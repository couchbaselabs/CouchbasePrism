"""The runtime pipeline (design/architecture.md §1).

`answer_question()` is the single entry point for the benchmark harness, the
per-question debugger AND the Streamlit demo. That is deliberate: a demo with
its own pipeline would drift from the thing being measured, and the demo's
credibility rests on it being the measured thing.

`PipelineOptions` is what makes the evaluation phases configurations of one
code path rather than forked copies. The previous iteration kept five separate
eval scripts, which drifted - a prompt fix applied to one silently failed to
reach the others and cost a passing question.
"""
from dataclasses import dataclass

from .. import catalog, config, dictionary, retrieval
from .answer import synthesize
from .calculation import candidate_facts, compute, propose_candidates
from .fact_binding import bind_facts, grounded_facts, to_identifier, validate_bindings
from .resolve_and_plan import resolve_and_plan
from .validation import validate_conclusion


@dataclass(frozen=True)
class PipelineOptions:
    """Which capabilities are active. Defaults are the full runtime."""
    catalog_filter: bool = True   # scope retrieval to one resolved document
    anchors: bool = True          # content-anchor retrieval from the plan
    bm25: bool = False            # hybrid BM25 + kNN via the Search Vector Index
    title_boost: float = 0.0      # boost associated-titles in BM25 (unreliable)
    governance: bool = True       # bind, compute, validate, consult the dictionary
    fusion: str = None            # "score" (Couchbase native) or "rrf"; None = config
    # Retrieval tuning. None means "use the configured default", so a run that
    # touches no dial is identical to one from before the dials existed.
    rank_constant: int = None     # RRF only: 1/(k + rank)
    window_size: int = None       # per-channel result set fusion considers
    bm25_weight: float = None     # lexical channel weight (a query boost)
    vector_weight: float = None   # vector channel weight (a knn boost)
    knn_k: int = None             # vector candidate depth
    top_k: int = None             # evidence budget handed to the model


def answer_question(question: str, catalog_docs: list, dictionary_data: dict = None,
                    options: PipelineOptions = None, model: str = None,
                    scope: str = None) -> dict:
    """One question, end to end. Returns every intermediate stage so the demo
    can show the pipeline and the debugger can diagnose it.

    `scope` picks which domain's manifest/catalog resolution runs against -
    defaults to config.DEFAULT_SCOPE. Retrieval itself (`retrieval.retrieve`,
    further down) is not yet threaded through an explicit scope - it still
    reads config.SCOPE, the transitional default - since nothing calls this
    with a second domain's chunks to retrieve against today. Resolution is
    threaded for real because the Intent Clarifier's manifest read
    genuinely needs to know which domain's manifest to fetch.
    """
    options = options or PipelineOptions()
    dictionary_data = (dictionary_data if dictionary_data is not None
                       else dictionary.load())

    # #...# spans are stripped here, once, before anything else sees the raw
    # question - the planner and the embedding model both get ordinary text,
    # unaware anything was marked (the words themselves stay in place, only
    # the hashes go). forced_phrases only reaches hybrid_search's lexical
    # leg, further down; it is a no-op for every other retrieval path.
    question, forced_phrases = retrieval.extract_phrase_terms(question)

    # Resolve + Plan (runtime/resolve_and_plan.py) - one LLM call doing what
    # used to be two (catalog.intent's Intent Clarifier, then
    # retrieval.planner's evidence planner): both read the same raw question
    # as their primary input, and neither's reasoning depended on the
    # other's structured output - the only thing that crossed between them,
    # a one-line "resolved doc type/period" hint into the planner, is now
    # unnecessary, since the model already knows what it just resolved
    # within the same completion. See that module for why combining the
    # CALL does not combine the CONCERNS (resolution stays manifest-driven
    # and corpus-specific; evidence planning stays domain-free).
    #
    # Only ONE document from the resolved set reaches retrieval below - a
    # known, real gap, not papered over: the multi-document retrieve+bind+
    # compute+synthesize pipeline a company-wide, multi-year resolution
    # exists to feed has not been built yet. The one used is the MOST RECENT
    # year (then quarter) in the set, not documents[0] - verified live this
    # was not cosmetic: the N1QL fetch carries no ORDER BY, so documents[0]
    # was whichever the server happened to return first, and that silently
    # answered a "2023 vs FY2022" question from 2022's own figures, and
    # separately a "third quarter of 2022" question from Q1's 10-Q - both
    # confident, cited, WRONG answers, not a decline. This sort is a
    # defensive floor, not a fix for the gap above: with only ever one
    # document reaching retrieval, latest-year-then-latest-quarter is the
    # least-wrong single guess when the model's own consolidation/quarter
    # guidelines (see resolve_and_plan.py's prompt) do not collapse the set
    # to one document on their own.
    #
    # catalog_filter=False (the unscoped baseline eval phase) skips
    # resolution but still needs a plan - plan_evidence() alone, not the
    # combined call, since there is no manifest-scoped resolution to do.
    resolution_detail = None
    if not options.catalog_filter:
        doc_name = None
        plan = retrieval.plan_evidence(question, model=model)
    else:
        resolution_detail = resolve_and_plan(question, scope=scope, model=model)
        documents = sorted(
            resolution_detail.get("documents") or [],
            key=lambda d: (d.get("doc_period") or 0,
                          catalog.quarter_of(d.get("period_end_date_iso")) or 0),
            reverse=True)
        doc_name = documents[0]["doc_name"] if documents else None
        plan = resolution_detail["plan"]
    entry = next((d for d in catalog_docs if d.get("doc_name") == doc_name), None)
    weights = {k: v for k, v in (("bm25", options.bm25_weight),
                                 ("vector", options.vector_weight)) if v is not None}
    chunks = retrieval.retrieve(question, plan, doc_name,
                                use_anchors=options.anchors, use_bm25=options.bm25,
                                title_boost=options.title_boost,
                                fusion=options.fusion, weights=weights or None,
                                rank_constant=options.rank_constant,
                                window_size=options.window_size,
                                knn_k=options.knn_k,
                                top_k=options.top_k or config.TOP_K,
                                forced_phrases=forced_phrases,
                                source_filename=(entry or {}).get("source_filename"))

    kind = plan.get("answer_kind")
    entry = policy = None
    bound, candidates, rejected = [], [], []
    calc = {"governed": False, "computed": [], "errors": []}
    conclusion = {}

    if options.governance:
        entry = dictionary.find_metric(dictionary_data, plan.get("concept", ""))
        policy = dictionary.find_policy(dictionary_data, entry["id"]) if entry else None

        # `attribution` is deliberately absent: the source states the
        # explanation, so there is nothing to compute. Including it cost 118 of
        # 160 seconds on one question - candidate proposal and fact binding both
        # ran, both produced nothing, and the trace showed two large confident
        # stages that contributed no part of the answer.
        if kind in ("derived_metric", "judgment"):
            planned = [to_identifier(f) for f in retrieval.fact_ids(plan)]
            if entry:
                needed = set(entry["interpretation"]["required_facts"])
                fact_specs = {fid: {"id": fid} for fid in needed}
            else:
                # Propose FIRST, then bind the union of what the plan asked for
                # and what the candidates reference. The planner under-specifies:
                # it omitted `inventory` for quick ratio, leaving the only
                # correct formula with an unbound name and nothing computable.
                candidates, rejected = propose_candidates(
                    plan.get("concept", ""), question, plan, model=model)
                # candidate_facts() carries each fact's own metadata (anchors,
                # period_role) rather than collapsing everything to bare ids -
                # a formula needing the same quantity from two years names two
                # facts and tags each with which period it is, so validation
                # can trust that instead of guessing from the id later.
                fact_specs = {to_identifier(f["id"]): {**f, "id": to_identifier(f["id"])}
                             for f in retrieval.plan_facts(plan) if f.get("id")}
                for fact in candidate_facts(candidates):
                    fact_specs.setdefault(fact["id"], fact)
                needed = set(planned)
                for candidate in candidates:
                    for ident in retrieval.formula_identifiers(candidate.get("formula", "")):
                        needed.add(ident)
                        fact_specs.setdefault(ident, {"id": ident})
            raw_bound = bind_facts(question, sorted(needed), chunks, model=model)
            for fact in raw_bound:
                role = fact_specs.get(fact.get("name"), {}).get("period_role")
                if role:
                    fact["period_role"] = role
            bound = validate_bindings(raw_bound, chunks)
            calc = compute(entry, candidates, grounded_facts(bound))
            conclusion = validate_conclusion(calc["computed"], policy)

    answer = synthesize(question, chunks, calc, conclusion, model=model)

    return {
        "question": question,
        "forced_phrases": forced_phrases,
        "resolution_detail": resolution_detail,
        "computation_skipped": kind == "attribution",
        "options": options,
        "resolved_doc": doc_name,
        "plan": plan,
        "answer_kind": kind,
        "concept": plan.get("concept"),
        "governed": calc["governed"],
        "dictionary_entry": entry["id"] if entry else None,
        "has_policy": policy is not None,
        "chunks": chunks,
        "bound_facts": bound,
        "candidates": candidates,
        "rejected_candidates": rejected,
        "calculation": calc,
        "conclusion": conclusion,
        "answer": answer,
    }
