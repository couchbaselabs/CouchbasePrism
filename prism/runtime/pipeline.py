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
from .calculation import compute, propose_candidates
from .fact_binding import bind_facts, grounded_facts, to_identifier, validate_bindings
from .validation import validate_conclusion


def describe_source(entry: dict) -> str:
    """One line naming the resolved document in its own terms.

    Every field is optional and a missing one is simply left out, so this
    degrades to less context rather than to a sentence with a hole in it. The
    form is normalised for the prompt ("10-K", not "FORM 10-K") even though the
    catalog stores what was printed - the prompt wants the concept, not the
    evidence.

    `gics_sector` is here because vocabulary is sector-specific in a way genre
    alone does not capture: a Financials filing has no cost of goods sold, and a
    planner that knows the sector can stop proposing labels that cannot exist.
    """
    if not entry:
        return None
    form = catalog.form_of(entry.get("doc_type")) or entry.get("doc_type")
    bits = []
    if form:
        bits.append(f"a {form}")
    if entry.get("company"):
        bits.append(f"for {entry['company']}")
    if entry.get("gics_sector"):
        bits.append(f"in the {entry['gics_sector']} sector")
    if entry.get("doc_period"):
        bits.append(f"covering period {entry['doc_period']}")
    return ("the source is " + " ".join(bits) + ".") if bits else None


@dataclass(frozen=True)
class PipelineOptions:
    """Which capabilities are active. Defaults are the full runtime."""
    catalog_filter: bool = True   # scope retrieval to one resolved document
    anchors: bool = True          # content-anchor retrieval from the plan
    bm25: bool = False            # hybrid BM25 + kNN via the Search Vector Index
    title_boost: float = 0.0      # boost associated-titles in BM25 (unreliable)
    governance: bool = True       # bind, compute, validate, consult the dictionary
    fusion: str = None            # "score" (Couchbase native) or "rrf"; None = config
    source_context: bool = True   # tell the planner the resolved doc type/period
    # Retrieval tuning. None means "use the configured default", so a run that
    # touches no dial is identical to one from before the dials existed.
    rank_constant: int = None     # RRF only: 1/(k + rank)
    window_size: int = None       # per-channel result set fusion considers
    bm25_weight: float = None     # lexical channel weight (a query boost)
    vector_weight: float = None   # vector channel weight (a knn boost)
    knn_k: int = None             # vector candidate depth
    top_k: int = None             # evidence budget handed to the model


def answer_question(question: str, catalog_docs: list, dictionary_data: dict = None,
                    options: PipelineOptions = None, model: str = None) -> dict:
    """One question, end to end. Returns every intermediate stage so the demo
    can show the pipeline and the debugger can diagnose it."""
    options = options or PipelineOptions()
    dictionary_data = (dictionary_data if dictionary_data is not None
                       else dictionary.load())

    doc_name = (catalog.resolve_for_question(catalog_docs, question)
                if options.catalog_filter else None)
    # Built from the catalog, so the prompt stays generic and the corpus
    # supplies the specifics.
    entry = next((d for d in catalog_docs if d.get("doc_name") == doc_name), None)
    context = describe_source(entry) if options.source_context else None
    plan = retrieval.plan_evidence(question, model=model, source_context=context)
    weights = {k: v for k, v in (("bm25", options.bm25_weight),
                                 ("vector", options.vector_weight)) if v is not None}
    chunks = retrieval.retrieve(question, plan, doc_name,
                                use_anchors=options.anchors, use_bm25=options.bm25,
                                title_boost=options.title_boost,
                                fusion=options.fusion, weights=weights or None,
                                rank_constant=options.rank_constant,
                                window_size=options.window_size,
                                knn_k=options.knn_k,
                                top_k=options.top_k or config.TOP_K)

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
            else:
                # Propose FIRST, then bind the union of what the plan asked for
                # and what the candidates reference. The planner under-specifies:
                # it omitted `inventory` for quick ratio, leaving the only
                # correct formula with an unbound name and nothing computable.
                candidates, rejected = propose_candidates(
                    plan.get("concept", ""), question, plan, model=model)
                needed = set(planned)
                for candidate in candidates:
                    needed |= retrieval.formula_identifiers(candidate.get("formula", ""))
            bound = validate_bindings(
                bind_facts(question, sorted(needed), chunks, model=model), chunks)
            calc = compute(entry, candidates, grounded_facts(bound))
            conclusion = validate_conclusion(calc["computed"], policy)

    answer = synthesize(question, chunks, calc, conclusion, model=model)

    return {
        "question": question,
        "source_context": context,
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
