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

from .. import catalog, dictionary, retrieval
from .answer import synthesize
from .calculation import compute, propose_candidates
from .fact_binding import bind_facts, grounded_facts, to_identifier, validate_bindings
from .validation import validate_conclusion


@dataclass(frozen=True)
class PipelineOptions:
    """Which capabilities are active. Defaults are the full runtime."""
    catalog_filter: bool = True   # scope retrieval to one resolved document
    anchors: bool = True          # content-anchor retrieval from the plan
    bm25: bool = False            # hybrid BM25 + kNN via the Search Vector Index
    title_boost: float = 0.0      # boost associated-titles in BM25 (unreliable)
    governance: bool = True       # bind, compute, validate, consult the dictionary


def answer_question(question: str, catalog_docs: list, dictionary_data: dict = None,
                    options: PipelineOptions = None, model: str = None) -> dict:
    """One question, end to end. Returns every intermediate stage so the demo
    can show the pipeline and the debugger can diagnose it."""
    options = options or PipelineOptions()
    dictionary_data = (dictionary_data if dictionary_data is not None
                       else dictionary.load())

    doc_name = (catalog.resolve_for_question(catalog_docs, question)
                if options.catalog_filter else None)
    plan = retrieval.plan_evidence(question, model=model)
    chunks = retrieval.retrieve(question, plan, doc_name,
                                use_anchors=options.anchors, use_bm25=options.bm25,
                                title_boost=options.title_boost)

    kind = plan.get("answer_kind")
    entry = policy = None
    bound, candidates = [], []
    calc = {"governed": False, "computed": [], "errors": []}
    conclusion = {}

    if options.governance:
        entry = dictionary.find_metric(dictionary_data, plan.get("concept", ""))
        policy = dictionary.find_policy(dictionary_data, entry["id"]) if entry else None

        if kind in ("derived_metric", "judgment"):
            planned = [to_identifier(f) for f in (plan.get("required_facts") or [])]
            if entry:
                needed = set(entry["interpretation"]["required_facts"])
            else:
                # Propose FIRST, then bind the union of what the plan asked for
                # and what the candidates reference. The planner under-specifies:
                # it omitted `inventory` for quick ratio, leaving the only
                # correct formula with an unbound name and nothing computable.
                candidates = propose_candidates(plan.get("concept", ""), question,
                                                planned, model=model)
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
        "calculation": calc,
        "conclusion": conclusion,
        "answer": answer,
    }
