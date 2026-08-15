"""Assembling the final chunk set from the available retrieval strategies.

Anchor hits lead - they are structurally identified and high precision - with
vector or hybrid results filling the remainder for coverage. Which strategies
run at all is decided by the caller (see prism.runtime.pipeline.PipelineOptions),
so the evaluation phases are configurations of this one path rather than forked
copies of it.
"""
from .. import config
from .anchor_search import anchor_search
from .anchor_probe import filter_anchors
from .anchor_repair import repair_anchors
from .planner import plan_anchors
from .hybrid_search import hybrid_search
from .vector_search import embed, vector_search


def combine(*chunk_lists, top_k: int = config.TOP_K) -> list:
    """Order-preserving dedupe across strategies, first occurrence wins."""
    seen, out = set(), []
    for chunks in chunk_lists:
        for chunk in chunks or []:
            if len(out) >= top_k:
                return out
            key = chunk.get("id") or (chunk.get("text") or "")[:200]
            if key not in seen:
                seen.add(key)
                out.append(chunk)
    return out


def retrieve(question: str, plan: dict, doc_name: str = None, *,
             use_anchors: bool = True, use_bm25: bool = False,
             title_boost: float = 0.0, top_k: int = config.TOP_K,
             repair: bool = True) -> list:
    """Every configuration issues exactly ONE statement.

    With BM25 on, scope + lexical + vector are all legs of a single SEARCH(),
    so there is one query plan, one score, and one thing to show when
    explaining the architecture. Without it, anchors would need their own
    round trip, so that path stays available but is not what the phases use.
    """
    embedding = embed(question)
    anchors = plan_anchors(plan) if use_anchors else []
    if anchors:
        # Drop anchors the corpus cannot match before they consume budget.
        # 62% of generated anchors were dead on the 3M subset.
        live, dead, _counts = filter_anchors(anchors, doc_name)
        if dead and repair:
            # Look at the document, then plan again. The planner could not have
            # known this vocabulary; it is in the corpus, not in the model.
            seen = vector_search(embedding, doc_name, top_k)
            fixed, _proposed = repair_anchors(question, dead, seen, doc_name)
            live = live + [a for a in fixed if a not in live]
        # Not `live or anchors`: when every anchor is dead the lexical leg is
        # worth nothing, and an empty list makes hybrid_search fall back to
        # matching the question text instead of phrase-matching known misses.
        anchors = live
    if use_bm25:
        return hybrid_search(question, embedding, doc_name, anchors=anchors,
                             top_k=top_k, title_boost=title_boost)
    anchor_chunks = anchor_search(anchors, doc_name) if (anchors and doc_name) else []
    return combine(anchor_chunks, vector_search(embedding, doc_name, top_k), top_k=top_k)
