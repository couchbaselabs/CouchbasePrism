"""Assembling the final chunk set from the available retrieval strategies.

Anchor hits lead - they are structurally identified and high precision - with
vector or hybrid results filling the remainder for coverage. Which strategies
run at all is decided by the caller (see prism.runtime.pipeline.PipelineOptions),
so the evaluation phases are configurations of this one path rather than forked
copies of it.
"""
from .. import config
from .anchor_search import anchor_search
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
             title_boost: float = 0.0, top_k: int = config.TOP_K) -> list:
    embedding = embed(question)
    anchors = (plan.get("content_anchors") or []) if use_anchors else []
    anchor_chunks = anchor_search(anchors, doc_name) if (anchors and doc_name) else []
    if use_bm25:
        semantic = hybrid_search(question, embedding, doc_name, top_k,
                                 title_boost=title_boost)
    else:
        semantic = vector_search(embedding, doc_name, top_k)
    return combine(anchor_chunks, semantic, top_k=top_k)
