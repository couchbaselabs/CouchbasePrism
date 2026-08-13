"""Tier 2 — retrieval over the chunk collection (design/architecture.md §3).

Chunks are produced by the Couchbase AI Data Plane Unstructured Data Workflow.
PRISM does not parse or chunk PDFs; improvements to chunk quality are feedback
to that service, not code here.

    planner          question -> concept, required facts, content anchors
    anchor_search    printed row labels -> the right chunk (IDF-ranked)
    vector_search    dense kNN over the Hyperscale Vector Index
    hybrid_search    BM25 + kNN through a Search Vector Index
    combined_search  assemble and dedupe
"""
from .anchor_search import anchor_search  # noqa: F401
from .combined_search import combine, retrieve  # noqa: F401
from .hybrid_search import hybrid_search  # noqa: F401
from .planner import (  # noqa: F401
    PLANNER_SYSTEM_PROMPT, fact_ids, formula_identifiers, plan_anchors,
    plan_evidence, plan_facts,
)
from .vector_search import embed, vector_search  # noqa: F401
