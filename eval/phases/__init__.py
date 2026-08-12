"""The evaluation phases, as CONFIGURATIONS of one runtime.

Each phase is a `PipelineOptions`, not a separate implementation. The previous
iteration kept five forked eval scripts and they drifted: a prompt fix applied
to one silently failed to reach the others and cost a question that had
previously passed. One code path, five settings, no drift.

Running the full ladder reproduces the progression that is the core of the
demo - each step isolates one capability, so the deltas mean something:

    1-vector    pure kNN, no document filter          25%  (2/8)
    2-catalog   + catalog-resolved document filter    38%  (3/8)
    3-hybrid    + BM25 over the whole question        38%  (3/8)
    4-planner   + content-anchor retrieval            50%  (4/8)
    5-runtime   + binding, calculation, governance    50% cold / 62% warm

Phase 3 is the instructive one: hybrid search over the raw question bought
nothing, because the failures were never retrieval-ranking problems. Phase 4's
anchors and phase 5's governance are what actually moved.
"""
from prism.runtime import PipelineOptions

PHASES = {
    "1-vector": PipelineOptions(catalog_filter=False, anchors=False, bm25=False,
                                governance=False),
    "2-catalog": PipelineOptions(catalog_filter=True, anchors=False, bm25=False,
                                 governance=False),
    "3-hybrid": PipelineOptions(catalog_filter=True, anchors=False, bm25=True,
                                title_boost=3.0, governance=False),
    "4-planner": PipelineOptions(catalog_filter=True, anchors=True, bm25=False,
                                 governance=False),
    "5-runtime": PipelineOptions(catalog_filter=True, anchors=True, bm25=False,
                                 governance=True),
}

DEFAULT_PHASE = "5-runtime"


def get(name: str) -> PipelineOptions:
    if name not in PHASES:
        raise KeyError(f"unknown phase {name!r}; choose from {sorted(PHASES)}")
    return PHASES[name]
