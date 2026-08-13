"""The evaluation phases, as CONFIGURATIONS of one runtime.

Each phase is a `PipelineOptions`, not a separate implementation. An earlier
iteration kept forked eval scripts and they drifted: a prompt fix applied to
one silently failed to reach the others and cost a question that had previously
passed. One code path, three settings, no drift.

Three phases, each adding one layer, so the deltas attribute cleanly:

    1-vector    embed the question, kNN across the whole corpus. This is
                textbook RAG, and it is the baseline the rest is measured
                against.
    2-catalog   + resolve WHICH document from the catalog first, then search
                only that filing. Scope before similarity.
    3-hybrid    + BM25 lexical matching and content anchors alongside kNN
                through the Search Vector Index, plus fact binding,
                deterministic calculation and dictionary governance. The
                preferred architecture; the one that emits SEARCH().

An earlier five-phase split separated content anchors ("planner") and
governance ("runtime") into their own steps, but that left the default phase
with bm25=False - so the architecture being demonstrated never actually
exercised the FTS index. Folding them into 3-hybrid means the preferred
configuration is the one that shows the SEARCH() query.

title_boost is deliberately 0. Boosting `meta-data.associated-titles` looks
attractive but that field is wrong on a meaningful fraction of table chunks
(three confirmed cases in this corpus, including a securities table titled
["Delaware", "41-0417775"]), so weighting it promotes confidently wrong chunks.
BM25 runs on `text-to-embed` only.
"""
from prism.runtime import PipelineOptions

PHASES = {
    "1-vector": PipelineOptions(catalog_filter=False, anchors=False, bm25=False,
                                governance=False),
    "2-catalog": PipelineOptions(catalog_filter=True, anchors=False, bm25=False,
                                 governance=False),
    "3-hybrid": PipelineOptions(catalog_filter=True, anchors=True, bm25=True,
                                title_boost=0.0, governance=True),
}

DEFAULT_PHASE = "3-hybrid"

LABELS = {
    "1-vector": "Vector only",
    "2-catalog": "Catalog + Vector",
    "3-hybrid": "Hybrid · BM25 + Catalog + Vector",
}


def get(name: str) -> PipelineOptions:
    if name not in PHASES:
        raise KeyError(f"unknown phase {name!r}; choose from {sorted(PHASES)}")
    return PHASES[name]
