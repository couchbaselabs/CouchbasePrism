"""Environment-derived configuration. Nothing here is corpus-specific."""
import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# --- Couchbase -------------------------------------------------------------
BUCKET = os.environ.get("COUCHBASE_BUCKET", "acme")
SCOPE = os.environ.get("COUCHBASE_SCOPE", "prism")
CATALOG_COLLECTION = "catalog"
DOCS_COLLECTION = "docs"

# FTS index names must be FULLY QUALIFIED in N1QL SEARCH() calls; the bare
# short name fails to resolve.
FTS_DOCS_INDEX = f"{BUCKET}.{SCOPE}.ftsFinanceBench"


def couchbase_host() -> str:
    return os.environ["COUCHBASE_CONN_STRING"].replace("couchbases://", "")


def couchbase_auth() -> tuple:
    return (os.environ["COUCHBASE_USERNAME"], os.environ["COUCHBASE_PASSWORD"])


# --- Source objects --------------------------------------------------------
def source_filename(doc_name: str) -> str:
    """The AI Data Plane workflow derives the stored filename from the S3
    location as {bucket}_{folder}_{name}.pdf, so a catalog doc_name does not
    match `xmeta-data.filename` directly."""
    return f"{os.environ['AWS_BUCKET']}_{os.environ['AWS_FOLDER']}_{doc_name}.pdf"


# --- Models ----------------------------------------------------------------
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")

# Model per ROLE, not per call site. Three roles, because the stages differ in
# what they actually need:
#
#   planner   proposes structure it cannot verify - the evidence plan and the
#             candidate calculation conventions. Reasoning quality matters,
#             volume is one call each.
#   answer    binding and synthesis, where an error is a wrong number in front
#             of a reader. Binding is grouped here rather than with utility
#             because picking the wrong column is not a tagging mistake.
#   utility   classification, tagging and grading - catalog field extraction,
#             the evaluation judge, anchor repair. Cheap and high volume.
MODEL_PLANNER = os.environ.get("PRISM_MODEL_PLANNER", OPENAI_MODEL)
MODEL_ANSWER = os.environ.get("PRISM_MODEL_ANSWER", OPENAI_MODEL)
MODEL_UTILITY = os.environ.get("PRISM_MODEL_UTILITY", OPENAI_MODEL)

# Reasoning models are slow enough that the old 60-90s ceilings dropped whole
# questions: one benchmark run lost a question to a 90s read timeout while the
# model was still reasoning. Generous by default; a hung call still fails.
LLM_TIMEOUT = int(os.environ.get("PRISM_LLM_TIMEOUT", 300))

# Which role each traced stage belongs to. A stage missing here falls back to
# the answer role, since an untagged call is more likely to be user-facing than
# throwaway.
STAGE_ROLE = {
    "planner": "planner",
    "candidate": "planner",
    "binder": "answer",
    "answer": "answer",
    "judge": "utility",
    "anchor_repair": "utility",
    "catalog": "utility",
}


def model_for(stage: str = None) -> str:
    return {"planner": MODEL_PLANNER, "answer": MODEL_ANSWER,
            "utility": MODEL_UTILITY}[STAGE_ROLE.get(stage, "answer")]


def stage_models(planner: str = None, answer: str = None,
                 utility: str = None) -> dict:
    """A per-stage mapping from three role choices, for callers that select
    models at runtime (the UI) rather than from the environment."""
    chosen = {"planner": planner or MODEL_PLANNER,
              "answer": answer or MODEL_ANSWER,
              "utility": utility or MODEL_UTILITY}
    return {stage: chosen[role] for stage, role in STAGE_ROLE.items()}
EMBED_ENDPOINT = os.environ.get("MODEL_END_POINT", "")
EMBED_MODEL = os.environ.get("MODEL_ID", "")

# --- Dictionary ------------------------------------------------------------
# Customer/environment state, not source. Lives outside the package so it can
# differ per deployment; eventually belongs in {bucket}.{scope}.dictionary.
DICTIONARY_PATH = pathlib.Path(
    os.environ.get("PRISM_DICTIONARY", REPO_ROOT / "dictionary.yaml"))

# --- Retrieval tuning ------------------------------------------------------
TOP_K = 10
MAX_ANCHOR_CHUNKS = 6
VECTOR_N_PROBES = 16
# kNN candidate depth inside the hybrid SEARCH(). NOT a result count - it is
# the pool the vector channel contributes. Couchbase unions the query and knn
# hits and sums their scores, so a lexical-only document IS eligible; it simply
# carries only its lexical score. Measured: 3M's FY2022 operating-margin table
# is present at rank 136 with score 0.6695, below the rank-10 floor of 0.8696.
# Outranked, not excluded. Boosting cannot rescue it either - scores are
# normalised, so a higher boost LOWERS the result (0.7394 at boost 1 down to
# 0.2731 at boost 1000).
KNN_CANDIDATES = 200
# "score" Couchbase native hybrid: one fused SEARCH(), the Search service sums
#         the lexical and vector scores. Default - nothing to tune, and one
#         blended SEARCH_SCORE() per row.
# "rrf"   two SEARCH channels unioned in one statement, merged in code by
#         reciprocal rank. Measurably better recall (0.67 vs 0.54 at 10, over
#         two runs) and it exposes each channel's contribution, at the cost of
#         three knobs. See eval.compare_fusion.
HYBRID_FUSION = os.environ.get("PRISM_HYBRID_FUSION", "score")
# Per-channel RRF weights. Equal by default: an unequal weighting is a claim
# that one channel is generally more trustworthy, which nothing measured here
# supports. Exposed so it can be tested rather than argued about.
RRF_BM25_WEIGHT = float(os.environ.get("PRISM_RRF_BM25_WEIGHT", 1.0))
RRF_VECTOR_WEIGHT = float(os.environ.get("PRISM_RRF_VECTOR_WEIGHT", 1.0))
# RRF rank constant. 60 is conventional. Measured over two independent runs of
# eval.compare_fusion, k=1 is better on recall@5 (0.50 vs 0.38) and mean gold
# rank (4.2 vs 5.0) while recall@10 is identical - so it reorders the evidence
# set without changing its membership, and all TOP_K chunks reach the model
# either way. Left at 60 because no end-to-end difference was measured; worth
# revisiting if TOP_K shrinks or a reranker is added, where rank order starts
# to matter.
RRF_K = int(os.environ.get("PRISM_RRF_K", 60))
