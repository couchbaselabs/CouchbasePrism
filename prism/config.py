"""Environment-derived configuration. Nothing here is corpus-specific."""
import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# --- Couchbase -------------------------------------------------------------
# Fixed, not environment-configurable, by deliberate choice - this is PRISM's
# own namespace contract, not a per-deployment setting. A customer's Couchbase
# AI Data Plane workflow has to be pointed at exactly these names for
# Initialize (and everything downstream of it) to find what it ingested;
# letting BUCKET/SCOPE vary per environment just adds a way for the app and
# the workflow to silently disagree about where the data lives. Only the
# connection string and credentials (couchbase_host/couchbase_auth, below)
# are meant to vary per deployment.
BUCKET = "acme"
SCOPE = "prism"
CATALOG_COLLECTION = "catalog"
DOCS_COLLECTION = "docs"
DICTIONARY_COLLECTION = "dictionary"

# FTS index names must be FULLY QUALIFIED in N1QL SEARCH() calls; the bare
# short name fails to resolve there. The Search Service's own REST endpoints
# (couchbase_io.py's create/delete/count/facet, scoped under /bucket/{b}/
# scope/{s}/index/{name}) take the bare name instead - the scope already
# names the bucket and scope, so repeating them in the index name 400s.
FTS_DOCS_INDEX_NAME = "ftsFinanceBench"
FTS_DOCS_INDEX = f"{BUCKET}.{SCOPE}.{FTS_DOCS_INDEX_NAME}"


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
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")

# Model per ROLE, not per call site. Three roles, because the stages differ in
# what they actually need:
#
#   planner   proposes structure it cannot verify - the evidence plan and the
#             candidate calculation conventions. Reasoning quality matters,
#             volume is one call each.
#   answer    binding and synthesis, where an error is a wrong number in front
#             of a reader. Binding is grouped here rather than with utility
#             because picking the wrong column is not a tagging mistake.
#   utility   classification and tagging - catalog field extraction, anchor
#             repair. Cheap and high volume.
#   judge      evaluation only, never part of an answer path. Defaults to the
#             answer model because grading is NOT a tagging task: on gpt-5.4-nano
#             the judge marked a verified-correct answer as a failure, and a
#             stronger judge on the same recorded answers scored 5/8 rather than
#             4/8. A weak grader silently understates the system.
# Defaults are measured, not assumed. Every one of these was chosen against a
# result recorded in this repository's history:
#
# planner  gpt-5.4. gpt-5.4-mini names an aggregate the source never prints
#          ("quick_assets") in 3 of 3 runs and cannot follow the union-of-
#          conventions rule; gpt-5.4 produces the union in 2 of 3, which is what
#          makes candidate sets materially distinct (spread 0.005-0.049 against
#          0.0). Costs roughly 70% more planning latency.
# answer   gpt-5.4. It produced the fully verified answer on the operating-margin
#          question - ten-plus figures, every one traced to the filing - in 28s
#          against gpt-5.5's 160s for no measured gain. Decisive tiebreak:
#          gpt-5.5 REJECTS temperature 0 and only accepts its default, so using
#          it here makes every run irreproducible.
# utility  gpt-5.4-nano. Extracted company, form and period across 354 documents
#          without a failure. Cheapest tier that did the job.
# judge    gpt-5.4, by explicit choice - which means it grades output from its
#          own tier. Self-grading is a measurable confound: it is how gpt-5.5
#          came to look worse than gpt-5.4 in a side-by-side. Set
#          PRISM_MODEL_JUDGE to a different model when a score has to be
#          defensible to someone else.
MODEL_PLANNER = os.environ.get("PRISM_MODEL_PLANNER", "gpt-5.4")
MODEL_ANSWER = os.environ.get("PRISM_MODEL_ANSWER", "gpt-5.4")
MODEL_UTILITY = os.environ.get("PRISM_MODEL_UTILITY", "gpt-5.4")
MODEL_JUDGE = os.environ.get("PRISM_MODEL_JUDGE", "gpt-5.4")

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
    "judge": "judge",
    "anchor_repair": "utility",
    "catalog": "utility",
}


def model_for(stage: str = None) -> str:
    return {"planner": MODEL_PLANNER, "answer": MODEL_ANSWER,
            "utility": MODEL_UTILITY,
            "judge": MODEL_JUDGE}[STAGE_ROLE.get(stage, "answer")]


def stage_models(planner: str = None, answer: str = None,
                 utility: str = None, judge: str = None) -> dict:
    """A per-stage mapping from role choices, for callers that select models at
    runtime (the UI) rather than from the environment. The judge follows the
    answer model unless overridden - it must not be the weakest model in the
    run, or it understates everything else."""
    chosen = {"planner": planner or MODEL_PLANNER,
              "answer": answer or MODEL_ANSWER,
              "utility": utility or MODEL_UTILITY,
              # NOT `judge or answer`: defaulting the grader to the model being
              # graded reintroduces self-grading the moment all roles are set
              # the same.
              "judge": judge or MODEL_JUDGE}
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
# "score"       Couchbase native, no fusion strategy: one SEARCH(), the Search
#               service SUMS the lexical and vector scores. Simple, but the two
#               scores are on different footings and a lexical-only hit is
#               outranked by every vector hit.
# "native-rrf"  Couchbase native reciprocal rank fusion, server-side, one
#               statement. Requires 8.1 (or the 8.0/7.6 backport). On 8.0.1 the
#               `score` field PARSED AND WAS IGNORED, which looks exactly like
#               success - verify with the checks in infra/README.md.
# "native-rsf"  Native relative score fusion.
# "native-dbsf" Native distribution-based score fusion.
# "rrf"         Our own: two SEARCH channels unioned in one statement, merged in
#               code. Kept because it works on any version and because it
#               reports each channel's rank and contribution per chunk.
# Measured over 143 questions against FinanceBench's annotated evidence pages:
# sum, native-rrf, native-rsf and our in-code rrf are indistinguishable on
# recall (0.58-0.60 resolved, 0.81-0.85 with the document forced). Fusion
# strategy is not a quality lever on this corpus. native-rrf is the default
# because it ties for best recall while being ~30% faster than the in-code
# version - one statement, no application-side merge - and `explain` still
# exposes the per-channel breakdown. native-dbsf is measurably WORSE (0.44) and
# is kept only as the counter-example.
HYBRID_FUSION = os.environ.get("PRISM_HYBRID_FUSION", "native-rrf")
# Only rrf and rsf are documented (Bleve v2.5.4 onwards). "dbsf" appears in the
# internal design document marked [TBD] and is absent from the released docs -
# the server accepts the value without implementing it, the same validation gap
# that lets "bogus" through. It measured worst of six configurations (recall@10
# 0.44 against 0.60-0.85), which is what an unimplemented strategy would look
# like, so it is not offered.
NATIVE_STRATEGIES = {"native-rrf": "rrf", "native-rsf": "rsf"}
# Native equivalents of RRF_K and LEG_CANDIDATES. score_window_size must be >=
# the requested size, and is the per-channel result set fusion considers.
NATIVE_RANK_CONSTANT = int(os.environ.get("PRISM_NATIVE_RANK_CONSTANT", 60))
NATIVE_WINDOW_SIZE = int(os.environ.get("PRISM_NATIVE_WINDOW_SIZE", 150))
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
