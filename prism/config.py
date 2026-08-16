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
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
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
# the pool the lexical leg is allowed to re-rank. Couchbase sums the two
# scores, and a document outside the kNN candidates cannot be surfaced by BM25
# no matter how well it matches: at k=50 the chunk holding 3M's FY2022
# operating-margin table was unreachable despite being the single best lexical
# hit in the filing; at k=100 it ranked 7th. Boosting cannot substitute, as
# scores are normalised and boosting lowers them.
KNN_CANDIDATES = 200
# "rrf"   two SEARCH legs unioned in one statement, merged by reciprocal rank.
# "score" the original single fused SEARCH() with the kNN clause inside it,
#         kept as a runnable fallback. It cannot surface a chunk the vector leg
#         missed, however well it matches lexically - see hybrid_search.
HYBRID_FUSION = os.environ.get("PRISM_HYBRID_FUSION", "rrf")
# Per-channel RRF weights. Equal by default: an unequal weighting is a claim
# that one channel is generally more trustworthy, which nothing measured here
# supports. Exposed so it can be tested rather than argued about.
RRF_BM25_WEIGHT = float(os.environ.get("PRISM_RRF_BM25_WEIGHT", 1.0))
RRF_VECTOR_WEIGHT = float(os.environ.get("PRISM_RRF_VECTOR_WEIGHT", 1.0))
