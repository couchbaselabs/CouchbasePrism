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
