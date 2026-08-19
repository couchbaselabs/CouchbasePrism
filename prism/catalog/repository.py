"""Persistence for catalog documents: {bucket}.{scope}.catalog, keyed by doc_name."""
from .. import config
from ..couchbase_io import ensure_primary_index, query


def upsert(document: dict) -> None:
    query(
        f"UPSERT INTO `{config.BUCKET}`.`{config.SCOPE}`.`{config.CATALOG_COLLECTION}` "
        f"(KEY, VALUE) VALUES ($doc_id, $doc_body)",
        {"$doc_id": document["doc_id"], "$doc_body": document},
    )


def load_all() -> list:
    """Only the fields resolution needs. `value` is a reserved word in N1QL and
    must be backticked in the projection."""
    return query(
        "SELECT d.doc_name, d.doc_type.`value` AS doc_type, d.doc_period, "
        "d.period_end_date_iso, d.company.`value` AS company, d.gics_sector "
        f"FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.CATALOG_COLLECTION}` AS d"
    )


def ingested_doc_names() -> set:
    """Which documents actually have chunks in the docs collection.

    The catalog is built from local PDFs, but ingestion can fail per document -
    24 of 368 did, mostly long filings timing out. A catalog entry for a
    document with no chunks is worse than no entry: resolution picks it, and the
    answer becomes "the excerpts do not contain..." for a reason that has
    nothing to do with retrieval quality.

    Filenames are reversed back to doc_name here rather than stored, so this
    stays correct if the S3 prefix changes.
    """
    prefix = config.source_filename("")[:-len(".pdf")]
    rows = query(
        "SELECT DISTINCT d.`xmeta-data`.`filename` AS filename "
        f"FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.DOCS_COLLECTION}` AS d "
        "WHERE d.`xmeta-data`.`filename` IS NOT MISSING"
    )
    out = set()
    for r in rows:
        name = r.get("filename") or ""
        if name.startswith(prefix) and name.endswith(".pdf"):
            out.add(name[len(prefix):-len(".pdf")])
    return out


def ensure_index() -> None:
    ensure_primary_index(config.CATALOG_COLLECTION)
