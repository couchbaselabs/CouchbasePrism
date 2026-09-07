"""Persistence for catalog documents: {bucket}.{scope}.catalog, keyed by doc_name."""
from .. import config
from ..couchbase_io import QueryError, ensure_primary_index, query, search_facet


def upsert(document: dict) -> None:
    query(
        f"UPSERT INTO `{config.BUCKET}`.`{config.SCOPE}`.`{config.CATALOG_COLLECTION}` "
        f"(KEY, VALUE) VALUES ($doc_id, $doc_body)",
        {"$doc_id": document["doc_id"], "$doc_body": document},
    )


def delete_all() -> int:
    """Empties the catalog collection. Used before a full rebuild - a stale
    entry from a previous extractor sitting alongside a freshly rebuilt one,
    with no way to tell them apart at read time, is worse than an empty
    collection a rebuild is about to repopulate."""
    rows = query(
        f"DELETE FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.CATALOG_COLLECTION}` "
        "RETURNING 1"
    )
    return len(rows)


def load_all() -> list:
    """Only the fields resolution needs. `value` is a reserved word in N1QL and
    must be backticked in the projection.

    Tolerates a missing primary index specifically - "no catalogued documents
    yet, Initialize has never run" is an expected state on a fresh
    environment (Initialize's own `ensure_primary_index` step exists exactly
    because this can happen), not an error worth crashing a caller over.
    Anything else still raises - a genuine query error should never read as
    an empty catalog."""
    try:
        return query(
            "SELECT d.doc_name, d.doc_type.`value` AS doc_type, d.doc_period, "
            "d.period_end_date_iso, d.company.`value` AS company, d.gics_sector, "
            "d.aliases, d.source_filename, d.search_label "
            f"FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.CATALOG_COLLECTION}` AS d"
        )
    except QueryError as e:
        if "No index available" in str(e):
            return []
        raise


def companies() -> list:
    """Distinct extracted company names in the catalog, for work that is per
    company rather than per document."""
    rows = query(
        "SELECT DISTINCT d.company.`value` AS company "
        f"FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.CATALOG_COLLECTION}` AS d "
        "WHERE d.company.`value` IS NOT MISSING")
    return sorted({r["company"] for r in rows if r.get("company")})


def set_aliases(company: str, aliases: list) -> int:
    """Attach aliases to every catalog document for one company. Returns the
    number of documents updated."""
    rows = query(
        f"UPDATE `{config.BUCKET}`.`{config.SCOPE}`.`{config.CATALOG_COLLECTION}` AS d "
        "SET d.aliases = $aliases, d.lineage.aliases = 'model_proposed' "
        "WHERE d.company.`value` = $company RETURNING d.doc_name",
        {"$aliases": aliases, "$company": company})
    return len(rows)


def set_period(doc_name: str, period: int, source: str) -> None:
    query(
        f"UPDATE `{config.BUCKET}`.`{config.SCOPE}`.`{config.CATALOG_COLLECTION}` AS d "
        "SET d.doc_period = $period, d.lineage.doc_period = $source "
        "WHERE d.doc_name = $doc_name",
        {"$period": period, "$source": source, "$doc_name": doc_name})


def all_periods() -> list:
    return query(
        "SELECT d.doc_name, d.doc_period, d.period_end_date_iso "
        f"FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.CATALOG_COLLECTION}` AS d")


def ingested_doc_names() -> set:
    """Which documents actually have chunks in the docs collection.

    The catalog is built from local PDFs, but ingestion can fail per document -
    24 of 368 did, mostly long filings timing out. A catalog entry for a
    document with no chunks is worse than no entry: resolution picks it, and the
    answer becomes "the excerpts do not contain..." for a reason that has
    nothing to do with retrieval quality.

    A search-index facet on xmeta-data.filename, not a N1QL `SELECT DISTINCT`
    scan - the same reasoning as cover_text_from_chunks(): no secondary index
    exists on that field, so the scan touched every document in the
    collection. A facet asks the search index for its own distinct terms
    directly. Verified live: identical result set to the old DISTINCT query
    (354 of 354 filenames, exact set match) in a fraction of the time.
    size=10000 is deliberately far above any real corpus's distinct-document
    count - facets cost nothing extra for an unmet size ceiling, so there is
    no reason to risk a real deployment silently losing documents past a
    tighter cap.
    """
    prefix = config.source_filename("")[:-len(".pdf")]
    terms = search_facet(config.FTS_DOCS_INDEX_NAME, "xmeta-data.filename", size=10000)
    out = set()
    for t in terms:
        name = t.get("term") or ""
        if name.startswith(prefix) and name.endswith(".pdf"):
            out.add(name[len(prefix):-len(".pdf")])
    return out


def ensure_index() -> None:
    ensure_primary_index(config.CATALOG_COLLECTION)
