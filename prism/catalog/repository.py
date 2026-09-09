"""Persistence for catalog documents: {bucket}.{scope}.catalog, keyed by doc_name.

Every function takes an explicit `scope`, defaulting to config.DEFAULT_SCOPE -
initialize.py's per-domain loop is the one caller that varies it; every other
caller (the app, manage.py) gets today's single-domain behaviour unchanged.
"""
from .. import config
from ..couchbase_io import QueryError, ensure_primary_index, query, search_facet


def upsert(document: dict, scope: str = None) -> None:
    scope = scope or config.DEFAULT_SCOPE
    query(
        f"UPSERT INTO `{config.BUCKET}`.`{scope}`.`{config.CATALOG_COLLECTION}` "
        f"(KEY, VALUE) VALUES ($doc_id, $doc_body)",
        {"$doc_id": document["doc_id"], "$doc_body": document},
    )


def delete_all(scope: str = None) -> int:
    """Empties the catalog collection. Used before a full rebuild - a stale
    entry from a previous extractor sitting alongside a freshly rebuilt one,
    with no way to tell them apart at read time, is worse than an empty
    collection a rebuild is about to repopulate."""
    scope = scope or config.DEFAULT_SCOPE
    rows = query(
        f"DELETE FROM `{config.BUCKET}`.`{scope}`.`{config.CATALOG_COLLECTION}` "
        "RETURNING 1"
    )
    return len(rows)


def load_all(scope: str = None) -> list:
    """Only the fields resolution needs. `value` is a reserved word in N1QL and
    must be backticked in the projection.

    Tolerates a missing primary index specifically - "no catalogued documents
    yet, Initialize has never run" is an expected state on a fresh
    environment (Initialize's own `ensure_primary_index` step exists exactly
    because this can happen), not an error worth crashing a caller over.
    Anything else still raises - a genuine query error should never read as
    an empty catalog.

    Filtered to `type = "catalog_document"` - the collection also holds one
    "catalog_manifest" document per domain (catalog.manifest), and a caller
    here (the app's document list, eval's catalog_docs) expects one row per
    real document, not a phantom entry with no doc_name."""
    scope = scope or config.DEFAULT_SCOPE
    try:
        return query(
            "SELECT d.doc_name, d.doc_type.`value` AS doc_type, d.doc_period, "
            "d.period_end_date_iso, d.company.`value` AS company, d.gics_sector, "
            "d.aliases, d.source_filename, d.search_label "
            f"FROM `{config.BUCKET}`.`{scope}`.`{config.CATALOG_COLLECTION}` AS d "
            "WHERE d.type = 'catalog_document'"
        )
    except QueryError as e:
        if "No index available" in str(e):
            return []
        raise


def companies(scope: str = None) -> list:
    """Distinct extracted company names in the catalog, for work that is per
    company rather than per document."""
    scope = scope or config.DEFAULT_SCOPE
    rows = query(
        "SELECT DISTINCT d.company.`value` AS company "
        f"FROM `{config.BUCKET}`.`{scope}`.`{config.CATALOG_COLLECTION}` AS d "
        "WHERE d.type = 'catalog_document' AND d.company.`value` IS NOT MISSING")
    return sorted({r["company"] for r in rows if r.get("company")})


def set_aliases(company: str, aliases: list, scope: str = None) -> int:
    """Attach aliases to every catalog document for one company. Returns the
    number of documents updated."""
    scope = scope or config.DEFAULT_SCOPE
    rows = query(
        f"UPDATE `{config.BUCKET}`.`{scope}`.`{config.CATALOG_COLLECTION}` AS d "
        "SET d.aliases = $aliases, d.lineage.aliases = 'model_proposed' "
        "WHERE d.company.`value` = $company RETURNING d.doc_name",
        {"$aliases": aliases, "$company": company})
    return len(rows)


def set_period(doc_name: str, period: int, source: str, scope: str = None) -> None:
    scope = scope or config.DEFAULT_SCOPE
    query(
        f"UPDATE `{config.BUCKET}`.`{scope}`.`{config.CATALOG_COLLECTION}` AS d "
        "SET d.doc_period = $period, d.lineage.doc_period = $source "
        "WHERE d.doc_name = $doc_name",
        {"$period": period, "$source": source, "$doc_name": doc_name})


def all_periods(scope: str = None) -> list:
    scope = scope or config.DEFAULT_SCOPE
    return query(
        "SELECT d.doc_name, d.doc_period, d.period_end_date_iso "
        f"FROM `{config.BUCKET}`.`{scope}`.`{config.CATALOG_COLLECTION}` AS d "
        "WHERE d.type = 'catalog_document'")


def ingested_doc_names(scope: str = None) -> set:
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

    Empty on a domain with no chunks ingested yet (iso20020, for now) - the
    facet simply returns no terms, same as an empty corpus always has; this
    is not an error case, it is what "nothing ingested here yet" looks like.
    """
    scope = scope or config.DEFAULT_SCOPE
    prefix = config.source_filename("", scope=scope)[:-len(".pdf")]
    terms = search_facet(config.fts_docs_index_name(scope), "xmeta-data.filename",
                         size=10000, scope=scope)
    out = set()
    for t in terms:
        name = t.get("term") or ""
        if name.startswith(prefix) and name.endswith(".pdf"):
            out.add(name[len(prefix):-len(".pdf")])
    return out


def ensure_index(scope: str = None) -> None:
    ensure_primary_index(config.CATALOG_COLLECTION, scope=scope)
