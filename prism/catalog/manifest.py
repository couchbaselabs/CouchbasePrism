"""The catalog metadata document ("manifest"): one aggregate document per
domain, computed AFTER a catalog rebuild, rolling up what's actually in that
domain's catalog - company names, the distinct FORM types present, sectors,
and which fiscal years have a complete annual filing (years_full) versus
only partial-year coverage so far (years_partial - the current, in-progress
fiscal year, most commonly).

This is what the Intent Clarifier (catalog.intent) reads instead of scanning
the whole catalog or searching ftsCatalog - one small, bounded document
regardless of how many documents the catalog itself holds, since it rolls up
DISTINCT values, not per-document rows. It stays small even at "millions of
documents" scale as long as the distinct company/doc_type/year vocabulary
stays bounded - the actual scaling assumption FTS-based resolution existed
to cover, now handled by never needing a per-document scan at question time
at all.

Stored in the SAME catalog collection as ordinary entries, distinguished by
`type: "catalog_manifest"` (ordinary entries are `type: "catalog_document"`)
rather than a new collection - it is one document, and repository.py's
read-time queries (load_all/companies/all_periods) are filtered to exclude
it, the same way the FTS index's own doc_config.mode already discriminates
document kinds by `type` everywhere else in this codebase.
"""
from .. import config
from ..couchbase_io import query
from . import repository
from .extraction import quarter_of
from .resolver import form_of

MANIFEST_DOC_ID = "_manifest"

EMPTY = {"companies": [], "doc_types": [], "sectors": [],
        "years_full": [], "years_partial": [], "quarters": {}}


def build(scope: str = None) -> dict:
    """Rolled up from repository.load_all(scope) - every field here is
    derived, nothing re-extracted or re-classified.

    doc_type is normalized through resolver.form_of() before being counted as
    distinct - the raw stored values ("FORM 10-K", "10-K", "SCHEDULE 14A",
    "DEF 14A") are cover-page phrasing differences for the SAME form across
    different filing years, not different forms. Skipping this normalization
    was a real bug the first time this was computed: the manifest surfaced
    six near-duplicate doc_types instead of four, which would have handed
    the Intent Clarifier the exact same noise the manifest exists to remove.

    company is deduped case-insensitively for the same reason - verified
    live, this corpus's own cover pages print "3M COMPANY" (SEC boilerplate
    registrant caps) on some filings and "3M Company" on others, and a naive
    set() surfaced both as distinct companies. One casing is kept per
    case-insensitive group (whichever sorts first, picked only for
    determinism - there is nothing to prefer between the two). intent.py's
    resolve_documents() still matches company case-insensitively against the
    catalog's own raw values, since this normalization happens only here in
    the manifest, not in the stored catalog documents themselves.

    years_full is which fiscal years have a 10-K catalogued - a complete
    annual filing. years_partial is every other year that has SOME
    catalogued document (a 10-Q, an 8-K) but no 10-K yet - almost always the
    current, in-progress fiscal year. This is the "2020 to 2025 full, 2026
    current" distinction the Intent Clarifier needs to know which years in a
    requested range are actually answerable in full.

    quarters (year -> [quarter, ...]) is which quarters have their OWN 10-Q
    for that year - added after a live miss: without it, a "third quarter of
    2022" question correctly resolved to company/doc_type/year but had no
    way to say WHICH of that year's three 10-Qs it meant, and the pipeline
    silently picked Q1 (whatever order N1QL happened to return). Derived
    from each 10-Q's own period_end_date_iso via quarter_of() - the same
    month-based arithmetic _search_label already uses, not a new guess. Only
    10-Qs contribute: a 10-K's period_end_date is fiscal year-end, not one
    quarter, and quarter_of() on it would misleadingly return Q4.
    """
    rows = repository.load_all(scope)
    companies_by_key, sectors, doc_types = {}, set(), set()
    years_with_10k, years_seen = set(), set()
    quarters_by_year = {}
    for row in rows:
        company = row.get("company")
        if company:
            # min(), not first-seen: load_all() carries no ORDER BY, so
            # "first" would be a different casing on every run. min() over
            # whatever variants exist is at least the same answer every time.
            companies_by_key[company.upper()] = min(
                company, companies_by_key.get(company.upper(), company))
        if row.get("gics_sector"):
            sectors.add(row["gics_sector"])
        form = form_of(row.get("doc_type")) or row.get("doc_type")
        if form:
            doc_types.add(form)
        period = row.get("doc_period")
        if period:
            years_seen.add(period)
            if form == "10-K":
                years_with_10k.add(period)
            if form == "10-Q":
                quarter = quarter_of(row.get("period_end_date_iso"))
                if quarter:
                    quarters_by_year.setdefault(period, set()).add(quarter)
    return {
        "companies": sorted(companies_by_key.values()),
        "doc_types": sorted(doc_types),
        "sectors": sorted(sectors),
        "years_full": sorted(years_with_10k),
        "years_partial": sorted(years_seen - years_with_10k),
        "quarters": {str(year): sorted(qs)
                    for year, qs in sorted(quarters_by_year.items())},
    }


def save(manifest: dict, scope: str = None) -> None:
    scope = scope or config.DEFAULT_SCOPE
    document = {**manifest, "doc_id": MANIFEST_DOC_ID, "type": "catalog_manifest"}
    query(
        f"UPSERT INTO `{config.BUCKET}`.`{scope}`.`{config.CATALOG_COLLECTION}` "
        "(KEY, VALUE) VALUES ($doc_id, $doc_body)",
        {"$doc_id": MANIFEST_DOC_ID, "$doc_body": document},
    )


def load(scope: str = None) -> dict:
    """A point read (USE KEYS on the fixed manifest id), not a scan - cheap
    regardless of catalog size, which is the whole reason the Intent
    Clarifier can read this instead of searching ftsCatalog. Returns EMPTY
    (every field an empty list) on a domain with no catalog yet, or one
    Initialize hasn't rebuilt since this module was introduced - callers
    should treat that the same way an empty catalog always has, not as an
    error."""
    scope = scope or config.DEFAULT_SCOPE
    rows = query(
        f"SELECT d.* FROM `{config.BUCKET}`.`{scope}`.`{config.CATALOG_COLLECTION}` AS d "
        "USE KEYS $doc_id",
        {"$doc_id": MANIFEST_DOC_ID},
    )
    return rows[0] if rows else dict(EMPTY)


def rebuild(scope: str = None) -> dict:
    """Compute and persist in one call - the step initialize.py runs
    immediately after the catalog rebuild, per domain."""
    manifest = build(scope)
    save(manifest, scope=scope)
    return manifest
