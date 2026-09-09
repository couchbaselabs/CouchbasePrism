"""Initialize: the one operator action a user runs after the Couchbase AI Data
Plane workflow finishes ingesting - no manual index setup required.

Runs the same thirteen steps ONCE PER CONFIGURED DOMAIN (design/domains.yaml) - one
domain, one scope, each with its own docs/catalog search indexes. A domain
with nothing ingested yet (iso20020, for now - architecture only, no real
content) is not an error: its catalog rebuild step returns an empty result and
every other step runs against an empty collection exactly the way a fresh
environment always has. That is deliberate, not a workaround - the whole
point of building the domain-scoping layout for real now is that a second
domain provisions and behaves correctly with zero content in it, the same way
the concepts collection does.

Destructive by design, per domain - drops and rebuilds that domain's catalog,
dictionary, and search indexes from scratch - but never touches `docs` or
anything the ingestion workflow itself owns, in any domain.

Ordering is deliberate, not incidental, and reverses an earlier version of
this module's own reasoning. The catalog rebuild reads cover-page text via
`prism.catalog.extraction.cover_text_from_chunks`, which goes through the
search index (SEARCH()) rather than a plain N1QL predicate - `docs` carries
no secondary index on xmeta-data.filename or meta-data.page-number, only a
primary index and the search index, so a plain WHERE clause there was a full
PrimaryScan3 over the whole 177K-document collection PER DOCUMENT catalogued
(confirmed live via EXPLAIN - this was the real cause of catalog rebuild
running at 1-2 documents/minute; a raw N1QL fix would have meant adding a
second, dedicated GSI, which is exactly the redundant index this design
avoids). `ingested_doc_names()` has the same shape: a search-index facet
instead of a `SELECT DISTINCT` scan.

That means the search index now has to exist and be FULLY caught up with
`docs` before the catalog rebuild starts, not after it - the opposite of an
earlier version of this file, which put the index recreate last specifically
to minimize how long retrieval was degraded. Recreating the index is what a
running app's own retrieval depends on too, so there is no way to make this
step free of a disruption window; putting it first just means everything
downstream (catalog rebuild, dictionary clear) waits until the index is
actually ready to serve the queries they need, rather than running against
an incomplete one and silently missing documents that have not finished
reindexing yet.
"""
import json

from . import catalog, config, couchbase_io, dictionary

STEPS = [
    "Drop the search index",
    "Recreate the search index",
    "Wait for the search index to catch up",
    "Ensure primary index — catalog",
    "Ensure primary index — dictionary",
    "Ensure primary index — concepts",
    "Ensure primary index — docs",
    "Rebuild the catalog from ingested chunks",
    "Build the catalog metadata document",
    "Drop the catalog search index",
    "Recreate the catalog search index",
    "Wait for the catalog search index to catch up",
    "Clear the dictionary",
]

FTS_INDEX_DEFINITION_PATH = config.REPO_ROOT / "design" / "fts-index.json"
FTS_CATALOG_INDEX_DEFINITION_PATH = config.REPO_ROOT / "design" / "fts-catalog-index.json"


def run(model: str = None, sectors: dict = None, on_step=None,
       on_catalog_progress=None, on_index_progress=None,
       domains: list = None) -> dict:
    """Runs all thirteen steps, in order, for each domain in `domains` (defaults to
    every domain design/domains.yaml configures - "init can do all scopes" is
    the point, not an opt-in). Stops at the first failure within a domain -
    a later step assumes every earlier one in THAT domain succeeded - but
    still attempts every remaining domain, since one domain's failure says
    nothing about another's independent collections and indexes.

    Each domain's own search index (`fts{Domain}Docs`) is created from the
    SAME design/fts-index.json template, with `create_search_index`
    rewriting the index name and scope binding per domain - one definition,
    N differently-named indexes, same reasoning as the catalog index below.

    The catalog's own search index (`fts{Domain}Catalog` - backs
    catalog.resolve_for_question_at_scale, fts_resolver.py's FTS-shortlist-
    then-LLM-pick resolution path, no longer the live one but still tested)
    is recreated AFTER the catalog rebuild, not before - the opposite order
    from the docs index. Each catalog entry's search_label (see
    extraction.py's _search_label) is computed as part of the rebuild
    itself, so recreating this index earlier would just index stale or
    absent labels from before the rebuild ran. The manifest step right after
    the catalog rebuild has no such dependency on this index at all - it is
    read by a direct point fetch (USE KEYS), never searched.

    on_step(i_1_based, total, name, status, detail=None) fires before
    ("running") and after ("done"/"error") each step, where `total` is the
    step count across ALL domains (len(domains) * len(STEPS)) and `name` is
    "{step} — {domain}" - the same STEPS prefixes a caller may already match
    on (e.g. `name.startswith("Rebuild the catalog")`) still hold, since the
    domain is appended, not prepended.
    on_catalog_progress passes straight through to catalog.rebuild_from_chunks
    for its own per-document progress; on_index_progress passes through to
    couchbase_io.wait_for_search_index for BOTH reindex-catch-up steps.

    Returns {"domains": {domain: {"steps": [...], "catalog_results": [...],
    "manifest": {...}, "dictionary_removed": [...]}}} - one entry per domain
    attempted, in the same order `domains` was given.
    """
    if not FTS_INDEX_DEFINITION_PATH.exists():
        raise FileNotFoundError(
            f"{FTS_INDEX_DEFINITION_PATH} is missing - Initialize needs the "
            "search index definition on disk to recreate the index from.")
    if not FTS_CATALOG_INDEX_DEFINITION_PATH.exists():
        raise FileNotFoundError(
            f"{FTS_CATALOG_INDEX_DEFINITION_PATH} is missing - Initialize needs "
            "the catalog search index definition on disk to recreate it from.")
    definition = json.loads(FTS_INDEX_DEFINITION_PATH.read_text())
    catalog_index_definition = json.loads(FTS_CATALOG_INDEX_DEFINITION_PATH.read_text())

    domains = list(domains or config.DOMAINS)
    total_steps = len(STEPS) * len(domains)
    overall = {"domains": {}}

    for d, scope in enumerate(domains):
        summary = {"steps": []}
        index_name = config.fts_docs_index_name(scope)
        catalog_index_name = config.fts_catalog_index_name(scope)

        def step(i, fn, _d=d, _scope=scope, _summary=summary):
            global_i = _d * len(STEPS) + i
            name = f"{STEPS[i]} — {_scope}"
            if on_step:
                on_step(global_i + 1, total_steps, name, "running")
            try:
                result = fn()
                _summary["steps"].append({"name": STEPS[i], "ok": True})
                if on_step:
                    on_step(global_i + 1, total_steps, name, "done")
                return result
            except Exception as e:
                _summary["steps"].append({"name": STEPS[i], "ok": False, "error": str(e)})
                if on_step:
                    on_step(global_i + 1, total_steps, name, "error", str(e))
                raise

        def wait_for_index(_scope=scope, _index_name=index_name):
            target = couchbase_io.query(
                f"SELECT RAW COUNT(*) FROM `{config.BUCKET}`.`{_scope}`."
                f"`{config.DOCS_COLLECTION}`")[0]
            return couchbase_io.wait_for_search_index(
                _index_name, target_count=target, on_progress=on_index_progress,
                scope=_scope)

        def wait_for_catalog_index(_scope=scope, _catalog_index_name=catalog_index_name):
            target = couchbase_io.query(
                f"SELECT RAW COUNT(*) FROM `{config.BUCKET}`.`{_scope}`."
                f"`{config.CATALOG_COLLECTION}`")[0]
            return couchbase_io.wait_for_search_index(
                _catalog_index_name, target_count=target, on_progress=on_index_progress,
                scope=_scope)

        try:
            step(0, lambda: couchbase_io.delete_search_index(index_name, scope=scope))
            step(1, lambda: couchbase_io.create_search_index(
                definition, scope=scope, index_name=index_name))
            step(2, wait_for_index)
            step(3, lambda: couchbase_io.ensure_primary_index(
                config.CATALOG_COLLECTION, scope=scope))
            step(4, lambda: couchbase_io.ensure_primary_index(
                config.DICTIONARY_COLLECTION, scope=scope))
            step(5, lambda: couchbase_io.ensure_primary_index(
                config.CONCEPTS_COLLECTION, scope=scope))
            step(6, lambda: couchbase_io.ensure_primary_index(
                config.DOCS_COLLECTION, scope=scope))
            summary["catalog_results"] = step(7, lambda: catalog.rebuild_from_chunks(
                model=model, sectors=sectors, on_progress=on_catalog_progress, scope=scope))
            summary["manifest"] = step(8, lambda: catalog.rebuild_manifest(scope=scope))
            step(9, lambda: couchbase_io.delete_search_index(catalog_index_name, scope=scope))
            step(10, lambda: couchbase_io.create_search_index(
                catalog_index_definition, scope=scope, index_name=catalog_index_name))
            step(11, wait_for_catalog_index)
            summary["dictionary_removed"] = step(12, lambda: dictionary.clear(scope=scope))
        except Exception:
            # This domain's remaining steps are skipped (a later one assumes
            # an earlier one in the SAME domain succeeded), but the next
            # domain still runs - see the module docstring.
            pass

        overall["domains"][scope] = summary

    return overall
