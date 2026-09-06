"""Initialize: the one operator action a user runs after the Couchbase AI Data
Plane workflow finishes ingesting - no manual index setup required.

Destructive by design - drops and rebuilds the catalog, the dictionary, and
the search index from scratch - but never touches `docs` or anything the
ingestion workflow itself owns.

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
    "Rebuild the catalog from ingested chunks",
    "Clear the dictionary",
]

FTS_INDEX_DEFINITION_PATH = config.REPO_ROOT / "design" / "fts-index.json"


def run(model: str = None, sectors: dict = None, on_step=None,
       on_catalog_progress=None, on_index_progress=None) -> dict:
    """Runs all seven steps in order, stopping at the first failure - a later
    step assumes every earlier one succeeded, and continuing past a real
    failure (a catalog rebuild reading from a half-built index, a dictionary
    clear after something upstream silently didn't finish) is worse than
    stopping loudly.

    on_step(index_1_based, total, name, status, detail=None) fires before
    ("running") and after ("done"/"error") each step, so a caller can render
    progress without depending on this module's internals.
    on_catalog_progress passes straight through to catalog.rebuild_from_chunks
    for its own per-document progress; on_index_progress passes through to
    couchbase_io.wait_for_search_index for the reindex-catch-up step.

    Returns {"steps": [{"name", "ok", "error"?}, ...], "catalog_results": [...],
    "dictionary_removed": [...]}.
    """
    if not FTS_INDEX_DEFINITION_PATH.exists():
        raise FileNotFoundError(
            f"{FTS_INDEX_DEFINITION_PATH} is missing - Initialize needs the "
            "search index definition on disk to recreate the index from.")
    definition = json.loads(FTS_INDEX_DEFINITION_PATH.read_text())
    index_name = definition["name"]

    summary = {"steps": []}

    def step(i, fn):
        if on_step:
            on_step(i + 1, len(STEPS), STEPS[i], "running")
        try:
            result = fn()
            summary["steps"].append({"name": STEPS[i], "ok": True})
            if on_step:
                on_step(i + 1, len(STEPS), STEPS[i], "done")
            return result
        except Exception as e:
            summary["steps"].append({"name": STEPS[i], "ok": False, "error": str(e)})
            if on_step:
                on_step(i + 1, len(STEPS), STEPS[i], "error", str(e))
            raise

    def wait_for_index():
        target = couchbase_io.query(
            f"SELECT RAW COUNT(*) FROM `{config.BUCKET}`.`{config.SCOPE}`."
            f"`{config.DOCS_COLLECTION}`")[0]
        return couchbase_io.wait_for_search_index(
            index_name, target_count=target, on_progress=on_index_progress)

    step(0, lambda: couchbase_io.delete_search_index(index_name))
    step(1, lambda: couchbase_io.create_search_index(definition))
    step(2, wait_for_index)
    step(3, lambda: couchbase_io.ensure_primary_index(config.CATALOG_COLLECTION))
    step(4, lambda: couchbase_io.ensure_primary_index(config.DICTIONARY_COLLECTION))
    summary["catalog_results"] = step(5, lambda: catalog.rebuild_from_chunks(
        model=model, sectors=sectors, on_progress=on_catalog_progress))
    summary["dictionary_removed"] = step(6, dictionary.clear)

    return summary
