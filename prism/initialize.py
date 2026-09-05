"""Initialize: the one operator action a user runs after the Couchbase AI Data
Plane workflow finishes ingesting - no manual index setup required.

Destructive by design - drops and rebuilds the catalog, the dictionary, and
the search index from scratch - but never touches `docs` or anything the
ingestion workflow itself owns.

Ordering is deliberate, not incidental: the catalog rebuild reads cover-page
text via a plain N1QL predicate (`prism.catalog.extraction.cover_text_from_chunks`),
never SEARCH(), so it has no dependency on the search index at all. That
means primary indexes, the catalog rebuild, and the dictionary clear can all
run first, while the EXISTING search index still serves any question asked
mid-run. Only once those steps have succeeded does this drop and recreate the
search index - the one step that actually degrades retrieval until it
finishes reindexing (measured live: under a minute for this corpus). Putting
it last keeps that window as short as it can be, and means a failure in an
earlier step never touches the index a running app depends on.
"""
import json

from . import catalog, config, couchbase_io, dictionary

STEPS = [
    "Ensure primary index — catalog",
    "Ensure primary index — dictionary",
    "Rebuild the catalog from ingested chunks",
    "Clear the dictionary",
    "Drop the search index",
    "Recreate the search index",
]

FTS_INDEX_DEFINITION_PATH = config.REPO_ROOT / "design" / "fts-index.json"


def run(model: str = None, sectors: dict = None, on_step=None,
       on_catalog_progress=None) -> dict:
    """Runs all six steps in order, stopping at the first failure - a later
    step assumes every earlier one succeeded, and continuing past a real
    failure (an index built against a definition that never loaded, a catalog
    rebuild that silently skipped) is worse than stopping loudly.

    on_step(index_1_based, total, name, status, detail=None) fires before
    ("running") and after ("done"/"error") each step, so a caller can render
    progress without depending on this module's internals. on_catalog_progress
    passes straight through to catalog.rebuild_from_chunks for its own
    per-document progress during the one step slow enough to need it.

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

    step(0, lambda: couchbase_io.ensure_primary_index(config.CATALOG_COLLECTION))
    step(1, lambda: couchbase_io.ensure_primary_index(config.DICTIONARY_COLLECTION))
    summary["catalog_results"] = step(2, lambda: catalog.rebuild_from_chunks(
        model=model, sectors=sectors, on_progress=on_catalog_progress))
    summary["dictionary_removed"] = step(3, dictionary.clear)
    step(4, lambda: couchbase_io.delete_search_index(index_name))
    step(5, lambda: couchbase_io.create_search_index(definition))

    return summary
