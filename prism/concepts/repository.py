"""Concepts persistence: one document per entry in {bucket}.{scope}.concepts,
keyed by the entry's own id - the exact same shape as
dictionary/repository.py, since both are "a small list of entries a human
reviewed," just governing different things.

`path` is a test seam, not a production option: when given, load()/save()
read and write a local YAML file instead of Couchbase, same reasoning as
dictionary's own test seam (no live cluster needed for a test).
"""
import yaml

from .. import config
from ..couchbase_io import QueryError, query


def load(path=None, scope: str = None) -> dict:
    """Tolerates a missing primary index, same reasoning as
    dictionary.repository.load() and catalog.repository.load_all() - an
    empty concepts collection on a fresh environment is expected, not an
    error."""
    if path is not None:
        if not path.exists():
            return {"entries": []}
        with open(path) as f:
            return yaml.safe_load(f) or {"entries": []}
    scope = scope or config.DEFAULT_SCOPE
    try:
        rows = query(
            "SELECT d.* FROM "
            f"`{config.BUCKET}`.`{scope}`.`{config.CONCEPTS_COLLECTION}` AS d"
        )
    except QueryError as e:
        if "No index available" in str(e):
            return {"entries": []}
        raise
    return {"entries": rows}


def save(concepts: dict, path=None, scope: str = None) -> None:
    if path is not None:
        with open(path, "w") as f:
            yaml.safe_dump(concepts, f, sort_keys=False, width=100)
        return
    scope = scope or config.DEFAULT_SCOPE
    # Full replace, same semantics as dictionary.repository.save() - empty
    # the collection first so a removed entry actually disappears rather
    # than leaving a stale document upsert can't reach because its id changed.
    query(f"DELETE FROM `{config.BUCKET}`.`{scope}`.`{config.CONCEPTS_COLLECTION}`")
    for entry in concepts.get("entries", []):
        query(
            f"UPSERT INTO `{config.BUCKET}`.`{scope}`.`{config.CONCEPTS_COLLECTION}` "
            "(KEY, VALUE) VALUES ($id, $entry)",
            {"$id": entry["id"], "$entry": entry},
        )
