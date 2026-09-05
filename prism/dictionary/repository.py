"""Dictionary persistence: one document per entry in {bucket}.{scope}.dictionary,
keyed by the entry's own id, alongside catalog and docs.

`path` is a test seam, not a production option: when given, load()/save() read
and write a local YAML file instead of Couchbase - tests use pytest's
tmp_path through this so dictionary tests need no live cluster (the
"no cluster needed" rule the rest of this codebase's tests already follow).
When omitted (every real call site - manage.py, the runtime pipeline, the
app), it talks to Couchbase. Approval's readability requirement ("a human
reading and editing this... needs comments and readable diffs") is served by
the Streamlit UI's own dictionary display now, not by hand-editing a file -
manage.py approve/forget/reset-dictionary were already the only sanctioned
write paths, never a text editor.
"""
import yaml

from .. import config
from ..couchbase_io import query


def load(path=None) -> dict:
    if path is not None:
        if not path.exists():
            return {"entries": []}
        with open(path) as f:
            return yaml.safe_load(f) or {"entries": []}
    rows = query(
        "SELECT d.* FROM "
        f"`{config.BUCKET}`.`{config.SCOPE}`.`{config.DICTIONARY_COLLECTION}` AS d"
    )
    return {"entries": rows}


def save(dictionary: dict, path=None) -> None:
    if path is not None:
        with open(path, "w") as f:
            yaml.safe_dump(dictionary, f, sort_keys=False, width=100)
        return
    # approve/forget/clear all pass the FULL entries list and expect a total
    # replace, not a merge - empty the collection first so a removed entry
    # (forget) or an emptied dictionary (clear) actually disappears rather
    # than leaving a stale document upsert can't reach because its id changed.
    query(f"DELETE FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.DICTIONARY_COLLECTION}`")
    for entry in dictionary.get("entries", []):
        query(
            f"UPSERT INTO `{config.BUCKET}`.`{config.SCOPE}`.`{config.DICTIONARY_COLLECTION}` "
            "(KEY, VALUE) VALUES ($id, $entry)",
            {"$id": entry["id"], "$entry": entry},
        )
