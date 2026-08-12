"""Dictionary persistence.

YAML on purpose: the approval step is a human reading and editing this file, so
it needs comments and readable diffs. It is environment/customer state rather
than source, and is gitignored. It eventually belongs in
{bucket}.{scope}.dictionary alongside catalog and docs; a file is the right
shape while the schema is still moving.
"""
import yaml

from .. import config


def load(path=None) -> dict:
    path = path or config.DICTIONARY_PATH
    if not path.exists():
        return {"entries": []}
    with open(path) as f:
        return yaml.safe_load(f) or {"entries": []}


def save(dictionary: dict, path=None) -> None:
    path = path or config.DICTIONARY_PATH
    with open(path, "w") as f:
        yaml.safe_dump(dictionary, f, sort_keys=False, width=100)
