"""Skills: domain-expert-owned knowledge about how documents in a domain are
typically organised - a proxy statement's filing lag, which form structurally
carries which period, table layout conventions. General to an industry,
never to one company's own vocabulary (concepts, `prism/concepts/`) or one
user's approved formula (dictionary, `prism/dictionary/`) - see
docs/adr/0003-skills-a-domain-expert-owned-knowledge-layer.md for why these
three stay separate, and separately owned.

A single local YAML file (`config.SKILLS_PATH`), not a Couchbase collection,
unlike dictionary and concepts - skills changes are rare, industry-wide edits
a domain reviewer makes, not the frequent, narrow, per-user edits those two
are built for. Kept as a file on purpose; see the ADR before "just make this
a collection too" - that's a real option, deliberately not decided here.

Every skill is a plain sentence, one per line - no structured fields, unlike
a dictionary entry or a concept. There is nothing to validate beyond "is this
a non-empty line" because nothing downstream parses a skill's content; it
only ever gets read back by a person and appended, verbatim, into a prompt.
"""
import pathlib

import yaml

from . import config


def _read(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load(scope: str = None, path: pathlib.Path = None) -> list:
    """This scope's skill statements, in the order they were saved. Empty
    when the file doesn't exist yet or has nothing for this scope - PRISM
    behaves exactly as it did before Skills existed in that case; nothing
    downstream needs to special-case an empty list."""
    scope = scope or config.DEFAULT_SCOPE
    path = path or config.SKILLS_PATH
    return list(_read(path).get(scope) or [])


def save(lines: list, scope: str = None, path: pathlib.Path = None) -> None:
    """Overwrites this scope's list. Other scopes already in the file are
    read first and preserved untouched - a save for "secfilings" must not
    erase a different domain's own skills."""
    scope = scope or config.DEFAULT_SCOPE
    path = path or config.SKILLS_PATH
    data = _read(path)
    data[scope] = [line.strip() for line in (lines or []) if line and line.strip()]
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def render(scope: str = None, path: pathlib.Path = None) -> str:
    """The block resolve_and_plan appends to its own system prompt - empty
    string, not a header with nothing under it, when this scope has no
    skills yet (a fresh deployment, or a non-finance domain that hasn't
    written any)."""
    lines = load(scope=scope, path=path)
    if not lines:
        return ""
    body = "\n".join(f"- {line}" for line in lines)
    return ("\n\nSKILLS (domain-expert-reviewed filing conventions - "
            f"informational, never a JSON-shape rule):\n{body}")
