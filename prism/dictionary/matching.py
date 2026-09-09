"""Matching a planner-identified concept to an approved entry.

Only `approved` entries are usable. A `proposed` entry is a record of
unresolved ambiguity, not an authority - treating it as one would skip the
human decision the whole design depends on.
"""


def _normalize(s: str) -> str:
    """Underscores and hyphens are separators, not characters. A model asked for
    a concept name returns "quick ratio", "quick-ratio" or "quick_ratio"
    interchangeably, and treating those as three concepts silently disables
    governance for two of them."""
    return " ".join((s or "").lower().replace("-", " ").replace("_", " ").split())


def find_metric(dictionary: dict, concept: str):
    """The first approved entry matching `concept`, UNLESS more than one
    matches - multiple approved conventions can coexist for one concept
    (dictionary.compile.build_metric_entry's formula_type), and when they
    do, the one explicitly tagged "primary" drives computation by default.
    An entry with no formula_type at all (every entry before this field
    existed) defaults to "primary" - a single, unambiguous entry behaves
    exactly as it always did."""
    target = _normalize(concept)
    if not target:
        return None
    matches = []
    for entry in dictionary.get("entries", []):
        if (entry.get("entry_type") != "metric"
                or entry.get("governance", {}).get("status") != "approved"):
            continue
        names = [entry.get("recognition", {}).get("canonical_name", "")]
        names += entry.get("recognition", {}).get("aliases") or []
        for name in names:
            normalized = _normalize(name)
            if normalized and (normalized == target
                               or normalized in target or target in normalized):
                matches.append(entry)
                break
    if not matches:
        return None
    primary = [e for e in matches if e.get("formula_type", "primary") == "primary"]
    return (primary or matches)[0]


def find_policy(dictionary: dict, metric_id: str):
    for entry in dictionary.get("entries", []):
        if (entry.get("entry_type") == "interpretation_policy"
                and entry.get("governance", {}).get("status") == "approved"
                and entry.get("applies_to") == metric_id):
            return entry
    return None
