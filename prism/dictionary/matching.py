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
    target = _normalize(concept)
    if not target:
        return None
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
                return entry
    return None


def find_policy(dictionary: dict, metric_id: str):
    for entry in dictionary.get("entries", []):
        if (entry.get("entry_type") == "interpretation_policy"
                and entry.get("governance", {}).get("status") == "approved"
                and entry.get("applies_to") == metric_id):
            return entry
    return None
