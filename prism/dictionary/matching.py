"""Matching a planner-identified concept to a dictionary entry.

Every entry that exists in the dictionary is authoritative by construction -
there is no separate "proposed" or "approved" governance status to check
anymore. An entry only ever gets there through dictionary.compile (a form)
or dictionary.approval (a governed correction from an answer already seen),
both human-reviewed acts - so presence in the collection IS the approval.
"""


def _normalize(s: str) -> str:
    """Underscores and hyphens are separators, not characters. A model asked for
    a concept name returns "quick ratio", "quick-ratio" or "quick_ratio"
    interchangeably, and treating those as three concepts silently disables
    governance for two of them."""
    return " ".join((s or "").lower().replace("-", " ").replace("_", " ").split())


def find_metric(dictionary: dict, concept: str, formula_type: str = None):
    """The dictionary entry matching `concept`, by metric name or
    abbreviation. When more than one entry matches (a primary/alternate
    pair - see dictionary.compile.build_metric_entry), `formula_type`
    (typically read from the plan's own formula_preference field, when the
    question explicitly asked for a specific convention) picks a specific
    one if it names one that exists; otherwise the entry with no
    formula_type at all, or explicitly tagged "Preferred", wins - an entry
    from before this field existed defaults to "Preferred" so a single
    unambiguous entry behaves exactly as it always did.
    """
    target = _normalize(concept)
    if not target:
        return None
    matches = []
    for entry in dictionary.get("entries", []):
        names = [entry.get("metric", "")]
        if entry.get("abbreviation"):
            names.append(entry["abbreviation"])
        for name in names:
            normalized = _normalize(name)
            if normalized and (normalized == target
                               or normalized in target or target in normalized):
                matches.append(entry)
                break
    if not matches:
        return None
    if formula_type:
        wanted = _normalize(formula_type)
        chosen = [e for e in matches
                 if _normalize(e.get("formula_type") or "preferred") == wanted]
        if chosen:
            return chosen[0]
    preferred = [e for e in matches
                if _normalize(e.get("formula_type") or "preferred") == "preferred"]
    return (preferred or matches)[0]
