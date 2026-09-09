"""The governed correction — the entire human step in the loop (ADR-0001).

Everything before this is the system proposing; everything after it is
deterministic. A person decides which surfaced candidate is authoritative, and
that decision becomes locked behaviour for every future matching question.

Distinct from dictionary.compile: this takes a formula ALREADY in executable,
snake_case form - the "approve one of the candidates this run just computed"
path from the answer page, where there is no user-friendly text to compile
and no anchors to derive (the candidate came with its own required_facts
already bound in this run).
"""
import uuid

from .evaluator import formula_facts
from .matching import _normalize, find_metric
from .repository import load, save


def approve(concept: str, formula: str, healthy_at_or_above=None, path=None,
           scope: str = None) -> dict:
    """Writes (or replaces) THE PREFERRED entry for `concept` - matched by
    metric name/abbreviation, not a stable id, since ids are now minted
    (uuid4), not derived from the concept. Any existing entry matching
    `concept` that is Preferred (or untyped - see matching.find_metric) is
    replaced; an Alternate entry for the same concept is left alone."""
    dictionary = load(path, scope=scope)
    target = _normalize(concept)

    def _is_preferred_match(entry):
        names = [entry.get("metric", "")]
        if entry.get("abbreviation"):
            names.append(entry["abbreviation"])
        matched = any(_normalize(n) and (_normalize(n) == target
                                        or _normalize(n) in target
                                        or target in _normalize(n)) for n in names)
        return matched and _normalize(entry.get("formula_type") or "preferred") == "preferred"

    dictionary["entries"] = [e for e in dictionary.get("entries", [])
                             if not _is_preferred_match(e)]
    entry = {
        "id": str(uuid.uuid4()),
        "metric": concept,
        "formula_type": "Preferred",
        "formula": formula,
        "required_facts": formula_facts(formula),
    }
    if healthy_at_or_above is not None:
        entry["threshold_operator"] = ">="
        entry["threshold_number"] = float(healthy_at_or_above)
    dictionary["entries"].append(entry)
    save(dictionary, path, scope=scope)
    return dictionary


def forget(concept: str, path=None, scope: str = None) -> list:
    """Remove EVERY entry matching `concept` (Preferred and any Alternates) -
    the inverse of `approve` - useful for re-demonstrating the ungoverned
    path for a single concept without discarding everything else that has
    been reviewed."""
    dictionary = load(path, scope=scope)
    entry = find_metric(dictionary, concept)
    if entry is None:
        return []
    target = _normalize(concept)

    def _matches(e):
        names = [e.get("metric", "")]
        if e.get("abbreviation"):
            names.append(e["abbreviation"])
        return any(_normalize(n) and (_normalize(n) == target
                                      or _normalize(n) in target
                                      or target in _normalize(n)) for n in names)

    removed = [e["id"] for e in dictionary["entries"] if _matches(e)]
    dictionary["entries"] = [e for e in dictionary["entries"] if e["id"] not in removed]
    save(dictionary, path, scope=scope)
    return removed


def clear(path=None, scope: str = None) -> list:
    """Empty the dictionary. PRISM operates fine like this - that is the whole
    claim - so this is how the cold half of the two-pass demo is set up."""
    dictionary = load(path, scope=scope)
    removed = [e.get("id") for e in dictionary.get("entries", [])]
    save({"entries": []}, path, scope=scope)
    return removed
