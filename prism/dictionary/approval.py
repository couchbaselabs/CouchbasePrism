"""The governed correction — the entire human step in the loop (ADR-0001).

Everything before this is the system proposing; everything after it is
deterministic. A person decides which surfaced candidate is authoritative, and
that decision becomes locked behaviour for every future matching question.
"""
import re

from .evaluator import formula_facts
from .repository import load, save


def approve(concept: str, formula: str, healthy_at_or_above=None,
            domain: str = "finance", document_types=None, path=None) -> dict:
    """Writes (or replaces) a metric entry and, optionally, its interpretation
    policy. Scoped deliberately: a document that breaks the scoping assumption
    (GAAP vs IFRS, an unusually structured balance sheet) should re-trigger
    ambiguity rather than silently inherit the wrong entry."""
    dictionary = load(path)
    slug = re.sub(r"[^a-z0-9]+", "_", concept.lower()).strip("_")
    metric_id = f"{domain}.{slug}"
    dictionary["entries"] = [
        e for e in dictionary.get("entries", [])
        if e.get("id") not in (metric_id, f"{metric_id}.policy")
    ]
    dictionary["entries"].append({
        "id": metric_id,
        "entry_type": "metric",
        "scope": {"domain": domain,
                  "document_types": document_types or ["10-K", "10-Q"]},
        "recognition": {"canonical_name": concept, "aliases": []},
        "interpretation": {"formula": formula, "required_facts": formula_facts(formula)},
        "governance": {"status": "approved", "source": "proposed_from_use", "version": 1},
    })
    if healthy_at_or_above is not None:
        dictionary["entries"].append({
            "id": f"{metric_id}.policy",
            "entry_type": "interpretation_policy",
            "applies_to": metric_id,
            "scope": {"domain": domain},
            "policy": {"healthy_at_or_above": float(healthy_at_or_above)},
            "governance": {"status": "approved", "source": "human_review", "version": 1},
        })
    save(dictionary, path)
    return dictionary
