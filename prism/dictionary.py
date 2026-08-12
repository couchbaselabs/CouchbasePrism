"""Tier 3 — the dictionary (design/architecture.md §4, ADR-0001).

Model-proposed from real use, human-approved when consequential. NOT
hand-curated, and NOT a prerequisite: PRISM answers with an empty dictionary,
surfacing labeled candidate interpretations instead of silently picking one.

Two entry types, deliberately separate and independently versioned:
  metric                — the arithmetic:  quick_ratio = (ca - inv) / cl
  interpretation_policy — the judgment:    healthy_at_or_above = 1.0

PRISM may compute a metric without holding authority to judge it. A quick
ratio of 0.9 is comfortable for a business with fast receivables and alarming
for one without; that is risk appetite, not arithmetic, so it versions
separately and its absence means the verdict is declined rather than guessed.

An approved formula is evaluated HERE, in code, through a restricted AST
walker - never by a model. Left to compute freely with correct inputs in hand,
both gpt-4o-mini and gpt-4o produced defensible-but-unintended conventions
(0.85 and 0.90 against an expected 0.96). That is the whole reason this tier
exists.
"""
import ast
import operator
import re

import yaml

from . import config

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv}


class FormulaError(ValueError):
    pass


def evaluate_formula(formula: str, facts: dict) -> float:
    """Arithmetic over bound fact names only. A formula originates as
    model-authored text, so it must never reach eval(): no calls, no
    attributes, no subscripts, no names that are not bound facts."""
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"unparseable formula {formula!r}: {e}") from e

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            raise FormulaError(f"non-numeric constant {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id not in facts:
                raise FormulaError(f"formula references unbound fact {node.id!r}")
            if facts[node.id] is None:
                raise FormulaError(f"fact {node.id!r} has no bound value")
            return float(facts[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = walk(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        raise FormulaError(f"disallowed expression element: {type(node).__name__}")

    try:
        return walk(tree)
    except ZeroDivisionError as e:
        raise FormulaError("division by zero") from e


# ------------------------------------------------------------------ store

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


def _normalize(s: str) -> str:
    return " ".join((s or "").lower().replace("-", " ").split())


def find_metric(dictionary: dict, concept: str):
    """Only `approved` entries are usable. A `proposed` entry is a record of
    unresolved ambiguity, not an authority."""
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
            n = _normalize(name)
            if n and (n == target or n in target or target in n):
                return entry
    return None


def find_policy(dictionary: dict, metric_id: str):
    for entry in dictionary.get("entries", []):
        if (entry.get("entry_type") == "interpretation_policy"
                and entry.get("governance", {}).get("status") == "approved"
                and entry.get("applies_to") == metric_id):
            return entry
    return None


def judge(value: float, policy: dict):
    """Returns (verdict, explanation), or (None, reason) when the policy cannot
    decide. Deliberately narrow - thresholds only. A richer judgment shape
    should be a new policy type, not a special case bolted on here."""
    rules = (policy or {}).get("policy", {})
    if "healthy_at_or_above" in rules:
        threshold = float(rules["healthy_at_or_above"])
        ok = value >= threshold
        return ok, (f"{value:.4g} is {'at or above' if ok else 'below'} the approved "
                    f"threshold of {threshold:g}")
    return None, "approved policy carries no rule this runtime understands"


def approve(concept: str, formula: str, healthy_at_or_above=None,
            domain: str = "finance", path=None) -> dict:
    """The governed correction. This is the entire human step: a person decides
    which surfaced candidate is authoritative, and it becomes deterministic
    behaviour from then on."""
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
        "scope": {"domain": domain, "document_types": ["10-K", "10-Q"]},
        "recognition": {"canonical_name": concept, "aliases": []},
        "interpretation": {
            "formula": formula,
            "required_facts": sorted(
                {n.id for n in ast.walk(ast.parse(formula, mode="eval"))
                 if isinstance(n, ast.Name)}),
        },
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
