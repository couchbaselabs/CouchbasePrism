"""Tier 3 — the dictionary (design/architecture.md §4, ADR-0001).

Model-proposed from real use, human-approved when consequential. NOT
hand-curated, and NOT a prerequisite: PRISM answers with an empty dictionary,
surfacing labeled candidate interpretations instead of silently picking one.

Two entry types, deliberately separate and independently versioned:
    metric                the arithmetic:  quick_ratio = (ca - inv) / cl
    interpretation_policy the judgment:    healthy_at_or_above = 1.0

PRISM may compute a metric without holding authority to judge it.

    evaluator   restricted AST evaluation of an approved formula
    repository  persistence in {bucket}.{scope}.dictionary (YAML for tests only)
    matching    concept -> approved entry (prefers "primary" when more than
                one approved convention exists for the same concept)
    policy      value + policy -> verdict
    approval    the human decision (approve / forget / clear) - a formula
                already in executable, snake_case form
    compile     a human-authored form (metric name, a formula in ordinary
                words, primary/alternate, a threshold) -> an approved entry,
                deriving the executable formula and its content anchors
"""
from .approval import approve, clear, forget  # noqa: F401
from .compile import build_metric_entry, compile_formula  # noqa: F401
from .evaluator import FormulaError, evaluate_formula, formula_facts  # noqa: F401
from .matching import find_metric, find_policy  # noqa: F401
from .policy import judge  # noqa: F401
from .repository import load, save  # noqa: F401
