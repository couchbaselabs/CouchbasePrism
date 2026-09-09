"""Tier 3 — the dictionary (design/architecture.md §4, ADR-0001).

Model-proposed from real use, human-approved when consequential. NOT
hand-curated, and NOT a prerequisite: PRISM answers with an empty dictionary,
surfacing labeled candidate interpretations instead of silently picking one.

One entry shape, one document per metric convention - the arithmetic and its
threshold live together (quick_ratio = (ca - inv) / cl, threshold >= 1.0),
not as two separately-versioned documents. Presence in the dictionary IS the
approval; there is no separate governance status to check. Multiple
conventions for the same concept coexist as separate entries sharing a
`metric` name, distinguished by `formula_type` ("Preferred"/"Alternate").

PRISM may compute a metric without holding a threshold to judge it - an
entry with a formula but no threshold_operator/threshold_number computes a
value and still declines any verdict.

    evaluator   restricted AST evaluation of a stored formula
    repository  persistence in {bucket}.{scope}.dictionary (YAML for tests only)
    matching    concept -> entry (prefers "Preferred" when more than one
                entry matches, unless a specific formula_type is requested)
    policy      value + entry -> verdict
    approval    the human decision (approve / forget / clear) - a formula
                already in executable, snake_case form
    compile     a human-authored form (metric name, a formula in ordinary
                words, primary/alternate, a threshold) -> a dictionary
                entry, deriving the executable formula and its content
                anchors
"""
from .approval import approve, clear, forget  # noqa: F401
from .compile import build_metric_entry, compile_formula  # noqa: F401
from .evaluator import FormulaError, evaluate_formula, formula_facts  # noqa: F401
from .matching import find_metric  # noqa: F401
from .policy import judge  # noqa: F401
from .repository import load, save  # noqa: F401
