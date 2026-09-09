"""Interpretation policy — the judgment, kept conceptually separate from the
arithmetic even though both now live on the SAME dictionary entry.

Computing a quick ratio of 0.96 is arithmetic. Concluding that 0.96 is
*unhealthy* is a threshold judgment: 0.9 is comfortable for a business with
fast, predictable receivables and alarming for one without. Conflating them
would have PRISM silently import a generic textbook threshold with exactly the
false confidence this design exists to eliminate - so a metric with a formula
but no threshold_operator/threshold_number computes a value and still
declines any verdict; see validation.validate_conclusion.
"""
import operator as _operator

# {"threshold_operator": ">=", "threshold_number": 1.0} - one general shape
# for all four comparisons, rather than a policy type per operator.
_OPERATORS = {">=": _operator.ge, "<=": _operator.le,
             ">": _operator.gt, "<": _operator.lt, "==": _operator.eq}


def judge(value: float, entry: dict):
    """Returns (verdict, explanation), or (None, reason) when the entry
    carries no threshold, or one this runtime does not understand.
    Deliberately narrow - thresholds only. A richer judgment shape should be
    a new field, not a special case bolted on here."""
    entry = entry or {}
    op_symbol = entry.get("threshold_operator")
    threshold_number = entry.get("threshold_number")
    if op_symbol is None or threshold_number is None:
        return None, "no approved threshold exists for this metric"
    op_fn = _OPERATORS.get(op_symbol)
    if op_fn is None:
        return None, f"approved policy uses an operator this runtime does not understand: {op_symbol!r}"
    threshold = float(threshold_number)
    ok = op_fn(value, threshold)
    return ok, (f"{value:.4g} is {'' if ok else 'not '}{op_symbol} the approved "
                f"threshold of {threshold:g}")
