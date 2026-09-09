"""Interpretation policies — the judgment, kept separate from the arithmetic.

Computing a quick ratio of 0.96 is arithmetic. Concluding that 0.96 is
*unhealthy* is a threshold judgment: 0.9 is comfortable for a business with
fast, predictable receivables and alarming for one without. Conflating them
would have PRISM silently import a generic textbook threshold with exactly the
false confidence this design exists to eliminate.

Policies version independently of the metric they apply to, because one metric
may carry different thresholds for different organisations, and a change of
risk appetite must not force re-approval of the arithmetic.
"""
import operator as _operator

# {"operator": ">=", "value": 1.0} - the shape dictionary.compile.
# build_metric_entry() writes, general enough to express any of the four
# comparisons without a new policy type per operator.
_OPERATORS = {">=": _operator.ge, "<=": _operator.le,
             ">": _operator.gt, "<": _operator.lt, "==": _operator.eq}


def judge(value: float, policy: dict):
    """Returns (verdict, explanation), or (None, reason) when the policy cannot
    decide. Deliberately narrow - thresholds only. A richer judgment shape
    should be a new policy type, not a special case bolted on here.

    "healthy_at_or_above" is the original, still-supported shape (implicitly
    ">="); {"operator", "value"} is the general form for any of the four
    comparisons. Both are checked, not one replacing the other - existing
    approved policies never need re-writing for this to keep working."""
    rules = (policy or {}).get("policy", {})
    if "healthy_at_or_above" in rules:
        threshold = float(rules["healthy_at_or_above"])
        ok = value >= threshold
        return ok, (f"{value:.4g} is {'at or above' if ok else 'below'} the approved "
                    f"threshold of {threshold:g}")
    if "operator" in rules and "value" in rules:
        op_symbol = rules["operator"]
        op_fn = _OPERATORS.get(op_symbol)
        if op_fn is None:
            return None, f"approved policy uses an operator this runtime does not understand: {op_symbol!r}"
        threshold = float(rules["value"])
        ok = op_fn(value, threshold)
        return ok, (f"{value:.4g} is {'' if ok else 'not '}{op_symbol} the approved "
                    f"threshold of {threshold:g}")
    return None, "approved policy carries no rule this runtime understands"
