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
