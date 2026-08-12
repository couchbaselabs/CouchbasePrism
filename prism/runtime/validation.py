"""Deciding whether the computed values support an answer.

The check is whether candidates agree on THE CONCLUSION THE QUESTION ASKS FOR,
not merely that they all computed. Two conventions that both land below a
threshold can support an answer before any approval exists; two that straddle
it cannot, and averaging or arbitrarily picking one is a wrong-answer risk
rather than a shortcut.

Absent an approved interpretation policy there is no conclusion to agree about.
That is itself the finding, not a gap to paper over with a default threshold.
"""
from .. import dictionary


def validate_conclusion(computed: list, policy) -> dict:
    values = [c["value"] for c in computed]
    if not values:
        return {"status": "no_value"}
    spread = round(max(values) - min(values), 6)
    if policy is None:
        return {"status": "no_policy", "spread": spread, "values": values,
                "note": "value(s) computed; no approved interpretation policy exists, "
                        "so no verdict is authorised"}
    verdicts = {dictionary.judge(v, policy)[0] for v in values}
    if len(verdicts) == 1:
        return {"status": "agreed", "verdict": verdicts.pop(), "spread": spread,
                "explanation": dictionary.judge(values[0], policy)[1]}
    return {"status": "conflicting", "spread": spread, "values": values,
            "note": "candidate conventions straddle the approved threshold - blocking on "
                    "review rather than picking one"}
