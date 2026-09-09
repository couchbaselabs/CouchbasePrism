"""Deciding whether the computed values support an answer.

The check is whether candidates agree on THE CONCLUSION THE QUESTION ASKS FOR,
not merely that they all computed. Two conventions that both land below a
threshold can support an answer before any approval exists; two that straddle
it cannot, and averaging or arbitrarily picking one is a wrong-answer risk
rather than a shortcut.

Absent a dictionary entry with an approved threshold there is no conclusion
to agree about. That is itself the finding, not a gap to paper over with a
default threshold - true whether nothing was approved at all, or an entry
exists but carries a formula with no threshold_operator/threshold_number.
"""
from .. import dictionary


def validate_conclusion(computed: list, entry) -> dict:
    values = [c["value"] for c in computed]
    if not values:
        return {"status": "no_value"}
    spread = round(max(values) - min(values), 6)
    has_threshold = bool(entry) and entry.get("threshold_operator") is not None \
        and entry.get("threshold_number") is not None
    if not has_threshold:
        return {"status": "no_policy", "spread": spread, "values": values,
                "note": "value(s) computed; no approved interpretation threshold exists, "
                        "so no verdict is authorised"}
    verdicts = {dictionary.judge(v, entry)[0] for v in values}
    if len(verdicts) == 1:
        return {"status": "agreed", "verdict": verdicts.pop(), "spread": spread,
                "explanation": dictionary.judge(values[0], entry)[1]}
    return {"status": "conflicting", "spread": spread, "values": values,
            "note": "candidate conventions straddle the approved threshold - blocking on "
                    "review rather than picking one"}
