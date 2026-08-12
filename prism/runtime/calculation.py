"""Producing values from bound facts.

Governed concept   -> the approved formula, evaluated deterministically.
Ungoverned concept -> one or more explicitly labeled candidate interpretations,
                      also evaluated deterministically.

Either way the arithmetic happens in code. The model's role is to propose
conventions, never to compute them - which is the asymmetry that makes "learns
as it goes" compatible with "doesn't drift".

The candidate set is never claimed exhaustive. No system can establish that it
enumerated every defensible convention, and the mechanism doesn't need it to:
two candidates disagreeing is already enough to detect ambiguity.
"""
import json
import re

from .. import dictionary, llm
from .fact_binding import to_identifier

CANDIDATE_SYSTEM_PROMPT = (
    "You are proposing how a named financial concept is computed. Several conventions are "
    "often defensible; your job is to surface them, clearly labeled, NOT to pick one.\n\n"
    "Work from the DEFINITION OF THE CONCEPT FIRST, using your own domain knowledge. You "
    "will also be shown line items that a preliminary evidence plan happened to locate - "
    "that list is frequently INCOMPLETE and must not constrain you. If the correct "
    "definition needs a line item missing from it, use that line item anyway as a new "
    "snake_case identifier; it will be looked up afterwards. Never bend a definition to "
    "fit the facts already on hand: proposing the current ratio when asked for the quick "
    "ratio, merely because inventory was not in the list, is exactly the failure this "
    "step exists to prevent.\n\n"
    "Respond with exactly one JSON object:\n"
    '{"candidates": [{"label": "short name for this convention", "formula": "an '
    'arithmetic expression", "rationale": "one sentence on who uses this convention"}]}\n\n'
    "Formulas are arithmetic over snake_case identifiers combined with + - * / and "
    "parentheses. No function calls, no identifiers containing spaces.\n\n"
    "Propose every convention you consider genuinely defensible - typically 1 to 3. Do "
    "not pad the list, and do not claim to have enumerated every possibility."
)


def propose_candidates(concept: str, question: str, known_facts: list,
                       model: str = None) -> list:
    user = (f"CONCEPT: {concept}\nQUESTION: {question}\n"
            f"LINE ITEMS A PRELIMINARY PLAN LOCATED (possibly incomplete, do not treat "
            f"as the full vocabulary): {json.dumps(known_facts)}")
    candidates = llm.chat_json(CANDIDATE_SYSTEM_PROMPT, user, model=model,
                               timeout=60).get("candidates", [])
    for candidate in candidates:
        # Safety net: the prompt asks for snake_case, but a stray "Total current
        # assets" would make the formula unparseable.
        candidate["formula"] = re.sub(
            r"[A-Za-z_][A-Za-z_0-9 ]*[A-Za-z_0-9]",
            lambda m: to_identifier(m.group(0)), candidate.get("formula", ""))
    return candidates


def compute(entry, candidates: list, facts: dict) -> dict:
    result = {"governed": entry is not None, "computed": [], "errors": []}
    specs = ([{"label": "approved", "formula": entry["interpretation"]["formula"]}]
             if entry else candidates)
    for spec in specs:
        formula = spec.get("formula", "")
        try:
            result["computed"].append({
                "label": spec.get("label", "unnamed"),
                "formula": formula,
                "rationale": spec.get("rationale"),
                "value": round(dictionary.evaluate_formula(formula, facts), 6),
            })
        except dictionary.FormulaError as e:
            result["errors"].append({"label": spec.get("label"), "formula": formula,
                                     "error": str(e)})
    return result
