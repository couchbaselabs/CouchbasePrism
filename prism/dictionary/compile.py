"""Compiling a human-authored formula into an approved dictionary entry.

The form a user fills in (metric name, alias, primary/alternate, a formula
written in ordinary words, a threshold) never asks for the row/column labels
a real filing prints - that would just be re-inventing the same
"content_anchors" reasoning retrieval/planner.py already does for the
evidence plan, asked of a human instead of a model. compile_metric_entry()
asks a model instead, under the SAME discipline planner.py's prompt already
established: structural knowledge only (how a quantity is conventionally
labeled), never a specific recalled fact - the anchors are a genre-level
guess about labeling convention, not a claim about any document's actual
printed numbers, so the same anti-hallucination guardrail applies for the
same reason.

One formula per entry, by design (the user's own rule) - multiple
conventions for the same concept coexist as separate entries sharing a
`metric` name, distinguished by `formula_type` ("Preferred" vs "Alternate",
open-ended past those two). matching.find_metric() prefers the entry tagged
"Preferred" when more than one entry matches, unless the plan's own
formula_preference names a different one explicitly - see that module.

The document key is a UUID, not a slug derived from the metric name: this
is meant to be created from a form (metric name, alias, primary/alternate,
a formula in ordinary words, a threshold), and a form doesn't compute a
stable id from what a person just typed - it mints one.
"""
import uuid

from .. import llm
from .evaluator import FormulaError, formula_facts

COMPILE_SYSTEM_PROMPT = """\
You are converting a human-readable financial formula into an executable \
one, for a governed dictionary entry a domain expert has already decided \
to approve - your job is translation, not judgment of whether the formula \
is correct.

Given a metric name and a formula written in ordinary words (e.g. \
"(Current Assets - Inventories) / Current Liabilities"), return:
1. The SAME formula rewritten as a snake_case arithmetic expression - only \
+, -, *, / and parentheses, one identifier per distinct quantity.
2. For each identifier, the label that quantity is CONVENTIONALLY printed \
as on a real financial statement - structural knowledge only (how such a \
line is typically labeled), never a fact you recall about any specific \
company's actual filed numbers.

Return ONLY one JSON object:
{
  "formula": "<snake_case arithmetic expression>",
  "facts": [{"id": "<snake_case identifier used in formula>",
             "printed_label": "<label as conventionally printed>"}]
}

GUIDELINES:
- Use the SAME identifier for the same underlying quantity wherever it
  recurs in the formula.
- printed_label is the CONVENTIONAL row/line label a filing prints, not a
  paraphrase - e.g. "Total current assets", not "current assets total" or
  "current assets".
- Return strictly valid JSON, no markdown, no explanation outside the
  object.
"""


def compile_formula(metric: str, user_friendly_formula: str, model: str = None) -> dict:
    """{"formula": "...", "facts": [{"id", "printed_label"}, ...]}. Raises
    FormulaError if the model's own formula does not parse as arithmetic -
    fail loud rather than store an entry evaluate_formula() would later
    reject at answer time, when the failure is much harder to trace back to
    its source."""
    user = f"METRIC: {metric}\nFORMULA: {user_friendly_formula}"
    result = llm.chat_json(COMPILE_SYSTEM_PROMPT, user, model=model, stage="catalog")
    formula = result.get("formula") or ""
    formula_facts(formula)  # raises FormulaError on anything not arithmetic
    return {"formula": formula, "facts": result.get("facts") or []}


def build_metric_entry(metric: str, user_friendly_formula: str, formula_type: str = "Preferred",
                       abbreviation: str = None, threshold_operator: str = None,
                       threshold_number: float = None, context: str = None,
                       model: str = None) -> dict:
    """One flat, self-contained entry - not yet saved. Every field but
    `formula`, `required_facts` and `bm25_table_anchors` maps directly to a
    form field (metric, abbreviation, formula_type, user_friendly_formula,
    threshold_operator, threshold_number, context); those three are the ones
    a form never asks for, generated here instead.
    """
    compiled = compile_formula(metric, user_friendly_formula, model=model)
    formula = compiled["formula"]
    required_facts = formula_facts(formula)
    anchors = [f["printed_label"] for f in compiled["facts"] if f.get("printed_label")]

    entry = {
        "id": str(uuid.uuid4()),
        "metric": metric,
        "formula_type": formula_type,
        "user_friendly_formula": user_friendly_formula,
        "formula": formula,
        "required_facts": required_facts,
        "bm25_table_anchors": {"target_strings": anchors},
    }
    if abbreviation:
        entry["abbreviation"] = abbreviation
    if context:
        entry["context"] = context
    if threshold_operator and threshold_number is not None:
        entry["threshold_operator"] = threshold_operator
        entry["threshold_number"] = float(threshold_number)
    return entry
