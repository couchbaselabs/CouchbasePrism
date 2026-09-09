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

One formula per entry, by design (the user's own rule) - multiple approved
conventions for the same concept coexist as separate entries sharing a
canonical_name, distinguished by `formula_type` ("primary" vs "alternate",
open-ended past those two). matching.find_metric() prefers the entry tagged
"primary" when more than one approved entry matches - see that module.
"""
import re

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


def build_metric_entry(metric: str, user_friendly_formula: str, formula_type: str = "primary",
                       abbreviation: str = None, aliases: list = None,
                       threshold_operator: str = None, threshold_number: float = None,
                       context: str = None, domain: str = "finance",
                       document_types: list = None, model: str = None) -> tuple:
    """Returns (metric_entry, policy_entry_or_None) - not yet saved.

    id is suffixed by formula_type for anything but "primary", so multiple
    conventions for one concept get distinct ids instead of colliding (and
    replacing each other) the way approve()'s single-entry-per-concept
    semantics would.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", metric.lower()).strip("_")
    suffix = "" if formula_type == "primary" else "_" + re.sub(r"[^a-z0-9]+", "_",
                                                               formula_type.lower()).strip("_")
    metric_id = f"{domain}.{slug}{suffix}"

    compiled = compile_formula(metric, user_friendly_formula, model=model)
    formula = compiled["formula"]
    required_facts = formula_facts(formula)
    anchors = [f["printed_label"] for f in compiled["facts"] if f.get("printed_label")]

    entry = {
        "id": metric_id,
        "entry_type": "metric",
        "scope": {"domain": domain, "document_types": document_types or ["10-K", "10-Q"]},
        "recognition": {
            "canonical_name": metric.lower(),
            "aliases": ([abbreviation] if abbreviation else []) + (aliases or []),
        },
        "interpretation": {
            "formula": formula,
            "required_facts": required_facts,
            "user_friendly_formula": user_friendly_formula,
            "content_anchors": anchors,
        },
        "formula_type": formula_type,
        "governance": {"status": "approved", "source": "human_review", "version": 1},
    }
    if context:
        entry["context"] = context

    policy_entry = None
    if threshold_operator and threshold_number is not None:
        policy_entry = {
            "id": f"{metric_id}.policy",
            "entry_type": "interpretation_policy",
            "applies_to": metric_id,
            "scope": {"domain": domain},
            "policy": {"operator": threshold_operator, "value": float(threshold_number)},
            "governance": {"status": "approved", "source": "human_review", "version": 1},
        }
    return entry, policy_entry
