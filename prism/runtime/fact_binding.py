"""Binding retrieved text to named facts, with provenance.

Retrieval returning a table is not the same as knowing which cell means what.
Binding asserts "15,754 IS total_current_assets, for 3M, June 30 2023 column,
in USD millions, page 5" - and every one of those claims can be wrong
independently. Picking the wrong column is not an arithmetic error and no
amount of validating the arithmetic will catch it.
"""
import json
import re

from .. import llm

BINDER_SYSTEM_PROMPT = """\
You are binding named facts to specific values found in source excerpts.

Each requested fact is a snake_case identifier naming a quantity that appears
in the excerpts under some printed label. Locate the single value each one
refers to and report where it came from.

Report only values printed in the excerpts. Do not compute anything, do not
derive one fact from another, and do not convert units.

Return ONLY one valid JSON object:

{
  "facts": [
    {
      "name": "<the requested identifier, echoed exactly>",
      "value": <number, no separators or symbols>,
      "units": "<unit and scale as printed>",
      "period": "<the column, period, or variant this value sits under>",
      "row_label": "<the printed label, verbatim>",
      "entity": "<the subject this value belongs to>",
      "source_page": <page number>
    }
  ]
}

GUIDELINES:

- Echo each requested identifier exactly; do not rename or normalise it.
- If a requested fact is genuinely not present, omit it. Do not guess and do
  not substitute a related quantity.
- Sources commonly print several columns or variants side by side. Selecting
  the wrong one produces a silently wrong answer, so identifying the correct
  column matters as much as identifying the correct row.
- Bind all facts from the same column or period unless the question asks
  otherwise.
- Return strictly valid JSON without markdown or explanatory text.
"""


def to_identifier(name: str) -> str:
    """Fact names travel as snake_case because they end up inside formulas. The
    planner emits human labels ("Total current assets"), which are not parseable
    as expressions - every candidate formula failed on this until normalized."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", (name or "").lower())).strip("_")


def format_chunks(chunks: list) -> str:
    return "\n\n---\n\n".join(
        f"[{i + 1}] {'[TABLE] ' if c.get('type') == 'table' else ''}"
        f"page {c.get('page')}\n{c.get('text', '')}"
        for i, c in enumerate(chunks))


def bind_facts(question: str, required: list, chunks: list, model: str = None) -> list:
    if not required:
        return []
    user = (f"QUESTION: {question}\n\nFACTS TO BIND: {json.dumps(required)}\n\n"
            f"EXCERPTS:\n{format_chunks(chunks)}")
    return llm.chat_json(BINDER_SYSTEM_PROMPT, user, model=model,
                         stage="binder").get("facts", [])


def validate_bindings(bound: list, chunks: list) -> list:
    """Grounding + period consistency. Checks the VALUE against the source text
    rather than the model's cited snippet, for the same reason the catalog does:
    a sloppy citation of a correct value is not a hallucination, and a plausible
    citation of an invented value is."""
    haystack = " ".join((c.get("text") or "") for c in chunks)
    periods = {}
    for fact in bound:
        issues = []
        value = fact.get("value")
        if value is None:
            issues.append("no value bound")
        else:
            n = abs(float(value))
            printed = f"{n:,.0f}" if float(n).is_integer() else f"{n:,}"
            # Filings print 15,754 / (1,749) / 15754 - accept any of those.
            if not any(v in haystack for v in
                       {printed, printed.replace(",", ""), f"{n:g}"}):
                issues.append(f"value {value} not found verbatim in retrieved text")
        if not fact.get("period"):
            issues.append("no period/column bound")
        else:
            periods.setdefault(fact["period"], []).append(fact.get("name"))
        fact["grounded"] = not issues
        fact["binding_issues"] = issues

    if len(periods) > 1:
        # Facts drawn from different columns produce a number that is wrong in a
        # way no arithmetic check can detect, so the whole set is rejected.
        for fact in bound:
            fact["binding_issues"].append(f"facts bound to mixed periods: {sorted(periods)}")
            fact["grounded"] = False
    return bound


def grounded_facts(bound: list) -> dict:
    return {f["name"]: f["value"] for f in bound
            if f.get("grounded") and f.get("value") is not None}
