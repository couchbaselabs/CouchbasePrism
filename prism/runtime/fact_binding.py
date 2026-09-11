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
    # period_role is the real signal: pipeline.py attaches it from the
    # candidate/planner's own fact spec BEFORE this runs, so a fact whose
    # requester explicitly declared "this is deliberately a different period"
    # is trusted outright. _period_siblings() is a fallback for anything
    # that arrives without one - governed facts from an approved dictionary
    # entry currently have no period_role at all, since dictionary.yaml only
    # stores bare identifiers.
    exempt = {f["name"] for f in bound if f.get("period_role") and f.get("name")}
    exempt |= _period_siblings(bound)
    periods = {}
    for fact in bound:
        issues = []
        value = fact.get("value")
        if value is None:
            issues.append("no value bound")
        else:
            try:
                n = abs(float(value))
            except (TypeError, ValueError):
                # The prompt asks for a bare number; nothing enforces it. A
                # dense multi-row table (e.g. several segments sharing a page)
                # is exactly where the model can bind a row label instead of
                # its value. One bad fact should not crash the whole batch -
                # it degrades to ungrounded like any other failed check.
                issues.append(f"value {value!r} is not numeric")
                n = None
            if n is not None:
                printed = f"{n:,.0f}" if float(n).is_integer() else f"{n:,}"
                # Filings print 15,754 / (1,749) / 15754 - accept any of those.
                if not any(v in haystack for v in
                           {printed, printed.replace(",", ""), f"{n:g}"}):
                    issues.append(f"value {value} not found verbatim in retrieved text")
        if not fact.get("period"):
            issues.append("no period/column bound")
        elif fact.get("name") not in exempt:
            periods.setdefault(fact["period"], []).append(fact.get("name"))
        fact["grounded"] = not issues
        fact["binding_issues"] = issues

    if len(periods) > 1 and _distinct_time_periods(periods):
        # Facts drawn from different columns produce a number that is wrong in a
        # way no arithmetic check can detect, so the whole set is rejected -
        # excluding facts a same-quantity sibling already showed are meant to
        # span periods. _distinct_time_periods() is the gate that keeps this
        # from firing on facts that are merely in different presentation
        # columns of the SAME period (see its own docstring) - without it,
        # 3M's own GAAP/adjusted-for-special-items columns looked identical
        # to a genuine cross-period mismatch.
        for fact in bound:
            if fact.get("name") not in exempt:
                fact["binding_issues"].append(f"facts bound to mixed periods: {sorted(periods)}")
                fact["grounded"] = False
    return bound


_YEAR_TOKEN = re.compile(r"^(fy)?(19|20)\d{2}$", re.I)
# Deliberately small and generic - words that describe WHICH period, never a
# quantity. "current" alone would wrongly match current_assets/
# current_liabilities if this were checked against the whole name (that pair
# shares "current" as a leading token, not as the diverging one) - it is only
# ever checked against the token where two same-prefix names diverge, below.
_PERIOD_TOKENS = {"beginning", "ending", "opening", "closing", "prior",
                  "current", "previous", "preceding", "balance", "year",
                  "period"}


def _looks_like_period(token: str) -> bool:
    return bool(token) and (bool(_YEAR_TOKEN.match(token)) or token.lower() in _PERIOD_TOKENS)


_YEAR_IN_TEXT = re.compile(r"(19|20)\d{2}")


def _distinct_time_periods(period_labels) -> bool:
    """True only if the distinct `period` labels bound actually carry some
    signal of different POINTS IN TIME - a year, a quarter marker, or a
    generic before/after word (_PERIOD_TOKENS) - not just different
    presentation-column wording for the SAME period.

    Observed false positive (3M_2023_10K, "Operating income margin"): facts
    bound to "Total Company GAAP amounts" and "Total Company Adjustments for
    special items" - two COLUMNS of 3M's own non-GAAP reconciliation table
    for one fiscal year, not two fiscal years - were rejected as mixed
    periods, so every fact in the set came back ungrounded and nothing was
    ever computed. Neither label contains a year, a quarter, or any of
    _PERIOD_TOKENS - nothing in either string actually claims a different
    time, so the mismatch below never should have fired.

    This checks the LABELS, not the fact names - deliberately a different
    axis from _period_siblings() below, which exempts a same-quantity fact
    pair that legitimately spans periods by NAME. The two are
    complementary: this stops a false "mixed periods" claim before it's
    made; _period_siblings() forgives a true one already made."""
    labels = list(period_labels)
    years = {m.group(0) for label in labels for m in _YEAR_IN_TEXT.finditer(label)}
    if len(years) > 1:
        return True
    quarters = {q.upper() for label in labels for q in re.findall(r"\bQ[1-4]\b", label, re.I)}
    if len(quarters) > 1:
        return True
    tokens = {tok.lower() for label in labels for tok in re.split(r"[^A-Za-z0-9]+", label)}
    return bool(tokens & _PERIOD_TOKENS)


def _period_siblings(bound: list) -> set:
    """Fact names that have a same-quantity sibling bound to a DIFFERENT
    period - INTENTIONALLY multi-period (a formula naming two years of the
    same quantity), not a binding accident, however the model happened to
    spell the distinguishing part. Observed so far, three different
    vocabularies for the identical pattern: total_assets_fy2021/fy2022,
    total_assets_2021/2022, inventories_beginning_balance/ending_balance.

    Two names are siblings if their underscore-tokens share a leading prefix
    AND the first token where they diverge looks like a period marker on at
    least one side (a year, or a small set of generic before/after words) -
    both conditions matter. Prefix alone is not enough: current_assets and
    current_liabilities share "current" as their leading token, and treating
    that as "the same quantity, two periods" would silently defeat the very
    check this replaces - current_assets bound to 2021 alongside
    current_liabilities bound to 2022 for what should be a single-period
    ratio is exactly the accidental mismatch the original guard existed to
    catch. Checking the DIVERGING token ("assets" vs "liabilities", neither
    period-like) rather than the shared one is what tells them apart from
    total_assets_fy2021 vs total_assets_fy2022 (diverging on "fy2021" vs
    "fy2022", both year-like).

    This assumes the quantity comes first and the period-qualifier is a
    trailing suffix, the convention every observed case has followed. A
    qualifier placed BEFORE the quantity ("opening_inventory" vs
    "closing_inventory") would not be caught and would need a different
    signal.
    """
    named = [f for f in bound if f.get("name")]
    tokens = {f["name"]: f["name"].split("_") for f in named}
    siblings = set()
    for f in named:
        for g in named:
            if g is f or g.get("period") == f.get("period"):
                continue
            a, b = tokens[f["name"]], tokens[g["name"]]
            shared = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                shared += 1
            if shared == 0:
                continue
            a_next = a[shared] if shared < len(a) else None
            b_next = b[shared] if shared < len(b) else None
            if _looks_like_period(a_next) or _looks_like_period(b_next):
                siblings.add(f["name"])
                break
    return siblings


def grounded_facts(bound: list) -> dict:
    return {f["name"]: f["value"] for f in bound
            if f.get("grounded") and f.get("value") is not None}
