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

The prompt is domain-free by design - see planner.py. A candidate carries its
own required facts, including any the evidence plan missed, so a fact that only
one convention needs can still be located.
"""
import json
import re

from .. import dictionary, llm
from ..retrieval.planner import formula_identifiers
from .fact_binding import to_identifier

CANDIDATE_SYSTEM_PROMPT = """\
You are proposing possible calculation methods for a named analytical concept.

You do not have access to the source documents. Work from the definition of
the concept using general domain knowledge.

A preliminary evidence plan will be provided. It was produced without
knowledge of any calculation method, so its facts may be incomplete or wrong.
It is a hint, not a specification.

The established definition of the concept takes precedence over that list. If
the definition requires a fact the list omits, or excludes one the list
includes, follow the definition and identify the facts you actually need, so
they can be retrieved later. A method that quietly reduces to a different,
more convenient concept is wrong even when every listed fact was used.

Your proposals are unapproved candidates. Do not select an authoritative
method and do not imply that organizational approval exists.

Return ONLY one valid JSON object:

{
  "candidates": [
    {
      "candidate_id": "<stable snake_case identifier>",
      "method_name": "<short descriptive name>",
      "formula": "<arithmetic expression>",
      "required_facts": [
        {
          "id": "<snake_case identifier used by the formula>",
          "description": "<what the fact represents>",
          "content_anchors": [
            "<likely verbatim row, column, field, or parameter label>"
          ],
          "period_role": "<omit for a single-period fact; otherwise a short "
                          "label distinguishing which period this is, e.g. "
                          "'prior' vs 'current', or 'fy2021' vs 'fy2022'>"
        }
      ],
      "rationale": "<why or when this method is defensible>"
    }
  ]
}

GUIDELINES:

- Return materially distinct, genuinely defensible methods. Two methods that
  differ only in how the same inclusion is expressed are NOT distinct: adding
  three components and adding two of the same three is one method, spelled
  twice. Distinct methods disagree about WHICH source values belong.
- Every candidate must still be the requested concept. A related but different
  measure is not an alternative convention for this one, and offering it makes
  the candidates disagree for a reason that has nothing to do with ambiguity.
- Do not claim that the candidates are exhaustive.
- Do not rank or choose among candidates.
- Do not add alternatives merely to increase the candidate count.
- If there is one generally accepted calculation, return one candidate.
- If no calculation applies, return an empty `candidates` array.
- Every identifier in a formula must appear in that candidate's
  `required_facts`.
- Every required fact must be a single value the source prints. If a term in
  your formula would itself have to be computed from other printed values,
  expand it into those values rather than naming the aggregate. A term that
  cannot be read directly off the source can never be bound, and the whole
  candidate then evaluates to nothing.
- Formulas may use only required-fact identifiers, numeric constants,
  parentheses, and the operators +, -, *, and /.
- Do not use function calls or identifiers containing spaces.
- Do not calculate numerical results.
- Name each required fact for the quantity only. Do not build the subject,
  date, period, or version into the ID - EXCEPT that a formula needing the
  same quantity from two or more periods (an average across fiscal years is
  the common case) genuinely needs distinct ids, since each becomes its own
  variable in the formula. When that happens, give each a short distinguishing
  id AND set `period_role` on both - the id can vary however is natural, but
  `period_role` is what downstream binding trusts to know these are
  deliberately different periods of the same thing, not a binding error. Every
  fact that is NOT part of a multi-period group should omit `period_role`.
- Content anchors must be labels written exactly as the source would print
  them, containing no subject name, date, period, or qualifier - sources
  print a label once and reuse it across every column.
- An anchor naming the concept being calculated is almost always wrong. Ask
  what the source prints, not what the method is called.
- An empty anchor list is better than an invented label.
- Return strictly valid JSON without markdown or explanatory text.
"""


def _normalize_formula(formula: str) -> str:
    """Safety net for identifiers that arrive with spaces despite the
    instruction: "Total current assets" is not parseable as an expression."""
    return re.sub(r"[A-Za-z_][A-Za-z_0-9 ]*[A-Za-z_0-9]",
                  lambda m: to_identifier(m.group(0)), formula or "")


def validate_candidate(candidate: dict) -> str:
    """Returns an error string, or "" when the candidate is well formed.

    The prompt asks that every formula identifier be declared in
    `required_facts`; this checks it rather than trusting it. An undeclared
    identifier means the fact has no anchors and nothing will ever bind it, so
    the candidate can only produce an unbound-fact error later - better to
    reject it here with a reason than to fail obscurely at evaluation.
    """
    formula = candidate.get("formula") or ""
    if not formula.strip():
        return "no formula"
    used = formula_identifiers(formula)
    if not used:
        return f"formula {formula!r} references no facts"
    declared = {f.get("id") for f in candidate.get("required_facts") or []
                if isinstance(f, dict)}
    declared |= {f for f in candidate.get("required_facts") or []
                 if isinstance(f, str)}
    undeclared = used - declared
    if undeclared:
        return (f"formula references {sorted(undeclared)} which are not declared "
                f"in required_facts")
    return ""


def propose_candidates(concept: str, question: str, plan: dict,
                       model: str = None) -> tuple:
    """Returns (accepted, rejected). Rejected candidates are kept so the trace
    can show that something was proposed and why it was not used."""
    user = json.dumps({
        "concept": concept,
        "question": question,
        "preliminary_evidence_plan": {
            "required_facts": plan.get("required_facts") or [],
            "answer_kind": plan.get("answer_kind"),
        },
    }, indent=1)
    raw = llm.chat_json(CANDIDATE_SYSTEM_PROMPT, user, model=model, stage="candidate").get("candidates", [])

    accepted, rejected = [], []
    for candidate in raw:
        candidate["formula"] = _normalize_formula(candidate.get("formula", ""))
        for fact in candidate.get("required_facts") or []:
            if isinstance(fact, dict) and fact.get("id"):
                fact["id"] = to_identifier(fact["id"])
        problem = validate_candidate(candidate)
        if problem:
            candidate["rejected_because"] = problem
            rejected.append(candidate)
        else:
            accepted.append(candidate)
    return accepted, rejected


def candidate_facts(candidates: list) -> list:
    """Fact objects across all candidates, de-duplicated by id - these carry
    the anchors for facts the evidence plan did not think to ask for."""
    seen, out = set(), []
    for candidate in candidates:
        for fact in candidate.get("required_facts") or []:
            if not isinstance(fact, dict):
                fact = {"id": str(fact), "content_anchors": []}
            fact_id = fact.get("id")
            if fact_id and fact_id not in seen:
                seen.add(fact_id)
                out.append(fact)
    return out


def compute(entry, candidates: list, facts: dict) -> dict:
    result = {"governed": entry is not None, "computed": [], "errors": []}
    specs = ([{"method_name": "approved", "candidate_id": "approved",
               "formula": entry["interpretation"]["formula"]}]
             if entry else candidates)
    for spec in specs:
        formula = spec.get("formula", "")
        name = spec.get("method_name") or spec.get("candidate_id") or "unnamed"
        try:
            result["computed"].append({
                "label": name,
                "candidate_id": spec.get("candidate_id"),
                "formula": formula,
                "rationale": spec.get("rationale"),
                "value": round(dictionary.evaluate_formula(formula, facts), 6),
            })
        except dictionary.FormulaError as e:
            result["errors"].append({"label": name, "formula": formula,
                                     "error": str(e)})
    return result
