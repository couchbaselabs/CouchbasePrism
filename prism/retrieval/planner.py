"""The evidence planner.

Runs BEFORE any dictionary lookup and needs no dictionary entry to work. That
ordering is the claim ADR-0001 rests on, and it is why retrieval can succeed
for concepts the system has never seen.

The prompt is deliberately DOMAIN-FREE. It never names a document type, an
industry or a corpus, because the moment it does, PRISM stops being a reference
architecture and becomes a product for one vertical. Anything domain-specific
belongs in the dictionary, which is data, not in a prompt, which is code.

Its most valuable output is `content_anchors` - literal labels expected to
appear verbatim in the source. Those sidestep chunk section titles, which are
unreliable in real corpora. Anchors hang off each required fact rather than off
the plan as a whole, so it is clear which anchor is meant to locate which fact.
"""
import ast

from .. import llm

PLANNER_SYSTEM_PROMPT = """\
You are planning the source evidence needed to answer a user query.

You do not have access to the source documents. Use general knowledge of how
the likely source material is structured, but do not invent facts, values, or
source labels.

Return ONLY one valid JSON object:

{
  "concept": "<canonical name of the requested concept>",
  "answer_kind": "<exactly one of: stated_fact, derived_metric, judgment>",
  "required_facts": [
    {
      "id": "<snake_case identifier>",
      "description": "<what the fact represents>",
      "content_anchors": [
        "<likely verbatim row, column, field, or parameter label>"
      ]
    }
  ],
  "preferred_artifacts": [
    "<likely source format, such as table, text, diagram, chart, or log>"
  ]
}

GUIDELINES:

- `concept` is the single measure or quantity whose value determines the
  answer. When the question names a specific measure to use, `concept` is
  that measure, even where the question frames it as a broader assessment.
- Write `concept` as ordinary lower-case words, as it would appear in a
  glossary, not as an identifier.
- Exclude the subject, entity, date, and period from `concept`. Questions about
  the same underlying concept should produce the same canonical name.
- Use `stated_fact` when the answer should be present directly in the source.
- Use `derived_metric` when source values must be combined or transformed.
- Use `judgment` when the question asks for an assessment, classification,
  sufficiency determination, or threshold-based conclusion.
- Include only facts reasonably necessary to answer the question.
- When the answer must be derived or assessed, list the source quantities you
  can confidently identify as direct inputs. Do not list the derived quantity
  itself as a required fact.
- Do not speculate about alternative calculation conventions. A later
  calculation-planning stage may add further required facts.
- Fact IDs must be unique snake_case identifiers that name only the quantity.
- Do not include the subject, entity, date, period, or version in a fact ID.
  Those dimensions are resolved separately when the value is located.
- Content anchors must be labels reasonably expected to appear verbatim in the
  source, written exactly as the source would print them.
- Do not add the entity name, requested date, fiscal period, or other
  question-specific context unless it is genuinely part of the printed label.
- Preserve qualifiers that belong to the printed label, such as "net",
  "current", "diluted", or "continuing operations".
- Do not use abstract topics, inferred section descriptions, calculation
  names, or paraphrases as anchors.
- An anchor naming a derived concept is usually wrong unless that concept is
  expected to be stated directly in the source.
- Return only anchors you are reasonably confident about. An empty anchor list
  is better than an invented label.
- Do not propose formulas, perform calculations, or make the final judgment.
- Return strictly valid JSON without markdown or explanatory text.
"""


def plan_evidence(question: str, model: str = None) -> dict:
    return llm.chat_json(PLANNER_SYSTEM_PROMPT, question, model=model, timeout=60,
                          stage="planner")


def _as_fact(entry) -> dict:
    """Tolerate a bare string where an object was asked for. Models drift back
    to the simpler shape occasionally, and a plan that loses its anchors is
    better than a run that crashes."""
    if isinstance(entry, dict):
        return entry
    return {"id": str(entry), "description": "", "content_anchors": []}


def plan_facts(plan: dict) -> list:
    return [_as_fact(f) for f in (plan.get("required_facts") or [])]


def fact_ids(plan: dict) -> list:
    return [f.get("id") for f in plan_facts(plan) if f.get("id")]


def plan_anchors(plan: dict) -> list:
    """Every anchor across every required fact, de-duplicated, order preserved.

    Retrieval matches anchors against chunk text without needing to know which
    fact each belongs to - the association matters for diagnosis and for
    re-retrieving a specific missing fact, not for the search itself.
    """
    seen, out = set(), []
    for fact in plan_facts(plan):
        for anchor in fact.get("content_anchors") or []:
            if anchor and anchor not in seen:
                seen.add(anchor)
                out.append(anchor)
    # A plan that put anchors at the top level is still usable.
    for anchor in plan.get("content_anchors") or []:
        if anchor and anchor not in seen:
            seen.add(anchor)
            out.append(anchor)
    return out


def formula_identifiers(formula: str) -> set:
    """Which bound facts a formula references - used to widen the binding set
    beyond what the planner thought to ask for."""
    try:
        return {n.id for n in ast.walk(ast.parse(formula, mode="eval"))
                if isinstance(n, ast.Name)}
    except SyntaxError:
        return set()
