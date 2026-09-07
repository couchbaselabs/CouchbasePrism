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
import re

from .. import llm

PLANNER_SYSTEM_PROMPT = """\
You are planning the source evidence needed to answer a user query.

You do not have access to the source documents. Use general knowledge of how
the likely source material is structured, but do not invent facts, values, or
source labels.

Return ONLY one valid JSON object:

{
  "concept": "<canonical name of the requested concept>",
  "answer_kind": "<exactly one of: stated_fact, derived_metric, judgment, attribution>",
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
  This is the answer_kind even when the question ALSO asks you to explain
  what drove a change in that metric ("Calculate X for FY2023, and explain
  the primary driver of its decline versus FY2022") - the calculation is not
  optional just because an explanation was also requested. In that compound
  case, required_facts must include BOTH the formula's own direct inputs AND
  the components or line items the source uses to explain the change (see
  the attribution guidance below for how to name those).
- Use `judgment` when the question asks for an assessment, classification,
  sufficiency determination, or threshold-based conclusion about a value that
  is already known or stated, not one that must first be calculated.
- Use `attribution` when the question asks what caused, drove, or explains
  something, the source is expected to state that explanation itself rather
  than requiring it to be derived, AND the question does not also ask you to
  calculate or compute a value from source figures - if it does, that makes
  the whole question `derived_metric` (see above), even though an explanation
  is also wanted. Choose `attribution` over `judgment` when the answer is an
  explanation to be reported, not a value to be assessed. A question may ask
  for an explanation and also ask whether a measure is meaningful (an
  assessment, not a calculation); that combination is still `attribution`.
- Include only facts reasonably necessary to answer the question, with one
  exception stated below.
- When the answer must be derived or assessed, list the source quantities you
  can confidently identify as direct inputs. Do not list the derived quantity
  itself as a required fact.
- For `attribution`, and for the explanatory half of a compound
  `derived_metric` question, the required facts are the components or line
  items whose movement the source uses to explain the change, not inputs to
  a formula.
- THE EXCEPTION TO MINIMALITY: where a derived metric has multiple materially
  distinct, established calculation conventions, include the union of the
  source-recorded facts those conventions need - not the inputs of whichever
  one you would choose. A fact that only one convention uses still belongs, and
  a shorter list is wrong here. Do not select, rank, or describe the formulas.
  Do not include inputs for merely related but different metrics.
- A required fact must be a single value the source prints on one line. If
  naming it would require combining, netting, or excluding other values, it is
  not a fact - list the printed components instead. Widely used analytical
  aggregates are still aggregates: they have names because analysts compute
  them, not because sources print them.
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


_HASH_PHRASE = re.compile(r"#([^#]+)#")


def extract_phrase_terms(question: str) -> tuple:
    """Pulls #...#-delimited spans out of a question as forced exact-phrase
    search terms - "How much did #John Doe# earn..." yields a cleaned
    question with the hashes stripped ("How much did John Doe earn...") and
    ["John Doe"] as a phrase to match verbatim, not just contribute words to
    the usual bag-of-terms leg.

    The hashes are removed but the words themselves stay in place - the
    embedding model and the LLM planner both see ordinary text, unaware
    anything was marked. Only hybrid_search's lexical clause sees the
    extracted phrase separately, as an additional match_phrase disjunct
    (Path A: folded into the existing lexical leg's score, not a separately
    fused channel - see the "why RRF" discussion this was measured against
    for match_phrase-per-anchor, a different and worse-measured case: a
    generic recurring caption printed on many pages, not a specific name).

    This runs BEFORE plan_evidence(), same as everything else in this
    module - the planner never sees the raw hash marks either way.
    """
    phrases = [m.group(1).strip() for m in _HASH_PHRASE.finditer(question)]
    phrases = [p for p in phrases if p]
    cleaned = _HASH_PHRASE.sub(lambda m: m.group(1), question)
    return cleaned, phrases


def plan_evidence(question: str, model=None, source_context: str = None) -> dict:
    """`source_context` describes the resolved document in the corpus's own
    terms - its type and period, taken from the catalog. The prompt template
    stays domain-free; the domain-specific value arrives as data.

    Worth knowing before tuning this: on gpt-4o-mini it measurably HURT.
    Over 24 plans, dead anchors rose from 62% to 66% and it degraded a question
    that had been passing. The model already infers the genre from the question
    wording; what it cannot infer is which strings the document prints. Kept
    because a stronger model may use it differently, and because it is now
    measurable per model rather than assumed either way.
    """
    user = (f"SOURCE CONTEXT: {source_context}\n\nQUERY: {question}"
            if source_context else question)
    return llm.chat_json(PLANNER_SYSTEM_PROMPT, user, model=model, stage="planner")


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
