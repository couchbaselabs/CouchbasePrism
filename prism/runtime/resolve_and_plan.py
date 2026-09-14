"""Resolve + Plan + Formula: EXPERIMENTAL single-call redesign.

Branch: experiment-single-llm-resolve-plan-formula. This replaces the
previous three-part combined call (resolution + evidence planning +
concept matching) with a leaner schema-driven prompt, and additionally
folds formula proposal into the SAME call - eliminating
calculation.py's separate CANDIDATE_SYSTEM_PROMPT call entirely for the
ungoverned/derived_metric path.

Why: the previous prompt accumulated several "Observed live: <specific
company, metric, and document>" anecdotes as bugs were root-caused
against this project's own small gold corpus. The rules they illustrated
were general, but stacking several real-corpus anecdotes in a PRODUCTION
prompt reads like the system was patched until the eval passed, not
designed to generalize - a fair concern to take seriously even though the
underlying logic wasn't actually overfit. This rewrite states rules
abstractly, with no corpus-specific examples anywhere.

Formula proposal moving here loses nothing evidence-wise: the CURRENT
propose_candidates() (calculation.py) already runs with zero retrieved
evidence - {concept, question, preliminary_evidence_plan} only, same
"general domain knowledge, no source text" position this call is already
in. Combining them removes a whole LLM round-trip for every
derived_metric question, not a capability.

ITERATION HISTORY: several earlier drafts of this prompt were measured
against the 3M gold set and rejected or revised based on what actually
happened, not just design review:
  - A draft that grouped several formula variables into one required_fact
    (e.g. numerator and denominator as a single fact) broke
    fact_binding, which only ever binds one value per fact id - the
    second variable silently vanished, and every formula depending on
    it went unbound. Reverted; one required_facts entry per printed
    value (rule 7 below) is now explicit rather than left implicit.
  - A draft that required formulas to reference variables by their own
    identifiers (rule 9 below) fixed a real failure mode: the model
    sometimes wrote plain-English formulas ("operating income (loss) /
    net sales") that nothing downstream could match to bound facts.
  - A draft with a single combined `content_anchor` STRING per fact
    (wrapped as a one-element list) broke retrieval outright:
    anchor_search.py matches each anchor as its own independent CONTAINS
    substring, so combining several labels into one phrase demanded the
    source contain that whole phrase verbatim, which real filing text
    essentially never does - confirmed directly against main's own
    (working) search terms on the same question. The CURRENT prompt
    below asks for a single space-separated, deduplicated-by-stem
    `content_anchor` again, but this time the code (see below) splits it
    into individual words before it reaches anchor_search.py, rather
    than passing the whole string as one anchor - a real precision
    tradeoff (single-word anchors are looser than curated multi-word
    labels) this prompt doesn't otherwise resolve, worth watching in
    eval results rather than assuming settled.
  - A draft with a detailed, multi-paragraph `content_anchor` rule full
    of worked examples was flagged as itself gaming the eval the same
    way the ORIGINAL production prompt's "Observed live: <company>"
    anecdotes did - specific enough to fix one gold question's own
    wording rather than stating a general principle. Replaced with the
    terser, example-light rules below.
  - The `doc_types`-guessing guard went through two different fixes:
    first dropped to "leave doc_types empty when the question names no
    form" (a strict no-guess rule), then changed again to "list every
    form that could structurally carry this evidence" (rule 3 below) -
    multi-candidate inclusion instead of either guessing one form or
    guessing none, intended to handle documents docs/findings/
    earnings-8k-not-catalogued-by-period.md describes (an earnings 8-K
    and a 10-Q covering the same period) without excluding either.
  - A period-granularity guard (rule 3) and a proxy-statement filing-lag
    guard (rule 4) were both added directly into these rules the same
    day they were discovered - and both are, in hindsight, SKILLS, not
    structural rules (see docs/adr/0003-skills-a-domain-expert-owned-
    knowledge-layer.md): general filing-mechanics facts a domain expert
    should own, not something an app developer bakes into the compiler
    prompt every time a new filing quirk surfaces. `skills_repo.render()`
    below is the fix - filing-mechanics knowledge belongs there from now
    on, not appended to the numbered rules.

CONCEPTS is deliberately NOT part of this prompt at all - the morning's
separate discussion (concepts collection as a search target, not
something embedded in every prompt) applies here. This experiment stubs
concept matching with a plain substring match against the concepts
collection (see _match_concepts below) rather than the real FTS-search
design discussed - a placeholder, not the intended final mechanism, kept
only so a concept-naming question doesn't silently lose forced-phrase
support during this experiment.
"""
from .. import concepts as concepts_repo
from .. import llm
from .. import skills as skills_repo
from ..catalog import manifest as catalog_manifest
from ..catalog.intent import resolve_documents

RESOLVE_PLAN_FORMULA_SYSTEM_PROMPT = """\
Extract routing and evidence-planning metadata as JSON from the QUESTION and MANIFEST.

1. Output only valid JSON matching the schema. Never compute a value, derive a
   result, or answer the question.

2. `companies`, `doc_types`, `years` and `quarters` must be copied exactly from
   MANIFEST. `[]` means any, not none.

3. Set `doc_types` from what the QUESTION itself names. If it names no form,
   list every form that could structurally carry this evidence - never only
   the most likely one - but exclude any form that cannot structurally cover
   the period granularity the QUESTION itself already states (SKILLS below,
   if present, describes which forms that applies to). This is about the
   period the question names, never a guess about which form a KIND of metric
   typically appears in - that guess is what this rule forbids in the first
   place.

4. `years`: the year the question asks about. A filing's own comparative column
   covers the prior year, so a comparison against the prior year does not add
   that year. SKILLS below, if present, may describe a form whose filing date
   differs from the period it discloses - apply that before setting `years`.

5. `subject_words`: key phrases copied verbatim from the QUESTION, naming its
   subject matter. Empty when the question names no subject matter.

6. `content_anchor`: labels you expect to find PRINTED in the document - row
   captions, table headings, a metric as a filing writes it. Never the
   question's own words with filler removed. Preserve printed qualifiers
   ("net", "current", "diluted", "continuing operations"). Lowercase,
   space-separated, deduplicated by stem. Omit a term rather than invent one.

7. One `required_facts` entry per value or explanation the answer needs. A fact
   is one value a source prints on one line; if naming it requires combining or
   netting other values, list those components instead.

8. `period_role`: omit for a single-period fact. Required whenever a formula
   names the same quantity for two or more periods, so binding knows the
   difference is deliberate.

9. `formulas`: only when `answer_kind` is `derived_metric`, otherwise `[]`.
   Reference required-fact ids exactly. `formula` is the bare expression only
   - never prefixed with a variable name and `=` (write
   `adjusted_free_cash_flow / adjusted_income`, not
   `adjusted_free_cash_flow_conversion = adjusted_free_cash_flow /
   adjusted_income`) - the evaluator parses a single expression, not an
   assignment statement. Return more than one only for materially distinct,
   independently defensible conventions - not rephrasings of one method.
   Never rank them, never choose between them, never claim the set is
   complete. Set `formula_preference` only when the QUESTION names a
   convention.

10. Use structural knowledge only - how documents are typically organised, how a
    concept is usually labelled. Never a specific figure, date or named event
    you recall about this entity, even where it is true.

Schema:
{
  "companies": [],
  "doc_types": [],
  "years": [],
  "quarters": [],
  "reasoning": "<one sentence>",
  "selection_complete": true|false,
  "missing_evidence": [],
  "concept": "<canonical measure name only - no entity, date, or period>",
  "answer_kind": "stated_fact|derived_metric|judgment|attribution",
  "required_facts": [
    {
      "id": "<snake_case>",
      "description": "<what fact is needed>",
      "content_anchor": "<lowercase, space-separated, deduplicated-by-stem terms>",
      "period_role": "<omit for a single-period fact; otherwise a short label>"
    }
  ],
  "preferred_artifacts": ["table"|"text"|"diagram"|"chart"|"log"],
  "formula_preference": "<null, or the convention the question explicitly named>",
  "formulas": [
    {
      "id": "<snake_case_id>",
      "description": "<formula description>",
      "formula": "<bare expression only, no leading `name =`>",
      "rationale": "<why or when this method is defensible>"
    }
  ],
  "subject_words": []
}

QUESTION: {question}
MANIFEST: {manifest}
"""


def _match_concepts(subject_words: list, concepts_data: dict) -> list:
    """PLACEHOLDER for this experiment only - a plain substring match
    against user_term/aliases, not the FTS-search-against-the-concepts-
    collection design discussed separately. Keeps concept-naming
    questions from silently losing forced-phrase support while this
    branch is evaluated; not the intended final mechanism."""
    words = " ".join(subject_words or []).lower()
    if not words:
        return []
    matched = []
    for entry in concepts_data.get("entries", []):
        terms = [entry.get("user_term", "")] + (entry.get("aliases") or [])
        if any(t and t.lower() in words for t in terms):
            matched.append(entry)
    return matched


def resolve_and_plan(question: str, manifest: dict = None, concepts_data: dict = None,
                     scope: str = None, model: str = None) -> dict:
    """One completion, then one non-LLM N1QL fetch. Returns:
    {"intent": {...the LLM's own resolution fields...},
     "documents": [...full catalog rows matched...],
     "selected_documents": [{"doc_name", "reason"}, ...] (fts_resolver.py's
        shape, kept for the trace UI that already reads it),
     "selection_complete": bool, "missing_evidence": [...],
     "plan": {...concept/answer_kind/required_facts/preferred_artifacts/
        formula_preference, same shape retrieval.planner.plan_evidence()
        always returned so nothing downstream needs to change},
     "formulas": [...the model's own proposed formula candidates - NEW,
        replaces calculation.propose_candidates() for this experiment...],
     "concepts": {"subject_words": [...], "matched": [...full concept
        records, looked up by id...], "expanded_terms": [...],
        "target_sections": [...]}}.
    """
    manifest = manifest if manifest is not None else catalog_manifest.load(scope)
    concepts_data = (concepts_data if concepts_data is not None
                     else concepts_repo.load(scope=scope))
    user = (f"QUESTION: {question}\n\nMANIFEST:\n{manifest}")
    # Skills (docs/adr/0003-...) are domain-expert-owned filing-mechanics
    # facts, appended after the structural rules rather than folded into
    # them - the rules above are this call's compiler and should stay
    # general; anything that only ever holds for one document form or one
    # filing convention belongs here instead, reviewed and edited by someone
    # who need not read this file. Empty string when this scope has none.
    system = RESOLVE_PLAN_FORMULA_SYSTEM_PROMPT + skills_repo.render(scope=scope)
    result = llm.chat_json(system, user, model=model, stage="resolve_and_plan")

    intent = {k: result.get(k) for k in
             ("companies", "doc_types", "years", "quarters", "reasoning",
              "selection_complete", "missing_evidence")}
    # required_facts arrives with the SINGULAR content_anchor this iteration's
    # prompt asks for - "lowercase, space-separated, deduplicated by stem" -
    # a bag-of-words description, not a verbatim phrase. anchor_search.py
    # matches each element of content_anchors as its own independent CONTAINS
    # substring (a chunk matches if it contains ANY one), so this is split on
    # whitespace into that list here - each individual TERM becomes its own
    # anchor, rather than wrapped whole as one long phrase (that was the
    # earlier, measured-broken iteration: a chunk then had to contain the
    # entire multi-word string verbatim, which real filing text essentially
    # never does). This word-level split is looser than a curated list of
    # precise multi-word labels ("operating income (loss)" becomes three
    # separate single-word anchors here, not one phrase) - a real tradeoff
    # against precision this prompt doesn't otherwise address, worth
    # confirming empirically rather than assuming either way.
    required_facts = []
    for f in result.get("required_facts") or []:
        if isinstance(f, dict):
            f = dict(f)
            anchor = f.pop("content_anchor", None)
            f["content_anchors"] = list(dict.fromkeys((anchor or "").split()))
        required_facts.append(f)
    plan = {"concept": result.get("concept"), "answer_kind": result.get("answer_kind"),
           "required_facts": required_facts,
           "preferred_artifacts": result.get("preferred_artifacts"),
           "formula_preference": result.get("formula_preference")}
    formulas = result.get("formulas") or []

    matched = _match_concepts(result.get("subject_words") or [], concepts_data)
    expanded_terms, target_sections = [], []
    for c in matched:
        for term in (c.get("official_filing_terms") or []) + (c.get("aliases") or []):
            if term and term not in expanded_terms:
                expanded_terms.append(term)
        for section in c.get("target_sections") or []:
            if section and section not in target_sections:
                target_sections.append(section)

    documents = resolve_documents(intent.get("companies") or [],
                                  intent.get("doc_types") or [],
                                  intent.get("years") or [],
                                  intent.get("quarters") or [], scope=scope)
    reason = intent.get("reasoning") or ""
    return {
        "intent": intent,
        "documents": documents,
        "selected_documents": [{"doc_name": d["doc_name"], "reason": reason}
                               for d in documents],
        "selection_complete": intent.get("selection_complete", False),
        "missing_evidence": intent.get("missing_evidence") or [],
        "plan": plan,
        "formulas": formulas,
        "concepts": {
            "subject_words": result.get("subject_words") or [],
            "matched": matched,
            "expanded_terms": expanded_terms,
            "target_sections": target_sections,
        },
    }
