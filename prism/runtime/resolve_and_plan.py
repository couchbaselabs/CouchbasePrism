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

Two things preserved here that the experimental prompt draft omitted -
noted explicitly since dropping them silently would regress fixes
already made, not just simplify the prompt:
  1. `period_role` per required fact - without it, a multi-period formula
     (e.g. an N-year average) has nothing to tell
     fact_binding.validate_bindings()'s mixed-period guard "this is
     deliberately several periods of the same quantity," and the guard
     rejects every fact in the set.
  2. A GENERAL form of "use structural knowledge only, never a specific
     recalled fact" for the evidence plan - an anti-hallucination
     discipline planner.py's own domain-free design already relied on,
     not an anecdote.
  `rationale` per formula and `formula_preference` are also kept (both
  existed in the previous design, neither appeared in the experimental
  draft - dropping either silently would have been a regression, not a
  simplification).

NOTE, latest revision: the rule guarding against guessing `doc_types`
from a metric's typical filing type (e.g. "non-GAAP measures usually
live in a 10-Q") - previously preserved above as item 2 - was
DELIBERATELY DROPPED in the prompt's second iteration, per explicit
instruction, not an oversight. That guard is exactly what
docs/findings/earnings-8k-not-catalogued-by-period.md's "why it
surfaced now" section credits with fixing 3m-003's doc-type guess -
dropping it is a real, deliberate re-exposure of that failure mode
under this experiment, to be watched for in the next eval run, not
silently re-added.

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
from ..catalog import manifest as catalog_manifest
from ..catalog.intent import resolve_documents

RESOLVE_PLAN_FORMULA_SYSTEM_PROMPT = """\
Extract RAG routing metadata into JSON based on the QUESTION and MANIFEST.

Rules:
1. Output ONLY valid JSON - no preamble, postscript, or code fences.
2. `companies`, `doc_types`, `years`, and `quarters` must strictly match
   values from MANIFEST (`[]` = any/all).
3. `subject_words` must be an array of key words and phrases extracted
   VERBATIM from the QUESTION string (do not derive, rephrase, or add
   external words).
4. `content_anchors` per required fact must be an array of SHORT,
   independent, verbatim-likely labels (e.g. "net sales", "operating
   income (loss)", "special charges") - each one is matched separately
   as its own substring, so a fact is found if the source contains ANY
   one of them, not all of them at once. Never combine several labels
   into one long phrase or sentence - that demands the source contain
   the whole combination verbatim, which real filing text never does.
   Derive them from formula variables (if `derived_metric`) and the
   verbatim `subject_words` relevant to that fact. Do NOT add generic
   conversational fluff or filing boilerplate (e.g., "full year",
   "consolidated statements", "results of operations").
5. `formulas` must be an array of
   `{"id": "...", "description": "...", "formula": "...", "rationale": "..."}`
   objects IF `answer_kind` is `derived_metric`; otherwise `[]`. Return
   MORE THAN ONE only when the concept has materially distinct,
   genuinely defensible conventions - two formulas that differ only in
   how the same inclusion is phrased are one method, not two. Never rank
   or choose among them, and never claim the set is exhaustive. Every
   `formula` string must reference variables by the EXACT `id` values used
   in `required_facts` - never plain-English restatements of the concept
   (e.g. write `operating_income_loss_2023 / net_sales_2023`, not
   "operating income (loss) / net sales") - downstream binding matches
   formulas to facts by these identifiers verbatim.
6. Never compute actual values or answer the question.
7. Evidence planning (required_facts, content_anchors, preferred_artifacts)
   uses STRUCTURAL knowledge only - typical section names, how a concept
   is usually labeled, what artifact usually carries it - never a
   specific fact, figure, or named event you recall about this entity
   from training, even where it happens to be true.
8. `period_role` on a required fact: omit for a single-period fact;
   otherwise a short label distinguishing which period this is (e.g.,
   "prior" vs "current", or "fy2022_fy2023") whenever a formula or
   query references quantities across periods, so downstream binding
   knows multi-period facts are deliberate.

Schema:
{
  "companies": [],
  "doc_types": [],
  "years": [],
  "quarters": [],
  "reasoning": "<one sentence>",
  "selection_complete": true|false,
  "missing_evidence": [],
  "concept": "<target metric requested>",
  "answer_kind": "stated_fact|derived_metric|judgment|attribution",
  "required_facts": [
    {
      "id": "<snake_case>",
      "description": "<what fact is needed>",
      "content_anchors": ["<short verbatim-likely label>"],
      "period_role": "<omit for a single-period fact; otherwise a short label>"
    }
  ],
  "preferred_artifacts": ["table"|"text"|"diagram"|"chart"|"log"],
  "formula_preference": "<null, or the convention the question explicitly named>",
  "formulas": [
    {
      "id": "<snake_case_id>",
      "description": "<formula description>",
      "formula": "<formula string>",
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
    result = llm.chat_json(RESOLVE_PLAN_FORMULA_SYSTEM_PROMPT, user, model=model,
                           stage="resolve_and_plan")

    intent = {k: result.get(k) for k in
             ("companies", "doc_types", "years", "quarters", "reasoning",
              "selection_complete", "missing_evidence")}
    # required_facts arrives with content_anchors already as a LIST of short,
    # independent labels - the same shape retrieval.planner.plan_evidence()
    # has always produced, and anchor_search.py's own CONTAINS-per-anchor
    # matching was built around (a chunk matches if it contains ANY one
    # anchor, not all of them). An earlier iteration of this experiment
    # collapsed this into a single combined content_anchor STRING (wrapped
    # here as a one-element list) to shrink the schema - measured, live,
    # to silently break retrieval: a chunk then had to contain the whole
    # multi-word combination verbatim, which real filing text essentially
    # never does. No bridging is needed now; the field is used as-is.
    required_facts = result.get("required_facts") or []
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
