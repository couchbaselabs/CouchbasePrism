"""Resolve + Plan: one LLM call doing what used to be two, now three.

catalog.intent.clarify_intent() (which document(s)?) and
retrieval.planner.plan_evidence() (what evidence, once there?) were always
two separate PROMPTS answering two separate questions - but both read the
SAME raw question as their primary input, and neither's own reasoning
depended on the other's structured output. The only thing that crossed
between them was `source_context`, a one-line string summarising the
resolved document's type/period fed INTO the planner - and that dependency
disappears entirely once resolution and planning happen in the same
completion: the model already knows what it just resolved.

Concept matching (subject-matter recognition against the concepts
collection) joins the same call for the same reason: the model is already
reading the question, and the concepts collection is small and bounded -
the same size class as the manifest, not something that scales with corpus
size, so including it wholesale costs little. It stays a DISTINCT part of
the prompt and a DISTINCT output field, same discipline as resolution vs.
planning below - the model reports WHICH known concepts it recognized
(exact ids from the supplied list), and the CODE deterministically looks up
each one's own official_filing_terms/target_sections rather than trusting
the model to reproduce them verbatim. A model asked to retype a curated
phrase is a model that might retype it slightly wrong.

Three real, distinct instruction sets remain inside the ONE combined prompt
below - resolution (manifest-driven, corpus-specific), evidence planning
(deliberately domain-free, per planner.py's own design principle: it never
names a document type, an industry, or a corpus), and concept matching
(concepts-collection-driven). Combining the CALL does not combine the
CONCERNS - a future domain with no catalog/manifest/concepts concept at all
could still reuse the evidence-planning third verbatim; the standalone
catalog.intent and retrieval.planner modules stay exactly as they were,
independently callable and independently tested, for that reason.

resolve_and_plan() is the live path pipeline.answer_question() now calls,
folding in catalog.intent.resolve_documents()'s exact N1QL membership
fetch (not an LLM call) after the combined completion returns.
"""
from .. import concepts as concepts_repo
from .. import llm
from ..catalog import manifest as catalog_manifest
from ..catalog.intent import resolve_documents

INTENT_AND_PLAN_SYSTEM_PROMPT = """\
You do three things from one question: (1) identify which catalog \
document(s) it needs, using ONLY catalog metadata; (2) plan what evidence \
to look for inside them, using ONLY general knowledge of how such \
documents are typically structured - you do not have the source text \
itself; (3) recognize whether the question names any KNOWN subject-matter \
concept from a small reference list.

You are given a MANIFEST: {"companies": [...], "doc_types": [...], \
"sectors": [...], "years_full": [...] (years with a complete annual \
filing), "years_partial": [...] (partial-year coverage so far, usually the \
current year), "quarters": {"<year>": [<quarter>, ...]} (which quarters \
that year has their OWN quarterly filing)}.

You are also given CONCEPTS: [{"id", "user_term", "aliases"}, ...] - a \
short list of subject matters this corpus is known to discuss, each named \
the way a person might refer to it informally.

Return ONLY one JSON object:
{
  "companies": ["<exactly as in manifest>", ...],
  "doc_types": ["<exactly as in manifest>", ...],
  "years": [<int>, ...],
  "quarters": [<int 1-4>, ...],
  "reasoning": "<one sentence: why these companies/doc_types/years/quarters>",
  "selection_complete": <true if the manifest's own coverage can answer this>,
  "missing_evidence": ["<what's missing, if incomplete>"],
  "concept": "<canonical name of the requested measure - lower-case, no entity/date/period>",
  "answer_kind": "stated_fact | derived_metric | judgment | attribution",
  "required_facts": [
    {"id": "<snake_case, the quantity only>", "description": "<what it represents>",
     "content_anchors": ["<label expected verbatim in the source>"]}
  ],
  "preferred_artifacts": ["table" | "text" | "diagram" | "chart" | "log", ...],
  "formula_preference": "<null, or the convention the question explicitly named>",
  "subject_words": ["<the question's own wording for its subject matter, if any>"],
  "matched_concepts": ["<id, exactly as in CONCEPTS - only ones genuinely named>"]
}

RESOLUTION (part 1):
- IGNORE subject matter, topic, product, or event words entirely - the
  catalog cannot filter on them. Do not let a name you recognize pull the
  company or year off course.
- A year range expands to every discrete year in it, not just the endpoints.
- One year vs. the year right before it: prefer years = [the later year]
  alone - a 10-K's own comparative column covers the prior year too. Use
  multiple years only for a wider or non-adjacent span.
- A named quarter MUST set quarters, checked against the manifest's own
  map - a year can have several quarterly filings, and quarters is the only
  way to say which one.
- Use EXACT manifest values only - never invent or normalize one.
- years_partial still counts as coverage, unless the question specifically
  needs a COMPLETE annual filing.
- Leave doc_types/years/quarters EMPTY when the question doesn't imply
  one - empty means "any", not "none". A guess about which FORM TYPICALLY
  CARRIES a kind of content is not the question implying one - it is you
  inventing a constraint the question never stated, and the two documents
  covering the identical quarter are not interchangeable, so guessing
  wrong silently answers from the wrong one. Observed live: asked for
  "Adjusted Free Cash Flow Conversion" (no document type named), this
  narrowed doc_types to ["10-Q"] with the reasoning "non-GAAP quarterly
  measures... are typically presented" there - a plausible-sounding guess
  that was simply wrong; that metric's own formula is in the earnings
  release 8-K's supplemental non-GAAP table, not the 10-Q. A metric's
  NAME or NATURE (non-GAAP, "Adjusted X", a ratio, a segment breakdown...)
  is never grounds to narrow doc_types by itself - only the question
  itself naming a form ("in the 10-Q", "per the earnings release") does.
- If no company plausibly matches, return companies empty, selection_complete
  false, and say what's missing - never guess the closest name.
- If no company plausibly matches, return companies empty, selection_complete
  false, and say what's missing - never guess the closest name.

EVIDENCE PLAN (part 2) - use STRUCTURAL knowledge only (typical section
names, how a concept is usually labeled, what artifact usually carries it),
NEVER a specific fact, figure, or named event you recall about this entity
from training, even where it happens to be true. Propose a generic label
instead unless the question itself names it - this matters MORE here than
it would in isolation, because part 1 just told you which company this is.
- concept: the single measure whose value determines the answer - canonical,
  phrasing-independent, excludes subject, entity, date, period.
- answer_kind: stated_fact (source states it directly) | derived_metric
  (must combine/transform source values - use this even when an explanation
  is ALSO requested alongside a calculation) | judgment (assess/classify an
  already-known value) | attribution (explain a cause, the source states the
  explanation itself, AND no calculation is also requested).
- required_facts: only what's needed. EXCEPTION: when a metric has multiple
  materially distinct established conventions, include the union of facts
  every convention needs, not just the ones for one you'd pick - do not
  rank or choose among them here.
- Attribution/explanatory facts: name the driver GENERICALLY ("special
  charges", "litigation-related charges") unless the question itself names
  it - a specific recalled name is a guess, not a structural label.
- A fact is ONE value the source prints on one line. If naming it requires
  combining, netting, or excluding other values, list the printed
  components instead, not the aggregate.
- Fact IDs: snake_case, the quantity only - no entity, date, period, or
  version.
- content_anchors: labels reasonably expected verbatim in the source,
  preserving printed qualifiers ("net", "current", "diluted", "continuing
  operations") - never abstract, derived, or paraphrased. An empty list is
  better than an invented label.
- Never propose a formula, compute a value, or make the final judgment -
  that happens downstream.
- formula_preference: null UNLESS the question itself explicitly names a
  specific convention to use ("use the alternate formula", "using the
  preferred method") - copy the wording it used (e.g. "alternate",
  "preferred"). Do not set it just because the concept happens to have more
  than one convention; only the question asking for a specific one does.

CONCEPT MATCHING (part 3):
- subject_words: the question's own informal wording for its subject
  matter (e.g. "forever chemicals", "earplug lawsuit") - empty if the
  question names no subject matter at all (most questions won't).
- matched_concepts: match subject_words against CONCEPTS' own user_term/
  aliases - use EXACT ids from the supplied list, never invent one. Empty
  if nothing in CONCEPTS plausibly matches, which is a normal outcome, not
  a failure.
- Do NOT match from background knowledge about what a concept usually
  involves - only from the question's own wording against user_term/
  aliases as given. A concept absent from the supplied list cannot match,
  no matter how well you recognize the subject from training.

Return strictly valid JSON - no markdown, no explanatory text outside the
object.
"""


def resolve_and_plan(question: str, manifest: dict = None, concepts_data: dict = None,
                     scope: str = None, model: str = None) -> dict:
    """One completion, then one non-LLM N1QL fetch. Returns:
    {"intent": {...the LLM's own resolution fields...},
     "documents": [...full catalog rows matched...],
     "selected_documents": [{"doc_name", "reason"}, ...] (fts_resolver.py's
        shape, kept for the trace UI that already reads it),
     "selection_complete": bool, "missing_evidence": [...],
     "plan": {...same shape retrieval.planner.plan_evidence() always
        returned - concept/answer_kind/required_facts/preferred_artifacts -
        so nothing downstream of this needs to change},
     "concepts": {"subject_words": [...], "matched": [...full concept
        records, looked up by id - never trusted from the model's own
        text...], "expanded_terms": [...], "target_sections": [...]}}.

    `manifest`/`concepts_data`, if not given, are read fresh via
    catalog.manifest.load(scope)/concepts.load(scope) - exposed as
    parameters mainly so a caller that already has them (or a test) is not
    forced into a second read.
    """
    manifest = manifest if manifest is not None else catalog_manifest.load(scope)
    concepts_data = (concepts_data if concepts_data is not None
                     else concepts_repo.load(scope=scope))
    # Only what matching needs, not official_filing_terms/target_sections -
    # those are looked up deterministically below once the model says which
    # ids matched, never generated by the model itself.
    concepts_for_prompt = [{"id": c.get("id"), "user_term": c.get("user_term"),
                            "aliases": c.get("aliases") or []}
                           for c in concepts_data.get("entries", [])]
    user = (f"QUESTION: {question}\n\nMANIFEST:\n{manifest}\n\n"
           f"CONCEPTS:\n{concepts_for_prompt}")
    result = llm.chat_json(INTENT_AND_PLAN_SYSTEM_PROMPT, user, model=model,
                           stage="resolve_and_plan")

    intent = {k: result.get(k) for k in
             ("companies", "doc_types", "years", "quarters", "reasoning",
              "selection_complete", "missing_evidence")}
    plan = {k: result.get(k) for k in
           ("concept", "answer_kind", "required_facts", "preferred_artifacts",
            "formula_preference")}

    by_id = {c.get("id"): c for c in concepts_data.get("entries", [])}
    matched = [by_id[i] for i in (result.get("matched_concepts") or []) if i in by_id]
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
        "concepts": {
            "subject_words": result.get("subject_words") or [],
            "matched": matched,
            "expanded_terms": expanded_terms,
            "target_sections": target_sections,
        },
    }
