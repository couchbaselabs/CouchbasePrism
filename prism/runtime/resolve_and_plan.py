"""Resolve + Plan: one LLM call doing what used to be two.

catalog.intent.clarify_intent() (which document(s)?) and
retrieval.planner.plan_evidence() (what evidence, once there?) were always
two separate PROMPTS answering two separate questions - but both read the
SAME raw question as their primary input, and neither's own reasoning
depended on the other's structured output. The only thing that crossed
between them was `source_context`, a one-line string summarising the
resolved document's type/period fed INTO the planner - and that dependency
disappears entirely once resolution and planning happen in the same
completion: the model already knows what it just resolved.

Two real, distinct instruction sets remain inside the ONE combined prompt
below - resolution (manifest-driven, corpus-specific) and evidence planning
(deliberately domain-free, per planner.py's own design principle: it never
names a document type, an industry, or a corpus). Combining the CALL does
not combine the CONCERNS - a future domain with no catalog/manifest concept
at all could still reuse the evidence-planning half verbatim; the standalone
catalog.intent and retrieval.planner modules stay exactly as they were,
independently callable and independently tested, for that reason.

resolve_and_plan() is the live path pipeline.answer_question() now calls,
folding in catalog.intent.resolve_documents()'s exact N1QL membership
fetch (not an LLM call) after the combined completion returns.
"""
from .. import llm
from ..catalog import manifest as catalog_manifest
from ..catalog.intent import resolve_documents

INTENT_AND_PLAN_SYSTEM_PROMPT = """\
You do two things from one question: (1) identify which catalog document(s) \
it needs, using ONLY catalog metadata; (2) plan what evidence to look for \
inside them, using ONLY general knowledge of how such documents are \
typically structured - you do not have the source text itself.

You are given a MANIFEST: {"companies": [...], "doc_types": [...], \
"sectors": [...], "years_full": [...] (years with a complete annual \
filing), "years_partial": [...] (partial-year coverage so far, usually the \
current year), "quarters": {"<year>": [<quarter>, ...]} (which quarters \
that year has their OWN quarterly filing)}.

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
  "preferred_artifacts": ["table" | "text" | "diagram" | "chart" | "log", ...]
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
  one - empty means "any", not "none".
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

Return strictly valid JSON - no markdown, no explanatory text outside the
object.
"""


def resolve_and_plan(question: str, manifest: dict = None, scope: str = None,
                     model: str = None) -> dict:
    """One completion, then one non-LLM N1QL fetch. Returns:
    {"intent": {...the LLM's own resolution fields...},
     "documents": [...full catalog rows matched...],
     "selected_documents": [{"doc_name", "reason"}, ...] (fts_resolver.py's
        shape, kept for the trace UI that already reads it),
     "selection_complete": bool, "missing_evidence": [...],
     "plan": {...same shape retrieval.planner.plan_evidence() always
        returned - concept/answer_kind/required_facts/preferred_artifacts -
        so nothing downstream of this needs to change}.

    `manifest`, if not given, is read fresh via catalog.manifest.load(scope) -
    exposed as a parameter mainly so a caller that already has it (or a
    test) is not forced into a second read.
    """
    manifest = manifest if manifest is not None else catalog_manifest.load(scope)
    user = f"QUESTION: {question}\n\nMANIFEST:\n{manifest}"
    result = llm.chat_json(INTENT_AND_PLAN_SYSTEM_PROMPT, user, model=model,
                           stage="resolve_and_plan")

    intent = {k: result.get(k) for k in
             ("companies", "doc_types", "years", "quarters", "reasoning",
              "selection_complete", "missing_evidence")}
    plan = {k: result.get(k) for k in
           ("concept", "answer_kind", "required_facts", "preferred_artifacts")}

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
    }
