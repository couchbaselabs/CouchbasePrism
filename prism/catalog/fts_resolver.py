"""Document resolution at scale: FTS narrows a shortlist, an LLM picks from it.

resolver.py's approach - load every catalog entry into Python, regex-parse
the question for company/period/quarter/date signals - does not survive past
a few thousand documents, and this session found three separate real bugs in
it (a spelled-out quarter, a mixed bare/FY-prefixed year, a two-date
comparison), each one a mechanism built for one phrasing failing on another.
Both problems share one fix: search instead of scan, and let a model that
actually understands the question's phrasing make the final call instead of
enumerating every way a person might phrase a period.

Two stages, not one call:
  1. search_candidates() - one SEARCH() against the catalog's own FTS index
     (ftsPrismCatalog), narrowing millions of catalog entries to a handful
     without ever loading the full catalog into Python. Matches company,
     aliases, sector, AND a deterministically-built search_label (see
     extraction.py's _search_label) that spells out a filing's period in
     natural language every way it might be asked about - no regex
     extracting the question's own phrasing into structured year/quarter
     first, which is exactly what the three bugs above were.
  2. llm_resolve() - the actual question, plus minimal per-candidate info
     (never full document content - that's retrieval's job, not this one's),
     goes to a cheap utility-tier model that picks which of the shortlisted
     documents actually answer it. Can return zero, one, or several - a
     shortlist that doesn't contain the right document, or a question that
     genuinely needs more than one filing, are both real outcomes this
     reports rather than papers over by forcing a single confident pick.

This WAS the resolution path pipeline.answer_question() used, briefly - it
existed as an opt-in alongside resolver.py's deterministic path specifically
so the two could be compared before committing; once compared (verified live
against all three bugs above, plus a genuine out-of-corpus question that
correctly declined rather than guessing), the choice itself became the thing
worth removing: "too many knobs leads to confusion" - one resolution
mechanism, not a runtime setting nobody but an engineer would know how to
pick between.

Superseded in turn by catalog/intent.py (the Intent Clarifier), for a reason
specific to this corpus rather than a rejection of the approach here: this
module's shortlist matches company/aliases/sector/search_label against the
QUESTION'S OWN free text, which means a subject-matter word in the question
("PFAS", "a product recall") still perturbs the shortlist even though the
catalog has no subject-matter field to match it against - the shortlist just
silently downranks toward whatever else in the text happens to overlap.
intent.py's manifest-based filter never sees the question's free text at
all; it asks a model to select only from values the catalog actually has,
which cannot be pulled off course by a word the catalog was never going to
answer. Both modules' functions stay in the codebase (still tested, form_of()
is a real dependency of both this module's and intent.py's normalization) but
only intent.py is called by the live pipeline now.

Still generalizes past what regex ever could, same as when this replaced
resolver.py: search_candidates()/llm_resolve() scale past loading the whole
catalog into Python, which intent.py's manifest read does not need to prove
again - a manifest is bounded by DISTINCT values, not document count, from
the start.
"""
from .. import config, llm
from ..couchbase_io import query

CANDIDATE_FIELDS = ("doc_name", "source_filename", "company", "doc_type",
                    "doc_period", "period_end_date_iso")

RESOLVER_SYSTEM_PROMPT = """\
You are choosing which document(s), from a shortlist already narrowed by \
search, contain the evidence to answer a question. You do not have access \
to the documents themselves - only this shortlist's metadata.

Return ONLY one valid JSON object:

{
  "selected_documents": [
    {"doc_name": "<exactly as given>", "reason": "<why this one>"}
  ],
  "selection_complete": <true if the shortlist contains everything needed, \
false if it does not>,
  "missing_evidence": ["<what kind of document is missing, if incomplete>"]
}

GUIDELINES:
- Pick the document(s) whose company, filing type, and period genuinely
  match what the question asks about.
- A 10-K's own comparative columns typically also cover the prior fiscal
  year. Prefer ONE 10-K over two separate filings when a year-over-year
  comparison could already be satisfied by that one annual filing's own
  disclosure.
- Prefer the most specific matching filing (e.g. a 10-Q for a
  quarter-specific question) over a broader one (e.g. an annual 10-K) when
  both could technically contain the answer.
- List more than one document only when the question genuinely needs
  separate filings - comparing two different companies, or two periods far
  enough apart that no single filing's own comparative column covers both.
- If NONE of the candidates plausibly answer the question, return an empty
  selected_documents list, set selection_complete to false, and describe
  what kind of document is missing in missing_evidence.
- Do not invent a doc_name that is not in the candidate list.
- Return strictly valid JSON without markdown or explanatory text.
"""


def search_candidates(question: str, company_hint: str = None, top_k: int = 10) -> list:
    """The catalog FTS shortlist - company/aliases/sector/search_label
    matched against the raw question text, ordered by relevance. No
    catalog-wide scan: this is one SEARCH(), same shape as the docs-index
    queries elsewhere in this codebase.

    company_hint, if given (e.g. from a UI's company selector), boosts that
    field further - optional, since the search_label/company disjuncts
    already carry the question's own wording of the company."""
    disjuncts = [
        '{"match": $q, "field": "company.value", "boost": 2}',
        '{"match": $q, "field": "aliases", "boost": 2}',
        '{"match": $q, "field": "gics_sector"}',
        '{"match": $q, "field": "search_label"}',
    ]
    params = {"$q": question}
    if company_hint:
        disjuncts.append('{"match": $company_hint, "field": "company.value", "boost": 3}')
        params["$company_hint"] = company_hint
    rows = query(
        f"""
        SELECT d.doc_name, d.source_filename, d.company.`value` AS company,
               d.doc_type.`value` AS doc_type, d.doc_period, d.period_end_date_iso,
               SEARCH_SCORE() AS score
        FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.CATALOG_COLLECTION}` AS d
        WHERE SEARCH(d, {{"query": {{"disjuncts": [{", ".join(disjuncts)}]}}}},
                    {{"index": "{config.FTS_CATALOG_INDEX}"}})
        ORDER BY SEARCH_SCORE() DESC
        LIMIT {int(top_k)}
        """,
        params,
    )
    return rows


def llm_resolve(question: str, candidates: list, model: str = None) -> dict:
    """Candidates get only CANDIDATE_FIELDS - never full document content,
    which is retrieval's job, not resolution's. Returns
    {"selected_documents": [...], "selection_complete": bool,
    "missing_evidence": [...]}."""
    trimmed = [{k: c.get(k) for k in CANDIDATE_FIELDS} for c in candidates]
    user = f"QUESTION: {question}\n\nCANDIDATES:\n{trimmed}"
    return llm.chat_json(RESOLVER_SYSTEM_PROMPT, user, model=model, stage="catalog")


def resolve_for_question_at_scale(question: str, company_hint: str = None,
                                  top_k: int = 10, model: str = None) -> dict:
    """One call combining both stages. Returns the full llm_resolve() result
    (not just a doc_name) so a caller can see selection_complete/
    missing_evidence rather than only the first pick - PipelineOptions'
    integration of this reads result["selected_documents"][0]["doc_name"]
    for now (single-document downstream pipeline), but the shape already
    supports more without a schema change once that's built."""
    candidates = search_candidates(question, company_hint=company_hint, top_k=top_k)
    if not candidates:
        return {"selected_documents": [], "selection_complete": False,
                "missing_evidence": ["no catalog entries matched this question at all"]}
    return llm_resolve(question, candidates, model=model)
