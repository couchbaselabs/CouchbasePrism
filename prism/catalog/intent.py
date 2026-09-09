"""Intent Clarifier: question -> which document(s), filtered ONLY on what the
catalog tracks (company, filing type, period) - never on subject matter,
since the catalog carries none. Replaces fts_resolver.py as the live
resolution path pipeline.answer_question() uses.

Two stages, same shape as fts_resolver.py's, but a fundamentally different
mechanism underneath:
  1. clarify_intent() - the question, plus the domain's manifest (one small
     aggregate document - catalog.manifest - never the catalog itself), goes
     to a cheap model that returns a STRUCTURED filter: which companies,
     which doc_types, which years, which quarters. A range in the question
     ("2020 to 2026") expands to every discrete year, not just its
     endpoints - the thing fts_resolver's free-text shortlist could never
     guarantee, because BM25 has no concept of a range, only of words that
     happen to co-occur.
  2. resolve_documents() - an exact catalog membership fetch (N1QL, no FTS,
     no ranking) using those structured values. There is nothing left to
     RANK once the filter is structured and drawn from the manifest's own
     known vocabulary - fts_resolver's FTS shortlist only ever existed to
     rank a free-text guess against catalog entries, and there is no free
     text left here to rank.

Genuinely different from fts_resolver in one respect worth being honest
about: this can return MORE than one document by design (a company-wide,
multi-year question is supposed to resolve to a document SET), but
pipeline.answer_question() downstream is still single-document - it reads
only the first. That is a real, known gap, not hidden by this module: the
multi-document retrieve+bind+compute+synthesize pipeline this exists to feed
has not been built yet. This module's job ends at "here is the correct
document set"; what happens with more than one document is the next piece.
"""
from .. import config, llm
from ..couchbase_io import query
from . import manifest as catalog_manifest
from .extraction import quarter_of
from .resolver import form_of

INTENT_SYSTEM_PROMPT = """\
You are identifying which document(s) in a catalog are relevant to a \
question, using ONLY the catalog's own metadata - never the document's \
actual content, which you do not have access to. The catalog tracks \
company, filing type, and period only. IGNORE any subject matter, topic, \
product, or event named in the question (e.g. "PFAS", "a product recall") \
entirely - the catalog cannot filter on it, and guessing a company or year \
from a topic you happen to recognize is exactly the failure mode this \
exists to avoid.

You are given a MANIFEST describing everything the catalog actually \
contains:
{"companies": [...], "doc_types": [...], "sectors": [...],
 "years_full": [...] (years with a complete annual filing),
 "years_partial": [...] (years with only partial-year filings so far -
 almost always the current, in-progress fiscal year),
 "quarters": {"<year>": [<quarter>, ...]} (which quarters that year has
 their OWN 10-Q for - a year missing from this map, or a quarter missing
 from its list, has no separate quarterly filing for that quarter; its
 figures live only in that year's 10-K)}

Return ONLY one valid JSON object:
{
  "companies": ["<exactly as printed in the manifest>", ...],
  "doc_types": ["<exactly as printed in the manifest>", ...],
  "years": [<integer>, ...],
  "quarters": [<integer 1-4>, ...],
  "reasoning": "<one sentence: why these values>",
  "selection_complete": <true if the manifest's own coverage can answer
                         this, false if a year/company/doc_type/quarter the
                         question needs is missing entirely>,
  "missing_evidence": ["<what's missing, if incomplete>"]
}

GUIDELINES:
- A year RANGE in the question ("between 2020 and 2026") expands to every
  discrete year in that range, not just the two endpoints.
- A 10-K's own comparative columns typically also cover the PRIOR fiscal
  year. When a question compares exactly one year to the year immediately
  before it (e.g. "2023 versus FY2022"), prefer years = [the later year]
  alone over both years - that one annual filing's own disclosure already
  covers both. Only include multiple years explicitly when the span is
  wider than one comparative column can cover (a multi-year range, or
  years that are not adjacent).
- A question naming a specific quarter ("the third quarter of 2022", "Q3
  2022", "as of September 30, 2022") MUST set quarters to that quarter's
  number, checked against the manifest's own "quarters" map for that year -
  a year can have several 10-Qs, and quarters is the ONLY way to say which
  one. Leaving it empty when a quarter is named is exactly the ambiguity
  this field exists to remove.
- Only use companies/doc_types values that are EXACTLY present in the
  manifest - never invent one, never normalize or guess a spelling.
- A year in years_partial is still real coverage - do not treat it as
  missing unless the question specifically needs a COMPLETE annual filing
  for that year and only years_full has one.
- Leave doc_types EMPTY if the question does not imply a specific filing
  type - an empty list means "any filing type", not "no filing type".
- Leave years EMPTY if the question is not period-specific at all - same
  reasoning as doc_types. Leave quarters EMPTY if the question is not
  quarter-specific (asks about a full year, or names no period at all).
- If no company in the manifest plausibly matches the question at all,
  return companies as an empty list, selection_complete false, and say so
  in missing_evidence - do not guess the closest company name.
"""


def clarify_intent(question: str, manifest: dict = None, scope: str = None,
                   model: str = None) -> dict:
    """manifest, if not given, is read fresh via catalog.manifest.load(scope) -
    exposed as a parameter mainly so a caller that already has it (or a test)
    is not forced into a second read."""
    manifest = manifest if manifest is not None else catalog_manifest.load(scope)
    user = f"QUESTION: {question}\n\nMANIFEST:\n{manifest}"
    return llm.chat_json(INTENT_SYSTEM_PROMPT, user, model=model, stage="catalog")


def resolve_documents(companies: list, doc_types: list, years: list,
                      quarters: list = None, scope: str = None) -> list:
    """Exact catalog membership, not a search. year is compared directly -
    the manifest's years come from the exact same doc_period the catalog
    itself stores, no representation gap. company and doc_type are each
    normalized before comparing, for the same underlying reason: catalog
    documents keep whatever the cover page actually printed ("3M COMPANY" on
    one filing, "3M Company" on another; "FORM 10-K", "SCHEDULE 14A", ...)
    while the manifest (and so the Clarifier's own output) reports one
    canonical form per company/doc_type - see manifest.build()'s and
    extraction.py's own reasoning for why extraction never normalizes at the
    source. Comparing canonical against raw directly would silently miss
    real documents on both dimensions. Company is matched case-insensitively
    in N1QL (UPPER() on both sides); doc_type still needs resolver.form_of()
    specifically, since it is not a casing difference but a wording one
    ("10-K" vs "FORM 10-K") that no case fold alone resolves.

    quarters filters via quarter_of(period_end_date_iso) in Python, same
    reasoning as doc_type - quarter is not a stored field, only derivable
    from the month of a 10-Q's own period end. Added after a live miss:
    without it, "third quarter of 2022" correctly narrowed to one year but
    matched all three of that year's 10-Qs, and nothing downstream could
    tell them apart. A document with no derivable quarter (a 10-K, a
    DEF 14A) is naturally excluded when quarters is given - it has none to
    match - which is correct: those forms cannot answer a quarter-specific
    question regardless of what doc_types says.

    Empty `companies` returns no documents rather than every document in the
    catalog - the Clarifier declining to name a company is a real "nothing
    matched" outcome, not "match everything".
    """
    if not companies:
        return []
    scope = scope or config.DEFAULT_SCOPE
    clauses = ["d.type = 'catalog_document'",
              "UPPER(d.company.`value`) IN $companies"]
    params = {"$companies": [c.upper() for c in companies]}
    if years:
        clauses.append("d.doc_period IN $years")
        params["$years"] = years
    rows = query(
        "SELECT d.doc_name, d.doc_type.`value` AS doc_type, d.doc_period, "
        "d.period_end_date_iso, d.company.`value` AS company, d.gics_sector, "
        "d.source_filename "
        f"FROM `{config.BUCKET}`.`{scope}`.`{config.CATALOG_COLLECTION}` AS d "
        f"WHERE {' AND '.join(clauses)}",
        params,
    )
    if doc_types:
        wanted = set(doc_types)
        rows = [r for r in rows if form_of(r.get("doc_type")) in wanted]
    if quarters:
        wanted_q = set(quarters)
        rows = [r for r in rows if quarter_of(r.get("period_end_date_iso")) in wanted_q]
    return rows


def resolve_for_question(question: str, scope: str = None, model: str = None) -> dict:
    """One call combining both stages. Returns:
    {"intent": {...clarify_intent's own output...},
     "documents": [...full catalog rows matched...],
     "selected_documents": [{"doc_name", "reason"}, ...] (fts_resolver.py's
        shape, kept for the trace UI that already reads it),
     "selection_complete": bool, "missing_evidence": [...]}
    """
    manifest = catalog_manifest.load(scope)
    intent = clarify_intent(question, manifest, scope=scope, model=model)
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
    }
