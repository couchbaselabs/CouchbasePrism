"""Deriving a catalog document from a document's cover page.

A fast cover-page pass, NOT a full layout parse. Two sources, same shape
after: `cover_text` reads pages 1-5 directly from the PDF via PyMuPDF
(~100-600ms); `cover_text_from_chunks` reads the same pages from chunks the
Couchbase AI Data Plane workflow already produced, since ingestion happens
before catalog build and the cover page's text is already sitting in
Couchbase with page-number indexed. The catalog never needs layout analysis,
so it should never pay for one - and building from chunks means it never
needs local PDF file access either.
"""
import datetime
import re
import time

import pymupdf

from .. import config, llm
from ..couchbase_io import query
from .resolver import form_of

_QUARTER_OF_MONTH = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3,
                     10: 4, 11: 4, 12: 4}
_QUARTER_WORDS = {1: "first Q1", 2: "second Q2", 3: "third Q3", 4: "fourth Q4"}

COVER_PAGES = 5
FIELDS = ["company", "doc_type", "period_end_date"]

CLASSIFY_SYSTEM_PROMPT = (
    "You are extracting three fields from the cover page(s) of an SEC filing, given only "
    'the cover-page text (pages separated by "--- page N ---" markers). This is a bounded '
    "classification task, not open extraction: only report a field if it is literally "
    "present in the given text. Do not infer, guess, or use outside knowledge (e.g. do "
    "not recall a company's real ticker or any other fact from memory if it is not "
    "printed in the text you were given).\n\n"
    "Respond with exactly one JSON object:\n"
    '{\n'
    '  "company": {"value": string|null, "confidence": number, "source": string, "source_span": string|null, "page": integer|null},\n'
    '  "doc_type": {...same shape...},\n'
    '  "period_end_date": {...same shape...}\n'
    "}\n\n"
    "- company: the exact registrant name as printed near \"(Exact name of registrant as "
    "specified in its charter)\".\n"
    '- doc_type: the SEC form as printed, e.g. "10-K", "10-Q", "8-K". Do not normalize.\n'
    "- period_end_date: the full date from \"for the fiscal/quarterly year/period ended "
    "[DATE]\", with the complete year - never truncate it.\n\n"
    'source must be "pdf_text" (literally printed) or "absent". confidence is 0.0-1.0.'
)

_DATE_FORMATS = ["%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%d %B %Y", "%d %b %Y", "%Y-%m-%d"]


def cover_text(pdf_path: str, pages: int = COVER_PAGES) -> str:
    """sort=True is load-bearing. PyMuPDF's default order follows the PDF
    content stream, not visual position, which scrambles some cover pages badly
    enough to corrupt extraction (Activision's 2015 10-K). The PDFBox
    equivalent is setSortByPosition(true) if this ever moves to the JVM."""
    doc = pymupdf.open(pdf_path)
    n = min(pages, doc.page_count)
    return "\n".join(f"--- page {i + 1} ---\n{doc[i].get_text(sort=True)}"
                     for i in range(n))


_ELEMENT_INDEX = re.compile(r"/(\d+)$")
# docling's element-id ("#/texts/N", "#/tables/N", ...) is not one counter -
# each element TYPE has its own independent sequence, so a plain sort of the
# id string interleaves them arbitrarily. Verified live on a real cover page:
# Couchbase's own result order came back as texts/10, texts/3, texts/1,
# tables/0, texts/0 - the actual title chunk (texts/0) arrived LAST. This
# needs the same care cover_text()'s sort=True gives PyMuPDF, just applied to
# a different kind of scrambling: sort by (page, is-a-table, index within
# that element's own counter). Tables sort after text on the same page since
# a cover page's company/form/period is essentially never inside one, and
# ordering text correctly matters far more than exactly where a table lands.
def _chunk_sort_key(chunk: dict) -> tuple:
    match = _ELEMENT_INDEX.search(chunk.get("element_id") or "")
    index = int(match.group(1)) if match else 0
    return (chunk.get("page") or 0, chunk.get("type") == "table", index)


def cover_text_from_chunks(doc_name: str, pages: int = COVER_PAGES,
                           scope: str = None) -> str:
    """Same output shape as cover_text() - "--- page N ---" markers, one block
    per page - but sourced from chunks the AI Data Plane workflow already
    ingested rather than opening the PDF again. No PDF file access needed.

    Goes through the Search Vector Index (SEARCH()), not a plain N1QL WHERE -
    `docs` carries no secondary index on xmeta-data.filename or
    meta-data.page-number, only a primary index and the search index, so a
    plain predicate here was a full PrimaryScan3 over the whole collection
    PER DOCUMENT catalogued (confirmed live via EXPLAIN - this was the actual
    cause of catalog rebuild running at 1-2 documents/minute). The same fix
    that made the document workbench fast applies here: both fields ARE
    mapped in the search index, so the fetch becomes an IndexFtsSearch
    instead - verified identical result sets against the old query (same 15
    rows, same ids) at a fraction of the time, no new index required.

    meta-data.type is NOT mapped in the search index at all, so the footnote
    exclusion happens in Python after the fetch rather than as a query
    predicate - the result set here is small enough (a few pages) that this
    costs nothing.
    """
    scope = scope or config.DEFAULT_SCOPE
    rows = query(
        f"""
        SELECT d.`meta-data`.`page-number` AS page,
               d.`meta-data`.type AS type,
               d.`element-id` AS element_id,
               d.`text-to-embed` AS text
        FROM `{config.BUCKET}`.`{scope}`.`{config.DOCS_COLLECTION}` AS d
        WHERE SEARCH(d, {{"query": {{"conjuncts": [
          {{"field": "xmeta-data.filename", "match": $filename}},
          {{"field": "meta-data.page-number", "max": $pages, "inclusive_max": true}}
        ]}}}}, {{"index": "{config.fts_docs_index(scope)}"}})
        """,
        {"$filename": config.source_filename(doc_name, scope=scope), "$pages": pages},
    )
    rows = [r for r in rows if r.get("type") != "footnote"]
    rows.sort(key=_chunk_sort_key)
    by_page = {}
    for row in rows:
        by_page.setdefault(row["page"], []).append(row.get("text") or "")
    return "\n".join(f"--- page {page} ---\n" + "\n".join(texts)
                     for page, texts in sorted(by_page.items()))


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def apply_grounding_check(extraction: dict, text: str) -> dict:
    """Verifies the claimed VALUE against the source, not the model's cited
    span. A model can cite sloppily while being right about the value (an 8-K
    period read correctly from "Date of Report" but cited against boilerplate),
    and can cite plausibly while being wrong (a ticker recalled from training
    data for a filing that prints none). Only the second is a hallucination;
    only value-checking tells them apart."""
    haystack = _normalize_ws(text)
    for name in FIELDS:
        field = extraction.get(name)
        if not field or field.get("source") != "pdf_text":
            continue
        value = _normalize_ws(str(field.get("value") or ""))
        if not value or value not in haystack:
            field.update(confidence=0.0, source="ungrounded", grounding_failed=True)
    return extraction


def classify_cover(text: str, model: str = None) -> dict:
    return apply_grounding_check(
        llm.chat_json(CLASSIFY_SYSTEM_PROMPT, text[:10000], model=model,
                      stage="catalog"), text)


def year_of(date_str: str):
    m = re.search(r"\b(19|20)\d{2}\b", date_str or "")
    return int(m.group()) if m else None


def to_iso_date(date_str: str):
    """Derived alongside - never replacing - the verbatim printed date, so the
    citable value survives and FTS still gets something range-queryable. A
    partial date ("March 2021") stays unparsed rather than inventing a day."""
    if not date_str:
        return None
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", date_str.strip(), flags=re.IGNORECASE)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


_NAME_PERIOD = re.compile(r"_((?:19|20)\d{2})")


def period_from_doc_name(doc_name: str):
    """A period for documents whose cover page states none.

    8-Ks and earnings releases carry no "for the fiscal year ended" line, so
    extraction leaves doc_period empty and they never match a period-bearing
    question - a 10-K wins by default. 12 of a multi-company eval corpus's
    resolution failures were exactly this. The document name carries the
    year, so it is used as a fallback and recorded as such.
    """
    match = _NAME_PERIOD.search(doc_name or "")
    return int(match.group(1)) if match else None


_NAME_FORM = re.compile(r"_(10K|10Q|DEF14A|8K)(?:_|$)")
_NAME_FORM_CANONICAL = {"10K": "10-K", "10Q": "10-Q", "DEF14A": "DEF 14A", "8K": "8-K"}


def doc_type_from_doc_name(doc_name: str):
    """A doc_type for documents whose cover page never states one within the
    classifier's page window - some of this corpus's "...earnings" 8-Ks never
    print "Form 8-K" (or any recognizable form phrase) in their first 5
    pages at all, so classify_cover comes back null rather than guessing.
    The document name already carries the form - this corpus's own naming
    convention - so it is used as a fallback and recorded as such, same
    reasoning as period_from_doc_name above."""
    match = _NAME_FORM.search(doc_name or "")
    return _NAME_FORM_CANONICAL.get(match.group(1)) if match else None


def fiscal_year(iso_date: str):
    """The fiscal year a period-end date belongs to.

    Companies on a 52/53-week calendar end the year on the weekday nearest 31
    December, which can fall in the first days of January: Johnson & Johnson's
    fiscal 2022 ended 1 January 2023. Taking the calendar year of that date
    labels it 2023 and every FY2022 question misses.

    Only the first week of January is adjusted. Retailers ending late January
    label the year by the calendar year it ends in - Best Buy's fiscal 2023
    ended 28 January 2023 - so a broader rule would break them, and a
    fiscal year ending in June is labelled by its ending year too.
    """
    if not iso_date or len(iso_date) < 10:
        return None
    year, month, day = int(iso_date[:4]), int(iso_date[5:7]), int(iso_date[8:10])
    return year - 1 if month == 1 and day <= 7 else year


def quarter_of(period_end_date_iso: str):
    """Which fiscal quarter a period-end date falls in, from its month alone -
    arithmetic, not a guess, same reasoning as fiscal_year() above. Only
    meaningful for a 10-Q's own period_end_date_iso; a 10-K's covers the
    whole year, not one quarter, so callers that care about the distinction
    should check doc_type/form_of() first (see _search_label and
    catalog.intent.resolve_documents, both of which do)."""
    if not period_end_date_iso or len(period_end_date_iso) < 7:
        return None
    return _QUARTER_OF_MONTH.get(int(period_end_date_iso[5:7]))


def _search_label(company: str, doc_type: str, doc_period, period_end_date_iso: str) -> str:
    """A rich, natural-language description of a catalog entry - built
    entirely from fields already known deterministically (no new
    classification, no LLM call), so a search index over it can match
    however a question happens to phrase a period ("third quarter of 2022",
    "Q3 2022", "September 2022") without any regex extracting that phrasing
    into structured year/quarter first. That regex extraction is exactly
    what produced three separate resolver bugs this session - each one a
    mechanism built for one phrasing failing on a different one. A BM25
    match against natural text sidesteps the whole class of gap: it doesn't
    care which words the question used, only whether they overlap with
    words already in this label."""
    form = form_of(doc_type) or (doc_type or "")
    bits = [company or "", form]
    if form == "10-Q":
        quarter = quarter_of(period_end_date_iso)
        if quarter:
            bits.append(f"{_QUARTER_WORDS[quarter]} quarter quarterly report")
    elif form == "10-K":
        bits.append("annual report full year fiscal year")
    elif form == "DEF 14A":
        bits.append("proxy statement annual meeting")
    elif form == "8-K":
        bits.append("current report")
    if doc_period:
        bits.append(f"fiscal {doc_period} FY{doc_period} {doc_period}")
    if period_end_date_iso:
        bits.append(f"period ended {period_end_date_iso}")
    return " ".join(b for b in bits if b)


def build_document(doc_name: str, extraction: dict, gics_sector: str = None,
                   extractor: str = "pymupdf-sort+closed-set-classification",
                   source_filename: str = None) -> dict:
    """Only the fields something actually reads.

    Each extracted field keeps its {value, confidence, source_span} envelope
    because those drive the grounding check. `gics_sector` has no envelope: it is
    not extracted from the document at all, so a confidence score would be
    fiction. Its provenance is recorded in `lineage` instead.

    `source_filename` is the actual `xmeta-data.filename` these chunks are
    stored under - resolved ONCE here, at build time, via
    config.source_filename(doc_name) (AWS_BUCKET/AWS_FOLDER-derived). Every
    retrieval-time call site used to recompute that mapping fresh on every
    query instead of reading it from here - which meant AWS_BUCKET/AWS_FOLDER
    had to stay correct FOREVER, not just once, for retrieval to keep working.
    Verified live: an AWS_FOLDER drift after this field was already correct in
    the catalog broke nothing, because retrieval now reads this instead of
    recomputing. Absent (None) for a document built via build_from_pdf, which
    has no S3-derived storage location to record.
    """
    period = extraction.get("period_end_date", {})
    raw = period.get("value")
    doc_type = extraction.get("doc_type") or {}
    if not doc_type.get("value"):
        # The classifier declined rather than guess - correct behaviour, but
        # this corpus's doc_name already carries the form when the cover page
        # doesn't state one clearly enough. source="doc_name" (not
        # "pdf_text") keeps this honestly distinct from an actual cover-page
        # read - the manifest and search_label both read doc_type.value the
        # same way either way, but a reader of this envelope can still tell
        # which one they got.
        fallback = doc_type_from_doc_name(doc_name)
        if fallback:
            doc_type = {**doc_type, "value": fallback, "source": "doc_name"}
    document = {
        "doc_id": doc_name,
        "doc_name": doc_name,
        "type": "catalog_document",
        "company": extraction.get("company"),
        "doc_type": doc_type,
        "period_end_date": period,
        "period_end_date_iso": to_iso_date(raw),
        "doc_period": (fiscal_year(to_iso_date(raw))
                       or period_from_doc_name(doc_name)),
        "lineage": {
            "extractor": extractor,
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    if gics_sector:
        document["gics_sector"] = gics_sector
        document["lineage"]["gics_sector"] = "corpus_metadata"
    if source_filename:
        document["source_filename"] = source_filename
    document["search_label"] = _search_label(
        (document["company"] or {}).get("value"),
        (document["doc_type"] or {}).get("value"),
        document["doc_period"], document["period_end_date_iso"])
    return document


def build_from_pdf(pdf_path: str, doc_name: str, model: str = None,
                   gics_sector: str = None) -> dict:
    return build_document(doc_name, classify_cover(cover_text(pdf_path), model=model),
                          gics_sector=gics_sector)


def build_from_chunks(doc_name: str, model: str = None, gics_sector: str = None,
                      pages: int = COVER_PAGES, scope: str = None) -> dict:
    """Same classification, sourced from already-ingested chunks instead of
    the PDF. This is the path that needs no local PDF file at all."""
    return build_document(
        doc_name,
        classify_cover(cover_text_from_chunks(doc_name, pages, scope=scope), model=model),
        gics_sector=gics_sector,
        extractor="chunks-sort+closed-set-classification",
        source_filename=config.source_filename(doc_name, scope=scope))
