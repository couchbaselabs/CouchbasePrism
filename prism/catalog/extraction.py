"""Deriving a catalog document from a PDF's cover page.

A fast cover-page pass, NOT a full layout parse: PyMuPDF reads pages 1-5 in
~100-600ms against 24-200s for a full conversion. The catalog never needs
layout analysis, so it should never pay for it.
"""
import datetime
import re
import time

import pymupdf

from .. import llm

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
    question - a 10-K wins by default. 12 of the FinanceBench resolution
    failures were exactly this. The document name carries the year, so it is used
    as a fallback and recorded as such.
    """
    match = _NAME_PERIOD.search(doc_name or "")
    return int(match.group(1)) if match else None


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


def build_document(doc_name: str, extraction: dict,
                   gics_sector: str = None) -> dict:
    """Only the fields something actually reads.

    Each extracted field keeps its {value, confidence, source_span} envelope
    because those drive the grounding check. `gics_sector` has no envelope: it is
    not extracted from the document at all, so a confidence score would be
    fiction. Its provenance is recorded in `lineage` instead.
    """
    period = extraction.get("period_end_date", {})
    raw = period.get("value")
    document = {
        "doc_id": doc_name,
        "doc_name": doc_name,
        "type": "catalog_document",
        "company": extraction.get("company"),
        "doc_type": extraction.get("doc_type"),
        "period_end_date": period,
        "period_end_date_iso": to_iso_date(raw),
        "doc_period": (fiscal_year(to_iso_date(raw))
                       or period_from_doc_name(doc_name)),
        "lineage": {
            "extractor": "pymupdf-sort+closed-set-classification",
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    if gics_sector:
        document["gics_sector"] = gics_sector
        document["lineage"]["gics_sector"] = "corpus_metadata"
    return document


def build_from_pdf(pdf_path: str, doc_name: str, model: str = None,
                   gics_sector: str = None) -> dict:
    return build_document(doc_name, classify_cover(cover_text(pdf_path), model=model),
                          gics_sector=gics_sector)
