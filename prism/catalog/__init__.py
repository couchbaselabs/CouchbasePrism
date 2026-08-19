"""Tier 1 — the catalog: one document per source PDF (design/architecture.md §2).

Answers "which document is this?" so retrieval can be scoped before any
semantic search happens.

    extraction  PDF cover page -> catalog document
    repository  persistence in {bucket}.{scope}.catalog
    resolver    question -> which doc_name it is about
"""
from .extraction import (  # noqa: F401
    CLASSIFY_SYSTEM_PROMPT, COVER_PAGES, FIELDS,
    apply_grounding_check, build_document, build_from_pdf, classify_cover,
    cover_text, to_iso_date, year_of,
)
from .repository import ingested_doc_names, ensure_index, load_all, upsert  # noqa: F401
from .resolver import period_from_question, resolve, resolve_for_question  # noqa: F401
