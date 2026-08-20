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
    cover_text, fiscal_year, period_from_doc_name, to_iso_date, year_of,
)
from .aliases import ALIAS_SYSTEM_PROMPT, propose_aliases  # noqa: F401
from .repository import (  # noqa: F401
    all_periods, companies, ensure_index, ingested_doc_names, load_all,
    set_aliases, set_period, upsert,
)
from .resolver import (  # noqa: F401
    event_date_from_question, form_of, period_from_question, resolve,
    resolve_for_question,
    subject_candidates,
)  # noqa: F401
