"""Tier 1 — the catalog: one document per source document (design/architecture.md §2).

Answers "which document is this?" so retrieval can be scoped before any
semantic search happens.

    extraction  cover page (PDF or already-ingested chunks) -> catalog document
    repository  persistence in {bucket}.{scope}.catalog
    resolver    question -> which doc_name it is about (deterministic, regex -
                not used by the live pipeline; see fts_resolver below for why.
                Still here, still tested: form_of() is a real dependency of
                extraction.py's search_label building.)
    fts_resolver  question -> which doc_name it is about (FTS shortlist + LLM
                  pick). Replaced by manifest+intent (below) as the live
                  resolution path - still here, still tested, kept for the
                  same reason resolver.py is: a comparison point, not a
                  toggle.
    manifest    one aggregate document per domain - companies/doc_types/
                sectors/year-coverage rolled up from the catalog, computed
                after every rebuild. What the Intent Clarifier reads instead
                of scanning the catalog or searching ftsCatalog.
    intent      question + manifest -> which document(s), filtered ONLY on
                company/doc_type/period (the catalog carries no subject
                matter at all) - THE resolution path pipeline.answer_question()
                uses now. Exposed as catalog.resolve_intent (catalog.resolve_
                for_question is already resolver.py's deterministic path).
"""
from .extraction import (  # noqa: F401
    CLASSIFY_SYSTEM_PROMPT, COVER_PAGES, FIELDS,
    apply_grounding_check, build_document, build_from_chunks, build_from_pdf,
    classify_cover, cover_text, cover_text_from_chunks, doc_type_from_doc_name,
    fiscal_year, period_from_doc_name, quarter_of, to_iso_date, year_of,
)
from .aliases import ALIAS_SYSTEM_PROMPT, propose_aliases  # noqa: F401
from .repository import (  # noqa: F401
    all_periods, companies, delete_all, ensure_index, ingested_doc_names,
    load_all, set_aliases, set_period, upsert,
)
from .resolver import (  # noqa: F401
    event_date_from_question, form_of, period_from_question, resolve,
    resolve_for_question,
    subject_candidates,
)  # noqa: F401
from .fts_resolver import (  # noqa: F401
    llm_resolve, resolve_for_question_at_scale, search_candidates,
)
from .intent import (  # noqa: F401
    clarify_intent, resolve_documents,
    resolve_for_question as resolve_intent,
)
from .manifest import build as build_manifest  # noqa: F401
from .manifest import load as load_manifest  # noqa: F401
from .manifest import rebuild as rebuild_manifest  # noqa: F401


def rebuild_from_chunks(model: str = None, sectors: dict = None,
                        on_progress=None, scope: str = None) -> list:
    """Empty the catalog and rebuild one entry per document that has chunks
    ingested, reading each one's cover page from those chunks. This is the
    operation the UI's "Run catalog" button, manage.py, AND initialize.py's
    per-domain loop all call, so none of the three drifts from the others -
    the same reason the eval phases are configurations of one runtime instead
    of forked copies of it.

    `sectors` is {doc_name: gics_sector}, corpus-external metadata (not
    extracted from the document) - the caller decides where it comes from,
    same as build_from_chunks always has.

    `on_progress(i, total, doc_name, result)` is called after each document
    if given, so a caller can show progress without depending on catalog
    internals. `result` is {"doc_name", "ok", "document"} or
    {"doc_name", "ok": False, "error"}.

    Empty `names` (a domain with nothing ingested yet - iso20020, for now) is
    not an error: it returns [] immediately, same as "no documents matched"
    always has, so Initialize can loop over every configured domain without
    special-casing the ones that are architecture-only so far.
    """
    names = sorted(ingested_doc_names(scope=scope))
    if not names:
        return []
    delete_all(scope=scope)
    sectors = sectors or {}
    results = []
    for i, doc_name in enumerate(names, 1):
        try:
            document = build_from_chunks(doc_name, model=model,
                                         gics_sector=sectors.get(doc_name),
                                         scope=scope)
            upsert(document, scope=scope)
            result = {"doc_name": doc_name, "ok": True, "document": document}
        except Exception as e:
            result = {"doc_name": doc_name, "ok": False, "error": str(e)}
        results.append(result)
        if on_progress:
            on_progress(i, len(names), doc_name, result)
    return results
