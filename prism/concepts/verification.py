"""Corpus verification: does a `filing_terms` entry actually appear, verbatim,
anywhere in this scope's own documents?

This is concepts' own test - the third leg of docs/adr/0003-skills-a-domain-
expert-owned-knowledge-layer.md's validation table. The dictionary is
validated by human approval (ADR-0001); skills are validated by a domain
expert's judgment, since they're meant to hold beyond any one corpus; a
filing_term makes neither kind of claim - it asserts "this corpus's own
filings print this phrase," which is mechanically checkable, so it should
be checked, not trusted. Zero hits means the term is wrong or invented for
THIS corpus (a plausible-sounding elaboration, a term from a different
company's filings, a generic label that never actually appears) - the same
mechanism, and the same "whole phrase, not a bag of its words" discipline,
as the forced-phrase probe used elsewhere in this codebase (`#term#` in the
Workbench; anchor_search.py's per-anchor CONTAINS).
"""
from .. import config
from ..couchbase_io import QueryError, query


def verify_filing_term(term: str, scope: str = None) -> int:
    """How many chunks in this scope's own corpus contain `term` verbatim
    (case-insensitive, the whole contiguous phrase - never a bag of its
    words, which would pass almost anything). 0 on a missing index, same
    tolerance as concepts.load()/dictionary.load() - a fresh environment
    with no docs indexed yet is a valid state, not an error."""
    term = (term or "").strip()
    if not term:
        return 0
    scope = scope or config.DEFAULT_SCOPE
    try:
        rows = query(
            f"""
            SELECT COUNT(*) AS n
            FROM `{config.BUCKET}`.`{scope}`.`{config.DOCS_COLLECTION}` AS d
            WHERE CONTAINS(LOWER(d.`text-to-embed`), LOWER($term))
            """,
            {"$term": term},
        )
    except QueryError as e:
        if "No index available" in str(e):
            return 0
        raise
    return rows[0]["n"] if rows else 0
