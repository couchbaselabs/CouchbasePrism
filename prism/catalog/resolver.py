"""Choosing WHICH document a question is about, before any semantic search.

Catalog-based document filtering was the single highest-leverage change in the
whole evaluation (25% -> 38%), ahead of hybrid search. Resolution is
deterministic regex: it matched LLM accuracy on structured signals during
catalog validation, so there is no reason to pay for a model call.
"""
import re


def period_from_question(question: str):
    quarter = None
    m = re.search(r"\bQ([1-4])\b", question, re.IGNORECASE)
    if m:
        quarter = int(m.group(1))
    m = (re.search(r"\bFY\s*(\d{4})\b", question, re.IGNORECASE)
         or re.search(r"\b(20\d{2})\b", question))
    return (int(m.group(1)) if m else None), quarter


def resolve(catalog_docs: list, year, quarter) -> str:
    if not catalog_docs:
        raise ValueError("catalog is empty - build it before resolving")
    if year is None:
        # No period mentioned: the question means "currently", i.e. the most
        # recent filing on hand.
        return max(catalog_docs, key=lambda d: d["period_end_date_iso"] or "")["doc_name"]
    matches = [d for d in catalog_docs if d["doc_period"] == year]
    if not matches:
        return max(catalog_docs, key=lambda d: d["period_end_date_iso"] or "")["doc_name"]
    if quarter is not None:
        quarterly = [d for d in matches if d["doc_type"] == "10-Q"]
        if quarterly:
            return quarterly[0]["doc_name"]
    annual = [d for d in matches if d["doc_type"] == "10-K"]
    return (annual[0] if annual else matches[0])["doc_name"]


def resolve_for_question(catalog_docs: list, question: str) -> str:
    return resolve(catalog_docs, *period_from_question(question))
