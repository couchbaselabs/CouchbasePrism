"""Choosing WHICH document a question is about, before any semantic search.

Catalog-based document filtering was the single highest-leverage change in the
whole evaluation (25% -> 38%), ahead of hybrid search. Resolution is
deterministic regex: it matched LLM accuracy on structured signals during
catalog validation, so there is no reason to pay for a model call.
"""
import re

# Canonical SEC form -> the shapes a cover page actually prints it as.
_FORMS = {"10K": "10-K", "10Q": "10-Q", "8K": "8-K", "20F": "20-F", "40F": "40-F"}


def form_of(doc_type) -> str:
    """The canonical form from whatever was printed.

    Extraction deliberately does NOT normalise - the printed string is evidence,
    and "FORM 10-K" is what the cover page says. So comparison has to normalise
    instead. This was a live bug: the resolver compared doc_type == "10-K"
    exactly, and one extraction model returned "10-K" while another returned
    "FORM 10-K", which silently sent every document down the fallback path
    (arbitrary filing within the year) with no error anywhere.
    """
    squashed = re.sub(r"[^A-Z0-9]", "", str(doc_type or "").upper())
    for key, canonical in _FORMS.items():
        if key in squashed:
            return canonical
    return None


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
        quarterly = [d for d in matches if form_of(d.get("doc_type")) == "10-Q"]
        if quarterly:
            return quarterly[0]["doc_name"]
    annual = [d for d in matches if form_of(d.get("doc_type")) == "10-K"]
    return (annual[0] if annual else matches[0])["doc_name"]


def resolve_for_question(catalog_docs: list, question: str) -> str:
    return resolve(catalog_docs, *period_from_question(question))
