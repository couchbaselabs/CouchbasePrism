"""Choosing WHICH document a question is about, before any semantic search.

Catalog-based document filtering was the single highest-leverage change in the
whole evaluation (25% -> 38%), ahead of hybrid search. Resolution is
deterministic: it matched LLM accuracy on structured signals during catalog
validation, so there is no reason to pay for a model call.

Resolution has to identify the SUBJECT before the period. That was invisible
while the corpus held one company: matching on year alone was correct 8/8. On a
354-document, 40-company corpus the same code was correct 12/143 (8%), happily
answering an Adobe question from 3M's 2015 10-K. Both signals are needed, and
the subject is the one that matters first.

Two subject signals, because neither covers the corpus alone:
  - the company extracted from the cover page ("THE COCA-COLA COMPANY"), which
    is real document evidence but is written formally
  - the document name's own prefix, which survives abbreviations the cover page
    never uses - a question says "PG&E" where the filing says "PACIFIC GAS AND
    ELECTRIC COMPANY"
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


# Corporate suffixes carry no identifying information and differ between how a
# filing writes its name and how a question refers to it.
_NOISE = ("COMPANY", "COMPANIES", "INCORPORATED", "CORPORATION", "HOLDINGS",
          "GROUP", "LIMITED", "PLC", "INC", "CORP", "CO", "LTD", "NV", "SA",
          "THE", "AND")


def _core(name: str) -> str:
    """A comparable identity string: letters and digits only, corporate noise
    removed. "THE COCA-COLA COMPANY" and "Coca-Cola" both reduce to COCACOLA."""
    words = re.split(r"[^A-Za-z0-9]+", str(name or "").upper())
    kept = [w for w in words if w and w not in _NOISE]
    return "".join(kept)


def subject_candidates(catalog_docs: list, question: str) -> list:
    """Documents whose subject the question names. Empty when nothing matches,
    so the caller can decide - silently falling back to the whole catalog is how
    an Adobe question came to be answered from a 3M filing.

    Longest match wins: "AMERICANWATERWORKS" must beat "AMERICAN" if both are
    present, or a question about one issuer resolves to another.
    """
    asked = _core(question)
    if not asked:
        return []
    best, matched = 0, []
    for doc in catalog_docs:
        for signal in (doc.get("company"), (doc.get("doc_name") or "").split("_")[0]):
            core = _core(signal)
            if len(core) < 2 or core not in asked:
                continue
            if len(core) > best:
                best, matched = len(core), [doc]
            elif len(core) == best and doc not in matched:
                matched.append(doc)
            break
    return matched


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
    """Subject first, then period. A question naming no known subject falls back
    to the whole catalog, which is right for a single-subject corpus and is the
    caller's problem to notice otherwise."""
    scoped = subject_candidates(catalog_docs, question) or catalog_docs
    return resolve(scoped, *period_from_question(question))
