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


def _cores(name: str) -> set:
    """Every spelling of one identity worth comparing.

    An ampersand is written three ways and readers use all of them: "J&J" is
    also "JnJ" and "J and J". Dropping the ampersand alone yields JJ, which
    matches none of the others - the alias was recorded correctly and still
    failed to match the question. Generated here rather than asked of a model,
    because it is orthography, not knowledge.
    """
    raw = str(name or "").upper()
    return {c for c in (_core(raw), _core(raw.replace("&", "N")),
                        _core(raw.replace("&", " AND "))) if c}


def _abbreviation_hit(core: str, question: str) -> int:
    """Length of the longest question word that PREFIXES this subject's core.

    Questions abbreviate what filings spell out: "JPM" for JPMorgan Chase, "MGM"
    for MGM Resorts International. Those are prefixes of the full name, so they
    can be matched without knowing anything about the company. Abbreviations that
    are not prefixes - "JnJ", "AMEX" - are not reachable this way and need
    recorded aliases.

    Three characters minimum: shorter prefixes collide across issuers.
    """
    best = 0
    for word in re.findall(r"[A-Za-z0-9]{3,}", question.upper()):
        if core.startswith(word) and len(word) > best:
            best = len(word)
    return best


def subject_candidates(catalog_docs: list, question: str) -> list:
    """Documents whose subject the question names. Empty when nothing matches,
    so the caller can decide - silently falling back to the whole catalog is how
    an Adobe question came to be answered from a 3M filing.

    Longest match wins: "AMERICANWATERWORKS" must beat "AMERICAN" if both are
    present, or a question about one issuer resolves to another. A full-name
    match always beats an abbreviation, since an abbreviation is weaker evidence.
    """
    asked = "|".join(sorted(_cores(question)))
    if not asked:
        return []
    best, matched = 0, []
    for doc in catalog_docs:
        signals = [doc.get("company"), (doc.get("doc_name") or "").split("_")[0]]
        signals += list(doc.get("aliases") or [])
        score = 0
        for signal in signals:
            for core in _cores(signal):
                if len(core) < 2:
                    continue
                if core in asked:
                    # Full name present: scored above any abbreviation of it.
                    score = max(score, len(core) + 100)
                else:
                    score = max(score, _abbreviation_hit(core, question))
        if not score:
            continue
        if score > best:
            best, matched = score, [doc]
        elif score == best and doc not in matched:
            matched.append(doc)
    return matched


_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}


def event_date_from_question(question: str):
    """An explicit calendar date the question names, as ISO, or None.

    Several questions ask about a specific filing: "the 8k filing dated 1st July
    2022", "the separation announced August 30, 2023". A report filed on a date
    is a different document from the annual report covering that year, and the
    date is the only thing distinguishing them.
    """
    text = (question or "").lower()
    for pattern in (r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+),?\s+(\d{4})\b",
                    r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"):
        for match in re.finditer(pattern, text):
            a, b, year = match.groups()
            day, month = (a, _MONTHS.get(b)) if a.isdigit() else (b, _MONTHS.get(a))
            if month:
                return f"{int(year):04d}-{month:02d}-{int(day):02d}"
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
            # The quarter was parsed and then ignored: quarterly[0] returned the
            # first 10-Q of the year whatever quarter was asked for, so "2022 Q2"
            # answered from Q1. Fiscal quarters do not map to fixed months across
            # issuers, but the year's 10-Qs in date order ARE Q1, Q2, Q3.
            quarterly.sort(key=lambda d: d.get("period_end_date_iso") or "")
            return quarterly[min(quarter, len(quarterly)) - 1]["doc_name"]
    annual = [d for d in matches if form_of(d.get("doc_type")) == "10-K"]
    return (annual[0] if annual else matches[0])["doc_name"]


def resolve_for_question(catalog_docs: list, question: str) -> str:
    """Subject, then a named date if there is one, then period.

    A named date outranks the period because it is more specific: a question
    about what was filed on 30 August 2023 is not asking about the annual report
    for 2023, even though both match the year.
    """
    scoped = subject_candidates(catalog_docs, question) or catalog_docs
    on_date = event_date_from_question(question)
    if on_date:
        dated = [d for d in scoped if on_date in (d.get("doc_name") or "")
                 or on_date == d.get("period_end_date_iso")]
        if dated:
            return dated[0]["doc_name"]
    return resolve(scoped, *period_from_question(question))
