**Status:** the catalog gap this doc describes is FIXED (see "Fix" below) -
`company`/`period_end_date_iso` are now populated for all 31 earnings 8-Ks.
3m-003 STILL does not resolve to the right document, but for a different,
later-stage reason: two same-period documents of different form now
correctly both match, and there is no tie-break between them - see
"What's still open" below. That remainder is deliberately not being patched
here; it is `docs/adr/0002-multi-document-retrieval.md` territory, still
`status: proposed`, awaiting review before any of it gets built.

## Fix

`extraction.py` gained two deterministic fallbacks, following the exact
pattern `doc_type_from_doc_name()`/`period_from_doc_name()` already used for
this same document class:

- `company_from_doc_name()` - the doc_name's own leading segment ("3M" from
  "3M_8K_...") mapped to the manifest's canonical form ("3M COMPANY").
- `period_end_from_earnings_headline()` - the press-release headline states
  a fiscal quarter directly ("3M Reports Second Quarter 2023 Results"); two
  tiers, because real headlines turned out messier than one shape: a year
  printed adjacent to the quarter word first, falling back to the filed
  date already in the document's own name (adjusted -1 year for a Q4/
  full-year release, which files in January of the FOLLOWING year) when no
  adjacent year is stated.

Verified: rebuilt the whole 3M catalog (`manage.py build-catalog`), all 31
earnings 8-Ks now carry both fields; `resolve_documents(["3M COMPANY"], [],
[2023], [2])` correctly returns BOTH `3M_2023_Q2_10Q` and
`3M_8K_2023-07-25_000056_earnings` where before it returned only the 10-Q.
6 new regression tests added to `tests/test_catalog_and_runtime.py`.

## What's still open

`pipeline.py`'s document-picking sort (`sorted(..., key=(doc_period,
quarter), reverse=True)`) has no tie-breaker for two documents that share a
period but differ in form - a stable sort over an inherently unordered N1QL
result, so whichever the server happened to return first silently wins.
This was always latent; the catalog fix above is what makes it visible for
the first time (previously only ONE document - the 10-Q - ever matched at
all). Deliberately not addressed here - see
`docs/adr/0002-multi-document-retrieval.md` for the actual multi-document
design this belongs to.

# Earnings-release 8-Ks are never catalogued by period - none of them

## Summary

All 31 earnings-release 8-Ks in this corpus have `period_end_date_iso: null`
and `company: null` in the catalog. Not a few of them - every single one.
Any question needing a *specific quarter's* earnings release therefore
cannot resolve to it through the normal companies/years/quarters filter,
no matter how the resolver reasons about document type: the field that
filter depends on was never populated in the first place.

## Root cause

`prism/catalog/extraction.py`'s `CLASSIFY_SYSTEM_PROMPT` looks for exact SEC
cover-page boilerplate:

- company: text near `"(Exact name of registrant as specified in its
  charter)"`
- period_end_date: text matching `"for the fiscal/quarterly year/period
  ended [DATE]"`

That boilerplate is real and reliable on a 10-K/10-Q/DEF 14A's own cover
page - it's the standardized SEC filing header. An "earnings release" 8-K
is structurally different: its first pages are the company's own **press
release** (a headline like "3M Reports Second-Quarter 2023 Results," a
dateline, no registrant-charter language at all). The extraction prompt
finds nothing to extract because the phrases it's built to recognize
genuinely are not there - this is not a parsing failure on a document that
has the data; the document is shaped differently from the ones this
extraction path was built for.

## Evidence

```
3M_8K_2020-01-28_007328_earnings | type: None | period_end: None | company: None
3M_8K_2020-04-28_051945_earnings | type: None | period_end: None | company: None
3M_8K_2020-07-28_087014_earnings | type: None | period_end: None | company: None
3M_8K_2020-10-27_118371_earnings | type: 10-K | period_end: None | company: None   <- doc_type even wrong here
3M_8K_2021-01-26_007403_earnings | type: None | period_end: None | company: None
```

`doc_type` is sometimes populated (inconsistently, and once outright wrong -
"10-K" for an earnings 8-K), but `period_end_date_iso` and `company` are
null across the board. `doc_type` being sometimes-right suggests the model
occasionally infers a plausible SEC form label anyway despite no clean
match in the text; `period_end_date` has no equivalently-guessable fallback
phrase, so it never gets one.

## Why it surfaced now

3m-003 asks for "Adjusted Free Cash Flow Conversion for the second quarter
of 2023," expecting `3M_8K_2023-07-25_000056_earnings`. Two separate,
already-diagnosed-and-partly-fixed layers sit in front of this one:

1. The resolver was guessing a document FORM from the metric's nature
   ("non-GAAP measures are typically in a 10-Q") - fixed, see the commit
   adding a negative example to `resolve_and_plan.py`'s resolution rules.
2. Even with that fixed (`doc_types` correctly left empty), the underlying
   `resolve_documents(quarters=[2])` query returns ONLY the 10-Q - not
   because of a tie-break issue between two matching documents (the
   original theory), but because the 8-K never matches the quarter filter
   AT ALL. `catalog.quarter_of(period_end_date_iso)` has nothing to compute
   from.

So the resolver reasoning is now correct, and it still cannot reach the
right document - the gap is entirely in catalog data, not in any prompt.

## How the open questions above were resolved

- The filed date alone was NOT trusted for the period end (it's a reporting
  lag - "2023-07-25" for a Q2 release would wrongly imply Q3 if read as a
  period end directly). It IS trusted for the YEAR, as tier 2's fallback,
  since it's reliable specifically for that: same calendar year for a
  Q1-Q3 release, one year prior for a Q4/full-year release filed in
  January.
- A separate extraction PROMPT path (LLM-based) was considered and
  rejected in favor of a plain regex fallback, same reasoning
  `doc_type_from_doc_name()`/`period_from_doc_name()` already used: the
  headline shape is regular enough (an ordinal + "quarter", optionally
  hyphenated) that no LLM judgment is needed, and a deterministic fallback
  can't hallucinate a quarter that isn't there the way an LLM prompt
  extension risked.
- `company` was fixed too (`company_from_doc_name()`) - it turned out NOT
  to be a pure convenience filter: `resolve_documents()`'s own query
  requires `UPPER(d.company.value) IN $companies` unconditionally, so a
  null company independently excluded every earnings 8-K from any query
  regardless of the period fix.

## Reproduction

```python
from prism import catalog
docs = catalog.repository.load_all(scope="secfilings")
earnings_8ks = [d for d in docs if "8K" in d["doc_name"] and "earnings" in d["doc_name"]]
# all 31 have period_end_date_iso=None, company=None
```
