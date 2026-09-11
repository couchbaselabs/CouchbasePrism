**Status:** open, not yet fixed — found live during gold-question testing,
2026-09-11, deliberately written up rather than patched on the spot (same
"note it, don't guess a fix" treatment as
`docs/findings/duplicate-year-tables-no-disambiguation.md`).

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

## Not yet a fix - open questions for whoever picks this up

- The filed date is already IN the document's own name
  (`3M_8K_2023-07-25_000056_earnings` - filed 2023-07-25). Could
  `period_end_date_iso` be inferred from the filed date for this document
  class specifically (an earnings release files ~3-4 weeks after quarter
  end, a fairly reliable convention), or is that too fragile/off-by-a-
  reporting-lag to trust for quarter matching?
  - `resolve_documents()`'s quarter filtering can hover the intended period
    if it uses the FILED date's own quarter/month directly rather than a
    derived fiscal period end.
- Is a separate classification prompt (looking for a press-release-style
  headline - "Reports Second-Quarter 2023 Results" - instead of SEC
  boilerplate) worth a dedicated extraction path for this document class,
  given it's a real, recurring shape (31 of them, and presumably every
  future earnings-release 8-K added to this corpus)?
- `company` is also null for all 31 - does anything currently depend on it
  for earnings 8-Ks specifically, or is it only used as a convenience
  filter (per `documents.yaml`'s own note) that happens not to matter here
  since this corpus is single-company?

## Reproduction

```python
from prism import catalog
docs = catalog.repository.load_all(scope="secfilings")
earnings_8ks = [d for d in docs if "8K" in d["doc_name"] and "earnings" in d["doc_name"]]
# all 31 have period_end_date_iso=None, company=None
```
