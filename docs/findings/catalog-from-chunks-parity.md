# Catalog built from chunks matches the PDF path, with one known gap

**Status:** measured, shipped - catalog build no longer needs a local PDF
**Scale:** 1 confirmed miss in 32 sampled documents (~3%), company field only

## What changed

`prism/catalog/extraction.py` derived every catalog entry from the PDF's own
cover page, read fresh via PyMuPDF at build time. Since the Couchbase AI Data
Plane workflow ingests every page into `docs` before catalog build ever runs,
that same cover-page text is already sitting in Couchbase with
`meta-data.page-number` indexed. `cover_text_from_chunks()` reads it from
there instead - `manage.py build-catalog` and the app's "Run catalog" button
no longer need local PDF file access at all.

## The ordering problem this needed, and how it was tested

docling's `element-id` (`#/texts/N`, `#/tables/N`, ...) is not one counter -
each element type has its own independent sequence. A live query for a real
cover page came back as `texts/10, texts/3, texts/1, tables/0, texts/0` - the
actual title chunk (`texts/0`) arrived LAST. This is the chunks equivalent of
the exact problem PyMuPDF's `sort=True` already exists to fix (documented in
`cover_text()` against Activision's 2015 10-K, whose PDF content-stream order
scrambled badly enough to corrupt extraction). `_chunk_sort_key()` sorts by
`(page, is-a-table, index-within-that-element-type's-own-counter)` and is
covered by a pure-logic test (`test_chunk_sort_key_undoes_scrambled_element_ids`)
using that exact scrambled ordering as input.

Verified against the same document the PDF path's fix was written for:
`ACTIVISIONBLIZZARD_2015_10K` extracts company/doc_type/period_end_date
cleanly (confidence 0.99 on all three) via the new chunks path.

## Parity measurement

Two random samples compared `build_from_pdf` against `build_from_chunks` on
the SAME documents, both against real ingested data (not a mock):

- 12 documents (seed 42): 2 came back `company=None` on chunks
  (`COCACOLA_2016_10K`, `NIKE_2020_10K`)
- 20 documents (seed 7): 0 mismatches

Investigating the two apparent misses split them into two different
outcomes:

**`COCACOLA_2016_10K` is not a regression.** The PDF path ALSO returns
`company=None` on this document - both paths cut off mid-sentence right
before "(Exact name of registrant..." Whatever this filing's cover-page
layout does defeats plain-text extraction either way. Parity, not a new gap.

**`NIKE_2020_10K` is a genuine, traced miss.** The chunk containing the
title-block paragraph reads:

    ...Commission File No. 1-10635 (Exact name of Registrant as specified
    in its charter) One Bowerman Drive, Beaverton, Oregon...

"NIKE, Inc." - which sits on its own line in the real filing, between
"Commission File No. 1-10635" and "(Exact name of Registrant...)" - is
simply ABSENT from the ingested chunk's text. PyMuPDF's raw PDF text
extraction still has it (`build_from_pdf` returns `'NIKE, Inc.'` at
confidence 0.99 on the identical document). This is not a page-range or
type-filter miss on the query side - a direct search across ALL pages of
this document for "Exact name" / "Registrant as specified" text returns
exactly one chunk, and that chunk is genuinely missing the company name the
raw PDF still contains.

## Why this isn't being coded around

This is the same category of issue as
`docs/findings/associated-titles-defect.md` - a real limitation in what the
AI Data Plane's chunking preserves, not something wrong in PRISM's query or
sort logic. A fallback to the PDF when `company` comes back null would
silently reintroduce the PDF dependency this change was meant to remove, for
a ~3% edge case where doc_type and doc_period still extract correctly
regardless - only the company name is lost. Per the project's standing
policy: chunk-quality issues are feedback to that service, not code here.

## What this means going forward

Catalog build (CLI and the UI's "Run catalog" button) is fully chunk-sourced
now. Expect the occasional document where `company` comes back null even
though the underlying filing states it clearly - `doc_type` and
`doc_period` are unaffected by this specific failure mode in every case
observed so far. A document missing `company` still catalogs; it just
resolves less precisely by subject until corrected (see
`prism.catalog.resolver` module docstring on why subject resolution needs
two independent signals in the first place).
