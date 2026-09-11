**Status:** open, not yet fixed — found live during gold-question testing,
2026-09-11, deliberately deferred rather than patched on the spot (see
`docs/adr/0002-multi-document-retrieval.md`'s own "note it, don't patch it"
precedent for a similarly-scoped decision).

# Two structurally identical year tables retrieve together, neither says which year it is

## Summary

3M's 10-K prints the same "Non-GAAP measures reconciliation" table twice —
once for the current fiscal year, once for the prior year, on adjacent pages
— with identical section structure and, critically, an **identical ingested
title on both**: `Table of Contents`. Neither chunk's metadata says which
year its own numbers belong to; the year exists only inside the table's own
printed rows. When a question needs "the full year 2023" figure and both the
2023 and 2022 copies of this table retrieve into the top-10 (as they did
here), the fact binder has to infer the year purely by reading the numbers -
and does not always get it right.

## Evidence

3M_2023_10K, `eval/corpora/ftsprism/questions/3m-002.yaml`'s question
("Calculate 3M's Operating Income (Loss) Margin for the full year 2023...").
Both tables retrieved, both titled identically:

```
[3] p47 table  titles=['Table of Contents']
[5] p46 table  titles=['Table of Contents']
```

Page 47's "Total Company" row (the FY2023 one, what the question actually
asks for):

```
| Total Company | | | | | | | | | | |
| GAAPamounts | $32,681 | (4.5)% | $ (9,128) | (27.9)% | ... |
```

Page 46's "Total Company" row (FY2022 - structurally identical position,
same column headers):

```
| Total Company | | | | | | | | | | |
| GAAPamounts | $ 34,229 | (3.2)% | $ 6,539 | 19.1 % | ... |
```

Both rows print `GAAPamounts` under a `Total Company` heading, in the same
column order, with no year token anywhere in either chunk's `titles` field.
The only place a year appears at all is deep in the page's own running text
above the table (a "Year ended December 31, 20XX" caption), which is not
what got captured as this chunk's title.

**Observed across separate runs of the identical question:**

- Run A: bound `operating_income_loss = -9128` from page 47 (correct - 2023)
  → computed margin -27.9%, matching gold exactly.
- Run B: bound `operating_income_loss = 6539` from page 46 (wrong - 2022)
  → computed margin 19.1%, which is FY2022's, not FY2023's.

Same question, same corpus, same retrieval configuration - the LLM binder's
own judgment call on which of two visually near-identical tables is the
right one is not stable run to run.

## Why it matters

This is the same root problem `docs/findings/associated-titles-defect.md`
already documents from a different angle (a table's ingested title not
reliably describing its own contents) - but that finding is about a *wrong*
title; this is about *no distinguishing* title at all between two
correctly-labeled-the-same tables. Fixing the workflow's title-assignment
bug would not fix this one: even a perfectly assigned "Table of Contents"
title (which is itself just the page's running header, not a real section
name) carries no year information to assign correctly in the first place.

## Not yet a fix - open questions for whoever picks this up

- Is the year recoverable onto the chunk at ingestion time (e.g. from the
  page's own preceding "Year ended December 31, 20XX" text), or does it need
  to be threaded through at query time instead?
- Could the fact binder be hardened by cross-checking a bound value's
  apparent year against the year(s) resolve_and_plan() already extracted from
  the question - reject a binding whose printed context contradicts the
  asked-for year, the same way validate_bindings() already rejects a binding
  whose value doesn't appear verbatim in the retrieved text?
- Is this specific to 3M's "current year + comparative prior year printed as
  two full copies of the same table" presentation, or does it recur across
  other filers/table shapes in this corpus?

## Reproduction

```
python -m eval.debug_question 3m-002 --chunks
```
Re-run a few times; the retrieved chunk set is stable but which page's row
gets bound is not.
