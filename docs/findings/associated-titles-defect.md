# Table chunks receive the wrong `associated-titles`

**Component:** Couchbase AI Data Plane, Unstructured Data Workflow
**Not a docling defect** — docling's output is correct; see "Docling is correct" below.
**Source:** 3M 2022 10-K, ingested to `acme.prism.docs`

## Summary

Every `type: table` chunk is assigned the **last `section_header` on its page**
instead of the **nearest preceding** one. Text chunks are unaffected — they get
the correct heading — so the two element types appear to follow different
association paths.

## Evidence

Docling assigns page 27's operating-expenses table to the heading directly
above it. The workflow assigns a heading from further down the page.

| docling ref | page | assigned by workflow | correct (nearest preceding) |
|---|---|---|---|
| `#/tables/12` | 27 | `Goodwill Impairment Expense:` | `Operating Expenses:` |
| `#/tables/13` | 28 | `PERFORMANCE BY BUSINESS SEGMENT` | `Provision for Income Taxes:` |
| `#/tables/14` | 28 | `PERFORMANCE BY BUSINESS SEGMENT` | `Income from Unconsolidated Subsidiaries, Net of Taxes:` |
| `#/tables/15` | 28 | `PERFORMANCE BY BUSINESS SEGMENT` | `Net Income (Loss) Attributable to Noncontrolling Interest:` |

In all four cases the assigned value is exactly the final `section_header`
appearing on that page.

Text elements on the same pages are correct. `#/texts/315` — the sentence
beginning "The Company is continuing the ongoing deployment of an enterprise
resource planning (ERP) system" — sits immediately *after* the same table and
is correctly given `Operating Expenses:`. Under a last-header-on-page rule it
would have been given `Goodwill Impairment Expense:` as well.

## Docling is correct

From `docling-serve` on the same PDF:

- `#/tables/12` has `parent: {"$ref": "#/body"}` and `prov[0].page_no: 27`.
- In `body.children` reading order it is the **immediately following sibling**
  of `#/texts/314`, whose `label` is `section_header` and whose text is
  `Operating Expenses:`.
- Every heading involved is labelled `section_header`, not plain `text`.
- The Markdown export renders the table under `## Operating Expenses:`, in the
  correct position.

So the nearest-preceding heading is not merely derivable — for this table it is
the previous element in `body.children`.

## Why it matters

`associated-titles` is not only metadata. It is prefixed into `text-to-embed`,
so the stored embedding for this table begins:

```
Section Title: Goodwill Impairment Expense: Content: | (Percent of net sales) | 2022 | 2021 | Change |
```

The wrong heading is therefore embedded into the vector, steering it away from
the table's actual subject. Measured effect on this chunk, which contains
3M's FY2022 operating margin (`19.1% / 20.8% / (1.7)%`):

- **Vector rank 81 of 730** chunks within its own filing.
- Simultaneously the **single best BM25 hit** in that filing for the query.

A retrieval configuration relying on the vector channel alone does not reach
it. Recall@10 against an annotated evidence-page benchmark is 0.54 with
score-fused hybrid search and 0.67 once the lexical channel is given an
independent quota — the difference is this chunk.

## Second instance

3M 2023 Q2 10-Q, page 1: the registered-securities table (`Title of each class
| Trading Symbol(s) | Name of each exchange on which registered`, containing
`MMM26`, `MMM30`, `MMM31`) is assigned `associated-titles: ['Delaware',
'41-0417775']` — the state of incorporation and the IRS Employer
Identification Number, not the table's actual subject.

## Related: document-level title

Every chunk in the 10-Q carries `Document Title: Delaware`, also prefixed into
`text-to-embed`. "Delaware" is the state of incorporation from the cover page.
Every embedding in the filing carries it.

## Suggested fixes

1. Associate a table with the nearest preceding `section_header` in
   `body.children` reading order, matching the behaviour already applied to
   text elements.
2. Make the title prefix in `text-to-embed` optional. Consumers can then avoid
   embedding a heading whose accuracy they cannot verify.

## Not defects

Worth preserving as-is:

- **Table parsing is accurate.** The page 27 grid comes through with correct
  columns, rows and values.
- **The structured companion documents are accurate** — one per table,
  `content.schema` + `content.data` in pandas orient, 1,059 of them in this
  corpus. They currently carry no `xmeta-data.filename` and no embedding, so
  nothing can retrieve or scope them; adding both would make them usable.

## Reproduction

Ingested with: RECURSIVE_SPLITTER, max tokens 2000, chunk overlap 50, footers
excluded, page range all. The same titles appear under SEMANTIC_SPLITTER with
different chunking parameters — table chunk count is identical (1,059) across
both ingestions and the titles do not change, so this is upstream of chunking.
