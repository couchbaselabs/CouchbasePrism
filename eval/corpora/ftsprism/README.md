# ftsPrism

PRISM's own open sample corpus - small on purpose, expandable on purpose,
built to carry zero licensing doubt rather than manage some.

## Why this exists

PRISM's public/GTM story needed a corpus with zero licensing doubt to
manage, not one to manage carefully - ftsPrism is what the public demo, the
field-enablement kit, and the Docker image all run on.

**Every piece of it is sourced or authored so there is no doubt to manage:**

- **PDFs** are pulled directly from [SEC EDGAR](https://www.sec.gov/edgar/search/) -
  public record, filed under legal disclosure requirements, not a copy of
  anyone's compiled dataset. `documents.yaml` records the exact filing URL
  each PDF came from.
- **Questions** are written in our own words, drafted directly against the
  filing itself.
- **Gold answers and evidence** are facts and short verbatim quotes from the
  filing itself - not reproductions of a third party's annotation of it.

MIT, same as the rest of this repo - see the root [`LICENSE`](../../../LICENSE).

## Layout

```
ftsprism/
  documents.yaml       one entry per PDF: doc_name, company (filter only, not
                        scored - see the comment in the file), source_url,
                        sector, lineage
  pdfs/
    3M/                 one subfolder per company, PDFs named {doc_name}.pdf
    BestBuy/            (a company added as HTML instead of PDF - see below -
                        gets its own subfolder here too, same rule)
  questions/            one YAML file per question; filename (minus .yaml)
                        is the question's id
  fetch_filings.py       pulls a company's filings straight from SEC EDGAR
                        and saves them into pdfs/{label}/ - see below
```

Only fields PRISM cannot derive itself get hand-supplied. `company`,
`doc_type` and the filing's period end date come from PRISM's own catalog
extraction (`prism/catalog/extraction.py`) and are *scored against*, not
handed to it - putting them in `documents.yaml` would mean grading the
system against its own homework. `sector` is the one true exception: it
isn't printed on any cover page, so no extraction step can ever produce it.

See `questions/_TEMPLATE.yaml` for the full question schema, field by field.

## Adding a company's filings

`fetch_filings.py` pulls straight from SEC EDGAR's submissions API - never a
company's own investor-relations site (see the module docstring for why that
distinction matters here). It's a dry run by default; nothing is written
until `--go` is passed, and that step needs a go-ahead each time since it's a
real download:

```
python fetch_filings.py --ticker MMM --label 3M --forms 10-K,10-Q --years 2020-2026
python fetch_filings.py --ticker MMM --label 3M --forms 8-K --item 2.02 --years 2020-2026 --go
python fetch_filings.py --ticker MMM --label 3M --forms "DEF 14A" --years 2020-2026 --go
```

`--label` sets both the `pdfs/{label}/` subfolder and the `doc_name` prefix;
`--ticker` only resolves a CIK, so the two can differ (e.g. ticker `MMM`,
label `3M`, matching how the rest of PRISM already names 3M's documents). An
Item-2.02 8-K's real content is a separate exhibit, not its thin cover form -
the script resolves that from the filing's own index page rather than
guessing a filename pattern. It still prints a manifest and appends rows to
`documents.yaml` for you to review - a filing with no clean EX-99 exhibit is
skipped outright rather than guessed at, and anything that looks like an
ancillary/supplementary filing rather than a full quarterly release is
called out for a look, not dropped silently.

3M's first 64 filings (10-K, 10-Q, earnings-8-K, DEF 14A across FY2020-2026)
were added this way - see the comment at the top of `documents.yaml` for the
exact commands.

A future company doesn't have to be PDF - the AI Data Plane workflow ingests
HTML directly, so a company added as HTML (planned for Best Buy) just needs
its own `pdfs/{label}/` subfolder of `.html` files instead of `.pdf`.
`documents()` in `ftsprism.py` currently globs `*.pdf` only, so that's a
small, known update to make when the HTML case actually arrives (glob both
extensions, or track each document's format explicitly in `documents.yaml`)
- not a redesign, but not done yet either.

## Adding a question

1. Confirm the document is already in `documents.yaml` (see above if not).
2. Optional: run it through PRISM's own catalog + answer pipeline first and
   use what it produces as a first draft of the question, answer, and
   evidence - note as much in `notes:`. Faster than authoring cold, but the
   human-verification step below is not optional either way.
3. Copy `questions/_TEMPLATE.yaml` to `3m-NNN.yaml` (the next free number -
   matches the `number:` field, which drives ordering in the UI dropdown),
   fill in every field, and open the actual filing to check the value and
   evidence quote against it directly. Only set `verified: true` once that
   check has actually happened - a wrong gold is worse than no gold, since
   it passes silently, forever.

## Running it

Same harness, just a different `--corpus`:

```
python -m eval.run_benchmark --corpus ftsprism
```
