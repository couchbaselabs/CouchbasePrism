# PRISM

Governed retrieval over document corpora on Couchbase.

PRISM answers natural-language questions about a document corpus with scoped,
cited, verifiable retrieval — and, critically, **it works before anyone teaches
it the domain.** It plans what evidence a question needs, retrieves it, binds
each figure to a specific row and column, and where a question needs a computed
value it surfaces the defensible interpretations rather than silently choosing
one. A human ratifies the consequential choices; those become deterministic
behaviour from then on.

> *Works immediately through model-generated evidence plans, learns from
> governed corrections, and progressively converts repeated ambiguity into
> scoped, approved, deterministic behaviour.*

See [`design/architecture.md`](design/architecture.md) for the full design and
[`docs/adr/`](docs/adr/) for the decisions that shaped it.

## Three tiers

| Tier | Grain | Answers | Populated by |
|---|---|---|---|
| `catalog` | one per source document | which document is this? | extraction at ingest |
| `docs` | one per chunk | what does it say, here? | Couchbase AI Data Plane workflow |
| `dictionary` | one per domain concept | how is this defined, and who says so? | **model-proposed, human-approved** |

PRISM does not parse or chunk PDFs. Chunking and embedding are done by the
Couchbase AI Data Plane Unstructured Data Workflow; improvements to chunk
quality are feedback to that service, not code here. The one exception is a
fast cover-page pass (~100–600ms with PyMuPDF) used to build the catalog, which
never needs layout analysis and so should never pay for it.

## Runtime

```
question
  → catalog          resolve the document
  → evidence planner identify the concept; plan facts, tables, content anchors
                     (needs no dictionary entry — this is the central claim)
  → dictionary       optional: approved formula / interpretation policy
  → retrieval        anchors + vector, scoped to the resolved document
  → fact binding     entity, period, column, units, provenance — each checkable
  → calculation      governed → deterministic in code
                     ungoverned → labeled candidates, also evaluated in code
  → validation       bindings sound? candidates agree on the conclusion?
  → answer           prose around numbers already computed
  → proposal loop    unresolved ambiguity becomes a reviewable proposal
```

The LLM's authority depends on governance state: for an unapproved concept it
may propose and compute candidates; for an approved one it is forbidden from
computing and only binds facts and writes prose. That switch is what makes
"learns as it goes" compatible with "doesn't drift."

## Setup

Docker is the only prerequisite for running PRISM - the image bundles Python
and every dependency, nothing else to install on the host.

```bash
curl -sSL https://raw.githubusercontent.com/couchbaselabs/CouchbasePrism/main/config.example.yaml -o config.yaml
$EDITOR config.yaml   # fill in Capella, AI Data Plane, AWS, OpenAI, domains

curl -sSL https://raw.githubusercontent.com/couchbaselabs/CouchbasePrism/main/install.sh | bash -s -- ./config.yaml
# → http://localhost:8501
```

`config.yaml` is mounted into the container at runtime, never baked into the
image - the image is on a public registry, and it should never carry your
secrets. Re-run the same `install.sh` line to pull a newer version.

For development, run from a checkout instead:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp config.example.yaml config.yaml   # fill in Capella, AI Data Plane, AWS,
                                     # OpenAI and your domains - one file,
                                     # prism/config.py loads it directly

# or, with Docker, from the same checkout:
docker compose up --build
```

## Use

```bash
# build the catalog for a corpus (one document per PDF)
python manage.py build-catalog --company 3M

# run the benchmark (3-hybrid, the preferred architecture, is the default)
python -m eval.run_benchmark --company 3M --out out/cold.json

# reproduce the progression — each phase adds one layer
for p in 1-vector 2-catalog 3-hybrid; do
  python -m eval.run_benchmark --company 3M --phase $p --out out/$p.json
done

# inspect one question, every stage
python -m eval.debug_question 3m-2022-q3-quick-ratio
python -m eval.debug_question 3m-002 --chunks

# the governed correction — the entire human step
python manage.py approve "quick ratio" \
  --formula "(total_current_assets - inventory) / total_current_liabilities" \
  --healthy-at-or-above 1.0
python manage.py show-dictionary

# clear it again — the cold half of the two-pass demo
python manage.py reset-dictionary          # everything
python manage.py forget "quick ratio"      # one concept only

# or run cold without touching the approved dictionary at all
PRISM_DICTIONARY=/tmp/empty.yaml python -m eval.run_benchmark --company 3M

# demo
streamlit run app/streamlit_app.py

pytest
```

## The two-pass demo

The point is the delta, not the score.

**Cold** (empty dictionary) — for *"does 3M have a healthy liquidity profile
based on its quick ratio for Q2 FY2023?"* PRISM binds `total_current_assets`,
`inventory` and `total_current_liabilities` to the right rows in the right
column of the balance sheet, computes **two** labeled candidates (0.9578 and
1.441 — both defensible conventions), notices they straddle the 1.0 threshold,
and **declines the verdict**: it may compute a metric without holding authority
to judge it.

**Approve** one convention and its threshold.

**Warm** — the same question is now governed: a single deterministic 0.9578 and
an authorised verdict.

Each architecture phase adds one layer, measured against the previous one:

| Architecture | What it adds |
|---|---|
| `1-vector` | textbook RAG — kNN over the whole corpus |
| `2-catalog` | resolve the document first, then search inside it |
| `3-hybrid` | BM25 + content anchors + kNN via `SEARCH()`, plus binding, deterministic calculation and governance |

"Converged" is the honest word for what `eval.run_benchmark` reports. Several
quick-ratio and ROA conventions are defensible finance; a gold answer picks
one particular convention. Reporting a score as "correct" would assert
exactly the objective truth PRISM is built not to assume.

## Layout

```
prism/                    the system, corpus-agnostic
  catalog/                tier 1 — which document is this?
    extraction.py         cover page → catalog document
    repository.py         persistence
    resolver.py           question → doc_name
  retrieval/              tier 2 — what does it say, here?
    planner.py            question → concept, facts, content anchors
    anchor_search.py      printed row labels → the right chunk (IDF-ranked)
    vector_search.py      dense kNN
    hybrid_search.py      BM25 + kNN via the Search Vector Index
    combined_search.py    assemble and dedupe
  dictionary/             tier 3 — how is this defined, and who says so?
    evaluator.py          restricted AST evaluation
    repository.py         YAML persistence
    matching.py           concept → approved entry
    policy.py             value + policy → verdict
    approval.py           the human decision
  runtime/
    fact_binding.py       text → facts with entity/period/column/provenance
    calculation.py        governed value, or labeled candidates
    validation.py         is a conclusion authorised?
    answer.py             prose around numbers already computed
    pipeline.py           answer_question() — the single entry point
eval/
  phases/                 the three architectures, as CONFIGURATIONS of one runtime
  corpora/                swappable corpus adapters
  run_benchmark.py        run and score
  debug_question.py       one question, every stage
app/                      Streamlit demo — calls the same runtime
design/ docs/             architecture and ADRs
```

The phases are `PipelineOptions`, not separate implementations. The previous
iteration kept five forked eval scripts and they drifted — a prompt fix applied
to one silently failed to reach the others and cost a question that had been
passing.

A corpus is a test suite, not the system. Nothing in `prism/` imports from
`eval/`; a second corpus is a new adapter under `eval/corpora/`.

## Known limitations

- **Table section titles are unreliable.** Three tables in the sample corpus
  carry titles describing something else on the page. Content-anchor planning
  mitigates this; it does not eliminate it.
- **Complex tables lose sectional structure** when serialized flat. Neither
  gpt-4o-mini nor gpt-4o correctly reads a two-waterfall operating-margin table
  even when handed it alone. This is upstream of everything here.
- **Multi-metric judgments** (e.g. "is this capital-intensive?") need several
  metrics and a policy each; only the single-metric path is built.

## License

MIT — see [`LICENSE`](LICENSE).
