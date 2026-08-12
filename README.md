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

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # fill in Capella, AI Data Plane, OpenAI, S3
set -a && source .env && set +a

git clone https://github.com/patronus-ai/financebench   # the test corpus
```

## Use

```bash
# build the catalog for a corpus (one document per PDF)
python manage.py build-catalog --corpus financebench --company 3M

# run the benchmark
python -m eval.run_benchmark --company 3M --out out/cold.json

# inspect one question, every stage
python -m eval.debug_question financebench_id_00807
python -m eval.debug_question financebench_id_00941 --chunks

# the governed correction — the entire human step
python manage.py approve "quick ratio" \
  --formula "(total_current_assets - inventory) / total_current_liabilities" \
  --healthy-at-or-above 1.0
python manage.py show-dictionary

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

On the 8 FinanceBench questions for 3M, convergence with FinanceBench's own
conventions moved 25% (vector only) → 38% (catalog-filtered) → 50% (evidence
planner) → **62%** (after one approval).

"Converged" is the honest word. Several quick-ratio and ROA conventions are
defensible finance; FinanceBench expects particular ones. Reporting this as
"62% correct" would assert exactly the objective truth PRISM is built not to
assume.

## Layout

```
prism/           the system, corpus-agnostic
  catalog.py     tier 1 — cover-page extraction, build, resolution
  retrieval.py   tier 2 — evidence planner, content anchors, search
  dictionary.py  tier 3 — entries, safe evaluator, policies, approval
  runtime.py     bind → calculate → validate → answer  (one entry point)
eval/            benchmark harness; corpora/ holds swappable adapters
app/             Streamlit demo — calls the same runtime the benchmark does
design/ docs/    architecture and ADRs
```

FinanceBench is one test corpus, not the system. Nothing in `prism/` imports
from `eval/`; a second suite is a new adapter under `eval/corpora/`.

## Known limitations

- **Table section titles are unreliable.** Three tables in the sample corpus
  carry titles describing something else on the page. Content-anchor planning
  mitigates this; it does not eliminate it.
- **Complex tables lose sectional structure** when serialized flat. Neither
  gpt-4o-mini nor gpt-4o correctly reads a two-waterfall operating-margin table
  even when handed it alone. This is upstream of everything here.
- **Multi-metric judgments** (e.g. "is this capital-intensive?") need several
  metrics and a policy each; only the single-metric path is built.
