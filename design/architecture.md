# PRISM architecture — three-tier model

Status: living document, reflects the state reached via internal eval work
(2026-08-06 through 2026-08-11) and the architecture review of 2026-08-11.
Supersedes `design/archive/catalog_document.md`, whose single-collection model has
since split into three tiers with distinct purposes and distinct names. Decisions
recorded here that are hard to reverse get their own ADR under `docs/adr/`; this
document is the map, the ADRs are the load-bearing walls.

---

## 0. Naming — three different things, three different names

The original design doc warned about "catalog" meaning two different things
(type-level vs. per-document). A third concept has since emerged, and the same
discipline applies: settle it once, in writing, before the words start
colliding in conversation.

| Name | Grain | Answers | How it's populated | Status |
|---|---|---|---|---|
| **`catalog`** | one per source document | "which document is this?" (company, doc type, period) | extracted at ingestion | Built, validated on a 50-document sample |
| **`docs`** | one per chunk | "what does the document say, here?" (text, embedding, page, section title) | chunked + embedded at ingestion | Built via Couchbase AI Data Plane Unstructured Data Workflow |
| **`dictionary`** | one per domain concept | "how is this concept defined, and who says so?" (formula, interpretation policy, recognition terms) | **model-proposed from real use, human-approved when consequential** | Designed (ADR-0001), not yet built |

`catalog` and `docs` are per-*document*/per-*chunk* — they scale with corpus
size. `dictionary` is per-*concept* — it scales with how many distinct
financial (or clinical, or legal) ideas the domain actually exercises, which is
a much smaller, slower-growing number.

The dictionary is **not hand-curated**. Nobody authors formulas from scratch
before the product is useful. The system proposes candidate interpretations
from real retrieved evidence; a human ratifies the consequential ones. That
distinction is the differentiator — see §4 and ADR-0001.

---

## 1. System overview

```
question
   │
   ▼
┌────────────────────┐
│ catalog            │  resolve the document (company, period, doc type)
└────────┬───────────┘
         ▼
┌────────────────────┐
│ evidence planner   │  identify the concept asked about; plan the facts,
│                    │  tables, and CONTENT ANCHORS needed to answer it.
│                    │  Requires no dictionary entry to do this.
└────────┬───────────┘
         ▼
┌────────────────────┐
│ dictionary lookup  │  OPTIONAL. If an approved entry exists for the
│                    │  identified concept, it supplies locked semantics
│                    │  (formula, interpretation policy). If not, the
│                    │  pipeline continues — it does not stall.
└────────┬───────────┘
         ▼
┌────────────────────┐
│ docs (retrieval)   │  vector + BM25, filtered to the resolved document,
│                    │  anchored on the plan's content anchors
└────────┬───────────┘
         ▼
┌────────────────────┐
│ fact binding       │  bind each value to a meaning: entity, period,
│                    │  column, units, and source provenance
└────────┬───────────┘
         ▼
┌────────────────────┐
│ calculation        │  governed   → deterministic execution in code
│                    │  ungoverned → LLM proposes labeled candidates
└────────┬───────────┘
         ▼
┌────────────────────┐
│ validation         │  bindings sound? candidates agree on the conclusion
│                    │  the question asks for? is there an approved
│                    │  interpretation policy to judge against?
└────────┬───────────┘
         ▼
┌────────────────────┐
│ answer synthesis   │  prose around numbers already computed — the LLM
│                    │  never re-derives a governed value here
└────────┬───────────┘
         ▼
   proposal / approval loop ──────► dictionary
   (unresolved ambiguity becomes a reviewable proposal)
```

**Evidence planning precedes dictionary dependence.** This ordering is
deliberate and was corrected from an earlier draft that placed the dictionary
between the catalog and retrieval — which wrongly implied PRISM needs a
dictionary entry before it can retrieve anything. It does not. The quick-ratio
walkthrough (§4.1) produced a complete balance-sheet evidence plan — the exact
rows to find, in the exact table — with no dictionary entry in existence. The
dictionary's role begins *after* evidence is in hand, to settle which
interpretation of that evidence is authoritative.

**Concept recognition lives in the evidence planner**, not the dictionary. The
planner must identify *what is being asked about* before it can plan evidence
for it, and it must be able to do so for concepts that have never been seen
before. A dictionary entry's `recognition` terms are a fast path for
already-known concepts, not the only way in.

## 2. Tier 1 — Catalog

One document per PDF. Current schema (validated): `doc_name` (join key,
a `{COMPANY}_{PERIOD}_{TYPE}` convention), `company`,
`doc_type`, `period_end_date` (verbatim, citable), `period_end_date_iso`
(derived, for FTS date-range queries — never overwrites the verbatim value).
Each field carries `{value, confidence, source, source_span, page}` —
`source: "pdf_text"` for genuine extractions, `"ungrounded"` when a mechanical
grounding check catches a value not actually present in the source text (see
§6).

**Extraction pipeline**: PyMuPDF (`sort=True` — critical, see below) pulls
cover-page text; OpenAI closed-set classification extracts the four fields;
regex alone matches OpenAI on `doc_type`/`doc_period` (both are structurally
fixed-vocabulary problems) but not `company` (17% miss rate — company names
aren't a fixed-vocabulary problem the way "FORM 10-K" is). Validated at 100%
agreement with independently-verified document metadata on a 50-document sample.

`sort=True` matters because PyMuPDF's default text order follows the PDF's
content stream, not visual position — on some filings (Activision's 2015
10-K) this scrambles cover-page paragraph order badly enough to corrupt
extraction. `sort=True` fixes it; the PDFBox equivalent is
`setSortByPosition(true)` if/when this moves to the Java backend.

## 3. Tier 2 — Docs (chunks)

Ingested via Couchbase's AI Data Plane Unstructured Data Workflow: PDFs in S3
(`{AWS_BUCKET}_{AWS_FOLDER}_{doc_name}.pdf` naming) → chunked, embedded
(2048-dim), written to `{scope}.docs`. Each chunk carries `text-to-embed`
(prefixed `"Section Title: ...\nContent: ..."`), `text-embedding`, `meta-data`
(`page-number`, `associated-titles`, `type` — `text`/`table`/`list_item`/
`page_footer`), `xmeta-data` (`filename`, source metadata).

**Known reliability issue, confirmed 3 times, not a one-off**:
`meta-data.associated-titles` is wrong on a meaningful fraction of table
chunks — not missing, *wrong*: a 3M operating-margin table, a debt-securities
table (titled `["Delaware", "41-0417775"]`), and a Consumer-segment table
(mislabeled "PERFORMANCE BY GEOGRAPHIC AREA") all carried titles describing
something else on the page (see `docs/findings/associated-titles-defect.md`
for the full detail on one of these). Any retrieval strategy that trusts this
field inherits the failure silently.

The evidence planner (§1) is the primary mitigation: a plan that specifies
**content anchors** — row labels that must appear verbatim in the right table,
e.g. `["Total current assets", "Total current liabilities"]` for a balance
sheet, or `["Organic sales", "Divestitures", "Translation"]` for a segment
performance table — does not depend on the title being correct at all. This is
how both remaining tables were located manually during the 2026-08-11 review.
A table's own row-label fingerprint is a far more reliable identifier than its
assigned title.

**Known ingestion-quality issue, unsolved**: complex tables with implicit
sectional structure (two mirrored "increase/(decrease) due to" waterfalls in
one table) lose that structure when serialized to flat pipe-delimited
markdown. Neither gpt-4o-mini nor gpt-4o correctly identifies which subtotal
answers a question even when given the table alone, isolated from all other
context. Not a retrieval or prompt problem — a representation problem,
upstream of everything in §1, and **not addressed by any part of this
architecture.**

## 4. Tier 3 — Dictionary (designed, not built)

### 4.1 Why it exists

Some questions ask for a value no document states verbatim — a ratio, a
computed change — because the value doesn't exist until it is computed from
retrieved raw facts. Retrieval can find the facts; it cannot decide which of
several genuinely defensible conventions is the one a given
benchmark/customer/analyst means. Confirmed directly, twice:

| Metric | Matches the gold answer's convention | Other defensible conventions, same retrieved facts |
|---|---|---|
| Quick ratio | `(current_assets − inventory) / current_liabilities` = **0.96** | `(cash + securities + receivables) / current_liabilities` = 0.85; `(current_assets − inventory − prepaids) / current_liabilities` = 0.90 |
| Return on assets | `net_income_attributable_to_3M / total_assets` = **12.44% ≈ 12.4%** | `net_income_including_NCI / total_assets` = 12.47% ≈ 12.5% |

**Note the column heading.** These are not "correct" values and the
alternatives are not errors — every listed convention is defensible finance.
0.96 is what *the gold answer expects*; a different organization could
reasonably approve 0.90 and be equally right. PRISM's value is precisely in
acknowledging and governing that ambiguity, so the documentation must not
quietly assert an objective truth the system is built to not assume. The same
discipline applies to how eval results are reported: a "38% pass rate" means
38% converged with the gold answers' chosen conventions, not that 38% were
objectively right.

In both cases retrieval was clean and the correct raw facts were all present
and correctly identified. The divergent answers came entirely from an
ungoverned compute step selecting a plausible-but-unintended convention.

### 4.2 Entry schema (not yet implemented)

Two entry types, deliberately separate:

```json
{
  "id": "finance.quick_ratio",
  "entry_type": "metric",
  "scope": { "domain": "finance", "document_types": ["10-K", "10-Q"] },
  "recognition": { "canonical_name": "quick ratio", "aliases": ["acid-test ratio"] },
  "interpretation": {
    "formula": "(current_assets - inventory) / current_liabilities",
    "required_facts": ["current_assets", "inventory", "current_liabilities"]
  },
  "governance": { "status": "approved", "source": "proposed_from_use", "version": 1 }
}
```

```json
{
  "id": "finance.quick_ratio.healthy_threshold",
  "entry_type": "interpretation_policy",
  "applies_to": "finance.quick_ratio",
  "scope": { "domain": "finance" },
  "policy": { "healthy_at_or_above": 1.0 },
  "governance": { "status": "approved", "source": "proposed_from_use", "version": 1 }
}
```

**Metric definition and judgment policy are separate entries, versioned
independently.** Computing a quick ratio of 0.96 is arithmetic. Concluding
that 0.96 is *unhealthy* is a threshold judgment — 0.9 is comfortable for a
business with fast, predictable receivables and alarming for one without.
Conflating them would have PRISM silently import a generic textbook threshold
with exactly the false confidence this whole design exists to eliminate. They
version separately because one metric may carry different policies for
different organizations or risk appetites, and a change of risk appetite must
not force re-approval of the arithmetic.

**PRISM may compute a metric without holding authority to judge it.** Where no
approved `interpretation_policy` exists, the honest behavior is to report the
computed value with its provenance and decline the verdict — or surface a
candidate policy ("the conventional threshold is 1.0; no approved policy
exists for this context") into the same proposal/approval loop. This directly
affects most of the eval questions in scope, which ask for judgments
("is 3M capital-intensive?", "reasonably healthy liquidity?"), not numbers.

**`required_facts` is a list of plain names, not references.** They are
descriptive — they tell a reviewer what went into the number and give the
evidence planner a starting point next time. Nothing should build a resolver
for them.

**No recursive fact ontology.** An earlier draft leaned toward promoting each
`required_fact` into its own dictionary entry with its own retrieval hints.
The evidence planner makes that unnecessary: it generates transient row/table
requirements per question, which is where retrieval guidance actually comes
from. The dictionary locks only the *consequential semantic decision*. This is
ADR-0001's own argument applied one level down — if we refuse to pre-author
formulas because that demands domain expertise before the product works, we
cannot pre-author several hundred line-item definitions either (`current_assets`
looks atomic but isn't: the printed subtotal row, or a sum? does the label
survive IFRS vs. GAAP? what about filers who print no subtotal at all?).
Reusable fact definitions get introduced **only when repeated binding failures
demonstrate their value** — and that trigger is only observable because the
fact-binding stage records what the planner asked for and whether each binding
succeeded. Without that telemetry the promotion rule is decorative.

### 4.3 Governance workflow — see ADR-0001

**Decision, in full:
[`docs/adr/0001-dictionary-not-a-prerequisite.md`](../docs/adr/0001-dictionary-not-a-prerequisite.md).**

Summary: a dictionary is **not a prerequisite for useful operation**. When a
concept has no approved entry, the system generates one or more *plausible,
explicitly labeled* candidate interpretations supported by the retrieved
evidence — never claiming that set is exhaustive — and surfaces them rather
than silently picking one. A human ratifies the consequential ones, and the
approved entry is thereafter executed deterministically in code, never
re-delegated to an LLM at answer time. Optional pre-built knowledge packs
(global, vertical, or organization-level) are permitted and may accelerate a
known vertical; they simply must never be *required*.

Consequence worth restating: the `validation` stage checks whether candidates
agree on **the conclusion the question actually asks for**, not merely that
they all computed. Candidates that diverge on the qualitative conclusion must
block on review — averaging them or picking one is a wrong-answer risk, not a
shortcut. Where the question asks for a judgment, that check also requires an
approved interpretation policy to judge against (§4.2).

## 5. Retrieval architecture — three phases, evaluated

All phases scoped to 3M's 9 filings, 8 matching eval questions, scored
with an LLM judge (bare-numeric expected answers use deterministic comparison
instead — free, unambiguous, no reason to route those through a judge).

| Phase | Mechanism | Convergence |
|---|---|---|
| `1-vector` | Hyperscale Vector Index, `APPROX_VECTOR_DISTANCE`, no filter. Textbook RAG. | 25% (2/8) |
| `2-catalog` | regex resolves `doc_name` from question year/quarter (8/8 correct) against `catalog`, then vector search restricted to that document | 38% (3/8) |
| `3-hybrid` | document scope + BM25 over content anchors + kNN, all in **one** `SEARCH()` against the docs Search Vector Index, plus fact binding, deterministic calculation and dictionary governance | **62% (5/8)** |

Every configuration issues exactly **one** SQL++ statement. In `3-hybrid` the
three legs are parts of a single `SEARCH()`:

```
query.conjuncts[0]   document scope    exact filename match
query.conjuncts[1]   BM25 lexical      the planner's content anchors, as phrases
knn                  vector kNN        with its own filter, so it is scoped too
```

Two details there are load-bearing:

**BM25 matches the anchors, not the question.** An earlier version sent the raw
prompt — *"Does 3M have a reasonably healthy liquidity profile… If the quick
ratio is not relevant, please state that and explain why."* — which scores
mostly on common words and measurably bought nothing. The planner already
produces the verbatim printed row labels worth matching (`Total current
assets`, `Total current liabilities`), and they go in as `match_phrase` so a
multi-word label matches as a sequence rather than an OR over its tokens. BM25
supplies rarity weighting natively, via IDF, which is what the standalone
anchor path had to hand-roll as `1/log2(2+df)`.

**The scope predicate lives inside `SEARCH()`, not in `WHERE`.** A scalar filter
outside is applied by the Query service only *after* the Search service has
returned its k results, so a selective filter against a small k can discard
nearly everything. The filename sits in the conjuncts (scoping the lexical leg)
and in the knn `filter` object (scoping the vector leg).

**Catalog-based document scoping was the single highest-leverage change**
(25% → 38%) — bigger than hybrid search on its own. An intermediate
configuration measured during development (catalog scoping + BM25 over the
whole raw question, no anchors, no governance) scored 38%, exactly the same as
catalog scoping alone: BM25 over an entire question adds nothing when the
remaining failures are not ranking problems. What moved the number was content
anchors and governance, both folded into `3-hybrid`.

An earlier five-phase split gave content anchors and governance their own
steps. That was dropped because it left the *default* phase with `bm25=False` —
the architecture being demonstrated never actually exercised the FTS index or
emitted a `SEARCH()` query. Folding them in means the preferred configuration
is the one that shows `SEARCH()`.

`title_boost` is deliberately 0 in `3-hybrid`. Boosting
`meta-data.associated-titles` is tempting but that field is wrong on a
meaningful fraction of table chunks (§3), so weighting it promotes confidently
wrong chunks. BM25 runs on `text-to-embed` only; the unreliable titles are
sidestepped by content anchors instead.

## 6. Governance principles (apply across all three tiers)

- **Every field distinguishes extracted / inferred / pending.** Extracted =
  literally present, citable. Inferred = derived by a rule or model, carries
  confidence. Pending = a later stage owns it. Never conflate "not yet
  extracted" with "extracted as absent."
- **Grounding checks verify the claimed value, not the model's cited span.**
  A model can quote its source inaccurately while still being right about the
  value (confirmed: an 8-K's `period_end_date` was correctly read from "Date
  of Report" language but cited against unrelated boilerplate) — checking the
  span punishes accurate values for sloppy citation. Checking the normalized
  *value* against the actual source text catches real hallucinations (a ticker
  fabricated from training-data memory) without that false-positive cost.
- **The LLM's authority depends on governance state.** For an *unapproved*
  concept the LLM may propose formulas and compute labeled candidates — that
  is how the dictionary gets seeded. For an *approved* concept the LLM is
  categorically forbidden from computing; it binds facts and writes prose
  around a number produced by code. Same pipeline, two trust levels, switched
  by whether an approved entry exists. This is the mechanism that makes
  "learns as it goes" compatible with "doesn't drift."
- **Bindings are validated, not assumed.** Retrieval returning a table is not
  the same as knowing which cell means what. Entity, period, column, units,
  and provenance are each independently checkable and each independently
  wrong-able — picking the wrong column is not an arithmetic error and
  validating the arithmetic will never catch it.
- **Provenance is bound, not reconstructed.** Every bound fact records the
  chunk and page it came from at binding time. Given `associated-titles` is
  demonstrably unreliable (§3), "this came from the table on page 5" must be
  recorded and checkable, not inferred after the fact.
- **Refusing beats guessing.** A field, a formula, or a *judgment* that cannot
  be established with confidence should say so rather than present a plausible
  guess at full confidence.

## 7. Open issues

1. `meta-data.associated-titles` reliability (§3) — mitigated by content-anchor
   planning, not eliminated.
2. Complex-table serialization losing sectional structure (§3) — an ingestion
   representation problem, **unaddressed by this architecture**, and the reason
   the current 8-question set cannot be fully answered even with everything
   here built.
3. Does a `proposed` (not yet approved) dictionary entry get used
   provisionally, or is it inert until ratified? Bears on whether "human
   approval" is blocking or asynchronous.
4. The §1 runtime — evidence planner, fact binding, calculation, validation,
   proposal loop — is designed but not built.

## Appendix — key locations

- Eval scripts: `eval/run_benchmark.py`, `eval/debug_question.py`
- Catalog collection: `{bucket}.{scope}.catalog`; chunks: `{bucket}.{scope}.docs`
- FTS indexes: one docs index and one catalog index per domain scope (see
  `design/domains.yaml`) — **must be fully-qualified** in N1QL `SEARCH()`
  calls, the bare short name fails
- S3 object naming: `{AWS_BUCKET}_{AWS_FOLDER}_{doc_name}.pdf`
- ADRs: `docs/adr/`
