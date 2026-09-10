---
status: proposed
---

# Multi-document retrieval: map per document, merge, reduce unchanged

`docs/findings/multi-period-questions.md` measured the gap this ADR closes:
14 of 143 answerable questions (~10%) in the eval corpus name more than one
fiscal period — a year-over-year change, an N-year average, a quarter
compared against a prior annual figure. Resolution already returns the full
set of matching documents (`resolve_and_plan()`'s `documents` field); the
pipeline discards all but the most recent one before retrieval, so these
questions are answered from a single filing with no indication a second
period went missing. That finding recommended detecting and declining these
questions as a stopgap; this ADR is the real fix instead, built directly, per
this project's decision to skip the interim stopgap.

**Decision:** retrieve and bind facts **once per resolved document, scoped to
that document alone**, then **merge** the validated, grounded fact sets
before handing off to computation. Computation, conclusion validation, and
answer synthesis do not change at all — they already consume a flat
`{fact_name: value}` dict and have no notion of how many documents produced
it. The new work is entirely upstream of that dict's assembly.

Capped at **2 documents per question** for this build. Covers the plurality
of the measured cases (a two-filing comparison) and keeps the retrieval/LLM
cost multiplier bounded and easy to reason about. A genuine 3+ year average is
explicitly out of scope for this version — see Limitations.

## Why this is a map + a merge, not a rewrite

Tracing the existing single-document path:

- `retrieve()` already takes one `doc_name` and scopes both the anchor and
  hybrid-search legs to it — calling it twice, once per resolved document,
  needs no signature change.
- `bind_facts()` already tolerates a requested fact that isn't actually in
  the excerpts it's given ("if a requested fact is genuinely not present,
  omit it" is already in `BINDER_SYSTEM_PROMPT`) — calling it with the SAME
  full required-facts list against each document's own chunks is safe. A
  document that doesn't cover a given period simply returns nothing for that
  period's facts; there is no new prompt to write for "which facts belong to
  which document."
- `validate_bindings()` already runs its grounding check (does the bound
  value appear verbatim in the retrieved text?) and its mixed-period guard
  scoped to ONE document's own chunks. Running it once per document, exactly
  as today, and merging the validated outputs AFTER — not merging raw
  bindings before validation — preserves that check's strictness exactly.
  Nothing about it needs to change.
- `compute()`, `validate_conclusion()`, `synthesize()` operate on
  `grounded_facts(bound)`, a flat dict keyed by fact name. A dict assembled
  from two documents' validated facts is indistinguishable to these
  functions from one assembled from a single document's own comparative
  columns (`period_role`'s existing use case) — no change required.

So the pipeline becomes:

```
for each of the (≤2) resolved documents:
    chunks   = retrieve(question, plan, doc_name)          # unchanged
    raw      = bind_facts(question, required_facts, chunks) # unchanged, same full list
    bound    = validate_bindings(raw, chunks)                # unchanged, per-document
merged = union of grounded facts across all `bound` lists     # NEW, small
calc   = compute(entry, candidates, merged)                   # unchanged
...    = validate_conclusion(...) / synthesize(...)            # unchanged
```

## What's actually new

1. **The map loop itself** in `pipeline.py`, replacing the current
   "collapse `resolution_detail["documents"]` to the single most-recent one"
   step with a loop over up to 2, sorted the same way as today (most recent
   year, then quarter) so the cap keeps the two most relevant periods when
   more were resolved.
2. **The merge step.** Union each document's validated `bound` list. Facts
   are expected to be disjoint by construction — each document only grounds
   the facts it actually contains. A genuine collision (two documents both
   grounding the same `period_role`) is a real conflict, not something to
   silently resolve by picking one: surface it as a binding issue rather
   than picking a value silently, the same discipline `validate_bindings`
   already applies to a same-document mismatch.
3. **Trace/UI.** Every trace view in `streamlit_app.py` assumes one resolved
   document, one SQL statement, one evidence set. Showing N resolutions, N
   retrievals, and a merge step is a real, separate UI lift — tracked here,
   not blocking the pipeline change.

## What does NOT need to change

`retrieval.retrieve()`, `fact_binding.bind_facts()`, `fact_binding.
validate_bindings()`, `calculation.compute()`, `calculation.
propose_candidates()`, `validation.validate_conclusion()`,
`answer.synthesize()` — all called exactly as they are today, just more than
once for the first three. This is what keeps the change small: the map/merge
layer is new orchestration above existing, already-correct single-document
logic, not a redesign of it.

## Limitations of this version

- **Capped at 2 documents.** A 3-year average (one of the measured question
  shapes) is not answerable by this build — it needs a third retrieve/bind
  pass and a cap increase, deliberately deferred rather than built uncapped
  on day one.
- **Governed (approved dictionary) formulas don't get this yet.** An
  approved entry's `required_facts` is a flat list of bare identifiers with
  no `period_role` — per `fact_binding.py`'s own note, dictionary entries
  carry no period metadata today. Multi-document binding therefore lands
  first for the **ungoverned candidate-proposal path** (`candidate_facts()`
  already carries `period_role` per fact). Extending it to an approved,
  multi-period metric needs the dictionary schema itself to carry
  `period_role` per required fact — a separate, later change.
- **No new document-to-period assignment logic.** Each document is asked to
  bind the FULL required-facts list, not just the subset it's expected to
  own. Simpler to build and verify first; if binder cost or cross-document
  mis-binding turns out to matter in practice, a targeted "assign this
  period_role to this document before asking" refinement is the next lever,
  not a redesign.
- **Trace UI not part of this change.** The pipeline will produce a correct
  merged answer before the trace can show how it got there across two
  documents.

## Consequences

- **Cost and latency scale with the document count for any question that
  resolves to more than one.** A single-document question is unaffected;
  a two-document question roughly doubles retrieval + binder LLM calls.
- **A collision between two documents' bound facts is a hard validation
  failure, not a silent pick.** Consistent with the existing mixed-period
  guard's own discipline — a wrong answer from silently averaging or
  arbitrarily choosing is worse than a declined one.
- **This does not change what a GOVERNED metric can compute yet.** A
  question naming an approved multi-period metric still needs the dictionary
  schema extension described above before it benefits from this ADR.
