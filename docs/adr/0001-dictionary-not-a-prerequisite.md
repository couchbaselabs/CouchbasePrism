---
status: accepted
---

# A domain dictionary is not a prerequisite for useful operation

PRISM must answer questions requiring a *derived* value that no source document
states verbatim — quick ratio, return on assets, CAPEX/revenue. The obvious
approach is a domain dictionary: recognized concepts with locked formulas,
authored before the product is used in a vertical. We rejected that as a
*requirement*, not as a possibility.

**Decision:** PRISM operates usefully with an empty dictionary. When a concept
has no approved entry, the evidence planner determines what facts are needed
and retrieval obtains them — neither step consults the dictionary. The system
then generates **one or more plausible, explicitly labeled candidate
interpretations supported by the retrieved evidence**, and surfaces them rather
than silently selecting one. A human ratifies the consequential ones; the
approved entry is written back as scoped and versioned, and thereafter executed
**deterministically in code** — never re-delegated to an LLM at answer time,
even with correct inputs in hand.

The candidate set is explicitly **not claimed to be exhaustive**. No system can
establish that it enumerated every defensible convention, and the mechanism
doesn't need it to: two candidates disagreeing is already sufficient to detect
ambiguity and block on review.

Optional pre-built knowledge packs — global, vertical, or organization-level —
are **permitted** and may accelerate a known vertical. They must never be
required for the product to function.

**Why:** letting an LLM freelance the interpretation, even given exactly the
right raw facts, produces confidently divergent answers — because common
financial ratios have multiple genuinely defensible conventions. Confirmed
directly, twice, on real 3M filings:

- **Quick ratio** (`financebench_id_00807`, 3M 2023 Q2 10-Q): FinanceBench
  expects `(current_assets − inventory) / current_liabilities = 0.96`. An
  ungoverned compute step produced `0.85` (cash + marketable securities +
  receivables, over liabilities) and `0.90` (current assets less inventory
  *and* prepaids) instead.
- **Return on assets** (`financebench_id_00499`, 3M 2022 10-K):
  `net income attributable to 3M / total assets` = 12.44%, matching the
  expected 12.4%; `net income including noncontrolling interest / total assets`
  = 12.47%, rounding to 12.5%, which would not match.

Every one of those alternatives is defensible finance. They are not errors and
0.96 is not objectively "correct" — it is the convention *this benchmark*
expects. That is the entire point: the value PRISM adds is acknowledging and
governing the ambiguity, not resolving it by fiat.

A pre-built dictionary gives more consistent answers on day one but demands
deep domain expertise in every vertical before the product is useful there,
which does not scale past one or two industries. Proposing from real evidence
and capturing a reviewed decision is slower on a fresh domain's first pass, but
the product works immediately in a domain nobody on the team understands and
sharpens from actual use rather than speculative authoring.

## Consequences

- **The dictionary is model-proposed and human-approved, not hand-curated.**
  Nobody authors formulas from scratch. The system does the work; the human
  makes a judgment call, and only when it is consequential.
- **Evidence planning must not depend on the dictionary.** The planner
  identifies the concept and plans required facts, tables, and content anchors
  with no entry in existence — demonstrated on quick ratio. A design that gates
  retrieval behind a dictionary lookup breaks the central claim of this ADR.
- **Metric definitions and interpretation policies are separate entries.**
  `quick_ratio = (current_assets − inventory) / current_liabilities` is
  arithmetic. `healthy_at_or_above = 1.0` is a threshold judgment, dependent on
  industry and risk appetite. PRISM may compute a metric without holding the
  authority to characterize it; absent an approved policy, it reports the value
  and declines the verdict, or proposes a candidate policy into the same loop.
- **Validation checks conclusion-agreement, not just computation.** Candidates
  that agree on the qualitative conclusion the question asks for can support an
  answer before approval; candidates that diverge on it must block. Averaging
  or arbitrary selection is a wrong-answer risk.
- **The surfacing-and-approval flow is product surface, not a debugging tool.**
  Someone must actually see and resolve the ambiguity for the dictionary to
  grow.
- **A locked formula is scoped**, and a document that breaks its scoping
  assumption (GAAP vs. IFRS, an unusually structured balance sheet) should
  re-trigger ambiguity rather than silently inherit the wrong entry.
- **This ADR assumes retrieval reliably finds the right facts once a concept is
  identified. It does not yet.** `meta-data.associated-titles` was wrong on 3
  of 3 checked cases (`financebench_id_01226`, `financebench_id_01865`,
  `financebench_id_00941`). Content-anchor planning mitigates this; it is
  tracked separately as an open issue in `design/architecture.md` §7.
