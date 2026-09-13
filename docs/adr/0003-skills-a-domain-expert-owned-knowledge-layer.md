---
status: proposed
---

# Skills: a domain-expert-owned knowledge layer, separate from prompt rules

`prism/runtime/resolve_and_plan.py`'s extraction prompt accumulated several
SEC-filing-specific facts over one session of live debugging: a proxy
statement is dated a year after the compensation it discloses; a 10-Q cannot
carry a full fiscal year's figures; non-GAAP quarterly metrics often surface
in an 8-K before a 10-Q. Each fix was individually correct and empirically
verified against the gold corpus. Stacked into one prompt, they read as the
system being patched until an eval passed — the same "gaming the system"
concern that motivated rewriting that prompt in the first place, recurring
one layer down. The rules themselves were not the problem; where they lived
was.

**Decision:** four kinds of knowledge feed `resolve_and_plan`, each with a
distinct owner, and none of the four should be authored by someone standing
in for one of the others:

| Artifact | Owner | Expertise needed | Change cadence | Blast radius |
|---|---|---|---|---|
| Prompt rules | App developer | The whole downstream contract (`fact_binding`, `calculation`, `retrieval`) | Rare — only when the extraction *mechanism* changes | Every company, every industry, every question |
| Skills | Domain expert | Filing mechanics for one industry; no code | Occasional — as filing quirks surface | Every company in one industry |
| Dictionary | End user | Trusts a specific formula for a specific metric | Frequent, incremental | One metric |
| Concepts | End user | Their own company's/vertical's jargon | Frequent, incremental | One term |

`Dictionary` and `Concepts` already work this way — `dictionary.matching`'s
own docstring states an entry only exists via `compile` or `approval`, "both
human-reviewed acts... presence in the collection IS the approval." `Skills`
fills the gap between "fully domain-free code" (the prompt) and "fully
user-specific data" (dictionary, concepts): general to an industry, but not
a mechanical JSON-shape rule and not one company's own vocabulary.

## Why the authority, not the content, was the actual problem

A proxy statement's filing lag is true regardless of who states it. What made
stating it inside `resolve_and_plan.py`'s rules feel like overfitting is that
an app developer has no standing to assert SEC filing mechanics — that
knowledge belongs to someone else, and its presence in "compiler" code that's
supposed to look domain-free is a category error, not a factual one. Move the
identical sentence into an artifact a domain expert owns, reviewed the way a
domain expert would review it, and it stops looking like a patch: it becomes
what it actually is, an SME's contribution to a knowledge base. The content
doesn't change. The authority under which it enters the system does.

This also explains why the pattern kept recurring rather than being fixed
once: no Skills layer existed, so every new filing quirk had nowhere
legitimate to go but the prompt.

## What this is not

**Not a caching optimization.** A separated Skills block does not newly
unlock prompt caching — the system prompt sent to the model is already one
static string per call regardless of how the source text is assembled before
that, so it is already cacheable today. The benefit is smaller diffs per
discovery, review by someone who understands the domain and not necessarily
the code, and an honest, explicitly-growing home for "we keep learning new
filing quirks" — not a technical capability that was previously missing.

**Not a lower bar for review.** A wrong skill is exactly as dangerous as a
wrong prompt rule — it still silently breaks every company in an industry,
not one metric. Domain-expert authorship changes who reviews a skill and how
(a domain checklist, not a code diff), never whether it gets re-run against
the eval suite before being trusted. "A domain expert wrote it" is not a
substitute for measuring it.

**Not always a clean sort.** Two categories are close enough to misclassify
in practice, and it is worth a fixed test rather than re-litigating per
discovery:

> Is this about JSON shape or output format, independent of any domain
> knowledge? → prompt rule.
> Does it hold for every company in the industry, not one? → skill.
> Is it this one company's or user's own vocabulary? → concept.
> Is it a formula a user is vouching for as correct? → dictionary.

The riskiest miscategorization is a skill phrased as a *preference* rather
than a *fact*. "Non-GAAP metrics originate in an 8-K before a 10-Q," phrased
as a reason to prefer or narrow to the 8-K, reproduces the exact bug
`docs/findings/earnings-8k-not-catalogued-by-period.md` documents — guessing
a document form from a metric's nature, and excluding the correct one. The
same fact phrased as "both structurally carry this; the 8-K usually appears
first" is safe: informative, not exclusionary. The distinction that matters
is that phrasing, not which file the sentence lives in.

## Consequences

- **`resolve_and_plan.py`'s rules become, and stay, purely structural** — JSON
  validity, field shapes, one-fact-per-printed-value granularity, formula
  identifier discipline. A rule that references how a *specific kind* of
  filing behaves belongs in Skills, not here, from this point forward.
- **A newly discovered filing quirk is a Skills edit, not a prompt edit.**
  Smaller diffs, and a reviewer who needs to know SEC filing practice, not
  `fact_binding`'s binding contract.
- **Skills is scoped like `concepts` and `dictionary`** — one domain's
  mechanics do not leak into another domain's questions. A non-finance
  deployment of this architecture starts with an empty Skills document, the
  same way it starts with an empty dictionary (ADR-0001).
- **Storage and review tooling are deliberately not decided here.** The first
  build (a separate branch) uses a local, per-scope, plain-text file — one
  skill statement per line, appended into the same system prompt at the same
  LLM call, no new round-trip. Whether Skills eventually needs a governed
  collection and CRUD surface like `concepts` already has is a question for
  after that first build is used, not before.
- **This does not change what Dictionary or Concepts already do.** Skills is
  additive: a third input to `resolve_and_plan`, not a replacement for
  either existing mechanism.
