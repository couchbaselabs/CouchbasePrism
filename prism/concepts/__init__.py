"""Concepts - "tribal knowledge": which entities/topics a corpus's own
documents refer to, and the term a person uses for one versus the term the
filing itself uses. Distinct from the dictionary, which governs formulas and
interpretation policy, not meaning; distinct from skills, which hold general
filing-mechanics knowledge rather than one corpus's own vocabulary - see
docs/adr/0003-skills-a-domain-expert-owned-knowledge-layer.md.

Two fields per entry, deliberately: `user_term` (what a person might type in
a question) and `filing_terms` (what this corpus's own documents actually
call it - verified, not guessed; see verification.py). An earlier version
carried `aliases`, `official_filing_terms`, `subsidiaries_involved`, and
`target_sections` - a distinction nobody maintained (both of the first two
fed the exact same expansion in resolve_and_plan.py, so the split was never
functional), a field nothing downstream ever read, and a field whose generic
section-header phrases diluted forced-phrase matching rather than helping
it. Collapsed to the two fields that do real, distinct work.

Consumed by resolve_and_plan.py: a question naming a concept's `user_term`
pulls its `filing_terms` into the same forced-phrase mechanism `#hash#`
spans use.
"""
from .repository import load, save  # noqa: F401
from .verification import verify_filing_term  # noqa: F401
