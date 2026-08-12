"""The runtime: bind -> calculate -> validate -> answer (design/architecture.md §1).

    fact_binding  retrieved text -> named facts with entity/period/column/provenance
    calculation   bound facts    -> governed value, or labeled candidates
    validation    values         -> is a conclusion authorised?
    answer        everything     -> prose around numbers already computed
    pipeline      answer_question(), the single entry point for eval + demo

The LLM's authority changes with governance state, and that switch is what
makes "learns as it goes" compatible with "doesn't drift":
    unapproved concept -> may propose formulas and compute labeled candidates
    approved concept   -> forbidden from computing; binds facts and writes prose
"""
from .answer import (  # noqa: F401
    BASE_ANSWER_PROMPT, COMPUTED_ANSWER_PROMPT, render_computed_block, synthesize,
)
from .calculation import compute, propose_candidates  # noqa: F401
from .fact_binding import (  # noqa: F401
    bind_facts, format_chunks, grounded_facts, to_identifier, validate_bindings,
)
from .pipeline import PipelineOptions, answer_question  # noqa: F401
from .validation import validate_conclusion  # noqa: F401
