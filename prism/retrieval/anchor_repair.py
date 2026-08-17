"""Repairing dead anchors by looking at the source before planning again.

The planner guesses labels without seeing the corpus, and 62% of those guesses
match nothing. The guess cannot be improved by better prompting, because the
missing information is not in the model: it is in the document. Telling the
planner the genre does not help either - that was measured, and dead anchors
went from 62% to 66%.

So the fix is to look. A scoped retrieval pass returns text the document
actually contains, the model is asked to replace its failed guesses with labels
copied verbatim from that text, and the replacements are probed the same way
the originals were. Only survivors are used.

This is "look before you plan" in its cheapest form: one extra search and one
extra model call, and only when something actually died. A question whose
anchors all matched pays nothing.
"""
import json

from .. import llm
from .anchor_probe import filter_anchors

REPAIR_SYSTEM_PROMPT = """\
You proposed labels expected to appear verbatim in a source document. Some of
them do not appear anywhere in it.

You are now shown excerpts taken from that document. Propose replacement
labels, copied exactly from the excerpts, that identify the same quantities the
failed labels were meant to identify.

Return ONLY one valid JSON object:

{
  "anchors": ["<label copied verbatim from the excerpts>"]
}

GUIDELINES:

- Copy each label character for character from the excerpts. Do not paraphrase,
  reformat, correct, expand, or abbreviate.
- A label must be a printed row, column, field, or parameter label, not a
  sentence, a value, or a description.
- Do not include the subject, date, period, or other question-specific context
  unless it is genuinely part of the printed label.
- Prefer labels that identify a specific quantity over generic headings.
- If the excerpts contain no suitable replacement for a failed label, omit it.
  Returning fewer labels is correct; inventing one is not.
- Return an empty array if nothing suitable appears.
- Return strictly valid JSON without markdown or explanatory text.
"""


def _format_excerpts(chunks: list, limit: int = 6, chars: int = 1200) -> str:
    return "\n\n---\n\n".join(
        f"[{i + 1}] {(chunk.get('text') or '')[:chars]}"
        for i, chunk in enumerate(chunks[:limit]))


def repair_anchors(question: str, dead: list, chunks: list, doc_name: str = None,
                   model: str = None) -> tuple:
    """Returns (repaired_live, proposed). `proposed` is kept for the trace so a
    repair that was itself dead is visible rather than silently discarded."""
    if not dead or not chunks:
        return [], []
    user = json.dumps({
        "query": question,
        "labels_that_do_not_appear": dead,
    }, indent=1) + "\n\nEXCERPTS FROM THE DOCUMENT:\n" + _format_excerpts(chunks)

    try:
        proposed = llm.chat_json(REPAIR_SYSTEM_PROMPT, user, model=model,
                                 stage="anchor_repair").get("anchors", [])
    except Exception:
        # A failed repair must not fail the question; the live anchors and the
        # vector leg still carry the retrieval.
        return [], []

    proposed = [a for a in proposed if isinstance(a, str) and a.strip()]
    if not proposed:
        return [], []
    live, _dead, _counts = filter_anchors(proposed, doc_name)
    return live, proposed
