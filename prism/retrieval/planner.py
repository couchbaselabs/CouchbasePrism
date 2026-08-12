"""The evidence planner.

Runs BEFORE any dictionary lookup and needs no dictionary entry to work. That
ordering is the claim ADR-0001 rests on, and it is why retrieval can succeed
for concepts the system has never seen: asked about a quick ratio with an empty
dictionary, the planner still produces a complete balance-sheet evidence plan.

Its most valuable output is `content_anchors` - verbatim printed row labels.
Those sidestep `meta-data.associated-titles`, which is wrong on a meaningful
fraction of table chunks in real corpora.
"""
import ast

from .. import llm

PLANNER_SYSTEM_PROMPT = (
    "You are planning what evidence is needed to answer a question about a specific SEC "
    "filing. You do NOT have the document. Plan from general knowledge of how SEC filings "
    "are structured.\n\n"
    "Respond with exactly one JSON object:\n"
    '{\n'
    '  "concept": "the financial concept being asked about, e.g. quick ratio",\n'
    '  "answer_kind": "stated_fact" | "derived_metric" | "judgment",\n'
    '  "required_facts": ["the specific figures needed to answer"],\n'
    '  "content_anchors": ["verbatim row or column labels"],\n'
    '  "preferred_artifacts": ["table" and/or "text"]\n'
    "}\n\n"
    "content_anchors is the important field and the easiest to get wrong. These must be "
    "labels that would appear VERBATIM as a row or column heading inside the relevant "
    "table of a real SEC filing - the literal printed text, not a paraphrase of the "
    'question and not a section title. Good: "Total current liabilities", "Organic '
    'sales", "Trading Symbol(s)", "Purchases of property, plant and equipment". Bad: '
    '"liquidity information", "the balance sheet" (not printed row labels).\n\n'
    "DO NOT PAD THE LIST. Emit only anchors you are genuinely confident are printed "
    "verbatim in this kind of filing. Two precise anchors are far better than six "
    'guesses: a plausible-sounding invention ("Maturity Date", "Description") will match '
    "unrelated tables. Prefer long, distinctive, complete phrases over short generic "
    'words - "Name of each exchange on which registered" is excellent, "Description" is '
    "useless. If you are confident about only one anchor, return only that one.\n\n"
    'answer_kind: "stated_fact" if the answer is printed in the document as-is; '
    '"derived_metric" if it must be computed from other figures; "judgment" if it asks '
    "whether something is good/healthy/stable, which requires a threshold opinion beyond "
    "the arithmetic."
)


def plan_evidence(question: str, model: str = None) -> dict:
    return llm.chat_json(PLANNER_SYSTEM_PROMPT, question, model=model, timeout=60)


def formula_identifiers(formula: str) -> set:
    """Which bound facts a candidate formula references - used to widen the
    binding set beyond what the planner thought to ask for."""
    try:
        return {n.id for n in ast.walk(ast.parse(formula, mode="eval"))
                if isinstance(n, ast.Name)}
    except SyntaxError:
        return set()
