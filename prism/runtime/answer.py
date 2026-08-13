"""Answer synthesis — prose around numbers that already exist.

Two prompts, because the model's job differs. With nothing computed it is
ordinary grounded extraction. With values computed it must present them and is
explicitly barred from denying them: the model twice refused a question whose
answer it had been handed, once claiming the excerpts lacked the data to
compute a quick ratio while holding the computed 0.9578.
"""
from .. import llm
from .fact_binding import format_chunks

BASE_ANSWER_PROMPT = (
    "Answer the question using ONLY the provided excerpts. If the answer is not in "
    "them, say so clearly. Cite sources by their [number].\n\n"
    "Sources marked [TABLE] contain structured data and are usually the authoritative "
    "source for a specific quantitative answer. When a question asks for a value, a "
    "rate, or what drove a change, check [TABLE] sources first and prefer their figures "
    "over a narrative source's paraphrase."
)

COMPUTED_ANSWER_PROMPT = (
    "You are given source excerpts AND a block of values that have already been "
    "computed deterministically from figures bound to specific rows and columns of "
    "those excerpts.\n\n"
    "THE COMPUTED VALUES ARE THE ANSWER to the quantitative part of the question. Lead "
    "with them. Use them exactly as given - do NOT recompute, re-derive or adjust them, "
    "and do not substitute your own arithmetic.\n\n"
    "Never claim the excerpts lack the data needed to compute these values: the binding "
    "step already located every figure used, so such a statement is false. If the "
    "question invites you to say a metric is 'not relevant' or cannot be determined, do "
    "not take that option when a value has been computed - report the value.\n\n"
    "If the computed block states a VERDICT, state that conclusion directly. If it says "
    "the verdict was DECLINED for lack of an approved interpretation policy, report the "
    "value and say plainly that characterising it (healthy/unhealthy, high/low) requires "
    "an approved threshold that does not exist yet - and do not supply your own "
    "threshold. If it says the verdict is BLOCKED, report the candidates and say the "
    "conventions disagree.\n\n"
    "Cite excerpts by [number] where they support the narrative."
)


def render_computed_block(calc: dict, conclusion: dict) -> str:
    lines = [f"- {c['label']}: {c['formula']} = {c['value']:.4g}"
             for c in calc["computed"]]
    lines.append("These come from an APPROVED dictionary entry and are authoritative."
                 if calc["governed"] else
                 "No approved entry exists; these are candidate interpretations, not "
                 "claimed exhaustive.")
    status = conclusion.get("status")
    if status == "agreed":
        lines.append(f"VERDICT (approved policy): "
                     f"{'yes/healthy' if conclusion['verdict'] else 'no/not healthy'} - "
                     f"{conclusion['explanation']}")
    elif status == "no_policy":
        lines.append("VERDICT DECLINED: no approved interpretation policy exists for this "
                     "concept, so no characterisation is authorised.")
    elif status == "conflicting":
        lines.append("VERDICT BLOCKED: candidate conventions disagree on which side of "
                     "the approved threshold this falls.")
    return "\n".join(lines)


def synthesize(question: str, chunks: list, calc: dict, conclusion: dict,
               model: str = None) -> str:
    """Only take the computed-values path when something was actually computed.
    Wrapping a stated-fact question in a "COMPUTED VALUES: (nothing)" frame made
    the model refuse a question it had previously answered correctly from the
    same chunks."""
    context = format_chunks(chunks)
    if calc["computed"]:
        prompt = (f"{COMPUTED_ANSWER_PROMPT}\n\nCOMPUTED VALUES:\n"
                  f"{render_computed_block(calc, conclusion)}\n\n"
                  f"EXCERPTS:\n{context}\n\nQUESTION: {question}\n\nANSWER:")
    else:
        prompt = (f"{BASE_ANSWER_PROMPT}\n\nEXCERPTS:\n{context}\n\n"
                  f"QUESTION: {question}\n\nANSWER:")
    return llm.chat_text(prompt, model=model, stage="answer")
