"""Answer synthesis — prose around numbers that already exist.

Two prompts, because the model's job differs. With nothing computed it is
ordinary grounded extraction. With values computed it must present them and is
explicitly barred from denying them: the model twice refused a question whose
answer it had been handed, once claiming the excerpts lacked the data to
compute a quick ratio while holding the computed 0.9578.
"""
from .. import llm
from .fact_binding import format_chunks

PLAIN_PROSE_RULE = (
    "Do not use markdown bold, italics, or other emphasis markup. "
    "Models tend to wrap numbers in **bold** with no surrounding space, which some "
    "renderers show as words running together with no space (e.g. \"3,404**million**\" "
    "- verified live, this is a real rendering and copy-paste corruption, not a "
    "cosmetic nitpick) - plain text has no such failure mode. This is about emphasis "
    "markup specifically, not about banning lists - an earlier \"write in plain prose\" "
    "framing read as the latter too, which is a live source of enumerated answers "
    "getting buried in a sentence instead of listed (see STRUCTURE_RULE)."
)

STRUCTURE_RULE = (
    "When the question asks for an enumeration and the source presents it as table "
    "rows, use an actual markdown list - one line per item, each starting with "
    "\"- \" - instead of burying them in a sentence; otherwise answer in ordinary "
    "prose. A bare newline between items is not enough: the renderer collapses a "
    "single line break into a space (verified live - a plain one-per-line answer "
    "with no \"- \" marker rendered back as one run-on sentence), so the leading "
    "\"- \" is what actually produces separate lines, not a stylistic choice. Three "
    "things matter beyond appearance:\n"
    "- Use the source's own printed row labels verbatim (\"Nontaxable or nondeductible "
    "items\", not \"non-deductible expenses\") - this is what makes an answer "
    "auditable line-by-line against the filing.\n"
    "- Keep every column the source shows for those rows (e.g. both the percent and "
    "the dollar amount) rather than silently picking one.\n"
    "- Never drop rows to shorten a list. If the question's own wording invites a "
    "subset (\"largest\", \"main\", \"primary\", \"key\"), say explicitly that the list "
    "is a subset and what criterion was applied - verified live, asking for the "
    "\"largest\" reconciling items in 3M's effective-tax-rate table silently dropped "
    "the smallest ones below an unstated cutoff, with nothing in the answer disclosing "
    "that a subset was applied or what decided it; the listed items then no longer "
    "summed to the stated rate, which is invisible to a casual reader. Asking for the "
    "items with no qualifier returned all nine rows correctly."
)

CURRENCY_RULE = (
    "Every dollar figure gets its $ sign, every time it is written - \"$4.9 billion\", "
    "never \"4.9 billion\" alone, whether the source or the computed block wrote it with "
    "one or not. Do not drop the sign on a repeated mention of a figure already stated "
    "once with it."
)

BASE_ANSWER_PROMPT = (
    "Answer the question using ONLY the provided excerpts. If the answer is not in "
    "them, say so clearly. Cite sources by their [number].\n\n"
    "Sources marked [TABLE] contain structured data and are usually the authoritative "
    "source for a specific quantitative answer. When a question asks for a value, a "
    "rate, or what drove a change, check [TABLE] sources first and prefer their figures "
    "over a narrative source's paraphrase.\n\n"
    f"{PLAIN_PROSE_RULE}\n\n{CURRENCY_RULE}\n\n{STRUCTURE_RULE}"
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
    "the verdict was DECLINED for lack of an approved interpretation policy: this matters "
    "ONLY when the question itself asks for a characterization (it uses words like "
    "healthy, stable, high, low, good, adequate, or similar, or otherwise asks you to "
    "judge the value). In that case, report the value and say plainly that characterising "
    "it against a threshold requires an approved policy that does not exist yet - and do "
    "not supply your own threshold. That restriction is about the computed value only, not "
    "about the excerpts: if an excerpt directly and explicitly states its own qualitative "
    "characterization or historical pattern (for example, a filing stating it has raised a "
    "dividend for a stated number of consecutive years), report that as a grounded, cited "
    "fact and let it answer the question's qualitative framing - that is the source's own "
    "characterization, not one you invented.\n\n"
    "If the question is PURELY quantitative (it does not ask for a characterization at "
    "all - e.g. \"what was the ratio\"), a DECLINED verdict is not something to mention: "
    "just report the value and how it was derived. Do not go looking through the excerpts "
    "for a qualitative statement to volunteer when nothing qualitative was asked - that is "
    "answering a question nobody asked.\n\n"
    "If it says the verdict is BLOCKED, report the candidates and say the conventions "
    "disagree.\n\n"
    "Answer in ONE short paragraph: state the answer itself, pithily, with citations. "
    "Do not also restate the formula or walk through the derivation in prose - the "
    "computed block above already gives the exact formula and figures used, and the UI "
    "shows it in its own separate section verbatim. A second paragraph re-deriving it in "
    "your own words only risks drifting out of sync with the one that's actually "
    "authoritative.\n\n"
    f"{PLAIN_PROSE_RULE}\n\n{CURRENCY_RULE}"
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
                     "concept, so THESE COMPUTED VALUES may not be characterised against a "
                     "threshold you invent. This does not restrict the excerpts themselves - "
                     "if they directly state their own qualitative characterization, report "
                     "it and let it answer the question.")
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
