"""Scoring.

Two paths, because they answer different questions:
  - A bare-numeric gold answer is compared deterministically. Free,
    unambiguous, and there is no reason to route it through a model. Rare
    under the current gold schema (answer is markdown prose/tables, not a
    bare number), but still checked first when it applies.
  - Everything else goes to an LLM judge, because manual eyeballing does not
    scale to a full corpus run.

A note on language: "pass" here means CONVERGED WITH THE CORPUS'S CHOSEN
CONVENTION, not "objectively correct". Several quick-ratio and ROA conventions
are defensible finance; the gold answer picks one particular convention.
Reporting a score as "62% correct" would assert exactly the objective truth
PRISM is built not to assume.

A gold question and a PRISM result are each reduced to the SAME three
sections - answer, formula, evidence - before judging, via gold_sections()/
prism_sections() below, so the judge always compares like to like regardless
of which of the two shapes (a yaml file, a pipeline result dict) they
started as. Only answer and formula are actually SCORED, though - evidence
is shown to the judge and to a human (side by side, in the UI) purely as
information. A 10-K prints its own key figures on several different pages
by design, so a citation mismatch is not evidence of anything wrong;
gating PASS on evidence overlap failed a fully correct answer that had
simply cited a different, equally valid location for the same figure
(3m-002, measured 2026-09-11) - never let it decide the verdict again.
"""
import re

from prism import llm

JUDGE_SYSTEM_PROMPT = (
    "You are grading whether a candidate answer to a financial question is substantively "
    "correct against a reference answer. Both are given as three sections - ANSWER, "
    "FORMULA, EVIDENCE - and scoring is NOT symmetric across them:\n\n"
    "ANSWER must substantively match - same key facts, figures and conclusion. The "
    "candidate is expected to be more verbose and phrased differently; that alone is not "
    "a failure. When the reference names SEVERAL contributing causes (a 'what drove this "
    "change' question), a 10-K's MD&A typically discusses more drivers than the reference "
    "happens to enumerate. Use common sense: pass the candidate if it reaches the same "
    "overall conclusion and correctly cites a majority of the reference's materially "
    "significant drivers, even if it phrases them differently, omits a minor one, or adds "
    "other well-sourced drivers the reference does not mention. Do not fail an answer "
    "solely for not naming every cause the reference lists, or for including extra "
    "correctly-cited context - fail it only when it reaches a different conclusion, "
    "misstates a key figure, or is missing enough of the material drivers that the "
    "explanation itself is thin.\n\n"
    "FORMULA must match ONLY when the reference has one - a direct-extraction question's "
    "reference has no formula, and that is not something to penalize. When the reference "
    "does have one, AT LEAST ONE of the candidate's formulas must compute the same thing, "
    "though it may name its identifiers differently (e.g. 'total_current_assets' vs "
    "'current_assets' is fine if both mean the same printed quantity) - judge computational "
    "equivalence, not string identity. The FORMULA section may legitimately list MORE THAN "
    "ONE candidate when no approved convention governs this concept yet - proposing several "
    "defensible interpretations instead of silently picking one is this system's whole "
    "design (see the multi-line format: each line is a separately labeled candidate). A "
    "second, differently-computed candidate that does NOT match the reference is not a "
    "failure by itself - it is expected, and the ANSWER section already reports the "
    "matching one as the stated result. Only fail the FORMULA section if NONE of the listed "
    "candidates compute the reference's value, or if the reference has a formula and the "
    "candidate has none at all.\n\n"
    "EVIDENCE decides NOTHING - it is informational only, never a reason to pass or fail. "
    "A 10-K repeats its key figures by design: the same net sales figure commonly prints on "
    "half a dozen different pages (a statement, an MD&A table, a segment note, a non-GAAP "
    "reconciliation...), and any one of them is a legitimate citation. Measured case: the "
    "reference cited the Consolidated Statement of Income; the candidate answered correctly "
    "from the MD&A instead, citing entirely different pages with zero overlap - both were "
    "right, because 3M prints that same net sales figure on six different pages of the same "
    "filing. Do not read anything into a citation mismatch, a citation overlap, or a missing "
    "citation - judge ANSWER and FORMULA alone and mention evidence in your comment only as "
    "an observation, never as a reason.\n\n"
    'Respond with exactly one JSON object: {"pass": true or false, "comment": "one '
    'sentence explaining the verdict, naming which section(s) drove it"}'
)


def gold_sections(question: dict) -> dict:
    """A gold question dict (eval/corpora's shared shape) reduced to the
    three sections shown to the judge - answer/formula exactly as authored,
    no reshaping needed since the gold schema already carries them in this
    shape. evidence rides along for the side-by-side display and the
    judge's own comment, but is advisory only - see JUDGE_SYSTEM_PROMPT,
    which explicitly tells the model not to let it affect pass/fail."""
    return {
        "answer": question.get("answer") or "",
        "formula": question.get("formula"),
        "evidence": list(question.get("evidence") or []),
    }


def prism_sections(result: dict) -> dict:
    """A pipeline result (prism.runtime.answer_question's return) reduced to
    the same three sections. formula/evidence do not exist as such on the
    result - they are assembled from calc["computed"] and result["chunks"],
    the same deterministic data the trace UI already displays, not a second
    LLM call re-describing them. formula is None (matching a gold question
    with no formula) when nothing was computed - an extraction-only answer
    has nothing to show there either.

    Evidence lines are numbered [1], [2], ... in the SAME order
    fact_binding.format_chunks() numbers them for the synthesis prompt -
    the answer's own [5]-style citations refer to that position, so the
    evidence list has to carry the same numbers or a citation has nothing
    to point at."""
    calc = result.get("calculation") or {}
    computed = calc.get("computed") or []
    formula = ("\n".join(f"{c['label']}: {c['formula']} = {c['value']:.4g}"
                         for c in computed) or None)
    doc = result.get("resolved_doc") or "?"
    evidence = []
    for i, c in enumerate(result.get("chunks") or [], 1):
        parts = [p for p in (
            f"Page {c['page']}" if c.get("page") is not None else None,
            " / ".join(c.get("titles") or []) or None,
            c.get("type")) if p]
        evidence.append(f"[{i}] {doc}: " + " · ".join(parts))
    return {"answer": result.get("answer") or "", "formula": formula, "evidence": evidence}


def strip_markdown(text: str) -> str:
    """Enough to keep markup out of what the judge reads as prose - not a
    full CommonMark parser, just free of characters an LLM comparison has no
    reason to see: emphasis markers, headers, links, inline code, and table
    pipes/divider rows (the gold answer is markdown and commonly a table -
    see _TEMPLATE.yaml)."""
    if not text:
        return text
    t = text
    t = re.sub(r"^\s*\|?[\s:|-]+\|?\s*$", "", t, flags=re.MULTILINE)  # |---|---:| rows
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"__(.+?)__", r"\1", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", t)
    t = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"^\s{0,3}#{1,6}\s+", "", t, flags=re.MULTILINE)
    t = t.replace("|", " ")
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def parse_expected_number(expected: str):
    """Only a whole-string bare number counts. Most gold answers are
    markdown prose and contain incidental digits; "contains a number
    somewhere" would mis-route them into deterministic scoring."""
    s = (expected or "").strip()
    if not re.fullmatch(r"-?\$?\d[\d,]*\.?\d*%?", s):
        return None
    try:
        return float(s.replace("$", "").replace(",", "").replace("%", ""))
    except ValueError:
        return None


def _numbers_in(text: str) -> list:
    return [float(m) for m in re.findall(r"-?\d+\.?\d*", re.sub(r"[,$]", "", text or ""))]


def numeric_match(answer: str, expected: str):
    target = parse_expected_number(expected)
    if target is None:
        return None
    found = _numbers_in(answer)
    if not found:
        return False
    return any(abs(n - target) <= max(1.0, abs(target) * 0.02) for n in found)


def llm_judge(question: str, gold: dict, prism: dict, model: str = None) -> dict:
    def render(sections: dict) -> str:
        lines = [f"ANSWER: {strip_markdown(sections['answer'])}"]
        formula = sections.get("formula")
        lines.append(f"FORMULA: {strip_markdown(formula) if formula else '(none)'}")
        evidence = sections.get("evidence") or []
        lines.append("EVIDENCE: " + ("; ".join(evidence) if evidence else "(none)"))
        return "\n".join(lines)

    user = (f"QUESTION: {question}\n\n"
            f"REFERENCE:\n{render(gold)}\n\n"
            f"CANDIDATE:\n{render(prism)}")
    return llm.chat_json(JUDGE_SYSTEM_PROMPT, user, model=model, stage="judge")


def score(question: str, gold: dict, prism: dict, model: str = None) -> dict:
    """gold/prism are the {answer, formula, evidence} shape gold_sections()/
    prism_sections() produce. Returns {passed, method, comment}."""
    if parse_expected_number(gold["answer"]) is not None:
        return {"passed": numeric_match(prism["answer"], gold["answer"]), "method": "numeric",
                "comment": "deterministic numeric comparison"}
    verdict = llm_judge(question, gold, prism, model=model)
    return {"passed": verdict.get("pass"), "method": "llm_judge",
            "comment": verdict.get("comment", "")}
