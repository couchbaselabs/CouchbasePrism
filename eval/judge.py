"""Scoring.

Two paths, because they answer different questions:
  - A bare-numeric expected answer is compared deterministically. Free,
    unambiguous, and there is no reason to route it through a model.
  - A narrative expected answer goes to an LLM judge, because manual eyeballing
    does not scale to a full corpus run.

A note on language: "pass" here means CONVERGED WITH THE CORPUS'S CHOSEN
CONVENTION, not "objectively correct". Several quick-ratio and ROA conventions
are defensible finance; the gold answer picks one particular convention.
Reporting a score as "62% correct" would assert exactly the objective truth
PRISM is built not to assume.
"""
import re

from prism import llm

JUDGE_SYSTEM_PROMPT = (
    "You are grading whether a candidate answer to a financial question is substantively "
    "correct against a reference answer. The candidate is expected to be more verbose and "
    "phrased differently - that alone is not a failure. Judge only whether the candidate "
    "contains the same key facts, figures and conclusion as the reference.\n"
    "When the reference names SEVERAL contributing causes (a 'what drove this change' "
    "question), a 10-K's MD&A typically discusses more drivers than the reference happens "
    "to enumerate. Use common sense: pass the candidate if it reaches the same overall "
    "conclusion and correctly cites a majority of the reference's materially significant "
    "drivers, even if it phrases them differently, omits a minor one, or adds other "
    "well-sourced drivers the reference does not mention. Do not fail an answer solely for "
    "not naming every cause the reference lists, or for including extra correctly-cited "
    "context - fail it only when it reaches a different conclusion, misstates a key figure, "
    "or is missing enough of the material drivers that the explanation itself is thin.\n"
    'Respond with exactly one JSON object: {"pass": true or false, "comment": "one '
    'sentence explaining the verdict"}'
)


def parse_expected_number(expected: str):
    """Only a whole-string bare number counts. Most expected answers are
    narrative and contain incidental digits; "contains a number somewhere"
    would mis-route them into deterministic scoring."""
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


def llm_judge(question: str, expected: str, answer: str, model: str = None) -> dict:
    return llm.chat_json(
        JUDGE_SYSTEM_PROMPT,
        f"QUESTION: {question}\n\nREFERENCE ANSWER: {expected}\n\nCANDIDATE ANSWER: {answer}",
        model=model, stage="judge")


def score(question: str, expected: str, answer: str, model: str = None) -> dict:
    """Returns {passed, method, comment}."""
    if parse_expected_number(expected) is not None:
        return {"passed": numeric_match(answer, expected), "method": "numeric",
                "comment": "deterministic numeric comparison"}
    verdict = llm_judge(question, expected, answer, model=model)
    return {"passed": verdict.get("pass"), "method": "llm_judge",
            "comment": verdict.get("comment", "")}
