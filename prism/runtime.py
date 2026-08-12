"""The runtime: bind -> calculate -> validate -> answer (design/architecture.md §1).

`answer_question()` is the single entry point for BOTH the benchmark harness
and the Streamlit demo. That is deliberate - if the demo had its own pipeline
it would drift from the thing being measured, and the demo's credibility rests
on it being the measured thing.

The LLM's authority changes with governance state, and that switch is the
mechanism that makes "learns as it goes" compatible with "doesn't drift":
  unapproved concept -> may propose formulas and compute labeled candidates
  approved concept   -> forbidden from computing; binds facts and writes prose
                        around a number produced by code
"""
import json
import re

from . import catalog, dictionary, llm, retrieval

BINDER_SYSTEM_PROMPT = (
    "You are binding named financial facts to specific values from filing excerpts. "
    "Each requested fact is a snake_case identifier (e.g. total_current_assets, "
    "inventory) naming a line item that appears in the excerpts under some printed row "
    "label. Find the single figure each one refers to and report where it came from. Do "
    "not compute anything and do not derive one fact from others - only report figures "
    "printed in the excerpts.\n\n"
    "Respond with exactly one JSON object:\n"
    '{"facts": [{"name": "<the requested identifier, echoed exactly>", "value": <number, '
    'no commas or currency symbols>, "units": "e.g. USD millions", "period": "the column '
    'this figure sits under, e.g. June 30, 2023", "row_label": "the printed row label, '
    'verbatim", "entity": "the company", "source_page": <page number>}]}\n\n'
    "If a requested fact is genuinely not present, omit it rather than guessing or "
    "substituting a related figure. Getting the COLUMN right matters as much as the row: "
    "filings print several periods side by side and the wrong column is a silently wrong "
    "answer."
)

CANDIDATE_SYSTEM_PROMPT = (
    "You are proposing how a named financial concept is computed. Several conventions are "
    "often defensible; your job is to surface them, clearly labeled, NOT to pick one.\n\n"
    "Work from the DEFINITION OF THE CONCEPT FIRST, using your own domain knowledge. You "
    "will also be shown line items that a preliminary evidence plan happened to locate - "
    "that list is frequently INCOMPLETE and must not constrain you. If the correct "
    "definition needs a line item missing from it, use that line item anyway as a new "
    "snake_case identifier; it will be looked up afterwards. Never bend a definition to "
    "fit the facts already on hand: proposing the current ratio when asked for the quick "
    "ratio, merely because inventory was not in the list, is exactly the failure this "
    "step exists to prevent.\n\n"
    "Respond with exactly one JSON object:\n"
    '{"candidates": [{"label": "short name for this convention", "formula": "an '
    'arithmetic expression", "rationale": "one sentence on who uses this convention"}]}\n\n'
    "Formulas are arithmetic over snake_case identifiers combined with + - * / and "
    "parentheses. No function calls, no identifiers containing spaces.\n\n"
    "Propose every convention you consider genuinely defensible - typically 1 to 3. Do "
    "not pad the list, and do not claim to have enumerated every possibility."
)

BASE_ANSWER_PROMPT = (
    "You are a financial analyst specializing in SEC regulatory filings. Answer the "
    "question using ONLY the provided excerpts. If the answer is not in them, say so "
    "clearly. Cite sources by their [number].\n\n"
    "Sources marked [TABLE] contain structured financial data and are usually the "
    "authoritative source for a specific quantitative answer. When a question asks for a "
    "number, a rate, or what drove a financial change, check [TABLE] sources first and "
    "prefer their figures over a narrative source's paraphrase."
)

COMPUTED_ANSWER_PROMPT = (
    "You are a financial analyst specializing in SEC regulatory filings. You are given "
    "filing excerpts AND a block of values that have already been computed "
    "deterministically from figures bound to specific rows and columns of those "
    "excerpts.\n\n"
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


def to_identifier(name: str) -> str:
    """Fact names travel as snake_case because they end up inside formulas. The
    planner emits human labels ("Total current assets"), which are not parseable
    as expressions - every candidate formula failed on this until normalized."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", (name or "").lower())).strip("_")


def _format_chunks(chunks: list) -> str:
    return "\n\n---\n\n".join(
        f"[{i + 1}] {'[TABLE] ' if c.get('type') == 'table' else ''}"
        f"page {c.get('page')}\n{c.get('text', '')}"
        for i, c in enumerate(chunks))


# -------------------------------------------------------------------- bind

def bind_facts(question: str, required: list, chunks: list, model: str = None) -> list:
    if not required:
        return []
    user = (f"QUESTION: {question}\n\nFACTS TO BIND: {json.dumps(required)}\n\n"
            f"EXCERPTS:\n{_format_chunks(chunks)}")
    return llm.chat_json(BINDER_SYSTEM_PROMPT, user, model=model).get("facts", [])


def validate_bindings(bound: list, chunks: list) -> list:
    """Retrieval returning a table is not the same as knowing which cell means
    what. Entity, period, column and units each fail independently, and picking
    the wrong column is not an arithmetic error - validating the arithmetic
    would never catch it."""
    haystack = " ".join((c.get("text") or "") for c in chunks)
    periods = {}
    for fact in bound:
        issues = []
        value = fact.get("value")
        if value is None:
            issues.append("no value bound")
        else:
            n = abs(float(value))
            printed = f"{n:,.0f}" if float(n).is_integer() else f"{n:,}"
            if not any(v in haystack for v in
                       {printed, printed.replace(",", ""), f"{n:g}"}):
                issues.append(f"value {value} not found verbatim in retrieved text")
        if not fact.get("period"):
            issues.append("no period/column bound")
        else:
            periods.setdefault(fact["period"], []).append(fact.get("name"))
        fact["grounded"] = not issues
        fact["binding_issues"] = issues

    if len(periods) > 1:
        for fact in bound:
            fact["binding_issues"].append(f"facts bound to mixed periods: {sorted(periods)}")
            fact["grounded"] = False
    return bound


# --------------------------------------------------------------- calculate

def propose_candidates(concept: str, question: str, known_facts: list,
                       model: str = None) -> list:
    user = (f"CONCEPT: {concept}\nQUESTION: {question}\n"
            f"LINE ITEMS A PRELIMINARY PLAN LOCATED (possibly incomplete, do not treat "
            f"as the full vocabulary): {json.dumps(known_facts)}")
    candidates = llm.chat_json(CANDIDATE_SYSTEM_PROMPT, user, model=model,
                               timeout=60).get("candidates", [])
    for candidate in candidates:
        candidate["formula"] = re.sub(
            r"[A-Za-z_][A-Za-z_0-9 ]*[A-Za-z_0-9]",
            lambda m: to_identifier(m.group(0)), candidate.get("formula", ""))
    return candidates


def compute(entry, candidates: list, facts: dict) -> dict:
    """Governed -> one deterministic value from the approved formula.
    Ungoverned -> every proposed candidate, still evaluated in code. Either way
    the arithmetic happens here, never in a model."""
    result = {"governed": entry is not None, "computed": [], "errors": []}
    specs = ([{"label": "approved", "formula": entry["interpretation"]["formula"]}]
             if entry else candidates)
    for spec in specs:
        formula = spec.get("formula", "")
        try:
            result["computed"].append({
                "label": spec.get("label", "unnamed"),
                "formula": formula,
                "rationale": spec.get("rationale"),
                "value": round(dictionary.evaluate_formula(formula, facts), 6),
            })
        except dictionary.FormulaError as e:
            result["errors"].append({"label": spec.get("label"), "formula": formula,
                                     "error": str(e)})
    return result


# ---------------------------------------------------------------- validate

def validate_conclusion(computed: list, policy) -> dict:
    """Do the candidates agree on the conclusion the question actually asks
    for? Absent an approved policy there is no conclusion to agree about, and
    that itself is the finding - not a gap to paper over with a default."""
    values = [c["value"] for c in computed]
    if not values:
        return {"status": "no_value"}
    spread = round(max(values) - min(values), 6)
    if policy is None:
        return {"status": "no_policy", "spread": spread, "values": values,
                "note": "value(s) computed; no approved interpretation policy exists, "
                        "so no verdict is authorised"}
    verdicts = {dictionary.judge(v, policy)[0] for v in values}
    if len(verdicts) == 1:
        return {"status": "agreed", "verdict": verdicts.pop(), "spread": spread,
                "explanation": dictionary.judge(values[0], policy)[1]}
    return {"status": "conflicting", "spread": spread, "values": values,
            "note": "candidate conventions straddle the approved threshold - blocking on "
                    "review rather than picking one"}


# ------------------------------------------------------------------ answer

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
    context = _format_chunks(chunks)
    if calc["computed"]:
        prompt = (f"{COMPUTED_ANSWER_PROMPT}\n\nCOMPUTED VALUES:\n"
                  f"{render_computed_block(calc, conclusion)}\n\n"
                  f"EXCERPTS:\n{context}\n\nQUESTION: {question}\n\nANSWER:")
    else:
        prompt = f"{BASE_ANSWER_PROMPT}\n\nEXCERPTS:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    return llm.chat_text(prompt, model=model)


# ---------------------------------------------------------------- pipeline

def answer_question(question: str, catalog_docs: list, dictionary_data: dict = None,
                    model: str = None) -> dict:
    """One question, end to end. Returns every intermediate stage so the demo
    can show the pipeline and the benchmark can diagnose it."""
    dictionary_data = dictionary_data if dictionary_data is not None else dictionary.load()

    doc_name = catalog.resolve_for_question(catalog_docs, question)
    plan = retrieval.plan_evidence(question, model=model)
    chunks = retrieval.retrieve(question, doc_name, plan)

    kind = plan.get("answer_kind")
    entry = dictionary.find_metric(dictionary_data, plan.get("concept", ""))
    policy = dictionary.find_policy(dictionary_data, entry["id"]) if entry else None

    bound, candidates = [], []
    calc = {"governed": False, "computed": [], "errors": []}
    conclusion = {}

    if kind in ("derived_metric", "judgment"):
        planned = [to_identifier(f) for f in (plan.get("required_facts") or [])]
        if entry:
            needed = set(entry["interpretation"]["required_facts"])
        else:
            # Propose first, then bind the UNION of what the plan asked for and
            # what the candidates reference. The planner under-specifies: it
            # omitted `inventory` for quick ratio, leaving the only correct
            # formula with an unbound name and nothing computable.
            candidates = propose_candidates(plan.get("concept", ""), question,
                                            planned, model=model)
            needed = set(planned)
            for candidate in candidates:
                needed |= retrieval.formula_identifiers(candidate.get("formula", ""))
        bound = validate_bindings(
            bind_facts(question, sorted(needed), chunks, model=model), chunks)
        facts = {f["name"]: f["value"] for f in bound
                 if f.get("grounded") and f.get("value") is not None}
        calc = compute(entry, candidates, facts)
        conclusion = validate_conclusion(calc["computed"], policy)

    answer = synthesize(question, chunks, calc, conclusion, model=model)

    return {
        "question": question,
        "resolved_doc": doc_name,
        "plan": plan,
        "answer_kind": kind,
        "concept": plan.get("concept"),
        "governed": calc["governed"],
        "dictionary_entry": entry["id"] if entry else None,
        "has_policy": policy is not None,
        "chunks": chunks,
        "bound_facts": bound,
        "candidates": candidates,
        "calculation": calc,
        "conclusion": conclusion,
        "answer": answer,
    }
