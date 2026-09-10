"""Run ONE question through the full runtime and print every stage.

    python -m eval.debug_question 3m-2022-q3-quick-ratio
    python -m eval.debug_question 3m-002 --chunks

This is the tool for answering "why did that happen?" - it shows the resolved
document, the evidence plan and its content anchors, which chunks the anchors
actually matched (and their rarity scores), every bound fact with its row,
column, page and grounding status, the candidate formulas with their computed
values, and the validation verdict.

Most failures are visible in the plan or the bindings, not the final answer.
"""
import argparse
import json
import sys

from eval import judge, phases
from eval.corpora import load as load_corpus
from prism import catalog, dictionary, retrieval, runtime, trace


def rule(title: str) -> None:
    print(f"\n{'─' * 78}\n{title}\n{'─' * 78}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question_id")
    ap.add_argument("--corpus", default="ftsprism")
    ap.add_argument("--chunks", action="store_true",
                    help="print full untruncated chunk text")
    ap.add_argument("--phase", default=phases.DEFAULT_PHASE,
                    choices=sorted(phases.PHASES))
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    question = next((q for q in corpus.questions() if q["id"] == args.question_id), None)
    if question is None:
        sys.exit(f"no question {args.question_id!r} in corpus {corpus.NAME}")

    rule("QUESTION")
    print(f"id       : {question['id']}")
    print(f"question : {question['question']}")
    print(f"expected : {question['expected_answer']}")
    print(f"expected document: {question['doc_name']}")

    # Capture at the external-boundary level so this is the actual execution,
    # not a second debug implementation that can drift from the benchmark.
    with trace.capture() as events:
        catalog_docs = catalog.load_all()
        dictionary_data = dictionary.load()
        result = runtime.answer_question(question["question"], catalog_docs,
                                         dictionary_data, options=phases.get(args.phase),
                                         model=args.model)
        verdict = judge.score(question["question"], question["expected_answer"],
                              result["answer"], model=args.model)

    rule("CATALOG — document resolution")
    ok = result["resolved_doc"] == question["doc_name"]
    print(f"resolved : {result['resolved_doc']}  {'✓' if ok else '✗ MISMATCH'}")

    rule("EVIDENCE PLAN (no dictionary needed)")
    plan = result["plan"]
    print(f"concept        : {plan.get('concept')}")
    print(f"answer_kind    : {plan.get('answer_kind')}")
    for fact in retrieval.plan_facts(plan):
        print(f"  fact {fact.get('id')}: {fact.get('description', '')}")
        print(f"       anchors: {fact.get('content_anchors')}")

    rule(f"RETRIEVAL — {len(result['chunks'])} chunks")
    for i, c in enumerate(result["chunks"], 1):
        score = c.get("anchor_score")
        tag = f"anchor={score}" if score is not None else "vector"
        print(f"[{i}] p{c.get('page')} {c.get('type'):<10} {tag:<16} "
              f"titles={c.get('titles')}")
        print(f"    {(c.get('text') or '')[:160].replace(chr(10), ' ')}")
        if args.chunks:
            print(f"    ---\n{c.get('text')}\n    ---")

    rule("DICTIONARY")
    print(f"matched entry : {result['dictionary_entry'] or '(none — ungoverned)'}")
    print(f"policy        : {'approved' if result['has_policy'] else '(none)'}")

    if result["bound_facts"]:
        rule("FACT BINDING")
        for f in result["bound_facts"]:
            mark = "✓" if f.get("grounded") else "✗"
            print(f"{mark} {f.get('name')} = {f.get('value')} {f.get('units', '')}")
            print(f"    row='{f.get('row_label')}' column='{f.get('period')}' "
                  f"page={f.get('source_page')} entity={f.get('entity')}")
            for issue in f.get("binding_issues") or []:
                print(f"    !! {issue}")

    if result["candidates"]:
        rule("CANDIDATE INTERPRETATIONS (proposed, not exhaustive)")
        for c in result["candidates"]:
            print(f"- {c.get('method_name') or c.get('candidate_id')}: {c.get('formula')}")
            print(f"    {c.get('rationale')}")
        for c in result.get("rejected_candidates") or []:
            print(f"- REJECTED {c.get('method_name')}: {c.get('rejected_because')}")

    calc = result["calculation"]
    if calc["computed"] or calc["errors"]:
        rule(f"CALCULATION — {'GOVERNED' if calc['governed'] else 'ungoverned'}")
        for c in calc["computed"]:
            print(f"  {c['label']}: {c['formula']} = {c['value']:.6g}")
        for e in calc["errors"]:
            print(f"  !! {e['label']}: {e['error']}")

    if result["conclusion"]:
        rule("VALIDATION")
        for k, v in result["conclusion"].items():
            print(f"  {k}: {v}")

    rule("ANSWER")
    print(result["answer"])

    rule("SCORE")
    print(f"passed : {verdict['passed']}  ({verdict['method']})")
    print(f"comment: {verdict['comment']}")

    rule("LOW-LEVEL EXECUTION TRACE — chronological")
    for index, event in enumerate(events, 1):
        kind = event.get("type")
        elapsed = event.get("elapsed_ms")
        print(f"\n[{index}] {kind}" + (f"  ({elapsed} ms)" if elapsed is not None else ""))
        if kind == "couchbase_query":
            print("SQL++:")
            print(event.get("statement"))
            print("PARAMETERS:")
            print(json.dumps(event.get("params"), indent=2, default=str))
            print(f"RESULT: rows={event.get('row_count')} status={event.get('status')}")
            if event.get("metrics"):
                print("METRICS:")
                print(json.dumps(event["metrics"], indent=2, default=str))
        elif kind == "llm_call":
            request = event.get("request") or {}
            print(f"ENDPOINT: {event.get('endpoint')}")
            print("REQUEST:")
            print(json.dumps(request, indent=2, default=str))
            print("RESPONSE:")
            print(event.get("response") or event.get("error"))
            if event.get("usage"):
                print("USAGE:")
                print(json.dumps(event["usage"], indent=2, default=str))
        elif kind == "embedding_call":
            print(f"ENDPOINT: {event.get('endpoint')}")
            print("REQUEST:")
            print(json.dumps(event.get("request"), indent=2, default=str))
            print(f"RESULT: dimensions={event.get('dimensions')}")
            if event.get("usage"):
                print("USAGE:")
                print(json.dumps(event["usage"], indent=2, default=str))
        else:
            print(json.dumps(event, indent=2, default=str))


if __name__ == "__main__":
    main()
