"""Run ONE question through the full runtime and print every stage.

    python -m eval.debug_question financebench_id_00807
    python -m eval.debug_question financebench_id_00941 --chunks

This is the tool for answering "why did that happen?" - it shows the resolved
document, the evidence plan and its content anchors, which chunks the anchors
actually matched (and their rarity scores), every bound fact with its row,
column, page and grounding status, the candidate formulas with their computed
values, and the validation verdict.

Most failures are visible in the plan or the bindings, not the final answer.
"""
import argparse
import sys

from eval import judge
from eval.corpora import load as load_corpus
from prism import catalog, dictionary, runtime


def rule(title: str) -> None:
    print(f"\n{'─' * 78}\n{title}\n{'─' * 78}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question_id")
    ap.add_argument("--corpus", default="financebench")
    ap.add_argument("--chunks", action="store_true",
                    help="print full untruncated chunk text")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    question = next((q for q in corpus.questions() if q["id"] == args.question_id), None)
    if question is None:
        sys.exit(f"no question {args.question_id!r} in corpus {corpus.NAME}")

    catalog_docs = catalog.load_all()
    dictionary_data = dictionary.load()

    rule("QUESTION")
    print(f"id       : {question['id']}")
    print(f"question : {question['question']}")
    print(f"expected : {question['expected_answer']}")
    print(f"expected document: {question['doc_name']}")

    result = runtime.answer_question(question["question"], catalog_docs,
                                     dictionary_data, model=args.model)

    rule("CATALOG — document resolution")
    ok = result["resolved_doc"] == question["doc_name"]
    print(f"resolved : {result['resolved_doc']}  {'✓' if ok else '✗ MISMATCH'}")

    rule("EVIDENCE PLAN (no dictionary needed)")
    plan = result["plan"]
    print(f"concept        : {plan.get('concept')}")
    print(f"answer_kind    : {plan.get('answer_kind')}")
    print(f"required_facts : {plan.get('required_facts')}")
    print(f"content_anchors: {plan.get('content_anchors')}")

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
            print(f"- {c.get('label')}: {c.get('formula')}")
            print(f"    {c.get('rationale')}")

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

    verdict = judge.score(question["question"], question["expected_answer"],
                          result["answer"], model=args.model)
    rule("SCORE")
    print(f"passed : {verdict['passed']}  ({verdict['method']})")
    print(f"comment: {verdict['comment']}")


if __name__ == "__main__":
    main()
