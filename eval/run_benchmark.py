"""Run a corpus of questions through the PRISM runtime and score the results.

    python -m eval.run_benchmark --corpus financebench --company 3M
    python -m eval.run_benchmark --company 3M --out out/cold.json

The dictionary is read from disk as-is, so "cold" and "warm" are not modes -
they are just what the dictionary happens to contain. That is the two-pass
demo: run, approve one entry, run again, compare.
"""
import argparse
import json
import pathlib
import sys
import time

from eval import judge
from eval.corpora import load as load_corpus
from prism import catalog, dictionary, runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="financebench")
    ap.add_argument("--company", default=None, help="restrict to one company")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="out/benchmark.json")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    questions = corpus.questions(company=args.company, limit=args.limit)
    catalog_docs = catalog.load_all()
    dictionary_data = dictionary.load()
    approved = sum(1 for e in dictionary_data.get("entries", [])
                   if e.get("governance", {}).get("status") == "approved")
    print(f"corpus={corpus.NAME} questions={len(questions)} "
          f"catalog={len(catalog_docs)} approved_entries={approved}", file=sys.stderr)

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']}: {q['question'][:60]}", file=sys.stderr)
        t0 = time.perf_counter()
        try:
            result = runtime.answer_question(q["question"], catalog_docs,
                                             dictionary_data, model=args.model)
            verdict = judge.score(q["question"], q["expected_answer"],
                                  result["answer"], model=args.model)
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            computed = ", ".join(f"{c['label']}={c['value']:.4g}"
                                 for c in result["calculation"]["computed"])
            results.append({
                "id": q["id"],
                "expected_doc": q["doc_name"],
                "resolved_doc": result["resolved_doc"],
                "resolution_correct": result["resolved_doc"] == q["doc_name"],
                "answer_kind": result["answer_kind"],
                "concept": result["concept"],
                "governed": result["governed"],
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "answer": result["answer"],
                "bound_facts": result["bound_facts"],
                "calculation": result["calculation"],
                "conclusion": result["conclusion"],
                "plan": result["plan"],
                "retrieved": [{"page": c.get("page"), "type": c.get("type"),
                               "anchor_score": c.get("anchor_score")}
                              for c in result["chunks"]],
                "passed": verdict["passed"],
                "scoring_method": verdict["method"],
                "comment": verdict["comment"],
                "elapsed_ms": elapsed,
            })
            print(f"    kind={result['answer_kind']} governed={result['governed']} "
                  f"[{computed}] {result['conclusion'].get('status', '-')} "
                  f"passed={verdict['passed']} ({elapsed}ms)", file=sys.stderr)
        except Exception as e:
            results.append({"id": q["id"], "question": q["question"], "error": str(e)})
            print(f"    ERROR: {e}", file=sys.stderr)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1))

    scored = [r for r in results if r.get("passed") is not None]
    passed = sum(1 for r in scored if r["passed"])
    resolved = sum(1 for r in results if r.get("resolution_correct"))
    pct = 100 * passed / len(scored) if scored else 0
    print(f"\n=== converged with {corpus.NAME}: {passed}/{len(scored)} ({pct:.0f}%) | "
          f"doc resolution {resolved}/{len(results)} | "
          f"{len(results) - len(scored)} errored", file=sys.stderr)
    print(f"full output: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
