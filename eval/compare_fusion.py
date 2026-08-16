"""Compare fusion strategies on retrieval quality, holding everything else fixed.

    python -m eval.compare_fusion --company 3M

Until now the case for RRF rested on observability and on one end-to-end
example. That is not evidence of better retrieval. This measures it against
FinanceBench's own annotated evidence pages, which are ground truth we did not
author.

Every configuration sees the SAME planner output, anchors, concept, embedding,
candidate depth and document scope: the plan is generated once per question and
reused, because the planner is nondeterministic and re-planning per config
would confound the comparison with prompt variance.

Only the retrieval stage is measured. Binding and answer convergence run
through several more model calls whose variance is larger than the effect being
measured, so they belong in a separate run rather than mixed in here.

Page numbering: FinanceBench evidence_page_num is 0-indexed, the chunk
`meta-data.page-number` is 1-indexed. Verified against three statements in
3M's 2022 10-K (gold 47/49/51 -> chunk 48/50/52).
"""
import argparse
import json
import pathlib
import statistics
import time

from prism import catalog, config
from prism.couchbase_io import query
from prism.retrieval import embed, plan_anchors, plan_evidence
from prism.retrieval.hybrid_search import (
    build_fused_statement, build_statement, rrf_merge,
)

GOLD = pathlib.Path("financebench/data/financebench_open_source.jsonl")

CONFIGS = [
    ("A native fused SEARCH_SCORE", {"mode": "fused"}),
    ("B RRF k=60 equal",            {"mode": "rrf", "k": 60}),
    ("C RRF k=10 equal",            {"mode": "rrf", "k": 10}),
    ("D RRF k=1  equal",            {"mode": "rrf", "k": 1}),
    ("E RRF k=60 bm25 x2",          {"mode": "rrf", "k": 60,
                                     "weights": {"bm25": 2.0, "vector": 1.0}}),
]


def gold_questions(company: str) -> list:
    out = []
    for line in GOLD.read_text().splitlines():
        row = json.loads(line)
        if company and row.get("company") != company:
            continue
        pages = {e["evidence_page_num"] + 1 for e in row.get("evidence", [])
                 if e.get("evidence_page_num") is not None}
        if pages:
            out.append({"id": row["financebench_id"], "question": row["question"],
                        "doc_name": row["doc_name"], "gold_pages": pages})
    return out


def run_config(spec: dict, question: str, embedding: list, doc_name: str,
               anchors: list, concept: str) -> tuple:
    started = time.perf_counter()
    if spec["mode"] == "fused":
        statement, params = build_fused_statement(
            question, embedding, doc_name, anchors, concept=concept)
        chunks = query(statement, params)
    else:
        statement, params = build_statement(
            question, embedding, doc_name, anchors, concept=concept)
        chunks = rrf_merge(query(statement, params), top_k=config.TOP_K,
                           k=spec.get("k", 60), weights=spec.get("weights"))
    return chunks, round((time.perf_counter() - started) * 1000, 1)


def score_chunks(chunks: list, gold_pages: set) -> dict:
    pages = [c.get("page") for c in chunks]
    first = next((i for i, p in enumerate(pages, 1) if p in gold_pages), None)
    return {
        "rank": first,
        "recall@5": len(gold_pages & set(pages[:5])) / len(gold_pages),
        "recall@10": len(gold_pages & set(pages[:10])) / len(gold_pages),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", default="3M")
    args = parser.parse_args()

    questions = gold_questions(args.company)
    catalog_docs = catalog.load_all()
    print(f"{len(questions)} questions with annotated evidence pages\n")

    results = {label: [] for label, _ in CONFIGS}
    for q in questions:
        doc = catalog.resolve_for_question(catalog_docs, q["question"]) or q["doc_name"]
        plan = plan_evidence(q["question"])          # ONCE - reused by every config
        anchors, concept = plan_anchors(plan), plan.get("concept")
        embedding = embed(q["question"])             # ONCE
        print(f"{q['id']}  gold pages {sorted(q['gold_pages'])}  "
              f"concept={concept!r}")
        for label, spec in CONFIGS:
            chunks, elapsed = run_config(spec, q["question"], embedding, doc,
                                         anchors, concept)
            scored = score_chunks(chunks, q["gold_pages"])
            scored["ms"] = elapsed
            results[label].append(scored)
            print(f"    {label:<30} gold rank {str(scored['rank']):<5} "
                  f"r@5 {scored['recall@5']:.2f}  r@10 {scored['recall@10']:.2f}  "
                  f"{elapsed:>7.0f} ms")
        print()

    print(f"{'config':<32}{'r@5':>7}{'r@10':>7}{'found':>8}{'mean rank':>11}{'ms':>8}")
    for label, _ in CONFIGS:
        rows = results[label]
        found = [r for r in rows if r["rank"]]
        print(f"  {label:<30}"
              f"{statistics.mean(r['recall@5'] for r in rows):>7.2f}"
              f"{statistics.mean(r['recall@10'] for r in rows):>7.2f}"
              f"{len(found):>5}/{len(rows):<3}"
              f"{(statistics.mean(r['rank'] for r in found) if found else 0):>11.1f}"
              f"{statistics.mean(r['ms'] for r in rows):>8.0f}")


if __name__ == "__main__":
    main()
