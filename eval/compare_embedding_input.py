"""Compare what text gets embedded for the vector channel, holding the
lexical channel and everything else fixed.

    python -m eval.compare_embedding_input --company 3M

The vector leg currently embeds the raw question verbatim (vector_search.py /
combined_search.py), while the lexical leg searches the planner's clarified
concept + anchors, not the raw question. That is a real asymmetry: the
question a user actually asks is often full of framing the embedding has to
see past ("If we exclude the impact of M&A, which segment..."), while BM25
gets the planner's already-distilled intent. This measures whether embedding
something closer to what BM25 sees helps, hurts, or does nothing - rather
than assuming any of the three.

Same discipline as compare_fusion.py: the plan is generated once per question
and reused across every variant, because the planner is nondeterministic and
re-planning per variant would confound the comparison with prompt variance.
Only retrieval is measured - binding and answer synthesis are separate,
noisier stages that belong in their own comparison.
"""
import argparse
import json
import pathlib
import statistics
import time

from prism import catalog
from prism.retrieval import embed, plan_anchors, plan_evidence
from prism.retrieval.hybrid_search import _merged_terms, hybrid_search

GOLD = pathlib.Path("financebench/data/financebench_open_source.jsonl")

# Each variant is (label, fn(question, concept, anchors) -> text_to_embed).
# The lexical leg is untouched in every variant - only what feeds embed()
# changes, via the separate `embedding` param hybrid_search already takes.
VARIANTS = [
    ("A raw question (current)", lambda q, c, a: q),
    ("B concept alone", lambda q, c, a: c or q),
    ("C question + concept", lambda q, c, a: f"{q} {c}" if c else q),
    ("D merged lexical bag", lambda q, c, a: _merged_terms(c, a) or q),
    ("E concept + anchors, natural", lambda q, c, a: "; ".join(
        [c] + list(a or [])) if c else q),
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
    parser.add_argument("--company", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    questions = gold_questions(args.company)
    ingested = catalog.ingested_doc_names()
    dropped = [q["id"] for q in questions if q["doc_name"] not in ingested]
    questions = [q for q in questions if q["doc_name"] in ingested]
    if args.limit:
        questions = questions[:args.limit]
    if dropped:
        print(f"skipped {len(dropped)} question(s) whose document was not ingested")
    print(f"{len(questions)} questions with annotated evidence pages\n")

    results = {label: [] for label, _ in VARIANTS}
    embed_cache = {}
    for q in questions:
        plan = plan_evidence(q["question"])          # ONCE - reused by every variant
        anchors, concept = plan_anchors(plan), plan.get("concept")
        print(f"{q['id']}  gold pages {sorted(q['gold_pages'])}  concept={concept!r}")
        for label, text_fn in VARIANTS:
            text = text_fn(q["question"], concept, anchors)
            if text not in embed_cache:
                embed_cache[text] = embed(text)
            embedding = embed_cache[text]
            started = time.perf_counter()
            chunks = hybrid_search(q["question"], embedding, q["doc_name"],
                                   anchors=anchors, concept=concept, top_k=args.top_k)
            elapsed = round((time.perf_counter() - started) * 1000, 1)
            scored = score_chunks(chunks, q["gold_pages"])
            scored["ms"] = elapsed
            results[label].append(scored)
            print(f"    {label:<30} gold rank {str(scored['rank']):<5} "
                  f"r@5 {scored['recall@5']:.2f}  r@10 {scored['recall@10']:.2f}")
        print()

    print(f"{'variant':<32}{'r@5':>7}{'r@10':>7}{'found':>8}{'mean rank':>11}")
    for label, _ in VARIANTS:
        rows = results[label]
        found = [r for r in rows if r["rank"]]
        print(f"  {label:<30}"
              f"{statistics.mean(r['recall@5'] for r in rows):>7.2f}"
              f"{statistics.mean(r['recall@10'] for r in rows):>7.2f}"
              f"{len(found):>5}/{len(rows):<3}"
              f"{(statistics.mean(r['rank'] for r in found) if found else 0):>11.1f}")


if __name__ == "__main__":
    main()
