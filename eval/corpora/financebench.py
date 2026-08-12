"""FinanceBench corpus adapter.

FinanceBench (github.com/patronus-ai/financebench) is ONE test suite, not the
system. Everything corpus-specific lives behind this adapter's interface so a
second suite is a new file here, not a refactor of the pipeline.

Their `financebench_document_information.jsonl` is deliberately NOT used as
input - PRISM builds its own catalog and is scored against theirs.
"""
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[2] / "financebench"
NAME = "financebench"


def pdf_dir() -> pathlib.Path:
    return ROOT / "pdfs"


def questions(company: str = None, limit: int = None) -> list:
    """Returns dicts with a stable shape the harness relies on:
    {id, question, expected_answer, doc_name, company}."""
    path = ROOT / "data" / "financebench_open_source.jsonl"
    out = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if company and row["company"] != company:
                continue
            out.append({
                "id": row["financebench_id"],
                "question": row["question"],
                "expected_answer": row["answer"],
                "doc_name": row["doc_name"],
                "company": row["company"],
                "evidence": [
                    {"doc_name": e["doc_name"], "page": e["evidence_page_num"],
                     "text": e["evidence_text"]}
                    for e in row.get("evidence", [])
                ],
            })
    return out[:limit] if limit else out


def documents(company: str = None) -> list:
    """Source PDFs, as (doc_name, path) pairs."""
    pattern = f"{company}_*.pdf" if company else "*.pdf"
    return [(p.stem, p) for p in sorted(pdf_dir().glob(pattern))]
