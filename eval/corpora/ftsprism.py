"""ftsPrism corpus adapter - PRISM's own open sample corpus.

Same shape FinanceBench established (eval/corpora/financebench.py): a
directory the adapter reads, not something baked into the package. The
difference is what's IN it. Every PDF here is pulled directly from SEC EDGAR
(public record - PRISM never redistributes a third party's compiled copy),
and every question, gold answer and evidence excerpt is either authored by
hand against the underlying filing or drafted by PRISM's own answer pipeline
and then human-verified - never paraphrased or derived from FinanceBench's
own compiled question set. See ftsprism/README.md for the full schema and
how to add to it.

Extra fields beyond the shared adapter interface (eval/corpora/__init__.py),
all optional and absent from financebench.py's questions() - a consumer must
tolerate their absence:
    formula      the exact expression that produces expected_answer, or None
                 for a direct-extraction question. Identifier-based (e.g.
                 "(total_current_assets - inventory) / total_current_liabilities"),
                 matching how prism/retrieval/planner.py's formula_identifiers()
                 parses a dictionary.yaml formula - not a numbers-substituted
                 derivation, which wouldn't be reusable the same way.
    keywords     literal lexical terms the question should be findable by -
                 lets a hybrid-search demo show the BM25 channel actually
                 contributing something, not just the vector one.
    tags         categorizes the QUESTION (reasoning type, shape - e.g.
                 numerical-reasoning, single-document), not what it's
                 findable by - that's keywords, above.
    gold_answer  optional {value, unit, tolerance} alongside expected_answer,
                 for a numeric question - not a replacement for it. Lets a
                 future scorer exact-match instead of relying on an LLM judge
                 for something that's actually deterministic.
"""
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parent / "ftsprism"
NAME = "ftsprism"


def pdf_dir() -> pathlib.Path:
    return ROOT / "pdfs"


def _documents() -> list:
    path = ROOT / "documents.yaml"
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text()) or []


def _load_question(path: pathlib.Path, doc_lookup: dict) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    doc = doc_lookup.get(data["doc_name"], {})
    return {
        "id": path.stem,
        "question": (data.get("question") or "").strip(),
        "expected_answer": str(data.get("expected_answer")),
        "doc_name": data["doc_name"],
        "company": doc.get("company"),
        "evidence": [
            {"doc_name": data["doc_name"], "page": e.get("page"),
             "text": e.get("text"), "title": e.get("title"),
             "type": e.get("type"), "table_name": e.get("table_name")}
            for e in data.get("evidence", [])
        ],
        "formula": data.get("formula"),
        "concept": data.get("concept"),
        "keywords": data.get("keywords") or [],
        "tags": data.get("tags") or [],
        "gold_answer": data.get("gold_answer"),
        "lineage": data.get("lineage") or {},
    }


def questions(company: str = None, limit: int = None) -> list:
    """Returns dicts with the same stable shape financebench.py returns
    ({id, question, expected_answer, doc_name, company, evidence}), plus
    formula, concept, keywords and lineage. Files under questions/ whose name
    starts with "_" (the template) are never loaded as real questions."""
    doc_lookup = {d["doc_name"]: d for d in _documents()}
    out = []
    for path in sorted((ROOT / "questions").glob("*.yaml")):
        if path.stem.startswith("_"):
            continue
        q = _load_question(path, doc_lookup)
        if company and q["company"] != company:
            continue
        out.append(q)
    return out[:limit] if limit else out


def documents(company: str = None) -> list:
    """Source PDFs, as (doc_name, path) pairs - only ones documents.yaml
    actually lists, so a PDF dropped in pdfs/ without a matching entry (not
    yet catalogued) is not silently picked up.

    pdfs/ holds one subfolder per company (pdfs/3M/, pdfs/BestBuy/, ...), so
    this globs recursively rather than assuming a flat directory - doc_name
    (the filename stem) is what identifies a document either way, not which
    subfolder it happens to live in."""
    names = {d["doc_name"] for d in _documents()
             if not company or d.get("company") == company}
    return [(p.stem, p) for p in sorted(pdf_dir().rglob("*.pdf")) if p.stem in names]


def sectors() -> dict:
    """{doc_name: sector} from this corpus's own document metadata - same
    reasoning as financebench.py's sectors(): not printed on any cover page,
    so it cannot come from PRISM's own extraction."""
    return {d["doc_name"]: d["sector"] for d in _documents() if d.get("sector")}
