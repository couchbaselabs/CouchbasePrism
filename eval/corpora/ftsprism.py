"""ftsPrism corpus adapter - PRISM's own open sample corpus.

A directory the adapter reads, not something baked into the package. Every
PDF here is pulled directly from SEC EDGAR (public record - PRISM never
redistributes a third party's compiled copy), and every question, gold
answer and evidence excerpt is either authored by hand against the
underlying filing or drafted by PRISM's own answer pipeline and then
human-verified. See ftsprism/questions/_TEMPLATE.yaml for the full schema
and how to add to it.

Extra fields beyond the shared adapter interface (eval/corpora/__init__.py),
all optional - a consumer must tolerate their absence:
    doc_names    ALWAYS a list, even for a single-document question - one
                 type for every consumer to handle, and a multi-doc-range
                 question (see category below) genuinely needs more than one.
    number       drives ordering in the UI dropdown.
    title        short label for the UI dropdown - a noun phrase, not a
                 sentence.
    category     one of single-doc-extraction, single-doc-computed,
                 governed-formula, phrase-precision, concept-semantic,
                 multi-doc-range - see _TEMPLATE.yaml for what each means.
    formula      the exact expression that produces `answer`, or None for a
                 direct-extraction question. Identifier-based (e.g.
                 "(total_current_assets - inventory) / total_current_liabilities"),
                 matching how prism/retrieval/planner.py's formula_identifiers()
                 parses a dictionary.yaml formula - not a numbers-substituted
                 derivation, which wouldn't be reusable the same way.
    verified     true only when a human (not just PRISM) has checked the
                 values and evidence against the actual filing.
    notes        free-text context for whoever reads this question next.
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
    doc_names = data.get("doc_names") or []
    # Company is a convenience filter only (see documents.yaml's own note) -
    # every doc_names entry belongs to the same company in this corpus today,
    # so the first is enough to look it up by. A genuinely multi-company
    # corpus would need this to actually check for agreement, not just read
    # doc_names[0].
    doc = doc_lookup.get(doc_names[0], {}) if doc_names else {}
    return {
        "id": path.stem,
        "number": data.get("number"),
        "title": data.get("title"),
        "category": data.get("category"),
        "question": (data.get("question") or "").strip(),
        "answer": str(data.get("answer") or "").strip(),
        "doc_names": doc_names,
        "company": doc.get("company"),
        "evidence": list(data.get("evidence") or []),
        "formula": data.get("formula"),
        "verified": bool(data.get("verified", False)),
        "notes": (data.get("notes") or "").strip(),
    }


def questions(company: str = None, limit: int = None) -> list:
    """Returns dicts with the shared adapter shape
    ({id, question, answer, doc_names, company, evidence}), plus number,
    title, category, formula, verified and notes. Files under questions/
    whose name starts with "_" (the template) are never loaded as real
    questions."""
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
    """{doc_name: sector} from this corpus's own document metadata - not
    printed on any cover page, so it cannot come from PRISM's own
    extraction."""
    return {d["doc_name"]: d["sector"] for d in _documents() if d.get("sector")}
