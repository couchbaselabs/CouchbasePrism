"""Tier 2 — retrieval over the chunk collection (design/architecture.md §3).

Chunks are produced by the Couchbase AI Data Plane Unstructured Data Workflow.
PRISM does not parse or chunk PDFs itself; improvements to chunk quality are
feedback to that service, not code here.

The evidence planner runs BEFORE any dictionary lookup and needs no dictionary
entry to work - that ordering is the claim ADR-0001 rests on, and it is why
retrieval can succeed for concepts the system has never seen.
"""
import ast
import math
import os

import requests

from . import config, llm
from .couchbase_io import query

PLANNER_SYSTEM_PROMPT = (
    "You are planning what evidence is needed to answer a question about a specific SEC "
    "filing. You do NOT have the document. Plan from general knowledge of how SEC filings "
    "are structured.\n\n"
    "Respond with exactly one JSON object:\n"
    '{\n'
    '  "concept": "the financial concept being asked about, e.g. quick ratio",\n'
    '  "answer_kind": "stated_fact" | "derived_metric" | "judgment",\n'
    '  "required_facts": ["the specific figures needed to answer"],\n'
    '  "content_anchors": ["verbatim row or column labels"],\n'
    '  "preferred_artifacts": ["table" and/or "text"]\n'
    "}\n\n"
    "content_anchors is the important field and the easiest to get wrong. These must be "
    "labels that would appear VERBATIM as a row or column heading inside the relevant "
    "table of a real SEC filing - the literal printed text, not a paraphrase of the "
    'question and not a section title. Good: "Total current liabilities", "Organic '
    'sales", "Trading Symbol(s)", "Purchases of property, plant and equipment". Bad: '
    '"liquidity information", "the balance sheet" (not printed row labels).\n\n'
    "DO NOT PAD THE LIST. Emit only anchors you are genuinely confident are printed "
    "verbatim in this kind of filing. Two precise anchors are far better than six "
    'guesses: a plausible-sounding invention ("Maturity Date", "Description") will match '
    "unrelated tables. Prefer long, distinctive, complete phrases over short generic "
    'words - "Name of each exchange on which registered" is excellent, "Description" is '
    "useless. If you are confident about only one anchor, return only that one.\n\n"
    'answer_kind: "stated_fact" if the answer is printed in the document as-is; '
    '"derived_metric" if it must be computed from other figures; "judgment" if it asks '
    "whether something is good/healthy/stable, which requires a threshold opinion beyond "
    "the arithmetic."
)


def plan_evidence(question: str, model: str = None) -> dict:
    return llm.chat_json(PLANNER_SYSTEM_PROMPT, question, model=model, timeout=60)


def formula_identifiers(formula: str) -> set:
    try:
        return {n.id for n in ast.walk(ast.parse(formula, mode="eval"))
                if isinstance(n, ast.Name)}
    except SyntaxError:
        return set()


# ------------------------------------------------------------------ embed

def embed(text: str) -> list:
    resp = requests.post(
        config.EMBED_ENDPOINT.rstrip("/") + "/v1/embeddings",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['API_KEY']}"},
        json={"model": config.EMBED_MODEL, "input": text}, timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


# ----------------------------------------------------------------- search

def anchor_search(anchors: list, doc_name: str,
                  limit: int = config.MAX_ANCHOR_CHUNKS) -> list:
    """Deterministic substring match on the chunk's own text. Never consults
    `associated-titles`, which is demonstrably unreliable - three separate
    tables in the sample corpus carry titles describing something else on the
    page (a securities table titled ["Delaware", "41-0417775"], a Consumer
    segment table titled "PERFORMANCE BY GEOGRAPHIC AREA").

    Ranked by anchor RARITY, not raw hit count. The planner reliably emits a
    plausible-but-invented label alongside the correct ones; under raw counting
    two junk anchors matching a wrong table outrank one precise anchor matching
    the right one. Weighting by 1/log2(2+df) makes a rare, specific anchor
    dominate several generic ones - IDF, for the reason BM25 uses it.
    """
    if not anchors:
        return []
    rows = query(
        f"""
        SELECT META(d).id AS id,
               d.`text-to-embed` AS text,
               d.`xmeta-data`.filename AS filename,
               d.`meta-data`.`page-number` AS page,
               d.`meta-data`.`associated-titles` AS titles,
               d.`meta-data`.type AS type,
               ARRAY a FOR a IN $anchors
                   WHEN CONTAINS(LOWER(d.`text-to-embed`), LOWER(a)) END AS matched_anchors
        FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.DOCS_COLLECTION}` AS d
        WHERE d.`xmeta-data`.filename = $filename
          AND ANY a IN $anchors
              SATISFIES CONTAINS(LOWER(d.`text-to-embed`), LOWER(a)) END
        LIMIT 100
        """,
        {"$anchors": anchors, "$filename": config.source_filename(doc_name)},
    )
    if not rows:
        return []

    doc_freq = {}
    for r in rows:
        for a in r.get("matched_anchors") or []:
            doc_freq[a] = doc_freq.get(a, 0) + 1
    for r in rows:
        matched = r.get("matched_anchors") or []
        r["anchor_hits"] = len(matched)
        r["anchor_score"] = round(
            sum(1.0 / math.log2(2 + doc_freq.get(a, 0)) for a in matched), 4)
    rows.sort(key=lambda r: r["anchor_score"], reverse=True)
    return rows[:limit]


def vector_search(embedding: list, doc_name: str, top_k: int = config.TOP_K) -> list:
    distance = (f'APPROX_VECTOR_DISTANCE(d.`text-embedding`, $query_vector, "L2", '
                f'{config.VECTOR_N_PROBES}, TRUE)')
    return query(
        f"""
        SELECT META(d).id AS id,
               d.`text-to-embed` AS text,
               d.`xmeta-data`.filename AS filename,
               d.`meta-data`.`page-number` AS page,
               d.`meta-data`.`associated-titles` AS titles,
               d.`meta-data`.type AS type,
               {distance} AS distance
        FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.DOCS_COLLECTION}` AS d
        WHERE d.`text-embedding` IS NOT MISSING
          AND d.`xmeta-data`.filename = $filename
        ORDER BY {distance}
        LIMIT {top_k}
        """,
        {"$query_vector": embedding, "$filename": config.source_filename(doc_name)},
    )


def combine(anchor_chunks: list, vector_chunks: list, top_k: int = config.TOP_K) -> list:
    """Anchor hits lead - they are structurally identified, high precision -
    and vector fills the remainder for coverage."""
    seen, out = set(), []
    for chunk in list(anchor_chunks) + list(vector_chunks):
        if len(out) >= top_k:
            break
        key = chunk.get("id") or (chunk.get("text") or "")[:200]
        if key not in seen:
            seen.add(key)
            out.append(chunk)
    return out


def retrieve(question: str, doc_name: str, plan: dict, top_k: int = config.TOP_K) -> list:
    return combine(anchor_search(plan.get("content_anchors") or [], doc_name),
                   vector_search(embed(question), doc_name, top_k), top_k)
