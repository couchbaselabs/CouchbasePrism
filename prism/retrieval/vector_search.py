"""Dense vector retrieval, kNN-only, over the SAME Search Vector Index that
hybrid_search uses.

Previously this issued APPROX_VECTOR_DISTANCE against a separate Hyperscale
(GSI) vector index. That meant two vector indexes over the same field: at corpus
scale, roughly 180k chunks x 2048 dimensions built and held twice, for two query
paths that answer the same question. A SEARCH() carrying only a `knn` clause and
no `query` gives vector-ranked results from the FTS index directly.

It also removes a false claim: the trace labelled these statements "Hyperscale
Vector Index" while no such index existed on this scope, so the query was in
fact brute-force scanning every chunk.
"""
import os
import time

import requests

from .. import config, trace
from ..couchbase_io import query


def embed(text: str) -> list:
    """Couchbase AI Data Plane, OpenAI-compatible /v1/embeddings. Must be the
    same model the workflow used to embed the chunks."""
    endpoint = config.EMBED_ENDPOINT.rstrip("/") + "/v1/embeddings"
    body = {"model": config.EMBED_MODEL, "input": text}
    started = time.perf_counter()
    try:
        resp = requests.post(
            endpoint,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['API_KEY']}"},
            json=body, timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        embedding = result["data"][0]["embedding"]
        trace.add("embedding_call", endpoint=endpoint, request=body,
                  dimensions=len(embedding), usage=result.get("usage"),
                  elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
        return embedding
    except Exception as exc:
        trace.add("embedding_call", endpoint=endpoint, request=body,
                  error=f"{type(exc).__name__}: {exc}",
                  elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
        raise


def build_statement(embedding: list, doc_name: str = None,
                    top_k: int = config.TOP_K, knn_k: int = None) -> tuple:
    """kNN-only SEARCH(). No `query` clause at all: adding one would union its
    hits with the kNN hits and sum the scores, which is how the fused hybrid
    query ended up letting a filename match perturb vector rank."""
    params = {"$query_vector": embedding}
    knn_filter = ""
    if doc_name:
        params["$filename"] = config.source_filename(doc_name)
        knn_filter = (', "filter": {"field": "xmeta-data.filename", '
                      '"match": $filename}')
    statement = f"""
        SELECT META(d).id AS id,
               d.`text-to-embed` AS text,
               d.`xmeta-data`.filename AS filename,
               d.`meta-data`.`page-number` AS page,
               d.`meta-data`.`associated-titles` AS titles,
               d.`meta-data`.type AS type,
               SEARCH_SCORE() AS score
        FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.DOCS_COLLECTION}` AS d
        WHERE SEARCH(d, {{"knn": [{{"field": "text-embedding",
              "vector": $query_vector, "k": {knn_k or max(top_k, 50)}{knn_filter}}}]}},
              {{"index": "{config.FTS_DOCS_INDEX}"}})
        ORDER BY SEARCH_SCORE() DESC
        LIMIT {top_k}
    """
    return statement, params


def vector_search(embedding: list, doc_name: str = None,
                  top_k: int = config.TOP_K, knn_k: int = None) -> list:
    """doc_name=None searches the whole corpus - that is phase 1's unscoped
    baseline, and the reason it retrieved chunks from five different filings
    for a question about one."""
    statement, params = build_statement(embedding, doc_name, top_k, knn_k)
    return query(statement, params)
