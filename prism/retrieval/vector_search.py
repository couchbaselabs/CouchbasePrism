"""Dense vector retrieval over the Hyperscale Vector Index."""
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


def _select(distance_expr: str, where: str) -> str:
    return f"""
        SELECT META(d).id AS id,
               d.`text-to-embed` AS text,
               d.`xmeta-data`.filename AS filename,
               d.`meta-data`.`page-number` AS page,
               d.`meta-data`.`associated-titles` AS titles,
               d.`meta-data`.type AS type,
               {distance_expr} AS distance
        FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.DOCS_COLLECTION}` AS d
        WHERE d.`text-embedding` IS NOT MISSING {where}
        ORDER BY {distance_expr}
    """


def vector_search(embedding: list, doc_name: str = None,
                  top_k: int = config.TOP_K) -> list:
    """doc_name=None searches the whole corpus - that is phase 1's unscoped
    baseline, and the reason it retrieved chunks from five different filings
    for a question about one."""
    distance = (f'APPROX_VECTOR_DISTANCE(d.`text-embedding`, $query_vector, "L2", '
                f'{config.VECTOR_N_PROBES}, TRUE)')
    params = {"$query_vector": embedding}
    where = ""
    if doc_name:
        where = "AND d.`xmeta-data`.filename = $filename"
        params["$filename"] = config.source_filename(doc_name)
    return query(_select(distance, where) + f" LIMIT {top_k}", params)
