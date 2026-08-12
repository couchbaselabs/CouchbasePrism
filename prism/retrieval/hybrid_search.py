"""Hybrid BM25 + vector retrieval through a Search Vector Index.

One FTS index carries both the `text-to-embed` text field (BM25) and the
`text-embedding` vector field, so a single SEARCH() combines lexical and
semantic scoring. Couchbase combines the `query` and `knn` clauses with OR by
default, ranking documents matching both higher.

Kept because it is a genuine Couchbase capability worth demonstrating, and
because it is phase 3 of the evaluation. Note what phase 3 measured: feeding
the whole raw question into BM25 with no document filter scored no better than
pure vector, and sometimes worse - a wrong-document chunk sharing generic words
("which", "trade", "under") could outrank the right one. BM25 earns its keep on
rare, distinctive terms, not whole questions.
"""
from .. import config
from ..couchbase_io import query


def hybrid_search(question: str, embedding: list, doc_name: str = None,
                  top_k: int = config.TOP_K, knn_k: int = 50,
                  title_boost: float = 0.0) -> list:
    """title_boost > 0 adds a boosted match against
    `meta-data.associated-titles`. Use sparingly: those titles are unreliable
    (see anchor_search), so boosting them can promote confidently wrong chunks.
    """
    disjuncts = ['{"match": $match_text, "field": "text-to-embed"}']
    if title_boost:
        disjuncts.append(
            '{"match": $match_text, "field": "meta-data.associated-titles", '
            f'"boost": {title_boost}}}')
    where_doc = "d.`xmeta-data`.filename = $filename AND " if doc_name else ""
    statement = (
        "SELECT META(d).id AS id, d.`text-to-embed` AS text, "
        "d.`xmeta-data`.filename AS filename, "
        "d.`meta-data`.`page-number` AS page, "
        "d.`meta-data`.`associated-titles` AS titles, "
        "d.`meta-data`.type AS type, SEARCH_SCORE() AS score "
        f"FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.DOCS_COLLECTION}` AS d "
        f"WHERE {where_doc}SEARCH(d, {{"
        f'"query": {{"disjuncts": [{", ".join(disjuncts)}]}}, '
        f'"knn": [{{"field": "text-embedding", "vector": $query_vector, "k": {knn_k}}}]'
        # The index name MUST be fully qualified; the bare short name does not
        # resolve and fails with "no search index".
        f'}}, {{"index": "{config.FTS_DOCS_INDEX}"}}) '
        f"ORDER BY SEARCH_SCORE() DESC LIMIT {top_k}"
    )
    params = {"$query_vector": embedding, "$match_text": question}
    if doc_name:
        params["$filename"] = config.source_filename(doc_name)
    return query(statement, params)
