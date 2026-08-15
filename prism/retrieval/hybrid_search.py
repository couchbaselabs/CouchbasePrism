"""Hybrid retrieval: BM25 + kNN + document scoping, in ONE SQL++ statement.

A single `SEARCH()` against a Search Vector Index carries all three legs:

    query.conjuncts[0]  document scope   — exact filename match
    query.conjuncts[1]  BM25 lexical     — the planner's content anchors,
                                           each as a phrase
    knn                 vector kNN       — with its own filter, so the vector
                                           leg is scoped too

Two things here are easy to get wrong and both were:

**BM25 matches the ANCHORS, not the question.** Sending the raw prompt to BM25
means scoring mostly on common words - "does", "have", "based", "please" - and
measurably bought nothing (a catalog+BM25 configuration scored the same as
catalog alone). The planner already produces the verbatim printed row labels
worth matching, e.g. ["Total current assets", "Total current liabilities"], and
those are used as `match_phrase` so a multi-word label matches as a sequence
rather than an OR over its tokens.

**The scope predicate lives INSIDE SEARCH().** A scalar filter written outside,
in the WHERE clause, is applied by the Query service only after the Search
service has already returned its k results - so with a selective filter and a
small k, the top hits can all belong to other documents and be discarded,
leaving few or no rows. Putting the filename in the conjuncts scopes the
lexical leg, and the knn `filter` object scopes the vector leg.

BM25 also gives anchor rarity for free: IDF is intrinsic to it, so a
distinctive label naturally outweighs a generic one without the hand-rolled
1/log2(2+df) weighting the standalone anchor path needs.
"""
from .. import config
from ..couchbase_io import query

SELECT_FIELDS = """
    SELECT META(d).id AS id,
           d.`text-to-embed` AS text,
           d.`xmeta-data`.filename AS filename,
           d.`meta-data`.`page-number` AS page,
           d.`meta-data`.`associated-titles` AS titles,
           d.`meta-data`.type AS type,
           SEARCH_SCORE() AS score
"""


def build_statement(question: str, embedding: list, doc_name: str = None,
                    anchors: list = None, top_k: int = config.TOP_K,
                    knn_k: int = config.KNN_CANDIDATES, title_boost: float = 0.0) -> tuple:
    """Returns (statement, params). Split from execution so the query shape can
    be tested without a cluster - this is the core retrieval path and its
    correctness is mostly in where the predicates land."""
    params = {"$query_vector": embedding}
    anchors = anchors or []

    if anchors:
        lexical = [f'{{"match_phrase": $a{i}, "field": "text-to-embed"}}'
                   for i in range(len(anchors))]
        for i, anchor in enumerate(anchors):
            params[f"$a{i}"] = anchor
    else:
        lexical = ['{"match": $match_text, "field": "text-to-embed"}']
        params["$match_text"] = question

    if title_boost:
        # Off by default: `associated-titles` is wrong on a meaningful fraction
        # of table chunks, so weighting it promotes confidently wrong chunks.
        lexical.append('{"match": $match_text, "field": "meta-data.associated-titles", '
                       f'"boost": {title_boost}}}')
        params.setdefault("$match_text", question)

    lexical_clause = f'{{"disjuncts": [{", ".join(lexical)}]}}'
    knn_filter = ""
    if doc_name:
        params["$filename"] = config.source_filename(doc_name)
        scope = '{"field": "xmeta-data.filename", "match": $filename}'
        query_clause = f'{{"conjuncts": [{scope}, {lexical_clause}]}}'
        knn_filter = f', "filter": {scope}'
    else:
        query_clause = lexical_clause

    statement = (
        SELECT_FIELDS
        + f"FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.DOCS_COLLECTION}` AS d\n"
        + "WHERE SEARCH(d, {"
        + f'"query": {query_clause}, '
        + f'"knn": [{{"field": "text-embedding", "vector": $query_vector, '
          f'"k": {knn_k}{knn_filter}}}]'
        # The index name MUST be fully qualified; the bare short name does not
        # resolve and fails with "no search index".
        + f'}}, {{"index": "{config.FTS_DOCS_INDEX}"}})\n'
        + f"ORDER BY SEARCH_SCORE() DESC LIMIT {top_k}"
    )
    return statement, params


def hybrid_search(question: str, embedding: list, doc_name: str = None,
                  anchors: list = None, top_k: int = config.TOP_K,
                  knn_k: int = config.KNN_CANDIDATES, title_boost: float = 0.0) -> list:
    """One statement, three legs. `anchors` empty falls back to matching the
    question text, so the lexical leg still contributes something rather than
    dropping out entirely."""
    statement, params = build_statement(question, embedding, doc_name, anchors,
                                        top_k, knn_k, title_boost)
    return query(statement, params)
