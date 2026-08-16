"""Hybrid retrieval: BM25 + kNN + document scoping, in ONE SQL++ statement.

Two legs, unioned, then merged by rank rather than by score:

    lexical leg   the concept as a term-AND, plus the planner's content
                  anchors as phrases, scoped to the resolved document
    vector leg    kNN over the same Search Vector Index, same scope

**Why not one fused SEARCH() with a knn clause.** That was the original design
and it could not retrieve a chunk that BM25 alone ranked first. Couchbase sums
the lexical and vector scores, but only across documents already in the kNN
candidate set, and the two scores are on different scales - kNN returns
0.86-1.00 where BM25 returns 0.15-0.33. A chunk found only by the lexical leg
therefore sorts below every vector hit and falls off the LIMIT. Measured on
3M's FY2022 operating-margin table: rank 1 lexically, absent from the fused
query at every candidate depth and every boost. Boosting makes it worse, since
scores are normalised and a higher boost lowers the normalised result.

`build_fused_statement` keeps that original shape as a live fallback - set
config.HYBRID_FUSION = "score" to use it. It is kept as code rather than as a
comment so it stays runnable and testable instead of drifting.

**Why RRF.** Reciprocal rank fusion scores a document by its RANK in each leg,
1/(k + rank), so the scale mismatch that broke score fusion cannot arise. It
also rewards agreement: a chunk both legs rank highly beats one that only a
single leg likes. And it has no tuning surface - no boost, no leg weights.

**BM25 matches the CONCEPT and the ANCHORS, not the question.** Sending the raw
prompt scores mostly on common words and measurably bought nothing. The concept
goes in as a term-AND: "operating margin change" matches a chunk printing
"Operating income margin" in a row label and "Change" as a column header, which
no verbatim anchor the planner produced could reach.

**The scope predicate lives INSIDE SEARCH().** A scalar filter in the WHERE
clause is applied only after the Search service has returned its results, so
with a selective filter the top hits can all belong to other documents and be
discarded.
"""
from .. import config
from ..couchbase_io import query

# Standard RRF constant. Large enough that the difference between ranks 1 and 2
# does not dominate, small enough that deep ranks still separate.
RRF_K = 60
LEG_CANDIDATES = 20

SELECT_FIELDS = """SELECT META(d).id AS id,
           d.`text-to-embed` AS text,
           d.`xmeta-data`.filename AS filename,
           d.`meta-data`.`page-number` AS page,
           d.`meta-data`.`associated-titles` AS titles,
           d.`meta-data`.type AS type,
           SEARCH_SCORE() AS score"""


def _lexical_clause(question: str, anchors: list, concept: str,
                    title_boost: float, params: dict) -> str:
    """The disjuncts BM25 scores against. Falls back to the question text only
    when there is nothing better, so the leg never drops out entirely."""
    disjuncts = []
    if concept:
        disjuncts.append('{"match": $concept, "field": "text-to-embed", '
                         '"operator": "and"}')
        params["$concept"] = concept
    for i, anchor in enumerate(anchors or []):
        disjuncts.append(f'{{"match_phrase": $a{i}, "field": "text-to-embed"}}')
        params[f"$a{i}"] = anchor
    if not disjuncts:
        disjuncts.append('{"match": $match_text, "field": "text-to-embed"}')
        params["$match_text"] = question
    if title_boost:
        # Off by default: `associated-titles` is wrong on a meaningful fraction
        # of table chunks, so weighting it promotes confidently wrong chunks.
        disjuncts.append('{"match": $match_text, '
                         '"field": "meta-data.associated-titles", '
                         f'"boost": {title_boost}}}')
        params.setdefault("$match_text", question)
    return f'{{"disjuncts": [{", ".join(disjuncts)}]}}'


def build_statement(question: str, embedding: list, doc_name: str = None,
                    anchors: list = None, top_k: int = config.TOP_K,
                    knn_k: int = config.KNN_CANDIDATES, title_boost: float = 0.0,
                    concept: str = None) -> tuple:
    """Returns (statement, params) for the two-leg union.

    Split from execution so the query shape can be tested without a cluster -
    this is the core retrieval path and its correctness is mostly in where the
    predicates land. ORDER BY / LIMIT sit inside subqueries because a UNION
    branch cannot carry them directly.
    """
    params = {"$query_vector": embedding}
    lexical = _lexical_clause(question, anchors, concept, title_boost, params)

    if doc_name:
        params["$filename"] = config.source_filename(doc_name)
        scope = '{"field": "xmeta-data.filename", "match": $filename}'
        lexical_query = f'{{"conjuncts": [{scope}, {lexical}]}}'
        vector_query = scope
        knn_filter = f', "filter": {scope}'
    else:
        lexical_query = lexical
        vector_query = '{"match_all": {}}'
        knn_filter = ""

    collection = f"`{config.BUCKET}`.`{config.SCOPE}`.`{config.DOCS_COLLECTION}`"
    # The index name MUST be fully qualified; the bare short name does not
    # resolve and fails with "no search index".
    index = f'{{"index": "{config.FTS_DOCS_INDEX}"}}'
    knn = (f'"knn": [{{"field": "text-embedding", "vector": $query_vector, '
           f'"k": {knn_k}{knn_filter}}}]')

    statement = f"""SELECT RAW lexical FROM (
  {SELECT_FIELDS}, "lexical" AS leg
  FROM {collection} AS d
  WHERE SEARCH(d, {{"query": {lexical_query}}}, {index})
  ORDER BY SEARCH_SCORE() DESC LIMIT {LEG_CANDIDATES}
) AS lexical
UNION ALL
SELECT RAW vector FROM (
  {SELECT_FIELDS}, "vector" AS leg
  FROM {collection} AS d
  WHERE SEARCH(d, {{"query": {vector_query}, {knn}}}, {index})
  ORDER BY SEARCH_SCORE() DESC LIMIT {LEG_CANDIDATES}
) AS vector"""
    return statement, params


def build_fused_statement(question: str, embedding: list, doc_name: str = None,
                          anchors: list = None, top_k: int = config.TOP_K,
                          knn_k: int = config.KNN_CANDIDATES,
                          title_boost: float = 0.0, concept: str = None) -> tuple:
    """The ORIGINAL single fused SEARCH(): lexical and kNN in one query object,
    scores summed by the Search service. Retained as a fallback - see the module
    docstring for why it is not the default. Reachable via
    config.HYBRID_FUSION = "score"."""
    params = {"$query_vector": embedding}
    lexical = _lexical_clause(question, anchors, concept, title_boost, params)
    knn_filter = ""
    if doc_name:
        params["$filename"] = config.source_filename(doc_name)
        scope = '{"field": "xmeta-data.filename", "match": $filename}'
        query_clause = f'{{"conjuncts": [{scope}, {lexical}]}}'
        knn_filter = f', "filter": {scope}'
    else:
        query_clause = lexical

    statement = (
        SELECT_FIELDS + "\n"
        + f"FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.DOCS_COLLECTION}` AS d\n"
        + "WHERE SEARCH(d, {"
        + f'"query": {query_clause}, '
        + f'"knn": [{{"field": "text-embedding", "vector": $query_vector, '
          f'"k": {knn_k}{knn_filter}}}]'
        + f'}}, {{"index": "{config.FTS_DOCS_INDEX}"}})\n'
        + f"ORDER BY SEARCH_SCORE() DESC LIMIT {top_k}"
    )
    return statement, params


def rrf_merge(rows: list, top_k: int = config.TOP_K, k: int = RRF_K) -> list:
    """Reciprocal rank fusion over the legs present in `rows`.

    Rank is derived here rather than trusted from row order: UNION ALL makes no
    ordering guarantee across branches, so each leg is re-sorted by its own
    score before ranking. Only ranks are compared, never scores from different
    legs - that comparison is precisely what broke score fusion.
    """
    legs = {}
    for row in rows:
        legs.setdefault(row.get("leg", "unknown"), []).append(row)

    fused = {}
    for leg_rows in legs.values():
        leg_rows.sort(key=lambda r: r.get("score") or 0, reverse=True)
        for rank, row in enumerate(leg_rows, 1):
            key = row.get("id")
            entry = fused.setdefault(key, {"row": row, "rrf": 0.0, "legs": {}})
            entry["rrf"] += 1.0 / (k + rank)
            entry["legs"][row.get("leg")] = rank

    ordered = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
    out = []
    for entry in ordered[:top_k]:
        chunk = dict(entry["row"])
        chunk["rrf_score"] = round(entry["rrf"], 6)
        chunk["leg_ranks"] = entry["legs"]
        # `leg` on a fused chunk would name whichever branch happened to be
        # read last, which is meaningless once both contributed.
        chunk["leg"] = "+".join(sorted(entry["legs"]))
        out.append(chunk)
    return out


def hybrid_search(question: str, embedding: list, doc_name: str = None,
                  anchors: list = None, top_k: int = config.TOP_K,
                  knn_k: int = config.KNN_CANDIDATES, title_boost: float = 0.0,
                  concept: str = None) -> list:
    """One statement, two legs, merged by rank."""
    if config.HYBRID_FUSION == "score":
        statement, params = build_fused_statement(
            question, embedding, doc_name, anchors, top_k, knn_k, title_boost, concept)
        return query(statement, params)

    statement, params = build_statement(
        question, embedding, doc_name, anchors, top_k, knn_k, title_boost, concept)
    return rrf_merge(query(statement, params), top_k=top_k)
