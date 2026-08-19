"""Hybrid retrieval: BM25 + kNN + document scoping, in ONE SQL++ statement.

Two legs, unioned, then merged by rank rather than by score:

    lexical leg   the concept as a term-AND, plus the planner's content
                  anchors as phrases, scoped to the resolved document
    vector leg    kNN over the same Search Vector Index, same scope

**Why not one fused SEARCH() with a knn clause.** Couchbase unions the query
and knn hits and sums their scores, so a lexical-only document is eligible -
it simply carries only its lexical score. That score is not competitive: 3M's
FY2022 operating-margin table, the best lexical hit in its filing, lands at
rank 136 with 0.6695 against a rank-10 floor of 0.8696. It is outranked rather
than excluded, and the effect is the same - the chunk never reaches the model.
Boosting makes it worse, since scores are normalised and a higher boost lowers
the result (0.7394 at boost 1, 0.2731 at boost 1000).

Measured over two independent runs of eval.compare_fusion against
FinanceBench's annotated evidence pages: recall@10 0.67 vs 0.54 for the fused
query, gold evidence found for 6 of 8 questions vs 5, at equal latency. The
whole difference is one question, so this is a modest and narrow result rather
than a decisive one.

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
discarded. On the lexical leg it is a conjunct; on the vector leg it is the
knn `filter`, and NOT also a `query`.

That distinction matters: `query` and `knn` in one search object are unioned
and their scores summed, so a `query` matching the filename would add every
chunk in the filing to the vector leg. Today that happens to be harmless -
`xmeta-data.filename` uses the keyword analyzer, so every chunk scores an
identical 1.1211 and a constant offset cannot reorder anything. It is harmless
by accident: analyze that field with `en` instead and the offset varies per
document, silently perturbing the leg that is supposed to represent pure
vector rank.
"""
from .. import config
from .. import trace
from ..couchbase_io import query

RRF_K = config.RRF_K
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
        knn_filter = f', "filter": {scope}'
    else:
        lexical_query = lexical
        knn_filter = ""

    collection = f"`{config.BUCKET}`.`{config.SCOPE}`.`{config.DOCS_COLLECTION}`"
    # The index name MUST be fully qualified; the bare short name does not
    # resolve and fails with "no search index".
    index = f'{{"index": "{config.FTS_DOCS_INDEX}"}}'
    knn = (f'"knn": [{{"field": "text-embedding", "vector": $query_vector, '
           f'"k": {knn_k}{knn_filter}}}]')

    statement = f"""SELECT RAW bm25 FROM (
  {SELECT_FIELDS}, "bm25" AS channel
  FROM {collection} AS d
  WHERE SEARCH(d, {{"query": {lexical_query}}}, {index})
  ORDER BY SEARCH_SCORE() DESC LIMIT {LEG_CANDIDATES}
) AS bm25
UNION ALL
SELECT RAW vector FROM (
  {SELECT_FIELDS}, "vector" AS channel
  FROM {collection} AS d
  WHERE SEARCH(d, {{{knn}}}, {index})
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


def build_native_statement(question: str, embedding: list, doc_name: str = None,
                          anchors: list = None, top_k: int = config.TOP_K,
                          knn_k: int = config.KNN_CANDIDATES, title_boost: float = 0.0,
                          concept: str = None, strategy: str = "rrf",
                          weights: dict = None) -> tuple:
    """One SEARCH() with the Search service doing the fusion.

    Weights are expressed as each query's TOP-LEVEL boost, which is how the
    server reads channel importance - verified: a lexical boost of 5 multiplies
    that channel's contribution by 5 (5/61 = 0.08197 at rank 1).
    """
    weights = weights or {}
    params = {"$query_vector": embedding}
    lexical = _lexical_clause(question, anchors, concept, title_boost, params)

    knn_filter = ""
    if doc_name:
        params["$filename"] = config.source_filename(doc_name)
        scope = '{"field": "xmeta-data.filename", "match": $filename}'
        query_clause = f'{{"conjuncts": [{scope}, {lexical}]'
        knn_filter = f', "filter": {scope}'
    else:
        query_clause = lexical[:-1] if lexical.endswith("}") else lexical
        query_clause = '{"disjuncts": [' + lexical.split('[', 1)[1].rsplit(']', 1)[0] + ']'
    bm25_weight = weights.get("bm25")
    query_clause += (f', "boost": {bm25_weight}}}' if bm25_weight is not None else "}")

    vector_weight = weights.get("vector")
    knn = (f'"knn": [{{"field": "text-embedding", "vector": $query_vector, '
           f'"k": {knn_k}{knn_filter}'
           + (f', "boost": {vector_weight}' if vector_weight is not None else "")
           + "}]")

    statement = (
        SELECT_FIELDS + "\n"
        + f"FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.DOCS_COLLECTION}` AS d\n"
        + "WHERE SEARCH(d, {"
        + f'"score": "{strategy}", '
        + f'"params": {{"score_rank_constant": {config.NATIVE_RANK_CONSTANT}, '
          f'"score_window_size": {max(config.NATIVE_WINDOW_SIZE, top_k)}}}, '
        + f'"query": {query_clause}, {knn}, "size": {top_k}'
        + f'}}, {{"index": "{config.FTS_DOCS_INDEX}"}})\n'
        + f"ORDER BY SEARCH_SCORE() DESC LIMIT {top_k}"
    )
    return statement, params


def rrf_merge(rows: list, top_k: int = config.TOP_K, k: int = RRF_K,
              weights: dict = None) -> list:
    """Reciprocal rank fusion over the channels present in `rows`.

    Rank is derived here rather than trusted from row order: UNION ALL makes no
    ordering guarantee across branches, so each channel is re-sorted by its own
    score before ranking. Only ranks are compared, never raw scores from
    different channels - that comparison is exactly what broke score fusion.

    Each returned chunk carries the full arithmetic: per-channel rank, the raw
    score it came from, and that channel's contribution. Retrieval ranking is
    otherwise the least inspectable stage in the pipeline, and a fused number
    with no derivation is not something anyone can check.
    """
    weights = weights or {"bm25": config.RRF_BM25_WEIGHT,
                          "vector": config.RRF_VECTOR_WEIGHT}
    channels = {}
    for row in rows:
        channels.setdefault(row.get("channel", "unknown"), []).append(row)

    fused = {}
    for name, channel_rows in channels.items():
        channel_rows.sort(key=lambda r: r.get("score") or 0, reverse=True)
        weight = weights.get(name, 1.0)
        for rank, row in enumerate(channel_rows, 1):
            contribution = weight / (k + rank)
            entry = fused.setdefault(row.get("id"),
                                     {"row": row, "rrf_score": 0.0, "channels": {}})
            entry["rrf_score"] += contribution
            entry["channels"][name] = {
                "rank": rank,
                "raw_score": round(row.get("score") or 0.0, 6),
                "contribution": round(contribution, 6),
            }

    ordered = sorted(fused.values(), key=lambda e: e["rrf_score"], reverse=True)
    out = []
    for entry in ordered[:top_k]:
        chunk = dict(entry["row"])
        chunk["rrf_score"] = round(entry["rrf_score"], 6)
        chunk["channels"] = entry["channels"]
        # `channel` on a fused chunk would name whichever branch was read last,
        # which is meaningless once both contributed.
        chunk["channel"] = "+".join(sorted(entry["channels"]))
        out.append(chunk)

    trace.add("rrf_fusion", rank_constant=k, weights=weights,
              candidates=len(fused), returned=len(out),
              ranking=[{"final_rank": i, "page": c.get("page"), "type": c.get("type"),
                        "rrf_score": c["rrf_score"],
                        "channels": {n: v["rank"] for n, v in c["channels"].items()}}
                       for i, c in enumerate(out, 1)])
    return out


def hybrid_search(question: str, embedding: list, doc_name: str = None,
                  anchors: list = None, top_k: int = config.TOP_K,
                  knn_k: int = config.KNN_CANDIDATES, title_boost: float = 0.0,
                  concept: str = None, fusion: str = None,
                  weights: dict = None) -> list:
    """One statement either way. Native fusion returns a single blended
    SEARCH_SCORE(); RRF returns the per-channel derivation as well."""
    mode = fusion or config.HYBRID_FUSION
    if mode in config.NATIVE_STRATEGIES:
        statement, params = build_native_statement(
            question, embedding, doc_name, anchors, top_k, knn_k, title_boost,
            concept, strategy=config.NATIVE_STRATEGIES[mode], weights=weights)
        return query(statement, params)
    if mode == "score":
        statement, params = build_fused_statement(
            question, embedding, doc_name, anchors, top_k, knn_k, title_boost, concept)
        return query(statement, params)

    statement, params = build_statement(
        question, embedding, doc_name, anchors, top_k, knn_k, title_boost, concept)
    return rrf_merge(query(statement, params), top_k=top_k, weights=weights)
