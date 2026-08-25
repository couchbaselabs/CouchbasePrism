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
import re

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


def _merged_terms(concept: str, anchors: list) -> str:
    """Concept + every anchor, lowercased, hyphens/slashes/punctuation split
    into word breaks, deduped, and joined into one bag of words for a single
    OR match - rather than requiring each anchor to appear as a verbatim
    phrase, which most of the planner's anchors never do."""
    seen = []
    for text in [concept or ""] + list(anchors or []):
        cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
        for word in cleaned.split():
            if word not in seen:
                seen.append(word)
    return " ".join(seen)


def _lexical_clause(question: str, anchors: list, concept: str,
                    title_boost: float, params: dict) -> str:
    """The disjuncts BM25 scores against. Falls back to the question text only
    when there is nothing better, so the leg never drops out entirely."""
    disjuncts = []
    terms = _merged_terms(concept, anchors)
    if terms:
        disjuncts.append('{"match": $terms, "field": "text-to-embed", '
                         '"operator": "or"}')
        params["$terms"] = terms
    # Tried adding match_phrase per anchor here to give "Total assets" and
    # "Net income" exact-phrase precision over the bag match's loose word
    # overlap. Measured worse, not better, on financebench_id_10420: those
    # captions are printed verbatim on a dozen pages of a real 10-K (a parent-
    # only Schedule I balance sheet, segment tables, 5-year Selected Financial
    # Data) - phrase-matching found MORE pages that legitimately say "Total
    # assets", not the right one, and some outranked the actual consolidated
    # balance sheet (fused rank 8 -> 16). The problem isn't phrase vs bag
    # matching; it's that BM25 alone cannot tell which of several genuine
    # occurrences is the consolidated statement for the specific years asked
    # about. Left as bag-only; see financebench_id_10420 in debug history.
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
                          title_boost: float = 0.0, concept: str = None,
                          weights: dict = None) -> tuple:
    """The ORIGINAL single fused SEARCH(): lexical and kNN in one query object,
    scores summed by the Search service. Retained as a fallback - see the module
    docstring for why it is not the default. Reachable via
    config.HYBRID_FUSION = "score"."""
    weights = weights or {}
    params = {"$query_vector": embedding}
    lexical = _lexical_clause(question, anchors, concept, title_boost, params)
    knn_filter = ""
    if doc_name:
        params["$filename"] = config.source_filename(doc_name)
        scope = '{"field": "xmeta-data.filename", "match": $filename}'
        query_clause = (f'{{"conjuncts": [{scope}, {lexical}], '
                        f'"boost": {weights.get("bm25", 1.0)}}}')
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


# The Search service explains its own fusion when `explain` is on:
#   "rrf score (weight=1.000, rank=1, rank_constant=60), normalized score of"
# That is the same arithmetic rrf_merge reports for the in-code path, so parsing
# it lets both paths show a reader per-channel rank and contribution rather than
# one opaque number.
FUSION_NODE = re.compile(
    r"(rrf|rsf) score \(weight=([\d.]+)"
    r"(?:,\s*rank=(\d+))?"
    r"(?:,\s*rank_constant=(\d+))?", re.I)
VECTOR_FIELD = "text-embedding"


def _channel_of(node: dict) -> str:
    """Which channel a fusion node came from, read from the scoring beneath it."""
    stack = [node]
    while stack:
        current = stack.pop()
        if VECTOR_FIELD in str(current.get("message", "")):
            return "vector"
        stack.extend(current.get("children") or [])
    return "bm25"


def parse_explanation(explanation: dict) -> dict:
    """{channel: {rank, weight, raw_score, contribution}} from the server's own
    explanation. Returns {} when fusion is off, since the additive default has no
    per-channel structure to report."""
    if not isinstance(explanation, dict):
        return {}
    channels, stack = {}, [explanation]
    while stack:
        node = stack.pop()
        match = FUSION_NODE.search(str(node.get("message", "")))
        if match:
            _, weight, rank, constant = match.groups()
            child = (node.get("children") or [{}])[0]
            # Relative score fusion has no ranks - it normalises scores. Emitting
            # "rank": null invites the reader to wonder what went wrong, so the
            # key is simply absent when the strategy does not use one.
            channel = {"weight": float(weight),
                       "raw_score": round(child.get("value") or 0.0, 6),
                       "contribution": round(node.get("value") or 0.0, 6)}
            if rank:
                channel["rank"] = int(rank)
            if constant:
                channel["rank_constant"] = int(constant)
            channels[_channel_of(node)] = channel
        stack.extend(node.get("children") or [])
    return channels


def build_native_statement(question: str, embedding: list, doc_name: str = None,
                          anchors: list = None, top_k: int = config.TOP_K,
                          knn_k: int = config.KNN_CANDIDATES, title_boost: float = 0.0,
                          concept: str = None, strategy: str = "rrf",
                          weights: dict = None, rank_constant: int = None,
                          window_size: int = None) -> tuple:
    """One SEARCH() with the Search service doing the fusion.

    Weights are expressed as each query's TOP-LEVEL boost, which is how the
    server reads channel importance - verified: a lexical boost of 5 multiplies
    that channel's contribution by 5 (5/61 = 0.08197 at rank 1).
    """
    # The boost is ALWAYS written, even at 1.0. It is how channel weight is
    # expressed, and a statement that omits it at the default hides where the
    # weight would go - which is the one thing a reader of this SQL wants to see.
    weights = weights or {}
    bm25_weight = weights.get("bm25", 1.0)
    vector_weight = weights.get("vector", 1.0)
    params = {"$query_vector": embedding}
    lexical = _lexical_clause(question, anchors, concept, title_boost, params)

    knn_filter = ""
    if doc_name:
        params["$filename"] = config.source_filename(doc_name)
        scope = '{"field": "xmeta-data.filename", "match": $filename}'
        query_clause = f'{{"conjuncts": [{scope}, {lexical}]'
        knn_filter = f', "filter": {scope}'
    else:
        query_clause = '{"disjuncts": [' + lexical.split('[', 1)[1].rsplit(']', 1)[0] + ']'
    query_clause += f', "boost": {bm25_weight}}}'

    knn = (f'"knn": [{{"field": "text-embedding", "vector": $query_vector, '
           f'"k": {knn_k}{knn_filter}, "boost": {vector_weight}}}]')

    constant = config.NATIVE_RANK_CONSTANT if rank_constant is None else rank_constant
    window = max(window_size or config.NATIVE_WINDOW_SIZE, top_k)
    statement = (
        SELECT_FIELDS + ",\n           SEARCH_META(d) AS meta\n"
        + f"FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.DOCS_COLLECTION}` AS d\n"
        + "WHERE SEARCH(d, {"
        + f'"score": "{strategy}", "explain": true, '
        + f'"params": {{"score_rank_constant": {constant}, '
          f'"score_window_size": {window}}}, '
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
                  weights: dict = None, rank_constant: int = None,
                  window_size: int = None) -> list:
    """One statement either way. Native fusion returns a single blended
    SEARCH_SCORE(); RRF returns the per-channel derivation as well."""
    mode = fusion or config.HYBRID_FUSION
    if mode in config.NATIVE_STRATEGIES:
        statement, params = build_native_statement(
            question, embedding, doc_name, anchors, top_k, knn_k, title_boost,
            concept, strategy=config.NATIVE_STRATEGIES[mode], weights=weights,
            rank_constant=rank_constant, window_size=window_size)
        rows = query(statement, params)
        for row in rows:
            channels = parse_explanation((row.pop("meta", None) or {}).get("explanation"))
            if channels:
                row["channels"] = channels
                row["channel"] = "+".join(sorted(channels))
        return rows
    if mode == "score":
        statement, params = build_fused_statement(
            question, embedding, doc_name, anchors, top_k, knn_k, title_boost,
            concept, weights=weights)
        return query(statement, params)

    statement, params = build_statement(
        question, embedding, doc_name, anchors, top_k, knn_k, title_boost, concept)
    return rrf_merge(query(statement, params), top_k=top_k, weights=weights,
                     k=rank_constant if rank_constant is not None else RRF_K)
