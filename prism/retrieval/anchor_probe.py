"""Checking anchors against the corpus before spending retrieval budget on them.

An anchor is the planner's *guess* at how a label is printed. Measured across
the 3M subset, 62% of generated anchors match no chunk at all - the planner
asks for "total revenue" where 3M prints "Net sales", and one question produced
three dead anchors on every attempt. A dead anchor is not a weak signal, it is
no signal: it contributes nothing to BM25 and silently consumes a slot.

Whether an anchor matches is a fact about the corpus, not a judgment, so it is
answered by counting rather than by asking a model. One statement covers every
anchor via conditional aggregation - the alternative, a query per anchor, costs
a round trip each to learn the same thing.

Knowing genre does not help here and was measured: telling the planner it was
reading a 10-K moved dead anchors from 62% to 66%. The model already infers the
genre from the question. What it cannot infer is which words this particular
issuer prints, and only the corpus knows that.
"""
from .. import config, trace
from ..couchbase_io import query


def probe_anchors(anchors: list, doc_name: str = None) -> dict:
    """Returns {anchor: matching chunk count}. Case-insensitive substring, which
    is deliberately looser than the `match_phrase` BM25 will run: the point is
    to identify anchors that cannot match under ANY analysis, so a false
    "alive" is safe and a false "dead" would not be."""
    anchors = [a for a in (anchors or []) if a]
    if not anchors:
        return {}

    params = {}
    projections = []
    for i, anchor in enumerate(anchors):
        params[f"$p{i}"] = f"%{anchor.lower()}%"
        projections.append(
            f"SUM(CASE WHEN LOWER(d.`text-to-embed`) LIKE $p{i} THEN 1 ELSE 0 END) AS a{i}")

    where = "d.`text-to-embed` IS NOT MISSING"
    if doc_name:
        params["$filename"] = config.source_filename(doc_name)
        where += " AND d.`xmeta-data`.`filename` = $filename"

    statement = (f"SELECT {', '.join(projections)}\n"
                 f"FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.DOCS_COLLECTION}` AS d\n"
                 f"WHERE {where}")
    rows = query(statement, params)
    counts = rows[0] if rows else {}
    return {anchor: int(counts.get(f"a{i}") or 0) for i, anchor in enumerate(anchors)}


def filter_anchors(anchors: list, doc_name: str = None) -> tuple:
    """Returns (live, dead, counts).

    Every anchor dead means the planner's whole vocabulary guess missed. In that
    case `live` is empty and the caller falls back to matching the question
    text, which is weak but is not nothing - dropping the lexical leg entirely
    would be strictly worse.
    """
    counts = probe_anchors(anchors, doc_name)
    live = [a for a in anchors or [] if counts.get(a, 0) > 0]
    dead = [a for a in anchors or [] if a and counts.get(a, 0) == 0]
    if counts:
        trace.add("anchor_probe", doc_name=doc_name, counts=counts,
                  live=live, dead=dead,
                  dead_fraction=round(len(dead) / max(len(counts), 1), 3))
    return live, dead, counts
