"""Content-anchor retrieval: find the chunk whose own text contains the printed
row labels the evidence plan asked for.

Never consults `meta-data.associated-titles`. That field is wrong on a
meaningful fraction of table chunks - confirmed three times in the sample
corpus, where a debt-securities table is titled ["Delaware", "41-0417775"] and
a Consumer segment table is titled "PERFORMANCE BY GEOGRAPHIC AREA". A table's
own row labels are a far more reliable fingerprint than its assigned title.
"""
import math

from .. import config
from ..couchbase_io import query


def anchor_search(anchors: list, doc_name: str,
                  limit: int = config.MAX_ANCHOR_CHUNKS) -> list:
    """Ranked by anchor RARITY, not raw hit count.

    The planner reliably emits a plausible-but-invented label alongside the
    correct ones. Under raw counting, two junk anchors matching a wrong table
    outrank one precise anchor matching the right one - observed exactly that,
    where Fair Value tables buried the actual securities table. Weighting each
    anchor by 1/log2(2+df), df being how many chunks in this document contain
    it, lets a rare specific anchor dominate several generic ones. This is IDF,
    for the same reason BM25 uses it.
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
    for row in rows:
        for anchor in row.get("matched_anchors") or []:
            doc_freq[anchor] = doc_freq.get(anchor, 0) + 1
    for row in rows:
        matched = row.get("matched_anchors") or []
        row["anchor_hits"] = len(matched)
        row["anchor_score"] = round(
            sum(1.0 / math.log2(2 + doc_freq.get(a, 0)) for a in matched), 4)
    rows.sort(key=lambda r: r["anchor_score"], reverse=True)
    return rows[:limit]
