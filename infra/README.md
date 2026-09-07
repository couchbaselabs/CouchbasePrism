# Rebuilding the cluster

Captured from the 8.0.1 cluster before it was replaced, so the 8.1 rebuild
matches rather than being reconstructed from memory. Keep every name identical:
`prism/config.py` derives `FTS_DOCS_INDEX` as `{BUCKET}.{SCOPE}.ftsPrism`,
so a rename means a code change.

    bucket      acme
    scope       prism
    collections docs, catalog
    FTS index   ftsPrism          (on acme.prism.docs)

## Order

1. Create bucket `acme`, scope `prism`, collections `docs` and `catalog`.
2. Run the AI Data Plane workflow into `acme.prism.docs` (see settings below).
3. Create the vector index, then the FTS index.
4. `COUCHBASE_SCOPE=prism python manage.py build-catalog --company 3M`
   (drop `--company` once the wider corpus is loaded).

Indexes go AFTER ingestion so they build once over a settled collection.

## Vector index

`gsi.vector-index.json` holds the captured definition. The statement:

```sql
CREATE VECTOR INDEX `hyperscale_prism_text-embedding`
ON `acme`.`prism`.`docs`(`text-embedding` VECTOR)
WITH { "defer_build": true, "num_replica": 1, "dimension": 2048,
       "similarity": "L2", "description": "IVF,SQ8", "scan_nprobes": 9 }
```

Then `BUILD INDEX ON `acme`.`prism`.`docs`(`hyperscale_prism_text-embedding`)`.

`dimension: 2048` must match the embedding model. Changing the model means
re-ingesting, not just reindexing.

**This index did not exist on `acme.prism.docs` in the 8.0.1 cluster** - only on
`acme.financebench.docs`. So `APPROX_VECTOR_DISTANCE` on the prism scope had no
vector index to use, which affects the `1-vector` and `2-catalog` phases. Create
it this time.

## FTS index

`fts.acme.prism.ftsPrism.json` is the captured definition. Four field
mappings matter, and two of them were bugs we already paid for once:

| field | setting | why |
|---|---|---|
| `text-to-embed` | `text`, analyzer `en`, **`include_term_vectors: true`** | BM25 matches anchors here. Term vectors are what make `match_phrase` work; without the field the lexical channel silently contributes nothing. |
| `xmeta-data.filename` | `text`, analyzer **`keyword`** | Document scoping. Under the `en` analyzer the filename tokenises and the scope predicate stops being selective, leaking results across filings. |
| `meta-data.associated-titles` | `text`, analyzer `en` | Indexed but unused for scoring - see `docs/findings/associated-titles-defect.md`. |
| `text-embedding` | `vector`, 2048 dims, `l2_norm`, optimized for `recall` | kNN channel. |

Also `scoring_model: bm25`, `doc_config.mode: scope.collection.type_field`, and
the type mapping keyed `prism.docs`.

## Ingestion workflow settings

    source            S3 kpd-couchbase / financebench, us-west-2
    destination       acme | prism | docs
    embedding model   MUST match MODEL_ID / MODEL_END_POINT in .env
    page range        all
    exclusions        Footer only
    chunking          RECURSIVE_SPLITTER, max tokens 2000, overlap 50
    S3 metadata       not extracted (nothing reads it)

Do not exclude Headers - section headers are needed even though table chunks
mis-attribute them. Do not exclude Tables; the answers are in them.

Document keys are random UUIDs, so re-running the workflow over the same PDFs
DUPLICATES rather than upserts. Ingest new files only, or clear the collection
first.

## Verify hybrid fusion is actually active on 8.1

On 8.0.1 the `score` field parsed and was ignored, which looks exactly like
success. Check all three before trusting it:

1. `"score": "bogus"` must be REJECTED. If accepted, fusion is not active.
2. `score_rank_constant` 1 vs 60 must change the scores.
3. With `"explain": true`, `SEARCH_META(d).explanation` should mention
   `RRF score (weight=..., window_size=..., rank_constant=...)`. Plain
   `"sum of:"` with `idf`/`fieldNorm` children means ordinary BM25.

Until all three pass, `config.HYBRID_FUSION = "rrf"` (the in-code merge) is the
one that actually fuses.

## Re-baseline afterwards

A fresh ingest changes chunk boundaries and ids. Recall figures, dead-anchor
rates and benchmark scores are not comparable across the rebuild. Re-run
`eval/compare_fusion.py` and the benchmark to establish new baselines before
comparing anything to them.
