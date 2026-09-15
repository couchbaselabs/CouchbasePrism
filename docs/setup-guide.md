# Setup guide: from a blank Capella project to a running PRISM

This is the full walkthrough behind the README's terse "fill in Capella, AI
Data Plane, AWS, OpenAI, domains" line - follow it once, end to end, before
your first `Initialize`. Every "Copy" callout below is a value you'll paste
into `config.yaml` or a workflow form a few steps later; the example values
shown (`acme`, `AKIA...`, `cb.p8x...`) are placeholders, not real credentials.

## Prerequisites

### 1. Capella cluster

Create (or use an existing) Capella cluster, then:

```
bucket:  prism
  scope: secfilings
    collections: docs, catalog, dictionary, concepts
```

Create a database user with read/write access to all buckets, scopes, and
collections (RBAC), and allow IP access from anywhere (or your own IP range,
if you'd rather scope it down).

Copy (example):
- **Connection endpoint**: Capella's own "Public Connection String" -
  `couchbases://cb.p8x********.cloud.couchbase.com` - paste it into
  `config.yaml`'s `couchbase.connectionString` exactly as Capella gives it
  to you. The `couchbases://` prefix is optional either way: this codebase
  talks to Capella over REST, not the native SDK, so `config.couchbase_host()`
  strips the prefix if present before building its own `https://` URLs -
  paste with or without it, both work.
- **User access credentials** (username/password)

### 2. AWS S3

Create an S3 bucket and a folder inside it to stage the sample PDFs for
ingestion:

- Bucket (example): `acme`
- Folder (example): `Prism/3M`

Copy (example):
- **Bucket**: `acme` → `config.yaml`'s `aws.bucket`
- **Region**: `us-west-2` → `config.yaml`'s `aws.region`
- **Folder**: `Prism/3M` → `config.yaml`'s `domains.secfilings.awsFolder`.
  Keep the `3M` subfolder exactly as shown if you're using the packaged
  sample corpus (`eval/corpora/ftsprism/pdfs/3M/`) - the AI Data Plane
  workflow's own ingested filenames encode this folder path
  (`prism/s3_upload.py`'s own docstring shows the exact shape:
  `..._Prism_3M_3M_2021_Q2_10Q.pdf`, where the first `3M` is this folder,
  the second is the PDF's own filename), so a mismatch here means retrieval
  can't map an ingested chunk back to its catalog entry. A different corpus
  would use a different folder - nothing in the code requires "3M"
  specifically, only that this value stays consistent across S3, the
  workflow, and `config.yaml`.
- **AWS access key ID** and **AWS secret access key** for a user with
  read/write on this bucket.

### 3. Upload the sample PDFs to S3

Once `config.yaml` has the bucket/folder above filled in, use the app's
Setup tab (see [README](../README.md#setup) for getting the app running) to
upload the sample PDFs - enter the access key ID/secret there and it stages
them into the S3 bucket/folder for you.

Notes:
- The bucket name and folder are read from `config.yaml` and aren't
  editable in that form - change them in `config.yaml` if you need a
  different destination.
- One-time step - you don't need to re-upload once the workflow below has
  ingested them.
- This S3 bucket/folder is what the AI Data Plane Workflow (next section)
  points at.

## AI Data Plane Workflow

### Deploy a model

You need a text-to-embedding model deployed. Example:
`nvidia/llama-3.2-nv-embedqa-1b-v2`.

1. **Name**: your choice.
2. **Region**: the same region as your Capella cluster.
3. **Compute and size**: your choice.
4. **Advanced configuration**: your choice - leave blank for a demo.

Once the model shows healthy, expand its row (the down-arrow in the UI) and
copy:
- **Model ID** (example: `a16def04-0345-*******-*****`)
- **Model endpoint** (example: `https://itwg*******-cw.ai.cloud.couchbase.com`)

Both go into `config.yaml`'s `aiDataPlane.modelId` / `aiDataPlane.modelEndpoint`.

### Generate API keys for the model

1. **API key name**: your choice.
2. **Expiration days**: default (180) is fine.
3. **Region**: same as the model's.
4. **Description**: your choice.
5. **Allowed IP address**: allow access from anywhere.
6. **CIDR block**: `0.0.0.0/0`.

Generate the key and copy:
- **API key ID** (example: `ec7186ee-a359********`) - keep this, but note it
  is NOT what goes in `config.yaml`.
- **API key token** (example: `cbsk-v1_*******`) - this is what goes in
  `config.yaml`'s `aiDataPlane.apiKey`. Using the key ID instead of the
  token produces a 401 that reads like a bad endpoint, not a bad key - a
  real, easy-to-make mistake worth double-checking.

### Create the workflow

1. Choose **Unstructured Data from External Sources**.
2. **Workflow name**: your choice.

### Configure the workflow

**Data source** - add a new S3 bucket integration:
1. **Integration name**: your choice.
2. **Bucket**: `acme` (from `config.yaml`'s `aws.bucket`).
3. **Folder path**: `Prism/3M` (from `config.yaml`'s
   `domains.secfilings.awsFolder`).
4. **Access key ID** / **Secret access key**: the same S3 credentials from
   above.

**Destination**:
1. **Extract S3 metadata fields**: unchecked.
2. **Destination cluster**: your Capella cluster.
3. **Destination bucket**: `prism`.
4. **Destination scope**: `secfilings`.
5. **Destination collection**: `docs`.

**Document processing** (next page):
1. **Include page range**: unchecked.
2. **Exclude tables**: unchecked.
3. **Exclude footer**: checked.
4. **Exclude header**: unchecked.
5. **Enable OCR**: unchecked.

**Chunking strategy**:
- **Strategy**: `RECURSIVE_SPLITTER`.
- **Max token**: `2000`.
- **Chunk overlap**: `50`.

**Embedding model** (next page):
1. **Model source**: Capella model.
2. Select the embedding model you deployed above.
3. **API key ID** / **API key token**: from the step above.

Deploy the workflow and wait for it to complete - it should ingest all 64
sample documents successfully.

## Then: Initialize

Once ingestion finishes, run **Initialize** - the Setup tab's own button, or
`manage.py initialize`. This is the one remaining step, and it's fully
automatic: it rebuilds the docs and catalog search indexes, rebuilds the
catalog from the ingested chunks, and seeds the dictionary and concepts
collections from `dictionary.yaml`/`concepts.yaml` - all from this one
ingested corpus, no manual data entry. Destructive-but-safe to re-run: each
step is a full rebuild/reseed, not an incremental update, so running it
again always leaves the same, known-good state. After it completes you're
ready to ask questions - nothing else to configure.
