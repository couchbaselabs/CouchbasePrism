"""Stages local PDFs into S3 for the Couchbase AI Data Plane workflow to
ingest - the step that used to be a manual AWS console upload.

Deliberately minimal: a straight upload, no S3 object metadata (no
version_tag/effective_date/status tagging - compare edgarDownloader's
upload scripts, which use those for a different, ongoing-versioning use
case this one doesn't need). Uploads flat into ONE folder - no per-company
subfolder in S3, even though the local layout has one (pdfs/3M/, pdfs/
BestBuy/, ...) purely for local organization. The company is already encoded
in each filename (3M_2022_10K.pdf), and a real ingested chunk's
`xmeta-data.filename` confirms the workflow expects exactly this flat shape:
"kpd-couchbase_Prism_3M_3M_2021_Q2_10Q.pdf" - one folder segment ("Prism_3M"
from AWS_FOLDER), not a second one per company.

The destination folder is config.aws_folder()/AWS_BUCKET - read from .env,
the same values config.source_filename() uses to map a catalog entry back to
its chunks. That's the whole point: one place those names live, not the
upload form re-typed separately from what retrieval expects.

Credentials are parameters, never read from ~/.aws/credentials or an
environment variable - the caller (the Setup tab) collects them from a
password-masked form field and passes them straight through for the
duration of one upload call. Nothing here writes them anywhere.
"""
import os
import pathlib

import boto3


def find_pdfs(local_root: pathlib.Path) -> list:
    """(company, path) pairs for every PDF under local_root/{company}/*.pdf -
    company is kept for progress reporting only; the actual upload key
    (below) does not use it."""
    return sorted(
        (p.parent.name, p) for p in local_root.rglob("*.pdf")
    )


def upload_pdfs(local_root: pathlib.Path, access_key: str, secret_key: str,
                session_token: str = None, on_progress=None) -> dict:
    """Uploads every PDF under local_root to
    s3://{AWS_BUCKET}/{AWS_FOLDER}/{filename} - AWS_BUCKET/AWS_FOLDER/
    AWS_REGION come from .env (config.py), not from the caller, so this is
    always the same destination config.source_filename() expects. Straight
    upload_file, no ExtraArgs - the destination is exactly what the Couchbase
    AI Data Plane workflow reads from, and it doesn't need any tagging to do
    that.

    on_progress(i, total, company, filename, ok, error=None) fires after each
    file, so a caller can render a progress bar without depending on this
    function's internals.

    Returns {"uploaded": [...], "failed": [...]} - both lists of
    {"company", "filename", "key"} (plus "error" for failed).
    """
    session = boto3.Session(
        aws_access_key_id=access_key, aws_secret_access_key=secret_key,
        aws_session_token=session_token, region_name=os.environ.get("AWS_REGION"),
    )
    s3 = session.client("s3")

    bucket = os.environ["AWS_BUCKET"]
    folder = os.environ["AWS_FOLDER"].strip("/")
    pairs = find_pdfs(local_root)
    uploaded, failed = [], []
    for i, (company, path) in enumerate(pairs, 1):
        key = f"{folder}/{path.name}" if folder else path.name
        try:
            s3.upload_file(str(path), bucket, key)
            uploaded.append({"company": company, "filename": path.name, "key": key})
            if on_progress:
                on_progress(i, len(pairs), company, path.name, True)
        except Exception as e:
            failed.append({"company": company, "filename": path.name, "key": key,
                           "error": str(e)})
            if on_progress:
                on_progress(i, len(pairs), company, path.name, False, error=str(e))

    return {"uploaded": uploaded, "failed": failed}
