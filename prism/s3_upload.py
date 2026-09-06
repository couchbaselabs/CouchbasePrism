"""Stages local PDFs into S3 for the Couchbase AI Data Plane workflow to
ingest - the step that used to be a manual AWS console upload.

Deliberately minimal: a straight upload, no S3 object metadata (no
version_tag/effective_date/status tagging - compare edgarDownloader's
upload scripts, which use those for a different, ongoing-versioning use
case this one doesn't need). One subfolder per company preserved as-is from
the local layout, so `pdfs/3M/foo.pdf` becomes `s3://{bucket}/{prefix}/3M/foo.pdf`.

Credentials are parameters, never read from ~/.aws/credentials or an
environment variable - the caller (the Setup tab) collects them from a
password-masked form field and passes them straight through for the
duration of one upload call. Nothing here writes them anywhere.
"""
import pathlib

import boto3


def find_pdfs(local_root: pathlib.Path) -> list:
    """(company, path) pairs for every PDF under local_root/{company}/*.pdf,
    sorted for a stable, predictable upload and progress-bar order."""
    return sorted(
        (p.parent.name, p) for p in local_root.rglob("*.pdf")
    )


def upload_pdfs(local_root: pathlib.Path, bucket: str, prefix: str,
                access_key: str, secret_key: str, region: str,
                session_token: str = None, on_progress=None) -> dict:
    """Uploads every PDF under local_root (one subfolder per company) to
    s3://{bucket}/{prefix}/{company}/{filename}. Straight upload_file, no
    ExtraArgs - the destination is exactly what the Couchbase AI Data Plane
    workflow reads from, and it doesn't need any tagging to do that.

    on_progress(i, total, company, filename, ok, error=None) fires after each
    file, so a caller can render a progress bar without depending on this
    function's internals.

    Returns {"uploaded": [...], "failed": [...]} - both lists of
    {"company", "filename", "key"} (plus "error" for failed).
    """
    session = boto3.Session(
        aws_access_key_id=access_key, aws_secret_access_key=secret_key,
        aws_session_token=session_token, region_name=region,
    )
    s3 = session.client("s3")

    pairs = find_pdfs(local_root)
    prefix = prefix.strip("/")
    uploaded, failed = [], []
    for i, (company, path) in enumerate(pairs, 1):
        key = f"{prefix}/{company}/{path.name}" if prefix else f"{company}/{path.name}"
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
