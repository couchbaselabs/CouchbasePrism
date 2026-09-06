"""find_pdfs is pure filesystem logic - tested directly. upload_pdfs' own
logic (key construction, progress reporting, partial failure) is tested
against a fake S3 client, never a real boto3.Session - this suite stays
cluster/network-free like the rest of it (see conftest.py)."""
import pathlib

import pytest

from prism import s3_upload


def _make_pdfs(root: pathlib.Path, layout: dict) -> None:
    for company, filenames in layout.items():
        d = root / company
        d.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            (d / name).write_bytes(b"%PDF-1.4 fake")


def test_find_pdfs_pairs_company_from_parent_folder(tmp_path):
    _make_pdfs(tmp_path, {"3M": ["3M_2022_10K.pdf"], "BestBuy": ["bby_2023_10K.pdf"]})
    pairs = s3_upload.find_pdfs(tmp_path)
    assert sorted((c, p.name) for c, p in pairs) == [
        ("3M", "3M_2022_10K.pdf"), ("BestBuy", "bby_2023_10K.pdf"),
    ]


def test_find_pdfs_ignores_non_pdf_files(tmp_path):
    _make_pdfs(tmp_path, {"3M": ["3M_2022_10K.pdf"]})
    (tmp_path / "3M" / "notes.txt").write_text("not a pdf")
    pairs = s3_upload.find_pdfs(tmp_path)
    assert [p.name for _, p in pairs] == ["3M_2022_10K.pdf"]


class _FakeS3:
    def __init__(self, fail_on: set = frozenset()):
        self.fail_on = fail_on
        self.calls = []

    def upload_file(self, path, bucket, key):
        self.calls.append((path, bucket, key))
        if pathlib.Path(path).name in self.fail_on:
            raise RuntimeError("simulated upload failure")


class _FakeSession:
    def __init__(self, fake_client, **kwargs):
        self.fake_client = fake_client
        self.kwargs = kwargs  # captured so a test can assert creds were passed through

    def client(self, service_name):
        assert service_name == "s3"
        return self.fake_client


def test_upload_pdfs_builds_company_prefixed_keys(tmp_path, monkeypatch):
    _make_pdfs(tmp_path, {"3M": ["3M_2022_10K.pdf", "3M_2022_Q3_10Q.pdf"]})
    fake = _FakeS3()
    monkeypatch.setattr(s3_upload.boto3, "Session",
                        lambda **kw: _FakeSession(fake, **kw))

    result = s3_upload.upload_pdfs(tmp_path, bucket="my-bucket", prefix="filings",
                                   access_key="AKIA...", secret_key="secret",
                                   region="us-west-2")

    assert len(result["uploaded"]) == 2
    assert not result["failed"]
    keys = {c[2] for c in fake.calls}
    assert keys == {"filings/3M/3M_2022_10K.pdf", "filings/3M/3M_2022_Q3_10Q.pdf"}


def test_upload_pdfs_strips_leading_and_trailing_slashes_from_prefix(tmp_path, monkeypatch):
    _make_pdfs(tmp_path, {"3M": ["3M_2022_10K.pdf"]})
    fake = _FakeS3()
    monkeypatch.setattr(s3_upload.boto3, "Session",
                        lambda **kw: _FakeSession(fake, **kw))

    s3_upload.upload_pdfs(tmp_path, bucket="b", prefix="/filings/",
                         access_key="k", secret_key="s", region="us-west-2")

    assert fake.calls[0][2] == "filings/3M/3M_2022_10K.pdf"


def test_upload_pdfs_empty_prefix_omits_leading_slash(tmp_path, monkeypatch):
    _make_pdfs(tmp_path, {"3M": ["3M_2022_10K.pdf"]})
    fake = _FakeS3()
    monkeypatch.setattr(s3_upload.boto3, "Session",
                        lambda **kw: _FakeSession(fake, **kw))

    s3_upload.upload_pdfs(tmp_path, bucket="b", prefix="",
                         access_key="k", secret_key="s", region="us-west-2")

    assert fake.calls[0][2] == "3M/3M_2022_10K.pdf"


def test_upload_pdfs_records_partial_failure_without_stopping(tmp_path, monkeypatch):
    _make_pdfs(tmp_path, {"3M": ["ok.pdf", "bad.pdf"]})
    fake = _FakeS3(fail_on={"bad.pdf"})
    monkeypatch.setattr(s3_upload.boto3, "Session",
                        lambda **kw: _FakeSession(fake, **kw))

    result = s3_upload.upload_pdfs(tmp_path, bucket="b", prefix="p",
                                   access_key="k", secret_key="s", region="us-west-2")

    assert [u["filename"] for u in result["uploaded"]] == ["ok.pdf"]
    assert len(result["failed"]) == 1
    assert result["failed"][0]["filename"] == "bad.pdf"
    assert "simulated upload failure" in result["failed"][0]["error"]


def test_upload_pdfs_reports_progress_per_file(tmp_path, monkeypatch):
    _make_pdfs(tmp_path, {"3M": ["a.pdf", "b.pdf"]})
    fake = _FakeS3()
    monkeypatch.setattr(s3_upload.boto3, "Session",
                        lambda **kw: _FakeSession(fake, **kw))
    seen = []

    s3_upload.upload_pdfs(tmp_path, bucket="b", prefix="p",
                         access_key="k", secret_key="s", region="us-west-2",
                         on_progress=lambda i, total, company, filename, ok, error=None:
                             seen.append((i, total, filename, ok)))

    assert seen == [(1, 2, "a.pdf", True), (2, 2, "b.pdf", True)]


def test_upload_pdfs_passes_credentials_to_session_not_environment(tmp_path, monkeypatch):
    """Credentials must flow through as explicit call arguments, never fall
    back to boto3's default environment/~/.aws credential chain - that's the
    whole point of collecting them in the UI instead of a config file."""
    _make_pdfs(tmp_path, {"3M": ["a.pdf"]})
    fake = _FakeS3()
    captured = {}

    def fake_session_ctor(**kw):
        captured.update(kw)
        return _FakeSession(fake, **kw)

    monkeypatch.setattr(s3_upload.boto3, "Session", fake_session_ctor)

    s3_upload.upload_pdfs(tmp_path, bucket="b", prefix="p",
                         access_key="AKIA_TEST", secret_key="secret_test",
                         region="us-west-2", session_token="token_test")

    assert captured["aws_access_key_id"] == "AKIA_TEST"
    assert captured["aws_secret_access_key"] == "secret_test"
    assert captured["aws_session_token"] == "token_test"
    assert captured["region_name"] == "us-west-2"
