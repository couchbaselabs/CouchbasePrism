"""query()'s error-surfacing logic, against a faked requests.post - no live
cluster, same monkeypatching pattern as test_s3_upload.py's fake boto3
client. This is specifically about the bug hit live on 2026-09-07: a missing
primary index came back as HTTP 404 with a perfectly informative Couchbase
error body, and raise_for_status() firing before the body was ever read threw
that detail away, surfacing only a bare "404 Client Error: Not Found"."""
import json

import pytest

from prism import couchbase_io


class _FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body) if body is not None else "not json"

    def json(self):
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise couchbase_io.requests.exceptions.HTTPError(
                f"{self.status_code} Client Error: fake for url: x")


def test_success_response_is_unaffected(monkeypatch):
    monkeypatch.setattr(couchbase_io.requests, "post", lambda *a, **k: _FakeResponse(
        200, {"status": "success", "results": [{"a": 1}], "metrics": {}}))
    assert couchbase_io.query("SELECT 1") == [{"a": 1}]


def test_structured_error_body_surfaces_even_on_a_404(monkeypatch):
    # The exact live shape: HTTP 404, but a JSON body with the real Couchbase
    # error - a missing primary index.
    monkeypatch.setattr(couchbase_io.requests, "post", lambda *a, **k: _FakeResponse(
        404, {"status": "fatal",
              "errors": [{"code": 4000, "msg": "No index available on "
                         "keyspace `default`:`acme`.`prism`.`catalog`..."}]}))
    with pytest.raises(couchbase_io.QueryError) as exc_info:
        couchbase_io.query("SELECT 1 FROM catalog")
    assert "No index available" in str(exc_info.value)


def test_structured_error_body_surfaces_on_a_200_too(monkeypatch):
    # Couchbase itself can return status: fatal with a 200 - the body is what
    # matters, not the HTTP status either way.
    monkeypatch.setattr(couchbase_io.requests, "post", lambda *a, **k: _FakeResponse(
        200, {"status": "fatal", "errors": [{"code": 5000, "msg": "syntax error"}]}))
    with pytest.raises(couchbase_io.QueryError) as exc_info:
        couchbase_io.query("SELECT ((")
    assert "syntax error" in str(exc_info.value)


def test_non_json_failure_falls_back_to_http_status(monkeypatch):
    # No JSON body at all (e.g. a proxy/network failure upstream of Couchbase
    # itself) - there is no structured error to prefer, so the HTTP status is
    # the only signal available.
    monkeypatch.setattr(couchbase_io.requests, "post", lambda *a, **k: _FakeResponse(
        502, None))
    with pytest.raises(couchbase_io.requests.exceptions.HTTPError):
        couchbase_io.query("SELECT 1")
