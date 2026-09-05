"""Couchbase access over the Query REST service.

Deliberately REST rather than the SDK: PRISM's data access is a handful of
N1QL statements, and the REST surface keeps this dependency-free and makes the
exact statement inspectable, which matters for the demo (showing the SQL++ is
part of the story).

Two things that cost real debugging time and are easy to regress:
  - Reserved words (`value`, `committed`) must be backticked in projections.
  - Array fields need `ANY ... SATISFIES ... END`; `=` against an array is
    always false and fails silently.
"""
import copy
import time

import requests
import urllib3

from . import config, trace

# Capella's query node presents a chain the local trust store doesn't carry, so
# requests are made with verify=False (still TLS). Silence the resulting
# per-call warning once here rather than letting it flood every run.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class QueryError(RuntimeError):
    pass


def query(statement: str, params: dict = None, timeout: int = 60) -> list:
    body = {"statement": statement}
    if params:
        body.update(params)
    started = time.perf_counter()
    try:
        resp = requests.post(
            f"https://{config.couchbase_host()}:18093/query/service",
            auth=config.couchbase_auth(),
            json=body,
            # Capella's query node presents a cert chain the local trust store
            # doesn't carry; the connection is still TLS.
            verify=False, timeout=timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("status") != "success":
            raise QueryError(f"{result.get('errors')}\nstatement: {statement}")
        rows = result.get("results", [])
        trace.add("couchbase_query", statement=statement.strip(), params=params or {},
                  row_count=len(rows), status=result.get("status"),
                  metrics=result.get("metrics"),
                  elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
        return rows
    except Exception as exc:
        trace.add("couchbase_query", statement=statement.strip(), params=params or {},
                  error=f"{type(exc).__name__}: {exc}",
                  elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
        raise


def ensure_primary_index(collection: str) -> None:
    query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON "
          f"`{config.BUCKET}`.`{config.SCOPE}`.`{collection}`")


def _search_admin_url(index_name: str = "") -> str:
    base = (f"https://{config.couchbase_host()}:18094/api/bucket/"
            f"{config.BUCKET}/scope/{config.SCOPE}/index")
    return f"{base}/{index_name}" if index_name else base


def get_search_index_definition(index_name: str) -> dict:
    """The live index definition, scoped-index REST path. The flat
    /api/index/{name} path 400s ("index not found") for a scoped index -
    verified live rather than assumed when this was first written."""
    resp = requests.get(_search_admin_url(index_name), auth=config.couchbase_auth(),
                        verify=False, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "ok":
        raise QueryError(f"{body}\nindex: {index_name}")
    return body["indexDef"]


def delete_search_index(index_name: str) -> None:
    """Deleting a nonexistent index is not an error here - Initialize always
    deletes-then-recreates, so a fresh environment with no index yet must not
    fail this step. A missing SCOPED index is not a 404, though - verified
    live: it comes back 400 with "index not found" in the body, the same
    shape as any other malformed-request error, so that specific message is
    what's checked for rather than the status code alone."""
    resp = requests.delete(_search_admin_url(index_name), auth=config.couchbase_auth(),
                           verify=False, timeout=30)
    if resp.status_code == 400 and "index not found" in resp.text:
        return
    resp.raise_for_status()


def create_search_index(definition: dict) -> None:
    """PUTs a new index from a definition read back from a live index (e.g.
    design/fts-index.json) - which carries three identifiers the SERVER
    assigns and the client must not resupply when creating a fresh index:
    the index's own `uuid`, the source bucket's `sourceUUID`, and each
    scoped collection's `uid`. Verified live: submitting the definition with
    these stripped creates a working index the server assigns fresh IDs to;
    querying it returns results identical to the index it was copied from.

    sourceName and the scope name are overridden from config rather than
    trusted from the file, so a definition captured against one bucket/scope
    still creates correctly against whatever this environment is configured
    for.
    """
    body = copy.deepcopy(definition)
    body.pop("uuid", None)
    body.pop("sourceUUID", None)
    body["sourceName"] = config.BUCKET
    for collection in (body.get("sourceParams", {})
                          .get("scopeParams", {}).get("collections", [])):
        collection.pop("uid", None)
    body["sourceParams"]["scopeParams"]["name"] = config.SCOPE
    resp = requests.put(_search_admin_url(body["name"]), auth=config.couchbase_auth(),
                        json=body, verify=False, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if result.get("status") != "ok":
        raise QueryError(f"{result}\nindex: {body.get('name')}")
