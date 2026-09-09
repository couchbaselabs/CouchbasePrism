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
        # Couchbase returns a structured error body (status/errors) even on a
        # non-2xx HTTP status - verified live: a missing primary index comes
        # back as HTTP 404 with a perfectly informative body (code 4000, "No
        # index available on keyspace ... CREATE PRIMARY INDEX ..."). Reading
        # that FIRST, before raise_for_status(), means a real Couchbase error
        # always surfaces with its own message; raise_for_status() only fires
        # when the response isn't valid JSON at all (a proxy/network failure
        # upstream of Couchbase), where there is no better message to give
        # than the HTTP status. Checking status naively first, then calling
        # raise_for_status() unconditionally, is exactly what silently threw
        # away this detail before this was fixed - a bare "404 Client Error:
        # Not Found for url: ..." with the actual cause never read.
        try:
            result = resp.json()
        except ValueError:
            result = None
        if result is not None and result.get("status") != "success":
            raise QueryError(f"{result.get('errors')}\nstatement: {statement}")
        if result is None:
            resp.raise_for_status()
            raise QueryError(f"non-JSON response from query service\nstatement: {statement}")
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


def ensure_primary_index(collection: str, scope: str = None) -> None:
    scope = scope or config.DEFAULT_SCOPE
    query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON "
          f"`{config.BUCKET}`.`{scope}`.`{collection}`")


def _search_admin_url(index_name: str = "", scope: str = None) -> str:
    scope = scope or config.DEFAULT_SCOPE
    base = (f"https://{config.couchbase_host()}:18094/api/bucket/"
            f"{config.BUCKET}/scope/{scope}/index")
    return f"{base}/{index_name}" if index_name else base


def get_search_index_definition(index_name: str, scope: str = None) -> dict:
    """The live index definition, scoped-index REST path. The flat
    /api/index/{name} path 400s ("index not found") for a scoped index -
    verified live rather than assumed when this was first written."""
    resp = requests.get(_search_admin_url(index_name, scope), auth=config.couchbase_auth(),
                        verify=False, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "ok":
        raise QueryError(f"{body}\nindex: {index_name}")
    return body["indexDef"]


def delete_search_index(index_name: str, scope: str = None) -> None:
    """Deleting a nonexistent index is not an error here - Initialize always
    deletes-then-recreates, so a fresh environment with no index yet must not
    fail this step. A missing SCOPED index is not a 404, though - verified
    live: it comes back 400 with "index not found" in the body, the same
    shape as any other malformed-request error, so that specific message is
    what's checked for rather than the status code alone."""
    resp = requests.delete(_search_admin_url(index_name, scope), auth=config.couchbase_auth(),
                           verify=False, timeout=30)
    if resp.status_code == 400 and "index not found" in resp.text:
        return
    resp.raise_for_status()


def search_index_count(index_name: str, scope: str = None) -> int:
    """How many documents the index has processed so far - not the corpus
    total, the index's OWN progress. Used to know when a freshly (re)created
    index has caught up, since a drop+recreate starts it from empty and a
    query against it returns incomplete results until it does."""
    resp = requests.get(f"{_search_admin_url(index_name, scope)}/count",
                        auth=config.couchbase_auth(), verify=False, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "ok":
        raise QueryError(f"{body}\nindex: {index_name}")
    return body["count"]


def wait_for_search_index(index_name: str, target_count: int, timeout: int = 300,
                          poll_interval: int = 3, on_progress=None,
                          scope: str = None) -> int:
    """Polls search_index_count() until it reaches target_count or timeout.
    Measured live on this corpus: a full reindex of 177,140 documents
    completes in under a minute, so the default timeout is generous rather
    than tight. Returns the final observed count; raises TimeoutError if it
    never reaches the target - callers should treat that as the recreate
    having failed to converge, not proceed on a partially-built index.
    """
    started = time.perf_counter()
    count = 0
    while time.perf_counter() - started < timeout:
        count = search_index_count(index_name, scope)
        if on_progress:
            on_progress(count, target_count)
        if count >= target_count:
            return count
        time.sleep(poll_interval)
    raise TimeoutError(
        f"{index_name} reached {count}/{target_count} documents after {timeout}s "
        "- did not finish reindexing in time")


def search_facet(index_name: str, field: str, size: int = 1000,
                 scope: str = None) -> list:
    """Distinct values of an indexed field, with each value's document count -
    a match_all query with size=0 (no document hits returned, only the
    facet), so this is cheap regardless of corpus size. Field must be mapped
    in the index (docs' xmeta-data.filename is text/keyword-analyzed, which
    facets on the whole string as one term - verified live: returns the same
    354 distinct filenames as a full N1QL `SELECT DISTINCT` scan, in ~300ms
    against a query that scan took much longer to run.
    """
    resp = requests.post(
        f"{_search_admin_url(index_name, scope)}/query", auth=config.couchbase_auth(),
        json={"query": {"match_all": {}}, "size": 0,
              "facets": {"_": {"field": field, "size": size}}},
        verify=False, timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    # The search-query endpoint's `status` is an object ({"total", "failed",
    # "successful", "errors"}), not a plain "ok"/"fail" string like the admin
    # endpoints above - verified live, not assumed. An unmapped/nonexistent
    # field is NOT an error here either: it comes back status.failed == 0
    # with an empty facet (no "terms" key), so that case falls through to the
    # empty-list default rather than raising.
    if body.get("status", {}).get("failed"):
        raise QueryError(f"{body}\nindex: {index_name}, field: {field}")
    return body.get("facets", {}).get("_", {}).get("terms", []) or []


def create_search_index(definition: dict, scope: str = None, index_name: str = None) -> None:
    """PUTs a new index from a definition read back from a live index (e.g.
    design/fts-index.json) - which carries three identifiers the SERVER
    assigns and the client must not resupply when creating a fresh index:
    the index's own `uuid`, the source bucket's `sourceUUID`, and each
    scoped collection's `uid`. Verified live: submitting the definition with
    these stripped creates a working index the server assigns fresh IDs to;
    querying it returns results identical to the index it was copied from.

    sourceName and the scope name are overridden from config/`scope` rather
    than trusted from the file, so a definition captured against one
    bucket/scope still creates correctly against whatever domain this is
    being created for. `index_name`, if given, overrides the definition's own
    `name` - one definition (design/fts-index.json) drives a differently-
    named index per domain (ftsSecfilingsDocs, ftsIso20020Docs, ...).

    The mapping.types key encodes "{scope}.{collection}" (doc_config.mode is
    scope.collection.type_field) - captured from whatever scope this
    definition was last read back from, so it is rewritten to the scope this
    index is actually being created in. Left unrewritten, a definition
    captured against one domain would create an index in another domain's
    scope that never matches any document there, since FTS reads that key to
    know which scope.collection to listen to.
    """
    scope = scope or config.DEFAULT_SCOPE
    body = copy.deepcopy(definition)
    body.pop("uuid", None)
    body.pop("sourceUUID", None)
    body["sourceName"] = config.BUCKET
    for collection in (body.get("sourceParams", {})
                          .get("scopeParams", {}).get("collections", [])):
        collection.pop("uid", None)
    body["sourceParams"]["scopeParams"]["name"] = scope
    if index_name:
        body["name"] = index_name
    types = body["params"]["mapping"]["types"]
    body["params"]["mapping"]["types"] = {
        f"{scope}.{key.rpartition('.')[2]}": mapping for key, mapping in types.items()
    }
    resp = requests.put(_search_admin_url(body["name"], scope), auth=config.couchbase_auth(),
                        json=body, verify=False, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if result.get("status") != "ok":
        raise QueryError(f"{result}\nindex: {body.get('name')}")
