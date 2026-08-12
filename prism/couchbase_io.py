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
import requests
import urllib3

from . import config

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
    return result.get("results", [])


def ensure_primary_index(collection: str) -> None:
    query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON "
          f"`{config.BUCKET}`.`{config.SCOPE}`.`{collection}`")
