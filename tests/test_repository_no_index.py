"""catalog.load_all() and dictionary.load() both tolerate ONE specific
condition - no primary index yet, the expected state before Initialize has
ever run - by returning empty, not raising. Anything else must still raise;
a genuine query error silently reading as "no catalog" would be worse than
the crash it replaces."""
import pytest

from prism import catalog, dictionary
from prism.couchbase_io import QueryError


def _no_index_error(keyspace: str) -> QueryError:
    return QueryError(
        f"[{{'code': 4000, 'msg': 'No index available on keyspace `default`:"
        f"`acme`.`prism`.`{keyspace}`...'}}]\nstatement: SELECT ...")


def test_catalog_load_all_returns_empty_on_missing_index(monkeypatch):
    monkeypatch.setattr(catalog.repository, "query",
                        lambda *a, **k: (_ for _ in ()).throw(_no_index_error("catalog")))
    assert catalog.load_all() == []


def test_catalog_load_all_reraises_other_query_errors(monkeypatch):
    monkeypatch.setattr(catalog.repository, "query",
                        lambda *a, **k: (_ for _ in ()).throw(QueryError("syntax error")))
    with pytest.raises(QueryError, match="syntax error"):
        catalog.load_all()


def test_dictionary_load_returns_empty_entries_on_missing_index(monkeypatch):
    monkeypatch.setattr(dictionary.repository, "query",
                        lambda *a, **k: (_ for _ in ()).throw(_no_index_error("dictionary")))
    assert dictionary.load() == {"entries": []}


def test_dictionary_load_reraises_other_query_errors(monkeypatch):
    monkeypatch.setattr(dictionary.repository, "query",
                        lambda *a, **k: (_ for _ in ()).throw(QueryError("syntax error")))
    with pytest.raises(QueryError, match="syntax error"):
        dictionary.load()


def test_dictionary_load_with_a_path_is_unaffected_by_the_couchbase_path(tmp_path):
    # The test seam (path=) never touches query() at all - confirming the
    # no-index tolerance added to the Couchbase branch didn't leak into it.
    missing = tmp_path / "dictionary.yaml"
    assert dictionary.load(path=missing) == {"entries": []}
