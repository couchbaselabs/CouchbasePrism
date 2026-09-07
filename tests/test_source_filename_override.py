"""A resolved source_filename (from the catalog entry) must be used verbatim
wherever it's given, not recomputed from AWS_BUCKET/AWS_FOLDER - across every
retrieval function that scopes a query to one document. This is what makes
the catalog, not the environment, the single source of truth for where a
document's chunks actually live (see prism/catalog/extraction.py's
build_document docstring)."""
import sys

from prism.retrieval import anchor_probe
# anchor_search/vector_search are re-exported as FUNCTIONS of the same name
# from prism/retrieval/__init__.py (from .anchor_search import anchor_search),
# which permanently overwrites that attribute on the prism.retrieval package
# object - `import prism.retrieval.anchor_search as x` still resolves via
# attribute traversal in CPython, so it gets the FUNCTION too, not the
# module, even in a completely different file. Only a sys.modules lookup
# bypasses the shadowing and gets the actual submodule - verified live,
# not assumed (the attribute-import form silently returned the function).
import prism.retrieval.anchor_search    # noqa: F401 - ensures it's loaded
import prism.retrieval.vector_search    # noqa: F401
anchor_search = sys.modules["prism.retrieval.anchor_search"]
vector_search = sys.modules["prism.retrieval.vector_search"]

RESOLVED = "kpd-couchbase_Prism_3M_3M_2023Q2_10Q.pdf"


def test_vector_search_build_statement_prefers_resolved_filename():
    _, params = vector_search.build_statement([0.1, 0.2], "3M_2023Q2_10Q",
                                               source_filename=RESOLVED)
    assert params["$filename"] == RESOLVED


def test_vector_search_build_statement_falls_back_without_it():
    _, params = vector_search.build_statement([0.1, 0.2], "3M_2023Q2_10Q")
    assert params["$filename"].endswith("3M_2023Q2_10Q.pdf")
    assert params["$filename"] != RESOLVED


def test_anchor_probe_prefers_resolved_filename(monkeypatch):
    captured = {}
    monkeypatch.setattr(anchor_probe, "query",
                        lambda stmt, params: captured.update(params) or [])
    anchor_probe.probe_anchors(["Total assets"], "3M_2023Q2_10Q",
                               source_filename=RESOLVED)
    assert captured["$filename"] == RESOLVED


def test_anchor_search_prefers_resolved_filename(monkeypatch):
    captured = {}
    monkeypatch.setattr(anchor_search, "query",
                        lambda stmt, params: captured.update(params) or [])
    anchor_search.anchor_search(["Total assets"], "3M_2023Q2_10Q",
                               source_filename=RESOLVED)
    assert captured["$filename"] == RESOLVED
