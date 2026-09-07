"""fts_resolver's query/prompt construction, against faked query()/llm.chat_json
- no live cluster, no live LLM. What's actually correct (the shortlist finds
the right document, the LLM picks it) is verified live and by hand each time
this changes; this suite covers the parts that are pure logic: which fields
get sent to the model, and how a candidate-less/decline outcome is handled."""
import pytest

from prism.catalog import fts_resolver


def test_search_candidates_builds_disjuncts_over_company_aliases_sector_and_label(monkeypatch):
    captured = {}
    monkeypatch.setattr(fts_resolver, "query",
                        lambda stmt, params: captured.update(statement=stmt, params=params) or [])
    fts_resolver.search_candidates("3M quick ratio Q3 2022")
    stmt = captured["statement"]
    for field in ("company.value", "aliases", "gics_sector", "search_label"):
        assert f'"field": "{field}"' in stmt
    assert captured["params"]["$q"] == "3M quick ratio Q3 2022"


def test_search_candidates_company_hint_adds_a_boosted_disjunct(monkeypatch):
    captured = {}
    monkeypatch.setattr(fts_resolver, "query",
                        lambda stmt, params: captured.update(statement=stmt, params=params) or [])
    fts_resolver.search_candidates("revenue", company_hint="3M")
    assert captured["params"]["$company_hint"] == "3M"
    assert captured["statement"].count('"field": "company.value"') == 2


def test_llm_resolve_sends_only_candidate_fields_never_full_document_content(monkeypatch):
    captured = {}
    monkeypatch.setattr(fts_resolver.llm, "chat_json",
                        lambda system, user, model=None, stage=None:
                            captured.update(system=system, user=user) or
                            {"selected_documents": [], "selection_complete": True,
                             "missing_evidence": []})
    candidates = [{"doc_name": "3M_2022_Q3_10Q", "source_filename": "x.pdf",
                  "company": "3M", "doc_type": "10-Q", "doc_period": 2022,
                  "period_end_date_iso": "2022-09-30",
                  "should_not_appear": "this is not a candidate field"}]
    fts_resolver.llm_resolve("What was the quick ratio?", candidates)
    assert "should_not_appear" not in captured["user"]
    assert "this is not a candidate field" not in captured["user"]
    assert "3M_2022_Q3_10Q" in captured["user"]  # the real fields did make it through


def test_resolve_for_question_at_scale_declines_with_no_candidates(monkeypatch):
    monkeypatch.setattr(fts_resolver, "search_candidates", lambda *a, **k: [])
    result = fts_resolver.resolve_for_question_at_scale("What was Acme Corp's revenue?")
    assert result["selected_documents"] == []
    assert result["selection_complete"] is False
    assert result["missing_evidence"]


def test_resolve_for_question_at_scale_passes_candidates_through_to_the_llm(monkeypatch):
    candidates = [{"doc_name": "3M_2022_Q3_10Q"}]
    monkeypatch.setattr(fts_resolver, "search_candidates", lambda *a, **k: candidates)
    seen = {}
    monkeypatch.setattr(fts_resolver, "llm_resolve",
                        lambda q, c, model=None: seen.update(q=q, c=c) or
                            {"selected_documents": [{"doc_name": "3M_2022_Q3_10Q", "reason": "x"}],
                             "selection_complete": True, "missing_evidence": []})
    result = fts_resolver.resolve_for_question_at_scale("q")
    assert seen["c"] == candidates
    assert result["selected_documents"][0]["doc_name"] == "3M_2022_Q3_10Q"
