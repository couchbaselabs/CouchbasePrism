"""Query-shape tests for the hybrid retrieval statement.

This is the core retrieval path and almost all of its correctness is in WHERE
the predicates land, which is invisible at a glance and silent when wrong - a
scope filter in the wrong place still returns rows, just the wrong ones. No
cluster needed: build_statement is separated from execution for exactly this.
"""
import json

from prism.retrieval.hybrid_search import build_statement

VEC = [0.1, 0.2, 0.3]
ANCHORS = ["Total current assets", "Total current liabilities"]


def search_object(statement: str) -> dict:
    """Pull the JSON passed to SEARCH() back out, with parameter placeholders
    swapped for quoted stand-ins so it parses."""
    start = statement.index("SEARCH(d, {") + len("SEARCH(d, ")
    end = statement.index(', {"index"', start)
    blob = statement[start:end]
    for token in ("$query_vector", "$filename", "$match_text", "$a0", "$a1", "$a2"):
        blob = blob.replace(token, f'"{token}"')
    return json.loads(blob)


def test_emits_exactly_one_statement_with_all_three_legs():
    statement, params = build_statement("q", VEC, "3M_2023Q2_10Q", ANCHORS)
    assert statement.count("SEARCH(") == 1
    obj = search_object(statement)
    conjuncts = obj["query"]["conjuncts"]
    assert conjuncts[0]["field"] == "xmeta-data.filename"          # scope
    assert "disjuncts" in conjuncts[1]                              # lexical
    assert obj["knn"][0]["field"] == "text-embedding"               # vector


def test_bm25_matches_anchors_as_phrases_not_the_question():
    # Sending the raw prompt scores mostly on common words and measurably
    # bought nothing; the anchors are the printed row labels worth matching.
    statement, params = build_statement("Does 3M have a healthy liquidity profile?",
                                        VEC, "3M_2023Q2_10Q", ANCHORS)
    disjuncts = search_object(statement)["query"]["conjuncts"][1]["disjuncts"]
    assert all("match_phrase" in d for d in disjuncts)
    assert [params["$a0"], params["$a1"]] == ANCHORS
    assert "$match_text" not in params


def test_falls_back_to_the_question_when_the_planner_found_no_anchors():
    statement, params = build_statement("liquidity profile", VEC, "3M_2023Q2_10Q",
                                        anchors=[])
    disjuncts = search_object(statement)["query"]["conjuncts"][1]["disjuncts"]
    assert disjuncts[0]["match"] == "$match_text"
    assert params["$match_text"] == "liquidity profile"


def test_scope_is_applied_inside_search_on_both_legs():
    # Outside SEARCH(), a scalar filter is applied only after the Search service
    # has returned k results, so a selective filter can leave nothing.
    statement, params = build_statement("q", VEC, "3M_2023Q2_10Q", ANCHORS)
    obj = search_object(statement)
    assert obj["query"]["conjuncts"][0]["match"] == "$filename"      # lexical leg
    assert obj["knn"][0]["filter"]["match"] == "$filename"           # vector leg
    assert params["$filename"].endswith("3M_2023Q2_10Q.pdf")
    # and never as a bare WHERE predicate
    assert "filename = $filename" not in statement


def test_unscoped_search_omits_the_filter_entirely():
    statement, params = build_statement("q", VEC, doc_name=None, anchors=ANCHORS)
    obj = search_object(statement)
    assert "conjuncts" not in obj["query"]
    assert "filter" not in obj["knn"][0]
    assert "$filename" not in params


def test_title_boost_is_off_by_default():
    # `associated-titles` is wrong on a meaningful fraction of table chunks, so
    # boosting it promotes confidently wrong chunks. It is still SELECTed as
    # metadata - what must be absent is a match clause scoring against it.
    statement, _ = build_statement("q", VEC, "3M_2023Q2_10Q", ANCHORS)
    fields = [d.get("field") for d in
              search_object(statement)["query"]["conjuncts"][1]["disjuncts"]]
    assert "meta-data.associated-titles" not in fields

    boosted, _ = build_statement("q", VEC, "3M_2023Q2_10Q", ANCHORS, title_boost=3.0)
    boosted_fields = [d.get("field") for d in
                      search_object(boosted)["query"]["conjuncts"][1]["disjuncts"]]
    assert "meta-data.associated-titles" in boosted_fields


def test_index_name_is_fully_qualified():
    # The bare short name does not resolve: "Search() function using KNN and no
    # search index".
    statement, _ = build_statement("q", VEC, "3M_2023Q2_10Q", ANCHORS)
    assert '"index": "' in statement
    index = statement.split('"index": "')[1].split('"')[0]
    assert index.count(".") == 2, f"expected bucket.scope.index, got {index!r}"
