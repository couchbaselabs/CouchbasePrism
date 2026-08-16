"""Query-shape tests for the hybrid retrieval statement.

This is the core retrieval path and almost all of its correctness is in WHERE
the predicates land, which is invisible at a glance and silent when wrong - a
scope filter in the wrong place still returns rows, just the wrong ones. No
cluster needed: build_statement is separated from execution for exactly this.
"""
import json

from prism.retrieval.hybrid_search import build_statement, rrf_merge

VEC = [0.1, 0.2, 0.3]
ANCHORS = ["Total current assets", "Total current liabilities"]


def search_objects(statement: str) -> list:
    """Pull back the JSON passed to each SEARCH() leg, with parameter
    placeholders swapped for quoted stand-ins so it parses. Order is
    [lexical, vector] - the union puts the lexical branch first."""
    objects, cursor = [], 0
    while True:
        try:
            start = statement.index("SEARCH(d, {", cursor) + len("SEARCH(d, ")
        except ValueError:
            return objects
        end = statement.index(', {"index"', start)
        blob = statement[start:end]
        for token in ("$query_vector", "$filename", "$match_text",
                      "$concept", "$a0", "$a1", "$a2"):
            blob = blob.replace(token, f'"{token}"')
        objects.append(json.loads(blob))
        cursor = end


def search_object(statement: str) -> dict:
    """The lexical leg, which is where the anchor and scope assertions live."""
    return search_objects(statement)[0]


def test_emits_one_statement_carrying_both_legs():
    # One SQL per question is a hard requirement: it is what the demo shows.
    # Two SEARCH legs inside it is the point - fusing them into a single
    # SEARCH() sums scores on incompatible scales and makes a lexical-only hit
    # unreachable.
    statement, params = build_statement("q", VEC, "3M_2023Q2_10Q", ANCHORS)
    assert ";" not in statement
    assert statement.count("UNION ALL") == 1
    lexical, vector = search_objects(statement)
    assert len(search_objects(statement)) == 2
    conjuncts = lexical["query"]["conjuncts"]
    assert conjuncts[0]["field"] == "xmeta-data.filename"          # scope
    assert "disjuncts" in conjuncts[1]                              # lexical
    assert "knn" not in lexical                                     # legs stay apart
    assert vector["knn"][0]["field"] == "text-embedding"            # vector


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
    lexical, vector = search_objects(statement)
    assert lexical["query"]["conjuncts"][0]["match"] == "$filename"  # lexical leg
    assert vector["knn"][0]["filter"]["match"] == "$filename"        # vector leg
    assert params["$filename"].endswith("3M_2023Q2_10Q.pdf")
    # and never as a bare WHERE predicate
    assert "filename = $filename" not in statement


def test_unscoped_search_omits_the_filter_entirely():
    statement, params = build_statement("q", VEC, doc_name=None, anchors=ANCHORS)
    lexical, vector = search_objects(statement)
    assert "conjuncts" not in lexical["query"]
    assert "filter" not in vector["knn"][0]
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


# ------------------------------------------------------------------- RRF

def test_rrf_ranks_by_position_not_by_score():
    # The whole point: BM25 scores (~0.2) and kNN scores (~1.0) are on
    # different scales, so summing them buried lexical-only hits. Ranking is
    # scale-free - a leg's top hit counts the same whatever it scored.
    rows = [{"id": "lex-top", "leg": "lexical", "score": 0.21},
            {"id": "vec-top", "leg": "vector", "score": 0.99}]
    assert [c["id"] for c in rrf_merge(rows)] == ["lex-top", "vec-top"]


def test_rrf_rewards_a_chunk_both_legs_found():
    # Ranked 2nd by each leg beats anything ranked 1st by only one of them.
    rows = [{"id": "lex-only", "leg": "lexical", "score": 0.9},
            {"id": "both", "leg": "lexical", "score": 0.5},
            {"id": "vec-only", "leg": "vector", "score": 0.9},
            {"id": "both", "leg": "vector", "score": 0.5}]
    merged = rrf_merge(rows)
    assert merged[0]["id"] == "both"
    assert merged[0]["leg"] == "lexical+vector"
    assert merged[0]["leg_ranks"] == {"lexical": 2, "vector": 2}


def test_rrf_does_not_trust_union_row_order():
    # UNION ALL guarantees no ordering across branches, so rank is derived from
    # each leg's own scores rather than from the order rows arrived in.
    rows = [{"id": "weak", "leg": "lexical", "score": 0.1},
            {"id": "strong", "leg": "lexical", "score": 0.8}]
    assert [c["id"] for c in rrf_merge(rows)] == ["strong", "weak"]
