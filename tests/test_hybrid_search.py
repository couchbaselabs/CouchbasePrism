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
                      "$terms", "$concept", "$a0", "$a1", "$a2",
                      "$phrase_0", "$phrase_1"):
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


def test_bm25_matches_the_merged_term_bag_not_the_question():
    # Sending the raw prompt scores mostly on common words and measurably
    # bought nothing. Concept + anchors are lowercased, split on punctuation,
    # deduped and OR-matched as one bag - not sent as match_phrase per anchor,
    # which was tried and measured WORSE (financebench_id_10420: exact-phrase
    # anchors matched MORE pages that legitimately print the same caption
    # elsewhere in the filing, not the right one - see hybrid_search.py).
    statement, params = build_statement("Does 3M have a healthy liquidity profile?",
                                        VEC, "3M_2023Q2_10Q", ANCHORS)
    disjuncts = search_object(statement)["query"]["conjuncts"][1]["disjuncts"]
    assert len(disjuncts) == 1
    assert disjuncts[0]["match"] == "$terms"
    assert disjuncts[0]["operator"] == "or"
    assert set(params["$terms"].split()) == {"total", "current", "assets", "liabilities"}
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


def test_a_resolved_source_filename_is_used_verbatim_not_recomputed():
    # config.source_filename() depends on AWS_BUCKET/AWS_FOLDER staying
    # correct forever; a source_filename already resolved (from the catalog
    # entry) must win outright, not just influence, whatever those env vars
    # say at query time.
    _, params = build_statement("q", VEC, "3M_2023Q2_10Q", ANCHORS,
                                source_filename="kpd-couchbase_Prism_3M_3M_2023Q2_10Q.pdf")
    assert params["$filename"] == "kpd-couchbase_Prism_3M_3M_2023Q2_10Q.pdf"


def test_no_source_filename_falls_back_to_computing_it():
    _, params = build_statement("q", VEC, "3M_2023Q2_10Q", ANCHORS)
    assert params["$filename"].endswith("3M_2023Q2_10Q.pdf")


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


def test_forced_phrases_add_match_phrase_disjuncts_alongside_the_term_bag():
    # Path A: a forced phrase (from planner.extract_phrase_terms's #...#
    # spans) is ADDED to the existing lexical leg, not a replacement for the
    # term bag - both should be present, each phrase as its own match_phrase
    # disjunct scored and summed by Bleve like any other.
    statement, params = build_statement("q", VEC, "3M_2023Q2_10Q", ANCHORS,
                                        forced_phrases=["John Doe"])
    disjuncts = search_object(statement)["query"]["conjuncts"][1]["disjuncts"]
    assert disjuncts[0]["match"] == "$terms"                    # term bag still first
    phrase_disjuncts = [d for d in disjuncts if "match_phrase" in d]
    assert len(phrase_disjuncts) == 1
    assert phrase_disjuncts[0]["match_phrase"] == "$phrase_0"
    assert phrase_disjuncts[0]["field"] == "text-to-embed"
    assert params["$phrase_0"] == "John Doe"


def test_forced_phrases_each_get_their_own_disjunct_and_parameter():
    statement, params = build_statement("q", VEC, "3M_2023Q2_10Q", ANCHORS,
                                        forced_phrases=["John Doe", "Jane Roe"])
    disjuncts = search_object(statement)["query"]["conjuncts"][1]["disjuncts"]
    phrase_disjuncts = [d for d in disjuncts if "match_phrase" in d]
    assert {d["match_phrase"] for d in phrase_disjuncts} == {"$phrase_0", "$phrase_1"}
    assert params["$phrase_0"] == "John Doe"
    assert params["$phrase_1"] == "Jane Roe"


def test_no_forced_phrases_adds_no_disjunct_or_parameter():
    statement, params = build_statement("q", VEC, "3M_2023Q2_10Q", ANCHORS)
    disjuncts = search_object(statement)["query"]["conjuncts"][1]["disjuncts"]
    assert not any("match_phrase" in d for d in disjuncts)
    assert not any(k.startswith("$phrase_") for k in params)


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
    rows = [{"id": "lex-top", "channel": "bm25", "score": 0.21},
            {"id": "vec-top", "channel": "vector", "score": 0.99}]
    assert [c["id"] for c in rrf_merge(rows)] == ["lex-top", "vec-top"]


def test_rrf_rewards_a_chunk_both_legs_found():
    # Ranked 2nd by each leg beats anything ranked 1st by only one of them.
    rows = [{"id": "lex-only", "channel": "bm25", "score": 0.9},
            {"id": "both", "channel": "bm25", "score": 0.5},
            {"id": "vec-only", "channel": "vector", "score": 0.9},
            {"id": "both", "channel": "vector", "score": 0.5}]
    merged = rrf_merge(rows)
    assert merged[0]["id"] == "both"
    assert merged[0]["channel"] == "bm25+vector"
    assert merged[0]["channels"]["bm25"]["rank"] == 2
    assert merged[0]["channels"]["vector"]["rank"] == 2


def test_rrf_does_not_trust_union_row_order():
    # UNION ALL guarantees no ordering across branches, so rank is derived from
    # each leg's own scores rather than from the order rows arrived in.
    rows = [{"id": "weak", "channel": "bm25", "score": 0.1},
            {"id": "strong", "channel": "bm25", "score": 0.8}]
    assert [c["id"] for c in rrf_merge(rows)] == ["strong", "weak"]


def test_rrf_exposes_the_arithmetic_per_channel():
    # A fused number nobody can derive is not inspectable. Each chunk carries
    # its rank, the raw score it came from, and that channel's contribution.
    rows = [{"id": "x", "channel": "bm25", "score": 0.6695},
            {"id": "y", "channel": "bm25", "score": 0.10},
            {"id": "x", "channel": "vector", "score": 0.8127}]
    top = rrf_merge(rows)[0]
    assert top["id"] == "x"
    assert top["channels"]["bm25"] == {
        "rank": 1, "raw_score": 0.6695, "contribution": round(1 / 61, 6)}
    assert top["channels"]["vector"]["rank"] == 1
    assert top["rrf_score"] == round(1 / 61 + 1 / 61, 6)


def test_channel_weights_shift_the_ordering():
    # Equal weights are a default, not a law. Weighting bm25 to zero must leave
    # the vector channel deciding on its own.
    rows = [{"id": "lex", "channel": "bm25", "score": 0.9},
            {"id": "vec", "channel": "vector", "score": 0.9}]
    assert [c["id"] for c in rrf_merge(rows)][0] == "lex"   # tie, insertion order
    weighted = rrf_merge(rows, weights={"bm25": 0.0, "vector": 1.0})
    assert weighted[0]["id"] == "vec"
    assert weighted[0]["rrf_score"] > weighted[1]["rrf_score"]
