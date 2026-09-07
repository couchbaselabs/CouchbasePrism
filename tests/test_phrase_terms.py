"""extract_phrase_terms is pure string logic - no LLM, no cluster. It runs
BEFORE the planner and the embedding call, so what it produces is what both
of those see; get this wrong and a raw "#" leaks into a prompt or an
embedding, not just into the retrieval leg these tests focus on."""
from prism.retrieval.planner import extract_phrase_terms


def test_extracts_one_hash_delimited_phrase_and_strips_only_the_hashes():
    cleaned, phrases = extract_phrase_terms(
        "How much did #John Doe# earn as per filings of 3M in 2023?")
    assert cleaned == "How much did John Doe earn as per filings of 3M in 2023?"
    assert phrases == ["John Doe"]


def test_extracts_multiple_phrases_in_order():
    cleaned, phrases = extract_phrase_terms(
        "Compare #John Doe# and #Jane Roe#'s compensation.")
    assert phrases == ["John Doe", "Jane Roe"]
    assert "#" not in cleaned


def test_no_hashes_returns_question_unchanged_and_no_phrases():
    cleaned, phrases = extract_phrase_terms("What was 3M's quick ratio in 2023?")
    assert cleaned == "What was 3M's quick ratio in 2023?"
    assert phrases == []


def test_empty_hash_span_is_not_returned_as_a_phrase():
    cleaned, phrases = extract_phrase_terms("Does ## mean anything here?")
    assert phrases == []


def test_whitespace_only_span_is_not_returned_as_a_phrase():
    cleaned, phrases = extract_phrase_terms("Odd input: #   # should be ignored.")
    assert phrases == []


def test_surrounding_whitespace_in_a_phrase_is_stripped():
    cleaned, phrases = extract_phrase_terms("Search for # John Doe # please.")
    assert phrases == ["John Doe"]
