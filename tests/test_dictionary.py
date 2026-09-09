"""The formula evaluator is a security boundary as much as a correctness one:
formulas originate as model-authored text, so anything it accepts is something
a model can make PRISM execute."""
import pytest

from prism.dictionary import FormulaError, evaluate_formula, find_metric, judge

FACTS = {"total_current_assets": 15754.0, "inventory": 5280.0,
         "total_current_liabilities": 10936.0}


def test_evaluates_approved_quick_ratio():
    value = evaluate_formula(
        "(total_current_assets - inventory) / total_current_liabilities", FACTS)
    assert round(value, 4) == 0.9578


def test_operator_precedence_is_pythons_not_left_to_right():
    assert evaluate_formula("inventory + total_current_liabilities * 2", FACTS) == \
        5280.0 + 10936.0 * 2


@pytest.mark.parametrize("formula", [
    "__import__('os').system('echo pwned')",
    "open('/etc/passwd').read()",
    "total_current_assets.__class__",
    "facts['inventory']",
    "sum([1, 2])",
])
def test_rejects_anything_that_is_not_arithmetic(formula):
    with pytest.raises(FormulaError):
        evaluate_formula(formula, FACTS)


def test_rejects_unbound_fact_rather_than_defaulting_it():
    # Silently treating a missing fact as zero would produce a plausible,
    # confidently wrong number - the exact failure mode this tier prevents.
    with pytest.raises(FormulaError, match="unbound fact"):
        evaluate_formula("total_current_assets - goodwill", FACTS)


def test_rejects_division_by_zero():
    with pytest.raises(FormulaError, match="division by zero"):
        evaluate_formula("inventory / zero", {**FACTS, "zero": 0.0})


# ------------------------------------------------------------------ lookup

def _dictionary():
    return {"entries": [
        {"id": "abc-1", "metric": "quick ratio", "abbreviation": "acid-test ratio",
         "formula": "a / b", "required_facts": ["a", "b"],
         "threshold_operator": ">=", "threshold_number": 1.0},
    ]}


def test_matches_on_alias_and_is_case_insensitive():
    assert find_metric(_dictionary(), "Acid-Test Ratio")["id"] == "abc-1"


def test_an_empty_dictionary_matches_nothing():
    # No separate "proposed"/"approved" status anymore - presence in the
    # dictionary IS the approval, so an empty dictionary is the only way to
    # get "nothing matches", not a status filter.
    assert find_metric({"entries": []}, "quick ratio") is None


def test_judge_is_inclusive_at_the_threshold():
    entry = {"threshold_operator": ">=", "threshold_number": 1.0}
    assert judge(1.0, entry)[0] is True
    assert judge(0.9578, entry)[0] is False


def test_judge_supports_every_comparison_operator():
    assert judge(5, {"threshold_operator": "<=", "threshold_number": 10})[0] is True
    assert judge(5, {"threshold_operator": ">", "threshold_number": 5})[0] is False
    assert judge(5, {"threshold_operator": "<", "threshold_number": 5})[0] is False
    assert judge(5, {"threshold_operator": "==", "threshold_number": 5})[0] is True


def test_judge_declines_an_entry_with_no_threshold():
    verdict, reason = judge(0.5, {"formula": "a / b"})
    assert verdict is None and "no approved threshold" in reason


def test_judge_declines_an_operator_it_does_not_understand():
    verdict, reason = judge(0.5, {"threshold_operator": "~=", "threshold_number": 1.0})
    assert verdict is None and "does not understand" in reason


def test_a_primary_and_alternate_entry_coexist_and_preferred_wins():
    dictionary = {"entries": [
        {"id": "p", "metric": "quick ratio", "formula": "a / b",
         "formula_type": "Preferred"},
        {"id": "alt", "metric": "quick ratio", "formula": "c / d",
         "formula_type": "Alternate"},
    ]}
    assert find_metric(dictionary, "quick ratio")["id"] == "p"
    assert find_metric(dictionary, "quick ratio", formula_type="Alternate")["id"] == "alt"
    # Asking for a formula_type that does not exist for this concept still
    # falls back to the preferred one rather than matching nothing.
    assert find_metric(dictionary, "quick ratio", formula_type="nonexistent")["id"] == "p"


def test_an_entry_with_no_formula_type_defaults_to_preferred():
    # Every entry before this field existed behaves exactly as it always did.
    entry = {"id": "x", "metric": "quick ratio", "formula": "a / b"}
    assert find_metric({"entries": [entry]}, "quick ratio") is entry


# ------------------------------------------------------- forget and clear

def _seeded(path):
    from prism.dictionary import approve
    approve("quick ratio", "(a - b) / c", healthy_at_or_above=1.0, path=path)
    approve("current ratio", "a / c", path=path)
    return path


def test_forget_removes_every_entry_matching_the_concept(tmp_path):
    from prism.dictionary import find_metric, forget, load
    path = _seeded(tmp_path / "d.yaml")
    removed = forget("quick ratio", path=path)
    assert len(removed) == 1
    data = load(path)
    assert find_metric(data, "quick ratio") is None
    # and leaves everything else that was reviewed alone
    assert find_metric(data, "current ratio") is not None


def test_forget_is_a_no_op_for_an_unknown_concept(tmp_path):
    from prism.dictionary import forget, load
    path = _seeded(tmp_path / "d.yaml")
    assert forget("gross margin", path=path) == []
    assert len(load(path)["entries"]) == 2


def test_clear_empties_the_dictionary(tmp_path):
    from prism.dictionary import clear, load
    path = _seeded(tmp_path / "d.yaml")
    removed = clear(path=path)
    assert len(removed) == 2
    assert load(path) == {"entries": []}


def test_a_cleared_dictionary_still_loads(tmp_path):
    # PRISM operating with an empty dictionary is the whole claim, so this must
    # be an ordinary state rather than an error path.
    from prism.dictionary import clear, find_metric, load
    path = _seeded(tmp_path / "d.yaml")
    clear(path=path)
    assert find_metric(load(path), "quick ratio") is None


def test_a_concept_named_with_underscores_still_matches_its_entry():
    # The planner is free to return "quick ratio", "quick-ratio" or
    # "quick_ratio". Treating those as three concepts left an approved entry
    # unmatched and silently disabled governance for the question.
    entry = {"metric": "quick ratio"}
    for spelling in ("quick ratio", "quick-ratio", "quick_ratio", "Quick Ratio"):
        assert find_metric({"entries": [entry]}, spelling) is entry


def test_approve_replaces_only_the_preferred_entry_for_the_same_concept(tmp_path):
    # A hand-written Alternate for this concept is left alone by a re-approval
    # of the Preferred one.
    from prism.dictionary import approve, find_metric, load, save
    path = tmp_path / "d.yaml"
    save({"entries": [{"id": "alt", "metric": "quick ratio", "formula": "c / d",
                       "formula_type": "Alternate"}]}, path=path)
    approve("quick ratio", "a / b", path=path)
    data = load(path)
    assert len(data["entries"]) == 2
    assert find_metric(data, "quick ratio")["formula"] == "a / b"
    assert find_metric(data, "quick ratio", formula_type="Alternate")["formula"] == "c / d"
