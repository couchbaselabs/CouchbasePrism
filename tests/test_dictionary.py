"""The formula evaluator is a security boundary as much as a correctness one:
formulas originate as model-authored text, so anything it accepts is something
a model can make PRISM execute."""
import pytest

from prism.dictionary import (
    FormulaError, evaluate_formula, find_metric, find_policy, judge,
)

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

def _dictionary(status="approved"):
    return {"entries": [
        {"id": "finance.quick_ratio", "entry_type": "metric",
         "recognition": {"canonical_name": "quick ratio", "aliases": ["acid-test ratio"]},
         "interpretation": {"formula": "a / b", "required_facts": ["a", "b"]},
         "governance": {"status": status}},
        {"id": "finance.quick_ratio.policy", "entry_type": "interpretation_policy",
         "applies_to": "finance.quick_ratio", "policy": {"healthy_at_or_above": 1.0},
         "governance": {"status": status}},
    ]}


def test_matches_on_alias_and_is_case_insensitive():
    assert find_metric(_dictionary(), "Acid-Test Ratio")["id"] == "finance.quick_ratio"


def test_proposed_entries_are_not_authoritative():
    # A proposed entry records unresolved ambiguity; using it would skip the
    # human decision the whole design depends on.
    assert find_metric(_dictionary(status="proposed"), "quick ratio") is None
    assert find_policy(_dictionary(status="proposed"), "finance.quick_ratio") is None


def test_judge_is_inclusive_at_the_threshold():
    assert judge(1.0, {"policy": {"healthy_at_or_above": 1.0}})[0] is True
    assert judge(0.9578, {"policy": {"healthy_at_or_above": 1.0}})[0] is False


def test_judge_declines_a_policy_shape_it_does_not_understand():
    verdict, reason = judge(0.5, {"policy": {"some_future_rule": 3}})
    assert verdict is None and "no rule" in reason
