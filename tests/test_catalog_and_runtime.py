"""Pure-logic tests for catalog resolution, date derivation, binding
validation and conclusion agreement. Each case below corresponds to a bug that
actually occurred during the FinanceBench evaluation."""
from prism import catalog, runtime

CATALOG = [
    {"doc_name": "3M_2018_10K", "doc_type": "10-K", "doc_period": 2018,
     "period_end_date_iso": "2018-12-31"},
    {"doc_name": "3M_2022_10K", "doc_type": "10-K", "doc_period": 2022,
     "period_end_date_iso": "2022-12-31"},
    {"doc_name": "3M_2023Q2_10Q", "doc_type": "10-Q", "doc_period": 2023,
     "period_end_date_iso": "2023-06-30"},
]


# ----------------------------------------------------------------- dates

def test_month_first_date_keeps_its_year():
    # The original regex matched "December 31" and dropped ", 2015", so every
    # US-style filing date lost its year.
    assert catalog.to_iso_date("December 31, 2015") == "2015-12-31"


def test_day_first_and_iso_still_parse():
    assert catalog.to_iso_date("12 March 2021") == "2021-03-12"
    assert catalog.to_iso_date("2021-03-12") == "2021-03-12"


def test_partial_date_is_left_unparsed_rather_than_invented():
    assert catalog.to_iso_date("March 2021") is None


# -------------------------------------------------------------- resolution

def test_quarter_mention_selects_the_10q_not_the_annual():
    assert catalog.resolve_for_question(
        CATALOG, "quick ratio for Q2 of FY2023?") == "3M_2023Q2_10Q"


def test_fiscal_year_selects_the_annual_filing():
    assert catalog.resolve_for_question(
        CATALOG, "FY2018 capital expenditure?") == "3M_2018_10K"


def test_no_period_mentioned_falls_back_to_most_recent():
    assert catalog.resolve_for_question(
        CATALOG, "Does 3M maintain a stable dividend?") == "3M_2023Q2_10Q"


# ------------------------------------------------------------- identifiers

def test_human_labels_become_formula_safe_identifiers():
    # Until this normalization existed, every candidate formula failed to parse
    # because the planner emits labels like "Total current assets".
    assert runtime.to_identifier("Total current assets") == "total_current_assets"
    assert runtime.to_identifier("Property, plant & equipment - net") == \
        "property_plant_equipment_net"


# ---------------------------------------------------------------- binding

CHUNKS = [{"text": "Total current assets | 15,754 | 14,688\n"
                   "Total current liabilities | 10,936 | 9,523"}]


def test_binding_accepts_a_value_printed_with_separators():
    bound = runtime.validate_bindings(
        [{"name": "total_current_assets", "value": 15754, "period": "June 30, 2023"}],
        CHUNKS)
    assert bound[0]["grounded"] is True


def test_binding_rejects_a_value_absent_from_the_source():
    bound = runtime.validate_bindings(
        [{"name": "invented", "value": 99999, "period": "June 30, 2023"}], CHUNKS)
    assert bound[0]["grounded"] is False
    assert "not found verbatim" in bound[0]["binding_issues"][0]


def test_mixed_periods_invalidate_every_binding():
    # Facts from different columns compute a number that is wrong in a way no
    # arithmetic check can detect, so the whole set is rejected.
    bound = runtime.validate_bindings([
        {"name": "total_current_assets", "value": 15754, "period": "June 30, 2023"},
        {"name": "total_current_liabilities", "value": 9523, "period": "December 31, 2022"},
    ], CHUNKS)
    assert all(f["grounded"] is False for f in bound)
    assert any("mixed periods" in i for f in bound for i in f["binding_issues"])


# ------------------------------------------------------------- conclusion

POLICY = {"policy": {"healthy_at_or_above": 1.0}}


def test_no_policy_means_no_verdict_is_authorised():
    result = runtime.validate_conclusion([{"value": 0.9578}], None)
    assert result["status"] == "no_policy"


def test_candidates_on_the_same_side_agree():
    result = runtime.validate_conclusion(
        [{"value": 0.85}, {"value": 0.9578}], POLICY)
    assert result["status"] == "agreed" and result["verdict"] is False


def test_candidates_straddling_the_threshold_block_on_review():
    # 0.9578 and 1.441 are both defensible quick-ratio conventions; picking one
    # or averaging them would be a wrong-answer risk, not a shortcut.
    result = runtime.validate_conclusion(
        [{"value": 0.9578}, {"value": 1.441}], POLICY)
    assert result["status"] == "conflicting"
