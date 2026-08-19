"""Pure-logic tests for catalog resolution, date derivation, binding
validation and conclusion agreement. Each case below corresponds to a bug that
actually occurred during the FinanceBench evaluation."""
from prism import catalog, retrieval, runtime, trace

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


# ------------------------------------------------------------------- trace

def test_trace_is_opt_in_and_preserves_event_order():
    trace.add("outside", ignored=True)
    with trace.capture() as events:
        trace.add("first", value=1)
        trace.add("second", value=2)
    assert [e["type"] for e in events] == ["first", "second", "trace_summary"]


def test_trace_compacts_large_vectors():
    with trace.capture() as events:
        trace.add("query", params={"vector": [0.1] * 2048})
    assert events[0]["params"]["vector"] == "<vector: 2048 dimensions>"


# ------------------------------------------------- generic plan/candidate schema

def test_anchors_are_collected_across_facts_and_deduplicated():
    plan = {"required_facts": [
        {"id": "a", "content_anchors": ["Total assets", "Total assets"]},
        {"id": "b", "content_anchors": ["Net income"]},
    ]}
    assert retrieval.plan_anchors(plan) == ["Total assets", "Net income"]
    assert retrieval.fact_ids(plan) == ["a", "b"]


def test_a_plan_that_drifts_to_the_old_flat_shape_still_runs():
    # Models occasionally return bare strings where objects were asked for. A
    # plan that loses its anchors is degraded; a run that crashes is broken.
    plan = {"required_facts": ["inventory"], "content_anchors": ["Inventories"]}
    assert retrieval.fact_ids(plan) == ["inventory"]
    assert retrieval.plan_anchors(plan) == ["Inventories"]


def test_a_formula_referencing_an_undeclared_fact_is_rejected():
    # An undeclared identifier carries no anchors, so nothing will ever bind it
    # and the candidate can only fail later with an opaque evaluation error.
    problem = runtime.validate_candidate({
        "formula": "a / (b + c)",
        "required_facts": [{"id": "a", "content_anchors": ["Total assets"]}],
    })
    assert "['b', 'c']" in problem


def test_a_well_formed_candidate_is_accepted():
    assert runtime.validate_candidate({
        "formula": "a / b",
        "required_facts": [{"id": "a"}, {"id": "b"}],
    }) == ""


def test_candidate_facts_carry_anchors_the_plan_never_asked_for():
    # This is the point of the schema change: a fact only one convention needs
    # still arrives with the anchors required to retrieve it.
    facts = runtime.candidate_facts([
        {"required_facts": [{"id": "inventory", "content_anchors": ["Inventories"]}]},
        {"required_facts": [{"id": "inventory", "content_anchors": ["Inventories"]},
                            {"id": "prepaid", "content_anchors": ["Prepaid expenses"]}]},
    ])
    assert [f["id"] for f in facts] == ["inventory", "prepaid"]


# ------------------------------------------------------- anchor repair

def test_repair_is_skipped_when_there_is_nothing_to_repair():
    # No dead anchors, or nothing retrieved to learn vocabulary from, must not
    # cost a model call - repair is only worth paying for when it can work.
    assert retrieval.repair_anchors("q", [], [{"text": "x"}]) == ([], [])
    assert retrieval.repair_anchors("q", ["missing label"], []) == ([], [])


# ------------------------------------------------ printed vs canonical form

def test_form_is_recognised_however_the_cover_page_printed_it():
    # Extraction preserves the printed string on purpose, so comparison must
    # normalise. Different extraction models returned "10-K" and "FORM 10-K",
    # and an exact == comparison silently sent everything to the fallback.
    for printed in ("10-K", "FORM 10-K", "Form 10-K", "10K", "ANNUAL REPORT FORM 10-K"):
        assert catalog.form_of(printed) == "10-K", printed
    for printed in ("10-Q", "FORM 10-Q", "form 10q"):
        assert catalog.form_of(printed) == "10-Q", printed
    assert catalog.form_of("8-K") == "8-K"
    assert catalog.form_of(None) is None
    assert catalog.form_of("ANNUAL REPORT") is None


def test_quarterly_question_resolves_to_the_10Q_despite_printed_form():
    docs = [
        {"doc_name": "X_2023_10K", "doc_type": "FORM 10-K", "doc_period": 2023,
         "period_end_date_iso": "2023-12-31"},
        {"doc_name": "X_2023Q2_10Q", "doc_type": "FORM 10-Q", "doc_period": 2023,
         "period_end_date_iso": "2023-06-30"},
    ]
    assert catalog.resolve(docs, 2023, 2) == "X_2023Q2_10Q"
    assert catalog.resolve(docs, 2023, None) == "X_2023_10K"


# ------------------------------------------------ subject before period

MULTI = [
    {"doc_name": "3M_2022_10K", "company": "3M COMPANY", "doc_type": "FORM 10-K",
     "doc_period": 2022, "period_end_date_iso": "2022-12-31"},
    {"doc_name": "ADOBE_2022_10K", "company": "ADOBE INC.", "doc_type": "FORM 10-K",
     "doc_period": 2022, "period_end_date_iso": "2022-12-02"},
    {"doc_name": "COCACOLA_2022_10K", "company": "THE COCA-COLA COMPANY",
     "doc_type": "FORM 10-K", "doc_period": 2022, "period_end_date_iso": "2022-12-31"},
    {"doc_name": "PG_E_2022_10K", "company": "PACIFIC GAS AND ELECTRIC COMPANY",
     "doc_type": "FORM 10-K", "doc_period": 2022, "period_end_date_iso": "2022-12-31"},
]


def test_resolution_identifies_the_subject_not_just_the_period():
    # With one company in the corpus, matching on year alone was 8/8 correct. On
    # 40 companies the same code was 8% correct - an Adobe question resolved to
    # 3M's filing, because every 2022 document matched equally.
    assert catalog.resolve_for_question(MULTI, "What was Adobe's FY2022 revenue?") \
        == "ADOBE_2022_10K"
    assert catalog.resolve_for_question(MULTI, "What was 3M's FY2022 revenue?") \
        == "3M_2022_10K"


def test_subject_matching_survives_how_the_name_is_written():
    # The filing says "THE COCA-COLA COMPANY"; a question says "Coca-Cola".
    assert catalog.resolve_for_question(MULTI, "Coca-Cola's 2022 net sales?") \
        == "COCACOLA_2022_10K"
    # The filing never writes "PG&E", but the document name carries it.
    assert catalog.resolve_for_question(MULTI, "What did PG&E report in 2022?") \
        == "PG_E_2022_10K"


def test_no_recognised_subject_leaves_the_catalog_unscoped():
    # Returning nothing would be wrong for a single-subject corpus, which is the
    # common deployment; the caller decides what an unscoped resolve means.
    assert catalog.subject_candidates(MULTI, "What were revenues in 2022?") == []
    assert catalog.resolve_for_question(MULTI, "What were revenues in 2022?") in {
        d["doc_name"] for d in MULTI}


def test_longer_subject_match_wins():
    docs = [{"doc_name": "AMERICAN_2022_10K", "company": "AMERICAN AIRLINES",
             "doc_period": 2022, "doc_type": "10-K", "period_end_date_iso": "2022-12-31"},
            {"doc_name": "AMERICANWATERWORKS_2022_10K",
             "company": "AMERICAN WATER WORKS COMPANY, INC.", "doc_period": 2022,
             "doc_type": "10-K", "period_end_date_iso": "2022-12-31"}]
    assert catalog.resolve_for_question(docs, "American Water Works 2022 revenue?") \
        == "AMERICANWATERWORKS_2022_10K"
