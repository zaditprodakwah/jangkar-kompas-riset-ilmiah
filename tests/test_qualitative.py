"""
Unit tests for deterministic Qualitative Synthesis and Inter-Coder Reliability (Kluster C).
Tests:
- Cohen's Kappa (kappa)
- Fleiss' Kappa for m > 2 raters
- Krippendorff's Alpha (alpha) with nominal metric and missing data
- MMAT 2018 Mixed Methods Appraisal Tool audit
"""

import pytest
from scholarly_core.qualitative import (
    calculate_cohens_kappa,
    calculate_fleiss_kappa,
    calculate_krippendorff_alpha,
    audit_mmat_study
)


def test_cohens_kappa_perfect_and_moderate():
    # Perfect agreement
    c1 = ["Theme_A", "Theme_B", "Theme_A", "Theme_C"]
    c2 = ["Theme_A", "Theme_B", "Theme_A", "Theme_C"]
    res_perf = calculate_cohens_kappa(c1, c2)
    assert res_perf["kappa"] == 1.0
    assert res_perf["observed_agreement"] == 1.0
    assert "Almost Perfect" in res_perf["interpretation"]

    # Substantial agreement
    c3 = ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
    c4 = ["A", "B", "A", "B", "A", "A", "A", "B", "A", "B"]  # 1 disagreement
    res_sub = calculate_cohens_kappa(c3, c4)
    assert res_sub["observed_agreement"] == 0.90
    assert res_sub["kappa"] > 0.70


def test_fleiss_kappa_canonical():
    # 5 items, 3 raters, 2 categories
    matrix = [
        [3, 0],  # Item 1: all in Cat 0
        [3, 0],  # Item 2: all in Cat 0
        [0, 3],  # Item 3: all in Cat 1
        [2, 1],  # Item 4: 2 in Cat 0, 1 in Cat 1
        [1, 2],  # Item 5: 1 in Cat 0, 2 in Cat 1
    ]
    res = calculate_fleiss_kappa(matrix)
    assert res["n_items"] == 5
    assert res["n_raters"] == 3
    assert res["fleiss_kappa"] == 0.4444
    assert res["interpretation"] == "Moderate Agreement"


def test_krippendorff_alpha_nominal_with_missing_data():
    # Units x Raters matrix
    # Unit 1: rater1="A", rater2="A"
    # Unit 2: rater1="B", rater2="B"
    # Unit 3: rater1="A", rater2="B" (disagreement)
    # Unit 4: rater1="B", rater2=None (missing, pair not formed)
    # Unit 5: rater1="A", rater2="A"
    data = [
        ["A", "A"],
        ["B", "B"],
        ["A", "B"],
        ["B", None],
        ["A", "A"],
    ]
    res = calculate_krippendorff_alpha(data)
    assert res["n_units"] == 5
    assert res["k_categories"] == 2
    assert res["total_paired_comparisons"] > 0
    assert res["alpha"] > 0.40


def test_mmat_2018_qualitative_audit():
    # Test qualitative study with 5/5 criteria met
    responses = {
        "1.1": True,
        "1.2": True,
        "1.3": True,
        "1.4": True,
        "1.5": True
    }
    res = audit_mmat_study("qualitative", responses)
    assert res["criteria_evaluated"] == 5
    assert res["criteria_met"] == 5
    assert res["quality_score_percent"] == 100.0
    assert res["is_publishable_tier1"] is True
    assert "Exemplary Quality" in res["verdict"]

    # Test with deficiency
    responses_def = {
        "1.1": True,
        "1.2": True,
        "1.3": False,
        "1.4": False,
        "1.5": True
    }
    res_def = audit_mmat_study("qualitative", responses_def)
    assert res_def["criteria_met"] == 3
    assert res_def["quality_score_percent"] == 60.0
    assert res_def["is_publishable_tier1"] is False
