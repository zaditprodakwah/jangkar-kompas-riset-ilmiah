"""
Unit Tests for Psychometric Validity Engine.
Validates:
- Aiken's V exact coefficient
- Exact p-value discrete polynomial convolution against analytical combinatorial truth
- Gwet's AC1 chance-corrected agreement
- Bootstrap confidence intervals
- Item-matrix batch auditing
"""

import pytest
from scholarly_core.validity import (
    calculate_aiken_v,
    calculate_aiken_exact_p_value,
    get_aiken_critical_value,
    calculate_gwet_ac1,
    bootstrap_aiken_ci,
    audit_single_item,
    audit_psychometric_matrix
)


def test_aiken_v_perfect_and_zero():
    # Perfect score: all 5s on 1-5 scale -> V = 1.0
    assert calculate_aiken_v([5, 5, 5, 5, 5], lo=1, hi=5) == 1.0
    # Minimum score: all 1s on 1-5 scale -> V = 0.0
    assert calculate_aiken_v([1, 1, 1, 1, 1], lo=1, hi=5) == 0.0
    # Mid-range: n=6, c=5 (denom = 24), sum(r-1) = 20 -> V = 20/24 = 0.8333
    assert calculate_aiken_v([5, 5, 4, 4, 4, 4], lo=1, hi=5) == 0.8333


def test_exact_p_value_combinatorial_ground_truth():
    # Scenario: n = 2 raters, c = 5 categories (ratings 1 to 5)
    # Total combinations = 5^2 = 25.
    
    # 1. Ratings [5, 5]: s_obs = 4 + 4 = 8 (max possible).
    # Only (4, 4) achieves s = 8.
    # Expected p = 1 / 25 = 0.040000 -> significant at alpha = 0.05
    p_perfect = calculate_aiken_exact_p_value([5, 5], lo=1, hi=5)
    assert p_perfect == 0.04

    # 2. Ratings [5, 4]: s_obs = 4 + 3 = 7.
    # Combinations with s >= 7: (4, 4) [s=8], (4, 3) [s=7], (3, 4) [s=7] -> exactly 3.
    # Expected p = 3 / 25 = 0.120000 -> not significant at alpha = 0.05
    p_slight_drop = calculate_aiken_exact_p_value([5, 4], lo=1, hi=5)
    assert p_slight_drop == 0.12

    # 3. Ratings [4, 4]: s_obs = 3 + 3 = 6.
    # Combinations with s >= 6: (4, 4) [s=8], (4, 3), (3, 4) [s=7], (4, 2), (2, 4), (3, 3) [s=6] -> 6.
    # Expected p = 6 / 25 = 0.240000
    p_mid = calculate_aiken_exact_p_value([4, 4], lo=1, hi=5)
    assert p_mid == 0.24


def test_exact_p_value_table_c2_scenarios():
    # Table C.2 Line 4: 6 raters with strong consensus [5, 5, 5, 5, 5, 4]
    p_dewan = calculate_aiken_exact_p_value([5, 5, 5, 5, 5, 4], lo=1, hi=5)
    # p should be very small (< 0.001)
    assert p_dewan < 0.001

    # Polarization: [5, 5, 5, 2, 2, 2]
    # S = 4+4+4+1+1+1 = 15, V = 15 / 24 = 0.625
    p_polar = calculate_aiken_exact_p_value([5, 5, 5, 2, 2, 2], lo=1, hi=5)
    # Should not be statistically significant at alpha = 0.05
    assert p_polar > 0.05


def test_gwet_ac1_behavior():
    # Perfect agreement among 6 raters
    assert calculate_gwet_ac1([5, 5, 5, 5, 5, 5], lo=1, hi=5) == 1.0

    # High agreement: 5 raters give 5, 1 gives 4 (20 of 30 pairs agree -> P_a = 0.667)
    ac1_high = calculate_gwet_ac1([5, 5, 5, 5, 5, 4], lo=1, hi=5)
    assert ac1_high > 0.60


    # Bimodal split: 3 give 5, 3 give 1 -> AC1 = 0.3143
    ac1_split = calculate_gwet_ac1([5, 5, 5, 1, 1, 1], lo=1, hi=5)
    assert 0.30 <= ac1_split <= 0.33

    # Total disagreement: 5 different ratings [1, 2, 3, 4, 5] -> P_a = 0 -> AC1 < 0
    ac1_disagree = calculate_gwet_ac1([1, 2, 3, 4, 5], lo=1, hi=5)
    assert ac1_disagree < 0.0



def test_bootstrap_ci_monotonicity():
    ratings = [5, 5, 4, 4, 4, 4]
    ci_lo, ci_hi = bootstrap_aiken_ci(ratings, lo=1, hi=5, n_resamples=1000, seed=42)
    assert 0.0 <= ci_lo <= ci_hi <= 1.0
    # Mean V is 0.8333, so CI should comfortably contain it
    assert ci_lo <= 0.8333 <= ci_hi


def test_audit_single_item_and_matrix():
    # Audit single item
    res = audit_single_item([5, 5, 5, 5, 5, 4], item_id="BUTIR-01", lo=1, hi=5)
    assert res["is_valid"] is True
    assert res["verdict"] == "VALID_SIGNIFIKAN"
    assert res["aiken_v"] == 0.9583

    # Audit matrix of 2 items
    matrix = [
        [5, 5, 5, 5, 5, 4], # valid
        [2, 2, 2, 2, 2, 1], # invalid
    ]
    report = audit_psychometric_matrix(matrix, lo=1, hi=5, n_resamples=500)
    assert report["summary"]["total_items"] == 2
    assert report["summary"]["valid_items"] == 1
    assert report["summary"]["invalid_items"] == 1
    assert report["summary"]["overall_status"] == "PERLU_REVISI_BUTIR"
