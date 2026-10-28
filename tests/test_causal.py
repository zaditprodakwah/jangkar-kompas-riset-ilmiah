"""
Unit tests for deterministic Causal Inference and Experimental Metrology (Kluster B).
Tests:
- Hake's Normalized Gain (<g>)
- Nissen et al. (2018) structural bias detection
- Paired Cohen's d and Hedges' g
- 2x2 Difference-in-Differences (DiD) Causal Estimator
- Parallel Trends Pre-Test
"""

import pytest
from scholarly_core.causal import calculate_hake_gain, calculate_did_2x2, evaluate_parallel_trends


def test_hake_gain_canonical_values():
    # Pre: 40, 50, 60 (mean=50), Post: 80, 85, 90 (mean=85), max=100
    # Gain = (85 - 50) / (100 - 50) = 35 / 50 = 0.70 (High Gain)
    pre = [40.0, 50.0, 60.0]
    post = [80.0, 85.0, 90.0]
    res = calculate_hake_gain(pre, post, max_score=100.0)

    assert res["n_students"] == 3
    assert res["mean_pre"] == 50.0
    assert res["mean_post"] == 85.0
    assert res["average_normalized_gain"] == 0.70
    assert res["hake_category"] == "High Gain"
    assert res["cohen_d_paired"] > 0
    assert res["hedges_g_paired"] > 0


def test_hake_gain_nissen_bias_alert():
    # Asymmetric distribution causing divergence between <g> and mean(g_i)
    pre = [10.0, 90.0]
    post = [80.0, 95.0]
    # Student 1: (80-10)/(100-10) = 70/90 = 0.7778
    # Student 2: (95-90)/(100-90) = 5/10 = 0.5000
    # Mean of individual gains = (0.7778 + 0.5000) / 2 = 0.6389
    # Average gain: mean(pre)=50, mean(post)=87.5 -> (87.5 - 50)/(100 - 50) = 37.5 / 50 = 0.7500
    # Divergence: |0.7500 - 0.6389| = 0.1111 > 0.05!
    res = calculate_hake_gain(pre, post, max_score=100.0)

    assert res["has_bias_alert"] is True
    assert res["bias_divergence"] > 0.05
    assert "Nissen et al. (2018)" in res["bias_diagnostic"]


def test_did_2x2_causal_effect():
    # Treatment group: starts at 50, jumps to 85 (Delta_T = +35)
    # Control group: starts at 50, rises to 60 naturally (Delta_C = +10)
    # True DiD = 35 - 10 = +25
    treat_pre = [48.0, 52.0, 50.0, 50.0]
    treat_post = [83.0, 87.0, 85.0, 85.0]
    control_pre = [50.0, 50.0, 50.0, 50.0]
    control_post = [60.0, 60.0, 60.0, 60.0]

    res = calculate_did_2x2(treat_pre, treat_post, control_pre, control_post)

    assert res["delta_treatment"] == 35.0
    assert res["delta_control"] == 10.0
    assert res["did_estimator"] == 25.0
    assert res["standard_error"] > 0
    assert res["t_statistic"] > 5.0
    assert res["p_value"] < 0.001
    assert res["is_significant_05"] is True
    assert res["percent_impact"] > 0


def test_parallel_trends_verification():
    # Both groups trend upward identically before intervention
    # T1 -> T2 -> T3: treat rises +5, +5; control rises +5, +5
    treat_history = [
        [40.0, 42.0],
        [45.0, 47.0],
        [50.0, 52.0]
    ]
    control_history = [
        [30.0, 32.0],
        [35.0, 37.0],
        [40.0, 42.0]
    ]
    res = evaluate_parallel_trends(treat_history, control_history)

    assert res["pre_treatment_periods"] == 3
    assert res["mean_divergence"] == 0.0
    assert res["parallel_trends_tenable"] is True
    assert "tenable" in res["verdict"]
