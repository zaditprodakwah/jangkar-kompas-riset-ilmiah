"""
Unit Tests for Forensic Statistical & Evidential Validity Engine
Plugin: scientific-research-os
Module: tests.test_forensics
"""

import pytest
import numpy as np
from scholarly_core.forensics import (
    calculate_ioannidis_ppv,
    calculate_type_s_m_error,
    run_grim_test,
    run_grimmer_test,
    run_debit_test,
    recalculate_statcheck,
    run_carlisle_stouffer_test,
    run_tiva_test,
    run_benford_first_digit_test,
    fit_zcurve_em
)


def test_ioannidis_ppv_single_team():
    # Power = 0.80, alpha = 0.05, R = 1.0 (50/50 prior), u = 0
    res = calculate_ioannidis_ppv(r_odds=1.0, alpha=0.05, beta=0.20, u_bias=0.0, n_teams=1)
    # Expected: 0.80 / (0.80 + 0.05) = 0.80 / 0.85 = 0.941176
    assert abs(res["ppv"] - 0.941176) < 1e-4
    assert res["critical_threshold_met"] is True
    assert res["diagnostic_zone"] == "high_evidential_validity"

    # Low prior R = 0.1, high bias u = 0.3
    res_biased = calculate_ioannidis_ppv(r_odds=0.1, alpha=0.05, beta=0.50, u_bias=0.3, n_teams=1)
    assert res_biased["ppv"] < 0.50
    assert res_biased["diagnostic_zone"] == "epistemic_failure_zone"


def test_ioannidis_ppv_multi_team():
    # 5 independent teams chasing the same hypothesis
    res_multi = calculate_ioannidis_ppv(r_odds=0.2, alpha=0.05, beta=0.20, u_bias=0.1, n_teams=5)
    assert 0.0 <= res_multi["ppv"] <= 1.0
    assert "fprp" in res_multi


def test_gelman_type_s_m_error():
    # True effect = 0.1, SE = 0.1 (low power ~17%)
    res = calculate_type_s_m_error(effect_true=0.1, se=0.1, alpha=0.05)
    assert res["power"] < 0.25
    assert res["type_m_exaggeration_ratio"] > 1.5  # Substantial exaggeration
    assert res["adequate_sample"] is False

    # Large effect = 2.0, SE = 0.2 (high power ~100%)
    res_large = calculate_type_s_m_error(effect_true=2.0, se=0.2, alpha=0.05)
    assert res_large["power"] > 0.99
    assert res_large["type_m_exaggeration_ratio"] <= 1.05
    assert res_large["type_s_error"] < 0.0001
    assert res_large["adequate_sample"] is True


def test_grim_test():
    # Sample size 20, 1 item.
    # A mean of 2.11 is algebraically impossible: 20 * 2.11 = 42.2.
    # 20 * (2.11 - 0.005) = 42.1 -> ceil = 43.
    # 20 * (2.11 + 0.005) = 42.3 -> floor = 42.
    # S_min (43) > S_max (42) -> GRIM inconsistent!
    res_fail = run_grim_test(mean=2.11, sample_size=20, items=1, decimals=2)
    assert res_fail["consistent"] is False
    assert res_fail["status"] == "GRIM_INCONSISTENT"

    # Mean of 2.10 is possible: 20 * 2.10 = 42.0 (integer sum 42)
    res_pass = run_grim_test(mean=2.10, sample_size=20, items=1, decimals=2)
    assert res_pass["consistent"] is True
    assert res_pass["status"] == "GRIM_CONSISTENT"


def test_grimmer_test():
    # Sample size 20, mean 2.10, SD 0.50
    res = run_grimmer_test(mean=2.10, sd=0.50, sample_size=20, items=1, decimals=2)
    assert "consistent" in res
    assert res["test"] == "GRIMMER"


def test_debit_test():
    # N = 100, binary proportion p = 0.50 -> expected SD = sqrt(100/99 * 0.5 * 0.5) = 0.5025 ~ 0.50
    res_pass = run_debit_test(p_mean=0.50, sd_reported=0.50, sample_size=100, decimals=2)
    assert res_pass["consistent"] is True

    # Impossible SD for binary proportion: p = 0.50, reported SD = 0.20
    res_fail = run_debit_test(p_mean=0.50, sd_reported=0.20, sample_size=100, decimals=2)
    assert res_fail["consistent"] is False
    assert res_fail["status"] == "DEBIT_INCONSISTENT"


def test_statcheck_recalculation():
    # Valid t-test: t(28) = 2.05 -> exact p ~ 0.0498
    res_t = recalculate_statcheck(test_type="t", test_stat=2.05, df1=28, p_reported=0.05, alpha=0.05)
    assert res_t["consistent"] is True
    assert res_t["gross_error"] is False

    # Gross error: reported p = 0.04 (significant), but actual t(28) = 1.20 has p ~ 0.24 (non-significant)
    res_gross = recalculate_statcheck(test_type="t", test_stat=1.20, df1=28, p_reported=0.04, alpha=0.05)
    assert res_gross["gross_error"] is True
    assert res_gross["status"] == "STATCHECK_GROSS_ERROR"

    # F-test: F(2, 60) = 4.50 -> exact p ~ 0.015
    res_f = recalculate_statcheck(test_type="f", test_stat=4.50, df1=2, df2=60, p_reported=0.015)
    assert res_f["consistent"] is True


def test_carlisle_stouffer_test():
    # Natural baseline: independent uniform p-values
    p_natural = [0.45, 0.62, 0.21, 0.78, 0.33, 0.55, 0.89]
    res_nat = run_carlisle_stouffer_test(p_natural)
    assert res_nat["is_anomaly"] is False
    assert res_nat["diagnostic"] == "NORMAL_RANDOMIZATION"

    # Severe under-dispersion (fabricated identical groups, p values all extremely close to 1)
    p_suspicious = [0.999, 0.998, 0.999, 0.997, 0.999]
    res_susp = run_carlisle_stouffer_test(p_suspicious)
    assert res_susp["is_anomaly"] is True
    assert res_susp["diagnostic"] == "SUSPICIOUS_UNDERDISPERSION_FABRICATION"


def test_tiva_test():
    # Highly variable z-scores
    z_natural = [2.1, 3.5, 1.8, 4.2, 2.7, 1.9, 3.1]
    res_nat = run_tiva_test(z_natural)
    assert res_nat["insufficient_variance"] is False

    # Suspiciously identical z-scores (variance ~ 0)
    z_suspicious = [2.01, 2.02, 2.00, 2.01, 2.02, 2.01]
    res_susp = run_tiva_test(z_suspicious)
    assert res_susp["insufficient_variance"] is True
    assert res_susp["status"] == "SUSPICIOUS_INSUFFICIENT_VARIANCE"


def test_benford_first_digit_test():
    # Generate data following Benford's law: powers of 2
    benford_data = [2 ** i for i in range(1, 100)]
    res = run_benford_first_digit_test(benford_data)
    assert res["testable"] is True
    assert res["conformity"] in ("close_conformity", "acceptable_conformity")


def test_zcurve_em():
    # Sample of z-scores with power and some publication selection
    np.random.seed(42)
    # Mixture of nulls and true effects
    z_sample = list(np.random.normal(loc=0.0, scale=1.0, size=50)) + list(np.random.normal(loc=2.5, scale=1.0, size=50))
    res = fit_zcurve_em(z_sample, sig_threshold=1.96)
    assert res["testable"] is True
    assert 0.0 <= res["expected_replication_rate"] <= 1.0
    assert 0.0 <= res["expected_discovery_rate"] <= 1.0
    assert res["publication_bias_gap"] >= 0.0
