"""
Forensic Statistical & Evidential Validity Engine (Metascience Core)
Plugin: scientific-research-os
Module: core.compute.forensics

Deterministic implementation of:
1. Ioannidis Positive Predictive Value (PPV) & Multi-Team Model
2. Gelman & Carlin Type S (Sign) and Type M (Magnitude/Exaggeration) Errors
3. Granularity Checks: GRIM, GRIMMER, and DEBIT tests
4. Inferential Verification: statcheck exact recalculation and df invariance
5. Carlisle-Stouffer Baseline Anomaly & TIVA (Test of Insufficient Variance)
6. Newcomb-Benford First Digit Test (Nigrini MAD)
7. Z-Curve 2.0 (Expectation-Maximization for ERR & EDR)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats


# =============================================================================
# 1. EPISTEMIC & EVIDENTIAL VALIDITY MODELS
# =============================================================================

def calculate_ioannidis_ppv(
    r_odds: float,
    alpha: float = 0.05,
    beta: float = 0.20,
    u_bias: float = 0.0,
    n_teams: int = 1
) -> Dict[str, Any]:
    """
    Calculate Positive Predictive Value (PPV) per Ioannidis (2005).
    Supports researcher bias u and independent multi-team competition n.
    """
    if r_odds < 0:
        raise ValueError("r_odds (pre-study odds) must be non-negative.")
    u_bias = max(0.0, min(1.0, float(u_bias)))
    alpha = max(1e-7, min(1.0, float(alpha)))
    beta = max(0.0, min(1.0, float(beta)))
    n_teams = max(1, int(n_teams))
    power = 1.0 - beta

    if n_teams == 1:
        numerator = power * r_odds + u_bias * beta * r_odds
        denominator = power * r_odds + alpha + u_bias * beta * r_odds + u_bias * (1.0 - alpha)
        ppv = numerator / denominator if denominator > 0 else 0.0
    else:
        # Multi-team competition model
        comp_beta = (beta ** n_teams) * ((1.0 - u_bias) ** n_teams)
        comp_alpha = ((1.0 - alpha) ** n_teams) * ((1.0 - u_bias) ** n_teams)
        numerator = r_odds * (1.0 - comp_beta)
        denominator = r_odds * (1.0 - comp_beta) + 1.0 - comp_alpha
        ppv = numerator / denominator if denominator > 0 else 0.0

    fprp = 1.0 - ppv
    critical_threshold_met = (power * r_odds) > alpha

    if ppv >= 0.80:
        diagnostic_zone = "high_evidential_validity"
    elif ppv >= 0.50:
        diagnostic_zone = "marginal_validity"
    else:
        diagnostic_zone = "epistemic_failure_zone"

    return {
        "ppv": round(float(ppv), 6),
        "fprp": round(float(fprp), 6),
        "power": round(float(power), 4),
        "pre_study_odds": r_odds,
        "bias_u": u_bias,
        "n_teams": n_teams,
        "critical_threshold_met": critical_threshold_met,
        "diagnostic_zone": diagnostic_zone
    }


def calculate_type_s_m_error(
    effect_true: float,
    se: float,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate Gelman & Carlin (2014) Type S (Sign) Error and Type M (Exaggeration Ratio).
    """
    if se <= 0:
        raise ValueError("Standard error must be strictly positive.")
    if effect_true == 0:
        raise ValueError("effect_true must be non-zero to evaluate Type S/M error.")

    abs_theta = abs(float(effect_true))
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    # Statistical power for two-sided test
    power = 1.0 - stats.norm.cdf(z_crit - abs_theta / se) + stats.norm.cdf(-z_crit - abs_theta / se)

    # Type S error: probability that significant result has opposite sign
    p_neg_sig = stats.norm.cdf(-z_crit - abs_theta / se)
    p_pos_sig = 1.0 - stats.norm.cdf(z_crit - abs_theta / se)
    type_s = p_neg_sig / (p_neg_sig + p_pos_sig) if (p_neg_sig + p_pos_sig) > 0 else 0.0

    # Type M error (Exaggeration Ratio):
    # E(|X| | |X| > c) / abs_theta where X ~ N(abs_theta, se^2) and c = z_crit * se
    c_val = z_crit * se
    mu = abs_theta

    alpha_upper = (c_val - mu) / se
    p_upper = 1.0 - stats.norm.cdf(alpha_upper)
    e_upper = mu + se * (stats.norm.pdf(alpha_upper) / p_upper) if p_upper > 1e-12 else c_val

    alpha_lower = (-c_val - mu) / se
    p_lower = stats.norm.cdf(alpha_lower)
    e_lower = -mu + se * (stats.norm.pdf(alpha_lower) / p_lower) if p_lower > 1e-12 else c_val

    total_sig_p = p_upper + p_lower
    if total_sig_p > 1e-12:
        expected_abs_estimate = (p_upper * e_upper + p_lower * e_lower) / total_sig_p
    else:
        expected_abs_estimate = c_val

    type_m = expected_abs_estimate / abs_theta

    adequate_sample = bool((type_m <= 1.20) and (type_s < 0.001))

    return {
        "effect_true": effect_true,
        "se": se,
        "power": round(float(power), 4),
        "type_s_error": round(float(type_s), 6),
        "type_m_exaggeration_ratio": round(float(type_m), 4),
        "adequate_sample": adequate_sample
    }


# =============================================================================
# 2. GRANULARITY & COMBINATORIC FORENSICS (GRIM, GRIMMER, DEBIT)
# =============================================================================

def run_grim_test(
    mean: float,
    sample_size: int,
    items: int = 1,
    decimals: int = 2
) -> Dict[str, Any]:
    """
    Granularity-Related Inconsistency of Means (GRIM) Test (Brown & Heathers, 2017).
    Applicable when sample_size * items < 10^decimals.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if items <= 0:
        raise ValueError("items must be positive.")

    n_eff = sample_size * items
    testable = n_eff < (10 ** decimals)

    delta = 0.5 * (10 ** (-decimals))
    s_min = math.ceil(n_eff * (mean - delta))
    s_max = math.floor(n_eff * (mean + delta))

    consistent = s_min <= s_max

    return {
        "test": "GRIM",
        "mean_reported": mean,
        "sample_size": sample_size,
        "items": items,
        "decimals": decimals,
        "n_effective": n_eff,
        "testable": testable,
        "s_min": s_min,
        "s_max": s_max,
        "consistent": consistent,
        "status": "GRIM_CONSISTENT" if consistent else "GRIM_INCONSISTENT"
    }


def run_grimmer_test(
    mean: float,
    sd: float,
    sample_size: int,
    items: int = 1,
    decimals: int = 2
) -> Dict[str, Any]:
    """
    GRIMMER Test (Anaya 2016, Allard 2018).
    Tests consistency of reported Mean and SD on integer scales using variance & parity.
    """
    if sample_size <= 1:
        raise ValueError("sample_size must be at least 2 for SD evaluation.")

    # Step 1: Run GRIM
    grim_res = run_grim_test(mean, sample_size, items, decimals)
    if not grim_res["consistent"]:
        return {
            "test": "GRIMMER",
            "consistent": False,
            "status": "GRIMMER_INCONSISTENT_GRIM_FAILED",
            "candidates_evaluated": 0,
            "reason": "Failed initial GRIM mean consistency test."
        }

    n = sample_size * items
    delta = 0.5 * (10 ** (-decimals))
    sd_low = max(0.0, sd - delta)
    sd_high = sd + delta

    candidates_found = False
    valid_candidates: List[Tuple[int, int]] = []

    for s_cand in range(grim_res["s_min"], grim_res["s_max"] + 1):
        q_min = math.ceil((n - 1) * (sd_low ** 2) + (s_cand ** 2) / n)
        q_max = math.floor((n - 1) * (sd_high ** 2) + (s_cand ** 2) / n)

        for q_cand in range(q_min, q_max + 1):
            if (q_cand % 2) == (s_cand % 2) and q_cand >= math.ceil((s_cand ** 2) / n):
                candidates_found = True
                valid_candidates.append((s_cand, q_cand))
                break
        if candidates_found:
            break

    return {
        "test": "GRIMMER",
        "mean_reported": mean,
        "sd_reported": sd,
        "sample_size": sample_size,
        "items": items,
        "consistent": candidates_found,
        "status": "GRIMMER_CONSISTENT" if candidates_found else "GRIMMER_INCONSISTENT",
        "valid_pairs_count": len(valid_candidates)
    }


def run_debit_test(
    p_mean: float,
    sd_reported: float,
    sample_size: int,
    decimals: int = 2
) -> Dict[str, Any]:
    """
    Descriptive Binary Test (DEBIT) (Heathers et al., 2019).
    For binary variables (0/1), SD is mathematically tied to mean p:
    SD = sqrt(N / (N - 1) * p * (1 - p)).
    """
    if sample_size <= 1:
        raise ValueError("sample_size must be at least 2.")
    if not (0.0 <= p_mean <= 1.0):
        return {
            "test": "DEBIT",
            "consistent": False,
            "status": "DEBIT_INCONSISTENT_PROPORTION_OUT_OF_BOUNDS",
            "reason": f"Proportion {p_mean} is outside [0, 1]."
        }

    delta = 0.5 * (10 ** (-decimals))
    c_min = math.ceil(sample_size * (p_mean - delta))
    c_max = math.floor(sample_size * (p_mean + delta))

    if c_min > c_max:
        return {
            "test": "DEBIT",
            "consistent": False,
            "status": "DEBIT_INCONSISTENT_MEAN_GRIM_FAILED",
            "reason": "Binary count c does not have valid integer representation."
        }

    consistent = False
    expected_sds = []
    for c_cand in range(c_min, c_max + 1):
        p_cand = c_cand / sample_size
        theo_sd = math.sqrt((sample_size / (sample_size - 1)) * p_cand * (1.0 - p_cand))
        expected_sds.append(round(theo_sd, decimals))
        if abs(theo_sd - sd_reported) <= (delta + 1e-9):
            consistent = True
            break

    return {
        "test": "DEBIT",
        "p_mean": p_mean,
        "sd_reported": sd_reported,
        "sample_size": sample_size,
        "consistent": consistent,
        "expected_sd_range": [min(expected_sds), max(expected_sds)] if expected_sds else [],
        "status": "DEBIT_CONSISTENT" if consistent else "DEBIT_INCONSISTENT"
    }


# =============================================================================
# 3. INFERENTIAL RECALCULATION & INVARIANCE (STATCHECK & BASELINE)
# =============================================================================

def recalculate_statcheck(
    test_type: str,
    test_stat: float,
    df1: float,
    df2: Optional[float] = None,
    p_reported: float = 0.05,
    alpha: float = 0.05,
    stat_decimals: int = 2
) -> Dict[str, Any]:
    """
    Statcheck recalculation of test statistic and p-value per Nuijten et al. (2016).
    Detects inconsistencies and significance-flipping gross errors.
    """
    t_type = test_type.lower().strip()
    stat_delta = 0.5 * (10 ** (-stat_decimals))
    stat_min = test_stat - stat_delta
    stat_max = test_stat + stat_delta

    def compute_p(stat_val: float) -> float:
        s = abs(stat_val)
        if t_type == "t":
            return float(2.0 * (1.0 - stats.t.cdf(s, df=df1)))
        elif t_type in ("f", "anova"):
            if df2 is None:
                raise ValueError("df2 is required for F-test.")
            return float(1.0 - stats.f.cdf(stat_val, dfn=df1, dfd=df2))
        elif t_type in ("chi2", "chisq", "chi_square"):
            return float(1.0 - stats.chi2.cdf(stat_val, df=df1))
        elif t_type == "r":
            if df1 <= 2:
                return 1.0
            r_val = max(-0.999999, min(0.999999, stat_val))
            t_val = r_val * math.sqrt((df1 - 2) / (1.0 - r_val ** 2))
            return float(2.0 * (1.0 - stats.t.cdf(abs(t_val), df=df1 - 2)))
        elif t_type == "z":
            return float(2.0 * (1.0 - stats.norm.cdf(s)))
        else:
            raise ValueError(f"Unsupported test type: {test_type}")

    p_exact = compute_p(test_stat)
    p_min = min(compute_p(stat_min), compute_p(stat_max))
    p_max = max(compute_p(stat_min), compute_p(stat_max))

    p_rep_margin = 0.005 if p_reported > 0.01 else 0.0005
    consistent = (p_reported >= (p_min - p_rep_margin)) and (p_reported <= (p_max + p_rep_margin))

    rep_sig = (p_reported <= alpha)
    act_sig = (p_exact <= alpha)
    gross_error = (rep_sig != act_sig)

    return {
        "test_type": t_type,
        "test_stat": test_stat,
        "df1": df1,
        "df2": df2,
        "p_reported": p_reported,
        "p_computed_exact": round(p_exact, 6),
        "p_range": [round(p_min, 6), round(p_max, 6)],
        "consistent": consistent,
        "gross_error": gross_error,
        "status": "STATCHECK_GROSS_ERROR" if gross_error else ("STATCHECK_CONSISTENT" if consistent else "STATCHECK_INCONSISTENT")
    }


def run_carlisle_stouffer_test(p_values: List[float]) -> Dict[str, Any]:
    """
    Carlisle-Stouffer baseline anomaly detection for RCT Table 1 (Carlisle, 2017).
    Under true random allocation, baseline p-values follow Uniform(0, 1).
    """
    if not p_values:
        raise ValueError("p_values list cannot be empty.")

    clean_ps = [max(1e-7, min(1.0 - 1e-7, float(p))) for p in p_values]
    m_vars = len(clean_ps)

    z_scores = [stats.norm.ppf(p) for p in clean_ps]
    z_carlisle = sum(z_scores) / math.sqrt(m_vars)
    p_global = float(stats.norm.cdf(z_carlisle))

    is_anomaly = (p_global > 0.999) or (p_global < 0.001)

    return {
        "test": "Carlisle-Stouffer",
        "m_variables": m_vars,
        "z_carlisle": round(float(z_carlisle), 4),
        "p_global": round(p_global, 6),
        "is_anomaly": is_anomaly,
        "diagnostic": (
            "SUSPICIOUS_UNDERDISPERSION_FABRICATION" if p_global > 0.999
            else ("SUSPICIOUS_OVERDISPERSION_RANDOMIZATION_FAILURE" if p_global < 0.001 else "NORMAL_RANDOMIZATION")
        )
    }


def run_tiva_test(z_scores: List[float]) -> Dict[str, Any]:
    """
    Test of Insufficient Variance (TIVA) per Schimmack (2014).
    Detects cherry-picking/file-drawer bias across studies in a single multi-study paper.
    """
    k_studies = len(z_scores)
    if k_studies < 3:
        raise ValueError("TIVA requires at least 3 independent studies/z-scores.")

    z_arr = np.array(z_scores, dtype=float)
    var_z = float(np.var(z_arr, ddof=1))
    chi2_stat = (k_studies - 1) * var_z

    p_tiva = float(stats.chi2.cdf(chi2_stat, df=k_studies - 1))
    insufficient_variance = p_tiva < 0.05

    return {
        "test": "TIVA",
        "k_studies": k_studies,
        "variance_z": round(var_z, 4),
        "chi2_stat": round(chi2_stat, 4),
        "p_tiva_left_tail": round(p_tiva, 6),
        "insufficient_variance": insufficient_variance,
        "status": "SUSPICIOUS_INSUFFICIENT_VARIANCE" if insufficient_variance else "NATURAL_VARIANCE"
    }


def run_benford_first_digit_test(values: List[float]) -> Dict[str, Any]:
    """
    Newcomb-Benford First Digit Test with Nigrini Mean Absolute Deviation (MAD).
    """
    first_digits = []
    for val in values:
        v = abs(float(val))
        if v == 0:
            continue
        s = f"{v:.10e}"
        d1 = int(s[0])
        if 1 <= d1 <= 9:
            first_digits.append(d1)

    n_digits = len(first_digits)
    if n_digits < 30:
        return {
            "test": "Benford",
            "n_observations": n_digits,
            "testable": False,
            "reason": f"Sample size {n_digits} < 30 (minimum sample threshold for Benford test)."
        }

    counts = {d: 0 for d in range(1, 10)}
    for d in first_digits:
        counts[d] += 1

    observed_props = {d: counts[d] / n_digits for d in range(1, 10)}
    benford_props = {d: math.log10(1.0 + 1.0 / d) for d in range(1, 10)}

    mad = sum(abs(observed_props[d] - benford_props[d]) for d in range(1, 10)) / 9.0

    if mad <= 0.006:
        conformity = "close_conformity"
    elif mad <= 0.012:
        conformity = "acceptable_conformity"
    elif mad <= 0.015:
        conformity = "marginal_conformity"
    else:
        conformity = "non_conformity"

    return {
        "test": "Benford",
        "n_observations": n_digits,
        "testable": True,
        "mad": round(float(mad), 6),
        "conformity": conformity,
        "observed_proportions": {d: round(observed_props[d], 4) for d in range(1, 10)},
        "benford_proportions": {d: round(benford_props[d], 4) for d in range(1, 10)}
    }


# =============================================================================
# 4. Z-CURVE 2.0 (EM ALGORITHM FOR REPLICATION & DISCOVERY RATES)
# =============================================================================

def fit_zcurve_em(
    z_scores: List[float],
    sig_threshold: float = 1.96,
    max_iter: int = 150
) -> Dict[str, Any]:
    """
    Z-Curve 2.0 per Bartoš & Schimmack (2022).
    Fits truncated folded normal mixture model via EM algorithm to estimate:
    - Expected Replication Rate (ERR)
    - Expected Discovery Rate (EDR)
    - Publication Bias Gap (ODR - EDR)
    """
    sig_z = [abs(float(z)) for z in z_scores if abs(float(z)) >= sig_threshold]
    n_sig = len(sig_z)
    total_z = len(z_scores)

    if n_sig < 5:
        return {
            "test": "Z-Curve 2.0",
            "testable": False,
            "reason": f"Only {n_sig} significant z-scores found (minimum 5 required)."
        }

    odr = n_sig / total_z if total_z > 0 else 1.0

    mu_components = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    j_comps = len(mu_components)
    c = sig_threshold

    p_sig_comp = np.array([
        (1.0 - stats.norm.cdf(c - m) + stats.norm.cdf(-c - m))
        for m in mu_components
    ])
    p_sig_comp = np.maximum(p_sig_comp, 1e-10)

    z_arr = np.array(sig_z)
    f_density = np.zeros((n_sig, j_comps))
    for j, m in enumerate(mu_components):
        f_density[:, j] = stats.norm.pdf(z_arr - m) + stats.norm.pdf(z_arr + m)

    weights = np.ones(j_comps) / j_comps

    for _ in range(max_iter):
        scaled_densities = (f_density / p_sig_comp) * weights
        row_sums = scaled_densities.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        gamma = scaled_densities / row_sums

        new_weights = gamma.mean(axis=0)
        if np.max(np.abs(new_weights - weights)) < 1e-5:
            weights = new_weights
            break
        weights = new_weights

    err = float(np.sum(weights * p_sig_comp))
    edr_denom = float(np.sum(weights / p_sig_comp))
    edr = (1.0 / edr_denom) if edr_denom > 0 else 0.0
    pub_bias_gap = max(0.0, odr - edr)

    return {
        "test": "Z-Curve 2.0",
        "testable": True,
        "n_significant": n_sig,
        "n_total": total_z,
        "observed_discovery_rate": round(odr, 4),
        "expected_replication_rate": round(err, 4),
        "expected_discovery_rate": round(edr, 4),
        "publication_bias_gap": round(pub_bias_gap, 4),
        "component_weights": [round(float(w), 4) for w in weights]
    }
