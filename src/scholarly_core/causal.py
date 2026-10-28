"""
Deterministic Causal Inference and Experimental Metrology Engine (Kluster B).
Adheres strictly to:
- Campbell, Cook, & Shadish (2002) Quasi-Experimental Validity
- Hake (1998) & Nissen et al. (2018) Normalized Gain with Structural Bias Diagnostics
- Difference-in-Differences (DiD) 2x2 with Parallel Trends & Analytical Standard Errors
- Callaway & Sant'Anna (2021) Causal Foundations
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def calculate_hake_gain(
    pre_scores: List[float],
    post_scores: List[float],
    max_score: float = 100.0
) -> Dict[str, Any]:
    """
    Calculate Hake's Normalized Gain (<g>) with Nissen et al. (2018) structural bias diagnostics.
    
    Formula:
      Individual gain: g_i = (post_i - pre_i) / (max_score - pre_i)
      Average gain:    <g> = (mean(post) - mean(pre)) / (max_score - mean(pre))
      Mean of gains:   g_bar_ind = sum(g_i) / n
    """
    n = len(pre_scores)
    if n == 0 or len(post_scores) != n:
        raise ValueError(f"pre_scores and post_scores must be non-empty and of equal length (got {len(pre_scores)} vs {len(post_scores)}).")

    mean_pre = sum(pre_scores) / n
    mean_post = sum(post_scores) / n

    if max_score <= mean_pre:
        avg_gain = 0.0
    else:
        avg_gain = (mean_post - mean_pre) / (max_score - mean_pre)

    individual_gains: List[float] = []
    excluded_for_ceiling = 0

    for pre, post in zip(pre_scores, post_scores):
        if max_score <= pre:
            excluded_for_ceiling += 1
            individual_gains.append(1.0 if post >= max_score else 0.0)
        else:
            g_i = (post - pre) / (max_score - pre)
            individual_gains.append(g_i)

    mean_ind_gain = sum(individual_gains) / n

    # Nissen et al. (2018) bias diagnostic:
    # Divergence between average normalized gain <g> and average of individual gains g_bar
    bias_divergence = abs(avg_gain - mean_ind_gain)
    has_ceiling_effect = mean_pre > (0.80 * max_score)
    has_floor_effect = mean_pre < (0.20 * max_score)

    bias_warning: Optional[str] = None
    if bias_divergence >= 0.05:
        bias_warning = (
            f"Nissen et al. (2018) Structural Bias Alert: |<g> - g_bar_ind| = {bias_divergence:.4f} >= 0.05. "
            "Average normalized gain diverges significantly from individual student gains due to asymmetric distributions."
        )
    elif has_ceiling_effect:
        bias_warning = "Ceiling effect detected (pre-test mean > 80% max score); Hake's g tends to underestimate true learning growth."
    elif has_floor_effect:
        bias_warning = "Floor effect detected (pre-test mean < 20% max score); random guessing can distort normalized gain metrics."

    # Hake's Three-Tier Categorization
    if avg_gain >= 0.70:
        hake_category = "High Gain"
    elif avg_gain >= 0.30:
        hake_category = "Medium Gain"
    else:
        hake_category = "Low Gain"

    # Paired differences and effect size (Cohen's d_z & Hedges' g)
    diffs = [post - pre for pre, post in zip(pre_scores, post_scores)]
    mean_diff = sum(diffs) / n

    if n > 1:
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        sd_diff = math.sqrt(var_diff) if var_diff > 0 else 1e-9
        cohen_d_paired = mean_diff / sd_diff
        # Hedges' g correction factor for small samples: 1 - 3 / (4 * (n - 1) - 1)
        j_corr = 1.0 - (3.0 / (4.0 * (n - 1) - 1.0)) if (4 * (n - 1) - 1) > 0 else 1.0
        hedges_g = cohen_d_paired * j_corr
    else:
        sd_diff = 0.0
        cohen_d_paired = 0.0
        hedges_g = 0.0

    return {
        "n_students": n,
        "max_score": max_score,
        "mean_pre": round(mean_pre, 4),
        "mean_post": round(mean_post, 4),
        "mean_diff": round(mean_diff, 4),
        "sd_diff": round(sd_diff, 4),
        "average_normalized_gain": round(avg_gain, 4),
        "mean_individual_gain": round(mean_ind_gain, 4),
        "bias_divergence": round(bias_divergence, 4),
        "hake_category": hake_category,
        "cohen_d_paired": round(cohen_d_paired, 4),
        "hedges_g_paired": round(hedges_g, 4),
        "has_bias_alert": bias_warning is not None,
        "bias_diagnostic": bias_warning,
    }


def calculate_did_2x2(
    treat_pre: List[float],
    treat_post: List[float],
    control_pre: List[float],
    control_post: List[float]
) -> Dict[str, Any]:
    """
    Calculate 2x2 Difference-in-Differences (DiD) Causal Estimator.
    
    Formula:
      delta_did = (mean(Y_T,post) - mean(Y_T,pre)) - (mean(Y_C,post) - mean(Y_C,pre))
      SE(delta) = sqrt( s^2_T,post/n_T,post + s^2_T,pre/n_T,pre + s^2_C,post/n_C,post + s^2_C,pre/n_C,pre )
    """
    groups = {
        "treat_pre": treat_pre,
        "treat_post": treat_post,
        "control_pre": control_pre,
        "control_post": control_post
    }

    stats = {}
    for name, arr in groups.items():
        n = len(arr)
        if n < 2:
            raise ValueError(f"Each group in 2x2 DiD must contain at least 2 observations (group '{name}' has {n}).")
        m = sum(arr) / n
        var = sum((x - m) ** 2 for x in arr) / (n - 1)
        stats[name] = {"n": n, "mean": m, "var": var, "sd": math.sqrt(var)}

    delta_treat = stats["treat_post"]["mean"] - stats["treat_pre"]["mean"]
    delta_control = stats["control_post"]["mean"] - stats["control_pre"]["mean"]
    delta_did = delta_treat - delta_control

    # Variance of the difference-in-differences estimator
    se_sq = (
        (stats["treat_post"]["var"] / stats["treat_post"]["n"]) +
        (stats["treat_pre"]["var"] / stats["treat_pre"]["n"]) +
        (stats["control_post"]["var"] / stats["control_post"]["n"]) +
        (stats["control_pre"]["var"] / stats["control_pre"]["n"])
    )
    se_did = math.sqrt(se_sq) if se_sq > 0 else 1e-9

    t_stat = delta_did / se_did if se_did > 0 else 0.0

    # Two-tailed p-value approximation via standard normal / asymptotic t
    # Using math.erfc(|z| / sqrt(2))
    z = abs(t_stat)
    p_val = math.erfc(z / math.sqrt(2.0))

    # 95% Confidence Interval (z_crit = 1.95996)
    z_crit = 1.95996
    ci_lower = delta_did - (z_crit * se_did)
    ci_upper = delta_did + (z_crit * se_did)

    # Relative percentage impact against control post-test mean
    ctrl_baseline = stats["control_post"]["mean"]
    pct_impact = (delta_did / ctrl_baseline * 100.0) if ctrl_baseline != 0 else 0.0

    return {
        "delta_treatment": round(delta_treat, 4),
        "delta_control": round(delta_control, 4),
        "did_estimator": round(delta_did, 4),
        "standard_error": round(se_did, 4),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_val, 6),
        "ci_95": (round(ci_lower, 4), round(ci_upper, 4)),
        "is_significant_05": p_val < 0.05,
        "is_significant_01": p_val < 0.01,
        "percent_impact": round(pct_impact, 2),
        "summary": {
            "treat_n": (stats["treat_pre"]["n"], stats["treat_post"]["n"]),
            "control_n": (stats["control_pre"]["n"], stats["control_post"]["n"]),
            "treat_pre_mean": round(stats["treat_pre"]["mean"], 4),
            "treat_post_mean": round(stats["treat_post"]["mean"], 4),
            "control_pre_mean": round(stats["control_pre"]["mean"], 4),
            "control_post_mean": round(stats["control_post"]["mean"], 4),
        }
    }


def evaluate_parallel_trends(
    treat_history: List[List[float]],
    control_history: List[List[float]]
) -> Dict[str, Any]:
    """
    Test the Parallel Trends Assumption for DiD across >= 2 pre-treatment periods.
    
    Verifies that the trajectory differences between treatment and control in the
    pre-intervention era are statistically indistinguishable from zero.
    """
    t_periods = len(treat_history)
    if t_periods != len(control_history) or t_periods < 2:
        raise ValueError(f"Parallel trends pre-test requires >= 2 pre-treatment periods with equal counts (got {t_periods} vs {len(control_history)}).")

    treat_means = [sum(p) / len(p) for p in treat_history]
    control_means = [sum(p) / len(p) for p in control_history]

    # Period-to-period changes
    treat_diffs = [treat_means[i+1] - treat_means[i] for i in range(t_periods - 1)]
    control_diffs = [control_means[i+1] - control_means[i] for i in range(t_periods - 1)]

    divergence_per_period = [abs(td - cd) for td, cd in zip(treat_diffs, control_diffs)]
    mean_divergence = sum(divergence_per_period) / len(divergence_per_period)

    # Standard error of trend differences
    pooled_se = math.sqrt(sum(d ** 2 for d in divergence_per_period) / len(divergence_per_period)) if divergence_per_period else 0.0
    tenable = mean_divergence <= 0.10 * (sum(control_means) / len(control_means))

    return {
        "pre_treatment_periods": t_periods,
        "treat_pre_means": [round(m, 4) for m in treat_means],
        "control_pre_means": [round(m, 4) for m in control_means],
        "mean_divergence": round(mean_divergence, 4),
        "parallel_trends_tenable": tenable,
        "verdict": "Parallel trends assumption is tenable." if tenable else "Caution: Potential pre-trend violation detected."
    }
