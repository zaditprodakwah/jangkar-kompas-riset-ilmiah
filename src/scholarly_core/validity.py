"""
Deterministic Psychometrics and Measurement Validity Engine.
Implements:
- Lewis R. Aiken (1985) Content Validity Coefficient (V)
- Exact P-Value via Discrete Polynomial Convolution Generating Function
- Kilem L. Gwet (2008, 2014) AC1 Chance-Corrected Inter-Rater Agreement
- Non-Parametric Bootstrap Resampling (10,000 iterations)
- Batch Item-Matrix Psychometric Auditing
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

# Aiken 1985 Table 1 Critical Values (alpha = 0.05) for reference and benchmarking
AIKEN_1985_CRITICAL_TABLE: Dict[Tuple[int, int], float] = {
    # c = 3
    (5, 3): 0.80, (6, 3): 0.78, (7, 3): 0.76, (8, 3): 0.72, (9, 3): 0.70, (10, 3): 0.70,
    (15, 3): 0.63, (20, 3): 0.60, (25, 3): 0.58,
    # c = 4
    (4, 4): 0.89, (5, 4): 0.80, (6, 4): 0.78, (7, 4): 0.74, (8, 4): 0.71, (9, 4): 0.70,
    (10, 4): 0.68, (15, 4): 0.62, (20, 4): 0.58, (25, 4): 0.56,
    # c = 5 (Likert 1 to 5)
    (2, 5): 1.00, (3, 5): 0.92, (4, 5): 0.83, (5, 5): 0.80, (6, 5): 0.77, (7, 5): 0.73,
    (8, 5): 0.70, (9, 5): 0.69, (10, 5): 0.67, (15, 5): 0.60, (20, 5): 0.57, (25, 5): 0.55,
    # c = 6
    (3, 6): 0.90, (4, 6): 0.82, (5, 6): 0.78, (6, 6): 0.74, (7, 6): 0.71, (8, 6): 0.68,
    (9, 6): 0.67, (10, 6): 0.65, (15, 6): 0.58, (20, 6): 0.55, (25, 6): 0.53,
    # c = 7
    (3, 7): 0.89, (4, 7): 0.80, (5, 7): 0.76, (6, 7): 0.72, (7, 7): 0.70, (8, 7): 0.67,
    (9, 7): 0.65, (10, 7): 0.63, (15, 7): 0.57, (20, 7): 0.54, (25, 7): 0.52
}


def calculate_aiken_v(ratings: List[int], lo: int = 1, hi: int = 5) -> float:
    """
    Calculate Aiken's V content validity coefficient.
    Formula: V = S / (n * (c - 1))
    where S = sum(r - lo), n = raters count, c = category range (hi - lo + 1).
    """
    n = len(ratings)
    c = hi - lo + 1
    if n == 0 or c <= 1:
        return 0.0

    s = sum(r - lo for r in ratings)
    denom = n * (c - 1)
    return round(s / denom, 4)


def calculate_aiken_exact_p_value(ratings: List[int], lo: int = 1, hi: int = 5) -> float:
    """
    Calculate exact one-tailed p-value for Aiken's V using discrete polynomial convolution.
    Under the null hypothesis of independent uniform random ratings r_i in {lo, ..., hi}:
    Probability Generating Function:
        G(x) = (1/c * sum_{j=0}^{c-1} x^j)^n = 1/c^n * (1 + x + x^2 + ... + x^{c-1})^n
    The exact distribution P(S = k) is determined via 1D dynamic programming convolution.
    """
    n = len(ratings)
    c = hi - lo + 1
    if n == 0 or c <= 1:
        return 1.0

    s_obs = sum(r - lo for r in ratings)
    s_max = n * (c - 1)

    if s_obs >= s_max:
        return round(1.0 / (c ** n), 6)
    if s_obs <= 0:
        return 1.0

    # DP array of size s_max + 1
    poly = [0] * (s_max + 1)
    poly[0] = 1

    for i in range(1, n + 1):
        next_poly = [0] * (s_max + 1)
        max_deg_prev = (i - 1) * (c - 1)
        for k in range(max_deg_prev + 1):
            if poly[k] == 0:
                continue
            for j in range(c):
                next_poly[k + j] += poly[k]
        poly = next_poly

    total_permutations = c ** n
    extreme_count = sum(poly[k] for k in range(s_obs, s_max + 1))
    return round(extreme_count / total_permutations, 6)


def get_aiken_critical_value(n_raters: int, n_categories: int, alpha: float = 0.05) -> float:
    """
    Retrieve Aiken (1985) critical value for alpha = 0.05.
    If not in canonical table, derive exact critical value analytically via polynomial convolution.
    """
    pair = (n_raters, n_categories)
    if alpha == 0.05 and pair in AIKEN_1985_CRITICAL_TABLE:
        return AIKEN_1985_CRITICAL_TABLE[pair]

    # Analytical derivation: find minimum V such that p_value <= alpha
    c = n_categories
    n = n_raters
    s_max = n * (c - 1)
    if s_max == 0:
        return 1.0

    # Build polynomial distribution once
    poly = [0] * (s_max + 1)
    poly[0] = 1
    for i in range(1, n + 1):
        next_poly = [0] * (s_max + 1)
        max_deg_prev = (i - 1) * (c - 1)
        for k in range(max_deg_prev + 1):
            if poly[k] == 0:
                continue
            for j in range(c):
                next_poly[k + j] += poly[k]
        poly = next_poly

    total_permutations = c ** n

    # Scan from s = s_max down to 0
    running_extreme = 0
    critical_s = s_max
    for s in range(s_max, -1, -1):
        running_extreme += poly[s]
        p_val = running_extreme / total_permutations
        if p_val <= alpha:
            critical_s = s
        else:
            break

    return round(critical_s / s_max, 4)


def calculate_gwet_ac1(ratings: List[int], lo: int = 1, hi: int = 5) -> float:
    """
    Calculate Gwet's AC1 chance-corrected inter-rater agreement coefficient for a single item rating vector.
    Gwet (2008, 2014) overcomes Fleiss' and Cohen's kappa paradoxes in high agreement contexts.
    """
    n = len(ratings)
    c = hi - lo + 1
    if n < 2 or c <= 1:
        return 1.0

    # Category frequency counts
    freq = {cat: 0 for cat in range(lo, hi + 1)}
    for r in ratings:
        if r in freq:
            freq[r] += 1

    # Observed agreement proportion P_a across all rater pairs
    agreed_pairs = sum(f * (f - 1) for f in freq.values())
    total_pairs = n * (n - 1)
    p_a = agreed_pairs / total_pairs

    # Marginal probabilities pi_k
    pi = [f / n for f in freq.values()]
    # Gwet's chance agreement probability P_e
    p_e = (1.0 / (c - 1)) * sum(p * (1.0 - p) for p in pi)

    if math.isclose(p_e, 1.0):
        return 1.0

    ac1 = (p_a - p_e) / (1.0 - p_e)
    return round(ac1, 4)


def bootstrap_aiken_ci(
    ratings: List[int],
    lo: int = 1,
    hi: int = 5,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """Calculate non-parametric bootstrap confidence interval for Aiken's V."""
    n = len(ratings)
    if n < 2:
        v = calculate_aiken_v(ratings, lo, hi)
        return (v, v)

    rng = random.Random(seed)
    boot_vs: List[float] = []

    for _ in range(n_resamples):
        sample = [rng.choice(ratings) for _ in range(n)]
        boot_vs.append(calculate_aiken_v(sample, lo, hi))

    boot_vs.sort()
    alpha = (1.0 - confidence_level) / 2.0
    idx_lo = int(alpha * n_resamples)
    idx_hi = int((1.0 - alpha) * n_resamples) - 1

    ci_lower = max(0.0, boot_vs[max(0, idx_lo)])
    ci_upper = min(1.0, boot_vs[min(len(boot_vs) - 1, idx_hi)])

    return (round(ci_lower, 4), round(ci_upper, 4))


def audit_single_item(
    ratings: List[int],
    item_id: str = "ITEM-01",
    lo: int = 1,
    hi: int = 5,
    alpha: float = 0.05,
    n_resamples: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """Audit single item ratings with triad psychometrics: Aiken V, exact p-value, Gwet's AC1, bootstrap CI."""
    n = len(ratings)
    c = hi - lo + 1
    s_obs = sum(r - lo for r in ratings)
    s_max = n * (c - 1)
    v = calculate_aiken_v(ratings, lo, hi)
    p_val = calculate_aiken_exact_p_value(ratings, lo, hi)
    crit_v = get_aiken_critical_value(n, c, alpha=alpha)
    ac1 = calculate_gwet_ac1(ratings, lo, hi)
    ci_lo, ci_hi = bootstrap_aiken_ci(ratings, lo, hi, n_resamples=n_resamples, confidence_level=1.0 - alpha, seed=seed)
    is_valid = p_val <= alpha and v >= crit_v

    return {
        "item_id": item_id,
        "n_raters": n,
        "n_categories": c,
        "raw_ratings": ratings,
        "s_sum": s_obs,
        "s_max": s_max,
        "aiken_v": v,
        "exact_p_value": p_val,
        "critical_v": crit_v,
        "gwet_ac1": ac1,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "is_valid": is_valid,
        "verdict": "VALID_SIGNIFIKAN" if is_valid else "GUGUR_REVISI_TOTAL"
    }


def audit_psychometric_matrix(
    matrix: List[List[int]],
    lo: int = 1,
    hi: int = 5,
    item_ids: Optional[List[str]] = None,
    alpha: float = 0.05,
    n_resamples: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """Audit a matrix of items (rows) across raters (columns)."""
    items_count = len(matrix)
    if items_count == 0:
        return {"items": [], "summary": {}}

    if not item_ids or len(item_ids) != items_count:
        item_ids = [f"BUTIR-{i+1:02d}" for i in range(items_count)]

    item_results = []
    valid_count = 0
    v_scores = []
    ac1_scores = []

    for idx, row in enumerate(matrix):
        res = audit_single_item(row, item_id=item_ids[idx], lo=lo, hi=hi, alpha=alpha, n_resamples=n_resamples, seed=seed)
        item_results.append(res)
        if res["is_valid"]:
            valid_count += 1
        v_scores.append(res["aiken_v"])
        ac1_scores.append(res["gwet_ac1"])

    mean_v = round(sum(v_scores) / items_count, 4) if items_count > 0 else 0.0
    mean_ac1 = round(sum(ac1_scores) / items_count, 4) if items_count > 0 else 0.0

    return {
        "items": item_results,
        "summary": {
            "total_items": items_count,
            "valid_items": valid_count,
            "invalid_items": items_count - valid_count,
            "validity_percentage": round((valid_count / items_count) * 100, 2),
            "mean_aiken_v": mean_v,
            "mean_gwet_ac1": mean_ac1,
            "overall_status": "MEMENUHI_STANDAR" if valid_count == items_count else "PERLU_REVISI_BUTIR"
        }
    }
