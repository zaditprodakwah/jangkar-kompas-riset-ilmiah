"""
Deterministic Statistical and Psychometric Computing Engine.
Includes:
- Aiken's V Content Validity with Aiken (1985) critical values
- Non-parametric bootstrap resampling (10,000 iterations)
- Effect size calculations (Cohen's d, Hake's Normalized Gain)
- MMAT 2018 (Mixed Methods Appraisal Tool) evaluation
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple


# Aiken 1985 Table 1 Critical Values (alpha = 0.05, m = 1 item)
AIKEN_1985_CRITICAL_TABLE: Dict[Tuple[int, int], float] = {
    # c = 3
    (5, 3): 0.80, (6, 3): 0.78, (7, 3): 0.76, (8, 3): 0.72, (9, 3): 0.70, (10, 3): 0.70,
    (15, 3): 0.63, (20, 3): 0.60, (25, 3): 0.58,
    # c = 4
    (4, 4): 0.89, (5, 4): 0.80, (6, 4): 0.78, (7, 4): 0.74, (8, 4): 0.71, (9, 4): 0.70,
    (10, 4): 0.68, (15, 4): 0.62, (20, 4): 0.58, (25, 4): 0.56,
    # c = 5 (Likert 1 to 5)
    (3, 5): 0.92, (4, 5): 0.83, (5, 5): 0.80, (6, 5): 0.77, (7, 5): 0.73, (8, 5): 0.70,
    (9, 5): 0.69, (10, 5): 0.67, (15, 5): 0.60, (20, 5): 0.57, (25, 5): 0.55,
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


def get_aiken_critical_value(n_raters: int, n_categories: int) -> float:
    """Retrieve Aiken (1985) critical value for alpha = 0.05 with conservative fallback."""
    pair = (n_raters, n_categories)
    if pair in AIKEN_1985_CRITICAL_TABLE:
        return AIKEN_1985_CRITICAL_TABLE[pair]

    exact_c = [c for (n, c) in AIKEN_1985_CRITICAL_TABLE.keys() if n == n_raters]
    if exact_c:
        closest_c = min(exact_c, key=lambda x: abs(x - n_categories))
        return AIKEN_1985_CRITICAL_TABLE[(n_raters, closest_c)]

    exact_n = [n for (n, c) in AIKEN_1985_CRITICAL_TABLE.keys() if c == n_categories]
    if exact_n:
        closest_n = min(exact_n, key=lambda x: abs(x - n_raters))
        return AIKEN_1985_CRITICAL_TABLE[(closest_n, n_categories)]

    return 0.70


def bootstrap_aiken_ci(
    ratings: List[int],
    lo: int = 1,
    hi: int = 5,
    n_resamples: int = 10000,
    seed: int = 42
) -> Tuple[float, float]:
    """Calculate 95% non-parametric bootstrap confidence interval for Aiken's V."""
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
    idx_lo = int(n_resamples * 0.025)
    idx_hi = int(n_resamples * 0.975)

    return (round(boot_vs[idx_lo], 4), round(boot_vs[idx_hi], 4))


def calculate_cohens_d(
    group1: Optional[List[float]] = None,
    group2: Optional[List[float]] = None,
    m1: Optional[float] = None,
    m2: Optional[float] = None,
    s1: Optional[float] = None,
    s2: Optional[float] = None,
    n1: Optional[int] = None,
    n2: Optional[int] = None
) -> float:
    """Calculate Cohen's d effect size from raw lists or summary statistics."""
    if m1 is not None and m2 is not None and s1 is not None and s2 is not None and n1 is not None and n2 is not None:
        if n1 < 2 or n2 < 2:
            return 0.0
        var1 = s1 ** 2
        var2 = s2 ** 2
        denom = n1 + n2 - 2
        if denom <= 0:
            return 0.0
        s_pooled = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / denom)
        if s_pooled == 0:
            return 0.0
        return round((m1 - m2) / s_pooled, 4)

    if group1 is not None and group2 is not None:
        n1_calc, n2_calc = len(group1), len(group2)
        if n1_calc < 2 or n2_calc < 2:
            return 0.0

        m1_calc = sum(group1) / n1_calc
        m2_calc = sum(group2) / n2_calc

        var1 = sum((x - m1_calc) ** 2 for x in group1) / (n1_calc - 1)
        var2 = sum((x - m2_calc) ** 2 for x in group2) / (n2_calc - 1)

        denom = n1_calc + n2_calc - 2
        if denom <= 0:
            return 0.0
        s_pooled = math.sqrt(((n1_calc - 1) * var1 + (n2_calc - 1) * var2) / denom)
        if s_pooled == 0:
            return 0.0

        return round((m1_calc - m2_calc) / s_pooled, 4)

    return 0.0


def calculate_normalized_gain(pre: float, post: float, max_score: float = 100.0) -> float:
    """
    Calculate Hake's Normalized Gain: <g> = (post - pre) / (max - pre).
    Returns value categorized into Low (<0.3), Medium (0.3 - 0.7), or High (>0.7).
    """
    if max_score <= pre:
        return 0.0
    gain = (post - pre) / (max_score - pre)
    return round(gain, 4)


def evaluate_mmat_2018(category: str, criteria_met: List[bool]) -> Dict[str, Any]:
    """
    Evaluate study against Mixed Methods Appraisal Tool (MMAT 2018).
    Scores 0% to 100% based on 5 category-specific criteria.
    """
    total = len(criteria_met)
    passed = sum(1 for c in criteria_met if c is True)
    score_pct = round((passed / max(1, total)) * 100, 1)

    return {
        "mmat_category": category,
        "criteria_passed": passed,
        "criteria_total": total,
        "score_percentage": score_pct,
        "quality_tier": "High" if score_pct >= 80 else ("Moderate" if score_pct >= 60 else "Low")
    }
