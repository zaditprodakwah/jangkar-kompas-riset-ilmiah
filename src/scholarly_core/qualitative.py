"""
Deterministic Qualitative Synthesis and Inter-Coder Reliability Engine (Kluster C).
Adheres strictly to:
- Robert K. Yin (2018) Case Study Chain of Evidence
- Kathy Charmaz (2014) Grounded Theory Systematic Coding
- Krippendorff (2011) Content Analysis Alpha Reliability Coefficient
- Cohen (1960) & Fleiss (1971) Agreement Diagnostics
- McGill University MMAT 2018 (Mixed Methods Appraisal Tool) Audit
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple


def calculate_cohens_kappa(
    coder1: List[Any],
    coder2: List[Any],
    categories: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Calculate Cohen's Kappa (kappa) inter-rater agreement for 2 raters on nominal data.
    
    Formula:
      kappa = (P_o - P_e) / (1 - P_e)
      where P_o is observed proportion of agreement, P_e is chance expected agreement.
    """
    n = len(coder1)
    if n == 0 or len(coder2) != n:
        raise ValueError(f"coder1 and coder2 must be non-empty and of equal length (got {len(coder1)} vs {len(coder2)}).")

    if categories is None:
        cat_set = set(coder1).union(set(coder2))
        cats = sorted(list(cat_set), key=lambda x: str(x))
    else:
        cats = categories

    k = len(cats)
    if k <= 1:
        return {
            "n_items": n,
            "k_categories": k,
            "observed_agreement": 1.0,
            "expected_agreement": 1.0,
            "kappa": 1.0,
            "interpretation": "Perfect agreement (single category)",
            "standard_error": 0.0,
            "z_score": 0.0,
            "is_significant_05": True
        }

    # Build contingency counts
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    matrix = [[0] * k for _ in range(k)]

    for c1, c2 in zip(coder1, coder2):
        if c1 in cat_to_idx and c2 in cat_to_idx:
            matrix[cat_to_idx[c1]][cat_to_idx[c2]] += 1

    # Observed agreement
    agree_count = sum(matrix[i][i] for i in range(k))
    p_o = agree_count / n

    # Expected agreement
    row_sums = [sum(matrix[i][j] for j in range(k)) for i in range(k)]
    col_sums = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    p_e = sum((row_sums[i] * col_sums[i]) for i in range(k)) / (n * n)

    if abs(1.0 - p_e) < 1e-9:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    # Standard error approximation
    if n > 1 and abs(1.0 - p_e) > 1e-9:
        se = math.sqrt(max(0.0, p_o * (1.0 - p_o)) / (n * ((1.0 - p_e) ** 2)))
        z = kappa / se if se > 0 else 0.0
    else:
        se = 0.0
        z = 0.0

    # Landis & Koch (1977) Interpretation Benchmarks
    if kappa >= 0.81:
        interp = "Almost Perfect Agreement"
    elif kappa >= 0.61:
        interp = "Substantial Agreement"
    elif kappa >= 0.41:
        interp = "Moderate Agreement"
    elif kappa >= 0.21:
        interp = "Fair Agreement"
    elif kappa >= 0.00:
        interp = "Slight Agreement"
    else:
        interp = "Poor Agreement (Worse than chance)"

    return {
        "n_items": n,
        "k_categories": k,
        "categories": [str(c) for c in cats],
        "observed_agreement": round(p_o, 4),
        "expected_agreement": round(p_e, 4),
        "kappa": round(kappa, 4),
        "standard_error": round(se, 4),
        "z_score": round(z, 4),
        "interpretation": interp,
        "is_significant_05": z > 1.96,
        "contingency_matrix": matrix
    }


def calculate_fleiss_kappa(matrix: List[List[int]]) -> Dict[str, Any]:
    """
    Calculate Fleiss' Kappa for m > 2 raters on nominal categories.
    
    matrix: N items x k categories, where matrix[i][j] is the count of raters
            who assigned item i to category j. Total raters per item (m) must be constant.
    """
    N = len(matrix)
    if N == 0:
        raise ValueError("Matrix must contain at least one item.")
    k = len(matrix[0])
    if k <= 1:
        raise ValueError("Matrix must contain at least 2 categories.")

    m = sum(matrix[0])
    if m < 2:
        raise ValueError(f"Fleiss' kappa requires at least 2 raters per item (got m={m}).")

    for row in matrix:
        if sum(row) != m:
            raise ValueError(f"Every item must have exactly m={m} ratings (got row sum {sum(row)}).")

    # Proportion of all assignments to the j-th category
    p_j = [sum(matrix[i][j] for i in range(N)) / (N * m) for j in range(k)]
    p_e = sum(pj ** 2 for pj in p_j)

    # Degree of agreement for i-th item
    p_i = []
    for row in matrix:
        sum_sq = sum(n_ij ** 2 for n_ij in row)
        p_i.append((sum_sq - m) / (m * (m - 1)))

    p_o = sum(p_i) / N

    if abs(1.0 - p_e) < 1e-9:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    # Landis & Koch (1977) interpretation
    if kappa >= 0.81:
        interp = "Almost Perfect Agreement"
    elif kappa >= 0.61:
        interp = "Substantial Agreement"
    elif kappa >= 0.41:
        interp = "Moderate Agreement"
    elif kappa >= 0.21:
        interp = "Fair Agreement"
    elif kappa >= 0.00:
        interp = "Slight Agreement"
    else:
        interp = "Poor Agreement"

    return {
        "n_items": N,
        "n_raters": m,
        "k_categories": k,
        "observed_agreement": round(p_o, 4),
        "expected_agreement": round(p_e, 4),
        "fleiss_kappa": round(kappa, 4),
        "interpretation": interp,
        "category_proportions": [round(p, 4) for p in p_j]
    }


def calculate_krippendorff_alpha(
    data: List[List[Optional[Any]]],
    level: str = "nominal"
) -> Dict[str, Any]:
    """
    Calculate Krippendorff's Alpha (alpha) across units and raters with support for missing data.
    
    data: List of units (rows), where each row is a list of ratings from r raters.
          Missing ratings must be represented as None.
    level: "nominal" (default).
    """
    n_units = len(data)
    if n_units == 0:
        raise ValueError("Data matrix must not be empty.")

    # Flatten to gather all unique values (excluding None)
    values = sorted(list(set(v for row in data for v in row if v is not None)), key=lambda x: str(x))
    k = len(values)
    if k <= 1:
        return {
            "n_units": n_units,
            "alpha": 1.0,
            "observed_disagreement": 0.0,
            "expected_disagreement": 0.0,
            "interpretation": "Perfect agreement (single category or invariant)",
            "is_reliable_080": True,
            "is_tentative_067": True
        }

    val_to_idx = {v: i for i, v in enumerate(values)}

    # Coincidence matrix C[v, c]
    C = [[0.0] * k for _ in range(k)]
    total_pair_comparisons = 0.0

    for row in data:
        valid_ratings = [v for v in row if v is not None]
        m_u = len(valid_ratings)
        if m_u < 2:
            continue  # Units with < 2 ratings cannot form pairs

        # For each pair in the unit, add weight 1 / (m_u - 1)
        for i in range(m_u):
            idx_i = val_to_idx[valid_ratings[i]]
            for j in range(m_u):
                if i != j:
                    idx_j = val_to_idx[valid_ratings[j]]
                    C[idx_i][idx_j] += 1.0 / (m_u - 1)
                    total_pair_comparisons += 1.0 / (m_u - 1)

    if total_pair_comparisons == 0:
        raise ValueError("No units with >= 2 ratings found to calculate inter-coder agreement.")

    # Marginal totals
    marginal = [sum(C[v][c] for c in range(k)) for v in range(k)]
    n_total = sum(marginal)

    # Observed disagreement D_o
    # For nominal data, metric delta(v, c) = 0 if v == c else 1
    d_o_sum = sum(C[v][c] for v in range(k) for c in range(k) if v != c)
    D_o = d_o_sum / n_total if n_total > 0 else 0.0

    # Expected disagreement D_e
    d_e_sum = sum(marginal[v] * marginal[c] for v in range(k) for c in range(k) if v != c)
    D_e = d_e_sum / (n_total * (n_total - 1)) if (n_total > 1) else 0.0

    if D_e == 0:
        alpha = 1.0
    else:
        alpha = 1.0 - (D_o / D_e)

    # Standard Krippendorff (2011) Thresholds:
    # alpha >= 0.800: Data are reliable; customary threshold for decisive findings.
    # 0.667 <= alpha < 0.800: Tentatively acceptable for exploratory findings.
    # alpha < 0.667: Unacceptable; unreliable agreement.
    if alpha >= 0.800:
        interp = "Reliable; meets the rigorous 0.800 threshold for definitive academic findings."
    elif alpha >= 0.667:
        interp = "Tentatively acceptable; meets the 0.667 threshold for exploratory research."
    else:
        interp = "Unreliable; falls below the 0.667 threshold for scientific evidence."

    return {
        "n_units": n_units,
        "k_categories": k,
        "values": [str(v) for v in values],
        "total_paired_comparisons": round(total_pair_comparisons, 2),
        "observed_disagreement": round(D_o, 4),
        "expected_disagreement": round(D_e, 4),
        "alpha": round(alpha, 4),
        "is_reliable_080": alpha >= 0.800,
        "is_tentative_067": alpha >= 0.667,
        "interpretation": interp
    }


def audit_mmat_study(
    category: str,
    responses: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Execute full MMAT 2018 (Mixed Methods Appraisal Tool) quality audit.
    
    Categories:
    - 'qualitative' (Yin case study / Charmaz grounded theory)
    - 'rct' (Randomized controlled trials)
    - 'non_randomized' (Quasi-experimental / Cohort)
    - 'quantitative_descriptive' (Surveys / Descriptive)
    - 'mixed_methods' (Sequential explanatory / exploratory / convergent)
    """
    CRITERIA_MAP = {
        "qualitative": [
            ("1.1", "Is the qualitative approach appropriate to answer the research question?"),
            ("1.2", "Are the qualitative data collection methods adequate to address the research question?"),
            ("1.3", "Are the findings adequately derived from the data?"),
            ("1.4", "Is the interpretation of results sufficiently substantiated by data?"),
            ("1.5", "Is there coherence between qualitative data sources, collection, analysis and interpretation?")
        ],
        "rct": [
            ("2.1", "Is randomization appropriately performed?"),
            ("2.2", "Are the groups comparable at baseline?"),
            ("2.3", "Are there complete outcome data?"),
            ("2.4", "Are outcome assessors blinded to the intervention provided?"),
            ("2.5", "Did the participants adhere to the assigned intervention?")
        ],
        "non_randomized": [
            ("3.1", "Are the participants representative of the target population?"),
            ("3.2", "Are measurements appropriate regarding both the exposure and outcome?"),
            ("3.3", "Are there complete outcome data?"),
            ("3.4", "Are the confounders accounted for in the design and analysis?"),
            ("3.5", "During the study period, is the intervention administered as intended?")
        ],
        "quantitative_descriptive": [
            ("4.1", "Is the sampling strategy relevant to address the research question?"),
            ("4.2", "Is the sample representative of the target population?"),
            ("4.3", "Are the measurements appropriate?"),
            ("4.4", "Is the risk of nonresponse bias low?"),
            ("4.5", "Is the statistical analysis appropriate to answer the research question?")
        ],
        "mixed_methods": [
            ("5.1", "Is there an adequate rationale for using a mixed methods design?"),
            ("5.2", "Are the different components of the study effectively integrated?"),
            ("5.3", "Are the outputs of the integration of qualitative and quantitative components adequately interpreted?"),
            ("5.4", "Are divergences and inconsistencies between quantitative and qualitative results adequately addressed?"),
            ("5.5", "Do the different components of the study adhere to the quality criteria of each tradition?")
        ]
    }

    cat_norm = category.lower().strip()
    if cat_norm not in CRITERIA_MAP:
        raise ValueError(f"Unknown MMAT category '{category}'. Supported: {list(CRITERIA_MAP.keys())}")

    criteria = CRITERIA_MAP[cat_norm]
    detailed_results = []
    met_count = 0

    for code, question in criteria:
        is_met = responses.get(code, responses.get(code.replace(".", "_"), False))
        if is_met:
            met_count += 1
        detailed_results.append({
            "code": code,
            "question": question,
            "met": bool(is_met)
        })

    score_pct = (met_count / len(criteria)) * 100.0

    if score_pct == 100.0:
        verdict = "Exemplary Quality (5/5 Criteria Met) - High Epistemic Trust"
    elif score_pct >= 80.0:
        verdict = "High Quality (4/5 Criteria Met) - Minor Methodological Gaps"
    elif score_pct >= 60.0:
        verdict = "Moderate Quality (3/5 Criteria Met) - Substantive Limitations Noted"
    else:
        verdict = "Low Methodological Rigor (<3/5 Criteria Met) - High Risk of Epistemic Bias"

    return {
        "category": cat_norm,
        "criteria_evaluated": len(criteria),
        "criteria_met": met_count,
        "quality_score_percent": round(score_pct, 1),
        "verdict": verdict,
        "is_publishable_tier1": score_pct >= 80.0,
        "checklist": detailed_results
    }
