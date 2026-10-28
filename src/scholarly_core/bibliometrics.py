"""
Bibliometrics & Knowledge Topology Engine
Plugin: scientific-research-os
Module: core.compute.bibliometrics

Deterministic implementation of:
1. CD-Index (Consolidation-Disruption Index) per Funk & Owen-Smith (2017) and Wu et al. (2019)
2. Uzzi Combinatorial Novelty Z-Score per Uzzi et al. (2013)
3. Rao-Stirling Interdisciplinary Diversity (Delta) per Stirling (2007) and Porter & Rafols (2009)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# 1. CANONICAL COGNITIVE DISTANCES (DISCIPLINARY TAXONOMY)
# Standardized global distance matrix based on Crossref/Scopus ASJC macro-domains
# =============================================================================

DEFAULT_COGNITIVE_DISTANCES: Dict[Tuple[str, str], float] = {
    # Distance between same discipline is 0.0
    ("natural_sciences", "formal_sciences"): 0.30,
    ("natural_sciences", "engineering"): 0.35,
    ("natural_sciences", "medicine"): 0.40,
    ("natural_sciences", "social_sciences"): 0.75,
    ("natural_sciences", "humanities"): 0.85,

    ("formal_sciences", "engineering"): 0.25,
    ("formal_sciences", "medicine"): 0.60,
    ("formal_sciences", "social_sciences"): 0.65,
    ("formal_sciences", "humanities"): 0.80,

    ("engineering", "medicine"): 0.50,
    ("engineering", "social_sciences"): 0.70,
    ("engineering", "humanities"): 0.85,

    ("medicine", "social_sciences"): 0.65,
    ("medicine", "humanities"): 0.80,

    ("social_sciences", "humanities"): 0.45,
}


def _get_distance(d1: str, d2: str, custom_matrix: Optional[Dict[Tuple[str, str], float]] = None) -> float:
    """Retrieve symmetric distance between two disciplines."""
    k1, k2 = d1.lower().strip(), d2.lower().strip()
    if k1 == k2:
        return 0.0
    matrix = custom_matrix or DEFAULT_COGNITIVE_DISTANCES
    if (k1, k2) in matrix:
        return matrix[(k1, k2)]
    if (k2, k1) in matrix:
        return matrix[(k2, k1)]
    # Default interdisciplinary distance if not explicitly mapped
    return 0.60


# =============================================================================
# 2. CD-INDEX (CONSOLIDATION-DISRUPTION INDEX)
# =============================================================================

def calculate_cd_index(
    n_i: int | Set[Any] | List[Any],
    n_j: int | Set[Any] | List[Any] = 0,
    n_k: int | Set[Any] | List[Any] = 0
) -> Dict[str, Any]:
    """
    Calculate the Consolidation-Disruption (CD) Index:
    CD = (n_i - n_j) / (n_i + n_j + n_k)

    Accepts either:
    1. Integer counts (n_i, n_j, n_k)
    2. Sets/lists of citing paper IDs (focal_citing, prior_citing, ref_only_citing)

    Definitions:
    - n_i: Papers that cite the focal paper WITHOUT citing its references (disruption)
    - n_j: Papers that cite the focal paper AND cite its references (consolidation)
    - n_k: Papers that cite the focal paper's references WITHOUT citing the focal paper

    Range: [-1.0, +1.0]
    Interpretation:
    - CD > 0: Destabilizing/Disruptive (creating a new research trajectory)
    - CD < 0: Consolidating/Developing (reinforcing existing paradigm)
    """
    # If sets or lists are passed, compute set differences and intersections
    if isinstance(n_i, (set, list, tuple)) and isinstance(n_j, (set, list, tuple)):
        focal_set = set(n_i)
        prior_set = set(n_j)
        calc_i = len(focal_set - prior_set)
        calc_j = len(focal_set & prior_set)
        if isinstance(n_k, (set, list, tuple)):
            calc_k = len(set(n_k) - focal_set)
        elif isinstance(n_k, (int, float)) and n_k > 0:
            calc_k = int(n_k)
        else:
            calc_k = len(prior_set - focal_set)
        n_i, n_j, n_k = calc_i, calc_j, calc_k

    n_i = max(0, int(n_i))
    n_j = max(0, int(n_j))
    n_k = max(0, int(n_k))

    total = n_i + n_j + n_k
    if total == 0:
        return {
            "cd_index": 0.0,
            "n_i": 0,
            "n_j": 0,
            "n_k": 0,
            "total_citation_network": 0,
            "classification": "NO_CITATIONS_NETWORK",
            "interpretation": "Insufficient forward citation network to compute CD-Index."
        }

    cd_val = (n_i - n_j) / float(total)

    if cd_val >= 0.25:
        classification = "PARADIGM_SHIFTING_DISRUPTION"
        interpretation = "Highly disruptive: future work tends to cite this work as a primary origin without needing past references."
    elif cd_val >= -0.15:
        classification = "BALANCED_MONOGRAPH_SYNTHESIS"
        interpretation = "Balanced scholarly synthesis: combines consolidation of foundational canon with innovative conceptual departures."
    else:
        classification = "CONSOLIDATION_INCREMENTAL"
        interpretation = "Consolidating/incremental: advances knowledge by directly strengthening existing intellectual lineage."

    return {
        "cd_index": round(cd_val, 4),
        "n_i": n_i,
        "n_j": n_j,
        "n_k": n_k,
        "total_citation_network": total,
        "classification": classification,
        "interpretation": interpretation
    }


# =============================================================================
# 3. UZZI COMBINATORIAL NOVELTY Z-SCORE
# =============================================================================

def calculate_uzzi_novelty(
    observed_pair_counts: Dict[str, int] | List[Any],
    total_tokens: Optional[int] = None,
    marginal_counts: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Calculate Uzzi Combinatorial Novelty Z-Score per Uzzi et al. (2013).
    Evaluates journal or disciplinary pair co-citations against a randomized null model.

    Input:
    - observed_pair_counts: Dict with key 'journal_a|journal_b' and value count, OR list of pairs/tuples
    - total_tokens: Total citations in reference network
    - marginal_counts: Optional individual citation counts per journal/discipline
    """
    if isinstance(observed_pair_counts, (list, tuple)):
        tally: Dict[str, int] = {}
        for item in observed_pair_counts:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                k = f"{item[0]}|{item[1]}"
            elif isinstance(item, str):
                k = item.replace(":", "|")
            else:
                continue
            tally[k] = tally.get(k, 0) + 1
        observed_pair_counts = tally

    if not observed_pair_counts:
        return {
            "median_z_score": 0.0,
            "tenth_percentile_z": 0.0,
            "n_pairs": 0,
            "classification": "INSUFFICIENT_PAIR_DATA",
            "is_high_novelty": False,
            "is_conventional_foundation": False
        }

    # Derive marginal frequencies if not provided
    if marginal_counts is None:
        marginal_counts = {}
        for pair_key, count in observed_pair_counts.items():
            parts = pair_key.split("|")
            if len(parts) == 2:
                p1, p2 = parts[0].strip(), parts[1].strip()
                marginal_counts[p1] = marginal_counts.get(p1, 0) + count
                marginal_counts[p2] = marginal_counts.get(p2, 0) + count

    total_n = total_tokens or max(1, sum(marginal_counts.values()) // 2)

    z_scores: List[float] = []
    pair_details: List[Dict[str, Any]] = []

    for pair_key, obs in observed_pair_counts.items():
        parts = pair_key.split("|")
        if len(parts) != 2:
            continue
        p1, p2 = parts[0].strip(), parts[1].strip()
        m1 = marginal_counts.get(p1, 1)
        m2 = marginal_counts.get(p2, 1)

        # Expected under random citation allocation
        expected = (float(m1) * float(m2)) / float(max(1, total_n))
        variance = expected * (1.0 - m1 / float(total_n)) * (1.0 - m2 / float(total_n))
        std_err = math.sqrt(max(1e-4, variance))

        z = (float(obs) - expected) / std_err
        z_scores.append(z)
        pair_details.append({
            "pair": f"{p1} | {p2}",
            "observed": obs,
            "expected": round(expected, 2),
            "z_score": round(z, 2)
        })

    if not z_scores:
        return {
            "median_z_score": 0.0,
            "tenth_percentile_z": 0.0,
            "n_pairs": 0,
            "classification": "INSUFFICIENT_PAIR_DATA",
            "is_high_novelty": False,
            "is_conventional_foundation": False
        }

    z_scores.sort()
    n = len(z_scores)

    # Median Z-score
    mid = n // 2
    if n % 2 == 1:
        med_z = z_scores[mid]
    else:
        med_z = (z_scores[mid - 1] + z_scores[mid]) / 2.0

    # 10th percentile Z-score
    idx_10 = max(0, int(0.10 * n))
    tenth_z = z_scores[idx_10]

    # 90th percentile Z-score
    idx_90 = min(n - 1, int(0.90 * n))
    ninetieth_z = z_scores[idx_90]

    # Uzzi classification:
    # High-impact scientific breakthroughs have HIGH conventional foundation (med_z > 0)
    # coupled with an ATYPICAL combination tail (tenth_z < 0 or ninetieth_z > 3.0)
    is_conventional_foundation = med_z >= 0.0
    is_high_novelty = (tenth_z < -0.5) or (ninetieth_z >= 3.0)

    if is_conventional_foundation and is_high_novelty:
        classification = "BREAKTHROUGH_BALANCED_NOVELTY"
        diagnostic = "Ideal Uzzi signature: strong conventional canon grounding coupled with high-novelty disciplinary combinations."
    elif not is_conventional_foundation and is_high_novelty:
        classification = "CHAOTIC_SPECULATIVE_NOVELTY"
        diagnostic = "High novelty without sufficient canonical grounding (high risk of epistemic drifting)."
    elif is_conventional_foundation and not is_high_novelty:
        classification = "CONVENTIONAL_LOW_NOVELTY"
        diagnostic = "Mainstream canon: solidly grounded but conservative in theoretical cross-pollination."
    else:
        classification = "MARGINAL_DIFFUSE_SYNTHESIS"
        diagnostic = "Diffuse citation structure with neither strong canon nor innovative combinations."

    return {
        "median_z_score": round(float(med_z), 4),
        "tenth_percentile_z": round(float(tenth_z), 4),
        "ninetieth_percentile_z": round(float(ninetieth_z), 4),
        "n_pairs_evaluated": n,
        "is_conventional_foundation": is_conventional_foundation,
        "is_high_novelty": is_high_novelty,
        "classification": classification,
        "diagnostic": diagnostic,
        "top_pairs": pair_details[:5]
    }


# =============================================================================
# 4. RAO-STIRLING INTERDISCIPLINARY DIVERSITY
# =============================================================================

def calculate_rao_stirling_diversity(
    disciplinary_proportions: Dict[str, float],
    distance_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    alpha: float = 1.0,
    beta: float = 1.0
) -> Dict[str, Any]:
    """
    Calculate Rao-Stirling Interdisciplinary Diversity (Delta):
    Delta = sum_{a != b} (d_ab)^alpha * (p_a * p_b)^beta

    Measures variety, balance, and disparity in bibliography/citation portfolio.
    Range: [0.0, 1.0]
    """
    if not disciplinary_proportions:
        return {
            "diversity_delta": 0.0,
            "variety": 0,
            "simpson_balance": 0.0,
            "classification": "MONO_DISCIPLINARY",
            "interpretation": "Single discipline or empty portfolio."
        }

    # Normalize proportions to sum to 1.0
    total_prop = sum(disciplinary_proportions.values())
    if total_prop <= 0:
        return {
            "diversity_delta": 0.0,
            "variety": 0,
            "simpson_balance": 0.0,
            "classification": "INVALID_PROPORTIONS",
            "interpretation": "Disciplinary weights must be non-zero."
        }

    p = {k: v / total_prop for k, v in disciplinary_proportions.items()}
    disciplines = list(p.keys())
    variety = len([k for k, val in p.items() if val > 0.01])

    delta = 0.0
    for i in range(len(disciplines)):
        for j in range(i + 1, len(disciplines)):
            d1, d2 = disciplines[i], disciplines[j]
            dist = _get_distance(d1, d2, distance_matrix)
            p_prod = p[d1] * p[d2]
            term = (dist ** alpha) * (p_prod ** beta)
            delta += 2.0 * term  # Symmetrical sum

    # Simpson Balance (1 - sum p_i^2)
    simpson = 1.0 - sum(val ** 2 for val in p.values())

    if delta >= 0.45 and variety >= 3:
        classification = "EXEMPLARY_INTERDISCIPLINARY_SYNTHESIS"
        interpretation = "High cognitive disparity and balanced integration across distant scientific domains."
    elif delta >= 0.20:
        classification = "MODERATE_CROSS_DISCIPLINARY"
        interpretation = "Healthy cross-disciplinary dialogue with moderate conceptual distance."
    else:
        classification = "MONO_DISCIPLINARY_FOCUSED"
        interpretation = "Specialized monodisciplinary focus with low cognitive boundary crossing."

    return {
        "diversity_delta": round(float(delta), 4),
        "variety_count": variety,
        "simpson_balance": round(float(simpson), 4),
        "classification": classification,
        "interpretation": interpretation,
        "proportions": {k: round(v, 4) for k, v in p.items()}
    }
