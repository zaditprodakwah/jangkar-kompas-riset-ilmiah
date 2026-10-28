"""
Unit tests for core.compute.bibliometrics
CD-Index, Uzzi Novelty Z-score, and Rao-Stirling Interdisciplinary Diversity
"""

import pytest
from scholarly_core.bibliometrics import (
    calculate_cd_index,
    calculate_uzzi_novelty,
    calculate_rao_stirling_diversity,
    DEFAULT_COGNITIVE_DISTANCES
)


def test_cd_index_disruptive_and_consolidating():
    # Disruptive case: future papers cite focal paper without past references
    res_disruptive = calculate_cd_index(n_i=45, n_j=5, n_k=10)
    assert res_disruptive["cd_index"] > 0.25
    assert res_disruptive["classification"] == "PARADIGM_SHIFTING_DISRUPTION"

    # Consolidating case: future papers cite focal paper AND past references
    res_consolidating = calculate_cd_index(n_i=2, n_j=40, n_k=8)
    assert res_consolidating["cd_index"] < -0.15
    assert res_consolidating["classification"] == "CONSOLIDATION_INCREMENTAL"

    # Balanced synthesis
    res_balanced = calculate_cd_index(n_i=15, n_j=14, n_k=20)
    assert -0.15 <= res_balanced["cd_index"] < 0.25
    assert res_balanced["classification"] == "BALANCED_MONOGRAPH_SYNTHESIS"

    # Zero citation edge case
    res_zero = calculate_cd_index(n_i=0, n_j=0, n_k=0)
    assert res_zero["cd_index"] == 0.0
    assert res_zero["classification"] == "NO_CITATIONS_NETWORK"


def test_uzzi_novelty_combinatorial():
    # Setup sample co-citation matrix of journal/discipline pairs
    pairs = {
        "computer_science|psychology": 15,
        "computer_science|mathematics": 80,
        "psychology|neuroscience": 45,
        "sociology|computer_science": 12,
        "theology|quantum_physics": 3   # Atypical rare combination
    }

    res = calculate_uzzi_novelty(pairs, total_tokens=250)
    assert res["n_pairs_evaluated"] == 5
    assert isinstance(res["median_z_score"], float)
    assert isinstance(res["tenth_percentile_z"], float)
    assert res["classification"] in [
        "BREAKTHROUGH_BALANCED_NOVELTY",
        "CONVENTIONAL_LOW_NOVELTY",
        "CHAOTIC_SPECULATIVE_NOVELTY",
        "MARGINAL_DIFFUSE_SYNTHESIS"
    ]

    # Empty pair edge case
    res_empty = calculate_uzzi_novelty({})
    assert res_empty["classification"] == "INSUFFICIENT_PAIR_DATA"


def test_rao_stirling_interdisciplinary_diversity():
    # Monodisciplinary case
    mono_portfolio = {"natural_sciences": 1.0}
    res_mono = calculate_rao_stirling_diversity(mono_portfolio)
    assert res_mono["diversity_delta"] == 0.0
    assert res_mono["classification"] == "MONO_DISCIPLINARY_FOCUSED"

    # High interdisciplinary portfolio: balanced across disparate disciplines
    diverse_portfolio = {
        "natural_sciences": 0.30,
        "social_sciences": 0.35,
        "formal_sciences": 0.35
    }
    res_diverse = calculate_rao_stirling_diversity(diverse_portfolio)
    assert res_diverse["diversity_delta"] > 0.20
    assert res_diverse["variety_count"] == 3
    assert res_diverse["classification"] in [
        "EXEMPLARY_INTERDISCIPLINARY_SYNTHESIS",
        "MODERATE_CROSS_DISCIPLINARY"
    ]
