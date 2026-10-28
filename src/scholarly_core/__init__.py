"""
scholarly-core: Standalone Scientific Metrology, Causal Inference, and Metascience Toolkit
Official computational companion for the monograph 'Jangkar & Kompas Riset Ilmiah' (18 Chapters).

Author: Muhammad Khoiruzzadittaqwa
"""

from .forensics import (
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

from .causal import (
    calculate_hake_gain,
    calculate_did_2x2,
    evaluate_parallel_trends
)

from .bibliometrics import (
    calculate_cd_index,
    calculate_uzzi_novelty,
    calculate_rao_stirling_diversity
)

from .qualitative import (
    calculate_cohens_kappa,
    calculate_fleiss_kappa,
    calculate_krippendorff_alpha,
    audit_mmat_study
)

from .validity import (
    calculate_aiken_v,
    calculate_aiken_exact_p_value,
    get_aiken_critical_value,
    calculate_gwet_ac1,
    bootstrap_aiken_ci,
    audit_single_item,
    audit_psychometric_matrix
)

__version__ = "1.0.0"
__author__ = "Muhammad Khoiruzzadittaqwa"

__all__ = [
    # Forensics
    "calculate_ioannidis_ppv",
    "calculate_type_s_m_error",
    "run_grim_test",
    "run_grimmer_test",
    "run_debit_test",
    "recalculate_statcheck",
    "run_carlisle_stouffer_test",
    "run_tiva_test",
    "run_benford_first_digit_test",
    "fit_zcurve_em",
    # Causal
    "calculate_hake_gain",
    "calculate_did_2x2",
    "evaluate_parallel_trends",
    # Bibliometrics
    "calculate_cd_index",
    "calculate_uzzi_novelty",
    "calculate_rao_stirling_diversity",
    # Qualitative
    "calculate_cohens_kappa",
    "calculate_fleiss_kappa",
    "calculate_krippendorff_alpha",
    "audit_mmat_study",
    # Validity & Metrology
    "calculate_aiken_v",
    "calculate_aiken_exact_p_value",
    "get_aiken_critical_value",
    "calculate_gwet_ac1",
    "bootstrap_aiken_ci",
    "audit_single_item",
    "audit_psychometric_matrix"
]
