"""
scholarly-core: Standalone Scientific Metrology & Metascience Toolkit
Companion library for the monograph 'Jangkar & Kompas Riset Ilmiah'.
"""

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
__all__ = [
    "calculate_aiken_v",
    "calculate_aiken_exact_p_value",
    "get_aiken_critical_value",
    "calculate_gwet_ac1",
    "bootstrap_aiken_ci",
    "audit_single_item",
    "audit_psychometric_matrix"
]
