"""Validation utilities for NeuroThera-Map.

This package implements the Phase 4 "mutual validation" hooks in a minimal,
dependency-light way:
- schema checks for exchanged artifacts
- a small ValidationReport container
- simple, interpretable loss/score components
"""

from .report import ValidationReport
from .schema import (
    validate_activity_map,
    validate_connectivity_graph,
    validate_drug_profile,
    validate_receptor_map,
)

__all__ = [
    "ValidationReport",
    "validate_activity_map",
    "validate_receptor_map",
    "validate_connectivity_graph",
    "validate_drug_profile",
]
