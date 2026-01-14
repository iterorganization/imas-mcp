"""COCOS (COordinate COnventions) identification and transformation.

This module provides utilities for working with tokamak coordinate conventions
as defined in Sauter & Medvedev, CPC 184 (2013) 293-302.

Key functionality:
- Calculate COCOS from equilibrium physics quantities
- Transform between COCOS conventions
- Validate COCOS consistency
"""

from imas_codex.cocos.calculator import (
    VALID_COCOS,
    COCOSParameters,
    ValidationResult,
    cocos_from_dd_version,
    cocos_to_parameters,
    determine_cocos,
    validate_cocos_consistency,
    validate_cocos_from_data,
)

__all__ = [
    "VALID_COCOS",
    "COCOSParameters",
    "ValidationResult",
    "cocos_from_dd_version",
    "cocos_to_parameters",
    "determine_cocos",
    "validate_cocos_consistency",
    "validate_cocos_from_data",
]
