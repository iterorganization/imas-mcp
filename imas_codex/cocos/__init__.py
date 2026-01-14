"""COCOS (COordinate COnventions) identification and transformation.

This module provides utilities for working with tokamak coordinate conventions
as defined in Sauter & Medvedev, CPC 184 (2013) 293-302.

Key functionality:
- Calculate COCOS from equilibrium physics quantities
- Transform between COCOS conventions
- Validate COCOS consistency
- Identify paths requiring sign flips between DD versions
- Load equilibrium data from IMAS IDS or EQDSK files
"""

from imas_codex.cocos.calculator import (
    KNOWN_CODE_COCOS,
    VALID_COCOS,
    COCOSParameters,
    ValidationResult,
    cocos_from_dd_version,
    cocos_to_parameters,
    determine_cocos,
    validate_cocos_consistency,
    validate_cocos_from_data,
)
from imas_codex.cocos.loaders import (
    EquilibriumData,
    load_from_eqdsk,
    load_from_imas_ids,
)
from imas_codex.cocos.transforms import (
    get_sign_flip_paths,
    list_ids_with_sign_flips,
    path_needs_cocos_transform,
)

__all__ = [
    "KNOWN_CODE_COCOS",
    "VALID_COCOS",
    "COCOSParameters",
    "EquilibriumData",
    "ValidationResult",
    "cocos_from_dd_version",
    "cocos_to_parameters",
    "determine_cocos",
    "get_sign_flip_paths",
    "list_ids_with_sign_flips",
    "load_from_eqdsk",
    "load_from_imas_ids",
    "path_needs_cocos_transform",
    "validate_cocos_consistency",
    "validate_cocos_from_data",
]
