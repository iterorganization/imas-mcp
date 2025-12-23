"""
Discovery Engine for remote facility exploration.

This module provides configuration management for remote fusion facilities.
Exploration is done via direct SSH; see imas_codex/config/README.md for guidance.
"""

from imas_codex.discovery.facility import (
    filter_private_fields,
    get_facilities_dir,
    get_facility,
    get_facility_private,
    get_facility_public,
    list_facilities,
    save_private,
    validate_no_private_fields,
)

__all__ = [
    "filter_private_fields",
    "get_facilities_dir",
    "get_facility",
    "get_facility_private",
    "get_facility_public",
    "list_facilities",
    "save_private",
    "validate_no_private_fields",
]
