"""
Discovery Engine for remote facility exploration.

This module provides configuration management for remote fusion facilities.
Exploration is done via direct SSH; see imas_codex/config/README.md for guidance.

Public API for facility configuration:
- get_facility(): Load complete config (public + private merged)
- get_facility_metadata(): Load public metadata only (graph-safe)
- get_facility_infrastructure(): Load private infrastructure only
- update_infrastructure(): Update private config (tools, paths, notes)
- update_metadata(): Update public config (name, description)
- add_exploration_note(): Add timestamped exploration note
"""

from imas_codex.discovery.facility import (
    add_exploration_note,
    filter_private_fields,
    get_facilities_dir,
    get_facility,
    get_facility_infrastructure,
    get_facility_metadata,
    list_facilities,
    update_infrastructure,
    update_metadata,
    validate_no_private_fields,
)

__all__ = [
    # Core API
    "get_facility",
    "get_facility_metadata",
    "get_facility_infrastructure",
    "update_infrastructure",
    "update_metadata",
    "add_exploration_note",
    # Utilities
    "list_facilities",
    "get_facilities_dir",
    "filter_private_fields",
    "validate_no_private_fields",
]
