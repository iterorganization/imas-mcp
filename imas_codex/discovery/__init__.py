"""
Discovery Engine for remote facility exploration.

This module provides:
1. Configuration management for remote fusion facilities
2. Graph-led discovery pipeline (scan → score → discover)

Public API for facility configuration:
- get_facility(): Load complete config (public + private merged)
- get_facility_metadata(): Load public metadata only (graph-safe)
- get_facility_infrastructure(): Load private infrastructure only
- update_infrastructure(): Update private config (tools, paths, notes)
- update_metadata(): Update public config (name, description)
- add_exploration_note(): Add timestamped exploration note

Graph-led discovery API:
- scan_facility_sync(): Scan paths in the frontier
- get_discovery_stats(): Get discovery statistics
- get_frontier(): Get paths awaiting scan
- seed_facility_roots(): Create initial root paths
- clear_facility_paths(): Delete all paths for fresh start
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
from imas_codex.discovery.frontier import (
    clear_facility_paths,
    get_discovery_stats,
    get_frontier,
    get_high_value_paths,
    get_scorable_paths,
    seed_facility_roots,
)
from imas_codex.discovery.scanner import scan_facility_sync

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
    # Graph-led discovery
    "scan_facility_sync",
    "get_discovery_stats",
    "get_frontier",
    "get_scorable_paths",
    "get_high_value_paths",
    "seed_facility_roots",
    "clear_facility_paths",
]
