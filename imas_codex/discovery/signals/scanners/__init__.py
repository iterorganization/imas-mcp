"""Data source scanner plugins for signal discovery.

Each scanner knows how to extract FacilitySignal nodes from a specific
data source type. Scanners are dispatched based on data_sources keys
in facility config YAML files.

Scanner lifecycle:
    1. scan() — discover signals from data source (SSH, API calls, parsing)
    2. Enrichment — LLM physics domain classification (shared, not per-scanner)
       with wiki context injection for cost savings and quality improvement
    3. check() — validate signals return data for reference shot

Available scanners:
    - TDIScanner: MDSplus TDI function files (.fun) — TCV
    - PPFScanner: JET Processed Pulse Files (SSH ppfdda/ppfdti) — JET
    - EDASScanner: JT-60SA Experiment Data Access System (eddbreadTable) — JT-60SA
    - MDSplusScanner: Direct MDSplus tree traversal — any MDSplus facility
    - IMASScanner: IMAS IDS signal enumeration — ITER, JET, JT-60SA
    - WikiScanner: Wiki-documented signal extraction — any facility with wiki

Usage:
    from imas_codex.discovery.signals.scanners import get_scanner, get_scanners_for_facility

    # Get a specific scanner
    scanner = get_scanner("tdi")
    result = await scanner.scan(facility="tcv", ssh_host="tcv", config={...})

    # Get all scanners for a facility based on data_sources config
    for scanner in get_scanners_for_facility("tcv"):
        result = await scanner.scan(...)
"""

from imas_codex.discovery.signals.scanners.base import (
    DataSourceScanner,
    ScanResult,
    get_facility_reference_shot,
    get_scanner,
    get_scanners_for_facility,
    list_scanners,
    register_scanner,
)

__all__ = [
    "DataSourceScanner",
    "ScanResult",
    "get_facility_reference_shot",
    "get_scanner",
    "get_scanners_for_facility",
    "list_scanners",
    "register_scanner",
]
