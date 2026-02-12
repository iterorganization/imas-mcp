"""EDAS (Experiment Data Access System) scanner plugin for JT-60SA.

EDAS is JT-60SA's primary data access system, wrapping proprietary data formats
with C, Fortran, and Python APIs. Signal definitions are in C header files
and Python wrapper code.

Access methods:
  - C API: eddb_get(channel, shot, &data, &ndata)
  - Python: eddb.get_data(channel, shot)
  - High-level tools: eGIS (viewer), eSLICE (slicer), eSURF (surface)

Config key: data_sources.edas
Facility: JT-60SA
"""

from __future__ import annotations

import logging
from typing import Any

from imas_codex.discovery.data.scanners.base import (
    ScanResult,
    register_scanner,
)
from imas_codex.graph.models import FacilitySignal

logger = logging.getLogger(__name__)


class EDASScanner:
    """Discover signals from JT-60SA EDAS system.

    EDAS discovery strategy:
    1. Parse C header files for signal/channel definitions
    2. Parse Python wrapper code for available channels
    3. Cross-reference with EDAS database tables
    4. Create FacilitySignal per channel with EDAS accessor format

    Config (data_sources.edas):
        api_path: str - Path to EDAS API source files
        header_path: str - Path to C headers with signal definitions
        reference_shot: int - Shot for validation
    """

    scanner_type: str = "edas"

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Discover signals from EDAS C headers and Python wrappers.

        TODO: Implement EDAS signal enumeration.
        Scaffold returns empty result with config metadata.
        """
        api_path = config.get("api_path")
        header_path = config.get("header_path")
        ref_shot = reference_shot or config.get("reference_shot")

        logger.info(
            "EDAS scanner scaffold: api=%s, headers=%s, shot=%s",
            api_path,
            header_path,
            ref_shot,
        )

        # TODO: Parse EDAS headers via SSH
        # ssh jt60sa "grep -E 'EDAS_CH_|eddb_channel' {header_path}/*.h"
        # Extract channel IDs, names, units, descriptions
        # Create FacilitySignal with accessor="eddb_get('{channel}', shot)"

        return ScanResult(
            signals=[],
            metadata={
                "api_path": api_path,
                "header_path": header_path,
                "reference_shot": ref_shot,
            },
            stats={
                "signals_discovered": 0,
                "status": "scaffold",
                "note": "EDAS scanner not yet implemented",
            },
        )

    async def check(
        self,
        facility: str,
        ssh_host: str,
        signals: list[FacilitySignal],
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> list[dict[str, Any]]:
        """Validate EDAS signals return data for reference shot.

        TODO: Implement via SSH eddb_get or Python wrapper.
        """
        return [
            {
                "signal_id": s.id,
                "valid": False,
                "error": "EDAS check not yet implemented",
            }
            for s in signals
        ]


# Auto-register on import
register_scanner(EDASScanner())
