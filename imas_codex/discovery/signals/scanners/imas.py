"""IMAS IDS scanner plugin.

Enumerates filled IDSs (Interface Data Structures) at facilities with native
IMAS backend support. Discovers which IDSs contain data for a reference shot,
creating FacilitySignal nodes for each filled IDS top-level signal.

Config key: data_sources.imas
Facility: ITER (primary), JET (post-migration), JT-60SA
"""

from __future__ import annotations

import logging
from typing import Any

from imas_codex.discovery.signals.scanners.base import (
    ScanResult,
    register_scanner,
)
from imas_codex.graph.models import FacilitySignal

logger = logging.getLogger(__name__)


class IMASScanner:
    """Discover signals from IMAS IDS data.

    IMAS discovery strategy:
    1. Open IMAS data entry for reference shot
    2. Enumerate all IDS types that contain data
    3. For each filled IDS, walk top-level arrays/signals
    4. Create FacilitySignal per (IDS, signal_path) with IMAS accessor format

    Config (data_sources.imas):
        backends: list[str] - Available IMAS backends (mdsplus, hdf5, memory)
        db_name: str - IMAS database name
        reference_shot: int - Shot for IDS enumeration
    """

    scanner_type: str = "imas"

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Discover filled IDSs from IMAS data entry.

        TODO: Implement IMAS IDS enumeration via SSH.
        Requires imas Python package on remote host.
        Scaffold returns empty result with config metadata.
        """
        backends = config.get("backends", [])
        db_name = config.get("db_name")
        ref_shot = reference_shot or config.get("reference_shot")

        logger.info(
            "IMAS scanner scaffold: backends=%s, db=%s, shot=%s",
            backends,
            db_name,
            ref_shot,
        )

        # TODO: Enumerate filled IDSs via SSH
        # ssh host "python3 -c 'import imas; entry = imas.DBEntry(...)'"
        # For each IDS: ids_info.get_filled_ids_list()
        # Walk top-level signal paths within each filled IDS
        # Create FacilitySignal with accessor="ids.{ids_name}.{signal_path}"

        return ScanResult(
            signals=[],
            metadata={
                "backends": backends,
                "db_name": db_name,
                "reference_shot": ref_shot,
            },
            stats={
                "signals_discovered": 0,
                "status": "scaffold",
                "note": "IMAS scanner not yet implemented",
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
        """Validate IMAS signals return data for reference shot.

        TODO: Implement via SSH imas Python API.
        """
        return [
            {
                "signal_id": s.id,
                "valid": False,
                "error": "IMAS check not yet implemented",
            }
            for s in signals
        ]


# Auto-register on import
register_scanner(IMASScanner())
