"""PPF (Processed Pulse File) scanner plugin for JET.

PPF is JET's primary data access system. Data is organized as:
  Owner / DDA (Diagnostic Data Area) / Dtype (Data Type)

Access methods (from graph DataAccess nodes):
  - SAL REST API: GET /salppc/rest/ppf/signal/{owner}/{pulse}/{seq}/{dda}/{dtype}
  - MATLAB: d = ppfget(pulse, dda, dtype)
  - IDL: ppfget, pulse, dda, dtype, data, x, irdat
  - CLI: getdat -a ppf -p <pulse> -o <owner> -i <dda>/<dtype>

Config key: data_sources.ppf
Facility: JET
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


class PPFScanner:
    """Discover signals from JET PPF (Processed Pulse File) system.

    PPF discovery strategy:
    1. Query SAL REST API to enumerate DDAs for a reference pulse
    2. For each DDA, enumerate available Dtypes
    3. Create FacilitySignal per DDA/Dtype with PPF accessor format
    4. Use default_owner (typically "jetppf") for standard processed data

    Config (data_sources.ppf):
        sal_endpoint: str - SAL REST API base URL
        reference_pulse: int - Pulse for signal enumeration
        default_owner: str - Default PPF owner (e.g., "jetppf")
        exclude_ddas: list[str] - DDAs to skip
    """

    scanner_type: str = "ppf"

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Discover signals from PPF via SAL REST API.

        TODO: Implement PPF enumeration via SAL endpoint.
        Scaffold returns empty result with config metadata.
        """
        sal_endpoint = config.get("sal_endpoint")
        ref_pulse = reference_shot or config.get("reference_pulse")
        default_owner = config.get("default_owner", "jetppf")
        exclude_ddas = set(config.get("exclude_ddas", []))

        logger.info(
            "PPF scanner scaffold: endpoint=%s, pulse=%s, owner=%s",
            sal_endpoint,
            ref_pulse,
            default_owner,
        )

        # TODO: Enumerate DDAs via SAL REST API
        # GET {sal_endpoint}/salppc/rest/ppf/signal/{owner}/{pulse}/0
        # Parse DDA list, then for each DDA get Dtypes
        # Create FacilitySignal with accessor="ppfget({pulse}, '{dda}', '{dtype}')"

        return ScanResult(
            signals=[],
            metadata={
                "sal_endpoint": sal_endpoint,
                "reference_pulse": ref_pulse,
                "default_owner": default_owner,
                "exclude_ddas": list(exclude_ddas),
            },
            stats={
                "signals_discovered": 0,
                "status": "scaffold",
                "note": "PPF scanner not yet implemented",
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
        """Validate PPF signals return data for reference pulse.

        TODO: Implement via SAL REST API data retrieval or SSH ppfget.
        """
        return [
            {
                "signal_id": s.id,
                "valid": False,
                "error": "PPF check not yet implemented",
            }
            for s in signals
        ]


# Auto-register on import
register_scanner(PPFScanner())
