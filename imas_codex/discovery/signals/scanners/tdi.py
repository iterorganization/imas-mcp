"""TDI function scanner plugin.

Wraps the existing TDI discovery pipeline (imas_codex.discovery.signals.tdi)
as a DataSourceScanner for the plugin registry.

Config key: data_sources.tdi
Facility: TCV (and any facility with TDI .fun files)
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


class TDIScanner:
    """Discover signals from MDSplus TDI function files (.fun).

    TDI functions provide physics-level abstraction over raw MDSplus paths.
    The scanner parses .fun files via SSH, extracts supported quantities,
    and creates FacilitySignal nodes with TDI accessor expressions.

    Config (data_sources.tdi):
        primary_path: str - Directory containing .fun files
        additional_paths: list[str] - Extra directories to scan
        reference_shot: int - Shot for validation
        exclude_functions: list[str] - Functions to skip (hardware/ops)
    """

    scanner_type: str = "tdi"

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Discover signals from TDI function files.

        Delegates to the existing tdi.py pipeline which handles:
        - SSH extraction of .fun file metadata
        - Exclude list filtering from facility config
        - Physics domain heuristic classification
        - Signal/function node creation
        """
        from imas_codex.discovery.signals.tdi import (
            create_tdi_data_access,
            discover_tdi_signals,
        )
        from imas_codex.graph import GraphClient

        tdi_path = config.get("primary_path")
        if not tdi_path:
            logger.warning("No primary_path in TDI config for %s", facility)
            return ScanResult(stats={"error": "no primary_path configured"})

        ref_shot = reference_shot or config.get("reference_shot")

        with GraphClient() as gc:
            # Create/get DataAccess node
            data_access = await create_tdi_data_access(gc, facility)

            # Discover signals (loads exclude_functions from facility config)
            signals, functions = await discover_tdi_signals(
                facility=facility,
                ssh_host=ssh_host,
                tdi_path=tdi_path,
                data_access_id=data_access.id,
            )

        return ScanResult(
            signals=signals,
            data_access=data_access,
            metadata={
                "functions": [
                    {
                        "name": f.name,
                        "path": f.path,
                        "quantity_count": len(f.quantities),
                    }
                    for f in functions
                ],
                "tdi_path": tdi_path,
                "reference_shot": ref_shot,
            },
            stats={
                "signals_discovered": len(signals),
                "functions_parsed": len(functions),
                "tdi_path": tdi_path,
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
        """Validate TDI signals return data for reference shot.

        Uses the check_signals_batch.py remote script which groups signals
        by tree/shot for efficient batch validation.
        """
        import asyncio
        import json

        from imas_codex.remote.executor import run_python_script

        ref_shot = reference_shot or config.get("reference_shot")
        if not ref_shot:
            return [
                {"signal_id": s.id, "valid": False, "error": "no reference_shot"}
                for s in signals
            ]

        batch_input = [
            {
                "id": s.id,
                "accessor": s.accessor,
                "tree_name": "tcv_shot",  # TDI default tree
                "shot": ref_shot,
            }
            for s in signals
        ]

        try:
            output = await asyncio.to_thread(
                run_python_script,
                "check_signals_batch.py",
                {"signals": batch_input, "timeout_per_group": 30},
                ssh_host=ssh_host,
                timeout=60 + len(batch_input),
                setup_commands=config.get("setup_commands"),
            )

            response = json.loads(output.strip().split("\n")[-1])
            return [
                {
                    "signal_id": r["id"],
                    "valid": r.get("success", False),
                    "shape": r.get("shape"),
                    "dtype": r.get("dtype"),
                    "error": r.get("error"),
                }
                for r in response.get("results", [])
            ]
        except Exception as e:
            logger.error("TDI check failed: %s", e)
            return [
                {"signal_id": s.id, "valid": False, "error": str(e)} for s in signals
            ]


# Auto-register on import
register_scanner(TDIScanner())
