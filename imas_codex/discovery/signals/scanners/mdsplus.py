"""MDSplus tree scanner plugin.

Direct MDSplus tree traversal for facilities that use raw tree/node access
rather than a higher-level abstraction layer (TDI, PPF, EDAS).

This scanner walks MDSplus tree nodes via SSH, extracting SIGNAL, NUMERIC,
and AXIS nodes as FacilitySignal candidates.

Config key: data_sources.mdsplus
Facility: Any facility with MDSplus (TCV, JET, JT-60SA, ITER)
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


class MDSplusScanner:
    """Discover signals from MDSplus tree traversal.

    Walks MDSplus trees via SSH to enumerate data-bearing nodes.
    Useful as a complement to higher-level scanners (e.g., TDI on TCV)
    to find signals not exposed through the abstraction layer.

    Config (data_sources.mdsplus):
        connection_tree: str - Default tree for TDI context
        trees: list[str] - Tree names to scan
        reference_shot: int - Shot for tree introspection
    """

    scanner_type: str = "mdsplus"

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Discover signals from MDSplus trees.

        Uses the existing discover_mdsplus_signals() function from parallel.py
        which walks tree nodes via SSH and extracts SIGNAL/NUMERIC/AXIS nodes.
        """
        trees = config.get("trees", [])
        ref_shot = reference_shot or config.get("reference_shot")
        connection_tree = config.get("connection_tree")

        if not trees:
            logger.info(
                "MDSplus scanner: no trees configured for %s (connection_tree=%s)",
                facility,
                connection_tree,
            )
            return ScanResult(
                metadata={"connection_tree": connection_tree},
                stats={
                    "signals_discovered": 0,
                    "note": "No trees configured for scanning; connection_tree used for TDI context only",
                },
            )

        if not ref_shot:
            logger.warning("MDSplus scanner: no reference_shot for %s", facility)
            return ScanResult(
                stats={"error": "no reference_shot configured"},
            )

        # Import the existing MDSplus discovery function
        from imas_codex.discovery.signals.parallel import discover_mdsplus_signals

        all_signals: list[dict] = []
        tree_stats: dict[str, int] = {}

        for tree_name in trees:
            data_access_id = f"{facility}:mdsplus:{tree_name}"
            try:
                signals = discover_mdsplus_signals(
                    facility=facility,
                    ssh_host=ssh_host,
                    tree_name=tree_name,
                    shot=ref_shot,
                    data_access_id=data_access_id,
                )
                all_signals.extend(signals)
                tree_stats[tree_name] = len(signals)
                logger.info(
                    "MDSplus %s:%s shot %d: %d signals",
                    facility,
                    tree_name,
                    ref_shot,
                    len(signals),
                )
            except Exception as e:
                logger.error(
                    "MDSplus scan failed for %s:%s: %s", facility, tree_name, e
                )
                tree_stats[tree_name] = -1

        # Convert raw dicts to FacilitySignal objects
        from imas_codex.graph.models import FacilitySignal as FS, FacilitySignalStatus

        facility_signals = []
        for sig_dict in all_signals:
            try:
                facility_signals.append(
                    FS(
                        id=sig_dict["id"],
                        facility_id=facility,
                        status=FacilitySignalStatus.discovered,
                        physics_domain=sig_dict.get("physics_domain", "general"),
                        name=sig_dict.get("name", ""),
                        accessor=sig_dict.get("accessor", ""),
                        data_access=sig_dict.get("data_access"),
                        tree_name=sig_dict.get("tree_name"),
                        node_path=sig_dict.get("node_path"),
                        units=sig_dict.get("units"),
                        discovery_source="tree_traversal",
                    )
                )
            except Exception as e:
                logger.debug(
                    "Could not create FacilitySignal from %s: %s", sig_dict.get("id"), e
                )

        return ScanResult(
            signals=facility_signals,
            metadata={
                "trees_scanned": list(trees),
                "reference_shot": ref_shot,
                "tree_signal_counts": tree_stats,
            },
            stats={
                "signals_discovered": len(facility_signals),
                "trees_scanned": len(trees),
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
        """Validate MDSplus signals return data for reference shot.

        Uses check_signals_batch.py which groups signals by tree/shot
        for efficient batch validation.
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
                "tree_name": s.tree_name or config.get("connection_tree", ""),
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
            logger.error("MDSplus check failed: %s", e)
            return [
                {"signal_id": s.id, "valid": False, "error": str(e)} for s in signals
            ]


# Auto-register on import
register_scanner(MDSplusScanner())
