"""MDSplus tree scanner plugin.

Direct MDSplus tree traversal for facilities that use raw tree/node access.
Discovers physics-relevant SIGNAL nodes by walking configured subtrees
via SSH, filtering out hardware/control channels and metadata nodes.

Strategy:
- Only scan configured physics subtrees (skip hardware: ATLAS, PCS, HYBRID)
- Only SIGNAL usage nodes (not NUMERIC/AXIS which are mostly config params)
- Filter: must have data for reference shot, skip metadata node names
- Deduplicate array channels (BPOL_003..BPOL_255 → single BPOL entry)

Config key: data_sources.mdsplus
Facility: Any facility with MDSplus (TCV, JET, JT-60SA, ITER)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from imas_codex.discovery.signals.scanners.base import (
    ScanResult,
    register_scanner,
)
from imas_codex.graph.models import DataAccess, FacilitySignal
from imas_codex.remote.executor import run_python_script

logger = logging.getLogger(__name__)


class MDSplusScanner:
    """Discover signals from MDSplus tree traversal.

    Walks MDSplus trees via SSH using enumerate_mdsplus.py to extract
    SIGNAL nodes with data. Produces FacilitySignal entries that
    complement higher-level scanners (TDI, PPF, EDAS).

    Config (data_sources.mdsplus):
        connection_tree: str - Default tree for TDI context
        trees: list[str] - Subtree names to scan for signals
        reference_shot: int - Shot for tree introspection
        exclude_node_names: list[str] - Node names to skip
    """

    scanner_type: str = "mdsplus"

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Discover signals from MDSplus trees via SSH.

        Uses enumerate_mdsplus.py remote script to walk trees and extract
        SIGNAL nodes with data, filtering metadata and deduplicating channels.
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
                    "note": "No trees configured; connection_tree used for TDI only",
                },
            )

        if not ref_shot:
            logger.warning("MDSplus scanner: no reference_shot for %s", facility)
            return ScanResult(
                stats={"error": "no reference_shot configured"},
            )

        # Build input for the remote enumerate script from facility config
        exclude_names = config.get("exclude_node_names", [])
        max_nodes = config.get("max_nodes_per_tree", 5000)
        node_usages = config.get("node_usages", ["SIGNAL"])
        input_data = {
            "trees": trees,
            "shot": ref_shot,
            "exclude_names": exclude_names,
            "max_nodes_per_tree": max_nodes,
            "node_usages": node_usages,
        }

        try:
            setup_cmds = config.get("setup_commands")
            output = await asyncio.to_thread(
                run_python_script,
                "enumerate_mdsplus.py",
                input_data,
                ssh_host=ssh_host,
                timeout=300,  # Trees can be large
                setup_commands=setup_cmds,
            )

            # Find JSON in output — MDSplus C libraries may print warnings
            # to stdout (fd 1) before our JSON. Find the line starting with {.
            response = None
            for line in reversed(output.strip().split("\n")):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        response = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

            if response is None:
                logger.error(
                    "MDSplus enumerate for %s: no JSON in output (%d bytes)",
                    facility,
                    len(output),
                )
                return ScanResult(
                    stats={"error": "no JSON in remote script output"},
                )
        except Exception as e:
            logger.error("MDSplus enumerate failed for %s: %s", facility, e)
            return ScanResult(
                stats={"error": f"enumerate failed: {e}"},
            )

        if "error" in response:
            logger.error(
                "MDSplus enumerate error for %s: %s", facility, response["error"]
            )
            return ScanResult(stats={"error": response["error"]})

        raw_signals = response.get("signals", [])
        tree_stats = response.get("tree_stats", {})

        # Create DataAccess for each tree
        data_access_nodes = {}
        for tree_name in trees:
            da_id = f"{facility}:mdsplus:{tree_name}"
            data_access_nodes[tree_name] = da_id

        # Convert to FacilitySignal objects
        from imas_codex.graph.models import FacilitySignalStatus

        facility_signals: list[FacilitySignal] = []
        for sig in raw_signals:
            tree_name = sig["tree"]
            name = sig["name"]
            path = sig["path"]
            group = sig.get("group", "TOP")

            # Build unique signal ID: facility:tree/group/name
            signal_id = f"{facility}:{tree_name}/{group.lower()}/{name.lower()}"

            # Build accessor: MDSplus path expression
            accessor = f"data({path})"

            # Note channel count in name if it's a channel group
            display_name = name
            channel_count = sig.get("channel_count")
            if channel_count and channel_count > 1:
                display_name = f"{name} [{channel_count} channels]"

            try:
                fs = FacilitySignal(
                    id=signal_id,
                    facility_id=facility,
                    status=FacilitySignalStatus.discovered,
                    physics_domain="general",  # Enriched by LLM later
                    name=display_name,
                    accessor=accessor,
                    data_access=data_access_nodes.get(tree_name, ""),
                    tree_name=tree_name,
                    node_path=path,
                    units=sig.get("units", ""),
                    discovery_source="tree_traversal",
                    example_shot=ref_shot,
                )
                facility_signals.append(fs)
            except Exception as e:
                logger.debug("Could not create FacilitySignal for %s: %s", path, e)

        # Build DataAccess model for the primary connection tree
        data_access = DataAccess(
            id=f"{facility}:mdsplus:{connection_tree or trees[0]}",
            facility_id=facility,
            method_type="mdsplus",
            library="MDSplus",
            access_type="local",
            connection_template=(
                f"import MDSplus\n"
                f"tree = MDSplus.Tree('{connection_tree or trees[0]}', "
                f"{{shot}}, 'readonly')"
            ),
            data_template="data = tree.getNode('{node_path}').data()",
            data_source="mdsplus",
        )

        logger.info(
            "MDSplus scanner %s: %d signals from %d trees (shot %d)",
            facility,
            len(facility_signals),
            len(trees),
            ref_shot,
        )

        return ScanResult(
            signals=facility_signals,
            data_access=data_access,
            metadata={
                "trees_scanned": list(trees),
                "reference_shot": ref_shot,
                "tree_stats": tree_stats,
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

        setup_cmds = config.get("setup_commands")

        try:
            output = await asyncio.to_thread(
                run_python_script,
                "check_signals_batch.py",
                {"signals": batch_input, "timeout_per_group": 30},
                ssh_host=ssh_host,
                timeout=60 + len(batch_input),
                setup_commands=setup_cmds,
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
