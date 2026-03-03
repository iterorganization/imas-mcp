"""Unified MDSplus scanner plugin.

Thin loop over config.trees, delegating to run_tree_discovery() for the
full EXTRACT → UNITS → PROMOTE pipeline per tree. After tree processing,
runs TDI linkage to connect TDI functions to TreeNodes.

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
    """Unified MDSplus tree scanner.

    Iterates over config.trees and runs the unified tree discovery
    pipeline (extract → units → promote) for each tree. Versioned trees
    use their configured versions; dynamic trees use the reference_shot.
    Subtrees are expanded and processed individually.

    After tree extraction, runs TDI linkage to connect TDI function
    build_path references to TreeNode nodes in the graph.

    Config (data_sources.mdsplus):
        connection_tree: str - Default tree for TDI context
        setup_commands: list[str] - MDSplus environment setup
        reference_shot: int - Shot for dynamic tree introspection
        trees: list[TreeConfig] - Unified tree configurations
        node_usages: list[str] - Node types to include
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
        """Discover signals from MDSplus trees via unified pipeline.

        For each tree in config.trees, runs run_tree_discovery() which
        handles extraction, units, and promotion to FacilitySignals.
        Signals are written directly to the graph — returned ScanResult
        has an empty signals list.
        """
        from imas_codex.discovery.mdsplus.pipeline import run_tree_discovery
        from imas_codex.discovery.mdsplus.tdi_linkage import link_tdi_to_tree_nodes

        connection_tree = config.get("connection_tree")
        ref_shot = reference_shot or config.get("reference_shot")

        all_stats: dict[str, Any] = {}
        total_promoted = 0

        # Process each tree through the unified pipeline
        for tree_config in config.get("trees", []):
            tree_name = tree_config.get("tree_name")
            if not tree_name:
                continue

            # Merge setup_commands from parent config
            merged_config = dict(tree_config)
            if "setup_commands" not in merged_config:
                merged_config["setup_commands"] = config.get("setup_commands", [])

            # Expand subtrees — each subtree is processed as its own tree
            subtrees = tree_config.get("subtrees", [])
            trees_to_process = (
                [
                    (st["tree_name"], {**merged_config, **st})
                    for st in subtrees
                    if st.get("tree_name")
                ]
                if subtrees
                else [(tree_name, merged_config)]
            )

            for sub_name, sub_config in trees_to_process:
                # Resolve version list: versioned trees use config,
                # dynamic trees use reference_shot
                versions = sub_config.get("versions", [])
                ver_list = [v["version"] for v in versions if "version" in v]
                if not ver_list and ref_shot:
                    ver_list = [ref_shot]
                elif not ver_list:
                    logger.warning(
                        "MDSplus scanner: no versions or reference_shot "
                        "for tree '%s' in %s",
                        sub_name,
                        facility,
                    )
                    all_stats[sub_name] = {
                        "error": "no versions or reference_shot",
                    }
                    continue

                logger.info(
                    "MDSplus scanner: processing tree '%s' (%d versions)",
                    sub_name,
                    len(ver_list),
                )
                try:
                    stats = await run_tree_discovery(
                        facility=facility,
                        ssh_host=ssh_host,
                        tree_name=sub_name,
                        tree_config=sub_config,
                        ver_list=ver_list,
                    )
                    all_stats[sub_name] = stats
                    total_promoted += stats.get("signals_promoted", 0)
                    logger.info(
                        "Tree '%s': %d promoted",
                        sub_name,
                        stats.get("signals_promoted", 0),
                    )
                except Exception as e:
                    logger.error("Tree '%s' failed: %s", sub_name, e)
                    all_stats[sub_name] = {"error": str(e)}

        # Run TDI linkage after all trees are processed
        tdi_links = 0
        try:
            tdi_links = link_tdi_to_tree_nodes(facility)
            if tdi_links:
                logger.info("TDI linkage: created %d edges", tdi_links)
        except Exception as e:
            logger.warning("TDI linkage failed: %s", e)

        # Build DataAccess node for the connection tree
        data_access = None
        first_tree = next(
            (t["tree_name"] for t in config.get("trees", []) if t.get("tree_name")),
            None,
        )
        primary_tree = connection_tree or first_tree
        if primary_tree:
            data_access = DataAccess(
                id=f"{facility}:mdsplus:tree_tdi",
                facility_id=facility,
                method_type="mdsplus",
                library="MDSplus",
                access_type="local",
                connection_template=(
                    f"import MDSplus\n"
                    f"tree = MDSplus.Tree('{primary_tree}', "
                    f"{{shot}}, 'readonly')"
                ),
                data_template="data = tree.getNode('{node_path}').data()",
                data_source="mdsplus",
            )

        logger.info(
            "MDSplus scanner %s: %d signals promoted, %d TDI links",
            facility,
            total_promoted,
            tdi_links,
        )

        return ScanResult(
            signals=[],
            data_access=data_access,
            metadata={"connection_tree": connection_tree},
            stats={
                "signals_promoted": total_promoted,
                "tdi_links": tdi_links,
                **all_stats,
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
        python_cmd = config.get("python_command", "python3")

        try:
            output = await asyncio.to_thread(
                run_python_script,
                "check_signals_batch.py",
                {"signals": batch_input, "timeout_per_group": 30},
                ssh_host=ssh_host,
                timeout=60 + len(batch_input),
                python_command=python_cmd,
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
