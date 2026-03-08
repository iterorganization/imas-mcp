"""Unified MDSplus scanner plugin.

Supports two access modes:
1. Tree-based (TCV, JT-60SA): Iterates config.trees, delegates to
   run_tree_discovery() for EXTRACT → UNITS → PROMOTE per tree.
2. Thin-client (JET): Connects to a remote MDSplus server via
   MDSplus.Connection and uses TDI functions (dpf/jpf/ppf) for
   data access. Enumerates JPF subsystems and validates access.

Config key: data_systems.mdsplus
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

logger = logging.getLogger(__name__)


class MDSplusScanner:
    """Unified MDSplus scanner for tree-based and thin-client facilities.

    Two access modes based on config.connection_type:

    **Tree-based** (default, TCV/JT-60SA):
        Iterates over config.trees and runs the unified tree discovery
        pipeline (extract → units → promote) for each tree. After tree
        extraction, runs TDI linkage to connect TDI function build_path
        references to DataNode nodes.

    **Thin-client** (JET):
        Connects to a remote MDSplus server via MDSplus.Connection and
        uses TDI functions (dpf/jpf/ppf) for data access. Enumerates
        JPF subsystems and creates DataAccess nodes for MDSplus-based
        access to existing signals.

    Config (data_systems.mdsplus):
        connection_type: str - "thin_client" for remote, omit for local
        server: str - MDSplus server hostname (thin_client only)
        connection_tree: str - Default tree for TDI context (tree-based)
        setup_commands: list[str] - MDSplus environment setup
        reference_shot: int - Shot for validation/introspection
        trees: list[TreeConfig] - Tree configurations (tree-based)
        jpf_subsystems: list[str] - JPF subsystem codes (thin_client)
    """

    scanner_type: str = "mdsplus"

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Discover signals from MDSplus (tree-based or thin-client).

        Dispatches to _scan_trees() or _scan_thin_client() based on
        config.connection_type.
        """
        connection_type = config.get("connection_type", "local")

        if connection_type == "thin_client":
            return await self._scan_thin_client(
                facility, ssh_host, config, reference_shot
            )
        return await self._scan_trees(facility, ssh_host, config, reference_shot)

    async def _scan_trees(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Tree-based scan: walk MDSplus trees via SSH (TCV, JT-60SA)."""
        from imas_codex.discovery.mdsplus.pipeline import run_tree_discovery
        from imas_codex.discovery.mdsplus.tdi_linkage import link_tdi_to_data_nodes

        connection_tree = config.get("connection_tree")
        ref_shot = reference_shot or config.get("reference_shot")

        all_stats: dict[str, Any] = {}
        total_promoted = 0

        # Process each tree through the unified pipeline
        for tree_config in config.get("trees", []):
            data_source_name = tree_config.get("source_name")
            if not data_source_name:
                continue

            # Merge setup_commands from parent config
            merged_config = dict(tree_config)
            if "setup_commands" not in merged_config:
                merged_config["setup_commands"] = config.get("setup_commands", [])

            # Expand subtrees — each subtree is processed as its own tree
            subtrees = tree_config.get("subtrees", [])
            trees_to_process = (
                [
                    (st["source_name"], {**merged_config, **st})
                    for st in subtrees
                    if st.get("source_name")
                ]
                if subtrees
                else [(data_source_name, merged_config)]
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
                        data_source_name=sub_name,
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
            tdi_links = link_tdi_to_data_nodes(facility)
            if tdi_links:
                logger.info("TDI linkage: created %d edges", tdi_links)
        except Exception as e:
            logger.warning("TDI linkage failed: %s", e)

        # Build DataAccess node for the connection tree
        data_access = None
        first_tree = next(
            (t["source_name"] for t in config.get("trees", []) if t.get("source_name")),
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
                data_template="data = tree.getNode('{data_source_path}').data()",
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

    async def _scan_thin_client(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Thin-client scan: connect to remote MDSplus server via TDI functions.

        Connects to the MDSplus server, enumerates JPF subsystems,
        validates data access, and creates a DataAccess node for
        MDSplus-based signal access.
        """
        from imas_codex.remote.executor import async_run_python_script

        server = config.get("server")
        ref_shot = reference_shot or config.get("reference_shot")

        if not server:
            logger.warning("MDSplus thin-client: no server configured for %s", facility)
            return ScanResult(stats={"error": "no server configured"})
        if not ref_shot:
            logger.warning("MDSplus thin-client: no reference_shot for %s", facility)
            return ScanResult(stats={"error": "no reference_shot"})

        logger.info(
            "MDSplus thin-client: connecting to %s for %s (shot %d)",
            server,
            facility,
            ref_shot,
        )

        # Sample signals to validate access patterns
        sample_signals = [
            {"path": "da/c2-ipla", "type": "jpf"},
            {"path": "EFIT/RBND", "type": "ppf"},
            {"path": "EFIT/RLIM", "type": "ppf"},
        ]

        try:
            output = await async_run_python_script(
                "enumerate_mdsplus_tdi.py",
                {
                    "server": server,
                    "shot": ref_shot,
                    "jpf_subsystems": config.get("jpf_subsystems", []),
                    "sample_signals": sample_signals,
                },
                ssh_host=ssh_host,
                timeout=120,
                python_command=config.get("python_command", "python3"),
                setup_commands=config.get("setup_commands"),
            )
            data = json.loads(output.strip().split("\n")[-1])
        except Exception as e:
            logger.error("MDSplus thin-client enumeration failed: %s", e)
            return ScanResult(stats={"error": str(e)[:300]})

        if "error" in data:
            logger.error("MDSplus thin-client error: %s", data["error"])
            return ScanResult(stats={"error": data["error"]})

        # Create DataAccess node for thin-client MDSplus
        data_access = DataAccess(
            id=f"{facility}:mdsplus:thin_client",
            facility_id=facility,
            name=f"MDSplus Thin-Client ({server})",
            method_type="mdsplus",
            library="MDSplus",
            access_type="remote",
            data_source="mdsplus",
            connection_template=(
                f"import MDSplus\nconn = MDSplus.Connection('{server}')"
            ),
            data_template=(
                "# JPF raw signal:\n"
                "data = conn.get('dpf(\"{signal_path}\", {shot})')\n"
                "# PPF processed signal:\n"
                "data = conn.get('ppf(\"{dda}/{dtype}\", {shot})')"
            ),
            setup_commands=config.get("setup_commands"),
        )

        valid_checks = sum(1 for c in data.get("signal_checks", []) if c.get("valid"))
        geometry = data.get("geometry", {})
        geometry_available = sum(1 for g in geometry.values() if g.get("available"))

        logger.info(
            "MDSplus thin-client %s: server=%s, %d/%d signals valid, "
            "%d subsystems active, %d geometry endpoints, %d TDI functions",
            facility,
            server,
            valid_checks,
            len(data.get("signal_checks", [])),
            len(data.get("active_subsystems", [])),
            geometry_available,
            data.get("tdi_function_count", 0),
        )

        return ScanResult(
            signals=[],
            data_access=data_access,
            metadata={
                "server": server,
                "connection_type": "thin_client",
                "all_subsystems": data.get("all_subsystems", []),
                "active_subsystems": data.get("active_subsystems", []),
                "geometry": geometry,
                "tdi_function_count": data.get("tdi_function_count", 0),
            },
            stats={
                "server": server,
                "connected": data.get("connected", False),
                "signals_checked": len(data.get("signal_checks", [])),
                "signals_valid": valid_checks,
                "active_subsystems": len(data.get("active_subsystems", [])),
                "total_subsystems": len(data.get("all_subsystems", [])),
                "geometry_available": geometry_available,
                "tdi_function_count": data.get("tdi_function_count", 0),
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

        Dispatches to tree-based or thin-client check based on
        config.connection_type.
        """
        connection_type = config.get("connection_type", "local")

        if connection_type == "thin_client":
            return await self._check_thin_client(
                facility, ssh_host, signals, config, reference_shot
            )
        return await self._check_trees(
            facility, ssh_host, signals, config, reference_shot
        )

    async def _check_trees(
        self,
        facility: str,
        ssh_host: str,
        signals: list[FacilitySignal],
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> list[dict[str, Any]]:
        """Validate signals via tree-based MDSplus access."""
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
                "data_source_name": s.data_source_name
                or config.get("connection_tree", ""),
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
            logger.error("MDSplus tree check failed: %s", e)
            return [
                {"signal_id": s.id, "valid": False, "error": str(e)} for s in signals
            ]

    async def _check_thin_client(
        self,
        facility: str,
        ssh_host: str,
        signals: list[FacilitySignal],
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> list[dict[str, Any]]:
        """Validate signals via thin-client MDSplus connection."""
        from imas_codex.remote.executor import async_run_python_script

        server = config.get("server")
        ref_shot = reference_shot or config.get("reference_shot")

        if not server or not ref_shot:
            return [
                {
                    "signal_id": s.id,
                    "valid": False,
                    "error": "no server or reference_shot",
                }
                for s in signals
            ]

        # Build signal batch — infer type from signal name/accessor
        batch = []
        for s in signals:
            name = s.name or ""
            accessor = s.accessor or ""
            parts = name.split("/")
            sig_type = "ppf"
            path = name

            if "dpf(" in accessor or "jpf(" in accessor:
                sig_type = "jpf"
            elif len(parts) == 2 and parts[0].islower():
                sig_type = "jpf"

            batch.append({"id": s.id, "path": path, "type": sig_type})

        try:
            output = await async_run_python_script(
                "check_mdsplus_tdi.py",
                {"server": server, "shot": ref_shot, "signals": batch},
                ssh_host=ssh_host,
                timeout=120,
                python_command=config.get("python_command", "python3"),
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
            logger.error("MDSplus thin-client check failed: %s", e)
            return [
                {"signal_id": s.id, "valid": False, "error": str(e)[:200]}
                for s in signals
            ]


# Auto-register on import
register_scanner(MDSplusScanner())
