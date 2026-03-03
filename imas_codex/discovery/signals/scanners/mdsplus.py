"""Unified MDSplus scanner plugin.

Handles both static (versioned, machine-description) and dynamic (shot-scoped)
MDSplus trees through a single scanner interface. Static trees go through the
full EXTRACT → UNITS → ENRICH → PROMOTE pipeline; dynamic trees use lightweight
SSH enumeration.

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
from imas_codex.graph import GraphClient
from imas_codex.graph.models import DataAccess, FacilitySignal, FacilitySignalStatus
from imas_codex.remote.executor import run_python_script

logger = logging.getLogger(__name__)


class MDSplusScanner:
    """Unified MDSplus tree scanner.

    Discovers signals from both static (versioned) and dynamic (shot-scoped)
    MDSplus trees. Static trees use the full parallel discovery pipeline
    (extract → units → enrich) then promote enriched TreeNodes to
    FacilitySignals. Dynamic trees use SSH enumeration via
    enumerate_mdsplus.py.

    Config (data_sources.mdsplus):
        connection_tree: str - Default tree for TDI context
        setup_commands: list[str] - MDSplus environment setup
        reference_shot: int - Shot for dynamic tree introspection
        trees: list[str] - Dynamic subtree names to scan
        static_trees: list[dict] - Static tree configurations
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
        """Discover signals from both static and dynamic MDSplus trees.

        Static trees run the full parallel discovery pipeline internally,
        then promote enriched leaf TreeNodes to FacilitySignals with
        SOURCE_NODE edges. Dynamic trees enumerate via SSH and return
        FacilitySignals directly.
        """
        connection_tree = config.get("connection_tree")
        ref_shot = reference_shot or config.get("reference_shot")

        all_signals: list[FacilitySignal] = []
        all_stats: dict[str, Any] = {}

        # Phase 1: Static trees (versioned, machine-description)
        static_trees = config.get("static_trees", [])
        for tree_config in static_trees:
            tree_name = tree_config.get("tree_name")
            if not tree_name:
                continue

            logger.info("MDSplus scanner: processing static tree '%s'", tree_name)
            try:
                static_stats = await self._scan_static_tree(
                    facility, ssh_host, tree_name, tree_config, config
                )
                promoted = await self._promote_static_signals(
                    facility, tree_name, tree_config, config
                )
                all_stats[f"static_{tree_name}"] = {
                    **static_stats,
                    "promoted": promoted,
                }
                logger.info(
                    "Static tree '%s': promoted %d signals", tree_name, promoted
                )
            except Exception as e:
                logger.error("Static tree '%s' failed: %s", tree_name, e)
                all_stats[f"static_{tree_name}"] = {"error": str(e)}

        # Phase 2: Dynamic trees (shot-scoped)
        dynamic_trees = config.get("trees", [])
        if dynamic_trees and ref_shot:
            try:
                dynamic_signals, dynamic_stats = await self._scan_dynamic_trees(
                    facility, ssh_host, config, dynamic_trees, ref_shot
                )
                all_signals.extend(dynamic_signals)
                all_stats["dynamic"] = dynamic_stats
            except Exception as e:
                logger.error("Dynamic tree scan failed: %s", e)
                all_stats["dynamic"] = {"error": str(e)}
        elif dynamic_trees and not ref_shot:
            logger.warning(
                "MDSplus scanner: no reference_shot for dynamic trees in %s",
                facility,
            )
            all_stats["dynamic"] = {"error": "no reference_shot configured"}

        # Build DataAccess node for the connection tree
        data_access = None
        primary_tree = connection_tree or (dynamic_trees[0] if dynamic_trees else None)
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

        total_signals = len(all_signals)
        static_promoted = sum(
            s.get("promoted", 0)
            for s in all_stats.values()
            if isinstance(s, dict) and "promoted" in s
        )

        logger.info(
            "MDSplus scanner %s: %d dynamic signals, %d static promoted",
            facility,
            total_signals,
            static_promoted,
        )

        return ScanResult(
            signals=all_signals,
            data_access=data_access,
            metadata={"connection_tree": connection_tree},
            stats={
                "signals_discovered": total_signals,
                "static_promoted": static_promoted,
                **all_stats,
            },
        )

    async def _scan_static_tree(
        self,
        facility: str,
        ssh_host: str,
        tree_name: str,
        tree_config: dict[str, Any],
        mdsplus_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the full static discovery pipeline for a versioned tree.

        Delegates to run_parallel_static_discovery() which handles:
        - Seeding TreeModelVersion nodes
        - SSH extraction of tree structure per version
        - Unit extraction from MDSplus metadata
        - LLM enrichment of TreeNode descriptions
        """
        from imas_codex.discovery.static.parallel import (
            run_parallel_static_discovery,
        )

        versions = tree_config.get("versions", [])
        ver_list = [v["version"] for v in versions if "version" in v]
        if not ver_list:
            return {"error": "no versions configured"}

        # Merge setup_commands from parent mdsplus config
        merged_config = dict(tree_config)
        if "setup_commands" not in merged_config:
            merged_config["setup_commands"] = mdsplus_config.get("setup_commands", [])

        result = await run_parallel_static_discovery(
            facility=facility,
            ssh_host=ssh_host,
            tree_name=tree_name,
            tree_config=merged_config,
            ver_list=ver_list,
            enrich=True,
        )
        return result

    async def _promote_static_signals(
        self,
        facility: str,
        tree_name: str,
        tree_config: dict[str, Any],
        mdsplus_config: dict[str, Any],
    ) -> int:
        """Promote enriched leaf TreeNodes to FacilitySignals with SOURCE_NODE edges.

        Queries the graph for enriched leaf TreeNodes (NUMERIC/SIGNAL) from the
        static tree, creates FacilitySignal nodes with pre-populated descriptions
        and physics domains, and links them via SOURCE_NODE.

        Returns count of promoted signals.
        """
        accessor_function = tree_config.get("accessor_function")
        data_access_id = f"{facility}:mdsplus:tree_tdi"

        # Create static-specific DataAccess if there's an accessor function
        if accessor_function:
            data_access_id = f"{facility}:mdsplus:static_{tree_name}"
            try:
                with GraphClient() as gc:
                    gc.query(
                        """
                        MERGE (da:DataAccess {id: $id})
                        SET da.facility_id = $facility,
                            da.method_type = 'mdsplus',
                            da.library = 'MDSplus',
                            da.access_type = 'local',
                            da.data_source = 'mdsplus',
                            da.name = $name,
                            da.data_template = $template
                        WITH da
                        MATCH (f:Facility {id: $facility})
                        MERGE (da)-[:AT_FACILITY]->(f)
                        """,
                        id=data_access_id,
                        facility=facility,
                        name=f"MDSplus static {tree_name} ({accessor_function})",
                        template=f"{accessor_function}('{{accessor}}')",
                    )
            except Exception as e:
                logger.warning("Failed to create static DataAccess: %s", e)

        # Promote enriched leaf nodes to FacilitySignals
        try:
            with GraphClient() as gc:
                result = gc.query(
                    """
                    MATCH (n:TreeNode {facility_id: $facility, tree_name: $tree})
                    WHERE n.description IS NOT NULL
                      AND n.node_type IN ['NUMERIC', 'SIGNAL']
                    WITH n
                    WITH n,
                         $facility + ':' +
                         COALESCE(n.physics_domain, 'general') + '/' +
                         $tree + '/' +
                         toLower(split(n.path, ':')[-1]) AS sig_id
                    MERGE (s:FacilitySignal {id: sig_id})
                    ON CREATE SET
                        s.facility_id = $facility,
                        s.status = $enriched,
                        s.physics_domain = COALESCE(n.physics_domain, 'general'),
                        s.name = n.description,
                        s.description = n.description,
                        s.accessor = split(n.path, ':')[-1],
                        s.data_access = $da_id,
                        s.tree_name = $tree,
                        s.node_path = n.path,
                        s.unit = n.unit,
                        s.temporality = 'static',
                        s.discovery_source = 'tree_traversal',
                        s.source_node = n.id,
                        s.discovered_at = datetime(),
                        s.enriched_at = datetime()
                    ON MATCH SET
                        s.description = COALESCE(n.description, s.description),
                        s.unit = COALESCE(n.unit, s.unit),
                        s.source_node = n.id
                    WITH s, n
                    MERGE (s)-[:SOURCE_NODE]->(n)
                    WITH s
                    MATCH (f:Facility {id: $facility})
                    MERGE (s)-[:AT_FACILITY]->(f)
                    WITH s
                    MATCH (da:DataAccess {id: $da_id})
                    MERGE (s)-[:DATA_ACCESS]->(da)
                    RETURN count(s) AS promoted
                    """,
                    facility=facility,
                    tree=tree_name,
                    da_id=data_access_id,
                    enriched=FacilitySignalStatus.enriched.value,
                )
                return result[0]["promoted"] if result else 0
        except Exception as e:
            logger.error("Failed to promote static signals: %s", e)
            return 0

    async def _scan_dynamic_trees(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        trees: list[str],
        ref_shot: int,
    ) -> tuple[list[FacilitySignal], dict[str, Any]]:
        """Discover signals from dynamic MDSplus trees via SSH enumeration.

        Uses enumerate_mdsplus.py to walk trees and extract SIGNAL/NUMERIC
        nodes with data, filtering metadata and deduplicating channels.
        """
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

        setup_cmds = config.get("setup_commands")
        python_cmd = config.get("python_command", "python3")

        try:
            output = await asyncio.to_thread(
                run_python_script,
                "enumerate_mdsplus.py",
                input_data,
                ssh_host=ssh_host,
                timeout=300,
                python_command=python_cmd,
                setup_commands=setup_cmds,
            )

            # Find JSON in output — MDSplus C libraries may print warnings
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
                return [], {"error": "no JSON in remote script output"}
        except Exception as e:
            logger.error("MDSplus enumerate failed for %s: %s", facility, e)
            return [], {"error": f"enumerate failed: {e}"}

        if "error" in response:
            logger.error(
                "MDSplus enumerate error for %s: %s", facility, response["error"]
            )
            return [], {"error": response["error"]}

        raw_signals = response.get("signals", [])
        tree_stats = response.get("tree_stats", {})

        # Create DataAccess IDs for each tree
        data_access_nodes = {}
        for tree_name in trees:
            da_id = f"{facility}:mdsplus:{tree_name}"
            data_access_nodes[tree_name] = da_id

        # Convert to FacilitySignal objects
        facility_signals: list[FacilitySignal] = []
        for sig in raw_signals:
            tree_name = sig["tree"]
            name = sig["name"]
            path = sig["path"]
            group = sig.get("group", "TOP")

            signal_id = f"{facility}:{tree_name}/{group.lower()}/{name.lower()}"
            accessor = f"data({path})"

            display_name = name
            channel_count = sig.get("channel_count")
            if channel_count and channel_count > 1:
                display_name = f"{name} [{channel_count} channels]"

            try:
                fs = FacilitySignal(
                    id=signal_id,
                    facility_id=facility,
                    status=FacilitySignalStatus.discovered,
                    physics_domain="general",
                    name=display_name,
                    accessor=accessor,
                    data_access=data_access_nodes.get(tree_name, ""),
                    tree_name=tree_name,
                    node_path=path,
                    unit=sig.get("units") or None,
                    temporality="dynamic",
                    discovery_source="tree_traversal",
                    example_shot=ref_shot,
                )
                facility_signals.append(fs)
            except Exception as e:
                logger.debug("Could not create FacilitySignal for %s: %s", path, e)

        return facility_signals, {
            "signals_discovered": len(facility_signals),
            "trees_scanned": len(trees),
            "reference_shot": ref_shot,
            "tree_stats": tree_stats,
        }

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
