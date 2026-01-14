"""
Agents MCP Server - Streamlined tools for LLM-driven facility exploration.

This server provides 4 core MCP tools:
- python: Persistent Python REPL with rich pre-loaded utilities
- get_graph_schema: Schema introspection for query generation
- ingest_nodes: Schema-validated node creation with privacy filtering
- private: Read/update sensitive infrastructure files

The python() REPL is the primary interface, providing:
- Graph: query(), semantic_search(), embed()
- Remote: ssh(), check_tools()
- Facility: get_facility(), get_exploration_targets(), get_tree_structure()
- IMAS DD: search_imas(), fetch_imas(), list_imas(), check_imas()
- COCOS: validate_cocos(), determine_cocos(), cocos_sign_flip_paths(), cocos_info()
- Code: search_code()

REPL state is loaded eagerly on server startup for instant tool response.
"""

import asyncio
import io
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Literal

from fastmcp import FastMCP
from neo4j.exceptions import ServiceUnavailable
from ruamel.yaml import YAML

from imas_codex.agents.prompt_loader import (
    PromptDefinition,
    load_prompts,
)
from imas_codex.discovery import (
    get_facility as _get_facility_config,
    get_facility_private,
    save_private,
)
from imas_codex.graph import GraphClient, get_schema
from imas_codex.graph.schema import to_cypher_props

logger = logging.getLogger(__name__)

# Configure ruamel.yaml for comment-preserving round-trips
_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 120

# Neo4j connection error message
NEO4J_NOT_RUNNING_MSG = (
    "Neo4j is not running. Start it with: uv run imas-codex neo4j start"
)


def _serialize_neo4j_value(value: Any) -> Any:
    """Serialize Neo4j values to JSON-compatible types."""
    if value is None:
        return None
    if hasattr(value, "isoformat") and hasattr(value, "tzinfo"):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialize_neo4j_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_neo4j_value(v) for v in value]
    return value


def _neo4j_error_message(e: Exception) -> str:
    """Format Neo4j errors with helpful instructions."""
    if isinstance(e, ServiceUnavailable):
        return NEO4J_NOT_RUNNING_MSG
    if "Connection refused" in str(e) or "ServiceUnavailable" in str(e):
        return NEO4J_NOT_RUNNING_MSG
    return str(e)


# =============================================================================
# Persistent Python REPL - Loaded eagerly on server startup
# =============================================================================

_repl_globals: dict[str, Any] | None = None
_imas_tools_instance = None


def _get_imas_tools():
    """Get or create singleton Tools instance with shared DocumentStore."""
    global _imas_tools_instance
    if _imas_tools_instance is None:
        from imas_codex.tools import Tools

        _imas_tools_instance = Tools()
    return _imas_tools_instance


def _run_async(coro):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)


def _init_repl() -> dict[str, Any]:
    """Initialize the persistent REPL environment with all utilities.

    Called once at server startup. Loads embedding model and IMAS tools.
    """
    global _repl_globals
    if _repl_globals is not None:
        return _repl_globals

    logger.info("Initializing Python REPL (loading embedding model and IMAS tools...)")

    from imas_codex.code_examples.pipeline import get_embed_model
    from imas_codex.code_examples.search import CodeExampleSearch
    from imas_codex.graph import GraphClient

    gc = GraphClient()
    embed_model = get_embed_model()

    # =========================================================================
    # Core utilities
    # =========================================================================

    def ssh(cmd: str, facility: str = "epfl", timeout: int = 60) -> str:
        """Run SSH command on remote facility.

        Args:
            cmd: Shell command to execute
            facility: SSH host alias (default: 'epfl')
            timeout: Command timeout in seconds

        Returns:
            Command output (stdout + stderr)

        Examples:
            ssh('ls /home/codes | head -5')
            ssh('rg -l equilibrium /home/codes --max-depth 2')
        """
        result = subprocess.run(
            ["ssh", facility, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        return output.strip() or "(no output)"

    def query(cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Execute Cypher query and return results.

        Args:
            cypher: Cypher query string
            **params: Query parameters

        Returns:
            List of result records as dicts

        Examples:
            query('MATCH (f:Facility) RETURN f.id, f.name')
            query('MATCH (t:TreeNode {tree_name: $tree}) RETURN t.path LIMIT 10', tree='results')
        """
        return gc.query(cypher, **params)

    def embed(text: str) -> list[float]:
        """Get embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            384-dim embedding vector
        """
        return embed_model.get_text_embedding(text)

    def semantic_search(
        text: str,
        index: str = "imas_path_embedding",
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Vector similarity search on graph embeddings.

        Args:
            text: Query text to embed and search
            index: Vector index name
            k: Number of results to return

        Available indexes:
            - imas_path_embedding: IMAS Data Dictionary paths (61k)
            - code_chunk_embedding: Code examples (8.5k chunks)
            - wiki_chunk_embedding: Wiki documentation (25k chunks)
            - cluster_centroid: Semantic clusters

        Returns:
            List of {node: ..., score: float} dicts
        """
        embedding = embed_model.get_text_embedding(text)
        return gc.query(
            f'CALL db.index.vector.queryNodes("{index}", $k, $embedding) '
            "YIELD node, score RETURN node, score ORDER BY score DESC",
            k=k,
            embedding=embedding,
        )

    # =========================================================================
    # Facility utilities
    # =========================================================================

    def get_facility(facility: str) -> dict[str, Any]:
        """Get comprehensive facility info including graph state.

        Args:
            facility: Facility ID (e.g., 'epfl')

        Returns:
            Dict with config, tools, paths, graph_summary, actionable_paths
        """
        result: dict[str, Any] = {"facility": facility}

        # Load facility config
        try:
            data = _get_facility_config(facility)
            result["config"] = {
                "ssh_host": data.get("ssh_host"),
                "description": data.get("description"),
                "machine": data.get("machine"),
                "name": data.get("name"),
            }
            result["tools"] = data.get("tools", {})
            result["paths"] = data.get("paths", {})
            result["exploration_notes"] = data.get("exploration_notes", [])
        except Exception as e:
            result["error"] = str(e)
            return result

        # Query graph for facility summary
        try:
            summary = gc.query(
                """
                MATCH (f:Facility {id: $fid})
                OPTIONAL MATCH (a:AnalysisCode)-[:FACILITY_ID]->(f)
                OPTIONAL MATCH (d:Diagnostic)-[:FACILITY_ID]->(f)
                OPTIONAL MATCH (t:TDIFunction)-[:FACILITY_ID]->(f)
                OPTIONAL MATCH (m:MDSplusTree)-[:FACILITY_ID]->(f)
                RETURN
                    count(DISTINCT a) AS analysis_codes,
                    count(DISTINCT d) AS diagnostics,
                    count(DISTINCT t) AS tdi_functions,
                    count(DISTINCT m) AS mdsplus_trees
                """,
                fid=facility,
            )
            if summary:
                result["graph_summary"] = summary[0]

            # Get actionable paths
            actionable = gc.query(
                """
                MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $fid})
                WHERE p.status = 'discovered'
                RETURN p.path AS path, p.interest_score AS score, p.description AS description
                ORDER BY COALESCE(p.interest_score, 0) DESC
                LIMIT 20
                """,
                fid=facility,
            )
            result["actionable_paths"] = actionable
        except Exception as e:
            result["graph_error"] = str(e)

        return result

    def get_exploration_targets(facility: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get prioritized exploration targets for a facility.

        Args:
            facility: Facility ID
            limit: Maximum targets to return

        Returns:
            List of targets with priority, type, action, and effort
        """
        targets: list[dict[str, Any]] = []

        try:
            # Get MDSplus tree coverage
            trees = gc.query(
                """
                MATCH (t:MDSplusTree)-[:FACILITY_ID]->(f:Facility {id: $fid})
                OPTIONAL MATCH (n:TreeNode {tree_name: t.name})-[:FACILITY_ID]->(f)
                RETURN t.name AS tree,
                       t.node_count_total AS total,
                       count(DISTINCT n) AS ingested
                ORDER BY t.name
                """,
                fid=facility,
            )

            for row in trees:
                total = row["total"] or 0
                ingested = row["ingested"] or 0
                if total > 0:
                    pct = round(100 * ingested / total, 1)
                    if pct == 0:
                        targets.append(
                            {
                                "priority": 1,
                                "type": "mdsplus_tree",
                                "target": row["tree"],
                                "action": f"Ingest {row['tree']} tree ({total} nodes)",
                                "effort": "high" if total > 1000 else "medium",
                            }
                        )
                    elif pct < 10:
                        targets.append(
                            {
                                "priority": 2,
                                "type": "mdsplus_tree",
                                "target": row["tree"],
                                "action": f"Continue {row['tree']} ({pct}% complete)",
                                "effort": "medium",
                            }
                        )

            # Get discovered paths
            paths = gc.query(
                """
                MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $fid})
                WHERE p.status = 'discovered'
                RETURN p.path AS path, p.interest_score AS score
                ORDER BY COALESCE(p.interest_score, 0) DESC
                LIMIT 5
                """,
                fid=facility,
            )

            for row in paths:
                targets.append(
                    {
                        "priority": 3,
                        "type": "facility_path",
                        "target": row["path"],
                        "action": f"Explore {row['path']}",
                        "score": row["score"],
                        "effort": "medium",
                    }
                )

        except Exception as e:
            targets.append({"error": str(e)})

        targets.sort(key=lambda x: x.get("priority", 99))
        return targets[:limit]

    def get_tree_structure(
        tree_name: str, path_prefix: str = "", limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get TreeNode structure from the graph.

        Args:
            tree_name: MDSplus tree name (e.g., 'results', 'magnetics')
            path_prefix: Optional path prefix to filter
            limit: Maximum nodes to return

        Returns:
            List of {path, description, units, physics_domain}
        """
        if path_prefix:
            return gc.query(
                """
                MATCH (n:TreeNode {tree_name: $tree})
                WHERE n.path STARTS WITH $prefix
                RETURN n.path AS path, n.description AS description,
                       n.units AS units, n.physics_domain AS domain
                ORDER BY n.path
                LIMIT $limit
                """,
                tree=tree_name,
                prefix=path_prefix,
                limit=limit,
            )
        else:
            return gc.query(
                """
                MATCH (n:TreeNode {tree_name: $tree})
                RETURN n.path AS path, n.description AS description,
                       n.units AS units, n.physics_domain AS domain
                ORDER BY n.path
                LIMIT $limit
                """,
                tree=tree_name,
                limit=limit,
            )

    # =========================================================================
    # Code search utilities
    # =========================================================================

    _code_searcher = CodeExampleSearch()

    def search_code(
        query_text: str,
        top_k: int = 5,
        facility: str | None = None,
        min_score: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Search code examples using semantic similarity.

        Args:
            query_text: Natural language query
            top_k: Maximum results to return
            facility: Optional facility filter
            min_score: Minimum similarity score

        Returns:
            List of code results with content, source_file, score
        """
        results = _code_searcher.search(
            query=query_text,
            top_k=top_k,
            facility=facility,
            min_score=min_score,
        )
        return [
            {
                "content": r.content[:500] + "..."
                if len(r.content) > 500
                else r.content,
                "function_name": r.function_name,
                "source_file": r.source_file,
                "facility_id": r.facility_id,
                "score": round(r.score, 3),
            }
            for r in results
        ]

    # =========================================================================
    # IMAS DD utilities
    # =========================================================================

    def search_imas(
        query_text: str,
        ids_filter: str | None = None,
        max_results: int = 10,
    ) -> str:
        """Search IMAS Data Dictionary using semantic search.

        Args:
            query_text: Natural language query
            ids_filter: Optional IDS name filter (space-delimited)
            max_results: Maximum results

        Returns:
            Formatted string with matching paths and documentation
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.search_imas_paths(
                    query=query_text,
                    ids_filter=ids_filter,
                    max_results=max_results,
                )
            )
            if not result.hits:
                return f"No IMAS paths found for '{query_text}'"
            output = []
            for hit in result.hits:
                line = f"{hit.path} (score: {hit.score:.2f})"
                output.append(line)
                if hit.documentation:
                    output.append(f"  {hit.documentation[:150]}...")
                if hit.units:
                    output.append(f"  Units: {hit.units}")
            return "\n".join(output)
        except Exception as e:
            return f"IMAS search error: {e}"

    def fetch_imas(paths: str) -> str:
        """Get full documentation for IMAS paths.

        Args:
            paths: Space-delimited IMAS paths

        Returns:
            Detailed path documentation
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(tools.fetch_imas_paths(paths=paths))
            return str(result)
        except Exception as e:
            return f"Fetch error: {e}"

    def list_imas(paths: str, leaf_only: bool = True, max_paths: int = 100) -> str:
        """List data paths in IDS.

        Args:
            paths: Space-separated IDS names or path prefixes
            leaf_only: Only return data fields
            max_paths: Limit output size

        Returns:
            Tree structure in YAML format
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.list_imas_paths(
                    paths=paths,
                    leaf_only=leaf_only,
                    max_paths=max_paths,
                )
            )
            return str(result)
        except Exception as e:
            return f"List error: {e}"

    def check_imas(paths: str) -> str:
        """Validate IMAS paths for existence.

        Args:
            paths: Space-delimited IMAS paths

        Returns:
            Validation results with existence status
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(tools.check_imas_paths(paths=paths))
            return str(result)
        except Exception as e:
            return f"Check error: {e}"

    def get_imas_overview(query_text: str | None = None) -> str:
        """Get high-level overview of IMAS Data Dictionary.

        Args:
            query_text: Optional keyword filter

        Returns:
            Overview with IDS list, physics domains, statistics
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(tools.get_imas_overview(query=query_text))
            return str(result)
        except Exception as e:
            return f"Overview error: {e}"

    # =========================================================================
    # COCOS utilities
    # =========================================================================

    def validate_cocos(
        declared_cocos: int,
        psi_axis: float,
        psi_edge: float,
        ip: float,
        b0: float,
        q: float | None = None,
        dp_dpsi: float | None = None,
    ) -> dict[str, Any]:
        """Validate declared COCOS against physics data.

        Uses Eq. 23 from Sauter & Medvedev paper to check consistency
        between declared COCOS and equilibrium physics quantities.

        Args:
            declared_cocos: COCOS value to validate (1-8 or 11-18)
            psi_axis: Poloidal flux at magnetic axis [Wb]
            psi_edge: Poloidal flux at plasma edge [Wb]
            ip: Plasma current [A] (sign matters)
            b0: Toroidal field at axis [T] (sign matters)
            q: Safety factor at mid-radius (optional)
            dp_dpsi: Pressure gradient dp/dψ (optional)

        Returns:
            Dict with is_consistent, calculated_cocos, confidence, inconsistencies

        Example:
            validate_cocos(17, psi_axis=0.5, psi_edge=-0.2, ip=-1e6, b0=-5.0, q=3.0)
        """
        from imas_codex.cocos import validate_cocos_from_data

        result = validate_cocos_from_data(
            declared_cocos=declared_cocos,
            psi_axis=psi_axis,
            psi_edge=psi_edge,
            ip=ip,
            b0=b0,
            q=q,
            dp_dpsi=dp_dpsi,
        )
        return {
            "is_consistent": result.is_consistent,
            "declared_cocos": result.declared_cocos,
            "calculated_cocos": result.calculated_cocos,
            "confidence": round(result.confidence, 2),
            "inconsistencies": result.inconsistencies,
        }

    def determine_cocos(
        psi_axis: float,
        psi_edge: float,
        ip: float,
        b0: float,
        q: float | None = None,
        dp_dpsi: float | None = None,
    ) -> dict[str, Any]:
        """Determine COCOS from equilibrium physics quantities.

        Uses Eq. 23 from Sauter & Medvedev paper to infer COCOS.

        Args:
            psi_axis: Poloidal flux at magnetic axis [Wb]
            psi_edge: Poloidal flux at plasma edge [Wb]
            ip: Plasma current [A] (sign matters)
            b0: Toroidal field at axis [T] (sign matters)
            q: Safety factor at mid-radius (optional)
            dp_dpsi: Pressure gradient dp/dψ (optional)

        Returns:
            Dict with cocos value and confidence

        Example:
            determine_cocos(psi_axis=0.5, psi_edge=-0.2, ip=-1e6, b0=-5.0)
        """
        from imas_codex.cocos import determine_cocos as _determine_cocos

        cocos, confidence = _determine_cocos(
            psi_axis=psi_axis,
            psi_edge=psi_edge,
            ip=ip,
            b0=b0,
            q=q,
            dp_dpsi=dp_dpsi,
        )
        return {"cocos": cocos, "confidence": round(confidence, 2)}

    def cocos_sign_flip_paths(ids_name: str | None = None) -> dict[str, Any]:
        """Get paths requiring COCOS sign flip between DD3/DD4.

        Args:
            ids_name: IDS name (e.g., 'equilibrium'). If None, lists all IDS.

        Returns:
            Dict with IDS name(s) and their sign-flip paths

        Example:
            cocos_sign_flip_paths('equilibrium')
            cocos_sign_flip_paths()  # List all IDS with sign flips
        """
        from imas_codex.cocos import get_sign_flip_paths, list_ids_with_sign_flips

        if ids_name:
            paths = get_sign_flip_paths(ids_name)
            return {"ids": ids_name, "paths": paths, "count": len(paths)}
        else:
            ids_list = list_ids_with_sign_flips()
            return {
                "ids_with_sign_flips": [
                    {"ids": ids, "count": len(get_sign_flip_paths(ids))}
                    for ids in ids_list
                ],
                "total_ids": len(ids_list),
            }

    def cocos_info(cocos_value: int) -> dict[str, Any]:
        """Get COCOS parameters for a given value.

        Args:
            cocos_value: COCOS index (1-8 or 11-18)

        Returns:
            Dict with the four COCOS parameters from Sauter Table I

        Example:
            cocos_info(17)  # IMAS DD4 / TCV convention
        """
        from imas_codex.cocos import KNOWN_CODE_COCOS, VALID_COCOS, cocos_to_parameters

        if cocos_value not in VALID_COCOS:
            return {"error": f"Invalid COCOS {cocos_value}. Valid: 1-8, 11-18"}

        params = cocos_to_parameters(cocos_value)
        codes = [code for code, val in KNOWN_CODE_COCOS.items() if val == cocos_value]
        return {
            "cocos": cocos_value,
            "sigma_bp": params.sigma_bp,
            "e_bp": params.e_bp,
            "sigma_r_phi_z": params.sigma_r_phi_z,
            "sigma_rho_theta_phi": params.sigma_rho_theta_phi,
            "used_by": codes,
        }

    # =========================================================================
    # Tool checking utilities
    # =========================================================================

    def check_tools(facility: str = "epfl") -> dict[str, Any]:
        """Check availability and versions of required tools on remote facility.

        Args:
            facility: SSH host alias (default: 'epfl')

        Returns:
            Dict with tool availability, versions, and overall status

        Example:
            check_tools('epfl')
            # Returns: {'fd': {'available': True, 'version': '10.2.0', ...}, ...}
        """
        from pathlib import Path

        import yaml

        # Load tool requirements from config
        config_path = Path(__file__).parent.parent / "config" / "tool_requirements.yaml"
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            return {"error": f"Tool requirements not found: {config_path}"}

        results: dict[str, Any] = {"facility": facility, "tools": {}}

        # Check required tools
        all_ok = True
        for tool in config.get("required_tools", []):
            name = tool["name"]
            min_version = tool.get("min_version", "0.0.0")
            version_cmd = tool.get("version_command", f"{name} --version")

            # Check ~/bin first, then PATH
            try:
                output = ssh(
                    f"~/bin/{name} --version 2>/dev/null || {version_cmd}", facility
                )
                version = output.strip().split("\n")[0] if output else ""
                # Extract version number (handle various formats)
                import re

                version_match = re.search(r"(\d+\.\d+\.?\d*)", version)
                version_str = version_match.group(1) if version_match else version
                available = bool(version_str)
            except Exception:
                available = False
                version_str = ""

            results["tools"][name] = {
                "available": available,
                "version": version_str,
                "required": min_version,
                "ok": available,  # TODO: version comparison
                "purpose": tool.get("purpose", ""),
            }
            if not available:
                all_ok = False

        # Check optional tools
        for tool in config.get("optional_tools", []):
            name = tool["name"]
            version_cmd = tool.get("version_command", f"{name} --version")
            try:
                output = ssh(
                    f"~/bin/{name} --version 2>/dev/null || {version_cmd}", facility
                )
                version = output.strip().split("\n")[0] if output else ""
                import re

                version_match = re.search(r"(\d+\.\d+\.?\d*)", version)
                version_str = version_match.group(1) if version_match else version
                available = bool(version_str)
            except Exception:
                available = False
                version_str = ""

            results["tools"][name] = {
                "available": available,
                "version": version_str,
                "optional": True,
                "purpose": tool.get("purpose", ""),
            }

        results["all_required_ok"] = all_ok
        return results

    # =========================================================================
    # Build REPL globals
    # =========================================================================

    _repl_globals = {
        # Core utilities
        "gc": gc,
        "embed_model": embed_model,
        "query": query,
        "ssh": ssh,
        "embed": embed,
        "semantic_search": semantic_search,
        # Facility utilities
        "get_facility": get_facility,
        "get_exploration_targets": get_exploration_targets,
        "get_tree_structure": get_tree_structure,
        "check_tools": check_tools,
        # Code search
        "search_code": search_code,
        # IMAS DD utilities
        "search_imas": search_imas,
        "fetch_imas": fetch_imas,
        "list_imas": list_imas,
        "check_imas": check_imas,
        "get_imas_overview": get_imas_overview,
        # COCOS utilities
        "validate_cocos": validate_cocos,
        "determine_cocos": determine_cocos,
        "cocos_sign_flip_paths": cocos_sign_flip_paths,
        "cocos_info": cocos_info,
        # Standard library
        "subprocess": subprocess,
        # Result storage
        "_": None,
    }

    logger.info(
        "Python REPL initialized with graph, IMAS, COCOS, and facility utilities"
    )
    return _repl_globals


def _get_repl() -> dict[str, Any]:
    """Get the persistent REPL environment (already initialized at startup)."""
    global _repl_globals
    if _repl_globals is None:
        return _init_repl()
    return _repl_globals


# =============================================================================
# MCP Server with 4 Core Tools
# =============================================================================


@dataclass
class AgentsServer:
    """
    Streamlined MCP server with 4 core tools for facility exploration.

    Tools:
    - python: Persistent REPL with rich utilities (primary interface)
    - get_graph_schema: Schema introspection for query generation
    - ingest_nodes: Schema-validated node creation with privacy filtering
    - private: Read/update sensitive infrastructure files

    The python() REPL provides access to:
    - Graph: query(), semantic_search(), embed()
    - Remote: ssh(), check_tools()
    - Facility: get_facility(), get_exploration_targets(), get_tree_structure()
    - IMAS DD: search_imas(), fetch_imas(), list_imas(), check_imas()
    - COCOS: validate_cocos(), determine_cocos(), cocos_sign_flip_paths(), cocos_info()
    - Code: search_code()
    """

    mcp: FastMCP = field(init=False, repr=False)
    _prompts: dict[str, PromptDefinition] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the MCP server and eagerly load REPL."""
        self.mcp = FastMCP(name="imas-codex-agents")
        self._prompts = load_prompts()

        # Eagerly initialize REPL on server startup
        logger.info("Eagerly loading REPL environment...")
        _init_repl()

        self._register_tools()
        self._register_prompts()

        logger.info(
            f"Agents MCP server initialized with 4 core tools and {len(self._prompts)} prompts"
        )

    def _register_tools(self):
        """Register the 4 core tools."""

        # =====================================================================
        # Tool 1: python - Persistent REPL (primary interface)
        # =====================================================================

        @self.mcp.tool()
        def python(code: str) -> str:
            """
            Execute Python code in a persistent REPL with rich pre-loaded utilities.

            The REPL maintains state between calls - variables persist across invocations.
            All utilities are loaded at server startup for instant response.

            Pre-loaded utilities:

            GRAPH:
            - query(cypher, **params): Execute Cypher, return list of dicts
            - semantic_search(text, index, k): Vector similarity search
            - embed(text): Get 384-dim embedding vector

            REMOTE:
            - ssh(cmd, facility="epfl", timeout=60): Run SSH command
            - check_tools(facility): Check tool availability and versions

            FACILITY:
            - get_facility(facility): Comprehensive facility info + graph state
            - get_exploration_targets(facility, limit): Prioritized work items
            - get_tree_structure(tree, prefix, limit): TreeNode hierarchy

            CODE SEARCH:
            - search_code(query, top_k, facility, min_score): Semantic code search

            IMAS DATA DICTIONARY:
            - search_imas(query, ids_filter, max_results): Semantic DD search
            - fetch_imas(paths): Full documentation for paths
            - list_imas(paths, leaf_only, max_paths): List IDS structure
            - check_imas(paths): Validate path existence
            - get_imas_overview(query): High-level DD summary

            Vector indexes for semantic_search:
            - imas_path_embedding: IMAS Data Dictionary paths (61k)
            - code_chunk_embedding: Code examples (8.5k chunks)
            - wiki_chunk_embedding: Wiki documentation (25k chunks)
            - cluster_centroid: Semantic clusters

            Args:
                code: Python code to execute (multi-line supported)

            Returns:
                stdout output, or repr of last expression if no print

            Examples:
                # Graph query
                python("paths = query('MATCH (t:TreeNode) RETURN t.path LIMIT 5')")

                # Semantic search
                python("hits = semantic_search('plasma current', 'code_chunk_embedding', 3)")

                # SSH command
                python("print(ssh('ls /home/codes | head -5'))")

                # IMAS search
                python("print(search_imas('electron temperature profile'))")

                # Facility info
                python("info = get_facility('epfl'); print(info['graph_summary'])")

                # Variables persist
                python("x = 42")
                python("print(x * 2)")  # prints 84
            """
            repl = _get_repl()

            stdout_capture = io.StringIO()
            old_stdout = sys.stdout

            try:
                sys.stdout = stdout_capture

                try:
                    result = eval(code, repl)
                    if result is not None:
                        repl["_"] = result
                        print(repr(result))
                except SyntaxError:
                    exec(code, repl)

                output = stdout_capture.getvalue()
                return output if output else "(no output)"

            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                return f"Error: {e}\n\n{tb}"

            finally:
                sys.stdout = old_stdout

        # =====================================================================
        # Tool 2: get_graph_schema - Schema introspection
        # =====================================================================

        @self.mcp.tool()
        def get_graph_schema() -> dict[str, Any]:
            """
            Get complete graph schema for Cypher query generation.

            Returns node labels with all properties, enums with valid values,
            relationship types, and private field annotations.
            Call this before writing Cypher queries in the python() REPL.

            Returns:
                Schema dict with node_labels, enums, relationship_types, notes
            """
            schema = get_schema()

            node_labels = {}
            for label in schema.node_labels:
                node_labels[label] = {
                    "identifier": schema.get_identifier(label),
                    "description": schema.get_class_description(label),
                    "properties": schema.get_all_slots(label),
                    "private_fields": schema.get_private_slots(label),
                }

            return {
                "node_labels": node_labels,
                "enums": schema.get_enums(),
                "relationship_types": schema.relationship_types,
                "notes": {
                    "private_fields": "Fields with is_private:true are never stored in graph",
                    "mutations": "Use ingest_nodes() tool for writes, or query() for reads in python REPL",
                },
            }

        # =====================================================================
        # Tool 3: ingest_nodes - Schema-validated writes
        # =====================================================================

        @self.mcp.tool()
        def ingest_nodes(
            node_type: str,
            data: dict[str, Any] | list[dict[str, Any]],
            create_facility_relationship: bool = True,
            batch_size: int = 50,
        ) -> dict[str, Any]:
            """
            Ingest nodes with schema validation and privacy filtering.

            Validates data against Pydantic models, filters out private fields,
            then writes to the graph. Use this instead of raw Cypher mutations.

            Special handling:
            - SourceFile: Auto-deduplicates already discovered/ingested files
            - TreeNode: Auto-creates TREE_NAME and ACCESSOR_FUNCTION relationships
            - FacilityPath: Links to parent Facility

            Args:
                node_type: Node label (use get_graph_schema() to see valid types)
                data: List of property dicts matching the schema
                create_facility_relationship: Auto-create FACILITY_ID relationship
                batch_size: Nodes per batch (default: 50)

            Returns:
                Dict with counts: {"processed": N, "skipped": K, "errors": [...]}

            Examples:
                # Add FacilityPaths
                ingest_nodes("FacilityPath", [
                    {"id": "epfl:/home/codes", "path": "/home/codes",
                     "facility_id": "epfl", "path_type": "code_directory",
                     "status": "discovered", "interest_score": 0.8}
                ])

                # Queue source files
                ingest_nodes("SourceFile", [
                    {"id": "epfl:/home/codes/liuqe.py", "path": "/home/codes/liuqe.py",
                     "facility_id": "epfl", "status": "discovered"}
                ])
            """
            schema = get_schema()

            if node_type not in schema.node_labels:
                msg = f"Unknown node type: {node_type}. Valid: {schema.node_labels}"
                raise ValueError(msg)

            items = [data] if isinstance(data, dict) else data
            if not items:
                return {"processed": 0, "skipped": 0, "errors": []}

            private_slots = set(schema.get_private_slots(node_type))
            model_class = schema.get_model(node_type)
            id_field = schema.get_identifier(node_type)
            if not id_field:
                raise ValueError(f"No identifier field found for {node_type}")

            valid_items: list[dict[str, Any]] = []
            errors: list[str] = []

            for i, item in enumerate(items):
                try:
                    filtered = {k: v for k, v in item.items() if k not in private_slots}
                    validated = model_class.model_validate(filtered)
                    props = to_cypher_props(validated)
                    valid_items.append(props)
                except Exception as e:
                    item_id = item.get(id_field, f"item[{i}]")
                    errors.append(f"{item_id}: {e}")
                    logger.warning(f"Validation failed for {node_type} {item_id}: {e}")

            if not valid_items:
                return {"processed": 0, "skipped": len(errors), "errors": errors}

            try:
                with GraphClient() as client:
                    skipped_dedup = 0

                    # SourceFile deduplication
                    if node_type == "SourceFile":
                        existing = client.query(
                            """
                            UNWIND $items AS item
                            OPTIONAL MATCH (sf:SourceFile {id: item.id})
                            OPTIONAL MATCH (ce:CodeExample {source_file: item.path, facility_id: item.facility_id})
                            RETURN item.id AS id, sf.status AS sf_status, ce.id AS ce_id
                            """,
                            items=valid_items,
                        )
                        skip_ids = {
                            row["id"]
                            for row in existing
                            if row["sf_status"] in ("discovered", "ingested")
                            or row["ce_id"]
                        }
                        if skip_ids:
                            valid_items = [
                                i for i in valid_items if i["id"] not in skip_ids
                            ]
                            skipped_dedup = len(skip_ids)
                        if not valid_items:
                            return {
                                "processed": 0,
                                "skipped": len(errors) + skipped_dedup,
                                "errors": errors,
                            }

                    facility_field = (
                        "facility_id" if create_facility_relationship else None
                    )
                    result = client.create_nodes(
                        label=node_type,
                        items=valid_items,
                        id_field=id_field,
                        batch_size=batch_size,
                        facility_id_field=facility_field,
                    )

                    # TreeNode relationship creation
                    if node_type == "TreeNode":
                        tree_names = {
                            item["tree_name"]
                            for item in valid_items
                            if item.get("tree_name")
                        }
                        if tree_names:
                            client.query(
                                """
                                UNWIND $names AS tn
                                MATCH (n:TreeNode {tree_name: tn})
                                MATCH (t:MDSplusTree {name: tn})
                                MERGE (n)-[:TREE_NAME]->(t)
                                """,
                                names=list(tree_names),
                            )

                    return {
                        "processed": result["processed"],
                        "skipped": len(errors) + skipped_dedup,
                        "errors": errors,
                    }

            except Exception as e:
                logger.exception("Failed to ingest nodes")
                raise RuntimeError(
                    f"Failed to ingest nodes: {_neo4j_error_message(e)}"
                ) from e

        # =====================================================================
        # Tool 4: private - Sensitive infrastructure files
        # =====================================================================

        @self.mcp.tool()
        def private(
            facility: str,
            data: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """
            Read or update private facility data (sensitive infrastructure info).

            Private files contain data marked is_private in the schema:
            OS versions, paths, tool availability, etc.
            This data is NEVER stored in the graph or OCI artifacts.

            Args:
                facility: Facility identifier (e.g., "epfl")
                data: If provided, deep-merge into private file and return result.
                      If None, just read and return current data.

            Returns:
                Current private data dict (after merge if data provided)

            Examples:
                # Read private data
                private("epfl")

                # Update tool availability
                private("epfl", {"tools": {"rg": "14.1.1"}})

                # Add exploration notes
                private("epfl", {"exploration_notes": ["Found new tree"]})
            """
            try:
                if data is not None:
                    save_private(facility, data)
                return get_facility_private(facility) or {}
            except Exception as e:
                logger.exception(f"Failed to access private data for {facility}")
                raise RuntimeError(f"Failed to access private data: {e}") from e

    def _register_prompts(self):
        """Register MCP prompts from markdown files."""
        for name, prompt_def in self._prompts.items():

            def make_prompt_fn(pd: PromptDefinition):
                def prompt_fn() -> str:
                    return pd.content

                prompt_fn.__name__ = pd.name.replace("-", "_")
                return prompt_fn

            self.mcp.prompt(name=name, description=prompt_def.description)(
                make_prompt_fn(prompt_def)
            )
            logger.debug(f"Registered prompt: {name}")

    def run(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
        host: str = "127.0.0.1",
        port: int = 8001,
    ):
        """Run the agents server."""
        if transport == "stdio":
            logger.debug("Starting Agents server with stdio transport")
            self.mcp.run(transport=transport)
        else:
            logger.info(f"Starting Agents server on {host}:{port}")
            self.mcp.run(transport=transport, host=host, port=port)
