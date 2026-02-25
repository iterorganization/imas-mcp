"""
Agents MCP Server - Streamlined tools for LLM-driven facility exploration.

This server provides 9 MCP tools organized by purpose:

Graph Operations:
- get_graph_schema: Schema introspection for query generation
- add_to_graph: Schema-validated node creation with privacy filtering

Facility Infrastructure (Private Data):
- update_facility_infrastructure: Deep-merge update to private YAML
- get_facility_infrastructure: Read private infrastructure data
- add_exploration_note: Append timestamped exploration note
- update_facility_paths: Update path mappings
- update_facility_tools: Update tool availability

Legacy/General:
- update_facility_config: Read/update facility config (public or private)
- python: Persistent Python REPL with rich pre-loaded utilities

The python() REPL provides advanced operations:
- Graph: query(), semantic_search(), embed()
- Remote: run(), check_tools() (auto-detects local vs SSH)
- Facility: get_facility(), get_exploration_targets(), get_tree_structure()
- IMAS DD: search_imas(), fetch_imas(), list_imas(), check_imas()
- COCOS: validate_cocos(), determine_cocos(), cocos_sign_flip_paths(), cocos_info()
- Code: search_code()

Use python() for:
- Complex multi-step operations requiring state
- Graph queries with Cypher
- Chained processing with intermediate logic
- IMAS/COCOS domain-specific operations

Use dedicated MCP tools for:
- Single-purpose infrastructure updates
- Clear, type-safe operations
- Better discoverability and documentation

REPL state is loaded eagerly on server startup for instant tool response.
"""

import asyncio
import io
import logging
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

from fastmcp import FastMCP
from neo4j.exceptions import ServiceUnavailable
from ruamel.yaml import YAML

from imas_codex.agentic.prompt_loader import (
    PromptDefinition,
    load_prompts,
)
from imas_codex.discovery import (
    get_facility as _get_facility_config,
    get_facility_infrastructure,
    get_facility_validated,
    update_infrastructure,
    update_metadata,
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
    "Neo4j is not running. Check service with: systemctl --user status imas-codex-neo4j"
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
    msg = str(e)
    if "Connection refused" in msg or "ServiceUnavailable" in msg:
        return NEO4J_NOT_RUNNING_MSG
    if "critical error" in msg.lower() or "needs to be restarted" in msg.lower():
        return (
            "Neo4j database has entered a critical error state and needs to be "
            "restarted. Run: imas-codex graph stop && imas-codex graph start"
        )
    return msg


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

    Called once at server startup. Uses lazy embedding via Encoder class
    which respects the embedding-backend config (local/remote).
    """
    global _repl_globals
    if _repl_globals is not None:
        return _repl_globals

    logger.info("Initializing Python REPL...")

    from imas_codex.embeddings.config import EncoderConfig
    from imas_codex.embeddings.encoder import EmbeddingBackendError, Encoder
    from imas_codex.graph import GraphClient
    from imas_codex.ingestion.search import ChunkSearch
    from imas_codex.settings import get_embedding_location

    gc = GraphClient()

    # Create encoder with lazy initialization - respects embedding-backend config
    # This will NOT load the model until actually used
    backend = get_embedding_location()
    logger.info(f"Embedding location: {backend}")

    # Track embedding availability
    _embedding_error: Exception | None = None
    _encoder: Encoder | None = None

    def _get_encoder() -> Encoder:
        """Get or create the encoder, raising any deferred initialization error."""
        nonlocal _encoder, _embedding_error

        if _embedding_error is not None:
            raise _embedding_error

        if _encoder is None:
            try:
                config = EncoderConfig()
                _encoder = Encoder(config)
                logger.info(f"Encoder initialized (backend={config.backend})")
            except Exception as e:
                _embedding_error = e
                logger.error(f"Embedding initialization failed: {e}")
                raise EmbeddingBackendError(
                    f"Embedding backend '{backend}' unavailable: {e}. "
                    f"Check configuration or use a different backend."
                ) from e

        return _encoder

    # =========================================================================
    # Core utilities
    # =========================================================================

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
            Embedding vector (dimension depends on configured model)

        Raises:
            EmbeddingBackendError: If embedding backend is unavailable
        """
        encoder = _get_encoder()
        embeddings = encoder.embed_texts([text])
        return embeddings[0].tolist()

    def semantic_search(
        text: str,
        index: str = "imas_path_embedding",
        k: int = 5,
        include_deprecated: bool = False,
    ) -> list[dict[str, Any]]:
        """Vector similarity search on graph embeddings.

        Args:
            text: Query text to embed and search
            index: Vector index name (use get_graph_schema() to list all)
            k: Number of results to return
            include_deprecated: If True, include deprecated IMAS paths in results.
                Only applies to imas_path_embedding index. Default False (active only).

        Returns:
            List of flat dicts with all node properties + score + labels.
            For wiki_chunk_embedding: also includes page_title, page_url.
            For code_chunk_embedding: also includes source_file.

        Raises:
            EmbeddingBackendError: If embedding backend is unavailable
        """
        encoder = _get_encoder()
        embeddings = encoder.embed_texts([text])
        embedding = embeddings[0].tolist()

        # Use index-specific queries for richer context
        if index == "wiki_chunk_embedding":
            return _search_wiki_chunks(embedding, k)
        if index == "code_chunk_embedding":
            return _search_code_chunks(embedding, k)

        # Filter deprecated paths for imas_path_embedding unless explicitly included
        where_clause = ""
        if index == "imas_path_embedding" and not include_deprecated:
            where_clause = "WHERE NOT (node)-[:DEPRECATED_IN]->(:DDVersion) "

        results = gc.query(
            f'CALL db.index.vector.queryNodes("{index}", $k, $embedding) '
            "YIELD node, score "
            f"{where_clause}"
            "RETURN [k IN keys(node) "
            "WHERE NOT k ENDS WITH 'embedding' | [k, node[k]]] "
            "AS properties, labels(node) AS labels, score "
            "ORDER BY score DESC",
            k=k,
            embedding=embedding,
        )
        # Flatten: properties at top level alongside score and labels
        return [
            {
                **dict(r["properties"]),
                "labels": r["labels"],
                "score": r["score"],
            }
            for r in results
        ]

    def _search_wiki_chunks(embedding: list[float], k: int) -> list[dict[str, Any]]:
        """Wiki-specific search that enriches results with parent page context."""
        results = gc.query(
            'CALL db.index.vector.queryNodes("wiki_chunk_embedding", $k, $embedding) '
            "YIELD node, score "
            "OPTIONAL MATCH (p:WikiPage)-[:HAS_CHUNK]->(node) "
            "OPTIONAL MATCH (wa:WikiArtifact)-[:HAS_CHUNK]->(node) "
            "RETURN [k IN keys(node) "
            "WHERE NOT k ENDS WITH 'embedding' | [k, node[k]]] "
            "AS properties, labels(node) AS labels, score, "
            "p.title AS page_title, p.url AS page_url, "
            "wa.id AS artifact_id, wa.title AS artifact_title, wa.url AS artifact_url "
            "ORDER BY score DESC",
            k=k,
            embedding=embedding,
        )
        out = []
        for r in results:
            d: dict[str, Any] = {
                **dict(r["properties"]),
                "labels": r["labels"],
                "score": r["score"],
            }
            # Add parent page context (WikiPage or WikiArtifact)
            if r.get("page_title"):
                d["page_title"] = r["page_title"]
                d["page_url"] = r["page_url"]
            elif r.get("artifact_id"):
                d["page_title"] = r["artifact_title"] or r["artifact_id"]
                if r.get("artifact_url"):
                    d["page_url"] = r["artifact_url"]
            out.append(d)
        return out

    def _search_code_chunks(embedding: list[float], k: int) -> list[dict[str, Any]]:
        """Code-specific search that enriches results with source file context."""
        results = gc.query(
            'CALL db.index.vector.queryNodes("code_chunk_embedding", $k, $embedding) '
            "YIELD node, score "
            "OPTIONAL MATCH (sf:SourceFile)-[:HAS_CHUNK]->(node) "
            "RETURN [k IN keys(node) "
            "WHERE NOT k ENDS WITH 'embedding' | [k, node[k]]] "
            "AS properties, labels(node) AS labels, score, "
            "sf.path AS source_file, sf.facility_id AS source_facility "
            "ORDER BY score DESC",
            k=k,
            embedding=embedding,
        )
        out = []
        for r in results:
            d: dict[str, Any] = {
                **dict(r["properties"]),
                "labels": r["labels"],
                "score": r["score"],
            }
            if r.get("source_file"):
                d["source_file"] = r["source_file"]
            if r.get("source_facility"):
                d["source_facility"] = r["source_facility"]
            out.append(d)
        return out

    # =========================================================================
    # Facility utilities
    # =========================================================================

    def get_facility(facility: str) -> dict[str, Any]:
        """Get comprehensive facility info including graph state.

        Loads the full facility config validated against the LinkML schema
        (FacilityConfig model) so all typed fields (data_sources, data_systems,
        data_access_patterns, wiki_sites, etc.) are included.

        Args:
            facility: Facility ID (e.g., 'tcv')

        Returns:
            Dict with config (full LinkML-validated), graph_summary,
            actionable_paths
        """
        result: dict[str, Any] = {"facility": facility}

        # Load facility config via LinkML-validated Pydantic model
        try:
            validated = get_facility_validated(facility)
            result["config"] = validated.model_dump(exclude_none=True)
        except Exception as e:
            # Fall back to raw dict if validation fails
            try:
                data = _get_facility_config(facility)
                result["config"] = data
                result["validation_error"] = str(e)
            except Exception as e2:
                result["error"] = str(e2)
                return result

        # Query graph for facility summary
        try:
            summary = gc.query(
                """
                MATCH (f:Facility {id: $fid})
                OPTIONAL MATCH (a:AnalysisCode)-[:AT_FACILITY]->(f)
                OPTIONAL MATCH (d:Diagnostic)-[:AT_FACILITY]->(f)
                OPTIONAL MATCH (t:TDIFunction)-[:AT_FACILITY]->(f)
                OPTIONAL MATCH (m:MDSplusTree)-[:AT_FACILITY]->(f)
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
                MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $fid})
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
                MATCH (t:MDSplusTree)-[:AT_FACILITY]->(f:Facility {id: $fid})
                OPTIONAL MATCH (n:TreeNode {tree_name: t.name})-[:AT_FACILITY]->(f)
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
                MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $fid})
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

    _code_searcher: ChunkSearch | None = None

    def _get_code_searcher() -> ChunkSearch:
        """Get or create ChunkSearch, loading embedding model on first use."""
        nonlocal _code_searcher
        if _code_searcher is None:
            logger.info("Initializing ChunkSearch (embedding loading)...")
            _code_searcher = ChunkSearch()
            logger.info("ChunkSearch ready")
        return _code_searcher

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
        results = _get_code_searcher().search(
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
    # Tool management utilities (from remote.tools)
    # =========================================================================

    # Import the new unified tool functions
    from imas_codex.agentic.tool_installer import quick_setup, setup_tools
    from imas_codex.remote.tools import (
        check_all_tools as _check_all_tools,
        install_all_tools as _install_all_tools,
        run as _run,
    )

    def run(cmd: str, facility: str | None = None, timeout: int = 60) -> str:
        """Execute command locally or via SSH depending on facility.

        This is the unified execution interface. If facility is None or
        the facility has local=True (e.g., 'iter'), runs locally.
        Otherwise uses SSH.

        Args:
            cmd: Shell command to execute
            facility: Facility ID (None = local, 'iter' = local, 'tcv' = SSH)
            timeout: Command timeout in seconds

        Returns:
            Command output (stdout + stderr)

        Examples:
            run('rg pattern', facility='iter')  # Local (ITER is local)
            run('rg pattern', facility='tcv')  # SSH to EPFL
            run('rg pattern')                   # Local (no facility)
        """
        return _run(cmd, facility=facility, timeout=timeout)

    def check_tools(facility: str | None = None) -> dict[str, Any]:
        """Check availability of all fast CLI tools.

        Args:
            facility: Facility ID (None = local)

        Returns:
            Dict with tool statuses and summary

        Example:
            check_tools('tcv')
            check_tools('iter')  # Local check
            check_tools()        # Local check
        """
        return _check_all_tools(facility=facility)

    def install_tools(
        facility: str | None = None,
        required_only: bool = False,
    ) -> dict[str, Any]:
        """Install all fast CLI tools on target system.

        Args:
            facility: Facility ID (None = local)
            required_only: Only install required tools (rg, fd)

        Returns:
            Dict with installation results

        Example:
            install_tools('tcv')           # Install all on EPFL
            install_tools('iter')           # Install all locally
            install_tools(required_only=True)  # Just rg and fd
        """
        return _install_all_tools(facility=facility, required_only=required_only)

    # =========================================================================
    # Build REPL globals
    # =========================================================================

    _repl_globals = {
        # Core utilities
        "gc": gc,
        "query": query,
        "embed": embed,
        "semantic_search": semantic_search,
        # Embedding (lazy - only initialized when used)
        "EmbeddingBackendError": EmbeddingBackendError,
        # Facility utilities
        "get_facility": get_facility,
        "get_facility_infrastructure": get_facility_infrastructure,
        "get_exploration_targets": get_exploration_targets,
        "get_tree_structure": get_tree_structure,
        # Facility configuration
        "update_infrastructure": update_infrastructure,
        "update_metadata": update_metadata,
        # Tool management (unified local/remote)
        "run": run,
        "check_tools": check_tools,
        "install_tools": install_tools,
        "setup_tools": setup_tools,
        "quick_setup": quick_setup,
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
        # Schema utilities
        "get_schema": get_schema,
        # REPL management
        "reload": _reload_repl,
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


def _reload_repl() -> str:
    """Reload the REPL environment after code changes.

    Clears cached modules and reinitializes all utilities.
    Use after editing imas_codex source files.

    Returns:
        Status message
    """
    global _repl_globals, _imas_tools_instance

    # Clear REPL state
    _repl_globals = None
    _imas_tools_instance = None

    # Invalidate imas_codex module cache
    modules_to_reload = [name for name in sys.modules if name.startswith("imas_codex")]
    for name in modules_to_reload:
        try:
            del sys.modules[name]
        except KeyError:
            pass

    logger.info(f"Cleared {len(modules_to_reload)} cached imas_codex modules")

    # Reinitialize
    _init_repl()

    return f"REPL reloaded. Cleared {len(modules_to_reload)} modules and reinitialized utilities."


# =============================================================================
# Background REPL initialization (non-blocking)
# =============================================================================

# Event signaling REPL initialization completion
_repl_ready = threading.Event()
_repl_init_error: Exception | None = None


def _init_repl_background():
    """Initialize REPL in background thread.

    Logs progress to stderr (visible in MCP output) and signals completion.
    """
    global _repl_init_error

    try:
        logger.info("Starting REPL initialization (embedding model + graph client)...")
        _init_repl()
        logger.info("REPL initialization complete - all tools ready")
    except Exception as e:
        logger.error(f"REPL initialization failed: {e}")
        _repl_init_error = e
    finally:
        _repl_ready.set()


def _wait_for_repl(timeout: float = 300.0) -> None:
    """Wait for REPL initialization to complete.

    Args:
        timeout: Maximum seconds to wait (default 5 minutes for embedding rebuild)

    Raises:
        RuntimeError: If initialization failed or timed out
    """
    if not _repl_ready.wait(timeout=timeout):
        raise RuntimeError(
            f"REPL initialization timed out after {timeout}s. "
            "Check MCP server logs for details."
        )

    if _repl_init_error is not None:
        raise RuntimeError(
            f"REPL initialization failed: {_repl_init_error}"
        ) from _repl_init_error


# =============================================================================
# MCP Server with 9 Core Tools
# =============================================================================


@dataclass
class AgentsServer:
    """
    MCP server with 9 core tools for facility exploration.

    Uses background initialization to eagerly load REPL without blocking
    the MCP handshake. The python() tool waits for initialization to complete.

    Tools:
    - python: Persistent REPL with rich utilities (primary interface)
    - get_graph_schema: Schema introspection for query generation
    - add_to_graph: Schema-validated node creation with privacy filtering
    - update_facility_config: Read/update facility config (public or private)
    - update_facility_infrastructure: Deep-merge update to private YAML
    - get_facility_infrastructure: Read private infrastructure data
    - add_exploration_note: Append timestamped exploration note
    - update_facility_paths: Update path mappings
    - update_facility_tools: Update tool availability

    The python() REPL provides access to:
    - Graph: query(), semantic_search(), embed()
    - Remote: run(), check_tools() (auto-detects local vs SSH)
    - Facility: get_facility(), get_exploration_targets(), get_tree_structure()
    - IMAS DD: search_imas(), fetch_imas(), list_imas(), check_imas()
    - COCOS: validate_cocos(), determine_cocos(), cocos_sign_flip_paths(), cocos_info()
    - Code: search_code()
    """

    mcp: FastMCP = field(init=False, repr=False)
    _prompts: dict[str, PromptDefinition] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the MCP server with background REPL loading.

        The background initialization pattern ensures:
        1. Server responds to MCP 'initialize' immediately (no timeout)
        2. REPL initialization runs in background with progress logging
        3. python() tool waits for REPL to be ready before executing
        """
        self.mcp = FastMCP(name="imas-codex-agents")
        self._prompts = load_prompts()

        # Start REPL initialization in background thread
        # This allows the server to respond to 'initialize' immediately
        init_thread = threading.Thread(
            target=_init_repl_background,
            name="repl-init",
            daemon=True,
        )
        init_thread.start()
        logger.info("REPL initialization started in background")

        self._register_tools()
        self._register_prompts()

        logger.info(
            f"Agents MCP server ready with 9 tools and {len(self._prompts)} prompts"
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

            === DISCOVERY ===
            List all functions: dir()
            Get help: help(function_name)
            Get signature: import inspect; inspect.signature(function_name)

            === GRAPH OPERATIONS ===
            query(cypher, **params) - Execute Cypher query, return list of dicts
            semantic_search(text, index, k, include_deprecated) - Vector similarity search
            embed(text) - Get 256-dim embedding vector

            === FACILITY CONFIGURATION ===
            get_facility(facility) - Load complete config (public + private merged)
            get_facility_infrastructure(facility) - Load private infrastructure only
            update_infrastructure(facility, data) - Update private config (tools, paths, notes)
            update_metadata(facility, data) - Update public config (name, description)
            get_exploration_targets(facility, limit) - Prioritized work items
            get_tree_structure(tree, prefix, limit) - TreeNode hierarchy

            === REMOTE EXECUTION ===
            run(cmd, facility, timeout) - Execute command (auto-detects local/SSH)
            check_tools(facility) - Check tool availability and versions

            === CODE SEARCH ===
            search_code(query, top_k, facility, min_score) - Semantic code search

            === IMAS DATA DICTIONARY ===
            search_imas(query, ids_filter, max_results) - Semantic DD search
            fetch_imas(paths) - Full documentation for paths
            list_imas(paths, leaf_only, max_paths) - List IDS structure
            check_imas(paths) - Validate path existence
            get_imas_overview(query) - High-level DD summary

            === COCOS ===
            validate_cocos(cocos) - Validate COCOS value
            determine_cocos(psi_axis, psi_edge, ip, b0) - Infer COCOS from data
            cocos_sign_flip_paths(ids_name) - Get sign-flip paths for DD3/DD4
            cocos_info(cocos_value) - Get COCOS parameters

            Vector indexes for semantic_search:
            Call get_graph_schema() to list all available indexes.

            Args:
                code: Python code to execute (multi-line supported)

            Returns:
                stdout output, or repr of last expression if no print

            Examples:
                # Discover what's available
                python("print([f for f in dir() if not f.startswith('_')])")

                # Check locality
                python("import socket; print(socket.gethostname())")

                # Graph query
                python("paths = query('MATCH (t:TreeNode) RETURN t.path LIMIT 5')")

                # Update infrastructure (private)
                python("update_infrastructure('iter', {'tools': {'rg': '14.1.1'}})")

                # Facility info
                python("info = get_facility('iter'); print(info['paths'])")

                # Variables persist
                python("x = 42")
                python("print(x * 2)")  # prints 84
            """
            # Wait for background initialization to complete (up to 5 minutes)
            _wait_for_repl()

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
            relationship types, vector indexes, and private field annotations.
            Call this before writing Cypher queries in the python() REPL.

            Returns:
                Schema dict with node_labels, enums, relationship_types,
                vector_indexes, notes
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

            # Derive vector indexes from all schemas (facility + DD)
            from imas_codex.graph.client import EXPECTED_VECTOR_INDEXES

            vector_indexes = {
                idx_name: {"label": label, "property": prop}
                for idx_name, label, prop in EXPECTED_VECTOR_INDEXES
            }

            return {
                "node_labels": node_labels,
                "enums": schema.get_enums(),
                "relationship_types": schema.relationship_types,
                "vector_indexes": vector_indexes,
                "notes": {
                    "private_fields": "Fields with is_private:true are never stored in graph",
                    "mutations": "Use add_to_graph() tool for writes, or query() for reads in python REPL",
                    "semantic_search": "Use semantic_search(text, index, k) with any index from vector_indexes",
                },
            }

        # =====================================================================
        # Tool 3: add_to_graph - Schema-validated writes
        # =====================================================================

        @self.mcp.tool()
        def add_to_graph(
            node_type: str,
            data: dict[str, Any] | list[dict[str, Any]],
            create_facility_relationship: bool = True,
            batch_size: int = 50,
        ) -> dict[str, Any]:
            """
            Create nodes in the knowledge graph with schema validation.

            Validates data against Pydantic models, filters out private fields,
            then writes to the graph. Use this for semantic data (files, codes, nodes).
            For infrastructure data (paths, tools, OS), use update_infrastructure() instead.

            Special handling:
            - SourceFile: Auto-deduplicates already discovered/ingested files
            - TreeNode: Auto-creates TREE_NAME and ACCESSOR_FUNCTION relationships
            - FacilityPath: Links to parent Facility

            Args:
                node_type: Node label (use get_graph_schema() to see valid types)
                data: List of property dicts matching the schema
                create_facility_relationship: Auto-create AT_FACILITY relationship
                batch_size: Nodes per batch (default: 50)

            Returns:
                Dict with counts: {"processed": N, "skipped": K, "errors": [...]}

            Examples:
                # Queue source files for ingestion
                add_to_graph("SourceFile", [
                    {"id": "tcv:/home/codes/liuqe.py", "path": "/home/codes/liuqe.py",
                     "facility_id": "tcv", "status": "discovered"}
                ])

                # Track discovered directories
                add_to_graph("FacilityPath", [
                    {"id": "tcv:/home/codes", "path": "/home/codes",
                     "facility_id": "tcv", "path_type": "code_directory",
                     "status": "discovered", "interest_score": 0.8}
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

                    # Ingestion gating: verify facility is allowed
                    facility_ids = {
                        item.get("facility_id")
                        for item in valid_items
                        if item.get("facility_id")
                    }
                    for fid in facility_ids:
                        try:
                            from imas_codex.graph.meta import gate_ingestion

                            gate_ingestion(client, fid)
                        except ValueError as gate_err:
                            return {
                                "processed": 0,
                                "skipped": len(valid_items),
                                "errors": [str(gate_err)],
                            }

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

                    result = client.create_nodes(
                        label=node_type,
                        items=valid_items,
                        id_field=id_field,
                        batch_size=batch_size,
                        create_relationships=create_facility_relationship,
                    )

                    return {
                        "processed": result["processed"],
                        "relationships": result.get("relationships", {}),
                        "skipped": len(errors) + skipped_dedup,
                        "errors": errors,
                    }

            except Exception as e:
                logger.exception("Failed to ingest nodes")
                raise RuntimeError(
                    f"Failed to ingest nodes: {_neo4j_error_message(e)}"
                ) from e

        # =====================================================================
        # Tool 4: update_facility_config - Facility configuration management
        # =====================================================================

        @self.mcp.tool()
        def update_facility_config(
            facility: str,
            data: dict[str, Any] | None = None,
            private: bool = True,
        ) -> dict[str, Any]:
            """
            Read or update facility configuration (public or private).

            Use this for infrastructure data (tools, paths, OS) that should NOT
            go in the graph. For semantic data (files, codes), use add_to_graph().

            Private data (private=True):
            - Tool versions and availability
            - File system paths
            - Hostnames and network info
            - OS and environment details
            - Exploration notes

            Public data (private=False):
            - Facility name and description
            - Machine name
            - Data system types

            Args:
                facility: Facility identifier (e.g., "tcv", "iter")
                data: If provided, update config. If None, just read.
                private: If True, update private config. If False, update public.

            Returns:
                Current config data (after update if data provided)

            Examples:
                # Read private infrastructure
                update_facility_config("iter")

                # Update tool availability (private)
                update_facility_config("iter", {"tools": {"rg": "14.1.1"}})

                # Add exploration notes (private)
                update_facility_config("iter", {
                    "exploration_notes": ["Found IMAS modules"]
                })

                # Update public metadata
                update_facility_config("iter", {
                    "description": "ITER SDCC - Updated"
                }, private=False)
            """
            try:
                if data is not None:
                    if private:
                        update_infrastructure(facility, data)
                    else:
                        update_metadata(facility, data)

                if private:
                    return get_facility_infrastructure(facility) or {}
                else:
                    from imas_codex.discovery import get_facility_metadata

                    return get_facility_metadata(facility) or {}
            except Exception as e:
                logger.exception(f"Failed to access config for {facility}")
                raise RuntimeError(f"Failed to access config: {e}") from e

        # =====================================================================
        # Tool 5: update_facility_infrastructure - Update private facility data
        # =====================================================================

        @self.mcp.tool()
        def update_facility_infrastructure(
            facility: str,
            data: dict[str, Any],
        ) -> dict[str, Any]:
            """
            Update private facility infrastructure data with deep merge.

            Use this for sensitive infrastructure data that should NOT go in the graph:
            - Tool versions and availability
            - File system paths and mounts
            - Hostnames and network info
            - OS and environment details
            - Exploration notes

            The data is deep-merged into the existing private YAML file,
            preserving comments and formatting.

            Args:
                facility: Facility identifier (e.g., "tcv", "iter")
                data: Data to merge into private file

            Returns:
                Updated private infrastructure data

            Examples:
                # Update tool availability
                update_facility_infrastructure("iter", {
                    "tools": {"rg": {"version": "14.1.1", "path": "~/bin/rg"}}
                })

                # Update file system paths
                update_facility_infrastructure("iter", {
                    "paths": {
                        "imas": {"/work/imas": "IMAS installation root"}
                    }
                })

                # Add multiple fields at once
                update_facility_infrastructure("iter", {
                    "file_systems": [{
                        "mount_point": "/mnt/HPC_T2",
                        "type": "GPFS",
                        "size": "1.5 PB"
                    }],
                    "exploration_notes": ["Discovered HPC storage"]
                })
            """
            try:
                from imas_codex.discovery import (
                    get_facility_infrastructure as _get_infra,
                    update_infrastructure as _update_infra,
                )

                _update_infra(facility, data)
                return _get_infra(facility) or {}
            except Exception as e:
                logger.exception(f"Failed to update infrastructure for {facility}")
                raise RuntimeError(f"Failed to update infrastructure: {e}") from e

        # =====================================================================
        # Tool 6: get_facility_infrastructure - Read private facility data
        # =====================================================================

        @self.mcp.tool()
        def get_facility_infrastructure(facility: str) -> dict[str, Any]:
            """
            Read private facility infrastructure data.

            Returns only the private infrastructure data (not public config).
            Use this to check what's already stored before updating.

            Args:
                facility: Facility identifier (e.g., "tcv", "iter")

            Returns:
                Private infrastructure data dict

            Example:
                # Check current infrastructure
                infra = get_facility_infrastructure("iter")
                print(infra.get("tools", {}))
                print(infra.get("exploration_notes", []))
            """
            try:
                from imas_codex.discovery import (
                    get_facility_infrastructure as _get_infra,
                )

                return _get_infra(facility) or {}
            except Exception as e:
                logger.exception(f"Failed to get infrastructure for {facility}")
                raise RuntimeError(f"Failed to get infrastructure: {e}") from e

        # =====================================================================
        # Tool 7: add_exploration_note - Append timestamped exploration note
        # =====================================================================

        @self.mcp.tool()
        def add_exploration_note(facility: str, note: str) -> list[str]:
            """
            Append a timestamped exploration note to facility's private data.

            Automatically adds ISO timestamp prefix to the note.

            Args:
                facility: Facility identifier (e.g., "tcv", "iter")
                note: Exploration note to add

            Returns:
                Updated exploration_notes list

            Examples:
                add_exploration_note("iter", "Found IMAS modules at /work/imas")
                add_exploration_note("iter", "Discovered 50 Python files in /work/codes")
            """
            try:
                from datetime import datetime

                from imas_codex.discovery import (
                    get_facility_infrastructure as _get_infra,
                )

                infra = _get_infra(facility) or {}
                notes = infra.get("exploration_notes", [])

                # Add timestamped note
                timestamp = datetime.now().strftime("%Y-%m-%d")
                timestamped_note = f"{timestamp}: {note}"
                notes.append(timestamped_note)

                update_infrastructure(facility, {"exploration_notes": notes})
                return notes
            except Exception as e:
                logger.exception(f"Failed to add exploration note for {facility}")
                raise RuntimeError(f"Failed to add exploration note: {e}") from e

        # =====================================================================
        # Tool 7b: get_discovery_context - Graph-derived discovery state
        # =====================================================================

        @self.mcp.tool()
        def get_discovery_context(facility: str) -> dict[str, Any]:
            """
            Get discovery context for a facility including graph-derived state.

            Returns comprehensive discovery state to guide exploration:
            - Configured roots and their categories
            - Coverage by category (what's already been discovered)
            - High-value paths found so far
            - Gap analysis (underrepresented categories)
            - Schema for valid category values

            Use this before exploring to identify gaps and avoid duplication.

            Args:
                facility: Facility identifier (e.g., "tcv", "iter")

            Returns:
                Dict with discovery_roots, coverage_by_category, high_value_paths,
                missing_categories, and schema with valid category values.

            Example:
                ctx = get_discovery_context("tcv")
                print("Missing categories:", ctx["missing_categories"])
                print("Coverage:", ctx["coverage_by_category"])
            """
            try:
                from imas_codex.agentic.prompt_loader import get_schema_context
                from imas_codex.discovery import (
                    get_facility_infrastructure as _get_infra,
                )
                from imas_codex.graph import GraphClient

                # Get configured roots from infrastructure
                infra = _get_infra(facility) or {}
                discovery_roots = infra.get("discovery_roots", [])

                # Get schema context for valid category values
                schema_ctx = get_schema_context()

                # Use GraphClient for graph queries
                with GraphClient() as client:
                    # Query coverage by category
                    coverage_query = """
                        MATCH (p:FacilityPath {facility_id: $facility})
                        WHERE p.status = 'scored' AND p.path_purpose IS NOT NULL
                        RETURN p.path_purpose AS purpose, count(*) AS count
                        ORDER BY count DESC
                    """
                    coverage_results = client.query(coverage_query, facility=facility)
                    coverage_by_category = {
                        record["purpose"]: record["count"]
                        for record in coverage_results
                    }

                    # Query high-value paths
                    high_value_query = """
                        MATCH (p:FacilityPath {facility_id: $facility})
                        WHERE p.score > 0.7
                        RETURN p.path AS path, p.path_purpose AS purpose,
                               p.score AS score, p.description AS description
                        ORDER BY p.score DESC LIMIT 15
                    """
                    high_value_paths = client.query(high_value_query, facility=facility)

                    # Determine missing categories (expected but not found)
                    expected_categories = [
                        c["value"] for c in schema_ctx["discovery_categories"]
                    ]
                    found_categories = set(coverage_by_category.keys())
                    missing_categories = [
                        c for c in expected_categories if c not in found_categories
                    ]

                    # Query containers not yet expanded (potential new roots)
                    unexplored_query = """
                        MATCH (p:FacilityPath {facility_id: $facility})
                        WHERE p.path_purpose = 'container'
                              AND p.score > 0.4
                              AND p.should_expand = false
                              AND p.terminal_reason IS NULL
                        RETURN p.path AS path, p.score AS score, p.description AS description
                        ORDER BY p.score DESC LIMIT 10
                    """
                    unexplored_containers = client.query(
                        unexplored_query, facility=facility
                    )

                return {
                    "facility": facility,
                    "discovery_roots": discovery_roots,
                    "coverage_by_category": coverage_by_category,
                    "total_scored_paths": sum(coverage_by_category.values()),
                    "high_value_paths": high_value_paths,
                    "missing_categories": missing_categories,
                    "unexplored_containers": unexplored_containers,
                    "schema": {
                        "valid_categories": schema_ctx["discovery_categories"],
                    },
                }
            except Exception as e:
                logger.exception(f"Failed to get discovery context for {facility}")
                raise RuntimeError(f"Failed to get discovery context: {e}") from e

        # =====================================================================
        # Tool 8: update_facility_paths - Update facility path mappings
        # =====================================================================

        @self.mcp.tool()
        def update_facility_paths(
            facility: str,
            paths: dict[str, dict[str, str]],
        ) -> dict[str, dict[str, str]]:
            """
            Update facility path mappings in private data.

            Use this to record important directory paths discovered during exploration.

            Args:
                facility: Facility identifier (e.g., "tcv", "iter")
                paths: Nested dict of path categories and paths

            Returns:
                Updated paths section

            Example:
                update_facility_paths("iter", {
                    "imas": {
                        "root": "/work/imas",
                        "core": "/work/imas/core",
                        "shared": "/work/imas/shared"
                    },
                    "codes": {
                        "chease": "/work/codes/chease",
                        "helena": "/work/codes/helena"
                    }
                })
            """
            try:
                update_infrastructure(facility, {"paths": paths})
                from imas_codex.discovery import (
                    get_facility_infrastructure as _get_infra,
                )

                infra = _get_infra(facility) or {}
                return infra.get("paths", {})
            except Exception as e:
                logger.exception(f"Failed to update paths for {facility}")
                raise RuntimeError(f"Failed to update paths: {e}") from e

        # =====================================================================
        # Tool 9: update_facility_tools - Update tool availability
        # =====================================================================

        @self.mcp.tool()
        def update_facility_tools(
            facility: str,
            tools: dict[str, dict[str, str]],
        ) -> dict[str, dict[str, str]]:
            """
            Update tool availability and versions in private data.

            Use this after running check_tools() to persist tool information.

            Args:
                facility: Facility identifier (e.g., "tcv", "iter")
                tools: Dict of tool_name -> {version, path, purpose}

            Returns:
                Updated tools section

            Example:
                update_facility_tools("iter", {
                    "rg": {
                        "version": "14.1.1",
                        "path": "/home/user/bin/rg",
                        "purpose": "Fast pattern search"
                    },
                    "fd": {
                        "version": "10.2.0",
                        "path": "/home/user/bin/fd",
                        "purpose": "Fast file finder"
                    }
                })
            """
            try:
                update_infrastructure(facility, {"tools": tools})
                from imas_codex.discovery import (
                    get_facility_infrastructure as _get_infra,
                )

                infra = _get_infra(facility) or {}
                return infra.get("tools", {})
            except Exception as e:
                logger.exception(f"Failed to update tools for {facility}")
                raise RuntimeError(f"Failed to update tools: {e}") from e

    def _register_prompts(self):
        """Register MCP prompts from markdown files.

        Static prompts: Return content as-is with includes resolved.
        Dynamic prompts (dynamic: true in frontmatter): Accept parameters
        and render with Jinja2 + schema context.
        """
        from imas_codex.agentic.prompt_loader import render_prompt

        for name, prompt_def in self._prompts.items():
            is_dynamic = prompt_def.metadata.get("dynamic", False)

            if is_dynamic:
                # Dynamic prompt: accept facility parameter, render with context
                def make_dynamic_prompt_fn(prompt_name: str, pd: PromptDefinition):
                    def prompt_fn(facility: str = "FACILITY") -> str:
                        """Render prompt with facility context and schema values.

                        Args:
                            facility: Facility identifier (e.g., "tcv", "iter")
                        """
                        try:
                            # Get facility infrastructure for ssh_host
                            from imas_codex.discovery import (
                                get_facility_infrastructure as _get_infra,
                            )

                            infra = _get_infra(facility) or {}
                            ssh_host = infra.get("ssh_host", facility)

                            context = {
                                "facility": facility,
                                "ssh_host": ssh_host,
                            }
                            return render_prompt(prompt_name, context)
                        except Exception as e:
                            logger.warning(f"Dynamic render failed: {e}, using static")
                            return pd.content

                    prompt_fn.__name__ = pd.name.replace("-", "_")
                    return prompt_fn

                self.mcp.prompt(name=name, description=prompt_def.description)(
                    make_dynamic_prompt_fn(name, prompt_def)
                )
                logger.debug(f"Registered dynamic prompt: {name}")
            else:
                # Static prompt: return content as-is
                def make_prompt_fn(pd: PromptDefinition):
                    def prompt_fn() -> str:
                        return pd.content

                    prompt_fn.__name__ = pd.name.replace("-", "_")
                    return prompt_fn

                self.mcp.prompt(name=name, description=prompt_def.description)(
                    make_prompt_fn(prompt_def)
                )
                logger.debug(f"Registered static prompt: {name}")

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
