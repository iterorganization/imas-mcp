"""
Agents MCP Server - Streamlined tools for LLM-driven facility exploration.

This server provides 15 MCP tools organized by purpose:

Unified Search (primary interface):
- search_signals: Signal search with data access and IMAS enrichment
- search_docs: Wiki/document/image search with cross-links
- search_code: Code search with data reference enrichment
- search_imas: IMAS DD search with cluster and facility cross-refs

Retrieval:
- fetch: Full content retrieval by ID or URL (WikiPage, Document, CodeFile, Image)

Graph Operations:
- get_graph_schema: Schema introspection for query generation
- add_to_graph: Schema-validated node creation with privacy filtering

Facility Infrastructure (Private Data):
- update_facility_infrastructure: Deep-merge update to private YAML
- get_facility_infrastructure: Read private infrastructure data
- add_exploration_note: Append timestamped exploration note

Configuration:
- update_facility_config: Read/update facility config (public or private)
- get_discovery_context: Get facility discovery context

Log Inspection:
- list_logs: List available log files with sizes and timestamps
- get_logs: Read logs with level/grep/time filtering
- tail_logs: Get most recent log entries

Advanced:
- python: Persistent Python REPL for custom queries

The python() REPL provides advanced operations not covered by search tools:
- Graph: query(), semantic_search(), embed()
- Domain: find_signals(), find_wiki(), find_imas(), find_code(), graph_search()
- Remote: run(), check_tools() (auto-detects local vs SSH)
- Facility: get_facility(), get_exploration_targets(), get_tree_structure()
- IMAS DD: search_imas(), fetch_imas(), list_imas(), check_imas()
- COCOS: validate_cocos(), determine_cocos(), cocos_sign_flip_paths(), cocos_info()

Use search_* MCP tools for:
- Common signal, documentation, code, and IMAS lookups
- Formatted reports with enriched results in one call

Use python() for:
- Complex multi-step operations requiring state
- Graph queries with Cypher
- Chained processing with intermediate logic
- IMAS/COCOS domain-specific operations
- Better discoverability and documentation

REPL state is initialized lazily on first use to avoid import deadlocks.
"""

import asyncio
import io
import logging
import subprocess
import sys
import threading
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from fastmcp import FastMCP
from neo4j.exceptions import ServiceUnavailable
from ruamel.yaml import YAML

from imas_codex.discovery import (
    get_facility as _get_facility_config,
    get_facility_infrastructure,
    get_facility_validated,
    update_infrastructure,
    update_metadata,
)
from imas_codex.embeddings.config import EncoderConfig
from imas_codex.embeddings.encoder import EmbeddingBackendError, Encoder
from imas_codex.graph import GraphClient, get_schema
from imas_codex.graph.schema import to_cypher_props
from imas_codex.llm.prompt_loader import (
    PromptDefinition,
    load_prompts,
)
from imas_codex.remote.tools import (
    check_all_tools as _check_all_tools,
    install_all_tools as _install_all_tools,
    run as _run,
)
from imas_codex.settings import get_embedding_location

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
# Persistent Python REPL - Initialized lazily on first use
# =============================================================================

_repl_globals: dict[str, Any] | None = None
_repl_lock = threading.Lock()
_imas_tools_instance = None


# =============================================================================
# API Reference — compact task-to-function mapping for python() docstring
# =============================================================================


def _generate_api_reference() -> str:
    """Generate compact API reference with inline parameter names.

    This runs at tool registration time — no REPL init needed.
    Keeps the reference short so agents actually read it.
    """
    return "\n".join(
        [
            "Use search_signals/search_docs/search_code/search_imas for common lookups.",
            "Use python() for custom queries not covered by the search tools.",
            "",
            "REPL functions (for custom queries in python()):",
            "  find_wiki(query, facility=, text_contains=, page_title_contains=, k=10)",
            "  wiki_page_chunks(title_contains, facility=, text_contains=, limit=50)",
            "  find_signals(query, facility=, diagnostic=, physics_domain=, limit=20)",
            "  find_imas(query) | find_code(query, facility=, limit=10)",
            "  find_data_nodes(query, facility=, data_source_name=)",
            "  map_signals_to_imas(facility, diagnostic=, physics_domain=)",
            "  facility_overview(facility)",
            "  graph_search(label, where={}, semantic=, traverse=[], return_props=[], limit=25)",
            "  query(cypher, **params)  — raw Cypher, only if no domain function fits",
            "  semantic_search(text, index=, k=5)",
            "",
            "  Format: as_table(pick(results, 'col1', 'col2'))",
            "  Schema: schema_for(task='wiki') before raw Cypher",
            "  Full API: repl_help()",
            "",
        ]
    )


def _get_imas_tools(gc: GraphClient | None = None):
    """Get or create singleton Tools instance with shared GraphClient."""
    global _imas_tools_instance
    if _imas_tools_instance is None:
        from imas_codex.tools import Tools

        if gc is None:
            gc = _get_repl()["gc"]
        _imas_tools_instance = Tools(graph_client=gc)
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


def _format_version_context_report(result: dict) -> str:
    """Format version context result into a readable report."""
    if result.get("error"):
        return f"Error: {result['error']}"

    paths_data = result.get("paths", {})
    lines = [
        f"Version Context Report: {result.get('total_paths', 0)} paths queried, "
        f"{len(result.get('paths_found', []))} found, "
        f"{result.get('paths_with_changes', 0)} with changes",
        "",
    ]

    if not paths_data:
        not_found = result.get("not_found", [])
        if not_found:
            lines.append(
                f"No matching graph paths found. Not found: {', '.join(not_found)}"
            )
            return "\n".join(lines)
        return "No version context found for the requested paths."

    for path_id, ctx in paths_data.items():
        changes = ctx.get("changes", [])
        introduced = ctx.get("introduced_in")
        deprecated = ctx.get("deprecated_in")
        lifecycle_parts = []
        if introduced:
            lifecycle_parts.append(f"introduced v{introduced}")
        if deprecated:
            lifecycle_parts.append(f"deprecated v{deprecated}")
        lifecycle_str = f" ({', '.join(lifecycle_parts)})" if lifecycle_parts else ""

        if changes:
            lines.append(f"**{path_id}**{lifecycle_str} — {len(changes)} change(s):")
            for c in changes:
                version = c.get("version", "?")
                change_type = c.get("change_type", "?")
                semantic = c.get("semantic_type")
                old_val = c.get("old_value", "")
                new_val = c.get("new_value", "")
                label = f"{change_type}" + (
                    f"/{semantic}" if semantic and semantic != "none" else ""
                )
                if old_val and new_val:
                    lines.append(f"  - v{version} [{label}]: `{old_val}` → `{new_val}`")
                elif new_val:
                    lines.append(f"  - v{version} [{label}]: added `{new_val}`")
                elif old_val:
                    lines.append(f"  - v{version} [{label}]: removed `{old_val}`")
                else:
                    lines.append(f"  - v{version} [{label}]")
        else:
            lines.append(f"**{path_id}**{lifecycle_str}: no metadata changes recorded")

    not_found = result.get("not_found", [])
    if not_found:
        lines.append("")
        lines.append(f"Not found: {', '.join(not_found)}")

    paths_without_changes = result.get("paths_without_changes", [])
    if paths_without_changes:
        lines.append("")
        lines.append(
            f"Paths without notable changes: {', '.join(paths_without_changes)}"
        )

    return "\n".join(lines)


def _format_dd_versions_report(result: dict) -> str:
    """Format DD version metadata into readable text."""
    if result.get("error"):
        return f"Error: {result['error']}"

    current_version = result.get("current_version") or "unknown"
    version_range = result.get("version_range") or "unknown"
    version_count = result.get("version_count", 0)
    versions = result.get("versions", []) or []

    lines = [
        "DD Version Metadata",
        f"Current version: {current_version}",
        f"Version range: {version_range}",
        f"Version count: {version_count}",
    ]
    if versions:
        lines.append(f"Version chain: {' -> '.join(versions)}")
    return "\n".join(lines)


def _format_error_fields_report(result: dict) -> str:
    """Format HAS_ERROR traversal results into readable text."""
    if result.get("error"):
        return f"Error: {result['error']}"

    path = result.get("path", "")
    if result.get("not_found"):
        return f"Path not found: {path}"

    error_fields = result.get("error_fields", []) or []
    if not error_fields:
        return f"No error fields found for '{path}'"

    lines = [f"Error fields for {path}:"]
    for ef in error_fields:
        line = f"  {ef.get('path', '')} ({ef.get('error_type', 'unknown')})"
        documentation = ef.get("documentation")
        if documentation:
            line += f" - {documentation[:100]}"
        lines.append(line)
    return "\n".join(lines)


def _init_repl() -> dict[str, Any]:
    """Initialize the persistent REPL environment with all utilities.

    Called lazily on first python() tool invocation. All heavy imports
    are at module level to avoid import deadlocks. Only I/O-bound work
    (Neo4j connection, encoder setup) happens here.
    """
    global _repl_globals
    if _repl_globals is not None:
        return _repl_globals

    logger.info("Initializing Python REPL...")

    from imas_codex.graph import domain_queries as _dq
    from imas_codex.graph.formatters import as_summary, as_table, pick
    from imas_codex.graph.query_builder import graph_search as _graph_search
    from imas_codex.graph.schema_context import schema_for as _schema_for

    gc = GraphClient.from_profile()

    # Create encoder with lazy initialization - respects embedding-backend config
    # This will NOT load the model until actually used
    backend = get_embedding_location()
    logger.info(f"Embedding location: {backend}")

    _encoder: Encoder | None = None

    def _get_encoder() -> Encoder:
        """Get or create the encoder, retrying on failure."""
        nonlocal _encoder

        if _encoder is None:
            try:
                config = EncoderConfig()
                _encoder = Encoder(config)
                logger.info(f"Encoder initialized (backend={config.backend})")
            except Exception as e:
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
            query('MATCH (t:SignalNode {data_source_name: $tree}) RETURN t.path LIMIT 10', tree='results')
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
        index: str = "imas_node_embedding",
        k: int = 5,
        include_deprecated: bool = False,
    ) -> list[dict[str, Any]]:
        """Vector similarity search on graph embeddings.

        Args:
            text: Query text to embed and search
            index: Vector index name (use get_graph_schema() to list all)
            k: Number of results to return
            include_deprecated: If True, include deprecated IMAS paths in results.
                Only applies to imas_node_embedding index. Default False (active only).

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

        # Filter deprecated paths for imas_node_embedding unless explicitly included
        where_clause = ""
        if index == "imas_node_embedding" and not include_deprecated:
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
            "OPTIONAL MATCH (wa:Document)-[:HAS_CHUNK]->(node) "
            "RETURN [k IN keys(node) "
            "WHERE NOT k ENDS WITH 'embedding' | [k, node[k]]] "
            "AS properties, labels(node) AS labels, score, "
            "p.title AS page_title, p.url AS page_url, "
            "wa.id AS document_id, wa.title AS document_title, wa.url AS document_url "
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
            # Add parent page context (WikiPage or Document)
            if r.get("page_title"):
                d["page_title"] = r["page_title"]
                d["page_url"] = r["page_url"]
            elif r.get("document_id"):
                d["page_title"] = r["document_title"] or r["document_id"]
                if r.get("document_url"):
                    d["page_url"] = r["document_url"]
            out.append(d)
        return out

    def _search_code_chunks(embedding: list[float], k: int) -> list[dict[str, Any]]:
        """Code-specific search that enriches results with source file context."""
        results = gc.query(
            'CALL db.index.vector.queryNodes("code_chunk_embedding", $k, $embedding) '
            "YIELD node, score "
            "OPTIONAL MATCH (sf:CodeFile)-[:HAS_CHUNK]->(node) "
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
        (FacilityConfig model) so all typed fields (data_systems, data_systems,
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
                OPTIONAL MATCH (m:DataSource)-[:AT_FACILITY]->(f)
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
                RETURN p.path AS path, p.score_composite AS score, p.description AS description
                ORDER BY COALESCE(p.score_composite, 0) DESC
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
                MATCH (t:DataSource)-[:AT_FACILITY]->(f:Facility {id: $fid})
                OPTIONAL MATCH (n:SignalNode {data_source_name: t.name})-[:AT_FACILITY]->(f)
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
                RETURN p.path AS path, p.score_composite AS score
                ORDER BY COALESCE(p.score_composite, 0) DESC
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
        data_source_name: str, path_prefix: str = "", limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get SignalNode structure from the graph.

        Args:
            data_source_name: MDSplus tree name (e.g., 'results', 'magnetics')
            path_prefix: Optional path prefix to filter
            limit: Maximum nodes to return

        Returns:
            List of {path, description, unit, physics_domain}
        """
        if path_prefix:
            return gc.query(
                """
                MATCH (n:SignalNode {data_source_name: $tree})
                WHERE n.path STARTS WITH $prefix
                RETURN n.path AS path, n.description AS description,
                       n.unit AS unit, n.physics_domain AS domain
                ORDER BY n.path
                LIMIT $limit
                """,
                tree=data_source_name,
                prefix=path_prefix,
                limit=limit,
            )
        else:
            return gc.query(
                """
                MATCH (n:SignalNode {data_source_name: $tree})
                RETURN n.path AS path, n.description AS description,
                       n.unit AS unit, n.physics_domain AS domain
                ORDER BY n.path
                LIMIT $limit
                """,
                tree=data_source_name,
                limit=limit,
            )

    # =========================================================================
    # Code search utilities
    # =========================================================================

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
        from imas_codex.ingestion.search import search_code_chunks

        results = search_code_chunks(
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
        dd_version: int | None = None,
    ) -> str:
        """Search IMAS Data Dictionary using semantic search.

        Excludes error fields and metadata subtrees from results.
        Use fetch_imas_paths to access error fields via HAS_ERROR relationships.

        Args:
            query_text: Natural language query
            ids_filter: Optional IDS name filter (space-delimited)
            max_results: Maximum results
            dd_version: Filter by DD major version (e.g., 3 or 4)

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
                    dd_version=dd_version,
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

    def fetch_imas(paths: str, dd_version: int | None = None) -> str:
        """Get full documentation for IMAS paths.

        Args:
            paths: Space-delimited IMAS paths
            dd_version: Filter by DD major version (e.g., 3 or 4)

        Returns:
            Detailed path documentation
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.fetch_imas_paths(paths=paths, dd_version=dd_version)
            )
            return str(result)
        except Exception as e:
            return f"Fetch error: {e}"

    def list_imas(
        paths: str,
        leaf_only: bool = True,
        max_paths: int = 100,
        dd_version: int | None = None,
    ) -> str:
        """List data paths in IDS.

        Args:
            paths: Space-separated IDS names or path prefixes
            leaf_only: Only return data fields
            max_paths: Limit output size
            dd_version: Filter by DD major version (e.g., 3 or 4)

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
                    dd_version=dd_version,
                )
            )
            return str(result)
        except Exception as e:
            return f"List error: {e}"

    def check_imas(paths: str, dd_version: int | None = None) -> str:
        """Validate IMAS paths for existence.

        Args:
            paths: Space-delimited IMAS paths
            dd_version: Filter by DD major version (e.g., 3 or 4)

        Returns:
            Validation results with existence status
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.check_imas_paths(paths=paths, dd_version=dd_version)
            )
            return str(result)
        except Exception as e:
            return f"Check error: {e}"

    def get_imas_overview(
        query_text: str | None = None,
        dd_version: int | None = None,
    ) -> str:
        """Get high-level overview of IMAS Data Dictionary.

        Args:
            query_text: Optional keyword filter
            dd_version: Filter by DD major version (e.g., 3 or 4)

        Returns:
            Overview with IDS list, physics domains, statistics
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.get_imas_overview(query=query_text, dd_version=dd_version)
            )
            return str(result)
        except Exception as e:
            return f"Overview error: {e}"

    def get_imas_path_context(
        path: str,
        relationship_types: str = "all",
        dd_version: int | None = None,
    ) -> str:
        """Get cross-IDS structural context for an IMAS path.

        Args:
            path: Exact IMAS path
            relationship_types: 'cluster', 'coordinate', 'unit', 'identifier', or 'all'
            dd_version: Filter by DD major version (e.g., 3 or 4)

        Returns:
            Cross-IDS connections grouped by type
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.get_imas_path_context(
                    path=path,
                    relationship_types=relationship_types,
                    dd_version=dd_version,
                )
            )
            return str(result)
        except Exception as e:
            return f"Path context error: {e}"

    def analyze_imas_structure(
        ids_name: str,
        dd_version: int | None = None,
    ) -> str:
        """Analyze the hierarchical structure of an IMAS IDS.

        Args:
            ids_name: IDS name (e.g. 'equilibrium')
            dd_version: Filter by DD major version (e.g., 3 or 4)

        Returns:
            Structural analysis with depth, types, domains
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.analyze_imas_structure(ids_name=ids_name, dd_version=dd_version)
            )
            return str(result)
        except Exception as e:
            return f"Structure analysis error: {e}"

    def export_imas_ids(
        ids_name: str,
        leaf_only: bool = False,
        dd_version: int | None = None,
    ) -> str:
        """Export full IDS structure with documentation.

        Args:
            ids_name: IDS name (e.g. 'equilibrium')
            leaf_only: If true, return only leaf nodes
            dd_version: Filter by DD major version (e.g., 3 or 4)

        Returns:
            Full IDS path listing
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.export_imas_ids(
                    ids_name=ids_name, leaf_only=leaf_only, dd_version=dd_version
                )
            )
            return str(result)
        except Exception as e:
            return f"Export error: {e}"

    def export_imas_domain(
        domain: str,
        ids_filter: str | None = None,
        dd_version: int | None = None,
    ) -> str:
        """Export all IMAS paths in a physics domain.

        Args:
            domain: Physics domain name (e.g. 'magnetics')
            ids_filter: Optional IDS name filter
            dd_version: Filter by DD major version (e.g., 3 or 4)

        Returns:
            Domain export grouped by IDS
        """
        try:
            tools = _get_imas_tools()
            result = _run_async(
                tools.export_imas_domain(
                    domain=domain, ids_filter=ids_filter, dd_version=dd_version
                )
            )
            return str(result)
        except Exception as e:
            return f"Domain export error: {e}"

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
    # Domain query functions (bound to this REPL's gc/embed)
    # =========================================================================

    import functools as _ft

    def _bind_dq(fn):
        """Bind gc and embed_fn into a domain query function."""

        @_ft.wraps(fn)
        def wrapper(*args, **kwargs):
            kwargs.setdefault("gc", gc)
            kwargs.setdefault("embed_fn", embed)
            return fn(*args, **kwargs)

        return wrapper

    find_signals = _bind_dq(_dq.find_signals)
    find_wiki = _bind_dq(_dq.find_wiki)
    wiki_page_chunks = _bind_dq(_dq.wiki_page_chunks)
    find_imas = _bind_dq(_dq.find_imas)
    find_code = _bind_dq(_dq.find_code)
    find_data_nodes = _bind_dq(_dq.find_data_nodes)
    map_signals_to_imas = _bind_dq(_dq.map_signals_to_imas)
    facility_overview = _bind_dq(_dq.facility_overview)
    graph_search = _bind_dq(_graph_search)

    # =========================================================================
    # REPL Registry — single source of truth for exposed functions
    # =========================================================================
    # To add a function: define it above, then add one entry here.
    # This registry drives: _repl_globals, repl_help(), and the python()
    # tool docstring. No other place needs updating.
    #
    # Format: list of (category, [(name, function), ...])
    # The name is what agents type in the REPL.

    _REPL_REGISTRY: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "DOMAIN QUERIES (prefer over raw Cypher)",
            [
                ("find_signals", find_signals),
                ("find_wiki", find_wiki),
                ("wiki_page_chunks", wiki_page_chunks),
                ("find_code", find_code),
                ("find_imas", find_imas),
                ("find_data_nodes", find_data_nodes),
                ("map_signals_to_imas", map_signals_to_imas),
                ("facility_overview", facility_overview),
            ],
        ),
        (
            "QUERY BUILDER",
            [("graph_search", graph_search)],
        ),
        (
            "FORMATTERS",
            [
                ("as_table", as_table),
                ("as_summary", as_summary),
                ("pick", pick),
            ],
        ),
        (
            "SCHEMA (call before writing Cypher)",
            [
                ("schema_for", _schema_for),
                ("get_schema", get_schema),
            ],
        ),
        (
            "GRAPH",
            [
                ("query", query),
                ("semantic_search", semantic_search),
                ("embed", embed),
            ],
        ),
        (
            "FACILITY",
            [
                ("get_facility", get_facility),
                ("get_facility_infrastructure", get_facility_infrastructure),
                ("update_infrastructure", update_infrastructure),
                ("get_exploration_targets", get_exploration_targets),
                ("get_tree_structure", get_tree_structure),
            ],
        ),
        (
            "REMOTE",
            [
                ("run", run),
                ("check_tools", check_tools),
            ],
        ),
        (
            "IMAS DD",
            [
                ("search_imas", search_imas),
                ("fetch_imas", fetch_imas),
                ("list_imas", list_imas),
                ("check_imas", check_imas),
                ("get_imas_overview", get_imas_overview),
                ("get_imas_path_context", get_imas_path_context),
                ("analyze_imas_structure", analyze_imas_structure),
                ("export_imas_ids", export_imas_ids),
                ("export_imas_domain", export_imas_domain),
            ],
        ),
        (
            "COCOS",
            [
                ("validate_cocos", validate_cocos),
                ("determine_cocos", determine_cocos),
                ("cocos_info", cocos_info),
            ],
        ),
    ]

    # Internal params injected by REPL binding — hidden from API reference
    _INTERNAL_PARAMS = {"gc", "embed_fn"}

    def _generate_repl_help() -> str:
        """Generate compact API reference from the registry."""
        import inspect

        lines = ["=== CODEX REPL API ===", ""]

        for cat, funcs in _REPL_REGISTRY:
            lines.append(f"  {cat}:")
            for name, fn in funcs:
                try:
                    sig = inspect.signature(fn)
                    params = {
                        k: v
                        for k, v in sig.parameters.items()
                        if k not in _INTERNAL_PARAMS
                    }
                    clean_sig = sig.replace(parameters=list(params.values()))
                    lines.append(f"    {name}{clean_sig}")
                except (ValueError, TypeError):
                    lines.append(f"    {name}(...)")
            lines.append("")

        lines.extend(
            [
                "  TIPS:",
                "  - Chain queries in a single python() call",
                "  - Call schema_for(task='wiki') before raw Cypher",
                "  - Use as_table(pick(results, 'col1', 'col2')) for output",
                "  - Call help(fn) for full docstring",
                "",
            ]
        )

        return "\n".join(lines)

    def repl_help() -> str:
        """Print auto-generated API reference for all REPL functions."""
        ref = _generate_repl_help()
        print(ref)
        return ref

    # =========================================================================
    # Build REPL globals from registry
    # =========================================================================

    _repl_globals = {name: fn for _, funcs in _REPL_REGISTRY for name, fn in funcs}
    _repl_globals.update(
        {
            # Core objects
            "gc": gc,
            "EmbeddingBackendError": EmbeddingBackendError,
            # Additional utilities not in the API reference
            "update_metadata": update_metadata,
            "install_tools": install_tools,
            "search_code": search_code,
            "get_imas_overview": get_imas_overview,
            "cocos_sign_flip_paths": cocos_sign_flip_paths,
            # REPL management
            "reload": _reload_repl,
            "repl_help": repl_help,
            # Standard library
            "subprocess": subprocess,
            # Result storage
            "_": None,
        }
    )

    logger.info(
        "Python REPL initialized with graph, IMAS, COCOS, and facility utilities"
    )
    return _repl_globals


def _get_repl() -> dict[str, Any]:
    """Get the persistent REPL environment, initializing lazily on first call.

    Thread-safe: uses a lock to prevent concurrent initialization.
    All imports are at module level, so no import deadlock risk.
    """
    global _repl_globals
    if _repl_globals is not None:
        return _repl_globals
    with _repl_lock:
        # Double-check after acquiring lock
        if _repl_globals is None:
            _init_repl()
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
# MCP Server with 9 Core Tools
# =============================================================================


@dataclass
class AgentsServer:
    """
    MCP server with 7 core tools for facility exploration.

    Uses lazy initialization for the REPL — the first python() call
    triggers GraphClient connection and encoder setup. This avoids
    import deadlocks that occur when background threads perform imports.

    Tools:
    - search_signals: Signal search with data access and IMAS enrichment
    - search_docs: Wiki/document/image search with cross-links
    - search_code: Code search with data reference enrichment
    - search_imas: IMAS DD search with cluster and facility cross-refs
    - python: Persistent REPL for custom queries not covered above
    - get_graph_schema: Schema introspection for query generation
    - add_to_graph: Schema-validated node creation with privacy filtering
    - update_facility_config: Read/update facility config (public or private)
    - update_facility_infrastructure: Deep-merge update to private YAML
    - get_facility_infrastructure: Read private infrastructure data
    - add_exploration_note: Append timestamped exploration note

    The python() REPL provides access to:
    - Graph: query(), semantic_search(), embed(), graph_search()
    - Domain: find_signals(), find_wiki(), wiki_page_chunks(), find_imas(), find_code()
    - Formatters: as_table(), as_summary(), pick()
    - Remote: run(), check_tools() (auto-detects local vs SSH)
    - Facility: get_facility(), get_exploration_targets(), get_tree_structure()
    - IMAS DD: search_imas(), fetch_imas(), list_imas(), check_imas()
    - COCOS: validate_cocos(), determine_cocos(), cocos_sign_flip_paths(), cocos_info()
    - Code: search_code()
    """

    read_only: bool = False
    dd_only: bool | None = None
    mcp: FastMCP = field(init=False, repr=False)
    _prompts: dict[str, PromptDefinition] = field(init=False, repr=False)
    _started_at: float = field(init=False, repr=False)

    # Tools that require facility data — not registered in DD-only mode
    FACILITY_TOOLS: ClassVar[frozenset[str]] = frozenset(
        {
            "search_signals",
            "signal_analytics",
            "search_docs",
            "search_code",
            "fetch_facility_resource",
            "get_discovery_context",
            "get_facility_infrastructure",
        }
    )

    def __post_init__(self):
        """Initialize the MCP server with lazy REPL loading.

        REPL initialization is deferred to the first python() tool call.
        All imports are at module level to avoid deadlocks. Only I/O-bound
        work (Neo4j connection) happens lazily.
        """
        import time

        self._started_at = time.monotonic()

        # Auto-detect DD-only mode from graph content
        if self.dd_only is None:
            self.dd_only = self._detect_dd_only()

        # DD-only implies read-only: no write tools needed for a DD-only deployment
        if self.dd_only:
            self.read_only = True

        name = "imas-codex-readonly" if self.read_only else "imas-codex"
        self.mcp = FastMCP(name=name)
        self._prompts = load_prompts()

        self._register_tools()
        self._register_prompts()
        self._register_health_check()

        # In DD-only mode, facility tools are never registered (see guards
        # in _register_tools). Log the active tool count for diagnostics.
        tool_count = sum(
            1 for k in self.mcp._local_provider._components if k.startswith("tool:")
        )
        mode_parts = []
        if self.read_only:
            mode_parts.append("read-only")
        if self.dd_only:
            mode_parts.append("dd-only")
        mode = ", ".join(mode_parts) if mode_parts else "read-write"
        logger.info(
            f"MCP server ready ({mode}) with {tool_count} tools and {len(self._prompts)} prompts"
        )

        # Pre-warm the embedding model in a background thread so the first
        # search_imas call doesn't pay the 30s+ cold-start penalty.
        # This does not block server startup.
        import threading

        def _warmup():
            from imas_codex.tools.graph_search import warmup_encoder

            warmup_encoder()

        threading.Thread(target=_warmup, daemon=True, name="encoder-warmup").start()

    @staticmethod
    def _detect_dd_only() -> bool:
        """Detect DD-only mode by checking for Facility nodes in the graph."""
        try:
            from imas_codex.graph.client import GraphClient
            from imas_codex.graph.meta import get_graph_meta

            gc = GraphClient.from_profile()
            try:
                meta = get_graph_meta(gc)
                if meta:
                    facilities = meta.get("facilities") or []
                    dd_only = len(facilities) == 0
                    if dd_only:
                        logger.info("Auto-detected DD-only graph (no facilities)")
                    else:
                        logger.info(
                            f"Auto-detected full graph with facilities: "
                            f"{', '.join(facilities)}"
                        )
                    return dd_only
            finally:
                gc.close()
        except Exception:
            logger.warning("Could not detect graph mode, defaulting to full mode")
        return False

    def _register_tools(self):
        """Register all MCP tools."""

        # Generate API reference at registration time from the source
        # functions (no REPL init needed — just inspect.signature on
        # the unbound originals).
        api_reference = _generate_api_reference()

        if not self.read_only:
            # =====================================================================
            # Tool 1: python - Persistent REPL (primary interface)
            # =====================================================================

            @self.mcp.tool()
            def python(code: str) -> str:
                repl = _get_repl()

                stdout_capture = io.StringIO()

                try:
                    with redirect_stdout(stdout_capture):
                        try:
                            result = eval(code, repl)
                            if result is not None:
                                repl["_"] = result
                                print(repr(result))
                        except SyntaxError:
                            exec(code, repl)

                    output = stdout_capture.getvalue()
                    if not output:
                        output = "(no output)"
                    return output

                except Exception as e:
                    import traceback

                    tb = traceback.format_exc()
                    return f"Error: {e}\n\n{tb}"

            # Set the docstring dynamically so it's always in sync with
            # the actual registered functions (generated from introspection)
            python.__doc__ = (
                "Execute Python in a persistent REPL for custom graph queries "
                "and operations not covered by the search_* tools. Variables "
                "persist across calls.\n\n"
                "Prefer search_signals/search_docs/search_code/search_imas for "
                "common lookups — they return formatted reports in one call.\n\n"
                f"{api_reference}\n"
                "Args:\n"
                "    code: Python code to execute (multi-line supported)\n\n"
                "Returns:\n"
                "    stdout output, or repr of last expression if no print"
            )

        # =====================================================================
        # Tool 2: get_graph_schema - Schema introspection
        # =====================================================================

        @self.mcp.tool()
        def get_graph_schema(
            scope: str = "overview",
        ) -> str:
            """
            Get graph schema context for Cypher query generation.

            Returns compact, task-relevant schema in text format. Use scope to
            get only the schema slice you need, reducing token usage.

            Args:
                scope: One of "overview" (compact summary of all labels),
                       "signals", "wiki", "imas", "code", "facility", "trees"
                       (detailed schema for that domain).

            Returns:
                Compact text schema context for LLM consumption.

            Examples:
                get_graph_schema()  # Overview of all labels
                get_graph_schema("signals")  # Signal-related schema
                get_graph_schema("imas")  # IMAS DD schema
            """
            from imas_codex.graph.schema_context import schema_for

            return schema_for(task=scope)

        if not self.read_only:
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
                - CodeFile: Auto-deduplicates already discovered/ingested files
                - SignalNode: Auto-creates IN_DATA_SOURCE and ACCESSOR_FUNCTION relationships
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
                    add_to_graph("CodeFile", [
                        {"id": "tcv:/home/codes/liuqe.py", "path": "/home/codes/liuqe.py",
                         "facility_id": "tcv", "status": "discovered"}
                    ])

                    # Track discovered directories
                    add_to_graph("FacilityPath", [
                        {"id": "tcv:/home/codes", "path": "/home/codes",
                         "facility_id": "tcv", "path_type": "code_directory",
                         "status": "discovered", "score_composite": 0.8}
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

                valid_items: list[dict[str, Any]] = []
                errors: list[str] = []

                for i, item in enumerate(items):
                    try:
                        filtered = {
                            k: v for k, v in item.items() if k not in private_slots
                        }
                        validated = model_class.model_validate(filtered)
                        props = to_cypher_props(validated)
                        valid_items.append(props)
                    except Exception as e:
                        item_id = item.get("id", f"item[{i}]")
                        errors.append(f"{item_id}: {e}")
                        logger.warning(
                            f"Validation failed for {node_type} {item_id}: {e}"
                        )

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

                        # CodeFile deduplication
                        if node_type == "CodeFile":
                            existing = client.query(
                                """
                                UNWIND $items AS item
                                OPTIONAL MATCH (sf:CodeFile {id: item.id})
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

        if not self.read_only:
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

        if not self.read_only:
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

        if not self.dd_only:

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

        if not self.read_only:
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

        if not self.dd_only:

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
                    from imas_codex.discovery import (
                        get_facility_infrastructure as _get_infra,
                    )
                    from imas_codex.graph import GraphClient
                    from imas_codex.llm.prompt_loader import get_schema_for_prompt

                    # Get configured roots from infrastructure
                    infra = _get_infra(facility) or {}
                    discovery_roots = infra.get("discovery_roots", [])

                    # Get schema context for valid category values
                    schema_ctx = get_schema_for_prompt(
                        "discovery/roots", ["discovery_categories"]
                    )

                    # Use GraphClient for graph queries
                    with GraphClient() as client:
                        # Query coverage by category
                        coverage_query = """
                            MATCH (p:FacilityPath {facility_id: $facility})
                            WHERE p.status = 'scored' AND p.path_purpose IS NOT NULL
                            RETURN p.path_purpose AS purpose, count(*) AS count
                            ORDER BY count DESC
                        """
                        coverage_results = client.query(
                            coverage_query, facility=facility
                        )
                        coverage_by_category = {
                            record["purpose"]: record["count"]
                            for record in coverage_results
                        }

                        # Query high-value paths
                        high_value_query = """
                            MATCH (p:FacilityPath {facility_id: $facility})
                            WHERE coalesce(p.score_composite, p.triage_composite) > 0.7
                            RETURN p.path AS path, p.path_purpose AS purpose,
                                   coalesce(p.score_composite, p.triage_composite) AS score, p.description AS description
                            ORDER BY score DESC LIMIT 15
                        """
                        high_value_paths = client.query(
                            high_value_query, facility=facility
                        )

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
                                  AND coalesce(p.score_composite, p.triage_composite) > 0.4
                                  AND p.should_expand = false
                                  AND p.terminal_reason IS NULL
                            RETURN p.path AS path, coalesce(p.score_composite, p.triage_composite) AS score, p.description AS description
                            ORDER BY score DESC LIMIT 10
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

        # NOTE: update_facility_paths and update_facility_tools were removed as
        # MCP tools (Phase 5 consolidation). Use update_infrastructure() in the
        # REPL instead: update_infrastructure('facility', {'paths': {...}})

        # =====================================================================
        # Unified search tools — multi-index vector search + graph enrichment
        # =====================================================================

        # Build dynamic docstring fragments for valid parameter values
        from imas_codex.discovery.base.scoring import (
            CODE_SCORE_DIMENSIONS,
            CONTENT_SCORE_DIMENSIONS,
        )
        from imas_codex.llm.search_tools import (
            _fetch,
            _search_code,
            _search_docs,
            _search_signals,
            _signal_analytics,
        )

        _content_scores_doc = ", ".join(sorted(CONTENT_SCORE_DIMENSIONS))
        _code_scores_doc = ", ".join(sorted(CODE_SCORE_DIMENSIONS))

        # Read physics domains from LinkML schema
        try:
            from linkml_runtime.utils.schemaview import SchemaView

            _sv = SchemaView(
                str(
                    __import__("pathlib").Path(__file__).resolve().parent.parent
                    / "schemas"
                    / "physics_domains.yaml"
                )
            )
            _pd_enum = _sv.get_enum("PhysicsDomain")
            _physics_domains_doc = ", ".join(sorted(_pd_enum.permissible_values.keys()))
            del _sv, _pd_enum
        except Exception:
            _physics_domains_doc = "(see physics_domains.yaml for valid values)"

        if not self.dd_only:

            @self.mcp.tool()
            def search_signals(
                query: str,
                facility: str,
                diagnostic: str | None = None,
                physics_domain: str | None = None,
                check_status: str | None = None,
                error_type: str | None = None,
                include_check_details: bool = False,
                k: int = 20,
            ) -> str:
                """Search facility signals with full graph enrichment.

                Performs hybrid search (vector + keyword) on signal descriptions,
                then enriches with data access templates, IMAS mappings, diagnostic
                context, and related tree nodes.

                Use this for: "How do I access [quantity] at [facility]?"

                Args:
                    query: Natural language search text (e.g. "plasma current")
                    facility: Facility id (required, e.g. "tcv", "jet")
                    diagnostic: Optional diagnostic filter (e.g. "magnetics")
                    physics_domain: Optional physics domain filter
                    check_status: Filter by check outcome: "passed", "failed", or "unchecked"
                    error_type: Filter by error classification (e.g. "not_available_for_shot")
                    include_check_details: Include CHECKED_WITH relationship data in results
                    k: Number of results (default 20)

                Returns:
                    Formatted report with signals, data access, IMAS mappings,
                    and related tree nodes.
                """
                return _search_signals(
                    query,
                    facility,
                    diagnostic=diagnostic,
                    physics_domain=physics_domain,
                    check_status=check_status,
                    error_type=error_type,
                    include_check_details=include_check_details,
                    k=k,
                )

            @self.mcp.tool()
            def signal_analytics(
                facility: str,
                group_by: list[str] | None = None,
                filters: dict[str, str] | None = None,
            ) -> str:
                """Aggregate signal counts by specified dimensions.

                Use this for batch analytics queries like "how many signals
                pass/fail checks?" or "signal breakdown by physics domain".

                Args:
                    facility: Facility id (required, e.g. "tcv", "jet")
                    group_by: Dimensions to group by (default: ["status"]).
                        Allowed: status, physics_domain, data_source_name,
                        discovery_source, diagnostic, check_status, error_type
                    filters: Optional key-value filters to narrow results
                        (e.g. {"status": "checked", "physics_domain": "magnetics"})

                Returns:
                    Formatted table with counts and percentages per group.
                """
                return _signal_analytics(facility, group_by=group_by, filters=filters)

            @self.mcp.tool()
            def search_docs(
                query: str,
                facility: str,
                k: int = 15,
                site: str | None = None,
                physics_domain: str | None = None,
                min_score: float | None = None,
                score_dimension: str | None = None,
            ) -> str:
                """Search documentation (wiki, documents, images) with cross-links.

                Performs hybrid search (vector + keyword) across wiki content,
                linked documents, and images, enriched with cross-references to
                signals, tree nodes, and IMAS paths.

                Use this for: "What does the knowledge base say about [topic] at [facility]?"

                Args:
                    query: Natural language search text (e.g. "fishbone instabilities")
                    facility: Facility id (required, e.g. "tcv", "jet")
                    k: Results per index (default 15)
                    site: Optional wiki site filter (substring match on wiki URL)
                    physics_domain: Filter by WikiPage physics domain
                    min_score: Minimum score threshold (0.0-1.0) for score_dimension
                    score_dimension: Score dimension to filter on

                Returns:
                    Formatted report with wiki documentation grouped by page,
                    cross-links to signals/IMAS paths, and related documents.
                """
                return _search_docs(
                    query,
                    facility,
                    k=k,
                    site=site,
                    physics_domain=physics_domain,
                    min_score=min_score,
                    score_dimension=score_dimension,
                )

            @self.mcp.tool()
            def search_code(
                query: str,
                facility: str | None = None,
                k: int = 10,
                physics_domain: str | None = None,
                min_score: float | None = None,
                score_dimension: str | None = None,
            ) -> str:
                """Search ingested code with data reference enrichment.

                Performs hybrid search (vector + keyword) on code chunks,
                enriched with MDSplus paths, TDI function calls, IMAS path
                references, and directory context.

                Use this for: "Show me code that does [task] at [facility]"

                Args:
                    query: Natural language search text (e.g. "equilibrium reconstruction")
                    facility: Optional facility filter (e.g. "tcv")
                    k: Number of results (default 10)
                    physics_domain: Filter by FacilityPath physics domain
                    min_score: Minimum score threshold (0.0-1.0) for score_dimension
                    score_dimension: Score dimension to filter on

                Returns:
                    Formatted report with code examples, data references,
                    and directory context.
                """
                return _search_code(
                    query,
                    facility=facility,
                    k=k,
                    physics_domain=physics_domain,
                    min_score=min_score,
                    score_dimension=score_dimension,
                )

        @self.mcp.tool()
        def search_imas(
            query: str,
            ids_filter: str | None = None,
            facility: str | None = None,
            include_version_context: bool = False,
            dd_version: int | None = None,
            k: int = 20,
        ) -> str:
            """Search IMAS Data Dictionary with cross-domain enrichment.

            Performs hybrid search (vector + keyword) across IMAS path and
            cluster embeddings, enriched with cluster membership, coordinate
            context, units, and optional facility cross-references and
            version history.

            Error fields (_error_upper, _error_lower, _error_index) and
            metadata subtrees (ids_properties/*, code/*) are excluded from
            search results. To access error fields for a data path, use
            fetch_imas_paths to get HAS_ERROR relationships.

            Use this for: "What IMAS paths represent [concept]?"

            Args:
                query: Natural language search text (e.g. "electron temperature")
                ids_filter: Optional IDS name filter (e.g. "core_profiles")
                facility: Optional facility for cross-references (e.g. "tcv")
                include_version_context: Include DD version change history
                dd_version: Filter by DD major version (e.g., 3 or 4)
                k: Number of results (default 20)

            Returns:
                Formatted report with IMAS paths, clusters, facility
                cross-references, and version context.
            """
            from concurrent.futures import ThreadPoolExecutor

            from imas_codex.llm.search_formatters import format_search_imas_report
            from imas_codex.models.error_models import ToolError

            tools = _get_imas_tools()

            # Run path search and cluster search in parallel — they are
            # independent operations sharing the same encoder singleton.
            def _path_search():
                return _run_async(
                    tools.search_imas_paths(
                        query=query,
                        ids_filter=ids_filter,
                        max_results=k,
                        facility=facility,
                        include_version_context=include_version_context,
                        dd_version=dd_version,
                    )
                )

            def _cluster_search():
                try:
                    cr = _run_async(
                        tools.clusters_tool.search_imas_clusters(
                            query=query,
                            ids_filter=ids_filter,
                            dd_version=dd_version,
                        )
                    )
                    if isinstance(cr, ToolError):
                        logger.debug(
                            "Cluster search returned ToolError, "
                            "omitting optional cluster context"
                        )
                        return None
                    return cr
                except Exception:
                    logger.debug(
                        "Cluster search failed, continuing without",
                        exc_info=True,
                    )
                    return None

            with ThreadPoolExecutor(max_workers=2) as executor:
                path_future = executor.submit(_path_search)
                cluster_future = executor.submit(_cluster_search)
                result = path_future.result()
                cluster_result = cluster_future.result()

            if isinstance(result, ToolError):
                return format_search_imas_report(result)

            return format_search_imas_report(result, cluster_result)

        # =====================================================================
        # Promoted IMAS DD tools — delegate to shared Tools via _get_imas_tools()
        # =====================================================================

        from imas_codex.llm.search_formatters import (
            format_check_report,
            format_cluster_report,
            format_fetch_paths_report,
            format_identifiers_report,
            format_list_report,
            format_overview_report,
        )

        @self.mcp.tool()
        def check_imas_paths(
            paths: str,
            ids: str | None = None,
            dd_version: int | None = None,
        ) -> str:
            """Validate IMAS paths against the Data Dictionary graph.

            Checks whether exact paths exist, reports data types and units,
            and suggests corrections for renamed or misspelled paths.

            Args:
                paths: Space or comma-delimited IMAS paths to validate
                    (e.g., "equilibrium/time_slice/profiles_1d/psi core_profiles/profiles_1d/electrons/temperature")
                ids: Optional IDS prefix to prepend to all paths
                dd_version: Filter by DD major version (e.g., 3 or 4)

            Returns:
                Formatted validation report with existence status per path.
            """
            tools = _get_imas_tools()
            result = _run_async(
                tools.check_imas_paths(paths=paths, ids=ids, dd_version=dd_version)
            )
            return format_check_report(result)

        @self.mcp.tool()
        def fetch_imas_paths(
            paths: str,
            ids: str | None = None,
            dd_version: int | None = None,
            include_version_history: bool = False,
        ) -> str:
            """Get full documentation for IMAS paths including units, coordinates, cluster membership.

            Returns detailed information for each path: documentation text,
            data type, units, coordinate specs, semantic cluster labels,
            physics domain classification, identifier schemas, and
            optionally version change history.

            Args:
                paths: Space or comma-delimited IMAS paths
                    (e.g., "equilibrium/time_slice/profiles_1d/psi")
                ids: Optional IDS prefix to prepend
                dd_version: Filter by DD major version (e.g., 3 or 4)
                include_version_history: Include notable version changes

            Returns:
                Formatted path documentation report.
            """
            tools = _get_imas_tools()
            result = _run_async(
                tools.fetch_imas_paths(
                    paths=paths,
                    ids=ids,
                    dd_version=dd_version,
                    include_version_history=include_version_history,
                )
            )
            return format_fetch_paths_report(result)

        @self.mcp.tool()
        def fetch_error_fields(
            path: str,
            dd_version: int | None = None,
        ) -> str:
            """Fetch error fields for a data path via HAS_ERROR relationships.

            Returns the error fields (_error_upper, _error_lower,
            _error_index) associated with a given data path. Error fields
            are not included in search or list results — use this tool
            to discover them for a known data path.

            Args:
                path: IMAS data path (e.g., "equilibrium/time_slice/profiles_1d/psi")
                dd_version: Filter by DD major version (e.g., 3 or 4)

            Returns:
                Formatted list of error fields with their types.
            """
            tools = _get_imas_tools()
            result = _run_async(
                tools.fetch_error_fields(path=path, dd_version=dd_version)
            )
            return _format_error_fields_report(result)

        @self.mcp.tool()
        def list_imas_paths(
            paths: str,
            leaf_only: bool = False,
            max_paths: int | None = None,
            dd_version: int | None = None,
        ) -> str:
            """List data paths within an IMAS IDS or subtree.

            Enumerates all data paths under the given IDS name(s) or
            path prefix(es). Error fields and metadata subtrees are
            excluded. Use leaf_only=True to get only data endpoints
            (excluding structures). Error fields are accessible via
            HAS_ERROR relationships from their parent data paths.

            Args:
                paths: Space-separated IDS names or path prefixes
                    (e.g., "equilibrium" or "equilibrium/time_slice")
                leaf_only: If true, return only leaf nodes (data fields)
                max_paths: Maximum paths per query (None for all)
                dd_version: Filter by DD major version (e.g., 3 or 4)

            Returns:
                Formatted path listing report.
            """
            tools = _get_imas_tools()
            result = _run_async(
                tools.list_imas_paths(
                    paths=paths,
                    leaf_only=leaf_only,
                    max_paths=max_paths,
                    dd_version=dd_version,
                )
            )
            return format_list_report(result)

        @self.mcp.tool()
        def get_imas_overview(
            query: str | None = None,
            dd_version: int | None = None,
        ) -> str:
            """Get overview of available IMAS IDS with statistics and physics domains.

            Returns a summary of all Interface Data Structures including
            descriptions, path counts, and physics domain classifications.

            Args:
                query: Optional filter keyword (e.g., "magnetics")
                dd_version: Filter by DD major version (e.g., 3 or 4)

            Returns:
                Formatted overview report with IDS statistics.
            """
            tools = _get_imas_tools()
            result = _run_async(
                tools.get_imas_overview(query=query, dd_version=dd_version)
            )
            return format_overview_report(result)

        @self.mcp.tool()
        def get_imas_identifiers(
            query: str | None = None,
            dd_version: int | None = None,
        ) -> str:
            """Browse IMAS identifier/enumeration schemas.

            Returns available identifier schemas (coordinate systems,
            grid types, probe types, etc.) with their valid options.

            Args:
                query: Optional filter (e.g., "coordinate" or "magnetics")
                dd_version: Filter by DD major version (e.g., 3 or 4)

            Returns:
                Formatted identifiers report with schemas and options.
            """
            tools = _get_imas_tools()
            result = _run_async(
                tools.get_imas_identifiers(query=query, dd_version=dd_version)
            )
            return format_identifiers_report(result)

        @self.mcp.tool()
        def search_imas_clusters(
            query: str | None = None,
            scope: str | None = None,
            ids_filter: str | None = None,
            section_only: bool = False,
            dd_version: int | None = None,
        ) -> str:
            """Search semantic clusters of related IMAS data paths.

            Finds groups of semantically related paths across IDS
            boundaries. Can search by natural language, find clusters
            containing a specific path, or list all clusters for an IDS.

            Args:
                query: Natural language description or exact IMAS path
                    (e.g., "boundary geometry" or "equilibrium/time_slice/boundary/outline/r").
                    Optional when ids_filter is provided (listing mode).
                scope: Filter by cluster scope: "global", "domain", or "ids"
                ids_filter: Limit to clusters from specific IDS
                section_only: If true, only return clusters containing structural sections
                dd_version: Filter by DD major version (e.g., 3 or 4)

            Returns:
                Formatted cluster report with paths and descriptions.
            """
            from imas_codex.models.error_models import ToolError

            tools = _get_imas_tools()
            result = _run_async(
                tools.clusters_tool.search_imas_clusters(
                    query=query,
                    scope=scope,
                    ids_filter=ids_filter,
                    section_only=section_only,
                    dd_version=dd_version,
                )
            )
            if isinstance(result, ToolError):
                return format_cluster_report(result)
            return format_cluster_report(result)

        from imas_codex.llm.search_formatters import (
            format_export_domain_report,
            format_export_ids_report,
            format_path_context_report,
            format_structure_report,
        )

        @self.mcp.tool()
        def find_related_imas_paths(
            path: str,
            relationship_types: str = "all",
            max_results: int = 20,
            dd_version: int | None = None,
        ) -> str:
            """Find IMAS paths related to a given path across different IDSs.

            Discovers cross-IDS relationships by combining vector embedding
            similarity, cluster membership, physics coordinate sharing,
            unit+domain affinity, and identifier schemas. Produces focused,
            noise-free results by filtering generic coordinate tokens
            (e.g. '1...N') that would otherwise match thousands of paths.

            Use this for: "What other IMAS paths measure the same quantity?"

            Args:
                path: Exact IMAS path (e.g. 'equilibrium/time_slice/profiles_1d/psi')
                relationship_types: Filter to 'semantic', 'cluster', 'coordinate',
                    'unit', 'identifier', or 'all' (default)
                max_results: Maximum results per section (default 20)
                dd_version: Filter by DD major version (e.g., 3 or 4)

            Returns:
                Formatted report with cross-IDS connections grouped by signal type.
            """
            tools = _get_imas_tools()
            result = _run_async(
                tools.get_imas_path_context(
                    path=path,
                    relationship_types=relationship_types,
                    max_results=max_results,
                    dd_version=dd_version,
                )
            )
            return format_path_context_report(result)

        @self.mcp.tool()
        def analyze_imas_structure(
            ids_name: str,
            dd_version: int | None = None,
        ) -> str:
            """Analyze the hierarchical structure of an IMAS IDS.

            Returns depth metrics, leaf/structure ratio, array patterns,
            physics domain distribution, coordinate usage, and COCOS fields.

            Args:
                ids_name: IDS name (e.g. 'equilibrium')
                dd_version: Filter by DD major version (e.g., 3 or 4)

            Returns:
                Formatted structural analysis report.
            """
            tools = _get_imas_tools()
            result = _run_async(
                tools.analyze_imas_structure(ids_name=ids_name, dd_version=dd_version)
            )
            return format_structure_report(result)

        @self.mcp.tool()
        def export_imas_ids(
            ids_name: str,
            leaf_only: bool = False,
            dd_version: int | None = None,
        ) -> str:
            """Export full IDS structure with documentation, units, and types.

            Returns all paths in an IDS with their complete metadata
            including units, coordinates, clusters, and COCOS labels.

            Args:
                ids_name: IDS name (e.g. 'equilibrium')
                leaf_only: If true, return only leaf nodes (default false)
                dd_version: Filter by DD major version (e.g., 3 or 4)

            Returns:
                Full IDS path listing with documentation.
            """
            tools = _get_imas_tools()
            result = _run_async(
                tools.export_imas_ids(
                    ids_name=ids_name, leaf_only=leaf_only, dd_version=dd_version
                )
            )
            return format_export_ids_report(result)

        @self.mcp.tool()
        def export_imas_domain(
            domain: str,
            ids_filter: str | None = None,
            dd_version: int | None = None,
        ) -> str:
            """Export all IMAS paths in a physics domain, grouped by IDS.

            Lists every path classified under the given physics domain,
            with documentation and units, organized by IDS.

            Args:
                domain: Physics domain name (e.g. 'magnetics', 'equilibrium')
                ids_filter: Optional IDS name filter
                dd_version: Filter by DD major version (e.g., 3 or 4)

            Returns:
                Domain export report grouped by IDS.
            """
            tools = _get_imas_tools()
            result = _run_async(
                tools.export_imas_domain(
                    domain=domain, ids_filter=ids_filter, dd_version=dd_version
                )
            )
            return format_export_domain_report(result)

        @self.mcp.tool()
        def get_dd_version_context(
            paths: str,
        ) -> str:
            """Get version change history for specific IMAS paths.

            Returns notable changes across DD versions for each path,
            including sign_convention, coordinate_convention, units,
            and definition_clarification changes.

            Args:
                paths: Space or comma-delimited IMAS paths to check

            Returns:
                Formatted version context report per path.
            """
            tools = _get_imas_tools()
            result = _run_async(tools.get_dd_version_context(paths=paths))
            return _format_version_context_report(result)

        @self.mcp.tool()
        def get_dd_versions() -> str:
            """Get Data Dictionary version metadata from the graph.

            Returns current version, available version range, version count,
            and the ordered version chain.
            """
            tools = _get_imas_tools()
            result = _run_async(tools.get_dd_versions())
            return _format_dd_versions_report(result)

        if not self.dd_only:

            @self.mcp.tool()
            def fetch_facility_resource(resource: str) -> str:
                """Fetch the full content of a facility resource by ID or URL.

                Use after search_docs, search_code, or search_signals
                identifies a resource of interest. Returns all chunks
                or content for the resource.

                Supported types: WikiPage, Document, CodeFile, Image.

                The resource parameter can be:
                - A graph node ID from search results (e.g. "jet:Fishbone_proposal_2018.ppt")
                - A URL (e.g. "https://wiki.jetdata.eu/tf/...")
                - A partial title for fuzzy matching

                Args:
                    resource: Node ID, URL, or title substring to fetch.

                Returns:
                    Full content report with all chunks in reading order,
                    or image description/OCR text for images.
                """
                return _fetch(resource)

        if not self.read_only:
            # =====================================================================
            # Log Tools (Phase 3: MCP Logs)
            # =====================================================================

            @self.mcp.tool()
            def list_logs() -> str:
                """List available imas-codex log files with sizes and last-modified times.

                Returns a formatted table of log files including file name, size,
                age, and last modified timestamp. Automatically reads from the
                configured log location (local or remote via SSH).

                Returns:
                    Formatted log file listing.
                """
                from imas_codex.cli.logging import list_log_files

                files = list_log_files()
                if not files:
                    return "No log files found in ~/.local/share/imas-codex/logs/"

                lines = ["Available log files:", ""]
                for f in files:
                    size = f["size_bytes"]
                    if size >= 1_000_000:
                        size_str = f"{size / 1_000_000:.1f}MB"
                    elif size >= 1_000:
                        size_str = f"{size / 1_000:.1f}KB"
                    else:
                        size_str = f"{size}B"
                    lines.append(
                        f"  {f['name']:<40} {size_str:>8}  "
                        f"modified {f['modified_iso']}  ({f['age_hours']:.1f}h ago)"
                    )
                return "\n".join(lines)

            @self.mcp.tool()
            def get_logs(
                command: str = "signals",
                facility: str | None = None,
                lines: int = 100,
                level: str = "WARNING",
                grep: str | None = None,
                since: str | None = None,
            ) -> str:
                """Read imas-codex log files with filtering.

                Reads <command>_<facility>.log with level, text, and time filtering.
                Automatically reads from the configured log location (local or
                remote via SSH based on [logs].location in pyproject.toml).

                Args:
                    command: CLI command name (e.g. "signals", "wiki", "paths",
                        "code", "documents").
                    facility: Facility ID (e.g. "jet", "tcv"). If omitted,
                        reads <command>.log.
                    lines: Maximum number of matching lines to return (default: 100).
                    level: Minimum log level to include. One of: DEBUG, INFO,
                        WARNING, ERROR, CRITICAL (default: WARNING).
                    grep: Case-insensitive text filter. Only lines containing
                        this substring are returned.
                    since: Time filter. Relative: "1h", "30m", "2d".
                        Absolute: "2024-03-13T10:00".

                Returns:
                    Filtered log content. Empty if no matching lines.
                """
                from imas_codex.cli.logging import read_log

                return read_log(
                    command=command,
                    facility=facility,
                    lines=lines,
                    level=level,
                    grep=grep,
                    since=since,
                )

            @self.mcp.tool()
            def tail_logs(
                command: str = "signals",
                facility: str | None = None,
                lines: int = 50,
            ) -> str:
                """Get the most recent log entries (tail -n).

                Returns the last N lines from the log file, regardless of
                level or content. Automatically reads from the configured
                log location (local or remote via SSH).

                Args:
                    command: CLI command name (e.g. "signals", "wiki", "paths").
                    facility: Facility ID (e.g. "jet", "tcv").
                    lines: Number of lines from the end (default: 50).

                Returns:
                    Last N lines of the log file.
                """
                from imas_codex.cli.logging import tail_log

                return tail_log(
                    command=command,
                    facility=facility,
                    lines=lines,
                )

    def _register_prompts(self):
        """Register MCP prompts from markdown files.

        Static prompts: Return content as-is with includes resolved.
        Dynamic prompts (dynamic: true in frontmatter): Accept parameters
        and render with Jinja2 + schema context.
        """
        from imas_codex.llm.prompt_loader import render_prompt

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

    def _register_health_check(self):
        """Register /health endpoint with graph metadata and uptime."""
        import importlib.metadata
        import time

        from starlette.requests import Request
        from starlette.responses import JSONResponse

        server = self

        def _get_version() -> str:
            try:
                return importlib.metadata.version("imas-codex")
            except Exception:
                return "unknown"

        def _format_uptime(seconds: float) -> str:
            if seconds < 0:
                seconds = 0
            remainder = int(seconds)
            days, remainder = divmod(remainder, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, secs = divmod(remainder, 60)
            parts: list[str] = []
            if days:
                parts.append(f"{days}d")
            if hours or days:
                parts.append(f"{hours}h")
            if minutes or hours or days:
                parts.append(f"{minutes}m")
            parts.append(f"{secs}s")
            return " ".join(parts)

        def _query_graph() -> dict:
            """Query graph for IMAS DD and facility metadata."""
            try:
                from imas_codex.graph.client import GraphClient

                gc = GraphClient.from_profile()
                result: dict = {}

                # Graph meta (name, facilities, imas flag)
                try:
                    from imas_codex.graph.meta import get_graph_meta

                    meta = get_graph_meta(gc)
                    if meta:
                        result["graph_name"] = meta.get("name")
                        result["facilities"] = meta.get("facilities") or []
                        result["imas"] = meta.get("imas", False)
                except Exception as exc:
                    logger.warning("Health: graph meta query failed: %s", exc)

                # Node and relationship counts
                try:
                    stats = gc.get_stats()
                    result["node_count"] = stats.get("nodes", 0)
                    result["relationship_count"] = stats.get("relationships", 0)
                except Exception as exc:
                    logger.warning("Health: graph stats query failed: %s", exc)

                # IMAS DD version info
                try:
                    dd_rows = gc.query(
                        "MATCH (d:DDVersion) "
                        "RETURN d.id AS version, d.is_current AS is_current "
                        "ORDER BY d.id"
                    )
                    if dd_rows:
                        versions = [r["version"] for r in dd_rows]
                        current = [r["version"] for r in dd_rows if r.get("is_current")]
                        result["imas_dd"] = {
                            "version": current[0] if current else versions[-1],
                            "min_version": versions[0],
                            "version_count": len(versions),
                        }

                    # IDS and path counts
                    imas_rows = gc.query(
                        "MATCH (p:IMASNode) RETURN count(p) AS paths, "
                        "count(DISTINCT p.ids) AS ids_count"
                    )
                    if imas_rows:
                        result["ids_count"] = imas_rows[0].get("ids_count", 0)
                        result["path_count"] = imas_rows[0].get("paths", 0)
                except Exception:
                    pass

                gc.close()
                return result
            except Exception as exc:
                return {"error": str(exc)}

        @self.mcp.custom_route("/health", methods=["GET"])
        async def health_check(request: Request) -> JSONResponse:
            uptime_seconds = time.monotonic() - server._started_at

            response: dict = {
                "status": "ok",
                "version": _get_version(),
                "uptime": _format_uptime(uptime_seconds),
                "uptime_seconds": round(uptime_seconds, 1),
            }

            graph = _query_graph()
            if "error" in graph:
                response["graph"] = {"status": "unavailable", "error": graph["error"]}
            else:
                response["graph"] = {
                    "status": "ok",
                    "name": graph.get("graph_name"),
                    "node_count": graph.get("node_count"),
                    "relationship_count": graph.get("relationship_count"),
                }
                if graph.get("imas_dd"):
                    response["imas_dd"] = graph["imas_dd"]
                    response["imas_dd"]["ids_count"] = graph.get("ids_count", 0)
                    response["imas_dd"]["path_count"] = graph.get("path_count", 0)
                response["facilities"] = graph.get("facilities", [])
                response["dd_only"] = server.dd_only

            return JSONResponse(response)

    def run(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        """Run the agents server."""
        if transport == "stdio":
            logger.debug("Starting Agents server with stdio transport")
            self.mcp.run(transport=transport)
        else:
            logger.info(f"Starting Agents server on {host}:{port}")
            # Pass host/port as transport_kwargs — FastMCP.run() forwards
            # them to run_http_async() which passes them to uvicorn.
            self.mcp.run(transport=transport, host=host, port=port)
