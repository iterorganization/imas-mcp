"""
Agents MCP Server - Tools for LLM-driven facility exploration.

This server provides MCP tools for:
- Executing Cypher queries (mutations restricted to _Discovery nodes)
- Ingesting validated nodes to the knowledge graph
- Reading/updating sensitive infrastructure files
- All IMAS DD tools (search, fetch, list, overview, etc.)

Local use only - provides full graph access for trusted agents.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml
from fastmcp import FastMCP
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from imas_codex.code_examples import CodeExampleIngester, CodeExampleSearch
from imas_codex.discovery import get_config
from imas_codex.discovery.config import get_facilities_dir
from imas_codex.graph import GraphClient, get_schema
from imas_codex.graph.schema import to_cypher_props
from imas_codex.tools import Tools

logger = logging.getLogger(__name__)

# Configure ruamel.yaml for comment-preserving round-trips
_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 120


def _deep_merge_ruamel(base: CommentedMap, updates: dict[str, Any]) -> CommentedMap:
    """Deep merge updates into ruamel CommentedMap, preserving comments."""
    for key, value in updates.items():
        if key in base:
            if isinstance(base[key], CommentedMap) and isinstance(value, dict):
                _deep_merge_ruamel(base[key], value)
            elif isinstance(base[key], list | CommentedSeq) and isinstance(value, list):
                # Extend lists, avoiding duplicates
                for item in value:
                    if item not in base[key]:
                        base[key].append(item)
            else:
                base[key] = value
        else:
            base[key] = value
    return base


@dataclass
class AgentsServer:
    """
    MCP server for facility exploration and knowledge graph management.

    Provides tools for:
    - cypher: Execute Cypher queries (read any, write only _Discovery)
    - ingest_node: Schema-validated node creation
    - read_infrastructure: Read sensitive infrastructure files
    - update_infrastructure: Merge updates to infrastructure files
    - get_graph_schema: Get complete schema for Cypher generation
    - get_facilities: List available facilities with SSH hosts
    - All IMAS DD tools (search_imas_paths, fetch_imas_paths, etc.)
    """

    include_imas_tools: bool = True
    mcp: FastMCP = field(init=False, repr=False)
    imas_tools: Tools | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Initialize the MCP server."""
        self.mcp = FastMCP(name="imas-codex-agents")
        self._register_tools()

        # Include IMAS DD tools by default
        if self.include_imas_tools:
            self.imas_tools = Tools()
            self.imas_tools.register(self.mcp)
            logger.debug("IMAS DD tools registered with agents server")

        logger.debug("Agents MCP server initialized")

    def _register_tools(self):
        """Register exploration and graph tools."""

        @self.mcp.tool()
        def cypher(query: str, params: dict[str, Any] | None = None) -> list[dict]:
            """
            Execute Cypher query against the knowledge graph.

            Read queries: unrestricted access to all nodes
            Write queries: restricted to _Discovery nodes only

            For schema-validated writes to proper nodes, use ingest_node().

            Args:
                query: Cypher query string
                params: Optional query parameters

            Returns:
                List of result records as dicts

            Examples:
                # Read query - explore the graph
                cypher("MATCH (f:Facility)-[:FACILITY_ID]-(d:Diagnostic) RETURN f.id, d.name")

                # Write to _Discovery - stage unstructured findings
                cypher('''
                    CREATE (d:_Discovery {
                        facility: 'epfl',
                        type: 'unknown_tree',
                        name: 'tcv_raw',
                        notes: 'Found on spcsrv1',
                        discovered_at: datetime()
                    })
                ''')
            """
            upper = query.upper()
            mutation_keywords = ["CREATE", "MERGE", "SET", "DELETE", "REMOVE"]
            has_mutation = any(kw in upper for kw in mutation_keywords)

            if has_mutation and "_DISCOVERY" not in upper:
                msg = (
                    "Write operations restricted to _Discovery nodes. "
                    "Use ingest_node() for schema-validated writes to proper nodes, "
                    "or write to _Discovery for staging unstructured findings."
                )
                raise ValueError(msg)

            try:
                with GraphClient() as client:
                    return client.query(query, **(params or {}))
            except Exception as e:
                logger.exception("Cypher query failed")
                raise RuntimeError(f"Cypher query failed: {e}") from e

        @self.mcp.tool()
        def ingest_node(
            node_type: str,
            data: dict[str, Any],
            create_facility_relationship: bool = True,
        ) -> str:
            """
            Ingest a node with schema validation.

            Validates data against the Pydantic model for the node type,
            then creates/updates the node in the graph.

            Args:
                node_type: Node label (must be valid LinkML class)
                data: Properties matching the Pydantic model
                create_facility_relationship: Auto-create FACILITY_ID relationship

            Returns:
                Success message with node identifier

            Examples:
                # Add a diagnostic
                ingest_node("Diagnostic", {
                    "name": "XRCS",
                    "facility_id": "epfl",
                    "category": "spectroscopy",
                    "description": "X-ray crystal spectrometer"
                })

                # Add an MDSplus tree
                ingest_node("MDSplusTree", {
                    "name": "tcv_raw",
                    "facility_id": "epfl",
                    "description": "Raw diagnostic data"
                })
            """
            schema = get_schema()

            # Validate node type
            if node_type not in schema.node_labels:
                msg = f"Unknown node type: {node_type}. Valid: {schema.node_labels}"
                raise ValueError(msg)

            # Get and validate against Pydantic model
            model_class = schema.get_model(node_type)
            try:
                validated = model_class.model_validate(data)
            except Exception as e:
                msg = f"Validation failed for {node_type}: {e}"
                raise ValueError(msg) from e

            # Get identifier field
            id_field = schema.get_identifier(node_type)
            if not id_field:
                msg = f"No identifier field found for {node_type}"
                raise ValueError(msg)

            node_id = getattr(validated, id_field)
            props = to_cypher_props(validated)

            try:
                with GraphClient() as client:
                    # Create/update the node
                    client.create_node(node_type, node_id, props, id_field=id_field)

                    # Create facility relationship if applicable
                    if (
                        create_facility_relationship
                        and "facility_id" in props
                        and node_type != "Facility"
                    ):
                        client.create_relationship(
                            node_type,
                            node_id,
                            "Facility",
                            props["facility_id"],
                            "FACILITY_ID",
                            from_id_field=id_field,
                        )

                return f"Created/updated {node_type} node: {node_id}"
            except Exception as e:
                logger.exception("Failed to ingest node")
                raise RuntimeError(f"Failed to ingest node: {e}") from e

        @self.mcp.tool()
        def read_infrastructure(facility: str) -> dict[str, Any] | None:
            """
            Read sensitive infrastructure data for a facility.

            Infrastructure files contain sensitive data (OS versions, paths,
            tool availability) that is NOT stored in the public graph.

            Args:
                facility: Facility identifier (e.g., "epfl")

            Returns:
                Infrastructure data dict, or None if file doesn't exist
            """
            infra_path = get_facilities_dir() / f"{facility}_infrastructure.yaml"
            if not infra_path.exists():
                return None

            try:
                return yaml.safe_load(infra_path.read_text())
            except Exception as e:
                logger.exception(f"Failed to read infrastructure for {facility}")
                raise RuntimeError(f"Failed to read infrastructure: {e}") from e

        @self.mcp.tool()
        def update_infrastructure(
            facility: str,
            data: dict[str, Any],
        ) -> str:
            """
            Merge updates into facility infrastructure file.

            Deep-merges the provided data into the existing infrastructure
            file, preserving comments and key order.

            Args:
                facility: Facility identifier (e.g., "epfl")
                data: Data to merge (will be deep-merged with existing)

            Returns:
                Success message

            Examples:
                # Update tool availability
                update_infrastructure("epfl", {
                    "knowledge": {"tools": {"rg": "unavailable"}}
                })

                # Add exploration notes
                update_infrastructure("epfl", {
                    "knowledge": {
                        "notes": ["New discovery about MDSplus config"]
                    }
                })

                # Add new paths discovered
                update_infrastructure("epfl", {
                    "paths": {"codes": {"root": "/home/codes"}}
                })
            """
            infra_path = get_facilities_dir() / f"{facility}_infrastructure.yaml"

            try:
                if infra_path.exists():
                    # Load with ruamel to preserve comments
                    with infra_path.open() as f:
                        existing = _yaml.load(f)
                    if existing is None:
                        existing = CommentedMap({"facility_id": facility})
                else:
                    existing = CommentedMap({"facility_id": facility})

                # Deep merge preserving comments
                _deep_merge_ruamel(existing, data)

                # Write back with preserved formatting
                with infra_path.open("w") as f:
                    _yaml.dump(existing, f)

                return f"Updated infrastructure for {facility}"
            except Exception as e:
                logger.exception(f"Failed to update infrastructure for {facility}")
                raise RuntimeError(f"Failed to update infrastructure: {e}") from e

        @self.mcp.tool()
        def get_graph_schema() -> dict[str, Any]:
            """
            Get complete graph schema for Cypher query generation.

            Returns node labels with all properties, enums with valid values,
            and relationship types. Call this before writing Cypher queries.

            Returns:
                Schema dict with node_labels, enums, relationship_types, special_nodes
            """
            schema = get_schema()

            node_labels = {}
            for label in schema.node_labels:
                node_labels[label] = {
                    "identifier": schema.get_identifier(label),
                    "description": schema.get_class_description(label),
                    "properties": schema.get_all_slots(label),
                }

            return {
                "node_labels": node_labels,
                "enums": schema.get_enums(),
                "relationship_types": schema.relationship_types,
                "special_nodes": {
                    "_Discovery": "Staging area for unstructured findings (write freely)",
                    "_GraphMeta": "Graph metadata (version, facilities)",
                },
            }

        @self.mcp.tool()
        def get_facilities() -> list[dict[str, str]]:
            """
            List available facilities with SSH connection info.

            Returns:
                List of facility dicts with id, ssh_host, description
            """
            facilities_dir = get_facilities_dir()
            if not facilities_dir.exists():
                return []

            result = []
            for path in sorted(facilities_dir.glob("*.yaml")):
                # Skip infrastructure files
                if path.stem.endswith("_infrastructure"):
                    continue
                try:
                    config = get_config(path.stem)
                    result.append(
                        {
                            "id": config.facility,
                            "ssh_host": config.ssh_host,
                            "description": config.description,
                        }
                    )
                except Exception:
                    # Skip invalid configs
                    pass
            return result

        @self.mcp.tool()
        def get_facility(facility: str) -> dict[str, Any]:
            """
            Get comprehensive facility info for exploration.

            Merges graph data, infrastructure, and exploration state.
            Call this before starting exploration to understand:
            - Available tools (rg, fd, etc.)
            - Actionable paths (discovered/scanned/flagged) by interest_score
            - Recently processed paths (analyzed/ingested)
            - Known analysis codes and diagnostics

            Args:
                facility: Facility ID (e.g., "epfl")

            Returns:
                Dict with: config, infrastructure, graph_summary, exploration_state
            """
            from imas_codex.discovery.config import (
                get_config,
                get_infrastructure,
            )

            result: dict[str, Any] = {"facility": facility}

            # Load public config
            try:
                config = get_config(facility)
                result["config"] = {
                    "ssh_host": config.ssh_host,
                    "description": config.description,
                    "machine": config.machine,
                }
            except Exception as e:
                result["config_error"] = str(e)

            # Load infrastructure (tools, paths, etc.)
            try:
                infra = get_infrastructure(facility)
                if infra:
                    # Extract key agent guidance
                    result["tools"] = infra.get("knowledge", {}).get("tools", {})
                    result["paths"] = infra.get("paths", {})
                    result["notes"] = infra.get("knowledge", {}).get("notes", [])
            except Exception as e:
                result["infrastructure_error"] = str(e)

            # Query graph for facility summary
            with self.graph_client:
                # Count nodes by type
                summary = self.graph_client.query(
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

                # Get actionable paths (discovered/scanned/flagged) for exploration
                actionable = self.graph_client.query(
                    """
                    MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $fid})
                    WHERE p.status IN ['discovered', 'listed', 'scanned', 'flagged']
                    RETURN p.id AS id, p.path AS path, p.path_type AS path_type,
                           p.status AS status, p.interest_score AS interest_score,
                           p.description AS description, p.depth AS depth
                    ORDER BY COALESCE(p.interest_score, 0) DESC, p.depth, p.path
                    """,
                    fid=facility,
                )
                result["actionable_paths"] = actionable

                # Get recently processed paths
                processed = self.graph_client.query(
                    """
                    MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $fid})
                    WHERE p.status IN ['analyzed', 'ingested']
                    RETURN p.id AS id, p.path AS path, p.status AS status,
                           p.last_examined AS last_examined, p.files_ingested AS files_ingested
                    ORDER BY p.last_examined DESC
                    LIMIT 10
                    """,
                    fid=facility,
                )
                result["recent_paths"] = processed

                # Get excluded paths (so agent knows what to skip)
                excluded = self.graph_client.query(
                    """
                    MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $fid})
                    WHERE p.status = 'excluded'
                    RETURN p.path AS path, p.description AS description
                    ORDER BY p.path
                    """,
                    fid=facility,
                )
                result["excluded_paths"] = [e["path"] for e in excluded]

            return result

        # =====================================================================
        # Code Example Tools
        # =====================================================================

        @self.mcp.tool()
        def ingest_code_files(
            facility: str,
            remote_paths: list[str],
            description: str | None = None,
        ) -> dict[str, int]:
            """
            Ingest code files from a remote facility into the knowledge graph.

            Fetches files via SCP, chunks them into searchable segments,
            generates embeddings, extracts IMAS IDS references, and stores
            in Neo4j with relationships to facility and IMAS paths.

            Args:
                facility: Facility SSH host alias (e.g., "epfl")
                remote_paths: List of remote file paths to ingest
                description: Optional description for all files

            Returns:
                Dict with counts: {"files": N, "chunks": M, "ids_found": K}

            Examples:
                # Ingest a single file
                ingest_code_files("epfl", ["/home/wilson/data/load_to_imas.py"])

                # Ingest multiple files
                ingest_code_files("epfl", [
                    "/home/duval/VNC_22/equil-tools-py/liuqeplot.py",
                    "/home/athorn/python/equilibrium.py"
                ], description="Equilibrium visualization examples")
            """
            try:
                ingester = CodeExampleIngester()
                return ingester.ingest_files(facility, remote_paths, description)
            except Exception as e:
                logger.exception("Failed to ingest code files")
                raise RuntimeError(f"Failed to ingest code files: {e}") from e

        @self.mcp.tool()
        def search_code_examples(
            query: str,
            top_k: int = 10,
            ids_filter: list[str] | None = None,
            facility: str | None = None,
        ) -> list[dict[str, Any]]:
            """
            Search for code examples using semantic similarity.

            Uses vector embeddings to find code snippets matching the query,
            with optional filtering by IMAS IDS or facility.

            Args:
                query: Natural language search query (e.g., "load equilibrium from MDSplus")
                top_k: Maximum number of results to return
                ids_filter: Optional list of IDS names to filter by (e.g., ["equilibrium"])
                facility: Optional facility ID to filter by (e.g., "epfl")

            Returns:
                List of code search results with content, source file, and relevance score

            Examples:
                # Search for equilibrium loading code
                search_code_examples("load equilibrium data from MDSplus")

                # Search with IDS filter
                search_code_examples("write profiles", ids_filter=["core_profiles"])

                # Search within a specific facility
                search_code_examples("LIUQE", facility="epfl")
            """
            try:
                searcher = CodeExampleSearch()
                results = searcher.search(
                    query=query,
                    top_k=top_k,
                    ids_filter=ids_filter,
                    facility=facility,
                )
                # Convert to dicts for JSON serialization
                return [
                    {
                        "chunk_id": r.chunk_id,
                        "content": r.content,
                        "function_name": r.function_name,
                        "source_file": r.source_file,
                        "facility_id": r.facility_id,
                        "related_ids": r.related_ids,
                        "score": r.score,
                        "start_line": r.start_line,
                        "end_line": r.end_line,
                    }
                    for r in results
                ]
            except Exception as e:
                logger.exception("Failed to search code examples")
                raise RuntimeError(f"Failed to search code examples: {e}") from e

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
