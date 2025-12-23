"""
Agents MCP Server - Tools for LLM-driven facility exploration.

This server provides MCP tools for:
- Executing Cypher queries (READ ONLY - mutations blocked)
- Ingesting validated nodes to the knowledge graph (private fields filtered)
- Reading/updating sensitive private facility files
- All IMAS DD tools (search, fetch, list, overview, etc.)

Local use only - provides read access to graph, write via ingest_node only.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from fastmcp import FastMCP
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from imas_codex.agents.prompt_loader import (
    PromptDefinition,
    list_prompts_summary,
    load_prompts,
)
from imas_codex.code_examples import CodeExampleIngester, CodeExampleSearch
from imas_codex.discovery import (
    get_facility,
    get_facility_private,
    list_facilities,
    save_private,
)
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
    _prompts: dict[str, PromptDefinition] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the MCP server."""
        self.mcp = FastMCP(name="imas-codex-agents")
        self._prompts = load_prompts()
        self._register_tools()
        self._register_prompts()

        # Include IMAS DD tools by default
        if self.include_imas_tools:
            self.imas_tools = Tools()
            self.imas_tools.register(self.mcp)
            logger.debug("IMAS DD tools registered with agents server")

        logger.debug(f"Agents MCP server initialized with {len(self._prompts)} prompts")

    def _register_tools(self):
        """Register exploration and graph tools."""

        @self.mcp.tool()
        def cypher(query: str, params: dict[str, Any] | None = None) -> list[dict]:
            """
            Execute READ-ONLY Cypher query against the knowledge graph.

            All write operations are blocked. Use ingest_node() for writes,
            which automatically filters private fields before graph storage.

            Args:
                query: Cypher query string (READ ONLY)
                params: Optional query parameters

            Returns:
                List of result records as dicts

            Examples:
                # Read queries - explore the graph
                cypher("MATCH (f:Facility) RETURN f.id, f.name")
                cypher("MATCH (f:Facility)-[:FACILITY_ID]-(d:Diagnostic) RETURN f.id, d.name")
                cypher("MATCH (p:FacilityPath {status: 'flagged'}) RETURN p.path, p.interest_score")
            """
            upper = query.upper()
            mutation_keywords = ["CREATE", "MERGE", "SET", "DELETE", "REMOVE", "DETACH"]

            if any(kw in upper for kw in mutation_keywords):
                msg = (
                    "Cypher mutations are blocked. Use ingest_node() for writes - "
                    "it validates data and filters private fields automatically."
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
            Ingest a node with schema validation and privacy filtering.

            Validates data against the Pydantic model, FILTERS OUT private
            fields (is_private: true in schema), then writes to the graph.

            Private fields are automatically excluded to prevent sensitive
            data from entering the graph or OCI artifacts.

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

            # Filter out private fields BEFORE validation
            private_slots = set(schema.get_private_slots(node_type))
            filtered_data = {k: v for k, v in data.items() if k not in private_slots}

            # Log if private fields were stripped
            stripped = set(data.keys()) & private_slots
            if stripped:
                logger.info(f"Filtered private fields from {node_type}: {stripped}")

            # Get and validate against Pydantic model
            model_class = schema.get_model(node_type)
            try:
                validated = model_class.model_validate(filtered_data)
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
        def read_private(facility: str) -> dict[str, Any] | None:
            """
            Read private facility data (sensitive infrastructure info).

            Private files contain data marked is_private in the schema:
            OS versions, paths, tool availability, etc.
            This data is NEVER stored in the graph or OCI artifacts.

            Args:
                facility: Facility identifier (e.g., "epfl")

            Returns:
                Private data dict, or None if file doesn't exist
            """
            try:
                return get_facility_private(facility)
            except Exception as e:
                logger.exception(f"Failed to read private data for {facility}")
                raise RuntimeError(f"Failed to read private data: {e}") from e

        @self.mcp.tool()
        def update_private(
            facility: str,
            data: dict[str, Any],
        ) -> str:
            """
            Merge updates into facility private file.

            Deep-merges the provided data into the existing private file,
            preserving comments and key order using ruamel.yaml.

            Args:
                facility: Facility identifier (e.g., "epfl")
                data: Data to merge (will be deep-merged with existing)

            Returns:
                Success message

            Examples:
                # Update tool availability
                update_private("epfl", {
                    "tools": {"rg": "14.1.1", "fd": "10.2.0"}
                })

                # Add exploration notes
                update_private("epfl", {
                    "exploration_notes": ["Discovered new MDSplus tree"]
                })

                # Add new paths discovered
                update_private("epfl", {
                    "paths": {"codes": {"root": "/home/codes"}}
                })
            """
            try:
                save_private(facility, data)
                return f"Updated private data for {facility}"
            except Exception as e:
                logger.exception(f"Failed to update private data for {facility}")
                raise RuntimeError(f"Failed to update private data: {e}") from e

        @self.mcp.tool()
        def get_graph_schema() -> dict[str, Any]:
            """
            Get complete graph schema for Cypher query generation.

            Returns node labels with all properties, enums with valid values,
            relationship types, and private field annotations.
            Call this before writing Cypher queries.

            Returns:
                Schema dict with node_labels, enums, relationship_types, private_fields
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
                    "mutations": "Cypher mutations are blocked - use ingest_node() for writes",
                },
            }

        @self.mcp.tool()
        def get_all_facilities() -> list[dict[str, str]]:
            """
            List available facilities with SSH connection info.

            Returns:
                List of facility dicts with id, ssh_host, description
            """
            result = []
            for facility_id in list_facilities():
                try:
                    data = get_facility(facility_id)
                    result.append(
                        {
                            "id": data.get("id", facility_id),
                            "ssh_host": data.get("ssh_host", ""),
                            "description": data.get("description", ""),
                            "machine": data.get("machine", ""),
                        }
                    )
                except Exception:
                    # Skip invalid configs
                    pass
            return result

        @self.mcp.tool()
        def get_facility_info(facility: str) -> dict[str, Any]:
            """
            Get comprehensive facility info for exploration.

            Merges public config, private data, and graph state.
            Call this before starting exploration to understand:
            - Available tools (rg, fd, etc.)
            - Actionable paths (discovered/scanned/flagged) by interest_score
            - Recently processed paths (analyzed/ingested)
            - Known analysis codes and diagnostics

            Args:
                facility: Facility ID (e.g., "epfl")

            Returns:
                Dict with: config, tools, paths, graph_summary, exploration_state
            """
            result: dict[str, Any] = {"facility": facility}

            # Load merged facility data (public + private)
            try:
                data = get_facility(facility)
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
                with GraphClient() as client:
                    # Count nodes by type
                    summary = client.query(
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
                    actionable = client.query(
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
                    processed = client.query(
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
                    excluded = client.query(
                        """
                        MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $fid})
                        WHERE p.status = 'excluded'
                        RETURN p.path AS path, p.description AS description
                        ORDER BY p.path
                        """,
                        fid=facility,
                    )
                    result["excluded_paths"] = [e["path"] for e in excluded]
            except Exception as e:
                result["graph_error"] = str(e)

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

        # =====================================================================
        # Prompt Tools
        # =====================================================================

        @self.mcp.tool()
        def list_prompts() -> list[dict[str, Any]]:
            """
            List available exploration prompts with descriptions.

            Prompts are pre-defined workflows for common exploration tasks.
            Use get_prompt() to retrieve a specific prompt for execution.

            Returns:
                List of prompt summaries with name, description, and arguments.

            Examples:
                prompts = list_prompts()
                # Returns: [
                #   {"name": "scout_depth", "description": "Explore directories...", "arguments": [...]},
                #   {"name": "triage_paths", "description": "Score discovered paths...", ...},
                #   ...
                # ]
            """
            return list_prompts_summary()

        @self.mcp.tool()
        def get_prompt(name: str, **kwargs: Any) -> str:
            """
            Get an exploration prompt rendered with arguments.

            Retrieves a prompt template and renders it with the provided arguments.
            Use list_prompts() to see available prompts and their arguments.

            Args:
                name: Prompt name (e.g., "scout_depth", "triage_paths")
                **kwargs: Arguments to render into the prompt template

            Returns:
                Rendered prompt text ready for execution

            Examples:
                # Get scout_depth prompt for EPFL
                prompt = get_prompt("scout_depth", facility="epfl", depth=3)

                # Get triage_paths prompt
                prompt = get_prompt("triage_paths", facility="epfl", limit=20)
            """
            if name not in self._prompts:
                available = list(self._prompts.keys())
                msg = f"Unknown prompt: {name}. Available: {available}"
                raise ValueError(msg)

            return self._prompts[name].render(**kwargs)

        @self.mcp.tool()
        def get_exploration_progress(facility: str) -> dict[str, Any]:
            """
            Get exploration progress metrics for a facility.

            Calculates completion metrics based on FacilityPath status distribution.

            Args:
                facility: Facility ID (e.g., "epfl")

            Returns:
                Dict with total_paths, actionable, processed, completion_pct, by_status
            """
            try:
                with GraphClient() as client:
                    rows = client.query(
                        """
                        MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $fid})
                        RETURN p.status AS status, count(*) AS count
                        ORDER BY status
                        """,
                        fid=facility,
                    )

                counts = {r["status"]: r["count"] for r in rows}
                total = sum(counts.values())

                # Categorize by workflow stage
                actionable = (
                    counts.get("discovered", 0)
                    + counts.get("listed", 0)
                    + counts.get("scanned", 0)
                )
                processed = (
                    counts.get("flagged", 0)
                    + counts.get("analyzed", 0)
                    + counts.get("ingested", 0)
                )
                skipped = counts.get("skipped", 0) + counts.get("excluded", 0)

                completion_pct = (
                    round(100 * (processed + skipped) / total, 1) if total else 0.0
                )

                # Recommend next action
                if total == 0:
                    recommendation = "Run /scout_depth to discover paths"
                elif actionable > processed:
                    recommendation = "Run /triage_paths to score discovered paths"
                elif counts.get("flagged", 0) > counts.get("ingested", 0):
                    recommendation = "Run /code_hunt to ingest flagged paths"
                else:
                    recommendation = "Increase depth or explore new root paths"

                return {
                    "facility": facility,
                    "total_paths": total,
                    "actionable": actionable,
                    "processed": processed,
                    "skipped": skipped,
                    "completion_pct": completion_pct,
                    "by_status": counts,
                    "recommendation": recommendation,
                }
            except Exception as e:
                logger.exception("Failed to get exploration progress")
                raise RuntimeError(f"Failed to get exploration progress: {e}") from e

    def _register_prompts(self):
        """Register MCP prompts from markdown files."""
        for name, prompt_def in self._prompts.items():
            # Build argument list for MCP prompt
            # Note: FastMCP prompts use a simpler signature
            # We register them as callable prompts

            @self.mcp.prompt(name=name, description=prompt_def.description)
            def make_prompt_fn(prompt_def: PromptDefinition = prompt_def):
                # Return a function that takes kwargs and renders
                def prompt_fn(**kwargs: Any) -> str:
                    return prompt_def.render(**kwargs)

                return prompt_fn

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
