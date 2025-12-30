"""
Agents MCP Server - Tools for LLM-driven facility exploration.

This server provides MCP tools for:
- Executing Cypher queries (READ ONLY - mutations blocked)
- Ingesting validated nodes to the knowledge graph (private fields filtered)
- Reading/updating sensitive private facility files
- All IMAS DD tools (search, fetch, list, overview, etc.)

Local use only - provides read access to graph, write via ingest_nodes only.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import anyio
from fastmcp import Context, FastMCP
from neo4j.exceptions import ServiceUnavailable
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

# Neo4j connection error message
NEO4J_NOT_RUNNING_MSG = (
    "Neo4j is not running. Start it with: uv run imas-codex neo4j start"
)


def _neo4j_error_message(e: Exception) -> str:
    """Format Neo4j errors with helpful instructions."""
    if isinstance(e, ServiceUnavailable):
        return NEO4J_NOT_RUNNING_MSG
    # Check for connection refused in the error chain
    if "Connection refused" in str(e) or "ServiceUnavailable" in str(e):
        return NEO4J_NOT_RUNNING_MSG
    return str(e)


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
    - ingest_nodes: Schema-validated node creation (single or batch)
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

            All write operations are blocked. Use ingest_nodes() for writes,
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
                    "Cypher mutations are blocked. Use ingest_nodes() for writes - "
                    "it validates data and filters private fields automatically."
                )
                raise ValueError(msg)

            try:
                with GraphClient() as client:
                    return client.query(query, **(params or {}))
            except Exception as e:
                logger.exception("Cypher query failed")
                raise RuntimeError(
                    f"Cypher query failed: {_neo4j_error_message(e)}"
                ) from e

        @self.mcp.tool()
        def ingest_nodes(
            node_type: str,
            data: dict[str, Any] | list[dict[str, Any]],
            create_facility_relationship: bool = True,
            batch_size: int = 50,
        ) -> dict[str, Any]:
            """
            Ingest nodes with schema validation and privacy filtering.

            Validates data against the Pydantic model, FILTERS OUT private
            fields (is_private: true in schema), then writes to the graph.

            Private fields are automatically excluded to prevent sensitive
            data from entering the graph or OCI artifacts.

            ALWAYS pass a list of dicts for batch ingestion, even for single
            items. Batch mode uses UNWIND for efficient Neo4j operations and
            supports partial success - valid items are ingested even if
            some items fail validation.

            Args:
                node_type: Node label (must be valid LinkML class)
                data: List of property dicts matching the Pydantic model
                create_facility_relationship: Auto-create FACILITY_ID relationship
                batch_size: Number of nodes per UNWIND batch (default: 50)

            Returns:
                Dict with counts: {"processed": N, "skipped": K, "errors": [...]}

            Examples:
                # Add diagnostics (always use list)
                ingest_nodes("Diagnostic", [
                    {"name": "XRCS", "facility_id": "epfl", "category": "spectroscopy"},
                    {"name": "Thomson", "facility_id": "epfl", "category": "spectroscopy"},
                ])

                # Add multiple paths in one call
                ingest_nodes("FacilityPath", [
                    {"id": "epfl:/home/codes", "path": "/home/codes", "facility_id": "epfl"},
                    {"id": "epfl:/home/anasrv", "path": "/home/anasrv", "facility_id": "epfl"},
                ])
            """
            schema = get_schema()

            # Validate node type
            if node_type not in schema.node_labels:
                msg = f"Unknown node type: {node_type}. Valid: {schema.node_labels}"
                raise ValueError(msg)

            # Normalize to list
            items = [data] if isinstance(data, dict) else data
            if not items:
                return {"processed": 0, "skipped": 0, "errors": []}

            # Get schema info once
            private_slots = set(schema.get_private_slots(node_type))
            model_class = schema.get_model(node_type)
            id_field = schema.get_identifier(node_type)
            if not id_field:
                msg = f"No identifier field found for {node_type}"
                raise ValueError(msg)

            # Validate all items, collecting valid ones and errors
            valid_items: list[dict[str, Any]] = []
            errors: list[str] = []

            for i, item in enumerate(items):
                try:
                    # Filter out private fields
                    filtered = {k: v for k, v in item.items() if k not in private_slots}

                    # Log if private fields were stripped
                    stripped = set(item.keys()) & private_slots
                    if stripped:
                        logger.info(
                            f"Filtered private fields from {node_type}: {stripped}"
                        )

                    # Validate against Pydantic model
                    validated = model_class.model_validate(filtered)
                    props = to_cypher_props(validated)
                    valid_items.append(props)

                except Exception as e:
                    item_id = item.get(id_field, f"item[{i}]")
                    errors.append(f"{item_id}: {e}")
                    logger.warning(f"Validation failed for {node_type} {item_id}: {e}")

            if not valid_items:
                return {"processed": 0, "skipped": len(errors), "errors": errors}

            # Batch write valid items
            try:
                with GraphClient() as client:
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

                return {
                    "processed": result["processed"],
                    "skipped": len(errors),
                    "errors": errors,
                }

            except Exception as e:
                logger.exception("Failed to ingest nodes")
                raise RuntimeError(
                    f"Failed to ingest nodes: {_neo4j_error_message(e)}"
                ) from e

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
                    "mutations": "Cypher mutations are blocked - use ingest_nodes() for writes",
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
                result["graph_error"] = _neo4j_error_message(e)

            return result

        # =====================================================================
        # Code Example Tools
        # =====================================================================

        @self.mcp.tool()
        async def ingest_code_files(
            facility: str,
            remote_paths: list[str],
            ctx: Context,
            description: str | None = None,
        ) -> dict[str, int]:
            """
            Ingest code files from a remote facility into the knowledge graph.

            Fetches files via SCP, chunks them into searchable segments,
            generates embeddings, extracts IMAS IDS references, and stores
            in Neo4j with relationships to facility and IMAS paths.

            Progress is reported during ingestion so you can monitor long
            operations. Each file fetch, chunk generation, and graph write
            step is reported.

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
            # Track last reported progress to send updates via Context
            last_message: list[str] = [""]

            def progress_callback(current: int, total: int, message: str) -> None:
                """Synchronous callback that stores progress for async reporting."""
                last_message[0] = message

            async def run_ingestion() -> dict[str, int]:
                """Run the blocking ingestion in a thread."""
                ingester = CodeExampleIngester(progress_callback=progress_callback)
                return await anyio.to_thread.run_sync(
                    lambda: ingester.ingest_files(facility, remote_paths, description)
                )

            try:
                await ctx.info(
                    f"Starting ingestion of {len(remote_paths)} files from {facility}"
                )

                # Run ingestion with progress reporting
                result = await run_ingestion()

                await ctx.info(
                    f"Completed: {result['files']} files, "
                    f"{result['chunks']} chunks, {result['ids_found']} IDS refs"
                )
                return result
            except Exception as e:
                logger.exception("Failed to ingest code files")
                error_msg = _neo4j_error_message(e)
                await ctx.error(f"Ingestion failed: {error_msg}")
                raise RuntimeError(f"Failed to ingest code files: {error_msg}") from e

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
                with searcher.graph_client:
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
            They are also registered as MCP prompts (accessible via /prompt-name).

            Returns:
                List of prompt summaries with name, description, and arguments.

            Examples:
                prompts = list_prompts()
                # Returns: [
                #   {"name": "scout-paths", "description": "Discover directories...", ...},
                #   {"name": "scout-code", "description": "Find and ingest...", ...},
                #   ...
                # ]
            """
            return list_prompts_summary()

        @self.mcp.tool()
        def get_exploration_progress(facility: str) -> dict[str, Any]:
            """
            Get exploration progress metrics for a facility.

            Calculates completion metrics based on FacilityPath status distribution
            and MDSplus tree/TDI function coverage.

            Args:
                facility: Facility ID (e.g., "epfl")

            Returns:
                Dict with total_paths, actionable, processed, completion_pct, by_status,
                mdsplus_coverage (per-tree stats), tdi_coverage
            """
            try:
                with GraphClient() as client:
                    # Path exploration progress
                    path_rows = client.query(
                        """
                        MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $fid})
                        RETURN p.status AS status, count(*) AS count
                        ORDER BY status
                        """,
                        fid=facility,
                    )

                    # MDSplus tree coverage (per-tree)
                    tree_rows = client.query(
                        """
                        MATCH (t:MDSplusTree)-[:FACILITY_ID]->(f:Facility {id: $fid})
                        OPTIONAL MATCH (n:TreeNode)-[:MDSPLUS_TREE]->(t)
                        OPTIONAL MATCH (tdi:TDIFunction)-[:FACILITY_ID]->(f)
                        WHERE tdi.physics_domain IS NOT NULL
                        RETURN t.name AS tree,
                               t.ingestion_status AS status,
                               t.node_count_total AS total_nodes,
                               t.node_count_ingested AS ingested_nodes,
                               t.population_type AS population_type,
                               count(DISTINCT n) AS nodes_in_graph,
                               t.last_ingested AS last_ingested
                        ORDER BY t.name
                        """,
                        fid=facility,
                    )

                    # TDI function coverage
                    tdi_rows = client.query(
                        """
                        MATCH (t:TDIFunction)-[:FACILITY_ID]->(f:Facility {id: $fid})
                        RETURN t.physics_domain AS domain,
                               count(*) AS count,
                               sum(CASE WHEN t.version IS NOT NULL THEN 1 ELSE 0 END) AS with_version
                        ORDER BY count DESC
                        """,
                        fid=facility,
                    )

                    # Analysis code coverage
                    code_rows = client.query(
                        """
                        MATCH (a:AnalysisCode)-[:FACILITY_ID]->(f:Facility {id: $fid})
                        RETURN a.code_type AS type,
                               count(*) AS count,
                               sum(CASE WHEN a.writes_to_tree IS NOT NULL THEN 1 ELSE 0 END) AS with_tree_link
                        ORDER BY count DESC
                        """,
                        fid=facility,
                    )

                counts = {r["status"]: r["count"] for r in path_rows}
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

                # MDSplus tree coverage summary
                mdsplus_coverage = {}
                for row in tree_rows:
                    tree_name = row["tree"]
                    total_nodes = row["total_nodes"] or 0
                    ingested = row["ingested_nodes"] or row["nodes_in_graph"] or 0
                    mdsplus_coverage[tree_name] = {
                        "status": row["status"] or "pending",
                        "population_type": row["population_type"],
                        "total_nodes": total_nodes,
                        "ingested_nodes": ingested,
                        "coverage_pct": round(100 * ingested / total_nodes, 1)
                        if total_nodes
                        else 0.0,
                        "last_ingested": row["last_ingested"],
                    }

                # TDI function coverage by domain
                tdi_coverage = {
                    "total": sum(r["count"] for r in tdi_rows),
                    "with_version": sum(r["with_version"] for r in tdi_rows),
                    "by_domain": {
                        r["domain"] or "unclassified": r["count"] for r in tdi_rows
                    },
                }

                # Analysis code coverage
                code_coverage = {
                    "total": sum(r["count"] for r in code_rows),
                    "with_tree_link": sum(r["with_tree_link"] for r in code_rows),
                    "by_type": {
                        r["type"] or "unclassified": r["count"] for r in code_rows
                    },
                }

                # Recommend next action
                if total == 0:
                    recommendation = "Run /scout-paths to discover paths"
                elif actionable > processed:
                    recommendation = "Run /score-paths to score discovered paths"
                elif counts.get("flagged", 0) > counts.get("ingested", 0):
                    recommendation = "Run /scout-code to ingest flagged paths"
                elif tdi_coverage["total"] == 0:
                    recommendation = (
                        "Run /ingest-tdi-functions to discover TDI functions"
                    )
                elif any(t["status"] == "pending" for t in mdsplus_coverage.values()):
                    recommendation = "Run /ingest-tree for pending trees"
                else:
                    recommendation = "Increase depth or explore new root paths"

                return {
                    "facility": facility,
                    "paths": {
                        "total": total,
                        "actionable": actionable,
                        "processed": processed,
                        "skipped": skipped,
                        "completion_pct": completion_pct,
                        "by_status": counts,
                    },
                    "mdsplus_coverage": mdsplus_coverage,
                    "tdi_coverage": tdi_coverage,
                    "code_coverage": code_coverage,
                    "recommendation": recommendation,
                }
            except Exception as e:
                logger.exception("Failed to get exploration progress")
                raise RuntimeError(
                    f"Failed to get exploration progress: {_neo4j_error_message(e)}"
                ) from e

    def _register_prompts(self):
        """Register MCP prompts from markdown files.

        Creates typed wrapper functions for each prompt so FastMCP can
        expose the arguments properly. Each argument gets a type annotation
        and default value from the prompt definition.
        """
        for name, prompt_def in self._prompts.items():
            self._register_single_prompt(name, prompt_def)
            logger.debug(f"Registered prompt: {name}")

    def _register_single_prompt(self, name: str, prompt_def: PromptDefinition):
        """Register a single prompt with proper argument handling.

        FastMCP prompts need typed function signatures (no **kwargs).
        We dynamically create a function with explicit parameters.
        """
        # Build argument documentation for the description
        arg_docs = []
        for arg in prompt_def.arguments:
            req = " (required)" if arg.required else ""
            default = f" [default: {arg.default}]" if arg.default is not None else ""
            arg_docs.append(f"  - {arg.name}: {arg.description}{req}{default}")

        enhanced_desc = prompt_def.description
        if arg_docs:
            enhanced_desc += "\n\nArguments:\n" + "\n".join(arg_docs)

        # For prompts with no arguments, use a simple function
        if not prompt_def.arguments:

            @self.mcp.prompt(name=name, description=enhanced_desc)
            def no_arg_prompt(_prompt_def: PromptDefinition = prompt_def) -> str:
                return _prompt_def.render()

            return

        # For prompts with arguments, dynamically build a function signature
        # FastMCP doesn't support **kwargs, so we need explicit parameters
        # All our prompts have defaults, so we can use simple typed params
        param_parts = []
        for arg in prompt_def.arguments:
            # Map YAML types to Python types
            py_type = {"string": "str", "integer": "int", "boolean": "bool"}.get(
                arg.type, "str"
            )
            default_repr = repr(arg.default) if arg.default is not None else '""'
            param_parts.append(f"{arg.name}: {py_type} = {default_repr}")

        params_str = ", ".join(param_parts)
        arg_names = [arg.name for arg in prompt_def.arguments]
        kwargs_str = ", ".join(f"{n}={n}" for n in arg_names)

        # Create the function dynamically
        func_code = f"""
def prompt_fn({params_str}) -> str:
    return _prompt_def.render({kwargs_str})
"""
        local_ns: dict[str, Any] = {"_prompt_def": prompt_def}
        exec(func_code, local_ns)  # noqa: S102
        prompt_fn = local_ns["prompt_fn"]
        prompt_fn.__name__ = name.replace("-", "_")
        prompt_fn.__doc__ = enhanced_desc

        # Register with FastMCP
        self.mcp.prompt(name=name, description=enhanced_desc)(prompt_fn)

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
