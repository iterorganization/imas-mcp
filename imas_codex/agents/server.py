"""
Agents MCP Server - Tools for LLM-driven facility exploration.

This server provides MCP tools for:
- Executing Cypher queries (READ ONLY - mutations blocked)
- Ingesting validated nodes to the knowledge graph (private fields filtered)
- Reading/updating sensitive private facility files

Note: IMAS DD tools are NOT included to keep startup fast (~2s vs ~40s).
Use the separate IMAS server (imas-codex serve imas) for DD search.

Local use only - provides read access to graph, write via ingest_nodes only.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any, Literal

from fastmcp import FastMCP
from neo4j.exceptions import ServiceUnavailable
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from imas_codex.agents.prompt_loader import (
    PromptDefinition,
    load_prompts,
)
from imas_codex.code_examples import CodeExampleSearch
from imas_codex.discovery import (
    get_facility,
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
    """Serialize Neo4j values to JSON-compatible types.

    Handles neo4j.time.DateTime, list, dict, and passes through primitives.
    """
    if value is None:
        return None
    # Check for Neo4j DateTime type (has isoformat method but isn't Python datetime)
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
    # Check for connection refused in the error chain
    if "Connection refused" in str(e) or "ServiceUnavailable" in str(e):
        return NEO4J_NOT_RUNNING_MSG
    return str(e)


# Cypher mutation keywords that modify the graph
_MUTATION_KEYWORDS = ["CREATE", "MERGE", "SET", "DELETE", "REMOVE", "DETACH"]
_MUTATION_PATTERN = re.compile(
    r"\b(" + "|".join(_MUTATION_KEYWORDS) + r")\b", re.IGNORECASE
)


def _is_cypher_mutation(query: str) -> bool:
    """
    Detect Cypher mutation keywords, ignoring content inside string literals.

    Strips string literals first to avoid false positives like SETTINGS,
    OFFSET, RESET, DATASET, CREATED_AT, and keywords inside string values.
    """
    # Strip string literals to avoid false positives
    stripped = re.sub(r'"[^"]*"', '""', query)
    stripped = re.sub(r"'[^']*'", "''", stripped)
    return bool(_MUTATION_PATTERN.search(stripped))


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

    Note: IMAS DD tools are NOT included here to keep startup fast (~2s vs ~40s).
    Use the separate IMAS server (imas-codex serve imas) for DD search.
    """

    mcp: FastMCP = field(init=False, repr=False)
    _prompts: dict[str, PromptDefinition] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the MCP server."""
        self.mcp = FastMCP(name="imas-codex-agents")
        self._prompts = load_prompts()
        self._register_tools()
        self._register_prompts()

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
                cypher("MATCH (p:FacilityPath {status: 'discovered'}) RETURN p.path, p.interest_score")
            """
            if _is_cypher_mutation(query):
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

            Special handling by node type:
            - SourceFile: Deduplicates automatically. Files already discovered,
              in progress, or with existing CodeExamples are skipped.
            - TreeNode: Auto-creates TREE_NAME and ACCESSOR_FUNCTION relationships.
            - FacilityPath: Links to parent Facility.

            ALWAYS pass a list of dicts for batch ingestion, even for single
            items. Batch mode uses UNWIND for efficient Neo4j operations.

            Args:
                node_type: Node label (must be valid LinkML class)
                data: List of property dicts matching the Pydantic model
                create_facility_relationship: Auto-create FACILITY_ID relationship
                batch_size: Number of nodes per UNWIND batch (default: 50)

            Returns:
                Dict with counts: {"processed": N, "skipped": K, "errors": [...]}

            Examples:
                # Queue source files for ingestion (idempotent, auto-deduplicates)
                ingest_nodes("SourceFile", [
                    {"id": "epfl:/home/codes/liuqe.py", "path": "/home/codes/liuqe.py",
                     "facility_id": "epfl", "status": "discovered", "interest_score": 0.8,
                     "patterns_matched": ["equilibrium", "IMAS"]},
                ])

                # Add FacilityPaths
                ingest_nodes("FacilityPath", [
                    {"id": "epfl:/home/codes", "path": "/home/codes", "facility_id": "epfl"},
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
                    # For SourceFile, skip items that are already discovered/ingested or have CodeExamples
                    skipped_dedup = 0
                    if node_type == "SourceFile":
                        existing = client.query(
                            """
                            UNWIND $items AS item
                            OPTIONAL MATCH (sf:SourceFile {id: item.id})
                            OPTIONAL MATCH (ce:CodeExample {source_file: item.path, facility_id: item.facility_id})
                            RETURN item.id AS id,
                                   sf.status AS sf_status,
                                   ce.id AS ce_id
                            """,
                            items=valid_items,
                        )
                        skip_ids = set()
                        for row in existing:
                            if row["sf_status"] in (
                                "discovered",
                                "ingested",
                            ):
                                skip_ids.add(row["id"])
                            elif row["ce_id"] is not None:
                                skip_ids.add(row["id"])
                        if skip_ids:
                            valid_items = [
                                i for i in valid_items if i["id"] not in skip_ids
                            ]
                            skipped_dedup = len(skip_ids)
                            logger.info(
                                f"Skipped {skipped_dedup} SourceFiles (already discovered/ingested)"
                            )
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

                    # For TreeNode, auto-create TREE_NAME and ACCESSOR_FUNCTION relationships
                    if node_type == "TreeNode":
                        # Create TREE_NAME relationships (TreeNode -> MDSplusTree)
                        tree_names = {
                            item["tree_name"]
                            for item in valid_items
                            if item.get("tree_name")
                        }
                        if tree_names:
                            client.query(
                                """
                                UNWIND $tree_names AS tn
                                MATCH (n:TreeNode {tree_name: tn})
                                MATCH (t:MDSplusTree {name: tn})
                                MERGE (n)-[:TREE_NAME]->(t)
                                """,
                                tree_names=list(tree_names),
                            )

                        # Create ACCESSOR_FUNCTION relationships (TreeNode -> TDIFunction)
                        accessor_funcs = {
                            item["accessor_function"]
                            for item in valid_items
                            if item.get("accessor_function")
                        }
                        if accessor_funcs:
                            client.query(
                                """
                                UNWIND $accessor_funcs AS af
                                MATCH (n:TreeNode {accessor_function: af})
                                MATCH (tdi:TDIFunction {name: af})
                                MERGE (n)-[:ACCESSOR_FUNCTION]->(tdi)
                                """,
                                accessor_funcs=list(accessor_funcs),
                            )

                    # For SourceFile, link to parent FacilityPath if provided
                    if node_type == "SourceFile":
                        parent_ids = {
                            item["parent_path_id"]
                            for item in valid_items
                            if item.get("parent_path_id")
                        }
                        if parent_ids:
                            for parent_id in parent_ids:
                                client.query(
                                    """
                                    MATCH (sf:SourceFile)
                                    WHERE sf.parent_path_id = $parent_id
                                    MATCH (p:FacilityPath {id: $parent_id})
                                    MERGE (p)-[:CONTAINS]->(sf)
                                    """,
                                    parent_id=parent_id,
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

                # Update tool availability (returns merged result)
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
        def get_facility_info(facility: str) -> dict[str, Any]:
            """
            Get comprehensive facility info for exploration.

            Merges public config, private data, and graph state.
            Call this before starting exploration to understand:
            - Available tools (rg, fd, etc.)
            - Actionable paths (discovered) by interest_score
            - Recently processed paths (explored)
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

                    # Get actionable paths (discovered) for exploration
                    actionable = client.query(
                        """
                        MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $fid})
                        WHERE p.status = 'discovered'
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
                        WHERE p.status = 'explored'
                        RETURN p.id AS id, p.path AS path, p.status AS status,
                               p.last_examined AS last_examined, p.files_ingested AS files_ingested
                        ORDER BY p.last_examined DESC
                        LIMIT 10
                        """,
                        fid=facility,
                    )
                    result["recent_paths"] = processed

                    # Get skipped paths (so agent knows what to skip)
                    skipped = client.query(
                        """
                        MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $fid})
                        WHERE p.status = 'skipped'
                        RETURN p.path AS path, p.description AS description
                        ORDER BY p.path
                        """,
                        fid=facility,
                    )
                    result["skipped_paths"] = [s["path"] for s in skipped]
            except Exception as e:
                result["graph_error"] = _neo4j_error_message(e)

            return result

        # =====================================================================
        # Code Example Tools
        # =====================================================================

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

        @self.mcp.tool()
        def get_exploration_progress(facility: str) -> dict[str, Any]:
            """
            Get exploration progress metrics for a facility.

            Calculates completion metrics based on FacilityPath status distribution,
            MDSplus tree coverage, and TreeNode graph population.

            Use this at the start of an exploration session to understand current
            state and identify high-priority targets.

            Terminology:
            - **ingested**: TreeNodes added to the graph (from tree introspection or
              code extraction). See MDSplusTree.node_count_ingested in schema.
            - **enriched**: TreeNodes enhanced by LLM with descriptions and metadata.
              Tracked separately via TreeNode.enrichment_status (discovered/enriched/stale).

            Args:
                facility: Facility ID (e.g., "epfl")

            Returns:
                Dict with:
                - paths: FacilityPath status distribution and completion
                - mdsplus_coverage: Per-tree node counts (total vs ingested to graph)
                - tree_node_coverage: TreeNodes by tree, domain, accessor function
                - tdi_coverage: TDI functions by physics domain
                - code_coverage: Analysis codes by type
                - next_targets: Top 5 prioritized exploration targets with actions
                - recommendation: Suggested next action (from top target)

            next_targets priorities:
                1. Trees with 0% coverage (breadth-first approach)
                2. Trees with <10% coverage (continue partial work)
                3. High-value physics domains with low coverage
                4. High-value results subtrees not yet explored
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

                    # MDSplus tree coverage (per-tree) - uses property match
                    tree_rows = client.query(
                        """
                        MATCH (t:MDSplusTree)-[:FACILITY_ID]->(f:Facility {id: $fid})
                        OPTIONAL MATCH (n:TreeNode {tree_name: t.name})-[:FACILITY_ID]->(f)
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

                    # TreeNode coverage by physics domain and accessor function
                    tree_node_rows = client.query(
                        """
                        MATCH (n:TreeNode)-[:FACILITY_ID]->(f:Facility {id: $fid})
                        RETURN n.tree_name AS tree,
                               n.physics_domain AS domain,
                               count(*) AS nodes,
                               sum(CASE WHEN n.accessor_function IS NOT NULL THEN 1 ELSE 0 END) AS with_accessor
                        ORDER BY nodes DESC
                        """,
                        fid=facility,
                    )

                    # Top subtrees in results tree
                    subtree_rows = client.query(
                        """
                        MATCH (n:TreeNode {tree_name: 'results'})-[:FACILITY_ID]->(f:Facility {id: $fid})
                        WITH split(replace(n.path, '\\\\RESULTS::', ''), ':')[0] AS subtree,
                             count(*) AS nodes
                        RETURN subtree, nodes
                        ORDER BY nodes DESC
                        LIMIT 15
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
                actionable = counts.get("discovered", 0)
                explored = counts.get("explored", 0)
                skipped = counts.get("skipped", 0)

                completion_pct = (
                    round(100 * (explored + skipped) / total, 1) if total else 0.0
                )

                # MDSplus tree coverage summary
                mdsplus_coverage = {}
                for row in tree_rows:
                    tree_name = row["tree"]
                    total_nodes = row["total_nodes"] or 0
                    ingested = row["ingested_nodes"] or row["nodes_in_graph"] or 0
                    mdsplus_coverage[tree_name] = {
                        "status": row["status"] or "discovered",
                        "population_type": row["population_type"],
                        "total_nodes": total_nodes,
                        "ingested_nodes": ingested,
                        "coverage_pct": round(100 * ingested / total_nodes, 1)
                        if total_nodes
                        else 0.0,
                        "last_ingested": _serialize_neo4j_value(row["last_ingested"]),
                    }

                # TreeNode coverage by tree and domain
                total_tree_nodes = sum(r["nodes"] for r in tree_node_rows)
                total_with_accessor = sum(r["with_accessor"] for r in tree_node_rows)
                by_tree: dict[str, int] = {}
                by_domain: dict[str, int] = {}
                for row in tree_node_rows:
                    tree = row["tree"] or "unknown"
                    domain = row["domain"] or "unclassified"
                    by_tree[tree] = by_tree.get(tree, 0) + row["nodes"]
                    by_domain[domain] = by_domain.get(domain, 0) + row["nodes"]

                tree_node_coverage = {
                    "total": total_tree_nodes,
                    "by_tree": dict(
                        sorted(by_tree.items(), key=lambda x: x[1], reverse=True)
                    ),
                    "by_domain": dict(
                        sorted(by_domain.items(), key=lambda x: x[1], reverse=True)
                    ),
                    "with_accessor": total_with_accessor,
                    "accessor_pct": round(
                        100 * total_with_accessor / total_tree_nodes, 1
                    )
                    if total_tree_nodes
                    else 0.0,
                    "top_subtrees": {r["subtree"]: r["nodes"] for r in subtree_rows},
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

                # Build next_targets: prioritized exploration targets for agents
                next_targets: list[dict[str, Any]] = []

                # Priority 1: Trees with 0% coverage (breadth-first)
                for tree_name, tree_info in mdsplus_coverage.items():
                    if tree_info["coverage_pct"] == 0.0 and tree_info["total_nodes"]:
                        next_targets.append(
                            {
                                "priority": 1,
                                "type": "mdsplus_tree",
                                "target": tree_name,
                                "action": f"Ingest {tree_name} tree structure",
                                "expected_nodes": tree_info["total_nodes"],
                                "population_type": tree_info["population_type"],
                                "effort": "high"
                                if tree_info["total_nodes"] > 1000
                                else "medium",
                            }
                        )

                # Priority 2: Trees with low coverage (<10%)
                for tree_name, tree_info in mdsplus_coverage.items():
                    cov = tree_info["coverage_pct"]
                    if 0.0 < cov < 10.0:
                        remaining = (
                            tree_info["total_nodes"] - tree_info["ingested_nodes"]
                        )
                        next_targets.append(
                            {
                                "priority": 2,
                                "type": "mdsplus_tree",
                                "target": tree_name,
                                "action": f"Continue {tree_name} ingestion ({cov:.1f}% complete)",
                                "remaining_nodes": remaining,
                                "effort": "high" if remaining > 1000 else "medium",
                            }
                        )

                # Priority 3: Physics domains with low coverage
                high_value_domains = ["equilibrium", "profiles", "magnetics", "heating"]
                for domain in high_value_domains:
                    domain_count = by_domain.get(domain, 0)
                    if domain_count < 50:  # Threshold for "low coverage"
                        next_targets.append(
                            {
                                "priority": 3,
                                "type": "physics_domain",
                                "target": domain,
                                "action": f"Expand {domain} domain coverage",
                                "current_nodes": domain_count,
                                "effort": "medium",
                            }
                        )

                # Priority 4: Subtrees in results tree not yet explored
                known_subtrees = set(tree_node_coverage["top_subtrees"].keys())
                high_value_subtrees = {
                    "THOMSON",
                    "LANGMUIR",
                    "CXRS",
                    "ECE",
                    "FIR",
                    "BOLOMETER",
                    "TORAY",
                    "LIUQE",
                    "PSITBX",
                    "ECRH",
                    "NBI",
                }
                missing_subtrees = high_value_subtrees - known_subtrees
                for subtree in sorted(missing_subtrees):
                    next_targets.append(
                        {
                            "priority": 4,
                            "type": "results_subtree",
                            "target": subtree,
                            "action": f"Explore \\\\RESULTS::{subtree} subtree",
                            "effort": "medium",
                        }
                    )

                # Sort by priority and limit to top 5
                next_targets.sort(key=lambda x: x["priority"])
                next_targets = next_targets[:5]

                # Recommend next action based on current state
                if total == 0:
                    recommendation = "Run /scout-paths to discover paths"
                elif next_targets:
                    top = next_targets[0]
                    recommendation = top["action"]
                elif actionable > explored:
                    recommendation = "Run exploration to process discovered paths"
                elif tdi_coverage["total"] == 0:
                    recommendation = (
                        "Run /ingest-tdi-functions to discover TDI functions"
                    )
                else:
                    recommendation = "Increase depth or explore new root paths"

                return {
                    "facility": facility,
                    "paths": {
                        "total": total,
                        "actionable": actionable,
                        "explored": explored,
                        "skipped": skipped,
                        "completion_pct": completion_pct,
                        "by_status": counts,
                    },
                    "mdsplus_coverage": mdsplus_coverage,
                    "tree_node_coverage": tree_node_coverage,
                    "tdi_coverage": tdi_coverage,
                    "code_coverage": code_coverage,
                    "next_targets": next_targets,
                    "recommendation": recommendation,
                }
            except Exception as e:
                logger.exception("Failed to get exploration progress")
                raise RuntimeError(
                    f"Failed to get exploration progress: {_neo4j_error_message(e)}"
                ) from e

        # =====================================================================
        # Agent State Tools (for autonomous agents)
        # =====================================================================

        @self.mcp.tool()
        def get_exploration_state(
            facility: str,
            root_path: str | None = None,
        ) -> dict[str, Any]:
            """
            Get current exploration state for a facility or subtree.

            Returns comprehensive state for resumable exploration:
            - Path statistics by status
            - Source file processing progress
            - Coverage percentage
            - Recommended next action

            Terminology:
            - **files_discovered**: SourceFiles discovered by scouts, awaiting processing
            - **files_ingested**: SourceFiles successfully processed (CodeExamples extracted)

            Args:
                facility: Facility ID (e.g., "epfl")
                root_path: Optional root path to scope the query
                           (e.g., "/home/codes" to get state for that subtree)

            Returns:
                Dict with:
                - paths_discovered: Count of paths in discovered status
                - paths_explored: Count of paths in explored status
                - files_discovered: SourceFiles awaiting processing (status=discovered)
                - files_ingested: SourceFiles with code extracted (status=ingested)
                - coverage_pct: Percentage of discovered paths that are explored
                - recommended_next: Suggested next action based on state

            Examples:
                # Full facility state
                get_exploration_state("epfl")

                # Subtree state
                get_exploration_state("epfl", "/home/codes/liuqe")
            """
            try:
                with GraphClient() as client:
                    # Query path status distribution with optional root filter
                    if root_path:
                        path_result = client.query(
                            """
                            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $fid})
                            WHERE p.path STARTS WITH $root
                            RETURN p.status AS status, count(*) AS count
                            """,
                            fid=facility,
                            root=root_path,
                        )
                        file_result = client.query(
                            """
                            MATCH (sf:SourceFile)-[:FACILITY_ID]->(f:Facility {id: $fid})
                            WHERE sf.path STARTS WITH $root
                            RETURN sf.status AS status, count(*) AS count
                            """,
                            fid=facility,
                            root=root_path,
                        )
                    else:
                        path_result = client.query(
                            """
                            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $fid})
                            RETURN p.status AS status, count(*) AS count
                            """,
                            fid=facility,
                        )
                        file_result = client.query(
                            """
                            MATCH (sf:SourceFile)-[:FACILITY_ID]->(f:Facility {id: $fid})
                            RETURN sf.status AS status, count(*) AS count
                            """,
                            fid=facility,
                        )

                    # Build status counts
                    path_counts = {r["status"]: r["count"] for r in path_result}
                    file_counts = {r["status"]: r["count"] for r in file_result}

                    # Calculate aggregates
                    discovered = path_counts.get("discovered", 0)
                    explored = path_counts.get("explored", 0)
                    total = sum(path_counts.values())

                    files_discovered = file_counts.get("discovered", 0)
                    files_ingested = file_counts.get("ingested", 0)

                    coverage_pct = round(100 * explored / total, 1) if total else 0.0

                    # Determine recommendation
                    if total == 0:
                        recommended = "Run discover paths to find code directories"
                    elif discovered > explored * 3:
                        recommended = "Run discover files to discover sources from discovered paths"
                    elif files_discovered > files_ingested:
                        recommended = (
                            f"Run ingest to process {files_discovered} discovered files"
                        )
                    else:
                        recommended = "Exploration complete for this scope"

                    return {
                        "facility": facility,
                        "root_path": root_path,
                        "paths_discovered": discovered,
                        "paths_explored": explored,
                        "paths_total": total,
                        "files_discovered": files_discovered,
                        "files_ingested": files_ingested,
                        "coverage_pct": coverage_pct,
                        "by_path_status": path_counts,
                        "by_file_status": file_counts,
                        "recommended_next": recommended,
                    }
            except Exception as e:
                logger.exception("Failed to get exploration state")
                raise RuntimeError(
                    f"Failed to get exploration state: {_neo4j_error_message(e)}"
                ) from e

        @self.mcp.tool()
        def mark_path_explored(
            facility: str,
            path: str,
            status: str,
            file_count: int | None = None,
            patterns_found: list[str] | None = None,
            interest_score: float | None = None,
            notes: str | None = None,
        ) -> dict[str, Any]:
            """
            Mark a FacilityPath as explored with the given status.

            This is the primary tool for updating path exploration state.
            Automatically sets last_examined timestamp. Creates the path if missing.

            Args:
                facility: Facility ID
                path: Absolute path to mark
                status: New PathStatus value (discovered, explored, skipped, stale)
                file_count: Optional file count (set during listing)
                patterns_found: Patterns that matched (e.g., ["equilibrium", "IMAS"])
                interest_score: Priority score 0.0-1.0 (higher = more interesting)
                notes: Free-form notes to append

            Returns:
                Updated path info: {id, path, status, interest_score, patterns_found}

            Examples:
                # Mark as discovered with patterns found
                mark_path_explored("epfl", "/home/codes/liuqe", "discovered",
                                  file_count=47, patterns_found=["equilibrium", "IMAS"],
                                  interest_score=0.9)

                # Mark as explored (all files discovered for ingestion)
                mark_path_explored("epfl", "/home/codes/liuqe", "explored",
                                  notes="All 47 files discovered for ingestion")
            """
            # Validate status
            schema = get_schema()
            valid_statuses = schema.get_enums().get("PathStatus", [])
            if status not in valid_statuses:
                raise ValueError(f"Invalid status '{status}'. Valid: {valid_statuses}")

            try:
                with GraphClient() as client:
                    path_id = f"{facility}:{path}"

                    # First try to update existing path
                    set_parts = ["p.status = $status", "p.last_examined = datetime()"]
                    params: dict[str, Any] = {
                        "fid": facility,
                        "path": path,
                        "status": status,
                    }

                    if file_count is not None:
                        set_parts.append("p.file_count = $file_count")
                        params["file_count"] = file_count
                    if patterns_found is not None:
                        set_parts.append("p.patterns_found = $patterns_found")
                        params["patterns_found"] = patterns_found
                    if interest_score is not None:
                        set_parts.append("p.interest_score = $interest_score")
                        params["interest_score"] = interest_score
                    if notes:
                        set_parts.append(
                            "p.notes = CASE WHEN p.notes IS NULL THEN $notes "
                            "ELSE p.notes + '\\n' + $notes END"
                        )
                        params["notes"] = notes

                    set_clause = ", ".join(set_parts)

                    result = client.query(
                        f"""
                        MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {{id: $fid}})
                        WHERE p.path = $path
                        SET {set_clause}
                        RETURN p.id AS id, p.path AS path, p.status AS status,
                               p.interest_score AS interest_score,
                               p.patterns_found AS patterns_found
                        """,
                        **params,
                    )

                    if result:
                        return dict(result[0])

                    # Path doesn't exist, create it
                    create_props = {
                        "id": path_id,
                        "path": path,
                        "path_type": "code_directory",
                        "status": status,
                        "facility_id": facility,
                    }
                    if file_count is not None:
                        create_props["file_count"] = file_count
                    if patterns_found is not None:
                        create_props["patterns_found"] = patterns_found
                    if interest_score is not None:
                        create_props["interest_score"] = interest_score
                    if notes:
                        create_props["notes"] = notes

                    result = client.query(
                        """
                        MATCH (f:Facility {id: $fid})
                        CREATE (p:FacilityPath $props)-[:FACILITY_ID]->(f)
                        SET p.discovered_at = datetime(), p.last_examined = datetime()
                        RETURN p.id AS id, p.path AS path, p.status AS status,
                               p.interest_score AS interest_score,
                               p.patterns_found AS patterns_found
                        """,
                        fid=facility,
                        props=create_props,
                    )

                    if result:
                        return dict(result[0])

                    return {"error": f"Failed to create path for facility {facility}"}
            except Exception as e:
                logger.exception("Failed to mark path explored")
                raise RuntimeError(
                    f"Failed to mark path: {_neo4j_error_message(e)}"
                ) from e

        @self.mcp.tool()
        def get_agent_progress(
            run_id: str | None = None,
            facility: str | None = None,
        ) -> dict[str, Any]:
            """
            Get progress for an agent run.

            If run_id is provided, returns that specific run's progress.
            If only facility is provided, returns the most recent run.
            If neither, returns all running agents.

            Args:
                run_id: Specific AgentRun ID to query
                facility: Facility to filter by (returns most recent)

            Returns:
                AgentRun info with progress metrics:
                - Basic info: id, agent_type, status, started_at
                - Progress: items_processed, items_total, completion_pct
                - Budget: budget_seconds, actual elapsed, budget_cost, actual_cost
                - Checkpoint: last_item_id, last_checkpoint_at
                - For running agents: estimated time remaining

            Examples:
                # Get specific run
                get_agent_progress(run_id="epfl-discover-paths-20260107-1234")

                # Get most recent run for facility
                get_agent_progress(facility="epfl")

                # Get all running agents
                get_agent_progress()
            """
            try:
                with GraphClient() as client:
                    if run_id:
                        result = client.query(
                            "MATCH (ar:AgentRun {id: $run_id}) RETURN ar",
                            run_id=run_id,
                        )
                    elif facility:
                        result = client.query(
                            """
                            MATCH (ar:AgentRun)-[:FACILITY_ID]->(f:Facility {id: $fid})
                            RETURN ar
                            ORDER BY ar.started_at DESC
                            LIMIT 1
                            """,
                            fid=facility,
                        )
                    else:
                        result = client.query(
                            """
                            MATCH (ar:AgentRun)
                            WHERE ar.status = 'running'
                            RETURN ar
                            ORDER BY ar.started_at DESC
                            """
                        )

                    if not result:
                        return {"error": "No matching agent runs found"}

                    from datetime import datetime

                    runs = []
                    for row in result:
                        ar = dict(row["ar"])

                        # Calculate completion percentage
                        if ar.get("items_total") and ar.get("items_total") > 0:
                            ar["completion_pct"] = round(
                                100
                                * (ar.get("items_processed", 0) / ar["items_total"]),
                                1,
                            )
                        else:
                            ar["completion_pct"] = None

                        # Calculate elapsed time
                        if ar.get("started_at"):
                            started = ar["started_at"]
                            if isinstance(started, str):
                                started = datetime.fromisoformat(
                                    started.replace("Z", "+00:00")
                                )
                            if hasattr(started, "tzinfo") and started.tzinfo is None:
                                started = started.replace(tzinfo=UTC)
                            elapsed = (datetime.now(UTC) - started).total_seconds()
                            ar["elapsed_seconds"] = round(elapsed, 1)

                            # Check budget
                            if ar.get("budget_seconds"):
                                remaining = ar["budget_seconds"] - elapsed
                                ar["budget_remaining_seconds"] = max(
                                    0, round(remaining, 1)
                                )
                                ar["budget_exceeded"] = remaining < 0

                        # Serialize neo4j datetime values
                        runs.append(_serialize_neo4j_value(ar))

                    return {"runs": runs} if len(runs) > 1 else runs[0]
            except Exception as e:
                logger.exception("Failed to get agent progress")
                raise RuntimeError(
                    f"Failed to get agent progress: {_neo4j_error_message(e)}"
                ) from e

        @self.mcp.tool()
        def checkpoint_agent_run(
            run_id: str,
            last_item_id: str,
            items_processed: int,
            items_skipped: int = 0,
            items_failed: int = 0,
            actual_cost: float | None = None,
            llm_calls: int | None = None,
            tokens_used: int | None = None,
        ) -> dict[str, Any]:
            """
            Save checkpoint for an agent run (for resumability).

            Call this periodically (e.g., every 10 items or 60 seconds) to enable
            crash recovery. On resume, query last_item_id to continue from there.

            Args:
                run_id: AgentRun ID to checkpoint
                last_item_id: ID of last successfully processed item
                items_processed: Total items processed so far
                items_skipped: Items skipped (already done, excluded)
                items_failed: Items that failed processing
                actual_cost: Cumulative cost in USD
                llm_calls: Cumulative LLM calls
                tokens_used: Cumulative tokens

            Returns:
                Updated AgentRun with checkpoint info

            Examples:
                # Checkpoint after processing a batch
                checkpoint_agent_run(
                    run_id="epfl-discover-paths-20260107-1234",
                    last_item_id="/home/codes/liuqe",
                    items_processed=150,
                    items_skipped=23,
                    actual_cost=0.42
                )
            """
            try:
                with GraphClient() as client:
                    # Build SET clause dynamically
                    set_parts = [
                        "ar.last_item_id = $last_item_id",
                        "ar.items_processed = $items_processed",
                        "ar.items_skipped = $items_skipped",
                        "ar.items_failed = $items_failed",
                        "ar.last_checkpoint_at = datetime()",
                    ]
                    params: dict[str, Any] = {
                        "run_id": run_id,
                        "last_item_id": last_item_id,
                        "items_processed": items_processed,
                        "items_skipped": items_skipped,
                        "items_failed": items_failed,
                    }

                    if actual_cost is not None:
                        set_parts.append("ar.actual_cost = $actual_cost")
                        params["actual_cost"] = actual_cost
                    if llm_calls is not None:
                        set_parts.append("ar.llm_calls = $llm_calls")
                        params["llm_calls"] = llm_calls
                    if tokens_used is not None:
                        set_parts.append("ar.tokens_used = $tokens_used")
                        params["tokens_used"] = tokens_used

                    set_clause = ", ".join(set_parts)

                    result = client.query(
                        f"""
                        MATCH (ar:AgentRun {{id: $run_id}})
                        SET {set_clause}
                        RETURN ar.id AS id,
                               ar.status AS status,
                               ar.items_processed AS items_processed,
                               ar.items_total AS items_total,
                               ar.last_item_id AS last_item_id,
                               ar.last_checkpoint_at AS last_checkpoint_at,
                               ar.actual_cost AS actual_cost
                        """,
                        **params,
                    )

                    if not result:
                        raise ValueError(f"AgentRun not found: {run_id}")

                    row = dict(result[0])

                    # Calculate completion percentage
                    if row.get("items_total") and row["items_total"] > 0:
                        row["completion_pct"] = round(
                            100 * row["items_processed"] / row["items_total"], 1
                        )

                    return row
            except Exception as e:
                logger.exception("Failed to checkpoint agent run")
                raise RuntimeError(
                    f"Failed to checkpoint: {_neo4j_error_message(e)}"
                ) from e

    def _register_prompts(self):
        """Register MCP prompts from markdown files."""
        for name, prompt_def in self._prompts.items():
            # Capture prompt_def in closure
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
