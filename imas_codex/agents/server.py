"""
Agents MCP Server - Tools for LLM-driven facility exploration.

This server provides MCP tools for:
- Executing Cypher queries (mutations restricted to _Discovery nodes)
- Ingesting validated nodes to the knowledge graph
- Reading/updating sensitive infrastructure files

Local use only - provides full graph access for trusted agents.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import yaml
from fastmcp import FastMCP

from imas_codex.discovery import get_config
from imas_codex.discovery.config import get_facilities_dir
from imas_codex.graph import GraphClient, get_schema
from imas_codex.graph.schema import to_cypher_props

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, updates: dict) -> dict:
    """Deep merge updates into base dict."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


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
    """

    mcp: FastMCP = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the MCP server."""
        self.mcp = FastMCP(name="imas-codex-agents")
        self._register_tools()
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
            file, creating it if it doesn't exist.

            Args:
                facility: Facility identifier (e.g., "epfl")
                data: Data to merge (will be deep-merged with existing)

            Returns:
                Success message

            Examples:
                # Update tool availability
                update_infrastructure("epfl", {
                    "tools": {
                        "rg": {"status": "unavailable"},
                        "grep": {"status": "available", "path": "/usr/bin/grep"}
                    }
                })

                # Add exploration notes
                update_infrastructure("epfl", {
                    "notes": ["MDSplus config at /usr/local/mdsplus/local/mdsplus.conf"]
                })
            """
            infra_path = get_facilities_dir() / f"{facility}_infrastructure.yaml"

            # Load existing or start fresh
            if infra_path.exists():
                existing = yaml.safe_load(infra_path.read_text()) or {}
            else:
                existing = {"facility_id": facility}

            # Deep merge
            merged = _deep_merge(existing, data)

            # Add timestamp
            merged["last_updated"] = datetime.now(UTC).isoformat()

            try:
                infra_path.write_text(yaml.dump(merged, default_flow_style=False))
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
