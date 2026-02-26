"""DD graph schema tool for MCP schema introspection.

Exposes the IMAS Data Dictionary portion of the LinkML-derived graph
schema to MCP clients, enabling agents to understand available node
types, relationships, and properties before composing Cypher queries.
"""

import logging
from functools import lru_cache
from pathlib import Path

from imas_codex.graph.schema import GraphSchema
from imas_codex.search.decorators import mcp_tool

logger = logging.getLogger(__name__)


def _get_dd_schema() -> GraphSchema:
    """Load the IMAS DD LinkML schema."""
    schema_path = Path(__file__).parent.parent / "schemas" / "imas_dd.yaml"
    return GraphSchema(schema_path)


@lru_cache(maxsize=1)
def _build_schema_summary() -> dict:
    """Build a comprehensive DD schema summary for MCP clients.

    Cached because the schema is static — LinkML doesn't change at runtime.
    """
    schema = _get_dd_schema()

    # Node types with properties
    node_types = {}
    for label in schema.node_labels:
        slots = schema.get_all_slots(label)
        # Filter out relationship slots and embedding vectors
        properties = {}
        for name, info in slots.items():
            if info.get("relationship"):
                continue
            properties[name] = {k: v for k, v in info.items() if k != "relationship"}
        node_types[label] = {
            "description": schema.get_class_description(label) or "",
            "properties": properties,
        }

    # Relationships
    relationships = []
    for rel in schema.relationships:
        relationships.append(
            {
                "type": rel.cypher_type,
                "from": rel.from_class,
                "to": rel.to_class,
                "multivalued": rel.multivalued,
            }
        )

    # Enums
    enums = schema.get_enums()

    # Vector indexes
    vector_indexes = [
        {"name": name, "label": label, "property": prop}
        for name, label, prop in schema.vector_indexes
    ]

    return {
        "node_types": node_types,
        "relationships": relationships,
        "enums": enums,
        "vector_indexes": vector_indexes,
        "notes": {
            "version_lifecycle": (
                "IMASPath nodes link to DDVersion via INTRODUCED_IN and "
                "DEPRECATED_IN relationships. DDVersion nodes chain via PREDECESSOR."
            ),
            "version_evolution": (
                "IMASPathChange nodes track metadata mutations between versions. "
                "Each links to the affected IMASPath via FOR_IMAS_PATH and to the "
                "target version via IN_VERSION. SemanticChangeType classifies the "
                "physics significance of changes."
            ),
            "clusters": (
                "IMASSemanticCluster groups related paths via IN_CLUSTER. "
                "Use vector search on cluster_description_embedding for semantic "
                "cluster discovery."
            ),
        },
    }


class SchemaTool:
    """DD graph schema introspection tool for MCP clients."""

    @mcp_tool(
        "Get the IMAS Data Dictionary graph schema: node types with properties, "
        "relationship types with directionality, enum values, and vector indexes. "
        "Call this before composing Cypher queries with query_imas_graph() to "
        "understand available data structures. Returns node types (DDVersion, IDS, "
        "IMASPath, Unit, IMASSemanticCluster, IMASPathChange, etc.), their properties, "
        "relationships (INTRODUCED_IN, DEPRECATED_IN, IN_IDS, IN_CLUSTER, PREDECESSOR, "
        "FOR_IMAS_PATH, etc.), and semantic search indexes."
    )
    async def get_dd_graph_schema(self) -> dict:
        """Return the DD graph schema for client inspection.

        Returns:
            Schema summary with node_types, relationships, enums,
            vector_indexes, and usage notes.
        """
        return _build_schema_summary()
