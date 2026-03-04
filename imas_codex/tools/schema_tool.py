"""DD graph schema tool for MCP schema introspection.

Exposes the IMAS Data Dictionary portion of the LinkML-derived graph
schema to MCP clients, enabling agents to understand available node
types, relationships, and properties before composing Cypher queries.
"""

import logging

from imas_codex.search.decorators import mcp_tool

logger = logging.getLogger(__name__)


class SchemaTool:
    """DD graph schema introspection tool for MCP clients."""

    @mcp_tool(
        "Get the IMAS Data Dictionary graph schema: node types with properties, "
        "relationship types with directionality, enum values, and vector indexes. "
        "Call this before composing Cypher queries with query_imas_graph() to "
        "understand available data structures. Returns node types (DDVersion, IDS, "
        "IMASPath, Unit, IMASSemanticCluster, IMASPathChange, etc.), their properties, "
        "relationships (INTRODUCED_IN, DEPRECATED_IN, IN_IDS, IN_CLUSTER, HAS_PREDECESSOR, "
        "FOR_IMAS_PATH, etc.), and semantic search indexes."
    )
    async def get_dd_graph_schema(self) -> str:
        """Return the DD graph schema for client inspection.

        Returns compact text schema context for the IMAS task group,
        auto-generated from LinkML schemas.
        """
        from imas_codex.graph.schema_context import schema_for

        return schema_for(task="imas")
