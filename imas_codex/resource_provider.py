# File: imas_codex/resources.py
"""
IMAS Codex Resources - Graph-Native MCP Resources.

All data dictionary data is served by graph-backed tools.
Resources provide usage guidance only.
"""

import json
import logging

from fastmcp import FastMCP

from imas_codex.providers import MCPProvider

logger = logging.getLogger(__name__)


def mcp_resource(description: str, uri: str):
    """Decorator to mark methods as MCP resources with description and URI."""

    def decorator(func):
        func._mcp_resource = True
        func._mcp_resource_uri = uri
        func._mcp_resource_description = description
        return func

    return decorator


class Resources(MCPProvider):
    """MCP resources providing usage examples."""

    def __init__(self, ids_set: set[str] | None = None):
        self.ids_set = ids_set

    @property
    def name(self) -> str:
        return "resources"

    def register(self, mcp: FastMCP):
        """Register resources with the MCP server."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, "_mcp_resource") and attr._mcp_resource:
                uri = attr._mcp_resource_uri
                description = attr._mcp_resource_description
                mcp.resource(uri=uri, description=description)(attr)

    @mcp_resource(
        "Resource Usage Examples - How to effectively use IMAS Codex resources.",
        "examples://resource-usage",
    )
    async def get_resource_usage_examples(self) -> str:
        """Resource Usage Examples - How to effectively use IMAS Codex resources."""
        examples = {
            "workflow_patterns": {
                "quick_orientation": {
                    "description": "Use tools for all data exploration",
                    "steps": [
                        "1. Use get_imas_overview for IDS overview and domain mapping",
                        "2. Use search_imas_paths to find specific data paths",
                        "3. Use list_imas_paths to browse IDS structure",
                        "4. Use get_imas_identifiers for enumeration options",
                        "5. Use search_imas_clusters for cross-IDS groupings",
                    ],
                },
                "schema_inspection": {
                    "description": "Inspect the data dictionary schema",
                    "steps": [
                        "1. Use get_dd_schema to see LinkML class definitions",
                        "2. Use get_dd_versions for version history",
                        "3. Use cypher_query for custom graph traversals",
                    ],
                },
            },
            "tools_reference": {
                "search_imas_paths": "Semantic search across IMAS paths",
                "check_imas_paths": "Validate specific IMAS paths",
                "fetch_imas_paths": "Retrieve path details by exact ID",
                "list_imas_paths": "Browse paths by IDS name",
                "get_imas_overview": "Summary of all IDS in the graph",
                "get_imas_identifiers": "Enumerated options and branching logic",
                "search_imas_clusters": "Cross-IDS semantic clusters",
                "cypher_query": "Custom Cypher queries against the graph",
                "get_dd_schema": "LinkML schema definitions",
                "get_dd_versions": "DD version history",
            },
        }
        return json.dumps(examples, indent=2)
