"""IMAS Codex Tools Package.

All tools are backed by Neo4j via GraphClient.
"""

import logging

from fastmcp import FastMCP

from imas_codex.graph.client import GraphClient
from imas_codex.providers import MCPProvider

from .base import BaseTool
from .cypher_tool import CypherTool
from .graph_search import (
    GraphClustersTool,
    GraphIdentifiersTool,
    GraphListTool,
    GraphOverviewTool,
    GraphPathTool,
    GraphSearchTool,
)
from .schema_tool import SchemaTool
from .version_tool import VersionTool

logger = logging.getLogger(__name__)


class Tools(MCPProvider):
    """MCP tools backed by Neo4j."""

    _tool_instances: list = []

    def __init__(
        self,
        ids_set: set[str] | None = None,
        graph_client: GraphClient | None = None,
    ):
        if graph_client is None:
            raise ValueError("GraphClient is required")

        self.ids_set = ids_set

        self.search_tool = GraphSearchTool(graph_client)
        self.path_tool = GraphPathTool(graph_client)
        self.list_tool = GraphListTool(graph_client)
        self.overview_tool = GraphOverviewTool(graph_client)
        self.clusters_tool = GraphClustersTool(graph_client)
        self.identifiers_tool = GraphIdentifiersTool(graph_client)
        self.cypher_tool = CypherTool(graph_client)
        self.schema_tool = SchemaTool()
        self.version_tool = VersionTool(graph_client)

        self._tool_instances = [
            self.search_tool,
            self.path_tool,
            self.list_tool,
            self.overview_tool,
            self.clusters_tool,
            self.identifiers_tool,
            self.cypher_tool,
            self.schema_tool,
            self.version_tool,
        ]

    @property
    def name(self) -> str:
        """Provider name for logging and identification."""
        return "tools"

    def register(self, mcp: FastMCP):
        """Register all IMAS tools with the MCP server."""
        for tool in self._tool_instances:
            for attr_name in dir(tool):
                if attr_name.startswith("_"):
                    continue
                attr = getattr(tool, attr_name)
                if getattr(attr, "_mcp_tool", None) is True:
                    mcp.tool(description=attr._mcp_description)(attr)

    def get_registered_tool_names(self) -> list[str]:
        """Get list of all registered MCP tool names."""
        tool_names = []
        for tool in self._tool_instances:
            for attr_name in dir(tool):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(tool, attr_name)
                    if getattr(attr, "_mcp_tool", None) is True:
                        tool_names.append(attr_name)
                except AttributeError:
                    continue
        return sorted(tool_names)

    # Primary method delegation
    async def search_imas_paths(self, *args, **kwargs):
        """Delegate to search tool."""
        return await self.search_tool.search_imas_paths(*args, **kwargs)

    async def check_imas_paths(self, *args, **kwargs):
        """Delegate to path tool."""
        return await self.path_tool.check_imas_paths(*args, **kwargs)

    async def fetch_imas_paths(self, *args, **kwargs):
        """Delegate to path tool."""
        return await self.path_tool.fetch_imas_paths(*args, **kwargs)

    async def list_imas_paths(self, *args, **kwargs):
        """Delegate to list tool."""
        return await self.list_tool.list_imas_paths(*args, **kwargs)

    async def get_imas_overview(self, *args, **kwargs):
        """Delegate to overview tool."""
        return await self.overview_tool.get_imas_overview(*args, **kwargs)

    async def get_imas_identifiers(self, *args, **kwargs):
        """Delegate to identifiers tool."""
        return await self.identifiers_tool.get_imas_identifiers(*args, **kwargs)

    async def search_imas_clusters(self, *args, **kwargs):
        """Delegate to clusters tool."""
        return await self.clusters_tool.search_imas_clusters(*args, **kwargs)


__all__ = [
    "BaseTool",
    "GraphSearchTool",
    "GraphPathTool",
    "GraphListTool",
    "GraphOverviewTool",
    "GraphClustersTool",
    "GraphIdentifiersTool",
    "CypherTool",
    "SchemaTool",
    "VersionTool",
    "Tools",
]
