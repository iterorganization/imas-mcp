"""IMAS Codex Tools Package.

The supported IMAS DD tool surface is graph-backed and implemented via
the Graph*Tool classes exported from graph_search.py plus VersionTool.
Legacy file-backed IMAS tool implementations remain in the repository only
until the clean-break removal phase and must not be wired into this provider.
"""

import logging

from fastmcp import FastMCP

from imas_codex.graph.client import GraphClient
from imas_codex.providers import MCPProvider

from .base import BaseTool
from .dd_analytics_tool import DDAnalyticsTool
from .graph_search import (
    GraphClustersTool,
    GraphIdentifiersTool,
    GraphListTool,
    GraphOverviewTool,
    GraphPathContextTool,
    GraphPathTool,
    GraphSearchTool,
    GraphStructureTool,
)
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

        # Graph-backed IMAS DD tools are the only supported provider path.
        self.search_tool = GraphSearchTool(graph_client)
        self.path_tool = GraphPathTool(graph_client)
        self.list_tool = GraphListTool(graph_client)
        self.overview_tool = GraphOverviewTool(graph_client)
        self.clusters_tool = GraphClustersTool(graph_client)
        self.identifiers_tool = GraphIdentifiersTool(graph_client)
        self.path_context_tool = GraphPathContextTool(graph_client)
        self.structure_tool = GraphStructureTool(graph_client)
        self.version_tool = VersionTool(graph_client)
        self.dd_analytics_tool = DDAnalyticsTool(graph_client)

        self._tool_instances = [
            self.search_tool,
            self.path_tool,
            self.list_tool,
            self.overview_tool,
            self.clusters_tool,
            self.identifiers_tool,
            self.path_context_tool,
            self.structure_tool,
            self.version_tool,
            self.dd_analytics_tool,
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
    async def search_dd_paths(self, *args, **kwargs):
        """Delegate to search tool."""
        return await self.search_tool.search_dd_paths(*args, **kwargs)

    async def check_dd_paths(self, *args, **kwargs):
        """Delegate to path tool."""
        return await self.path_tool.check_dd_paths(*args, **kwargs)

    async def fetch_dd_paths(self, *args, **kwargs):
        """Delegate to path tool."""
        return await self.path_tool.fetch_dd_paths(*args, **kwargs)

    async def list_dd_paths(self, *args, **kwargs):
        """Delegate to list tool."""
        return await self.list_tool.list_dd_paths(*args, **kwargs)

    async def get_dd_overview(self, *args, **kwargs):
        """Delegate to overview tool."""
        return await self.overview_tool.get_dd_overview(*args, **kwargs)

    async def get_dd_identifiers(self, *args, **kwargs):
        """Delegate to identifiers tool."""
        return await self.identifiers_tool.get_dd_identifiers(*args, **kwargs)

    async def get_dd_path_context(self, *args, **kwargs):
        """Delegate to path context tool."""
        return await self.path_context_tool.get_dd_path_context(*args, **kwargs)

    async def analyze_dd_structure(self, *args, **kwargs):
        """Delegate to structure tool."""
        return await self.structure_tool.analyze_dd_structure(*args, **kwargs)

    async def export_imas_ids(self, *args, **kwargs):
        """Delegate to structure tool."""
        return await self.structure_tool.export_imas_ids(*args, **kwargs)

    async def export_imas_domain(self, *args, **kwargs):
        """Delegate to structure tool."""
        return await self.structure_tool.export_imas_domain(*args, **kwargs)

    async def get_dd_versions(self, *args, **kwargs):
        """Delegate to version tool."""
        return await self.version_tool.get_dd_versions(*args, **kwargs)

    async def search_dd_clusters(self, *args, **kwargs):
        """Delegate to clusters tool."""
        return await self.clusters_tool.search_dd_clusters(*args, **kwargs)

    async def get_dd_version_context(self, *args, **kwargs):
        """Delegate to version tool."""
        return await self.version_tool.get_dd_version_context(*args, **kwargs)

    async def get_dd_changelog(self, *args, **kwargs):
        """Delegate to version tool."""
        return await self.version_tool.get_dd_changelog(*args, **kwargs)

    async def fetch_dd_error_fields(self, *args, **kwargs):
        """Delegate to path tool."""
        return await self.path_tool.fetch_dd_error_fields(*args, **kwargs)

    async def analyze_dd_coverage(self, *args, **kwargs):
        """Delegate to DD analytics tool."""
        return await self.dd_analytics_tool.analyze_dd_coverage(*args, **kwargs)

    async def check_dd_units(self, *args, **kwargs):
        """Delegate to DD analytics tool."""
        return await self.dd_analytics_tool.check_dd_units(*args, **kwargs)

    async def analyze_dd_changes(self, *args, **kwargs):
        """Delegate to DD analytics tool."""
        return await self.dd_analytics_tool.analyze_dd_changes(*args, **kwargs)


__all__ = [
    "BaseTool",
    "DDAnalyticsTool",
    "GraphSearchTool",
    "GraphPathTool",
    "GraphListTool",
    "GraphOverviewTool",
    "GraphClustersTool",
    "GraphIdentifiersTool",
    "VersionTool",
    "Tools",
]
