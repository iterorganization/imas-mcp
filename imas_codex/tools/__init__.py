"""IMAS Codex Tools Package.

The supported IMAS DD tool surface is graph-backed and implemented via
the Graph*Tool classes exported from graph_search.py plus VersionTool.

Callers access tool methods directly on the tool instances:
    tools.path_tool.check_dd_paths(...)
    tools.version_tool.get_dd_versions(...)
"""

import logging

from fastmcp import FastMCP

from imas_codex.graph.client import GraphClient
from imas_codex.providers import MCPProvider

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
    """Container for graph-backed DD tool instances.

    No facade methods — callers access tool instances directly:
        tools.path_tool.check_dd_paths(...)
        tools.search_tool.search_dd_paths(...)
        tools.version_tool.get_dd_versions(...)
    """

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
        self.path_context_tool = GraphPathContextTool(graph_client)
        self.structure_tool = GraphStructureTool(graph_client)
        self.version_tool = VersionTool(graph_client)

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


__all__ = [
    "GraphSearchTool",
    "GraphPathTool",
    "GraphListTool",
    "GraphOverviewTool",
    "GraphClustersTool",
    "GraphIdentifiersTool",
    "VersionTool",
    "Tools",
]
