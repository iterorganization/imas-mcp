"""IMAS Codex Tools Package.

This package contains the Tools implementation split into focused modules.
Each module handles a specific tool functionality with clean separation of concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imas_codex.providers import MCPProvider
from imas_codex.search.document_store import DocumentStore

from .base import BaseTool
from .clusters_tool import ClustersTool
from .identifiers_tool import IdentifiersTool
from .list_tool import ListTool
from .overview_tool import OverviewTool
from .path_tool import PathTool
from .search_tool import SearchTool

if TYPE_CHECKING:
    from fastmcp import FastMCP  # Only used in type hints, not function signatures


class Tools(MCPProvider):
    """Main Tools class that delegates to individual tool implementations."""

    # Class-level registry of tool instances for dynamic discovery
    _tool_instances: list = []

    def __init__(
        self,
        ids_set: set[str] | None = None,
    ):
        """Initialize the IMAS tools provider.

        Args:
            ids_set: Optional set of IDS names to limit processing to.
                    If None, will process all available IDS.
        """
        self.ids_set = ids_set

        # Create shared DocumentStore with ids_set
        self.document_store = DocumentStore(ids_set=ids_set)

        # Initialize individual tools with shared document store
        self.search_tool = SearchTool(self.document_store)
        self.path_tool = PathTool(self.document_store)
        self.list_tool = ListTool(self.document_store)
        self.overview_tool = OverviewTool(self.document_store)
        self.clusters_tool = ClustersTool(self.document_store)
        self.identifiers_tool = IdentifiersTool(self.document_store)

        # Store tool instances for dynamic discovery
        self._tool_instances = [
            self.search_tool,
            self.path_tool,
            self.list_tool,
            self.overview_tool,
            self.clusters_tool,
            self.identifiers_tool,
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
                if hasattr(attr, "_mcp_tool") and attr._mcp_tool:
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
                    if hasattr(attr, "_mcp_tool") and attr._mcp_tool:
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
    "SearchTool",
    "PathTool",
    "ListTool",
    "OverviewTool",
    "ClustersTool",
    "IdentifiersTool",
    "Tools",
]
