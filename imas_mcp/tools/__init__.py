"""
IMAS MCP Tools Package.

This package contains the refactored Tools implementation split into focused modules.
Each module handles a specific tool functionality with clean separation of concerns.
"""

from typing import Optional
from fastmcp import FastMCP
from imas_mcp.providers import MCPProvider

# Import individual tool classes
from .base import BaseTool
from .search import Search
from .explain import Explain
from .overview import Overview
from .analysis import Analysis
from .relationships import Relationships
from .identifiers import Identifiers
from .export import Export


class Tools(MCPProvider):
    """Main Tools class that delegates to individual tool implementations."""

    def __init__(self, ids_set: Optional[set[str]] = None):
        """Initialize the IMAS tools provider.

        Args:
            ids_set: Optional set of IDS names to limit processing to.
                    If None, will process all available IDS.
        """
        self.ids_set = ids_set

        # Initialize individual tools
        self.search_tool = Search(ids_set)
        self.explain_tool = Explain()
        self.overview_tool = Overview()
        self.analysis_tool = Analysis()
        self.relationships_tool = Relationships()
        self.identifiers_tool = Identifiers()
        self.export_tool = Export()

    @property
    def name(self) -> str:
        """Provider name for logging and identification."""
        return "tools"

    def register(self, mcp: FastMCP):
        """Register all IMAS tools with the MCP server."""
        # Register tools from each module
        for tool in [
            self.search_tool,
            self.explain_tool,
            self.overview_tool,
            self.analysis_tool,
            self.relationships_tool,
            self.identifiers_tool,
            self.export_tool,
        ]:
            for attr_name in dir(tool):
                attr = getattr(tool, attr_name)
                if hasattr(attr, "_mcp_tool") and attr._mcp_tool:
                    mcp.tool(description=attr._mcp_description)(attr)

    # Primary method delegation
    async def search_imas(self, *args, **kwargs):
        """Delegate to search tool."""
        return await self.search_tool.search_imas(*args, **kwargs)

    # TODO: Implement other tool delegations when those tools are completed
    # async def explain_concept(self, *args, **kwargs):
    #     """Delegate to explain tool."""
    #     return await self.explain_tool.explain_concept(*args, **kwargs)


__all__ = [
    "BaseTool",
    "Search",
    "Explain",
    "Overview",
    "Analysis",
    "Relationships",
    "Identifiers",
    "Export",
    "Tools",  # Main Tools class for backward compatibility
]
