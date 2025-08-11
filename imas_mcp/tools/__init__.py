"""
IMAS MCP Tools Package.

This package contains the refactored Tools implementation split into focused modules.
Each module handles a specific tool functionality with clean separation of concerns.
"""

from typing import Optional

from fastmcp import FastMCP

from imas_mcp.providers import MCPProvider
from imas_mcp.search.document_store import DocumentStore

from .analysis_tool import AnalysisTool

# Import individual tool classes
from .base import BaseTool
from .explain_tool import ExplainTool
from .export_tool import ExportTool
from .identifiers_tool import IdentifiersTool
from .overview_tool import OverviewTool
from .relationships_tool import RelationshipsTool
from .search_tool import SearchTool


class Tools(MCPProvider):
    """Main Tools class that delegates to individual tool implementations."""

    def __init__(self, ids_set: set[str] | None = None):
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
        self.explain_tool = ExplainTool(self.document_store)
        self.overview_tool = OverviewTool(self.document_store)
        self.analysis_tool = AnalysisTool(self.document_store)
        self.relationships_tool = RelationshipsTool(self.document_store)
        self.identifiers_tool = IdentifiersTool(self.document_store)
        self.export_tool = ExportTool(self.document_store)

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

    async def explain_concept(self, *args, **kwargs):
        """Delegate to explain tool."""
        return await self.explain_tool.explain_concept(*args, **kwargs)

    async def get_overview(self, *args, **kwargs):
        """Delegate to overview tool."""
        return await self.overview_tool.get_overview(*args, **kwargs)

    async def explore_identifiers(self, *args, **kwargs):
        """Delegate to identifiers tool."""
        return await self.identifiers_tool.explore_identifiers(*args, **kwargs)

    async def analyze_ids_structure(self, *args, **kwargs):
        """Delegate to analysis tool."""
        return await self.analysis_tool.analyze_ids_structure(*args, **kwargs)

    async def explore_relationships(self, *args, **kwargs):
        """Delegate to relationships tool."""
        return await self.relationships_tool.explore_relationships(*args, **kwargs)

    async def export_ids(self, *args, **kwargs):
        """Delegate to export tool."""
        return await self.export_tool.export_ids(*args, **kwargs)

    async def export_physics_domain(self, *args, **kwargs):
        """Delegate to export tool."""
        return await self.export_tool.export_physics_domain(*args, **kwargs)


__all__ = [
    "BaseTool",
    "SearchTool",
    "ExplainTool",
    "OverviewTool",
    "AnalysisTool",
    "RelationshipsTool",
    "IdentifiersTool",
    "ExportTool",
    "Tools",
]
