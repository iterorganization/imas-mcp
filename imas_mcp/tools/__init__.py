"""IMAS MCP Tools Package.

This package contains the refactored Tools implementation split into focused modules.
Each module handles a specific tool functionality with clean separation of concerns.
"""

from fastmcp import FastMCP

from imas_mcp.providers import MCPProvider
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.services.docs_server_manager import DocsServerManager

# Import individual tool classes
from .analysis_tool import AnalysisTool
from .base import BaseTool
from .docs_tool import DocsTool
from .explain_tool import ExplainTool
from .export_tool import ExportTool
from .identifiers_tool import IdentifiersTool
from .list_tool import ListTool
from .overview_tool import OverviewTool
from .path_tool import PathTool
from .relationships_tool import RelationshipsTool
from .search_tool import SearchTool


class Tools(MCPProvider):
    """Main Tools class that delegates to individual tool implementations."""

    def __init__(
        self,
        ids_set: set[str] | None = None,
        docs_manager: DocsServerManager | None = None,
    ):
        """Initialize the IMAS tools provider.

        Args:
            ids_set: Optional set of IDS names to limit processing to.
                    If None, will process all available IDS.
            docs_manager: Optional shared docs server manager for documentation tools.
        """
        self.ids_set = ids_set

        # Create shared DocumentStore with ids_set
        self.document_store = DocumentStore(ids_set=ids_set)

        # Initialize individual tools with shared document store
        self.search_tool = SearchTool(self.document_store)
        self.path_tool = PathTool(self.document_store)
        self.list_tool = ListTool(self.document_store)
        self.explain_tool = ExplainTool(self.document_store)
        self.overview_tool = OverviewTool(self.document_store)
        self.analysis_tool = AnalysisTool(self.document_store)
        self.relationships_tool = RelationshipsTool(self.document_store)
        self.identifiers_tool = IdentifiersTool(self.document_store)
        self.export_tool = ExportTool(self.document_store)

        # Initialize docs tool with injected docs manager
        if docs_manager is None:
            from imas_mcp.services.docs_server_manager import DocsServerManager

            docs_manager = DocsServerManager()
        self.docs_tool = DocsTool(docs_manager)

    @property
    def name(self) -> str:
        """Provider name for logging and identification."""
        return "tools"

    def register(self, mcp: FastMCP):
        """Register all IMAS tools with the MCP server."""
        # Register tools from each module
        for tool in [
            self.search_tool,
            self.path_tool,
            self.list_tool,
            self.explain_tool,
            self.overview_tool,
            self.analysis_tool,
            self.relationships_tool,
            self.identifiers_tool,
            self.export_tool,
            self.docs_tool,
        ]:
            for attr_name in dir(tool):
                attr = getattr(tool, attr_name)
                if hasattr(attr, "_mcp_tool") and attr._mcp_tool:
                    mcp.tool(description=attr._mcp_description)(attr)

    # Primary method delegation
    async def search_imas(self, *args, **kwargs):
        """Delegate to search tool."""
        return await self.search_tool.search_imas(*args, **kwargs)

    async def check_imas_paths(self, *args, **kwargs):
        """Delegate to path tool."""
        return await self.path_tool.check_imas_paths(*args, **kwargs)

    async def fetch_imas_paths(self, *args, **kwargs):
        """Delegate to path tool."""
        return await self.path_tool.fetch_imas_paths(*args, **kwargs)

    async def list_imas_paths(self, *args, **kwargs):
        """Delegate to list tool."""
        return await self.list_tool.list_imas_paths(*args, **kwargs)

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

    async def update_imas_path(self, *args, **kwargs):
        """Delegate to path tool."""
        return await self.path_tool.update_imas_path(*args, **kwargs)

    # Documentation search delegation methods
    async def search_docs(self, *args, **kwargs):
        """Delegate to docs tool."""
        return await self.docs_tool.search_docs(*args, **kwargs)

    async def list_docs(self, *args, **kwargs):
        """Delegate to docs tool."""
        return await self.docs_tool.list_docs(*args, **kwargs)


__all__ = [
    "BaseTool",
    "SearchTool",
    "PathTool",
    "ListTool",
    "ExplainTool",
    "OverviewTool",
    "AnalysisTool",
    "RelationshipsTool",
    "IdentifiersTool",
    "ExportTool",
    "DocsTool",
    "Tools",
]
