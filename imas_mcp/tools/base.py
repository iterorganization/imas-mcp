"""
Base tool functionality for IMAS MCP tools.

This module contains common functionality shared across all tool implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.error_models import ToolError
from imas_mcp.models.result_models import SearchResult
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.engines.hybrid_engine import HybridSearchEngine
from imas_mcp.search.engines.lexical_engine import LexicalSearchEngine
from imas_mcp.search.engines.semantic_engine import SemanticSearchEngine
from imas_mcp.search.services.search_service import SearchService
from imas_mcp.services import (
    DocumentService,
    PhysicsService,
    ResponseService,
    SearchConfigurationService,
)

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Base class for all IMAS MCP tools with service injection."""

    def __init__(self, document_store: DocumentStore | None = None):
        self.logger = logger
        self.document_store = document_store or DocumentStore()

        # Initialize search service
        self._search_service = self._create_search_service()

        # Initialize services
        self.physics = PhysicsService()
        self.response = ResponseService()
        self.documents = DocumentService(self.document_store)
        self.search_config = SearchConfigurationService()

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Return the name of this tool - must be implemented by subclasses."""
        pass

    # =====================================
    # CORE TOOL METHODS
    # =====================================

    async def execute_search(
        self,
        query: str,
        search_mode: str | SearchMode = "auto",
        max_results: int = 10,
        ids_filter: list[str] | None = None,
    ) -> SearchResult:
        """
        Unified search execution that returns a complete SearchResult.

        Args:
            query: Search query
            search_mode: Search mode to use
            max_results: Maximum results to return
            ids_filter: Optional IDS filter

        Returns:
            SearchResult with all search data and context
        """
        # Create and optimize configuration
        config = self.search_config.create_config(
            search_mode=search_mode,
            max_results=max_results,
            ids_filter=ids_filter,
        )
        config = self.search_config.optimize_for_query(query, config)

        # Execute search - now returns SearchResponse with hits and all_paths
        search_result = await self._search_service.search(query, config)

        # Build response using search response service
        response = self.response.build_search_response(
            results=search_result.hits,  # Extract hits from SearchResponse
            query=query,
            search_mode=config.search_mode,
            ids_filter=ids_filter,
            max_results=max_results,
        )

        # Build summary from returned hits
        if hasattr(response, "summary"):
            # Update existing summary with path information from hits
            path_list = [hit.document.metadata.path_name for hit in search_result.hits]
            response.summary.update({"path_list": path_list})
        else:
            # Create new summary from hits
            path_list = [hit.document.metadata.path_name for hit in search_result.hits]
            ids_coverage = {
                hit.document.metadata.ids_name for hit in search_result.hits
            }
            response.summary = {
                "hits_returned": len(search_result.hits),
                "path_list": path_list,
                "ids_coverage": sorted(ids_coverage),
            }

        return response

    def build_prompt(self, prompt_type: str, tool_context: dict[str, Any]) -> str:
        """Override in subclasses to build tool-specific AI prompts."""
        return ""

    def system_prompt(self) -> str:
        """Override in subclasses to provide tool-specific system prompts."""
        return "You are an expert assistant analyzing IMAS fusion physics data. Provide detailed, accurate insights."

    def build_sample_tasks(self, tool_result) -> list[dict[str, Any]]:
        """Build sampling tasks for sampling tool result content.

        Default implementation provides base sampling tasks for common fields.
        Override in subclasses to add tool-specific sampling tasks.

        Args:
            tool_result: The tool result to build sampling tasks for

        Returns:
            List of sampling tasks with field, prompt_type, and context
        """
        tasks = []

        # Base sampling for any tool with nodes
        if hasattr(tool_result, "nodes") and tool_result.nodes:
            tasks.append(
                {
                    "field": "nodes_documentation",
                    "prompt_type": "sample_documentation",
                    "context": {
                        "nodes": tool_result.nodes[:5],  # Limit context size
                        "query": getattr(tool_result, "query", ""),
                        "tool_type": type(self).__name__,
                    },
                }
            )

        return tasks

    def _create_search_service(self) -> SearchService:
        """Create search service with appropriate engines."""
        # Create engines for each mode
        engines = {}
        for mode in [SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID]:
            engine = self._create_engine(mode.value)
            engines[mode] = engine

        return SearchService(engines)

    def _create_engine(self, engine_type: str):
        """Create a search engine of the specified type."""
        engine_map = {
            "semantic": SemanticSearchEngine,
            "lexical": LexicalSearchEngine,
            "hybrid": HybridSearchEngine,
        }

        if engine_type not in engine_map:
            raise ValueError(f"Unknown engine type: {engine_type}")

        engine_class = engine_map[engine_type]
        return engine_class(self.document_store)

    def _create_error_response(self, error_message: str, query: str = "") -> ToolError:
        """Create a standardized error response."""
        return ToolError(
            error=error_message,
            suggestions=[],
            context={
                "query": query,
                "tool": self.tool_name,
                "status": "error",
            },
        )
