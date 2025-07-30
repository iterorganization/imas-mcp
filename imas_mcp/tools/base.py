"""
Base tool functionality for IMAS MCP tools.

This module contains common functionality shared across all tool implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from imas_mcp.models.response_models import ErrorResponse

from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.services.search_service import SearchService
from imas_mcp.search.engines.semantic_engine import SemanticSearchEngine
from imas_mcp.search.engines.lexical_engine import LexicalSearchEngine
from imas_mcp.search.engines.hybrid_engine import HybridSearchEngine
from imas_mcp.models.constants import SearchMode

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Base class for all IMAS MCP tools."""

    def __init__(self, document_store: Optional[DocumentStore] = None):
        self.logger = logger
        self.document_store = document_store or DocumentStore()
        self._search_service = self._create_search_service()

    @abstractmethod
    def get_tool_name(self) -> str:
        """Return the name of this tool."""
        pass

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

    def _create_error_response(
        self, error_message: str, query: str = ""
    ) -> "ErrorResponse":
        """Create a standardized error response."""
        from imas_mcp.models.response_models import ErrorResponse

        return ErrorResponse(
            error=error_message,
            suggestions=[],
            context={
                "query": query,
                "tool": self.get_tool_name(),
                "status": "error",
            },
        )
