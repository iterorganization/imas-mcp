"""
Base search engine classes for IMAS MCP.

This module provides the abstract base classes for implementing different
search engines (semantic, lexical, hybrid) in a clean, composable way.
"""

import logging
from abc import ABC, abstractmethod

from imas_mcp.search.search_strategy import SearchConfig, SearchMatch, SearchResponse

logger = logging.getLogger(__name__)


class SearchEngine(ABC):
    """Abstract base class for all search engines.

    This class defines the interface that all search engines must implement.
    It enforces a clean separation between search logic and other concerns
    like caching, validation, and AI enhancement.
    """

    def __init__(self, name: str):
        """Initialize the search engine with a name for identification."""
        self.name = name
        self.logger = logger

    @abstractmethod
    async def search(
        self, query: str | list[str], config: SearchConfig
    ) -> SearchResponse:
        """Execute search with given query and configuration.

        Args:
            query: Search query string or list of strings
            config: Search configuration with parameters like max_results, filters

        Returns:
            SearchResponse containing limited hits and all matching paths

        Raises:
            SearchEngineError: When search execution fails
        """
        pass

    @abstractmethod
    def get_engine_type(self) -> str:
        """Get the type identifier for this engine.

        Returns:
            String identifier like 'semantic', 'lexical', 'hybrid'
        """
        pass

    def validate_query(self, query: str | list[str]) -> bool:
        """Validate if the query is suitable for this engine.

        Args:
            query: Query to validate

        Returns:
            True if query is valid for this engine
        """
        if isinstance(query, str):
            return len(query.strip()) > 0
        elif isinstance(query, list):
            return len(query) > 0 and any(len(q.strip()) > 0 for q in query)
        return False

    def normalize_query(self, query: str | list[str]) -> str:
        """Normalize query to standard string format.

        Args:
            query: Query string or list of strings

        Returns:
            Normalized query string
        """
        if isinstance(query, str):
            return query.strip()
        elif isinstance(query, list):
            return " ".join(q.strip() for q in query if q.strip())
        return ""

    def log_search_execution(
        self, query: str | list[str], config: SearchConfig, result_count: int
    ) -> None:
        """Log search execution for monitoring and debugging.

        Args:
            query: The search query
            config: Search configuration used
            result_count: Number of results returned
        """
        query_str = self.normalize_query(query)
        self.logger.info(
            f"[{self.name}] Search executed: query='{query_str[:50]}...' "
            f"search_mode={config.search_mode.value} max_results={config.max_results} "
            f"returned={result_count}"
        )


class SearchEngineError(Exception):
    """Exception raised when search engine operations fail."""

    def __init__(self, engine_name: str, message: str, query: str = ""):
        """Initialize search engine error.

        Args:
            engine_name: Name of the engine that failed
            message: Error description
            query: Query that caused the error (optional)
        """
        self.engine_name = engine_name
        self.query = query
        super().__init__(f"[{engine_name}] {message}")


class MockSearchEngine(SearchEngine):
    """Mock search engine for testing and development.

    This engine provides predictable results for testing the search service
    architecture without requiring full search infrastructure.
    """

    def __init__(self):
        super().__init__("mock")

    async def search(
        self, query: str | list[str], config: SearchConfig
    ) -> SearchResponse:
        """Return mock search results for testing."""
        from imas_mcp.models.constants import SearchMode
        from imas_mcp.search.document_store import Document, DocumentMetadata

        # Create mock document for testing
        mock_metadata = DocumentMetadata(
            path_name="core_profiles/profiles_1d/electrons/temperature",
            path_id="mock_path_1",
            ids_name="core_profiles",
            data_type="temperature",
            physics_domain="plasma_core",
        )

        mock_document = Document(
            metadata=mock_metadata,
            documentation=f"Mock result for query: {self.normalize_query(query)}",
            units=None,
        )

        # Create mock search result
        mock_result = SearchMatch(
            document=mock_document,
            score=0.95,
            rank=0,
            search_mode=SearchMode.SEMANTIC,
            highlights="temperature",
        )

        mock_hits = [mock_result]

        return SearchResponse(hits=mock_hits)

    def get_engine_type(self) -> str:
        """Get engine type identifier."""
        return "mock"
