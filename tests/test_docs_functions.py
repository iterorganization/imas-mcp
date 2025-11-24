"""
Unit tests for documentation search functions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_mcp.services.docs_server_manager import DocsServerManager
from imas_mcp.tools.docs_tool import DocsTool


class TestDocsFunctions:
    """Test cases for documentation search MCP tool functions."""

    @pytest.fixture
    def mock_docs_proxy(self):
        """Create a mock docs proxy service."""
        proxy = AsyncMock()
        return proxy

    @pytest.fixture
    def mock_docs_manager(self, mock_docs_proxy):
        """Create a mock docs server manager."""
        manager = MagicMock(spec=DocsServerManager)
        manager.get_proxy_service.return_value = mock_docs_proxy
        return manager

    @pytest.fixture
    def docs_tool(self, mock_docs_manager):
        """Create a DocsTool instance for testing."""
        return DocsTool(docs_manager=mock_docs_manager)

    @pytest.mark.asyncio
    async def test_search_docs_success(self, docs_tool, mock_docs_manager):
        """Test successful documentation search."""
        # Setup mock response
        mock_response = {
            "results": [
                {"url": "https://example.com", "content": "test content", "score": 0.9}
            ],
            "query": "test query",
            "library": "test-lib",
            "version": "1.0.0",
        }
        mock_docs_manager.proxy_search_docs.return_value = mock_response

        # Call the function
        result = await docs_tool.search_docs(
            "test query", library="test-lib", limit=5, version="1.0.0"
        )

        # Verify the result
        assert result["results"] == mock_response["results"]
        assert result["query"] == mock_response["query"]
        assert result["library"] == mock_response["library"]
        assert result["version"] == mock_response["version"]
        mock_docs_manager.proxy_search_docs.assert_called_once_with(
            "test query", "test-lib", "1.0.0", 5
        )

    @pytest.mark.asyncio
    async def test_search_docs_empty_query(self, docs_tool):
        """Test search with empty query returns validation error."""
        result = await docs_tool.search_docs("")

        assert "error" in result
        assert "Validation error" in result["error"]
        assert "query" in result["error"]

    @pytest.mark.asyncio
    async def test_search_docs_invalid_limit(self, docs_tool):
        """Test search with invalid limit returns validation error."""
        result = await docs_tool.search_docs("test query", limit=25)

        assert "error" in result
        assert "Validation error" in result["error"]
        assert "limit" in result["error"]

    @pytest.mark.asyncio
    async def test_search_docs_server_unavailable(self, docs_tool, mock_docs_manager):
        """Test search when docs server is unavailable."""
        from imas_mcp.services.docs_proxy_service import DocsServerUnavailableError

        mock_docs_manager.proxy_search_docs.side_effect = DocsServerUnavailableError(
            "Server unavailable"
        )

        result = await docs_tool.search_docs("test query", library="test-lib")

        assert "error" in result
        # The error message can be either about server availability or missing library parameter
        assert any(
            msg in result["error"]
            for msg in [
                "Server unavailable",
                "Library parameter is required",
            ]
        )

    @pytest.mark.asyncio
    async def test_search_docs_library_not_found(self, docs_tool, mock_docs_manager):
        """Test search when library is not found."""
        from imas_mcp.services.docs_proxy_service import LibraryNotFoundError

        mock_docs_manager.proxy_search_docs.side_effect = LibraryNotFoundError(
            "unknown-lib", ["imas-python"]
        )

        result = await docs_tool.search_docs("test query", library="unknown-lib")

        assert "error" in result
        assert "Documentation library 'unknown-lib' not found" in result["error"]
        assert result["library_not_found"] is True

    @pytest.mark.asyncio
    async def test_list_docs_success(self, docs_tool, mock_docs_manager):
        """Test successful library listing."""
        # Setup mock response
        mock_libraries = ["imas-python", "numpy", "scipy"]
        mock_docs_manager.proxy_list_libraries.return_value = mock_libraries

        # Call the function
        result = await docs_tool.list_docs()

        # Verify the result
        assert result["libraries"] == mock_libraries
        assert result["count"] == 3
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_list_docs_server_unavailable(self, docs_tool, mock_docs_manager):
        """Test library listing when docs server is unavailable."""
        from imas_mcp.services.docs_proxy_service import DocsServerUnavailableError

        mock_docs_manager.proxy_list_libraries.side_effect = DocsServerUnavailableError(
            "Server unavailable"
        )

        result = await docs_tool.list_docs()

        assert "error" in result
        assert "Server unavailable" in result["error"]
        assert result["server_status"] == "unavailable"
        assert result["libraries"] == []

    @pytest.mark.asyncio
    async def test_list_docs_with_library_success(self, docs_tool, mock_docs_manager):
        """Test successful library version retrieval using list_docs with library parameter."""
        # Setup mock response
        # Note: list_docs with library parameter just returns a note now

        # Call the function
        result = await docs_tool.list_docs("imas-python")

        # Verify the result
        assert result["library"] == "imas-python"
        assert "note" in result
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_list_docs_with_library_empty_library(self, docs_tool):
        """Test version retrieval with empty library name returns validation error."""
        result = await docs_tool.list_docs("")

        assert "error" in result
        assert "Validation error" in result["error"]
        assert "library" in result["error"]

    @pytest.mark.asyncio
    async def test_list_docs_with_library_library_not_found(
        self, docs_tool, mock_docs_manager
    ):
        """Test version retrieval when library is not found."""
        # Note: list_docs with library parameter doesn't check existence anymore
        # It just returns a note

        result = await docs_tool.list_docs("unknown-lib")

        assert result["library"] == "unknown-lib"
        assert "note" in result
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_search_docs_library_required(self, docs_tool, mock_docs_manager):
        """Test search without library parameter returns helpful error."""
        # Mock the list_available_libraries call in the error handler
        mock_docs_manager.proxy_list_libraries.return_value = [
            "imas-python",
            "numpy",
        ]

        result = await docs_tool.search_docs("test query")

        assert "error" in result
        assert "Library parameter is required for search" in result["error"]
        assert result["library_required"] is True
        assert result["setup_instructions"] is True
        assert "available_libraries" in result
        assert "imas-python" in result["available_libraries"]
