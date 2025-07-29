"""
Tests for ExportTool implementation.

This module tests the export_search_results tool with all decorators applied.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from imas_mcp.tools.export_tool import ExportTool
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.services.search_service import SearchService
from imas_mcp.search.search_strategy import SearchResult


class TestExportTool:
    """Test cases for ExportTool."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store."""
        mock_store = Mock(spec=DocumentStore)
        return mock_store

    @pytest.fixture
    def mock_search_service(self):
        """Create a mock search service."""
        mock_service = Mock(spec=SearchService)
        return mock_service

    @pytest.fixture
    def export_tool(self, mock_document_store, mock_search_service):
        """Create ExportTool with mocked dependencies."""
        tool = ExportTool(mock_document_store)
        tool._search_service = mock_search_service
        return tool

    @pytest.fixture
    def mock_search_results(self):
        """Create mock search results."""
        mock_results = []
        for i in range(3):
            result = Mock(spec=SearchResult)
            result.document.metadata.path_name = f"core_profiles/path_{i}"
            result.document.documentation = f"Documentation for path {i}"
            result.document.metadata.physics_domain = "core_plasma"
            result.document.metadata.ids_name = "core_profiles"
            result.document.metadata.data_type = "float"
            result.document.units.unit_str = "eV" if result.document.units else None
            result.score = 0.9 - (i * 0.1)
            mock_results.append(result)
        return mock_results

    @pytest.mark.asyncio
    async def test_export_search_results_csv(self, export_tool, mock_search_results):
        """Test exporting search results to CSV format."""
        # Setup
        export_tool._search_service.search = AsyncMock(return_value=mock_search_results)

        # Execute
        result = await export_tool.export_search_results(
            query="temperature", export_format="csv", max_results=10
        )

        # Verify
        assert result is not None
        assert "export_data" in result
        export_tool._search_service.search.assert_called()

    @pytest.mark.asyncio
    async def test_export_search_results_json(self, export_tool, mock_search_results):
        """Test exporting search results to JSON format."""
        # Setup
        export_tool._search_service.search = AsyncMock(return_value=mock_search_results)

        # Execute
        result = await export_tool.export_search_results(
            query="temperature", export_format="json", max_results=10
        )

        # Verify
        assert result is not None
        assert "export_data" in result
        export_tool._search_service.search.assert_called()

    @pytest.mark.asyncio
    async def test_export_search_results_with_ids_filter(
        self, export_tool, mock_search_results
    ):
        """Test exporting with IDS filtering."""
        # Setup
        export_tool._search_service.search = AsyncMock(return_value=mock_search_results)

        # Execute
        result = await export_tool.export_search_results(
            query="temperature",
            export_format="csv",
            ids_filter=["core_profiles"],
            max_results=10,
        )

        # Verify
        assert result is not None
        export_tool._search_service.search.assert_called()
        # Verify the search config includes the ids_filter
        call_args = export_tool._search_service.search.call_args
        config = call_args[0][1]  # Second argument is the config
        assert config.ids_filter == ["core_profiles"]

    @pytest.mark.asyncio
    async def test_export_search_results_empty_results(self, export_tool):
        """Test exporting with no search results."""
        # Setup
        export_tool._search_service.search = AsyncMock(return_value=[])

        # Execute
        result = await export_tool.export_search_results(
            query="nonexistent", export_format="csv", max_results=10
        )

        # Verify
        assert result is not None
        assert "export_data" in result

    @pytest.mark.asyncio
    async def test_export_search_results_search_error(self, export_tool):
        """Test export with search service error."""
        # Setup
        export_tool._search_service.search = AsyncMock(
            side_effect=Exception("Search failed")
        )

        # Execute & Verify - should handle gracefully due to error handling decorator
        result = await export_tool.export_search_results(
            query="temperature", export_format="csv", max_results=10
        )

        # Should return some result due to error handling
        assert result is not None

    def test_get_tool_name(self, export_tool):
        """Test tool name retrieval."""
        assert export_tool.get_tool_name() == "export_search_results"

    @pytest.mark.asyncio
    async def test_export_search_results_different_search_modes(
        self, export_tool, mock_search_results
    ):
        """Test exporting with different search modes."""
        # Setup
        export_tool._search_service.search = AsyncMock(return_value=mock_search_results)

        # Test semantic mode
        result = await export_tool.export_search_results(
            query="plasma temperature",
            export_format="json",
            search_mode="semantic",
            max_results=5,
        )

        assert result is not None
        export_tool._search_service.search.assert_called()

    @pytest.mark.asyncio
    async def test_export_search_results_decorators_applied(
        self, export_tool, mock_search_results
    ):
        """Test that decorators are properly applied to the tool method."""
        # Setup
        export_tool._search_service.search = AsyncMock(return_value=mock_search_results)

        # Execute
        result = await export_tool.export_search_results(
            query="temperature", export_format="csv", max_results=10
        )

        # Verify that result has decorator-enhanced content
        assert result is not None
        # The actual decorator behavior would be tested in integration tests
        # Here we just verify the method executes successfully

    @pytest.mark.asyncio
    async def test_export_search_results_invalid_format(
        self, export_tool, mock_search_results
    ):
        """Test export with invalid format parameter."""
        # Setup
        export_tool._search_service.search = AsyncMock(return_value=mock_search_results)

        # Execute with invalid format - should handle gracefully
        result = await export_tool.export_search_results(
            query="temperature", export_format="invalid_format", max_results=10
        )

        # Should return some result due to error handling
        assert result is not None
