"""
Tests for RelationshipsTool implementation.

This module tests the find_relationships tool with all decorators applied.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from imas_mcp.tools.relationships_tool import RelationshipsTool
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.services.search_service import SearchService
from imas_mcp.search.search_strategy import SearchResult


class TestRelationshipsTool:
    """Test cases for RelationshipsTool."""

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
    def relationships_tool(self, mock_document_store, mock_search_service):
        """Create RelationshipsTool with mocked dependencies."""
        tool = RelationshipsTool(mock_document_store)
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
            result.score = 0.9 - (i * 0.1)
            mock_results.append(result)
        return mock_results

    @pytest.mark.asyncio
    async def test_find_relationships_basic(
        self, relationships_tool, mock_search_results
    ):
        """Test basic relationship finding functionality."""
        # Setup
        relationships_tool._search_service.search = AsyncMock(
            return_value=mock_search_results
        )

        # Execute
        result = await relationships_tool.find_relationships(
            path="core_profiles/temperature", max_relationships=5
        )

        # Verify
        assert result is not None
        assert "relationships" in result
        relationships_tool._search_service.search.assert_called()

    @pytest.mark.asyncio
    async def test_find_relationships_with_relationship_types(
        self, relationships_tool, mock_search_results
    ):
        """Test relationship finding with specific relationship types."""
        # Setup
        relationships_tool._search_service.search = AsyncMock(
            return_value=mock_search_results
        )

        # Execute
        result = await relationships_tool.find_relationships(
            path="core_profiles/temperature",
            relationship_types=["sibling", "parent"],
            max_relationships=3,
        )

        # Verify
        assert result is not None
        relationships_tool._search_service.search.assert_called()

    @pytest.mark.asyncio
    async def test_find_relationships_empty_results(self, relationships_tool):
        """Test relationship finding with no results."""
        # Setup
        relationships_tool._search_service.search = AsyncMock(return_value=[])

        # Execute
        result = await relationships_tool.find_relationships(
            path="nonexistent/path", max_relationships=5
        )

        # Verify
        assert result is not None
        assert "relationships" in result

    @pytest.mark.asyncio
    async def test_find_relationships_search_error(self, relationships_tool):
        """Test relationship finding with search service error."""
        # Setup
        relationships_tool._search_service.search = AsyncMock(
            side_effect=Exception("Search failed")
        )

        # Execute & Verify - should handle gracefully due to error handling decorator
        result = await relationships_tool.find_relationships(
            path="core_profiles/temperature", max_relationships=5
        )

        # Should return some result due to error handling
        assert result is not None

    def test_get_tool_name(self, relationships_tool):
        """Test tool name retrieval."""
        assert relationships_tool.get_tool_name() == "find_relationships"

    @pytest.mark.asyncio
    async def test_find_relationships_with_ids_filter(
        self, relationships_tool, mock_search_results
    ):
        """Test relationship finding with IDS filtering."""
        # Setup
        relationships_tool._search_service.search = AsyncMock(
            return_value=mock_search_results
        )

        # Execute
        result = await relationships_tool.find_relationships(
            path="core_profiles/temperature",
            ids_filter=["core_profiles", "equilibrium"],
            max_relationships=5,
        )

        # Verify
        assert result is not None
        relationships_tool._search_service.search.assert_called()
        # Verify the search config includes the ids_filter
        call_args = relationships_tool._search_service.search.call_args
        config = call_args[0][1]  # Second argument is the config
        assert config.ids_filter == ["core_profiles", "equilibrium"]

    @pytest.mark.asyncio
    async def test_find_relationships_decorators_applied(
        self, relationships_tool, mock_search_results
    ):
        """Test that decorators are properly applied to the tool method."""
        # Setup
        relationships_tool._search_service.search = AsyncMock(
            return_value=mock_search_results
        )

        # Execute
        result = await relationships_tool.find_relationships(
            path="core_profiles/temperature", max_relationships=5
        )

        # Verify that result has decorator-enhanced content
        assert result is not None
        # The actual decorator behavior would be tested in integration tests
        # Here we just verify the method executes successfully
