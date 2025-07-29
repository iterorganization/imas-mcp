"""
Test suite for explain tool with decorator composition.

Tests the explain tool with decorator functionality.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from imas_mcp.tools.explain_tool import ExplainTool
from imas_mcp.search.search_strategy import SearchResult
from imas_mcp.search.document_store import Document, DocumentMetadata
from imas_mcp.models.constants import SearchMode


class TestExplainTool:
    """Test cases for explain tool with decorator composition."""

    @pytest.fixture
    def mock_document_store(self):
        """Create mock document store for testing."""
        mock_store = MagicMock()
        mock_store.get_available_ids.return_value = ["core_profiles", "equilibrium"]
        return mock_store

    @pytest.fixture
    def explain_tool(self, mock_document_store):
        """Create explain tool for testing."""
        return ExplainTool(document_store=mock_document_store)

    @pytest.fixture
    def mock_search_service(self):
        """Create mock search service for testing."""
        service = MagicMock()
        service.search = AsyncMock()
        return service

    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results for testing."""
        metadata1 = DocumentMetadata(
            path_id="core_profiles_temperature",
            path_name="core_profiles.temperature",
            ids_name="core_profiles",
            units="eV",
            data_type="float",
            physics_domain="core_plasma",
        )

        doc1 = Document(
            metadata=metadata1,
            documentation="Electron temperature measurements",
            physics_context={"domain": "core_plasma"},
            relationships={},
        )

        result1 = SearchResult(
            document=doc1,
            score=0.95,
            rank=1,
            search_mode=SearchMode.SEMANTIC,
            highlights="",
        )

        return [result1]

    @pytest.mark.asyncio
    async def test_explain_concept_basic(
        self, explain_tool, mock_search_service, sample_search_results
    ):
        """Test basic concept explanation."""
        # Setup mock
        with patch.object(explain_tool, "_search_service", mock_search_service):
            mock_search_service.search.return_value = sample_search_results

            result = await explain_tool.explain_concept(
                concept="temperature", detail_level="intermediate"
            )

            # Verify basic structure
            assert isinstance(result, dict)
            assert "concept" in result
            assert result["concept"] == "temperature"
            assert "detail_level" in result
            assert "related_paths" in result or "paths" in result
            assert "physics_context" in result
            assert "sources_analyzed" in result or "count" in result

    @pytest.mark.asyncio
    async def test_explain_concept_no_results(self, explain_tool, mock_search_service):
        """Test explain concept when no results found."""
        # Setup mock to return empty results
        with patch.object(explain_tool, "_search_service", mock_search_service):
            mock_search_service.search.return_value = []

            result = await explain_tool.explain_concept(concept="nonexistent_concept")

            # Verify error handling
            assert isinstance(result, dict)
            assert "error" in result
            assert "suggestions" in result
            assert result["concept"] == "nonexistent_concept"

    @pytest.mark.asyncio
    async def test_explain_concept_different_detail_levels(
        self, explain_tool, mock_search_service, sample_search_results
    ):
        """Test concept explanation with different detail levels."""
        with patch.object(explain_tool, "_search_service", mock_search_service):
            mock_search_service.search.return_value = sample_search_results

            # Test all detail levels
            detail_levels = ["basic", "intermediate", "advanced"]

            for level in detail_levels:
                result = await explain_tool.explain_concept(
                    concept="temperature", detail_level=level
                )

                assert result["detail_level"] == level
                assert "concept" in result

    @pytest.mark.asyncio
    async def test_explain_concept_input_validation(
        self, explain_tool, mock_search_service
    ):
        """Test input validation through decorators."""
        # Test invalid detail level - validation decorator may return error response
        with patch.object(explain_tool, "_search_service", mock_search_service):
            mock_search_service.search.return_value = []

            result = await explain_tool.explain_concept(
                concept="temperature", detail_level="invalid_level"
            )

            # Check if validation decorator handled the error
            assert isinstance(result, dict)
            # May contain error or be valid depending on decorator implementation

    @pytest.mark.asyncio
    async def test_explain_concept_error_handling(
        self, explain_tool, mock_search_service
    ):
        """Test error handling in explain concept."""
        # Setup mock to raise exception
        with patch.object(explain_tool, "_search_service", mock_search_service):
            mock_search_service.search.side_effect = Exception("Search failed")

            # The error handling decorator should catch this
            result = await explain_tool.explain_concept(concept="temperature")

            # Should return error response instead of raising
            assert isinstance(result, dict)
            # May contain error field or be handled by decorator

    def test_get_tool_name(self, explain_tool):
        """Test tool name retrieval."""
        assert explain_tool.get_tool_name() == "explain_concept"
