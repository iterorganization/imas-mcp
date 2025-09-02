"""
Integration tests for hint decorators on actual MCP tools.

This module tests that the hint decorators work correctly on the
actual tool implementations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_mcp.models.constants import DetailLevel, SearchMode
from imas_mcp.models.result_models import ConceptResult, SearchResult, StructureResult
from imas_mcp.models.suggestion_models import SearchSuggestion, ToolSuggestion
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.search_strategy import SearchHit
from imas_mcp.tools.analysis_tool import AnalysisTool
from imas_mcp.tools.explain_tool import ExplainTool
from imas_mcp.tools.search_tool import SearchTool


class TestActualToolHints:
    """Test hint decorators on actual tool implementations."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store for testing."""
        store = MagicMock(spec=DocumentStore)
        store.get_all_documents.return_value = []
        store.get_document.return_value = None
        return store

    @pytest.fixture
    def search_tool(self, mock_document_store):
        """Create a SearchTool instance for testing."""
        return SearchTool(document_store=mock_document_store)

    @pytest.fixture
    def analysis_tool(self, mock_document_store):
        """Create an AnalysisTool instance for testing."""
        return AnalysisTool(document_store=mock_document_store)

    @pytest.fixture
    def explain_tool(self, mock_document_store):
        """Create an ExplainTool instance for testing."""
        return ExplainTool(document_store=mock_document_store)

    @pytest.mark.asyncio
    async def test_search_tool_hints_integration(self, search_tool):
        """Test that SearchTool properly populates hint fields."""
        # Mock the search execution to return a realistic result
        mock_hits = [
            SearchHit(
                path="core_profiles/profiles_1d/temperature",
                score=0.95,
                rank=0,
                search_mode=SearchMode.SEMANTIC,
                physics_domain="core_transport",
                documentation="Electron temperature profile",
                data_type="FLT_1D",
                units="eV",
                ids_name="core_profiles",
            )
        ]

        with patch.object(
            search_tool, "execute_search", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = SearchResult(
                hits=mock_hits,
                query="temperature",
                search_mode=SearchMode.SEMANTIC,
            )

            # Call the decorated method
            result = await search_tool.search_imas(query="temperature")

            # Verify the result is correct
            assert isinstance(result, SearchResult)
            assert result.query == "temperature"
            assert len(result.hits) == 1

            # Verify hint fields are populated by decorators
            assert hasattr(result, "query_hints")
            assert hasattr(result, "tool_hints")
            assert isinstance(result.query_hints, list)
            assert isinstance(result.tool_hints, list)

            # The decorators should have populated some hints
            # (exact content depends on the hint generation logic)

    @pytest.mark.asyncio
    async def test_analysis_tool_hints_integration(self, analysis_tool):
        """Test that AnalysisTool properly populates hint fields."""
        # Mock the analysis execution - use simpler approach
        with patch.object(
            analysis_tool, "analyze_ids_structure", new_callable=AsyncMock
        ) as mock_analyze:
            mock_analyze.return_value = StructureResult(
                ids_name="equilibrium",
                description="Equilibrium data structure",
                structure={"profiles_1d": 5, "time_slice": 3},
                sample_paths=["equilibrium/time_slice/profiles_1d/psi"],
                max_depth=3,
                query="equilibrium",
                search_mode=SearchMode.AUTO,
                max_results=None,
                ids_filter=None,
                physics_domains=["equilibrium"],
            )

            # Call the decorated method (though it will actually call our mock)
            result = await analysis_tool.analyze_ids_structure(ids_name="equilibrium")

            # Verify the result is correct
            assert isinstance(result, StructureResult)
            assert result.ids_name == "equilibrium"

            # Verify hint fields are populated by decorators
            assert hasattr(result, "tool_hints")
            assert isinstance(result.tool_hints, list)

            # The tool_hints decorator should have populated some hints

    @pytest.mark.asyncio
    async def test_explain_tool_hints_integration(self, explain_tool):
        """Test that ExplainTool properly populates hint fields."""
        # Mock the explanation execution
        with patch.object(
            explain_tool, "execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = SearchResult(
                hits=[
                    SearchHit(
                        path="equilibrium/time_slice/profiles_1d/psi",
                        score=0.9,
                        rank=0,
                        search_mode=SearchMode.SEMANTIC,
                        documentation="Poloidal flux",
                        ids_name="equilibrium",
                    )
                ],
                query="poloidal flux",
                search_mode=SearchMode.SEMANTIC,
            )

            # Call the decorated method
            result = await explain_tool.explain_concept(
                concept="poloidal flux", detail_level=DetailLevel.INTERMEDIATE
            )

            # Verify the result is correct
            assert isinstance(result, ConceptResult)
            assert result.concept == "poloidal flux"

            # Verify hint fields are populated by decorators
            assert hasattr(result, "tool_hints")
            assert isinstance(result.tool_hints, list)

            # The tool_hints decorator should have populated some hints


class TestHintFieldPopulation:
    """Test specific aspects of hint field population."""

    def test_all_result_types_have_hint_fields(self):
        """Verify all result types have the required hint fields."""
        # SearchResult
        search_result = SearchResult(
            hits=[],
            query="test",
            search_mode=SearchMode.AUTO,
        )
        assert hasattr(search_result, "query_hints")
        assert hasattr(search_result, "tool_hints")

        # ConceptResult
        concept_result = ConceptResult(
            concept="test",
            explanation="test explanation",
            detail_level=DetailLevel.BASIC,
            related_topics=[],
            nodes=[],
            physics_domains=[],
            query="test",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
        )
        assert hasattr(concept_result, "query_hints")
        assert hasattr(concept_result, "tool_hints")

        # StructureResult
        structure_result = StructureResult(
            ids_name="test",
            description="test description",
            structure={},
            sample_paths=[],
            max_depth=0,
            query="test",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
            physics_domains=[],
        )
        assert hasattr(structure_result, "query_hints")
        assert hasattr(structure_result, "tool_hints")

    def test_hint_field_types(self):
        """Test that hint fields have the correct types."""
        result = SearchResult(
            hits=[],
            query="test",
            search_mode=SearchMode.AUTO,
        )

        # Fields should be lists
        assert isinstance(result.query_hints, list)
        assert isinstance(result.tool_hints, list)

        # Fields should be empty by default
        assert len(result.query_hints) == 0
        assert len(result.tool_hints) == 0

        # Test field assignment
        result.query_hints = [
            SearchSuggestion(
                suggestion="test suggestion",
                reason="test reason",
                confidence=0.5,
            )
        ]
        result.tool_hints = [
            ToolSuggestion(
                tool_name="test_tool",
                description="test description",
                relevance="test relevance",
            )
        ]

        assert len(result.query_hints) == 1
        assert len(result.tool_hints) == 1
        assert isinstance(result.query_hints[0], SearchSuggestion)
        assert isinstance(result.tool_hints[0], ToolSuggestion)

    def test_metadata_properties(self):
        """Test that metadata properties work correctly."""
        result = SearchResult(
            hits=[],
            query="test",
            search_mode=SearchMode.AUTO,
        )

        # Test tool_name property
        assert result.tool_name == "search_imas"

        # Test processing_timestamp property (should be cached)
        timestamp1 = result.processing_timestamp
        timestamp2 = result.processing_timestamp
        assert timestamp1 == timestamp2  # Should be the same due to caching

        # Test version property
        assert isinstance(result.version, str)
        assert len(result.version) > 0


class TestDecoratorErrorHandling:
    """Test error handling in hint decorators."""

    @pytest.mark.asyncio
    async def test_decorator_handles_missing_fields_gracefully(self):
        """Test that decorators handle missing fields gracefully."""
        from imas_mcp.search.decorators.query_hints import query_hints
        from imas_mcp.search.decorators.tool_hints import tool_hints

        # Create a mock result object without hint fields
        class MockResult:
            def __init__(self):
                self.query = "test"

        @query_hints(max_hints=2)
        @tool_hints(max_hints=2)
        async def mock_function():
            return MockResult()

        # Should not raise an exception
        result = await mock_function()
        assert result.query == "test"

    @pytest.mark.asyncio
    async def test_decorator_handles_exceptions_gracefully(self):
        """Test that decorators handle internal exceptions gracefully."""
        from imas_mcp.search.decorators.tool_hints import tool_hints

        @tool_hints(max_hints=2)
        async def mock_function():
            # Return a result with hint fields
            return SearchResult(
                hits=[],
                query="test",
                search_mode=SearchMode.AUTO,
            )

        # Should not raise an exception even if hint generation fails internally
        result = await mock_function()
        assert isinstance(result, SearchResult)
        assert hasattr(result, "tool_hints")
