"""
Comprehensive test suite for SearchService architecture.

This module tests the SearchService class and related search functionality
using the updated architecture with proper parameter names.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from imas_mcp.search.services.search_service import SearchService, SearchServiceError
from imas_mcp.search.search_strategy import SearchConfig, SearchResult
from imas_mcp.models.constants import SearchMode
from imas_mcp.search.document_store import DocumentStore


@pytest.fixture
def mock_document_store():
    """Create a mock document store."""
    store = Mock(spec=DocumentStore)

    # Mock documents for testing
    mock_doc1 = Mock()
    mock_doc1.metadata.path_name = "core_profiles/temperature"
    mock_doc1.metadata.ids_name = "core_profiles"
    mock_doc1.metadata.physics_domain = "core_plasma"
    mock_doc1.metadata.data_type = "float"
    mock_doc1.documentation = "Temperature profile"
    mock_doc1.units.unit_str = "eV"

    mock_doc2 = Mock()
    mock_doc2.metadata.path_name = "equilibrium/rho_tor"
    mock_doc2.metadata.ids_name = "equilibrium"
    mock_doc2.metadata.physics_domain = "equilibrium"
    mock_doc2.metadata.data_type = "float"
    mock_doc2.documentation = "Toroidal flux coordinate"
    mock_doc2.units.unit_str = "m"

    store.documents = [mock_doc1, mock_doc2]
    return store


@pytest.fixture
def mock_search_results(mock_document_store):
    """Create mock search results."""
    results = []
    for i, doc in enumerate(mock_document_store.documents):
        result = SearchResult(
            document=doc,
            score=0.9 - (i * 0.1),
            rank=i + 1,
            search_mode=SearchMode.SEMANTIC,
            highlights=f"Highlight {i + 1}",
        )
        results.append(result)
    return results


class TestSearchService:
    """Test suite for SearchService."""

    def test_search_service_creation(self):
        """Test SearchService can be created."""
        service = SearchService()
        assert service is not None
        assert hasattr(service, "engines")

    @pytest.mark.asyncio
    async def test_search_service_semantic_mode(self, mock_search_results):
        """Test SearchService with semantic search mode."""
        service = SearchService()

        # Mock the semantic engine
        mock_engine = AsyncMock()
        mock_engine.search = AsyncMock(return_value=mock_search_results)
        service.engines = {SearchMode.SEMANTIC: mock_engine}

        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=5)

        results = await service.search("plasma temperature", config)

        assert isinstance(results, list)
        assert len(results) == len(mock_search_results)
        assert all(isinstance(result, SearchResult) for result in results)
        assert all(result.search_mode == SearchMode.SEMANTIC for result in results)

    @pytest.mark.asyncio
    async def test_search_service_lexical_mode(self, mock_search_results):
        """Test SearchService with lexical search mode."""
        service = SearchService()

        # Mock the lexical engine
        mock_engine = AsyncMock()
        mock_engine.search = AsyncMock(return_value=mock_search_results)
        service.engines = {SearchMode.LEXICAL: mock_engine}

        config = SearchConfig(search_mode=SearchMode.LEXICAL, max_results=10)

        results = await service.search("electron density", config)

        assert isinstance(results, list)
        assert len(results) == len(mock_search_results)
        mock_engine.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_service_hybrid_mode(self, mock_search_results):
        """Test SearchService with hybrid search mode."""
        service = SearchService()

        # Mock the hybrid engine
        mock_engine = AsyncMock()
        mock_engine.search = AsyncMock(return_value=mock_search_results)
        service.engines = {SearchMode.HYBRID: mock_engine}

        config = SearchConfig(search_mode=SearchMode.HYBRID, max_results=5)

        results = await service.search("plasma physics", config)

        assert isinstance(results, list)
        assert len(results) == len(mock_search_results)
        mock_engine.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_service_with_ids_filter(self, mock_search_results):
        """Test SearchService with IDS filtering."""
        service = SearchService()

        # Mock the semantic engine
        mock_engine = AsyncMock()
        mock_engine.search = AsyncMock(return_value=mock_search_results)
        service.engines = {SearchMode.SEMANTIC: mock_engine}

        config = SearchConfig(
            search_mode=SearchMode.SEMANTIC,
            max_results=5,
            ids_filter=["core_profiles", "equilibrium"],
        )

        results = await service.search("temperature", config)

        assert isinstance(results, list)
        mock_engine.search.assert_called_once()

        # Verify config was passed correctly
        call_args = mock_engine.search.call_args
        passed_config = call_args[0][1]  # Second argument is config
        assert passed_config.ids_filter == ["core_profiles", "equilibrium"]

    @pytest.mark.asyncio
    async def test_search_service_auto_mode_resolution(self, mock_search_results):
        """Test SearchService AUTO mode resolution."""
        service = SearchService()

        # Mock the resolved engine (should resolve to semantic for conceptual query)
        mock_engine = AsyncMock()
        mock_engine.search = AsyncMock(return_value=mock_search_results)
        service.engines = {SearchMode.SEMANTIC: mock_engine}

        config = SearchConfig(search_mode=SearchMode.AUTO, max_results=5)

        results = await service.search("plasma temperature physics", config)

        assert isinstance(results, list)
        # AUTO mode should be resolved to a specific mode
        mock_engine.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_service_empty_results(self):
        """Test SearchService with no results."""
        service = SearchService()

        # Mock engine that returns empty results
        mock_engine = AsyncMock()
        mock_engine.search = AsyncMock(return_value=[])
        service.engines = {SearchMode.LEXICAL: mock_engine}

        config = SearchConfig(search_mode=SearchMode.LEXICAL, max_results=5)

        results = await service.search("nonexistent_query", config)

        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_service_error_handling(self):
        """Test SearchService error handling."""
        service = SearchService()

        # Mock engine that raises an exception
        mock_engine = AsyncMock()
        mock_engine.search = AsyncMock(side_effect=Exception("Search engine error"))
        service.engines = {SearchMode.SEMANTIC: mock_engine}

        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=5)

        with pytest.raises(SearchServiceError):
            await service.search("test query", config)

    @pytest.mark.asyncio
    async def test_search_service_list_query(self, mock_search_results):
        """Test SearchService with list query input."""
        service = SearchService()

        # Mock the semantic engine
        mock_engine = AsyncMock()
        mock_engine.search = AsyncMock(return_value=mock_search_results)
        service.engines = {SearchMode.SEMANTIC: mock_engine}

        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=5)

        query_list = ["plasma", "temperature", "profile"]
        results = await service.search(query_list, config)

        assert isinstance(results, list)
        mock_engine.search.assert_called_once()

        # Verify the query was properly formatted
        call_args = mock_engine.search.call_args
        passed_query = call_args[0][0]  # First argument is query
        assert isinstance(passed_query, (str, list))


class TestSearchConfig:
    """Test suite for SearchConfig with updated parameters."""

    def test_search_config_defaults(self):
        """Test SearchConfig default values."""
        config = SearchConfig()
        assert config.search_mode == SearchMode.AUTO
        assert config.max_results == 10
        assert config.ids_filter is None
        assert config.similarity_threshold == 0.0
        assert config.boost_exact_matches is True
        assert config.enable_physics_enhancement is True

    def test_search_config_custom_values(self):
        """Test SearchConfig with custom values."""
        config = SearchConfig(
            search_mode=SearchMode.SEMANTIC,
            max_results=20,
            ids_filter=["core_profiles", "equilibrium"],
            similarity_threshold=0.5,
            boost_exact_matches=False,
            enable_physics_enhancement=False,
        )
        assert config.search_mode == SearchMode.SEMANTIC
        assert config.max_results == 20
        assert config.ids_filter == ["core_profiles", "equilibrium"]
        assert config.similarity_threshold == 0.5
        assert config.boost_exact_matches is False
        assert config.enable_physics_enhancement is False

    def test_search_config_field_validators(self):
        """Test SearchConfig field validators."""
        # Test search_mode string conversion
        config = SearchConfig(search_mode="semantic")
        assert config.search_mode == SearchMode.SEMANTIC

        # Test ids_filter string conversion
        config = SearchConfig(ids_filter="core_profiles")
        assert config.ids_filter == ["core_profiles"]

        # Test ids_filter list handling
        config = SearchConfig(ids_filter=["core_profiles", "equilibrium"])
        assert config.ids_filter == ["core_profiles", "equilibrium"]


class TestSearchResult:
    """Test suite for SearchResult model."""

    def test_search_result_creation(self, mock_document_store):
        """Test SearchResult creation."""
        doc = mock_document_store.documents[0]

        result = SearchResult(
            document=doc,
            score=0.85,
            rank=1,
            search_mode=SearchMode.SEMANTIC,
            highlights="Test highlights",
        )

        assert result.document == doc
        assert result.score == 0.85
        assert result.rank == 1
        assert result.search_mode == SearchMode.SEMANTIC
        assert result.highlights == "Test highlights"

    def test_search_result_field_access(self, mock_document_store):
        """Test direct field access on SearchResult."""
        doc = mock_document_store.documents[0]

        result = SearchResult(
            document=doc, score=0.9, rank=2, search_mode=SearchMode.LEXICAL
        )

        # Test direct field access (no to_dict needed)
        assert result.document.metadata.path_name == "core_profiles/temperature"
        assert result.document.metadata.ids_name == "core_profiles"
        assert result.document.documentation == "Temperature profile"
        assert result.score == 0.9
        assert result.rank == 2
        assert result.search_mode == SearchMode.LEXICAL
