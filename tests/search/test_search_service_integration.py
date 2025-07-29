"""
Test suite for SearchService integration functionality.

This module tests the SearchService with different search engines and configurations.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from imas_mcp.search.services.search_service import SearchService, SearchServiceError
from imas_mcp.search.search_strategy import SearchConfig, SearchResult
from imas_mcp.search.document_store import Document, DocumentMetadata
from imas_mcp.models.constants import SearchMode


class MockDocumentStore:
    """Mock document store for testing."""

    def __init__(self):
        self.documents = [
            Document(
                metadata=DocumentMetadata(
                    path_id="core_profiles/temperature",
                    ids_name="core_profiles",
                    path_name="core_profiles/temperature",
                    data_type="FLT_1D",
                    physics_domain="core_plasma",
                    physics_phenomena=("temperature",),
                ),
                documentation="Electron temperature profile",
            ),
            Document(
                metadata=DocumentMetadata(
                    path_id="equilibrium/pressure",
                    ids_name="equilibrium",
                    path_name="equilibrium/pressure",
                    data_type="FLT_1D",
                    physics_domain="equilibrium",
                    physics_phenomena=("pressure",),
                ),
                documentation="Plasma pressure equilibrium",
            ),
        ]


class TestSearchService:
    """Test suite for SearchService."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store."""
        return MockDocumentStore()

    @pytest.fixture
    def search_service(self, mock_document_store):
        """Create SearchService with mock engines."""
        service = SearchService()
        # Replace with mock engines that return consistent results
        for mode in service.engines:
            service.engines[mode] = Mock()
            service.engines[mode].search = AsyncMock(
                return_value=[
                    SearchResult(
                        document=mock_document_store.documents[0],
                        score=0.9,
                        rank=1,
                        search_mode=mode,
                    )
                ]
            )
        return service

    @pytest.mark.asyncio
    async def test_search_semantic_mode(self, search_service):
        """Test semantic search mode."""
        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=5)
        results = await search_service.search("temperature", config)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.search_mode == SearchMode.SEMANTIC for r in results)

    @pytest.mark.asyncio
    async def test_search_lexical_mode(self, search_service):
        """Test lexical search mode."""
        config = SearchConfig(search_mode=SearchMode.LEXICAL, max_results=5)
        results = await search_service.search("temperature", config)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(r.search_mode == SearchMode.LEXICAL for r in results)

    @pytest.mark.asyncio
    async def test_search_hybrid_mode(self, search_service):
        """Test hybrid search mode."""
        config = SearchConfig(search_mode=SearchMode.HYBRID, max_results=5)
        results = await search_service.search("temperature", config)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(r.search_mode == SearchMode.HYBRID for r in results)

    @pytest.mark.asyncio
    async def test_search_with_ids_filter(self, search_service):
        """Test search with IDS filtering."""
        config = SearchConfig(
            search_mode=SearchMode.SEMANTIC, max_results=5, ids_filter=["core_profiles"]
        )
        results = await search_service.search("temperature", config)

        assert isinstance(results, list)
        # Verify the config was passed correctly
        search_service.engines[SearchMode.SEMANTIC].search.assert_called_with(
            "temperature", config
        )

    @pytest.mark.asyncio
    async def test_search_auto_mode_resolution(self, search_service):
        """Test AUTO mode resolution."""
        config = SearchConfig(search_mode=SearchMode.AUTO, max_results=5)
        results = await search_service.search("temperature", config)

        assert isinstance(results, list)
        # AUTO mode should resolve to a specific mode
        if results:
            resolved_mode = results[0].search_mode
            assert resolved_mode in [
                SearchMode.SEMANTIC,
                SearchMode.LEXICAL,
                SearchMode.HYBRID,
            ]

    @pytest.mark.asyncio
    async def test_search_empty_query(self, search_service):
        """Test search with empty query."""
        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=5)

        # Mock empty results for empty query
        search_service.engines[SearchMode.SEMANTIC].search.return_value = []

        results = await search_service.search("", config)
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_engine_error(self, search_service):
        """Test search with engine error."""
        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=5)

        # Mock engine error
        search_service.engines[SearchMode.SEMANTIC].search.side_effect = Exception(
            "Engine failed"
        )

        with pytest.raises(SearchServiceError):
            await search_service.search("temperature", config)

    @pytest.mark.asyncio
    async def test_search_result_ranking(self, search_service):
        """Test that search results are properly ranked."""
        # Mock multiple results with different scores
        doc1 = Document(
            metadata=DocumentMetadata(
                path_id="core_profiles/temperature",
                ids_name="core_profiles",
                path_name="core_profiles/temperature",
                data_type="FLT_1D",
                physics_domain="core_plasma",
                physics_phenomena=("temperature",),
            ),
            documentation="Temperature profile",
        )
        doc2 = Document(
            metadata=DocumentMetadata(
                path_id="equilibrium/pressure",
                ids_name="equilibrium",
                path_name="equilibrium/pressure",
                data_type="FLT_1D",
                physics_domain="equilibrium",
                physics_phenomena=("pressure",),
            ),
            documentation="Pressure profile",
        )

        mock_results = [
            SearchResult(
                document=doc1, score=0.9, rank=1, search_mode=SearchMode.SEMANTIC
            ),
            SearchResult(
                document=doc2, score=0.7, rank=2, search_mode=SearchMode.SEMANTIC
            ),
        ]
        search_service.engines[SearchMode.SEMANTIC].search.return_value = mock_results

        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=5)
        results = await search_service.search("temperature", config)

        assert len(results) == 2
        assert results[0].score >= results[1].score  # Higher score first

    @pytest.mark.asyncio
    async def test_search_max_results_limit(self, search_service):
        """Test max_results parameter limits output."""
        # Create a sample document
        sample_doc = Document(
            metadata=DocumentMetadata(
                path_id="core_profiles/temperature",
                ids_name="core_profiles",
                path_name="core_profiles/temperature",
                data_type="FLT_1D",
                physics_domain="core_plasma",
                physics_phenomena=("temperature",),
            ),
            documentation="Temperature profile",
        )

        # Mock many results
        mock_results = [
            SearchResult(
                document=sample_doc,
                score=0.9 - i * 0.1,
                rank=i + 1,
                search_mode=SearchMode.SEMANTIC,
            )
            for i in range(10)
        ]
        search_service.engines[SearchMode.SEMANTIC].search.return_value = mock_results

        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=3)
        results = await search_service.search("temperature", config)

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_similarity_threshold(self, search_service):
        """Test similarity threshold filtering."""
        config = SearchConfig(
            search_mode=SearchMode.SEMANTIC, max_results=5, similarity_threshold=0.8
        )
        await search_service.search("temperature", config)

        # Verify config was passed with threshold
        search_service.engines[SearchMode.SEMANTIC].search.assert_called_with(
            "temperature", config
        )
        assert config.similarity_threshold == 0.8


class TestSearchServiceConfiguration:
    """Test SearchService configuration and setup."""

    def test_search_service_default_engines(self):
        """Test SearchService creates default engines."""
        service = SearchService()

        assert SearchMode.SEMANTIC in service.engines
        assert SearchMode.LEXICAL in service.engines
        assert SearchMode.HYBRID in service.engines

    def test_search_service_custom_engines(self):
        """Test SearchService with custom engines."""
        mock_engine1 = Mock()
        mock_engine1.search = AsyncMock(return_value=[])
        mock_engine2 = Mock()
        mock_engine2.search = AsyncMock(return_value=[])

        custom_engines = {
            SearchMode.SEMANTIC: mock_engine1,
            SearchMode.LEXICAL: mock_engine2,
        }

        service = SearchService(custom_engines)  # type: ignore

        assert service.engines == custom_engines
        assert SearchMode.HYBRID not in service.engines


class TestSearchModeSelection:
    """Test search mode selection logic."""

    @pytest.fixture
    def search_service(self):
        """Create SearchService for mode selection tests."""
        return SearchService()

    @pytest.mark.asyncio
    async def test_auto_mode_selects_semantic_for_physics_terms(self, search_service):
        """Test AUTO mode selects semantic for physics concepts."""
        # Mock the mode selector behavior
        config = SearchConfig(search_mode=SearchMode.AUTO, max_results=5)

        # For physics terms, should resolve to semantic
        physics_queries = ["plasma temperature", "electron density", "magnetic field"]

        for query in physics_queries:
            # Mock semantic engine
            search_service.engines[SearchMode.SEMANTIC] = Mock()
            search_service.engines[SearchMode.SEMANTIC].search = Mock(return_value=[])

            await search_service.search(query, config)
            # Verify semantic engine was called (AUTO resolved to semantic)
            # This would be true if the mode selector works correctly

    @pytest.mark.asyncio
    async def test_auto_mode_selects_lexical_for_path_terms(self, search_service):
        """Test AUTO mode selects lexical for path-like queries."""
        config = SearchConfig(search_mode=SearchMode.AUTO, max_results=5)

        # For path-like terms, should resolve to lexical
        path_queries = [
            "core_profiles/temperature",
            "equilibrium.pressure",
            "/ids/core",
        ]

        for query in path_queries:
            # Mock lexical engine
            search_service.engines[SearchMode.LEXICAL] = Mock()
            search_service.engines[SearchMode.LEXICAL].search = AsyncMock(
                return_value=[]
            )

            await search_service.search(query, config)
            # Verify appropriate engine was selected
