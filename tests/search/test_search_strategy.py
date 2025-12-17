"""Tests for search/search_strategy.py module."""

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.models.constants import SearchMode
from imas_codex.search.document_store import Document, DocumentMetadata
from imas_codex.search.search_strategy import (
    HybridSearchStrategy,
    LexicalSearchStrategy,
    SearchConfig,
    SearchHit,
    SearchMatch,
    SearchModeSelector,
    SearchResponse,
    SemanticSearchStrategy,
)


class TestSearchConfig:
    """Tests for the SearchConfig class."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = SearchConfig()

        assert config.search_mode == SearchMode.AUTO
        assert config.max_results == 50
        assert config.similarity_threshold == 0.0
        assert config.boost_exact_matches is True

    def test_validate_search_mode_from_string(self):
        """Search mode validates from string."""
        config = SearchConfig(search_mode="lexical")
        assert config.search_mode == SearchMode.LEXICAL

        config = SearchConfig(search_mode="semantic")
        assert config.search_mode == SearchMode.SEMANTIC

        config = SearchConfig(search_mode="hybrid")
        assert config.search_mode == SearchMode.HYBRID

        config = SearchConfig(search_mode="auto")
        assert config.search_mode == SearchMode.AUTO

    def test_validate_search_mode_invalid(self):
        """Invalid search mode raises error."""
        with pytest.raises(ValueError, match="Invalid search_mode"):
            SearchConfig(search_mode="invalid_mode")

    def test_validate_ids_filter_none(self):
        """ids_filter can be None."""
        config = SearchConfig(ids_filter=None)
        assert config.ids_filter is None

    def test_validate_ids_filter_string(self):
        """ids_filter validates from single string."""
        config = SearchConfig(ids_filter="equilibrium")
        assert config.ids_filter == ["equilibrium"]

    def test_validate_ids_filter_space_separated(self):
        """ids_filter validates from space-separated string."""
        config = SearchConfig(ids_filter="equilibrium core_profiles")
        assert config.ids_filter == ["equilibrium", "core_profiles"]

    def test_validate_ids_filter_list(self):
        """ids_filter validates from list."""
        config = SearchConfig(ids_filter=["equilibrium", "core_profiles"])
        assert config.ids_filter == ["equilibrium", "core_profiles"]

    def test_validate_ids_filter_invalid_type(self):
        """Invalid ids_filter type raises error."""
        with pytest.raises(ValueError, match="ids_filter must be"):
            SearchConfig(ids_filter=123)

    def test_validate_ids_filter_invalid_list_items(self):
        """Invalid list items in ids_filter raises error."""
        with pytest.raises(ValueError, match="All items in ids_filter"):
            SearchConfig(ids_filter=["valid", 123])


class TestSearchMatch:
    """Tests for the SearchMatch class."""

    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        metadata = DocumentMetadata(
            path_id="equilibrium/time_slice/profiles_1d/psi",
            ids_name="equilibrium",
            path_name="profiles_1d/psi",
            data_type="FLT_1D",
            physics_domain="MHD",
        )
        return Document(
            metadata=metadata,
            documentation="Poloidal flux profile",
            raw_data={
                "lifecycle": "stable",
                "type": "dynamic",
                "timebase": "time",
                "coordinate1": "rho_tor_norm",
                "structure_reference": None,
            },
        )

    def test_to_hit(self, sample_document):
        """to_hit converts SearchMatch to SearchHit."""
        match = SearchMatch(
            document=sample_document,
            score=0.95,
            rank=0,
            search_mode=SearchMode.SEMANTIC,
            highlights="test highlight",
        )

        hit = match.to_hit()

        assert isinstance(hit, SearchHit)
        assert hit.score == 0.95
        assert hit.rank == 0
        assert hit.search_mode == SearchMode.SEMANTIC
        assert hit.highlights == "test highlight"
        assert hit.path == "profiles_1d/psi"
        assert hit.documentation == "Poloidal flux profile"
        assert hit.ids_name == "equilibrium"
        assert hit.physics_domain == "MHD"
        assert hit.lifecycle == "stable"
        assert hit.node_type == "dynamic"
        assert hit.timebase == "time"
        assert hit.coordinate1 == "rho_tor_norm"

    def test_to_hit_with_units(self):
        """to_hit handles units correctly."""
        from imas_codex.search.document_store import Units

        metadata = DocumentMetadata(
            path_id="test/path",
            ids_name="test_ids",
            path_name="path",
        )
        doc = Document(
            metadata=metadata,
            documentation="Test doc",
            units=Units(unit_str="eV"),
        )
        match = SearchMatch(
            document=doc,
            score=0.8,
            rank=1,
            search_mode=SearchMode.LEXICAL,
        )

        hit = match.to_hit()
        assert hit.units == "eV"


class TestSearchModeSelector:
    """Tests for the SearchModeSelector class."""

    def test_select_mode_with_technical_operators(self):
        """Technical operators select lexical mode."""
        selector = SearchModeSelector()

        # Explicit operators
        assert selector.select_mode("ids_name:equilibrium") == SearchMode.LEXICAL
        assert selector.select_mode("path:*/temperature") == SearchMode.LEXICAL
        assert selector.select_mode("term AND another") == SearchMode.LEXICAL
        assert selector.select_mode("term OR another") == SearchMode.LEXICAL
        assert selector.select_mode('"exact phrase"') == SearchMode.LEXICAL
        assert selector.select_mode("temp*") == SearchMode.LEXICAL

    def test_select_mode_imas_paths(self):
        """IMAS paths select lexical mode."""
        selector = SearchModeSelector()

        # Path-like queries with IMAS indicators
        assert selector.select_mode("equilibrium/profiles_1d/psi") == SearchMode.LEXICAL
        assert (
            selector.select_mode("core_profiles/global_quantities")
            == SearchMode.LEXICAL
        )
        assert selector.select_mode("transport/time_slice") == SearchMode.LEXICAL

    def test_select_mode_imas_technical_terms(self):
        """IMAS technical terms select lexical mode."""
        selector = SearchModeSelector()

        assert selector.select_mode("core_profiles") == SearchMode.LEXICAL
        assert selector.select_mode("equilibrium") == SearchMode.LEXICAL
        assert selector.select_mode("profiles_1d") == SearchMode.LEXICAL
        assert selector.select_mode("global_quantities") == SearchMode.LEXICAL

    def test_select_mode_conceptual_queries(self):
        """Conceptual queries select semantic mode."""
        selector = SearchModeSelector()

        assert selector.select_mode("what is plasma temperature") == SearchMode.SEMANTIC
        assert (
            selector.select_mode("how does the magnetic field work")
            == SearchMode.SEMANTIC
        )
        assert selector.select_mode("explain electron density") == SearchMode.SEMANTIC
        assert selector.select_mode("describe fusion physics") == SearchMode.SEMANTIC
        assert selector.select_mode("meaning of flux") == SearchMode.SEMANTIC

    def test_select_mode_hybrid_for_mixed(self):
        """Mixed queries select hybrid mode."""
        selector = SearchModeSelector()

        # Both technical and conceptual elements
        # Note: if technical takes precedence if strongly matching
        result = selector.select_mode("explain the temperature profile")
        assert result in [SearchMode.SEMANTIC, SearchMode.HYBRID]

    def test_select_mode_list_query(self):
        """List queries work correctly."""
        selector = SearchModeSelector()

        result = selector.select_mode(["core_profiles", "temperature"])
        # 'temperature' is conceptual, 'core_profiles' is technical, so hybrid or semantic is possible
        assert result in [SearchMode.LEXICAL, SearchMode.HYBRID, SearchMode.SEMANTIC]

    def test_is_technical_query_underscore_terms(self):
        """Underscore-separated technical terms are detected."""
        selector = SearchModeSelector()

        # Test internal method
        assert selector._is_technical_query("profiles_1d") is True
        assert selector._is_technical_query("rho_tor_norm") is True
        assert selector._is_technical_query("time_slice") is True

    def test_is_conceptual_query(self):
        """Conceptual indicators are detected."""
        selector = SearchModeSelector()

        assert selector._is_conceptual_query("what is fusion") is True
        assert selector._is_conceptual_query("plasma physics") is True
        assert selector._is_conceptual_query("temperature measurement") is True

    def test_default_to_hybrid(self):
        """Ambiguous queries default to hybrid mode."""
        selector = SearchModeSelector()

        # Generic query without clear indicators
        result = selector.select_mode("data values")
        assert result == SearchMode.HYBRID


class TestLexicalSearchStrategy:
    """Tests for the LexicalSearchStrategy class."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store."""
        store = MagicMock()
        metadata = DocumentMetadata(
            path_id="test/path",
            ids_name="test_ids",
            path_name="path",
        )
        doc = Document(metadata=metadata, documentation="Test doc")
        store.search_full_text.return_value = [doc]
        return store

    def test_search_basic(self, mock_document_store):
        """Basic lexical search works."""
        strategy = LexicalSearchStrategy(mock_document_store)
        config = SearchConfig(max_results=10)

        results = strategy.search("temperature", config)

        assert len(results) == 1
        assert results[0].search_mode == SearchMode.LEXICAL
        mock_document_store.search_full_text.assert_called_once()

    def test_search_with_ids_filter(self, mock_document_store):
        """Search applies IDS filter."""
        strategy = LexicalSearchStrategy(mock_document_store)
        config = SearchConfig(
            max_results=10, ids_filter=["equilibrium", "core_profiles"]
        )

        strategy.search("temperature", config)

        # Check that IDS filter was applied to query
        call_args = mock_document_store.search_full_text.call_args
        query = call_args[0][0]
        assert "ids_name:equilibrium" in query
        assert "ids_name:core_profiles" in query

    def test_search_list_query(self, mock_document_store):
        """Search handles list queries."""
        strategy = LexicalSearchStrategy(mock_document_store)
        config = SearchConfig(max_results=10)

        strategy.search(["electron", "temperature"], config)

        call_args = mock_document_store.search_full_text.call_args
        query = call_args[0][0]
        assert "electron temperature" in query

    def test_search_handles_exception(self, mock_document_store):
        """Search handles exceptions gracefully."""
        mock_document_store.search_full_text.side_effect = Exception("Search error")
        strategy = LexicalSearchStrategy(mock_document_store)
        config = SearchConfig()

        results = strategy.search("query", config)

        assert results == []

    def test_get_mode(self, mock_document_store):
        """get_mode returns LEXICAL."""
        strategy = LexicalSearchStrategy(mock_document_store)
        assert strategy.get_mode() == SearchMode.LEXICAL


class TestSemanticSearchStrategy:
    """Tests for the SemanticSearchStrategy class."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store."""
        store = MagicMock()
        store.ids_set = None
        return store

    def test_init(self, mock_document_store):
        """Strategy initializes correctly."""
        strategy = SemanticSearchStrategy(mock_document_store)
        assert strategy._semantic_search is None

    def test_get_mode(self, mock_document_store):
        """get_mode returns SEMANTIC."""
        strategy = SemanticSearchStrategy(mock_document_store)
        assert strategy.get_mode() == SearchMode.SEMANTIC


class TestHybridSearchStrategy:
    """Tests for the HybridSearchStrategy class."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store."""
        store = MagicMock()
        store.ids_set = None
        return store

    def test_init(self, mock_document_store):
        """Strategy initializes with both sub-strategies."""
        strategy = HybridSearchStrategy(mock_document_store)

        assert isinstance(strategy.semantic_strategy, SemanticSearchStrategy)
        assert isinstance(strategy.lexical_strategy, LexicalSearchStrategy)

    def test_get_mode(self, mock_document_store):
        """get_mode returns HYBRID."""
        strategy = HybridSearchStrategy(mock_document_store)
        assert strategy.get_mode() == SearchMode.HYBRID


class TestSearchResponse:
    """Tests for the SearchResponse class."""

    def test_empty_response(self):
        """Empty response works."""
        response = SearchResponse(hits=[])
        assert response.hits == []

    def test_response_with_hits(self):
        """Response with hits works."""
        metadata = DocumentMetadata(
            path_id="test/path",
            ids_name="test_ids",
            path_name="path",
        )
        doc = Document(metadata=metadata, documentation="Test")
        match = SearchMatch(
            document=doc,
            score=0.9,
            rank=0,
            search_mode=SearchMode.SEMANTIC,
        )

        response = SearchResponse(hits=[match])
        assert len(response.hits) == 1
        assert response.hits[0].score == 0.9
