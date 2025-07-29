"""
Tests for search engine base classes and implementations.

Tests the abstract SearchEngine interface and mock implementation.
"""

import pytest

from imas_mcp.search.engines.base_engine import (
    SearchEngine,
    SearchEngineError,
    MockSearchEngine,
)
from imas_mcp.search.search_strategy import SearchConfig, SearchResult
from imas_mcp.models.constants import SearchMode


class ConcreteSearchEngine(SearchEngine):
    """Concrete implementation for testing SearchEngine."""

    def __init__(self):
        super().__init__("concrete_test")

    async def search(self, query, config):
        # Simple implementation for testing
        return []

    def get_engine_type(self):
        return "concrete"


class TestBaseSearchEngine:
    """Test SearchEngine abstract base class."""

    def test_search_engine_is_abstract(self):
        """Test that SearchEngine cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SearchEngine("test")  # type: ignore

    def test_concrete_engine_instantiation(self):
        """Test that concrete engine implementations can be instantiated."""
        engine = ConcreteSearchEngine()
        assert engine is not None
        assert isinstance(engine, SearchEngine)
        assert engine.name == "concrete_test"

    def test_validate_query_string(self):
        """Test query validation with string input."""
        engine = ConcreteSearchEngine()

        assert engine.validate_query("valid query") is True
        assert engine.validate_query("") is False
        assert engine.validate_query("   ") is False

    def test_validate_query_list(self):
        """Test query validation with list input."""
        engine = ConcreteSearchEngine()

        assert engine.validate_query(["query1", "query2"]) is True
        assert engine.validate_query([]) is False
        assert engine.validate_query(["", ""]) is False
        assert engine.validate_query(["valid", ""]) is True

    def test_validate_query_invalid_types(self):
        """Test query validation with invalid types."""
        engine = ConcreteSearchEngine()

        assert engine.validate_query(None) is False  # type: ignore
        assert engine.validate_query(123) is False  # type: ignore
        assert engine.validate_query({}) is False  # type: ignore

    def test_normalize_query_string(self):
        """Test query normalization with string input."""
        engine = ConcreteSearchEngine()

        assert engine.normalize_query("test query") == "test query"
        assert engine.normalize_query("  spaced  ") == "spaced"

    def test_normalize_query_list(self):
        """Test query normalization with list input."""
        engine = ConcreteSearchEngine()

        assert engine.normalize_query(["one", "two"]) == "one two"
        assert engine.normalize_query(["", "valid", " "]) == "valid"

    def test_log_search_execution(self):
        """Test search execution logging."""
        engine = ConcreteSearchEngine()
        config = SearchConfig(mode=SearchMode.SEMANTIC, max_results=10)

        # Should not raise exception
        engine.log_search_execution("test query", config, 5)

    @pytest.mark.asyncio
    async def test_concrete_engine_search_method(self):
        """Test that concrete engine implements search method."""
        engine = ConcreteSearchEngine()
        config = SearchConfig()

        results = await engine.search("test", config)
        assert isinstance(results, list)


class TestBaseMockSearchEngine:
    """Test MockSearchEngine implementation."""

    def test_mock_engine_instantiation(self):
        """Test MockSearchEngine can be instantiated."""
        engine = MockSearchEngine()
        assert engine.name == "mock"
        assert engine.get_engine_type() == "mock"

    @pytest.mark.asyncio
    async def test_mock_engine_search(self):
        """Test MockSearchEngine returns predictable results."""
        engine = MockSearchEngine()
        config = SearchConfig(max_results=1)

        results = await engine.search("temperature", config)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert "temperature" in results[0].document.documentation

    @pytest.mark.asyncio
    async def test_mock_engine_different_queries(self):
        """Test MockSearchEngine with different query types."""
        engine = MockSearchEngine()
        config = SearchConfig()

        # String query
        results1 = await engine.search("test string", config)
        assert len(results1) == 1

        # List query
        results2 = await engine.search(["test", "list"], config)
        assert len(results2) == 1

    def test_mock_engine_query_validation(self):
        """Test MockSearchEngine inherits validation methods."""
        engine = MockSearchEngine()

        assert engine.validate_query("valid") is True
        assert engine.validate_query("") is False


class TestBaseSearchEngineError:
    """Test SearchEngineError exception."""

    def test_search_engine_error_creation(self):
        """Test SearchEngineError can be created with required parameters."""
        error = SearchEngineError("test_engine", "Test error message")

        assert error.engine_name == "test_engine"
        assert error.query == ""
        assert "test_engine" in str(error)
        assert "Test error message" in str(error)

    def test_search_engine_error_with_query(self):
        """Test SearchEngineError with query parameter."""
        error = SearchEngineError("test_engine", "Test error", "test query")

        assert error.engine_name == "test_engine"
        assert error.query == "test query"

    def test_search_engine_error_inheritance(self):
        """Test SearchEngineError inherits from Exception."""
        error = SearchEngineError("test", "message")
        assert isinstance(error, Exception)
