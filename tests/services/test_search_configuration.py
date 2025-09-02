"""Tests for SearchConfigurationService."""

import pytest

from imas_mcp.models.constants import SearchMode
from imas_mcp.search.search_strategy import SearchConfig
from imas_mcp.services.search_configuration import SearchConfigurationService


class TestSearchConfigurationService:
    """Test SearchConfigurationService functionality."""

    def test_initialization(self):
        """Test SearchConfigurationService initializes correctly."""
        service = SearchConfigurationService()
        assert service.logger is not None

    def test_create_config_defaults(self):
        """Test creating configuration with defaults."""
        service = SearchConfigurationService()

        config = service.create_config()

        assert isinstance(config, SearchConfig)
        assert config.search_mode == SearchMode.AUTO
        assert config.max_results == 50  # Service default, not class default
        assert config.ids_filter is None
        # Physics is now always enabled at the core level, no longer a config parameter

    def test_create_config_with_params(self):
        """Test creating configuration with specific parameters."""
        service = SearchConfigurationService()

        config = service.create_config(
            search_mode="semantic",
            max_results=20,
            ids_filter=["equilibrium"],
        )

        assert config.search_mode == SearchMode.SEMANTIC
        assert config.max_results == 20
        assert config.ids_filter == ["equilibrium"]
        # Physics is now always enabled at the core level, no longer a config parameter

    def test_create_config_string_ids_filter(self):
        """Test creating configuration with string IDS filter."""
        service = SearchConfigurationService()

        config = service.create_config(ids_filter="equilibrium")

        assert config.ids_filter == ["equilibrium"]

    def test_optimize_for_query_complex(self):
        """Test query optimization for complex queries."""
        service = SearchConfigurationService()

        base_config = SearchConfig(
            search_mode=SearchMode.AUTO,
            max_results=10,
            ids_filter=None,
            similarity_threshold=0.0,
        )

        optimized = service.optimize_for_query(
            "complex query with many words that should trigger semantic search",
            base_config,
        )

        assert optimized.search_mode == SearchMode.SEMANTIC

    def test_optimize_for_query_boolean(self):
        """Test query optimization for boolean queries."""
        service = SearchConfigurationService()

        base_config = SearchConfig(
            search_mode=SearchMode.AUTO,
            max_results=10,
            ids_filter=None,
            similarity_threshold=0.0,
        )

        optimized = service.optimize_for_query("plasma AND temperature", base_config)

        assert optimized.search_mode == SearchMode.LEXICAL

    def test_optimize_for_query_list_input(self):
        """Test query optimization with list input."""
        service = SearchConfigurationService()

        base_config = SearchConfig(
            search_mode=SearchMode.AUTO,
            max_results=10,
            ids_filter=None,
            similarity_threshold=0.0,
        )

        optimized = service.optimize_for_query(["plasma", "temperature"], base_config)

        # Should handle list input
        assert isinstance(optimized, SearchConfig)
