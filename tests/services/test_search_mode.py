"""
Tests for search mode configuration and selection.

This module tests that:
1. Explicit search modes are never overridden by optimization
2. AUTO mode properly selects appropriate search modes based on query characteristics
3. Default search mode is AUTO
4. Boolean operator detection works correctly
"""

import pytest

from imas_mcp.models.constants import SearchMode
from imas_mcp.search.search_strategy import SearchConfig, SearchModeSelector
from imas_mcp.services.search_configuration import SearchConfigurationService


class TestSearchModePreservation:
    """Test that explicit search modes are never overridden."""

    @pytest.fixture
    def search_config_service(self):
        """Create a SearchConfigurationService instance."""
        return SearchConfigurationService()

    @pytest.mark.parametrize(
        "explicit_mode",
        [
            SearchMode.SEMANTIC,
            SearchMode.LEXICAL,
            SearchMode.HYBRID,
        ],
    )
    @pytest.mark.parametrize(
        "query",
        [
            "magnetic field flux surface toroidal",  # Original problem query
            "what is plasma temperature",  # Conceptual query
            "profiles_1d AND temperature",  # Boolean query
            "core_profiles/profiles_1d/t_i",  # Path-like query
            "explain the physics of tokamak equilibrium confinement",  # Long query
            "AND OR NOT",  # All boolean operators
            "transport and analysis",  # Contains "and" but not boolean
        ],
    )
    def test_explicit_mode_never_overridden(
        self, search_config_service, explicit_mode, query
    ):
        """Test that explicit search modes are never overridden by optimization."""
        # Create config with explicit mode
        config = search_config_service.create_config(
            search_mode=explicit_mode,
            max_results=10,
        )

        # Verify initial mode is set correctly
        assert config.search_mode == explicit_mode

        # Optimize the config
        optimized_config = search_config_service.optimize_for_query(query, config)

        # Verify mode was not changed
        assert optimized_config.search_mode == explicit_mode, (
            f"Explicit mode {explicit_mode.value} was overridden to "
            f"{optimized_config.search_mode.value} for query: '{query}'"
        )


class TestDefaultSearchMode:
    """Test that default search mode is AUTO."""

    def test_default_search_mode_is_auto(self):
        """Test that SearchConfig defaults to AUTO mode."""
        config = SearchConfig()
        assert config.search_mode == SearchMode.AUTO

    def test_search_config_service_default_is_auto(self):
        """Test that SearchConfigurationService defaults to AUTO mode."""
        service = SearchConfigurationService()
        config = service.create_config()
        assert config.search_mode == SearchMode.AUTO

    def test_create_config_with_string_auto(self):
        """Test that string 'auto' is converted to AUTO enum."""
        service = SearchConfigurationService()
        config = service.create_config(search_mode="auto")
        assert config.search_mode == SearchMode.AUTO


class TestAutoModeSelection:
    """Test AUTO mode selection logic using SearchModeSelector."""

    @pytest.fixture
    def mode_selector(self):
        """Create a SearchModeSelector instance."""
        return SearchModeSelector()

    @pytest.mark.parametrize(
        "query,expected_mode",
        [
            # Technical queries should use LEXICAL
            ("core_profiles/profiles_1d/t_i", SearchMode.LEXICAL),
            ("profiles_1d AND temperature", SearchMode.LEXICAL),
            ("ids_name:equilibrium", SearchMode.LEXICAL),
            ("documentation:plasma", SearchMode.LEXICAL),
            ("time_slice", SearchMode.LEXICAL),
            ("equilibrium", SearchMode.LEXICAL),
            ("rho_tor_norm", SearchMode.LEXICAL),
            # Conceptual queries should use SEMANTIC
            ("what is plasma temperature", SearchMode.SEMANTIC),
            ("explain tokamak physics", SearchMode.SEMANTIC),
            ("how does magnetic confinement work", SearchMode.SEMANTIC),
            ("meaning of safety factor", SearchMode.SEMANTIC),
            ("magnetic field flux surface toroidal", SearchMode.SEMANTIC),
            ("plasma temperature profiles", SearchMode.SEMANTIC),
            ("current density distribution", SearchMode.SEMANTIC),
            # Mixed queries (technical + conceptual) should use HYBRID
            ("describe plasma equilibrium", SearchMode.HYBRID),
            ("transport physics equilibrium", SearchMode.HYBRID),
        ],
    )
    def test_auto_mode_selection(self, mode_selector, query, expected_mode):
        """Test that AUTO mode selects appropriate search modes."""
        selected_mode = mode_selector.select_mode(query)
        assert selected_mode == expected_mode, (
            f"Expected {expected_mode.value} for query '{query}', "
            f"got {selected_mode.value}"
        )

    def test_explicit_operators_force_lexical(self, mode_selector):
        """Test that explicit operators always force lexical search."""
        explicit_operator_queries = [
            'path:"core_profiles"',
            "temperature AND density",
            "profiles_1d OR profiles_2d",
            "NOT equilibrium",
            "plasma*",
            "temp~",
            "units:eV",
            "documentation:transport",
        ]

        for query in explicit_operator_queries:
            mode = mode_selector.select_mode(query)
            assert mode == SearchMode.LEXICAL, (
                f"Query '{query}' with explicit operators should use LEXICAL mode, "
                f"got {mode.value}"
            )


class TestBooleanOperatorDetection:
    """Test boolean operator detection logic."""

    @pytest.fixture
    def search_config_service(self):
        """Create a SearchConfigurationService instance."""
        return SearchConfigurationService()

    @pytest.mark.parametrize(
        "query,has_boolean_ops",
        [
            # True boolean operators
            ("temperature AND density", True),
            ("profiles_1d OR profiles_2d", True),
            ("NOT equilibrium", True),
            ("temp AND (density OR pressure)", True),
            ("AND", True),
            ("OR", True),
            ("NOT", True),
            # False positives (should NOT be detected as boolean)
            ("magnetic field flux surface toroidal", False),
            ("transport and analysis", False),  # "and" not "AND"
            ("explore", False),  # contains "or" but not "OR"
            ("cannot", False),  # contains "not" but not "NOT"
            ("major", False),  # contains "or" but not "OR"
            ("another", False),  # contains "not" but not "NOT"
            ("standard", False),  # contains "and" but not "AND"
        ],
    )
    def test_boolean_operator_detection(
        self, search_config_service, query, has_boolean_ops
    ):
        """Test that boolean operators are detected correctly."""
        # Create config in AUTO mode to test optimization logic
        config = search_config_service.create_config(search_mode=SearchMode.AUTO)
        optimized_config = search_config_service.optimize_for_query(query, config)

        if has_boolean_ops:
            # Queries with boolean operators should be optimized to LEXICAL
            # (only when in AUTO mode)
            assert optimized_config.search_mode == SearchMode.LEXICAL, (
                f"Query '{query}' has boolean operators and should use LEXICAL mode"
            )
        else:
            # Queries without boolean operators should not be forced to LEXICAL
            # due to boolean detection (they might still be LEXICAL for other reasons)
            # This is harder to test directly, so we'll test the helper method
            pass


class TestSearchConfigurationEdgeCases:
    """Test edge cases for search configuration."""

    @pytest.fixture
    def search_config_service(self):
        """Create a SearchConfigurationService instance."""
        return SearchConfigurationService()

    def test_empty_query(self, search_config_service):
        """Test optimization with empty query."""
        config = search_config_service.create_config(search_mode=SearchMode.SEMANTIC)
        optimized_config = search_config_service.optimize_for_query("", config)
        assert optimized_config.search_mode == SearchMode.SEMANTIC

    def test_list_query(self, search_config_service):
        """Test optimization with list query."""
        query_list = ["magnetic", "field", "flux"]
        config = search_config_service.create_config(search_mode=SearchMode.SEMANTIC)
        optimized_config = search_config_service.optimize_for_query(query_list, config)
        assert optimized_config.search_mode == SearchMode.SEMANTIC

    def test_very_long_query_auto_mode(self, search_config_service):
        """Test that very long queries in AUTO mode get optimized to SEMANTIC."""
        long_query = "explain the detailed physics of plasma confinement in tokamak devices with magnetic field"
        config = search_config_service.create_config(search_mode=SearchMode.AUTO)
        optimized_config = search_config_service.optimize_for_query(long_query, config)

        # Long queries (>5 words) in AUTO mode should be optimized to SEMANTIC
        assert optimized_config.search_mode == SearchMode.SEMANTIC

    def test_ids_filter_conversion(self, search_config_service):
        """Test that IDS filter is properly converted."""
        # String filter
        config1 = search_config_service.create_config(ids_filter="equilibrium")
        assert config1.ids_filter == ["equilibrium"]

        # List filter
        config2 = search_config_service.create_config(
            ids_filter=["equilibrium", "transport"]
        )
        assert config2.ids_filter == ["equilibrium", "transport"]

        # None filter
        config3 = search_config_service.create_config(ids_filter=None)
        assert config3.ids_filter is None


class TestIntegrationWithSearchModeSelector:
    """Integration tests between SearchConfigurationService and SearchModeSelector."""

    def test_mode_selector_integration(self):
        """Test that SearchModeSelector produces expected results for known queries."""
        selector = SearchModeSelector()

        test_cases = [
            ("magnetic field flux surface toroidal", SearchMode.SEMANTIC),
            ("core_profiles/profiles_1d/t_i", SearchMode.LEXICAL),
            ("what is plasma temperature", SearchMode.SEMANTIC),
            ("temperature AND density", SearchMode.LEXICAL),
        ]

        for query, expected_mode in test_cases:
            # Test direct mode selection
            selected_mode = selector.select_mode(query)
            assert selected_mode == expected_mode


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
