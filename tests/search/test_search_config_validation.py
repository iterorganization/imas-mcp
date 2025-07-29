"""
Tests for SearchConfig validation and enum conversion.

This module tests the field validation that converts string inputs to SearchMode enums,
which was the core issue we resolved in the refactoring.
"""

import pytest
from pydantic import ValidationError

from imas_mcp.models.constants import SearchMode
from imas_mcp.search.search_strategy import SearchConfig


class TestSearchConfigValidation:
    """Test SearchConfig field validation and enum conversion."""

    def test_search_mode_enum_input(self):
        """Test that SearchMode enum input is accepted and preserved."""
        config = SearchConfig(search_mode=SearchMode.SEMANTIC)

        assert config.search_mode == SearchMode.SEMANTIC
        assert isinstance(config.search_mode, SearchMode)
        assert config.search_mode.value == "semantic"

    def test_search_mode_string_input_valid_values(self):
        """Test that valid string inputs are converted to SearchMode enums."""
        valid_string_modes = ["auto", "semantic", "lexical", "hybrid"]
        expected_enums = [
            SearchMode.AUTO,
            SearchMode.SEMANTIC,
            SearchMode.LEXICAL,
            SearchMode.HYBRID,
        ]

        for string_mode, expected_enum in zip(valid_string_modes, expected_enums):
            config = SearchConfig(search_mode=string_mode)

            assert config.search_mode == expected_enum
            assert isinstance(config.search_mode, SearchMode)
            assert config.search_mode.value == string_mode

    def test_search_mode_string_input_invalid_value(self):
        """Test that invalid string inputs raise ValidationError."""
        invalid_modes = ["invalid", "wrong", "unknown", ""]

        for invalid_mode in invalid_modes:
            with pytest.raises(ValidationError) as exc_info:
                SearchConfig(search_mode=invalid_mode)

            error = exc_info.value.errors()[0]
            assert error["type"] == "value_error"
            assert "Invalid search_mode" in error["msg"]
            assert "Valid options:" in error["msg"]

    def test_search_mode_default_value(self):
        """Test that default search_mode is SearchMode.AUTO."""
        config = SearchConfig()

        assert config.search_mode == SearchMode.AUTO
        assert isinstance(config.search_mode, SearchMode)
        assert config.search_mode.value == "auto"

    def test_search_mode_case_sensitivity(self):
        """Test that string inputs are case sensitive."""
        # These should fail because SearchMode values are lowercase
        invalid_cases = ["AUTO", "Semantic", "LEXICAL", "Hybrid"]

        for invalid_case in invalid_cases:
            with pytest.raises(ValidationError):
                SearchConfig(search_mode=invalid_case)

    def test_search_mode_whitespace_handling(self):
        """Test that strings with whitespace are rejected."""
        invalid_with_whitespace = [
            " auto",
            "semantic ",
            " lexical ",
            "\thybrid",
            "auto\n",
        ]

        for invalid_input in invalid_with_whitespace:
            with pytest.raises(ValidationError):
                SearchConfig(search_mode=invalid_input)

    def test_search_mode_type_safety_after_validation(self):
        """Test that after validation, search_mode can be used safely as SearchMode."""
        config = SearchConfig(search_mode="hybrid")

        # These operations should work without type errors
        # (these were the specific issues we fixed)

        # 1. Dictionary lookup (was causing type checker errors)
        strategies = {
            SearchMode.AUTO: "auto_strategy",
            SearchMode.SEMANTIC: "semantic_strategy",
            SearchMode.LEXICAL: "lexical_strategy",
            SearchMode.HYBRID: "hybrid_strategy",
        }
        strategy = strategies[config.search_mode]
        assert strategy == "hybrid_strategy"

        # 2. Accessing .value attribute (was causing type checker errors)
        value = config.search_mode.value
        assert value == "hybrid"

        # 3. Enum comparison
        assert config.search_mode == SearchMode.HYBRID
        assert config.search_mode != SearchMode.AUTO

    def test_ids_filter_field_validator(self):
        """Test ids_filter field validator handles different input types."""
        # Test None input
        config1 = SearchConfig(ids_filter=None)  # type: ignore[arg-type]
        assert config1.ids_filter is None

        # Test single string input
        config2 = SearchConfig(ids_filter="core_profiles")  # type: ignore[arg-type]
        assert config2.ids_filter == ["core_profiles"]

        # Test space-separated string input
        config3 = SearchConfig(ids_filter="core_profiles equilibrium")  # type: ignore[arg-type]
        assert config3.ids_filter == ["core_profiles", "equilibrium"]

        # Test list input
        config4 = SearchConfig(ids_filter=["core_profiles", "equilibrium"])  # type: ignore[arg-type]
        assert config4.ids_filter == ["core_profiles", "equilibrium"]

    def test_ids_filter_invalid_inputs(self):
        """Test ids_filter field validator rejects invalid inputs."""
        # Test invalid type
        with pytest.raises(
            ValueError, match="ids_filter must be None, string, or list of strings"
        ):
            SearchConfig(ids_filter=123)  # type: ignore[arg-type]

        # Test list with non-string items
        with pytest.raises(
            ValueError, match="All items in ids_filter list must be strings"
        ):
            SearchConfig(ids_filter=["core_profiles", 123])  # type: ignore[arg-type]

    def test_ids_filter_edge_cases(self):
        """Test ids_filter field validator edge cases."""
        # Test empty string
        config1 = SearchConfig(ids_filter="")  # type: ignore[arg-type]
        assert config1.ids_filter == [""]

        # Test empty list
        config2 = SearchConfig(ids_filter=[])  # type: ignore[arg-type]
        assert config2.ids_filter == []

        # Test string with multiple spaces
        config3 = SearchConfig(ids_filter="core_profiles   equilibrium   transport")  # type: ignore[arg-type]
        assert config3.ids_filter == ["core_profiles", "equilibrium", "transport"]

    def test_config_with_all_fields(self):
        """Test SearchConfig with all fields including string search_mode and ids_filter."""
        config = SearchConfig(
            search_mode="semantic",  # type: ignore[arg-type]
            max_results=20,
            ids_filter=["core_profiles", "equilibrium"],  # type: ignore[arg-type]
            similarity_threshold=0.8,
            boost_exact_matches=False,
            enable_physics_enhancement=False,
        )

        assert config.search_mode == SearchMode.SEMANTIC
        assert config.max_results == 20
        assert config.ids_filter == ["core_profiles", "equilibrium"]
        assert config.similarity_threshold == 0.8
        assert config.boost_exact_matches is False
        assert config.enable_physics_enhancement is False

    def test_search_mode_validator_preserves_enum_input(self):
        """Test that the validator doesn't unnecessarily convert enum inputs."""
        original_enum = SearchMode.LEXICAL
        config = SearchConfig(search_mode=original_enum)

        # Should be the same enum instance, not converted
        assert config.search_mode is original_enum
        assert config.search_mode == SearchMode.LEXICAL

    def test_search_mode_all_enum_members_supported(self):
        """Test that all SearchMode enum members are supported."""
        for search_mode in SearchMode:
            # Test enum input
            config1 = SearchConfig(search_mode=search_mode)
            assert config1.search_mode == search_mode

            # Test string input
            config2 = SearchConfig(search_mode=search_mode.value)  # type: ignore[arg-type]
            assert config2.search_mode == search_mode


class TestSearchConfigIntegration:
    """Test SearchConfig integration with other components."""

    def test_config_serialization_with_validated_enum(self):
        """Test that SearchConfig can be serialized properly after validation."""
        config = SearchConfig(search_mode="hybrid", max_results=15)  # type: ignore[arg-type]

        # Test model_dump (Pydantic v2)
        data = config.model_dump()
        assert data["search_mode"] == SearchMode.HYBRID
        assert data["max_results"] == 15

    def test_config_comparison_after_validation(self):
        """Test that SearchConfig instances can be compared after validation."""
        config1 = SearchConfig(search_mode="semantic")  # type: ignore[arg-type]
        config2 = SearchConfig(search_mode=SearchMode.SEMANTIC)
        config3 = SearchConfig(search_mode="lexical")  # type: ignore[arg-type]

        # Same mode should be equal
        assert config1.search_mode == config2.search_mode

        # Different modes should not be equal
        assert config1.search_mode != config3.search_mode

    def test_config_copy_preserves_validated_enum(self):
        """Test that copying SearchConfig preserves the validated enum."""
        original = SearchConfig(search_mode="hybrid")  # type: ignore[arg-type]
        copied = original.model_copy()

        assert copied.search_mode == SearchMode.HYBRID
        assert isinstance(copied.search_mode, SearchMode)
        # Enum instances are singletons, so they will be the same object
        assert copied.search_mode is original.search_mode  # Same enum instance
        assert copied.search_mode == original.search_mode  # Same value


class TestSearchConfigEdgeCases:
    """Test edge cases and error conditions for SearchConfig."""

    def test_none_search_mode_input(self):
        """Test that None input for search_mode raises ValidationError."""
        with pytest.raises(ValidationError):
            SearchConfig(search_mode=None)  # type: ignore[arg-type]

    def test_numeric_search_mode_input(self):
        """Test that numeric input for search_mode raises ValidationError."""
        with pytest.raises(ValidationError):
            SearchConfig(search_mode=0)  # type: ignore[arg-type]

        with pytest.raises(ValidationError):
            SearchConfig(search_mode=1.5)  # type: ignore[arg-type]

    def test_list_search_mode_input(self):
        """Test that list input for search_mode raises ValidationError."""
        with pytest.raises(ValidationError):
            SearchConfig(search_mode=["semantic"])  # type: ignore[arg-type]

    def test_dict_search_mode_input(self):
        """Test that dict input for search_mode raises ValidationError."""
        with pytest.raises(ValidationError):
            SearchConfig(search_mode={"mode": "semantic"})  # type: ignore[arg-type]

    def test_boolean_search_mode_input(self):
        """Test that boolean input for search_mode raises ValidationError."""
        with pytest.raises(ValidationError):
            SearchConfig(search_mode=True)  # type: ignore[arg-type]

        with pytest.raises(ValidationError):
            SearchConfig(search_mode=False)  # type: ignore[arg-type]
