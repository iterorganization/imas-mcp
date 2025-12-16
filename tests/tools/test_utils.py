"""Tests for tool utility functions."""

import pytest

from imas_mcp.tools.utils import (
    normalize_ids_filter,
    normalize_paths_input,
    validate_query,
)


class TestNormalizeIdsFilter:
    """Tests for normalize_ids_filter function."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        assert normalize_ids_filter(None) is None

    def test_empty_string_returns_none(self):
        """Test that empty string returns None."""
        assert normalize_ids_filter("") is None
        assert normalize_ids_filter("   ") is None

    def test_list_passthrough(self):
        """Test that list input is passed through."""
        result = normalize_ids_filter(["equilibrium", "magnetics"])
        assert result == ["equilibrium", "magnetics"]

    def test_list_strips_whitespace(self):
        """Test that whitespace is stripped from list items."""
        result = normalize_ids_filter(["  equilibrium ", " magnetics  "])
        assert result == ["equilibrium", "magnetics"]

    def test_list_filters_empty(self):
        """Test that empty items are filtered from list."""
        result = normalize_ids_filter(["equilibrium", "", "magnetics"])
        assert result == ["equilibrium", "magnetics"]

    def test_empty_list_returns_none(self):
        """Test that empty list returns None."""
        assert normalize_ids_filter([]) is None
        assert normalize_ids_filter(["", "  "]) is None

    def test_space_delimited_string(self):
        """Test space-delimited string parsing."""
        result = normalize_ids_filter("equilibrium magnetics")
        assert result == ["equilibrium", "magnetics"]

    def test_comma_delimited_string(self):
        """Test comma-delimited string parsing."""
        result = normalize_ids_filter("equilibrium, magnetics")
        assert result == ["equilibrium", "magnetics"]

    def test_comma_delimited_no_space(self):
        """Test comma-delimited string without spaces."""
        result = normalize_ids_filter("equilibrium,magnetics,core_profiles")
        assert result == ["equilibrium", "magnetics", "core_profiles"]

    def test_single_string(self):
        """Test single string returns single-element list."""
        result = normalize_ids_filter("equilibrium")
        assert result == ["equilibrium"]

    def test_multiple_spaces(self):
        """Test that multiple spaces are handled correctly."""
        result = normalize_ids_filter("equilibrium   magnetics")
        assert result == ["equilibrium", "magnetics"]

    def test_leading_trailing_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        result = normalize_ids_filter("  equilibrium magnetics  ")
        assert result == ["equilibrium", "magnetics"]


class TestNormalizePathsInput:
    """Tests for normalize_paths_input function."""

    def test_empty_string_returns_empty_list(self):
        """Test that empty string returns empty list."""
        assert normalize_paths_input("") == []
        assert normalize_paths_input("   ") == []

    def test_list_passthrough(self):
        """Test that list input is passed through."""
        result = normalize_paths_input(["path/one", "path/two"])
        assert result == ["path/one", "path/two"]

    def test_list_strips_whitespace(self):
        """Test that whitespace is stripped from list items."""
        result = normalize_paths_input(["  path/one ", " path/two  "])
        assert result == ["path/one", "path/two"]

    def test_list_filters_empty(self):
        """Test that empty items are filtered from list."""
        result = normalize_paths_input(["path/one", "", "path/two"])
        assert result == ["path/one", "path/two"]

    def test_space_delimited(self):
        """Test space-delimited string parsing."""
        result = normalize_paths_input("path/one path/two")
        assert result == ["path/one", "path/two"]

    def test_single_path(self):
        """Test single path returns single-element list."""
        result = normalize_paths_input("equilibrium/time_slice")
        assert result == ["equilibrium/time_slice"]

    def test_multiple_spaces(self):
        """Test that multiple spaces are handled correctly."""
        result = normalize_paths_input("path/one   path/two")
        assert result == ["path/one", "path/two"]


class TestValidateQuery:
    """Tests for validate_query function."""

    def test_valid_query(self):
        """Test that valid query returns True with no error."""
        is_valid, error = validate_query("electron temperature", "test_tool")
        assert is_valid is True
        assert error is None

    def test_empty_string_query(self):
        """Test that empty string returns False with error."""
        is_valid, error = validate_query("", "test_tool")
        assert is_valid is False
        assert error is not None
        assert "cannot be empty" in error
        assert "test_tool" in error

    def test_none_query(self):
        """Test that None returns False with error."""
        is_valid, error = validate_query(None, "test_tool")
        assert is_valid is False
        assert error is not None

    def test_whitespace_query(self):
        """Test that whitespace-only returns False with error."""
        is_valid, error = validate_query("   ", "test_tool")
        assert is_valid is False
        assert error is not None

    def test_error_includes_guidance(self):
        """Test that error message includes helpful guidance."""
        is_valid, error = validate_query("", "search_imas_paths")
        assert is_valid is False
        assert "get_imas_overview" in error
