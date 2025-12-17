"""Tests for mappings/__init__.py module."""

import pytest

from imas_codex.mappings import (
    PathMap,
    PathMapping,
    RenameHistoryEntry,
    get_path_map,
)


class TestPathMap:
    """Tests for the PathMap class."""

    def test_initialization_with_mock_data(self):
        """PathMap initializes with provided mapping data."""
        mock_data = {
            "old_to_new": {},
            "new_to_old": {},
            "metadata": {"total_mappings": 0, "target_version": "4.0.0"},
            "exclusion_reasons": {},
            "excluded_paths": {},
        }

        path_map = PathMap(mapping_data=mock_data)

        assert path_map.total_mappings == 0
        assert path_map.target_version == "4.0.0"

    def test_get_mapping_returns_none_for_unknown_path(self):
        """get_mapping returns None for unknown path."""
        mock_data = {"old_to_new": {}, "new_to_old": {}, "metadata": {}}
        path_map = PathMap(mapping_data=mock_data)

        result = path_map.get_mapping("unknown/path")

        assert result is None

    def test_get_mapping_returns_path_mapping_for_known_path(self):
        """get_mapping returns PathMapping for known path."""
        mock_data = {
            "old_to_new": {
                "old/path": {
                    "new_path": "new/path",
                    "deprecated_in": "4.0.0",
                    "last_valid_version": "3.9.0",
                }
            },
            "new_to_old": {},
            "metadata": {},
        }
        path_map = PathMap(mapping_data=mock_data)

        result = path_map.get_mapping("old/path")

        assert isinstance(result, PathMapping)
        assert result.new_path == "new/path"
        assert result.deprecated_in == "4.0.0"

    def test_get_rename_history_returns_entries(self):
        """get_rename_history returns history entries."""
        mock_data = {
            "old_to_new": {},
            "new_to_old": {
                "new/path": [
                    {"old_path": "old/path1", "deprecated_in": "3.5.0"},
                    {"old_path": "old/path2", "deprecated_in": "4.0.0"},
                ]
            },
            "metadata": {},
        }
        path_map = PathMap(mapping_data=mock_data)

        history = path_map.get_rename_history("new/path")

        assert len(history) == 2
        assert all(isinstance(h, RenameHistoryEntry) for h in history)

    def test_get_exclusion_reason_from_map(self):
        """get_exclusion_reason returns reason from pre-computed map."""
        mock_data = {
            "old_to_new": {},
            "new_to_old": {},
            "metadata": {},
            "excluded_paths": {"test/error_upper": "error_field"},
            "exclusion_reasons": {"error_field": "Error field paths are excluded"},
        }
        path_map = PathMap(mapping_data=mock_data)

        reason = path_map.get_exclusion_reason("test/error_upper")

        assert reason == "error_field"

    def test_get_exclusion_description(self):
        """get_exclusion_description returns human-readable description."""
        mock_data = {
            "old_to_new": {},
            "new_to_old": {},
            "metadata": {},
            "exclusion_reasons": {"error_field": "Error fields are excluded"},
        }
        path_map = PathMap(mapping_data=mock_data)

        desc = path_map.get_exclusion_description("error_field")

        assert desc == "Error fields are excluded"

    def test_is_excluded(self):
        """is_excluded returns True for excluded paths."""
        mock_data = {
            "old_to_new": {},
            "new_to_old": {},
            "metadata": {},
            "excluded_paths": {"excluded/path": "test_reason"},
        }
        path_map = PathMap(mapping_data=mock_data)

        assert path_map.is_excluded("excluded/path") is True

    def test_get_mapping_returns_none_when_data_is_none(self):
        """get_mapping returns None when internal data is None."""
        path_map = PathMap(mapping_data=None)
        path_map._loaded = True
        path_map._data = None

        result = path_map.get_mapping("any/path")

        assert result is None

    def test_get_rename_history_returns_empty_when_data_is_none(self):
        """get_rename_history returns empty list when internal data is None."""
        path_map = PathMap(mapping_data=None)
        path_map._loaded = True
        path_map._data = None

        result = path_map.get_rename_history("any/path")

        assert result == []

    def test_get_exclusion_reason_returns_none_when_data_is_none(self):
        """get_exclusion_reason returns None when internal data is None."""
        path_map = PathMap(mapping_data=None)
        path_map._loaded = True
        path_map._data = None

        result = path_map.get_exclusion_reason("any/path")

        assert result is None

    def test_metadata_returns_empty_when_data_is_none(self):
        """metadata property returns empty dict when data is None."""
        path_map = PathMap(mapping_data=None)
        path_map._loaded = True
        path_map._data = None

        result = path_map.metadata

        assert result == {}

    def test_get_rename_history_empty_for_unknown_path(self):
        """get_rename_history returns empty list for unknown path."""
        mock_data = {
            "old_to_new": {},
            "new_to_old": {},
            "metadata": {},
        }
        path_map = PathMap(mapping_data=mock_data)

        result = path_map.get_rename_history("unknown/path")

        assert result == []

    def test_get_exclusion_description_fallback(self):
        """get_exclusion_description falls back to key when not found."""
        mock_data = {
            "old_to_new": {},
            "new_to_old": {},
            "metadata": {},
            "exclusion_reasons": {},
        }
        path_map = PathMap(mapping_data=mock_data)

        desc = path_map.get_exclusion_description("unknown_reason")

        assert desc == "unknown_reason"

    def test_is_excluded_returns_false_for_non_excluded(self):
        """is_excluded returns False for non-excluded paths."""
        mock_data = {
            "old_to_new": {},
            "new_to_old": {},
            "metadata": {},
            "excluded_paths": {},
        }
        path_map = PathMap(mapping_data=mock_data)

        assert path_map.is_excluded("normal/path") is False


class TestGetPathMap:
    """Tests for the get_path_map singleton function."""

    def test_returns_singleton(self):
        """get_path_map returns the same instance on repeated calls."""
        get_path_map.cache_clear()

        map1 = get_path_map()
        map2 = get_path_map()

        assert map1 is map2
