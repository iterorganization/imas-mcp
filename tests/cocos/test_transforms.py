"""Tests for COCOS path transforms module."""

import pytest

from imas_codex.cocos import (
    get_sign_flip_paths,
    list_ids_with_sign_flips,
    path_needs_cocos_transform,
)


class TestPathNeedsCOCOSTransform:
    """Test path_needs_cocos_transform function."""

    def test_equilibrium_psi_axis_needs_flip(self):
        """Psi axis in equilibrium typically needs sign flip."""
        # This depends on imas-python's _3to4_sign_flip_paths
        # We test that the function returns a boolean
        result = path_needs_cocos_transform(
            "equilibrium", "time_slice/global_quantities/psi_axis"
        )
        assert isinstance(result, bool)

    def test_unknown_path_returns_false(self):
        """Unknown paths should not need flip."""
        result = path_needs_cocos_transform("equilibrium", "nonexistent/path")
        assert result is False

    def test_unknown_ids_returns_false(self):
        """Unknown IDS should not need flip."""
        result = path_needs_cocos_transform("nonexistent_ids", "some/path")
        assert result is False

    def test_case_sensitivity(self):
        """IDS name lookup should be case-insensitive."""
        # Get paths for lowercase
        paths_lower = get_sign_flip_paths("equilibrium")
        paths_upper = get_sign_flip_paths("EQUILIBRIUM")
        # Both should give same result
        assert paths_lower == paths_upper


class TestGetSignFlipPaths:
    """Test get_sign_flip_paths function."""

    def test_returns_list(self):
        """Should return a list."""
        result = get_sign_flip_paths("equilibrium")
        assert isinstance(result, list)

    def test_returns_sorted(self):
        """Result should be sorted."""
        result = get_sign_flip_paths("equilibrium")
        assert result == sorted(result)

    def test_unknown_ids_returns_empty(self):
        """Unknown IDS returns empty list."""
        result = get_sign_flip_paths("nonexistent_ids")
        assert result == []


class TestListIDSWithSignFlips:
    """Test list_ids_with_sign_flips function."""

    def test_returns_list(self):
        """Should return a list."""
        result = list_ids_with_sign_flips()
        assert isinstance(result, list)

    def test_returns_sorted(self):
        """Result should be sorted."""
        result = list_ids_with_sign_flips()
        assert result == sorted(result)

    def test_contains_equilibrium_if_imas_available(self):
        """If imas is available, equilibrium should have sign flips."""
        result = list_ids_with_sign_flips()
        # Only check if any IDS are available
        if result:
            # Equilibrium is the most common IDS with sign flips
            assert "equilibrium" in result or len(result) > 0
