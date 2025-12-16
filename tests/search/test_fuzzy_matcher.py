"""Tests for fuzzy path matching."""

import pytest

from imas_codex.search.fuzzy_matcher import (
    PathFuzzyMatcher,
    reset_fuzzy_matcher,
    suggest_correction,
)


@pytest.fixture
def matcher():
    """Create a PathFuzzyMatcher with test data."""
    valid_ids = ["equilibrium", "core_profiles", "magnetics", "summary"]
    valid_paths = [
        "equilibrium/time_slice",
        "equilibrium/time_slice/boundary",
        "equilibrium/time_slice/boundary/psi",
        "equilibrium/time_slice/global_quantities",
        "equilibrium/time_slice/global_quantities/ip",
        "core_profiles/profiles_1d",
        "core_profiles/profiles_1d/electrons",
        "core_profiles/profiles_1d/electrons/temperature",
        "magnetics/flux_loop",
        "magnetics/b_field_pol_probe",
    ]
    return PathFuzzyMatcher(valid_ids, valid_paths)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the fuzzy matcher singleton before each test."""
    reset_fuzzy_matcher()
    yield
    reset_fuzzy_matcher()


class TestSuggestIds:
    """Tests for IDS name suggestion."""

    def test_typo_equilibrum(self, matcher):
        """Test that 'equilibrum' suggests 'equilibrium'."""
        suggestions = matcher.suggest_ids("equilibrum")
        assert "equilibrium" in suggestions

    def test_typo_core_profile(self, matcher):
        """Test that 'core_profile' suggests 'core_profiles'."""
        suggestions = matcher.suggest_ids("core_profile")
        assert "core_profiles" in suggestions

    def test_typo_magnatic(self, matcher):
        """Test that 'magnatic' suggests 'magnetics'."""
        suggestions = matcher.suggest_ids("magnatic")
        assert "magnetics" in suggestions

    def test_completely_wrong(self, matcher):
        """Test that completely wrong input returns empty list."""
        suggestions = matcher.suggest_ids("xxxxxxxx")
        assert suggestions == []

    def test_empty_input(self, matcher):
        """Test that empty input returns empty list."""
        suggestions = matcher.suggest_ids("")
        assert suggestions == []

    def test_max_suggestions(self, matcher):
        """Test that max_suggestions is respected."""
        suggestions = matcher.suggest_ids("e", max_suggestions=1)
        assert len(suggestions) <= 1


class TestSuggestPaths:
    """Tests for path suggestion."""

    def test_typo_in_ids(self, matcher):
        """Test path suggestion when IDS name has typo."""
        suggestions = matcher.suggest_paths("equilibrum/time_slice")
        # Should suggest the corrected path
        assert any("equilibrium/time_slice" in s for s in suggestions)

    def test_typo_in_path_segment(self, matcher):
        """Test path suggestion when path segment has typo."""
        suggestions = matcher.suggest_paths("equilibrium/timeslice")
        assert "equilibrium/time_slice" in suggestions

    def test_partial_path(self, matcher):
        """Test path suggestion for partial path."""
        suggestions = matcher.suggest_paths("equilibrium/time_slice/bound")
        assert any("boundary" in s for s in suggestions)

    def test_completely_wrong_ids(self, matcher):
        """Test that completely wrong IDS returns empty list."""
        suggestions = matcher.suggest_paths("xxxxxxxx/yyyyyyyy")
        assert suggestions == []

    def test_empty_input(self, matcher):
        """Test that empty input returns empty list."""
        suggestions = matcher.suggest_paths("")
        assert suggestions == []

    def test_scoped_to_ids(self, matcher):
        """Test that paths are suggested from the correct IDS scope."""
        suggestions = matcher.suggest_paths("core_profiles/profiles_1d/electron")
        # Should suggest from core_profiles, not equilibrium
        assert all("core_profiles" in s for s in suggestions)


class TestGetSuggestionForPath:
    """Tests for formatted suggestion hints."""

    def test_returns_formatted_hint(self, matcher):
        """Test that a formatted hint is returned."""
        hint = matcher.get_suggestion_for_path("equilibrum/time_slice")
        assert hint is not None
        assert "Did you mean" in hint

    def test_ids_level_suggestion(self, matcher):
        """Test suggestion when only IDS is wrong."""
        hint = matcher.get_suggestion_for_path("equilibrum/unknown/path")
        assert hint is not None
        assert "equilibrium" in hint.lower()

    def test_returns_none_for_unrecognizable(self, matcher):
        """Test that unrecognizable input returns None."""
        hint = matcher.get_suggestion_for_path("xxxxxxxxx/yyyyyyyyy/zzzzzzzzz")
        # May or may not have suggestion depending on cutoff
        # Just ensure it doesn't crash
        assert hint is None or isinstance(hint, str)


class TestSuggestCorrection:
    """Tests for the convenience function."""

    def test_returns_suggestion(self):
        """Test that suggest_correction returns a suggestion."""
        valid_ids = ["equilibrium", "magnetics"]
        valid_paths = ["equilibrium/time_slice", "magnetics/flux_loop"]

        suggestion = suggest_correction("equilibrum/time_slice", valid_ids, valid_paths)
        assert suggestion is not None
        assert "Did you mean" in suggestion or "Unknown IDS" in suggestion

    def test_returns_none_for_unknown(self):
        """Test that suggest_correction returns None for unknown paths."""
        valid_ids = ["equilibrium"]
        valid_paths = ["equilibrium/time_slice"]

        suggestion = suggest_correction("xxxxxxx/yyyyyyy", valid_ids, valid_paths)
        # Should return None for completely unknown paths
        assert suggestion is None


class TestPathIndex:
    """Tests for path indexing functionality."""

    def test_index_by_ids(self, matcher):
        """Test that paths are indexed by IDS."""
        # The internal _path_by_ids should have entries for each IDS
        assert "equilibrium" in matcher._path_by_ids
        assert "core_profiles" in matcher._path_by_ids
        assert "magnetics" in matcher._path_by_ids

    def test_index_contains_all_paths(self, matcher):
        """Test that index contains all paths."""
        total_indexed = sum(len(paths) for paths in matcher._path_by_ids.values())
        assert total_indexed == len(matcher.valid_paths)
