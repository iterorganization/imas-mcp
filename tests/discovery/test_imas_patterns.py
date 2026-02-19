"""Tests for shared IMAS path detection and extraction."""

from __future__ import annotations

import pytest

from imas_codex.discovery.base.imas_patterns import (
    build_imas_pattern,
    build_imas_rg_pattern,
    extract_ids_names,
    extract_imas_paths,
    get_all_ids_names,
    normalize_imas_path,
)

# =============================================================================
# get_all_ids_names
# =============================================================================


class TestGetAllIdsNames:
    def test_returns_tuple(self):
        names = get_all_ids_names()
        assert isinstance(names, tuple)

    def test_contains_core_ids(self):
        names = get_all_ids_names()
        for ids in ("equilibrium", "core_profiles", "magnetics", "wall", "summary"):
            assert ids in names

    def test_all_lowercase(self):
        names = get_all_ids_names()
        for name in names:
            assert name == name.lower()

    def test_sorted(self):
        names = get_all_ids_names()
        assert names == tuple(sorted(names))


# =============================================================================
# normalize_imas_path
# =============================================================================


class TestNormalizeImasPath:
    """Tests for IMAS path normalization."""

    def test_slash_separated(self):
        assert (
            normalize_imas_path("equilibrium/time_slice/profiles_1d/psi")
            == "equilibrium/time_slice/profiles_1d/psi"
        )

    def test_dot_separated(self):
        assert (
            normalize_imas_path("equilibrium.time_slice.profiles_1d.psi")
            == "equilibrium/time_slice/profiles_1d/psi"
        )

    def test_ids_dot_prefix_stripped(self):
        assert (
            normalize_imas_path("ids.equilibrium.global_quantities.ip")
            == "equilibrium/global_quantities/ip"
        )

    def test_ids_slash_prefix_stripped(self):
        assert (
            normalize_imas_path("ids/equilibrium/global_quantities/ip")
            == "equilibrium/global_quantities/ip"
        )

    def test_index_colon_stripped(self):
        assert (
            normalize_imas_path("equilibrium/time_slice[:]/profiles_1d/psi")
            == "equilibrium/time_slice/profiles_1d/psi"
        )

    def test_index_integer_stripped(self):
        assert (
            normalize_imas_path("core_profiles/profiles_1d[0]/electrons/temperature")
            == "core_profiles/profiles_1d/electrons/temperature"
        )

    def test_index_variable_stripped(self):
        assert (
            normalize_imas_path("equilibrium/time_slice[i]/profiles_1d[j]/psi")
            == "equilibrium/time_slice/profiles_1d/psi"
        )

    def test_index_range_stripped(self):
        assert (
            normalize_imas_path("magnetics/flux_loop[1:N]/flux/data")
            == "magnetics/flux_loop/flux/data"
        )

    def test_index_star_stripped(self):
        assert (
            normalize_imas_path("wall/description_2d[*]/limiter/unit")
            == "wall/description_2d/limiter/unit"
        )

    def test_mixed_dots_and_indices(self):
        assert (
            normalize_imas_path("core_profiles.profiles_1d[0].electrons.temperature")
            == "core_profiles/profiles_1d/electrons/temperature"
        )

    def test_uppercase_lowered(self):
        assert (
            normalize_imas_path("Core_Profiles/Profiles_1D/Electrons/Temperature")
            == "core_profiles/profiles_1d/electrons/temperature"
        )

    def test_trailing_slash_stripped(self):
        assert (
            normalize_imas_path("equilibrium/time_slice/") == "equilibrium/time_slice"
        )

    def test_double_slash_collapsed(self):
        assert (
            normalize_imas_path("equilibrium//time_slice") == "equilibrium/time_slice"
        )

    def test_multiple_indices_in_one_path(self):
        assert (
            normalize_imas_path("equilibrium/time_slice[:]/profiles_1d[:]/psi")
            == "equilibrium/time_slice/profiles_1d/psi"
        )


# =============================================================================
# extract_imas_paths
# =============================================================================


class TestExtractImasPaths:
    """Tests for IMAS path extraction from text."""

    def test_simple_slash_path(self):
        text = "Read equilibrium/time_slice/profiles_1d/psi from the database."
        paths = extract_imas_paths(text)
        assert "equilibrium/time_slice/profiles_1d/psi" in paths

    def test_dot_notation(self):
        text = "Access ids.equilibrium.global_quantities.ip for plasma current."
        paths = extract_imas_paths(text)
        assert "equilibrium/global_quantities/ip" in paths

    def test_index_variables(self):
        text = "Loop over equilibrium/time_slice[:]/profiles_1d/psi values."
        paths = extract_imas_paths(text)
        assert "equilibrium/time_slice/profiles_1d/psi" in paths

    def test_integer_indices(self):
        text = "core_profiles/profiles_1d[0]/electrons/temperature is the electron temperature."
        paths = extract_imas_paths(text)
        assert "core_profiles/profiles_1d/electrons/temperature" in paths

    def test_multiple_paths(self):
        text = (
            "Compare equilibrium/time_slice/global_quantities/ip "
            "with magnetics/flux_loop/flux/data."
        )
        paths = extract_imas_paths(text)
        assert "equilibrium/time_slice/global_quantities/ip" in paths
        assert "magnetics/flux_loop/flux/data" in paths

    def test_duplicates_removed(self):
        text = (
            "equilibrium/time_slice/psi equilibrium.time_slice.psi "
            "ids.equilibrium.time_slice.psi"
        )
        paths = extract_imas_paths(text)
        assert paths.count("equilibrium/time_slice/psi") == 1

    def test_sorted_output(self):
        text = "wall/description_2d/limiter and bolometer/channel/name"
        paths = extract_imas_paths(text)
        assert paths == sorted(paths)

    def test_empty_text(self):
        assert extract_imas_paths("") == []

    def test_no_matches(self):
        assert extract_imas_paths("This is plain text with no IMAS paths.") == []

    def test_code_context_python(self):
        text = """
        ids = imas.ids_factory.new("equilibrium")
        eq = ids.equilibrium.time_slice[0].profiles_1d.psi
        te = ids.core_profiles.profiles_1d[0].electrons.temperature
        """
        paths = extract_imas_paths(text)
        assert "equilibrium/time_slice/profiles_1d/psi" in paths
        assert "core_profiles/profiles_1d/electrons/temperature" in paths

    def test_code_context_fortran(self):
        text = "CALL imas_put(equilibrium/time_slice/global_quantities/ip, data)"
        paths = extract_imas_paths(text)
        assert "equilibrium/time_slice/global_quantities/ip" in paths

    def test_wiki_markup(self):
        text = (
            "The signal is stored at `core_profiles/profiles_1d[:]/electrons/density`"
            " and available via the IMAS Access Layer."
        )
        paths = extract_imas_paths(text)
        assert "core_profiles/profiles_1d/electrons/density" in paths

    def test_case_insensitive(self):
        text = "Access Core_Profiles/Profiles_1D/Electrons/Temperature"
        paths = extract_imas_paths(text)
        assert "core_profiles/profiles_1d/electrons/temperature" in paths

    def test_partial_path_matches(self):
        """Single-segment after IDS name should still match."""
        text = "equilibrium/global_quantities is the container."
        paths = extract_imas_paths(text)
        assert "equilibrium/global_quantities" in paths

    def test_pre_compiled_pattern(self):
        """Accept a pre-compiled pattern object."""
        pattern = build_imas_pattern()
        text = "Read equilibrium/time_slice/profiles_1d/psi"
        paths = extract_imas_paths(text, pattern=pattern)
        assert "equilibrium/time_slice/profiles_1d/psi" in paths

    def test_no_false_positive_on_bare_ids_name(self):
        """Bare IDS names (no sub-path) should NOT match."""
        text = "The equilibrium is computed using EFIT."
        paths = extract_imas_paths(text)
        assert paths == []  # no sub-path after 'equilibrium'

    def test_multiline_text(self):
        text = """
        # Equilibrium reconstruction
        path1 = "equilibrium/time_slice/profiles_1d/psi"
        path2 = "core_profiles/profiles_1d/electrons/temperature"
        # Wall description
        wall_path = "wall/description_2d/limiter/unit"
        """
        paths = extract_imas_paths(text)
        assert len(paths) == 3
        assert "equilibrium/time_slice/profiles_1d/psi" in paths
        assert "core_profiles/profiles_1d/electrons/temperature" in paths
        assert "wall/description_2d/limiter/unit" in paths


# =============================================================================
# extract_ids_names
# =============================================================================


class TestExtractIdsNames:
    def test_factory_new_pattern(self):
        text = 'ids = imas.ids_factory.new("equilibrium")'
        names = extract_ids_names(text)
        assert "equilibrium" in names

    def test_factory_call_pattern(self):
        text = "eq = factory.equilibrium()"
        names = extract_ids_names(text)
        assert "equilibrium" in names

    def test_string_literal(self):
        text = 'ids_name = "core_profiles"'
        names = extract_ids_names(text)
        assert "core_profiles" in names

    def test_no_false_positives(self):
        text = 'name = "not_an_ids_name"'
        names = extract_ids_names(text)
        assert len(names) == 0

    def test_multiple_ids(self):
        text = """
        eq = factory.new("equilibrium")
        cp = factory.new("core_profiles")
        mag = factory.new("magnetics")
        """
        names = extract_ids_names(text)
        assert "equilibrium" in names
        assert "core_profiles" in names
        assert "magnetics" in names


# =============================================================================
# build_imas_pattern
# =============================================================================


class TestBuildImasPattern:
    def test_returns_compiled_pattern(self):
        pattern = build_imas_pattern()
        assert hasattr(pattern, "findall")

    def test_case_insensitive(self):
        pattern = build_imas_pattern()
        assert pattern.flags & 2  # re.IGNORECASE

    def test_matches_slash_path(self):
        pattern = build_imas_pattern()
        assert pattern.search("equilibrium/time_slice/psi")

    def test_matches_dot_path(self):
        pattern = build_imas_pattern()
        assert pattern.search("equilibrium.time_slice.psi")

    def test_matches_ids_prefix(self):
        pattern = build_imas_pattern()
        assert pattern.search("ids.equilibrium.time_slice.psi")

    def test_matches_with_indices(self):
        pattern = build_imas_pattern()
        assert pattern.search("equilibrium/time_slice[:]/profiles_1d/psi")

    def test_no_match_bare_ids_name(self):
        pattern = build_imas_pattern()
        # 'equilibrium' alone without sub-path should not match
        # because the pattern requires [./] after the IDS name
        text = "equilibrium is a great solver"
        match = pattern.search(text)
        # If it matches, it should require at least one sub-segment
        if match:
            assert "/" in match.group() or "." in match.group()


# =============================================================================
# build_imas_rg_pattern
# =============================================================================


class TestBuildImasRgPattern:
    def test_returns_string(self):
        pattern = build_imas_rg_pattern()
        assert isinstance(pattern, str)

    def test_valid_regex(self):
        import re

        pattern = build_imas_rg_pattern()
        re.compile(pattern)  # Should not raise

    def test_contains_ids_names(self):
        pattern = build_imas_rg_pattern()
        assert "equilibrium" in pattern
        assert "core_profiles" in pattern
