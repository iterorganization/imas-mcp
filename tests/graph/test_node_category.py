"""Tests for node_category classification logic."""

import pytest

from imas_codex.core.exclusions import ExclusionChecker


def _classify_node(path_id: str, name: str) -> str:
    """Classify a node using ExclusionChecker (mirrors build_dd._classify_node)."""
    checker = ExclusionChecker()
    if checker._is_error_field(name):
        return "error"
    if checker._is_metadata_path(path_id):
        return "metadata"
    return "data"


class TestClassifyNode:
    """Tests for node classification which classifies IMASNode paths."""

    # --- Error fields ---

    @pytest.mark.parametrize(
        "path,name",
        [
            ("equilibrium/time_slice/boundary/psi_error_upper", "psi_error_upper"),
            ("equilibrium/time_slice/boundary/psi_error_lower", "psi_error_lower"),
            ("equilibrium/time_slice/boundary/psi_error_index", "psi_error_index"),
            (
                "core_profiles/profiles_1d/electrons/temperature_error_upper",
                "temperature_error_upper",
            ),
            (
                "core_profiles/profiles_1d/electrons/density_error_lower",
                "density_error_lower",
            ),
        ],
    )
    def test_error_fields(self, path, name):
        assert _classify_node(path, name) == "error"

    # --- Metadata subtrees ---

    @pytest.mark.parametrize(
        "path,name",
        [
            ("equilibrium/ids_properties/homogeneous_time", "homogeneous_time"),
            ("equilibrium/ids_properties/comment", "comment"),
            ("equilibrium/ids_properties/provider", "provider"),
            ("equilibrium/code/name", "name"),
            ("equilibrium/code/version", "version"),
            ("equilibrium/code/parameters", "parameters"),
            (
                "core_profiles/ids_properties/creation_date",
                "creation_date",
            ),
        ],
    )
    def test_metadata_subtrees(self, path, name):
        assert _classify_node(path, name) == "metadata"

    # --- Generic metadata leaf fields ---

    @pytest.mark.parametrize(
        "path,name",
        [
            ("equilibrium/time_slice/boundary/description", "description"),
            ("equilibrium/time_slice/boundary/name", "name"),
            ("equilibrium/time_slice/boundary/comment", "comment"),
            ("equilibrium/time_slice/boundary/source", "source"),
            ("equilibrium/time_slice/boundary/provider", "provider"),
            ("equilibrium/time_slice/identifier/description", "description"),
            ("equilibrium/time_slice/identifier/name", "name"),
        ],
    )
    def test_metadata_leaf_fields(self, path, name):
        assert _classify_node(path, name) == "metadata"

    # --- Data fields ---

    @pytest.mark.parametrize(
        "path,name",
        [
            ("equilibrium/time_slice/boundary/psi", "psi"),
            (
                "core_profiles/profiles_1d/electrons/temperature",
                "temperature",
            ),
            (
                "core_profiles/profiles_1d/electrons/density",
                "density",
            ),
            ("equilibrium/time_slice/profiles_1d/q", "q"),
            ("magnetics/flux_loop/flux/data", "data"),
            ("equilibrium/time", "time"),
        ],
    )
    def test_data_fields(self, path, name):
        assert _classify_node(path, name) == "data"

    # --- Edge cases ---

    def test_ids_root_is_data(self):
        """IDS root paths (only 1 segment) should be data."""
        assert _classify_node("equilibrium", "equilibrium") == "data"

    def test_two_segment_path_is_data(self):
        """Two-segment paths should be data (not enough parts for metadata check)."""
        assert _classify_node("equilibrium/time_slice", "time_slice") == "data"

    def test_error_in_name_but_not_suffix(self):
        """Fields with 'error' in name but not the _error_X suffix pattern."""
        assert _classify_node("pf_active/coil/current/data", "data") == "data"

    def test_code_as_first_segment_not_excluded(self):
        """'code' as the IDS name (first segment) should not be metadata."""
        # parts[1:] check skips the first segment
        assert _classify_node("code/something", "something") == "data"
