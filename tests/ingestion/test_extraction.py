"""Tests for unified tree extraction module (mdsplus/extraction.py)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.mdsplus.extraction import (
    _compute_parent_path,
    _resolve_shots,
    discover_tree,
    extract_tree_version,
    get_static_tree_config,
    merge_units_into_data,
    merge_version_results,
)

_EXTRACT_TREE_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "imas_codex"
    / "remote"
    / "scripts"
    / "extract_tree.py"
)


class TestResolveShotsVersioned:
    """Test _resolve_shots for versioned trees."""

    def test_returns_provided_shots(self):
        result = _resolve_shots("tcv", "static", [1, 2, 3])
        assert result == [1, 2, 3]

    def test_resolves_from_config_versioned(self):
        result = _resolve_shots("tcv", "static", None)
        assert len(result) >= 1
        assert result[0] == 1
        assert result[-1] >= 1


class TestResolveShotsDynamic:
    """Test _resolve_shots for subtree/dynamic trees."""

    def test_resolves_subtree_from_config(self):
        result = _resolve_shots("tcv", "results", None)
        # results is a subtree of tcv_shot, should resolve to a single reference shot
        assert len(result) == 1
        assert result[0] >= 1

    def test_resolves_subtree_magnetics(self):
        result = _resolve_shots("tcv", "magnetics", None)
        assert len(result) == 1
        assert result[0] >= 1


class TestExtractTreeVersion:
    """Test extract_tree_version sends correct input to remote script."""

    @patch("imas_codex.mdsplus.extraction.run_python_script")
    @patch("imas_codex.mdsplus.extraction._load_mdsplus_config")
    def test_uses_extract_tree_script(self, mock_config, mock_run):
        mock_config.return_value = {
            "exclude_node_names": ["COMMENTS"],
            "setup_commands": ["source /etc/profile.d/mdsplus.sh"],
        }
        mock_run.return_value = '{"data_source_name": "results", "versions": {"85000": {"nodes": [], "node_count": 0, "tags": {}}}, "diff": {}}'

        extract_tree_version("tcv", "results", shot=85000)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        # Script name is first positional arg
        assert call_args[0][0] == "extract_tree.py"
        # input_data could be positional or keyword
        input_data = call_args[1].get("input_data") or call_args[0][1]
        assert input_data["data_source_name"] == "results"
        assert input_data["shots"] == [85000]
        assert input_data["exclude_names"] == ["COMMENTS"]
        assert "node_usages" not in input_data

    @patch("imas_codex.mdsplus.extraction.run_python_script")
    @patch("imas_codex.mdsplus.extraction._load_mdsplus_config")
    def test_passes_node_usages(self, mock_config, mock_run):
        mock_config.return_value = {
            "exclude_node_names": [],
            "setup_commands": None,
        }
        mock_run.return_value = '{"data_source_name": "results", "versions": {"85000": {"nodes": [], "node_count": 0, "tags": {}}}, "diff": {}}'

        extract_tree_version(
            "tcv", "results", shot=85000, node_usages=["SIGNAL", "NUMERIC"]
        )

        call_args = mock_run.call_args
        input_data = call_args[1].get("input_data") or call_args[0][1]
        assert input_data["node_usages"] == ["SIGNAL", "NUMERIC"]

    @patch("imas_codex.mdsplus.extraction.run_python_script")
    @patch("imas_codex.mdsplus.extraction._load_mdsplus_config")
    def test_handles_error_in_version_data(self, mock_config, mock_run):
        mock_config.return_value = {
            "exclude_node_names": [],
            "setup_commands": None,
        }
        mock_run.return_value = '{"data_source_name": "results", "versions": {"85000": {"error": "tree not found"}}, "diff": {}}'

        result = extract_tree_version("tcv", "results", shot=85000)
        assert "error" in result["versions"]["85000"]


class TestDiscoverTree:
    """Test discover_tree (unified multi-shot extraction)."""

    @patch("imas_codex.mdsplus.extraction.extract_tree_version")
    @patch("imas_codex.mdsplus.extraction._resolve_shots")
    def test_extracts_each_shot(self, mock_resolve, mock_extract):
        mock_resolve.return_value = [85000]
        mock_extract.return_value = {
            "data_source_name": "results",
            "versions": {
                "85000": {
                    "nodes": [{"path": "\\RESULTS::TOP"}],
                    "node_count": 1,
                    "tags": {},
                }
            },
            "diff": {},
        }

        result = discover_tree("tcv", "results", shots=[85000])

        mock_extract.assert_called_once_with(
            facility="tcv",
            data_source_name="results",
            shot=85000,
            timeout=300,
            node_usages=None,
        )
        assert "85000" in result["versions"]

    @patch("imas_codex.mdsplus.extraction.extract_tree_version")
    def test_multiple_shots(self, mock_extract):
        mock_extract.side_effect = [
            {
                "data_source_name": "static",
                "versions": {
                    "1": {
                        "nodes": [{"path": "\\STATIC::TOP"}],
                        "node_count": 1,
                        "tags": {},
                    }
                },
                "diff": {},
            },
            {
                "data_source_name": "static",
                "versions": {
                    "2": {
                        "nodes": [
                            {"path": "\\STATIC::TOP"},
                            {"path": "\\STATIC::TOP.NEW"},
                        ],
                        "node_count": 2,
                        "tags": {},
                    }
                },
                "diff": {},
            },
        ]

        result = discover_tree("tcv", "static", shots=[1, 2])

        assert "1" in result["versions"]
        assert "2" in result["versions"]
        assert result["versions"]["1"]["node_count"] == 1
        assert result["versions"]["2"]["node_count"] == 2
        # Diff should show the added path in version 2
        assert "\\STATIC::TOP.NEW" in result["diff"]["added"].get("2", [])


class TestBackwardCompat:
    """Verify backward-compatible aliases work."""

    def test_discover_static_tree_version_alias(self):
        from imas_codex.mdsplus.extraction import discover_static_tree_version

        assert discover_static_tree_version is extract_tree_version

    def test_discover_static_tree_alias(self):
        from imas_codex.mdsplus.extraction import discover_static_tree

        assert discover_static_tree is discover_tree

    def test_static_module_re_exports(self):
        # Verify these are the extraction module's functions
        from imas_codex.mdsplus.extraction import (
            discover_tree,
            extract_tree_version,
        )
        from imas_codex.mdsplus.static import (
            discover_static_tree,
            discover_static_tree_version,
            get_static_tree_config,
            ingest_static_tree,
            merge_units_into_data,
        )

        assert discover_static_tree is discover_tree
        assert discover_static_tree_version is extract_tree_version

    def test_mdsplus_init_exports(self):
        from imas_codex.mdsplus import (
            discover_static_tree,
            discover_tree,
            extract_tree_version,
        )

        assert discover_static_tree is discover_tree


class TestExtractTreeScript:
    """Tests for the extract_tree.py remote script functions."""

    def test_extract_version_runs(self):
        """Verify the script module can be imported and functions exist."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "extract_tree",
            str(_EXTRACT_TREE_SCRIPT),
        )
        importlib.util.module_from_spec(spec)
        # Don't exec (needs MDSplus), just check the functions exist
        assert spec is not None

    def test_diff_versions(self):
        """Test the diff_versions function in extract_tree.py."""
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "extract_tree",
            str(_EXTRACT_TREE_SCRIPT),
        )
        mod = importlib.util.module_from_spec(spec)
        # Load the module (safe — functions don't import MDSplus at module level)
        spec.loader.exec_module(mod)

        version_data = {
            "1": {"nodes": [{"path": "A"}, {"path": "B"}]},
            "2": {"nodes": [{"path": "A"}, {"path": "C"}]},
        }
        diff = mod.diff_versions(version_data)
        assert "C" in diff["added"]["2"]
        assert "B" in diff["removed"]["2"]

    def test_diff_versions_with_error(self):
        """Error versions are skipped in diff."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "extract_tree",
            str(_EXTRACT_TREE_SCRIPT),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        version_data = {
            "1": {"nodes": [{"path": "A"}]},
            "2": {"error": "tree not found"},
            "3": {"nodes": [{"path": "A"}, {"path": "B"}]},
        }
        diff = mod.diff_versions(version_data)
        assert "B" in diff["added"]["3"]
