"""Tests for check_dd_paths with specific DD version support.

Covers:
- Existing path in latest version — no flag
- Path not yet introduced in an early version
- Path renamed — suggestion includes rename version when dd_version is given
- _dd_version_clause semver string mode
- Backward-compatible integer mode unchanged
"""

from unittest.mock import MagicMock

import pytest

from imas_codex.tools.graph_search import GraphPathTool, _dd_version_clause

# ============================================================================
# _dd_version_clause — semver string mode
# ============================================================================


class TestDDVersionClauseSemver:
    """Test _dd_version_clause with string semver versions."""

    def test_string_version_returns_and_clause(self):
        clause = _dd_version_clause("p", dd_version="3.42.0")
        assert clause.startswith("AND EXISTS")

    def test_string_version_uses_major_minor_patch_params(self):
        params: dict = {}
        _dd_version_clause("p", dd_version="3.42.0", params=params)
        assert params["dd_ver_major"] == 3
        assert params["dd_ver_minor"] == 42
        assert params["dd_ver_patch"] == 0

    def test_string_version_checks_introduced_in(self):
        clause = _dd_version_clause("p", dd_version="4.1.1")
        assert "INTRODUCED_IN" in clause
        assert "iv.major" in clause

    def test_string_version_checks_not_deprecated(self):
        clause = _dd_version_clause("p", dd_version="4.1.1")
        assert "NOT EXISTS" in clause
        assert "DEPRECATED_IN" in clause
        assert "dv.major" in clause

    def test_string_version_patch_defaults_to_zero(self):
        """Two-component version 'x.y' treated as 'x.y.0'."""
        params: dict = {}
        _dd_version_clause("p", dd_version="3.42", params=params)
        assert params["dd_ver_patch"] == 0

    def test_unparseable_string_returns_empty(self):
        """Unparseable version string must not crash — returns empty clause."""
        clause = _dd_version_clause("p", dd_version="not-a-version")
        assert clause == ""

    def test_int_mode_unchanged(self):
        """Integer mode must still use dd_major_version parameter."""
        params: dict = {}
        clause = _dd_version_clause("p", dd_version=4, params=params)
        assert "dd_major_version" in params
        assert params["dd_major_version"] == 4
        assert "toInteger(split(" in clause

    def test_none_returns_empty(self):
        assert _dd_version_clause("p", dd_version=None) == ""


# ============================================================================
# check_dd_paths — path exists in latest version
# ============================================================================


class TestCheckDDPathsExistsLatest:
    """Path exists in latest version — should return exists=True without error."""

    @pytest.mark.asyncio
    async def test_existing_path_returns_true(self):
        gc = MagicMock()
        gc.query.return_value = [
            {
                "check_path": "equilibrium/time_slice/profiles_1d/psi",
                "id": "equilibrium/time_slice/profiles_1d/psi",
                "ids": "equilibrium",
                "data_type": "FLT_1D",
                "units": "Wb",
                "lifecycle_status": "active",
                "renamed_from": None,
                "renamed_to": None,
                "rename_version": None,
            }
        ]
        tool = GraphPathTool(gc)
        result = await tool.check_dd_paths(
            "equilibrium/time_slice/profiles_1d/psi", dd_version="4.1.1"
        )
        assert result.results[0].exists is True
        assert result.results[0].error is None
        assert result.summary["found"] == 1

    @pytest.mark.asyncio
    async def test_existing_path_no_dd_version(self):
        """Without dd_version no version-filter parameters should be passed."""
        gc = MagicMock()
        gc.query.return_value = [
            {
                "check_path": "equilibrium/time_slice/profiles_1d/psi",
                "id": "equilibrium/time_slice/profiles_1d/psi",
                "ids": "equilibrium",
                "data_type": "FLT_1D",
                "units": "Wb",
                "lifecycle_status": "active",
                "renamed_from": None,
                "renamed_to": None,
                "rename_version": None,
            }
        ]
        tool = GraphPathTool(gc)
        await tool.check_dd_paths("equilibrium/time_slice/profiles_1d/psi")
        kwargs = gc.query.call_args_list[0][1]
        # No version filter params should be present
        assert "dd_ver_major" not in kwargs
        assert "dd_major_version" not in kwargs


# ============================================================================
# check_dd_paths — path not introduced in an old version
# ============================================================================


class TestCheckDDPathsNotInOldVersion:
    """Path that doesn't exist (or wasn't introduced yet) in the requested version."""

    @pytest.mark.asyncio
    async def test_not_found_in_old_version(self):
        """When graph returns no match for the path, exists must be False."""
        gc = MagicMock()
        # Graph returns NULL id — path not found in v3.30.0
        gc.query.return_value = [
            {
                "check_path": "new_ids/some/path",
                "id": None,
                "ids": None,
                "data_type": None,
                "units": None,
                "lifecycle_status": None,
                "renamed_from": None,
                "renamed_to": None,
                "rename_version": None,
            }
        ]
        tool = GraphPathTool(gc)
        result = await tool.check_dd_paths("new_ids/some/path", dd_version="3.30.0")
        assert result.results[0].exists is False
        assert result.summary["not_found"] == 1

    @pytest.mark.asyncio
    async def test_semver_params_passed_to_query(self):
        """Parameters dd_ver_major/minor/patch must be forwarded to graph query."""
        gc = MagicMock()
        gc.query.return_value = [
            {
                "check_path": "test/path",
                "id": None,
                "ids": None,
                "data_type": None,
                "units": None,
                "lifecycle_status": None,
                "renamed_from": None,
                "renamed_to": None,
                "rename_version": None,
            }
        ]
        tool = GraphPathTool(gc)
        await tool.check_dd_paths("test/path", dd_version="3.35.0")
        kwargs = gc.query.call_args_list[0][1]
        assert kwargs.get("dd_ver_major") == 3
        assert kwargs.get("dd_ver_minor") == 35
        assert kwargs.get("dd_ver_patch") == 0


# ============================================================================
# check_dd_paths — renamed path with version context
# ============================================================================


class TestCheckDDPathsRenamedWithVersion:
    """Renamed paths should surface version-aware rename suggestion."""

    @pytest.mark.asyncio
    async def test_renamed_path_includes_rename_version(self):
        """When dd_version is given and rename_version returned, error text must mention both."""
        gc = MagicMock()
        gc.query.return_value = [
            {
                "check_path": "magnetics/bpol_probe/polarisation_angle",
                "id": None,
                "ids": None,
                "data_type": None,
                "units": None,
                "lifecycle_status": None,
                "renamed_from": "magnetics/bpol_probe/polarisation_angle",
                "renamed_to": "magnetics/bpol_probe/polarization_angle",
                "rename_version": "4.1.0",
            }
        ]
        tool = GraphPathTool(gc)
        result = await tool.check_dd_paths(
            "magnetics/bpol_probe/polarisation_angle", dd_version="4.1.1"
        )
        item = result.results[0]
        assert item.exists is False
        assert item.suggestion == "magnetics/bpol_probe/polarization_angle"
        # error text should mention the rename version
        assert item.error is not None
        assert "4.1.0" in item.error
        assert "polarization_angle" in item.error

    @pytest.mark.asyncio
    async def test_renamed_path_migration_includes_rename_version(self):
        """migration dict must carry rename_version when available."""
        gc = MagicMock()
        gc.query.return_value = [
            {
                "check_path": "old/path",
                "id": None,
                "ids": None,
                "data_type": None,
                "units": None,
                "lifecycle_status": None,
                "renamed_from": "old/path",
                "renamed_to": "new/path",
                "rename_version": "3.40.0",
            }
        ]
        tool = GraphPathTool(gc)
        result = await tool.check_dd_paths("old/path", dd_version="3.42.0")
        item = result.results[0]
        assert item.migration is not None
        assert item.migration["type"] == "renamed"
        assert item.migration["target"] == "new/path"
        assert item.migration.get("rename_version") == "3.40.0"

    @pytest.mark.asyncio
    async def test_renamed_path_no_dd_version_no_error_text(self):
        """Without dd_version, renamed paths must not generate a version-in-error message."""
        gc = MagicMock()
        gc.query.return_value = [
            {
                "check_path": "old/path",
                "id": None,
                "ids": None,
                "data_type": None,
                "units": None,
                "lifecycle_status": None,
                "renamed_from": "old/path",
                "renamed_to": "new/path",
                "rename_version": "3.40.0",
            }
        ]
        tool = GraphPathTool(gc)
        result = await tool.check_dd_paths("old/path")
        item = result.results[0]
        assert item.exists is False
        assert item.suggestion == "new/path"
        # Without version context there should be no error text
        assert item.error is None

    @pytest.mark.asyncio
    async def test_renamed_path_renamed_from_includes_rename_version(self):
        """renamed_from list entry must include rename_version when available."""
        gc = MagicMock()
        gc.query.return_value = [
            {
                "check_path": "old/path",
                "id": None,
                "ids": None,
                "data_type": None,
                "units": None,
                "lifecycle_status": None,
                "renamed_from": "old/path",
                "renamed_to": "new/path",
                "rename_version": "3.40.0",
            }
        ]
        tool = GraphPathTool(gc)
        result = await tool.check_dd_paths("old/path", dd_version="3.42.0")
        item = result.results[0]
        assert item.renamed_from is not None
        entry = item.renamed_from[0]
        assert entry["old_path"] == "old/path"
        assert entry["new_path"] == "new/path"
        assert entry.get("rename_version") == "3.40.0"
