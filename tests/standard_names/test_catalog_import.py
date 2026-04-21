"""Tests for the catalog feedback import module (Phase 4).

Tests error handling, SHA resolution, check mode, field normalization,
and deprecation of the legacy import_catalog() API.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

imas_sn = pytest.importorskip("imas_standard_names")


# =============================================================================
# Fixtures
# =============================================================================

SAMPLE_CATALOG_ENTRY = {
    "name": "electron_temperature",
    "description": "Electron temperature",
    "documentation": "The electron temperature Te is measured by Thomson scattering.",
    "kind": "scalar",
    "unit": "eV",
    "tags": [],
    "links": [],
    "dd_paths": ["core_profiles/profiles_1d/electrons/temperature"],
    "validity_domain": "core plasma",
    "constraints": ["T_e > 0"],
    "physics_domain": "core_plasma_physics",
    "status": "active",
}

SAMPLE_CATALOG_ENTRY_MINIMAL = {
    "name": "plasma_current",
    "description": "Plasma current",
    "documentation": "Total toroidal plasma current.",
    "kind": "scalar",
    "unit": "A",
    "tags": [],
    "links": [],
    "dd_paths": [],
    "validity_domain": "",
    "constraints": [],
    "physics_domain": "equilibrium",
    "status": "active",
}


@pytest.fixture()
def catalog_dir(tmp_path: Path) -> Path:
    """Create a temporary catalog directory with sample YAML files."""
    d = tmp_path / "catalog"
    d.mkdir()
    (d / "electron_temperature.yaml").write_text(yaml.safe_dump(SAMPLE_CATALOG_ENTRY))
    (d / "plasma_current.yaml").write_text(yaml.safe_dump(SAMPLE_CATALOG_ENTRY_MINIMAL))
    return d


# =============================================================================
# Error handling
# =============================================================================


class TestErrorHandling:
    """Test graceful error handling."""

    def test_handles_invalid_yaml(self, tmp_path: Path) -> None:
        """Should report errors for invalid YAML files without crashing."""
        d = tmp_path / "catalog"
        d.mkdir()
        (d / "bad.yaml").write_text(": : : invalid yaml [[[")

        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(d, dry_run=True)

        assert result.imported == 0
        assert len(result.errors) == 1
        assert "bad.yaml" in result.errors[0]

    def test_handles_non_mapping_yaml(self, tmp_path: Path) -> None:
        """Should report errors for YAML files that aren't dicts."""
        d = tmp_path / "catalog"
        d.mkdir()
        (d / "list.yaml").write_text(yaml.safe_dump(["a", "b", "c"]))

        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(d, dry_run=True)

        assert result.imported == 0
        assert len(result.errors) == 1
        assert "not a YAML mapping" in result.errors[0]

    def test_handles_incomplete_entry(self, tmp_path: Path) -> None:
        """Should report errors for entries missing required fields."""
        d = tmp_path / "catalog"
        d.mkdir()
        incomplete = {"name": "test", "kind": "scalar"}  # missing required fields
        (d / "incomplete.yaml").write_text(yaml.safe_dump(incomplete))

        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(d, dry_run=True)

        assert result.imported == 0
        assert len(result.errors) == 1
        assert "incomplete.yaml" in result.errors[0]

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory should return empty result."""
        d = tmp_path / "empty"
        d.mkdir()

        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(d, dry_run=True)

        assert result.imported == 0
        assert result.skipped == 0
        assert len(result.errors) == 0


# =============================================================================
# SHA resolution
# =============================================================================


class TestResolveCatalogSha:
    """Tests for _resolve_catalog_sha()."""

    def test_returns_sha_in_git_repo(self, tmp_path: Path) -> None:
        """Should return a 40-char SHA when run in a git repo."""
        from imas_codex.standard_names.catalog_import import _resolve_catalog_sha

        # Use the project repo itself as the catalog dir
        project_root = Path(__file__).resolve().parents[2]
        sha = _resolve_catalog_sha(project_root)
        assert sha is not None
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_returns_none_for_non_git_dir(self, tmp_path: Path) -> None:
        """Should return None for a directory that isn't a git repo."""
        from imas_codex.standard_names.catalog_import import _resolve_catalog_sha

        sha = _resolve_catalog_sha(tmp_path)
        assert sha is None

    def test_returns_none_when_git_not_found(self, tmp_path: Path) -> None:
        """Should return None when git binary is missing."""
        from imas_codex.standard_names.catalog_import import _resolve_catalog_sha

        with patch(
            "imas_codex.standard_names.catalog_import.subprocess.run"
        ) as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            sha = _resolve_catalog_sha(tmp_path)
            assert sha is None


# =============================================================================
# Check mode
# =============================================================================


class TestCheckMode:
    """Tests for check_catalog() — the --check sync comparison."""

    def test_only_in_catalog(self, catalog_dir: Path) -> None:
        """Entries in catalog but not graph should appear in only_in_catalog."""
        from imas_codex.standard_names.catalog_import import check_catalog

        # Graph has no entries
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with (
            patch("imas_codex.graph.client.GraphClient") as MockGC,
            patch(
                "imas_codex.standard_names.catalog_import._resolve_catalog_sha",
                return_value=None,
            ),
        ):
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            cr = check_catalog(catalog_dir=catalog_dir)

        assert cr.only_in_catalog == ["electron_temperature", "plasma_current"]
        assert cr.in_sync == 0

    def test_check_with_tag_filter(self, tmp_path: Path) -> None:
        """Tag filter should limit which catalog entries are checked."""
        from imas_codex.standard_names.catalog_import import check_catalog

        # Create catalog with tagged entry
        d = tmp_path / "catalog"
        d.mkdir()
        entry_with_tag = dict(SAMPLE_CATALOG_ENTRY)
        entry_with_tag["tags"] = ["spatial-profile"]
        (d / "electron_temperature.yaml").write_text(yaml.safe_dump(entry_with_tag))
        (d / "plasma_current.yaml").write_text(
            yaml.safe_dump(SAMPLE_CATALOG_ENTRY_MINIMAL)
        )

        # Graph has no entries
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with (
            patch("imas_codex.graph.client.GraphClient") as MockGC,
            patch(
                "imas_codex.standard_names.catalog_import._resolve_catalog_sha",
                return_value=None,
            ),
        ):
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            cr = check_catalog(
                catalog_dir=d,
                tag_filter=["spatial-profile"],
            )

        # Only electron_temperature has the tag — plasma_current should be filtered out
        assert cr.only_in_catalog == ["electron_temperature"]
        assert cr.in_sync == 0

    def test_check_empty_catalog(self, tmp_path: Path) -> None:
        """Empty catalog directory should return empty CheckResult."""
        from imas_codex.standard_names.catalog_import import check_catalog

        d = tmp_path / "empty_catalog"
        d.mkdir()

        with patch(
            "imas_codex.standard_names.catalog_import._resolve_catalog_sha",
            return_value=None,
        ):
            cr = check_catalog(catalog_dir=d)

        assert cr.in_sync == 0
        assert cr.only_in_catalog == []
        assert cr.only_in_graph == []
        assert cr.diverged == []


# =============================================================================
# Normalize field
# =============================================================================


class TestNormalizeField:
    """Tests for _normalize_field() comparison normalization."""

    def test_none(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field(None) is None

    def test_empty_string(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field("") is None
        assert _normalize_field("  ") is None

    def test_normal_string(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field("hello") == "hello"
        assert _normalize_field("  hello  ") == "hello"

    def test_empty_list(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field([]) is None

    def test_list_sorted(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field(["b", "a"]) == ("a", "b")
        assert _normalize_field(["a", "b"]) == ("a", "b")

    def test_numeric_passthrough(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field(42) == 42
        assert _normalize_field(3.14) == 3.14


# =============================================================================
# Deprecation warning
# =============================================================================


class TestDeprecationWarning:
    """import_catalog() should emit DeprecationWarning."""

    def test_import_catalog_warns(self, tmp_path: Path) -> None:
        """Calling import_catalog() should raise DeprecationWarning."""
        d = tmp_path / "catalog"
        d.mkdir()

        from imas_codex.standard_names.catalog_import import import_catalog

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import_catalog(d, dry_run=True)

        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        assert "run_import" in str(dep_warnings[0].message)
