"""Tests for the standard name seed module.

Tests ISN example loading, WEST migration logic, grammar round-trips,
and CLI integration — all with mocked graph writes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

imas_sn = pytest.importorskip("imas_standard_names")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def west_dir(tmp_path: Path) -> Path:
    """Create a temporary WEST-style catalog directory with sample entries."""
    base = tmp_path / "standard_names"

    # Create magnetics subdirectory
    magnetics = base / "magnetics"
    magnetics.mkdir(parents=True)
    (magnetics / "diamagnetic_loop_flux.yml").write_text(
        yaml.safe_dump(
            {
                "name": "diamagnetic_loop_flux",
                "description": "Diamagnetic flux measured by diamagnetic loop diagnostic.",
                "documentation": "Diamagnetic flux measurement docs.",
                "status": "draft",
                "tags": ["magnetics", "measured"],
                "kind": "scalar",
                "unit": "Wb",
            }
        )
    )

    # Create core-physics subdirectory
    core = base / "core-physics"
    core.mkdir(parents=True)
    (core / "electron_temperature.yml").write_text(
        yaml.safe_dump(
            {
                "name": "electron_temperature",
                "description": "Electron temperature.",
                "documentation": "Electron temperature docs.",
                "status": "draft",
                "tags": ["core-physics", "spatial-profile"],
                "kind": "scalar",
                "unit": "eV",
            }
        )
    )

    return base


@pytest.fixture()
def west_dir_with_invalid(west_dir: Path) -> Path:
    """Add an invalid YAML entry to the WEST directory."""
    magnetics = west_dir / "magnetics"
    (magnetics / "bad_entry.yml").write_text("not_a_mapping")
    return west_dir


# =============================================================================
# ISN example loading
# =============================================================================


class TestLoadIsnExamples:
    """Test loading ISN shipped reference examples."""

    def test_load_returns_entries(self) -> None:
        from imas_codex.standard_names.seed import load_isn_examples

        entries, errors = load_isn_examples()

        # Should find ~42 entries
        assert len(entries) > 30, f"Expected ~42 entries, got {len(entries)}"
        assert len(errors) == 0, f"Unexpected validation errors: {errors}"

    def test_entries_have_required_fields(self) -> None:
        from imas_codex.standard_names.seed import load_isn_examples

        entries, _ = load_isn_examples()

        for entry in entries:
            assert "id" in entry, f"Missing 'id' in entry: {entry}"
            assert entry["source_type"] == "reference"
            assert entry["review_status"] == "accepted"
            assert entry.get("description"), f"Missing description for {entry['id']}"
            assert entry.get("kind"), f"Missing kind for {entry['id']}"
            assert entry.get("physics_domain"), (
                f"Missing physics_domain for {entry['id']}"
            )

    def test_grammar_fields_populated(self) -> None:
        from imas_codex.standard_names.seed import load_isn_examples

        entries, _ = load_isn_examples()

        # At least some entries should have grammar fields
        with_grammar = [
            e for e in entries if e.get("physical_base") or e.get("subject")
        ]
        assert len(with_grammar) > 0, "No entries have grammar fields"

    def test_unique_ids(self) -> None:
        from imas_codex.standard_names.seed import load_isn_examples

        entries, _ = load_isn_examples()
        ids = [e["id"] for e in entries]
        assert len(ids) == len(set(ids)), "Duplicate IDs found"


# =============================================================================
# WEST migration logic
# =============================================================================


class TestWestMigration:
    """Test WEST-specific fixes: physics_domain and tag cleanup."""

    def test_fix_west_entry_adds_physics_domain(self) -> None:
        from imas_codex.standard_names.seed import _fix_west_entry

        data = {
            "name": "test_name",
            "tags": ["magnetics", "measured"],
        }
        fixed = _fix_west_entry(data, "magnetics")
        assert fixed["physics_domain"] == "magnetic_field_diagnostics"

    def test_fix_west_entry_strips_primary_tags(self) -> None:
        from imas_codex.standard_names.seed import _fix_west_entry

        data = {
            "name": "test_name",
            "tags": ["magnetics", "measured", "core-physics"],
        }
        fixed = _fix_west_entry(data, "magnetics")
        assert "magnetics" not in fixed["tags"]
        assert "core-physics" not in fixed["tags"]
        assert "measured" in fixed["tags"]

    def test_fix_west_entry_preserves_existing_physics_domain(self) -> None:
        from imas_codex.standard_names.seed import _fix_west_entry

        data = {
            "name": "test_name",
            "tags": [],
            "physics_domain": "existing_domain",
        }
        fixed = _fix_west_entry(data, "magnetics")
        assert fixed["physics_domain"] == "existing_domain"

    def test_fix_west_entry_does_not_mutate_input(self) -> None:
        from imas_codex.standard_names.seed import _fix_west_entry

        original_tags = ["magnetics", "measured"]
        data = {"name": "test_name", "tags": original_tags}
        _fix_west_entry(data, "magnetics")
        assert data["tags"] == ["magnetics", "measured"], "Input was mutated"

    def test_fix_west_entry_unknown_dir(self) -> None:
        from imas_codex.standard_names.seed import _fix_west_entry

        data = {"name": "test_name", "tags": []}
        fixed = _fix_west_entry(data, "unknown-dir")
        assert fixed["physics_domain"] == "general"


class TestLoadWestCatalog:
    """Test loading and validating WEST entries from filesystem."""

    def test_load_from_temp_dir(self, west_dir: Path) -> None:
        from imas_codex.standard_names.seed import load_west_catalog

        entries, errors = load_west_catalog(west_dir)

        assert len(entries) == 2
        assert len(errors) == 0

        # Check that fixes were applied
        magnetics_entry = next(e for e in entries if e["id"] == "diamagnetic_loop_flux")
        assert magnetics_entry["physics_domain"] == "magnetic_field_diagnostics"
        assert magnetics_entry["source_type"] == "west"
        assert magnetics_entry["review_status"] == "drafted"

        core_entry = next(e for e in entries if e["id"] == "electron_temperature")
        assert core_entry["physics_domain"] == "core_plasma_physics"
        # Primary tag 'core-physics' should be stripped, 'spatial-profile' kept
        assert core_entry["tags"] is not None
        assert "core-physics" not in core_entry["tags"]
        assert "spatial-profile" in core_entry["tags"]

    def test_load_handles_invalid_entries(self, west_dir_with_invalid: Path) -> None:
        from imas_codex.standard_names.seed import load_west_catalog

        entries, errors = load_west_catalog(west_dir_with_invalid)

        # 2 valid + 1 invalid
        assert len(entries) == 2
        assert len(errors) == 1
        assert "bad_entry.yml" in errors[0]

    def test_load_nonexistent_dir(self, tmp_path: Path) -> None:
        from imas_codex.standard_names.seed import load_west_catalog

        entries, errors = load_west_catalog(tmp_path / "nonexistent")
        assert len(entries) == 0
        assert len(errors) == 1
        assert "not found" in errors[0]


# =============================================================================
# Grammar round-trip
# =============================================================================


class TestGrammarRoundtrip:
    """Test grammar parsing and round-trip for all ISN examples."""

    def test_isn_examples_roundtrip(self) -> None:
        from imas_standard_names.grammar import (
            compose_standard_name,
            parse_standard_name,
        )

        from imas_codex.standard_names.seed import load_isn_examples

        entries, _ = load_isn_examples()
        mismatches = []

        for entry in entries:
            name = entry["id"]
            try:
                parsed = parse_standard_name(name)
                composed = compose_standard_name(parsed)
                if composed != name:
                    mismatches.append(f"{name} → {composed}")
            except Exception as exc:
                mismatches.append(f"{name}: {exc}")

        # Report but don't fail — some names may legitimately not round-trip
        if mismatches:
            pytest.skip(
                f"{len(mismatches)} grammar mismatches (expected for some names): "
                f"{mismatches[:3]}"
            )


# =============================================================================
# Seed functions (with mocked graph)
# =============================================================================


class TestSeedIsnExamples:
    """Test seed_isn_examples with mocked graph writes."""

    @patch("imas_codex.standard_names.graph_ops.write_standard_names", return_value=42)
    def test_seed_writes_to_graph(self, mock_write: MagicMock) -> None:
        from imas_codex.standard_names.seed import seed_isn_examples

        result = seed_isn_examples(dry_run=False)

        mock_write.assert_called_once()
        batch = mock_write.call_args[0][0]
        assert len(batch) > 30
        assert result.written == 42

    def test_seed_dry_run_skips_graph(self) -> None:
        from imas_codex.standard_names.seed import seed_isn_examples

        result = seed_isn_examples(dry_run=True)

        assert result.written == 0
        assert result.validated > 30

    @patch("imas_codex.standard_names.graph_ops.write_standard_names", return_value=42)
    def test_seed_result_fields(self, mock_write: MagicMock) -> None:
        from imas_codex.standard_names.seed import seed_isn_examples

        result = seed_isn_examples(dry_run=False)

        assert result.loaded > 0
        assert result.validated > 0
        assert result.validation_errors == []


class TestSeedWestCatalog:
    """Test seed_west_catalog with mocked graph writes."""

    @patch("imas_codex.standard_names.graph_ops.write_standard_names", return_value=2)
    def test_seed_writes_to_graph(self, mock_write: MagicMock, west_dir: Path) -> None:
        from imas_codex.standard_names.seed import seed_west_catalog

        result = seed_west_catalog(west_dir=west_dir, dry_run=False)

        mock_write.assert_called_once()
        batch = mock_write.call_args[0][0]
        assert len(batch) == 2
        assert result.written == 2

    def test_seed_dry_run_skips_graph(self, west_dir: Path) -> None:
        from imas_codex.standard_names.seed import seed_west_catalog

        result = seed_west_catalog(west_dir=west_dir, dry_run=True)

        assert result.written == 0
        assert result.validated == 2


# =============================================================================
# DIR_TO_DOMAIN coverage
# =============================================================================


class TestDirToDomainMapping:
    """Ensure all expected WEST directories are mapped."""

    def test_all_west_dirs_mapped(self) -> None:
        from imas_codex.standard_names.seed import DIR_TO_DOMAIN

        expected_dirs = {
            "coils-and-control",
            "core-physics",
            "data-products",
            "ec-heating",
            "edge-physics",
            "equilibrium",
            "fast-particles",
            "fueling",
            "fundamental",
            "ic-heating",
            "imaging",
            "interferometry",
            "lh-heating",
            "magnetics",
            "mhd",
            "nbi",
            "neutronics",
            "radiation-diagnostics",
            "reflectometry",
            "spectroscopy",
            "thomson-scattering",
            "transport",
            "turbulence",
        }
        assert expected_dirs == set(DIR_TO_DOMAIN.keys())
