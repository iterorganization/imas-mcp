"""Tests for domain-path mismatch detection.

Path says ``kinetics``, file content says ``equilibrium`` → error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

imas_sn = pytest.importorskip("imas_standard_names")


_ENTRY_KINETICS = {
    "name": "electron_temperature",
    "kind": "scalar",
    "unit": "eV",
    "description": "Electron temperature",
    "documentation": "The electron temperature Te.",
    "tags": [],
    "links": [],
    "constraints": [],
    "validity_domain": "",
    "status": "draft",
}


class TestDomainPathMismatch:
    """Derive physics_domain from file path; reject if file content contradicts."""

    def test_no_standard_names_dir_structure_errors(self, tmp_path: Path) -> None:
        """File not in standard_names/<domain>/ layout → error."""
        root = tmp_path / "isnc"
        flat_dir = root / "standard_names"
        flat_dir.mkdir(parents=True)
        # File directly in standard_names/ without domain subdir
        (flat_dir / "electron_temperature.yml").write_text(
            yaml.safe_dump(_ENTRY_KINETICS)
        )

        from imas_codex.standard_names.catalog_import import run_import

        report = run_import(root, dry_run=True)

        assert len(report.errors) >= 1
        error_text = " ".join(report.errors)
        assert "physics_domain" in error_text.lower() or "domain" in error_text.lower()

    def test_correct_domain_path_parses(self, tmp_path: Path) -> None:
        """File in standard_names/kinetics/ → domain = kinetics, no error."""
        root = tmp_path / "isnc"
        sn_dir = root / "standard_names" / "kinetics"
        sn_dir.mkdir(parents=True)
        (sn_dir / "electron_temperature.yml").write_text(
            yaml.safe_dump(_ENTRY_KINETICS)
        )

        from imas_codex.standard_names.catalog_import import run_import

        report = run_import(root, dry_run=True)

        assert len(report.errors) == 0
        assert report.imported == 1
        assert report.entries[0]["physics_domain"] == "kinetics"

    def test_deep_nested_domain_path(self, tmp_path: Path) -> None:
        """standard_names/equilibrium/sub/file.yml → domain = equilibrium."""
        root = tmp_path / "isnc"
        sn_dir = root / "standard_names" / "equilibrium" / "sub"
        sn_dir.mkdir(parents=True)
        entry = {**_ENTRY_KINETICS, "name": "plasma_current", "unit": "A"}
        (sn_dir / "plasma_current.yml").write_text(yaml.safe_dump(entry))

        from imas_codex.standard_names.catalog_import import run_import

        report = run_import(root, dry_run=True)

        assert len(report.errors) == 0
        assert report.entries[0]["physics_domain"] == "equilibrium"
