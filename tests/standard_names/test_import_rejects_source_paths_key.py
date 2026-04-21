"""Tests that import rejects YAML containing source_paths or dd_paths keys.

These keys are graph-only; the error message points users at ``sn run``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

imas_sn = pytest.importorskip("imas_standard_names")

_VALID_ENTRY = {
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


def _make_isnc(tmp_path: Path, entry: dict, domain: str = "kinetics") -> Path:
    root = tmp_path / "isnc"
    sn_dir = root / "standard_names" / domain
    sn_dir.mkdir(parents=True)
    (sn_dir / f"{entry['name']}.yml").write_text(yaml.safe_dump(entry))
    return root


class TestRejectsSourcePathsKey:
    """YAML with source_paths or dd_paths key is rejected."""

    def test_source_paths_key_rejected(self, tmp_path: Path) -> None:
        entry = {
            **_VALID_ENTRY,
            "source_paths": ["dd:core_profiles/profiles_1d/electrons/temperature"],
        }
        isnc = _make_isnc(tmp_path, entry)

        from imas_codex.standard_names.catalog_import import run_import

        report = run_import(isnc, dry_run=True)

        assert len(report.errors) >= 1
        # Error should mention forbidden key and sn run
        error_text = " ".join(report.errors)
        assert "source_paths" in error_text
        assert "sn run" in error_text

    def test_dd_paths_key_rejected(self, tmp_path: Path) -> None:
        entry = {
            **_VALID_ENTRY,
            "dd_paths": ["core_profiles/profiles_1d/electrons/temperature"],
        }
        isnc = _make_isnc(tmp_path, entry)

        from imas_codex.standard_names.catalog_import import run_import

        report = run_import(isnc, dry_run=True)

        assert len(report.errors) >= 1
        error_text = " ".join(report.errors)
        assert "dd_paths" in error_text
        assert "sn run" in error_text

    def test_valid_entry_accepted(self, tmp_path: Path) -> None:
        """Entry without forbidden keys should parse fine in dry_run."""
        isnc = _make_isnc(tmp_path, _VALID_ENTRY)

        from imas_codex.standard_names.catalog_import import run_import

        report = run_import(isnc, dry_run=True)

        assert len(report.errors) == 0
        assert report.imported == 1
