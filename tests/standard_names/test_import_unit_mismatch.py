"""Tests for unit mismatch rejection on import.

Catalog unit differs from graph HAS_UNIT → rejected unless override flag.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

imas_sn = pytest.importorskip("imas_standard_names")

_GC_PATCH = "imas_codex.graph.client.GraphClient"

_ENTRY = {
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


def _mock_gc_with_unit(graph_unit: str):
    """Build mock GC where the graph has a different unit."""
    gc = MagicMock()

    def _query(cypher, **params):
        if "ImportLock" in cypher and "holder IS NULL" in cypher:
            return [{"acquired": True}]
        if "ImportLock" in cypher and "holder = $holder" in cypher:
            return []
        if "ImportLock" in cypher and "RETURN" in cypher:
            return [{"holder": None, "acquired_at": None}]
        if "ImportWatermark" in cypher and "SET" in cypher:
            return [{"sha": "abc123"}]
        if "ImportWatermark" in cypher:
            return [
                {
                    "last_commit_sha": None,
                    "last_import_at": None,
                    "source_repo": None,
                }
            ]
        # Unit validation query — contains "sn.unit <> b.unit"
        if "sn.unit <> b.unit" in cypher:
            # Simulate WHERE filter: only return rows if unit actually differs
            catalog_unit = "eV"  # from _ENTRY
            if graph_unit != catalog_unit:
                return [
                    {
                        "name": "electron_temperature",
                        "existing_unit": graph_unit,
                        "incoming_unit": catalog_unit,
                    }
                ]
            return []
        if "StandardName" in cypher and "origin" in cypher:
            return []
        return []

    gc.query = MagicMock(side_effect=_query)
    return gc


def _patch_gc(gc):
    mock_cls = MagicMock()
    mock_cls.return_value.__enter__ = MagicMock(return_value=gc)
    mock_cls.return_value.__exit__ = MagicMock(return_value=False)
    return patch(_GC_PATCH, mock_cls)


class TestUnitMismatch:
    """Unit validation: reject mismatch unless --accept-unit-override."""

    def test_unit_mismatch_rejected(self, tmp_path: Path) -> None:
        """Catalog has eV but graph has keV → error."""
        isnc = _make_isnc(tmp_path, _ENTRY)
        gc = _mock_gc_with_unit("keV")

        with _patch_gc(gc):
            from imas_codex.standard_names.catalog_import import run_import

            report = run_import(isnc, accept_unit_override=False)

        assert len(report.errors) >= 1
        error_text = " ".join(report.errors)
        assert "unit" in error_text.lower()
        assert "accept-unit-override" in error_text

    def test_unit_mismatch_accepted_with_override(self, tmp_path: Path) -> None:
        """With override flag, unit mismatch is allowed."""
        isnc = _make_isnc(tmp_path, _ENTRY)
        gc = _mock_gc_with_unit("keV")

        with _patch_gc(gc):
            from imas_codex.standard_names.catalog_import import run_import

            report = run_import(isnc, accept_unit_override=True)

        unit_errors = [e for e in report.errors if "unit" in e.lower()]
        assert len(unit_errors) == 0

    def test_matching_unit_no_error(self, tmp_path: Path) -> None:
        """Same unit in catalog and graph → no error."""
        isnc = _make_isnc(tmp_path, _ENTRY)
        gc = _mock_gc_with_unit("eV")

        with _patch_gc(gc):
            from imas_codex.standard_names.catalog_import import run_import

            report = run_import(isnc)

        unit_errors = [e for e in report.errors if "unit" in e.lower()]
        assert len(unit_errors) == 0
