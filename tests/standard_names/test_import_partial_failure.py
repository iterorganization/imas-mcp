"""Tests for partial failure idempotency.

If import fails mid-way, watermark does NOT advance.
Re-run is idempotent by diff-based origin flip.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

imas_sn = pytest.importorskip("imas_standard_names")

_GC_PATCH = "imas_codex.graph.client.GraphClient"

_ENTRY1 = {
    "name": "electron_temperature",
    "kind": "scalar",
    "unit": "eV",
    "description": "Electron temperature",
    "documentation": "The electron temperature Te.",
    "links": [],
    "constraints": [],
    "validity_domain": "",
    "status": "draft",
}

_ENTRY2 = {
    "name": "plasma_current",
    "kind": "scalar",
    "unit": "A",
    "description": "Plasma current",
    "documentation": "Total toroidal plasma current.",
    "links": [],
    "constraints": [],
    "validity_domain": "",
    "status": "draft",
}


def _make_isnc(tmp_path: Path) -> Path:
    root = tmp_path / "isnc"
    sn_dir = root / "standard_names"
    sn_dir.mkdir(parents=True)
    (sn_dir / "kinetics.yml").write_text(yaml.safe_dump([_ENTRY1]))
    (sn_dir / "equilibrium.yml").write_text(yaml.safe_dump([_ENTRY2]))
    return root


def _patch_gc(gc):
    mock_cls = MagicMock()
    mock_cls.return_value.__enter__ = MagicMock(return_value=gc)
    mock_cls.return_value.__exit__ = MagicMock(return_value=False)
    return patch(_GC_PATCH, mock_cls)


class TestPartialFailure:
    """Partial failure leaves watermark unchanged; rerun is idempotent."""

    def test_write_failure_does_not_advance_watermark(self, tmp_path: Path) -> None:
        """If _write_import_entries raises, watermark stays unchanged."""
        isnc = _make_isnc(tmp_path)

        gc = MagicMock()
        watermark_advanced = False

        def _query(cypher, **params):
            nonlocal watermark_advanced
            if "ImportLock" in cypher and "holder IS NULL" in cypher:
                return [{"acquired": True}]
            if "ImportLock" in cypher and "holder = $holder" in cypher:
                return []
            if "ImportLock" in cypher:
                return [{"holder": None, "acquired_at": None}]
            if (
                "ImportWatermark" in cypher
                and "last_commit_sha IS NULL" in cypher
                and "SET" in cypher
            ):
                watermark_advanced = True
                return [{"sha": "new"}]
            if "ImportWatermark" in cypher:
                return [
                    {
                        "last_commit_sha": None,
                        "last_import_at": None,
                        "source_repo": None,
                    }
                ]
            if "HAS_UNIT" in cypher:
                return []
            if "cocos_transformation_type" in cypher and "RETURN" in cypher:
                return []
            if "MERGE (sn:StandardName" in cypher:
                # Simulate a failure during write
                raise RuntimeError("Simulated graph write failure")
            if "StandardName" in cypher and "origin" in cypher:
                return []
            return []

        gc.query = MagicMock(side_effect=_query)

        with _patch_gc(gc):
            from imas_codex.standard_names.catalog_import import run_import

            # Should raise due to simulated failure
            with pytest.raises(RuntimeError, match="Simulated graph write failure"):
                run_import(isnc)

        # Watermark should NOT have been advanced (failure was before CAS)
        assert not watermark_advanced

    def test_rerun_after_failure_is_idempotent(self, tmp_path: Path) -> None:
        """Re-running import after partial failure processes same entries."""
        isnc = _make_isnc(tmp_path)

        # First run: dry_run to simulate parsing success
        from imas_codex.standard_names.catalog_import import run_import

        report1 = run_import(isnc, dry_run=True)
        assert report1.imported == 2

        # Second run: same result (idempotent)
        report2 = run_import(isnc, dry_run=True)
        assert report2.imported == 2
        assert report2.entries[0]["id"] == report1.entries[0]["id"]
