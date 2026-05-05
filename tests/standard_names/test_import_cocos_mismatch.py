"""Tests for COCOS transformation type mismatch rejection on import.

Catalog cocos_transformation_type differs from graph → rejected unless override.
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
    "links": [],
    "constraints": [],
    "validity_domain": "",
    "status": "draft",
    "cocos_transformation_type": "psi_like",
}


def _make_isnc(tmp_path: Path, entry: dict, domain: str = "kinetics") -> Path:
    root = tmp_path / "isnc"
    sn_dir = root / "standard_names" / domain
    sn_dir.mkdir(parents=True)
    (sn_dir / f"{entry['name']}.yml").write_text(yaml.safe_dump(entry))
    return root


def _mock_gc_with_cocos(graph_cocos: str):
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
        # COCOS validation query — only return mismatch rows when cocos differs
        if "sn.cocos_transformation_type <> b.cocos" in cypher:
            # Simulate WHERE filter: only return rows if cocos actually differs
            catalog_cocos = "psi_like"  # from _ENTRY
            if graph_cocos != catalog_cocos:
                return [
                    {
                        "name": "electron_temperature",
                        "existing": graph_cocos,
                        "incoming": catalog_cocos,
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


class TestCocosMismatch:
    """COCOS validation: reject mismatch unless --accept-cocos-override."""

    def test_cocos_mismatch_rejected(self, tmp_path: Path) -> None:
        isnc = _make_isnc(tmp_path, _ENTRY)
        gc = _mock_gc_with_cocos("ip_like")

        with _patch_gc(gc):
            from imas_codex.standard_names.catalog_import import run_import

            report = run_import(isnc, accept_cocos_override=False)

        assert len(report.errors) >= 1
        error_text = " ".join(report.errors)
        assert "cocos" in error_text.lower()
        assert "accept-cocos-override" in error_text

    def test_cocos_mismatch_accepted_with_override(self, tmp_path: Path) -> None:
        isnc = _make_isnc(tmp_path, _ENTRY)
        gc = _mock_gc_with_cocos("ip_like")

        with _patch_gc(gc):
            from imas_codex.standard_names.catalog_import import run_import

            report = run_import(isnc, accept_cocos_override=True)

        cocos_errors = [e for e in report.errors if "cocos" in e.lower()]
        assert len(cocos_errors) == 0

    def test_matching_cocos_no_error(self, tmp_path: Path) -> None:
        isnc = _make_isnc(tmp_path, _ENTRY)
        gc = _mock_gc_with_cocos("psi_like")

        with _patch_gc(gc):
            from imas_codex.standard_names.catalog_import import run_import

            report = run_import(isnc)

        cocos_errors = [e for e in report.errors if "cocos" in e.lower()]
        assert len(cocos_errors) == 0
