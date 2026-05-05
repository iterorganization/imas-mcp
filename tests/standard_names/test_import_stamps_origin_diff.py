"""Tests for diff-based origin stamping on import.

No-op imports preserve ``origin='pipeline'``; edits flip to ``origin='catalog_edit'``.
All graph interactions are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

imas_sn = pytest.importorskip("imas_standard_names")

# Patch target: GraphClient is imported locally inside run_import and import_sync
_GC_PATCH = "imas_codex.graph.client.GraphClient"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ENTRY_BASE = {
    "name": "electron_temperature",
    "kind": "scalar",
    "unit": "eV",
    "description": "Electron temperature",
    "documentation": "The electron temperature Te.",
    "links": [],
    "constraints": ["T_e > 0"],
    "validity_domain": "core plasma",
    "status": "draft",
}


def _make_isnc(tmp_path: Path, entries: list[dict], domain: str = "kinetics") -> Path:
    """Create a minimal ISNC checkout with standard_names/<domain>.yml."""
    root = tmp_path / "isnc"
    sn_dir = root / "standard_names"
    sn_dir.mkdir(parents=True)
    (sn_dir / f"{domain}.yml").write_text(yaml.safe_dump(entries))
    return root


def _mock_gc(graph_state: dict | None = None):
    """Build a mock GraphClient with configurable query responses."""
    gc = MagicMock()

    def _query(cypher, **params):
        # Route based on query content
        if "ImportLock" in cypher and "holder IS NULL" in cypher:
            return [{"acquired": True}]
        if "ImportLock" in cypher and "holder = $holder" in cypher:
            return []  # release
        if "ImportLock" in cypher:
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
        if "StandardName" in cypher and "RETURN" in cypher and "origin" in cypher:
            # Graph state query for diff
            if graph_state:
                return [
                    {"id": k, **v, "origin": v.get("origin", "pipeline")}
                    for k, v in graph_state.items()
                ]
            return []
        if "HAS_UNIT" in cypher and "RETURN" in cypher:
            return []
        if "cocos_transformation_type" in cypher and "RETURN" in cypher:
            return []
        return []

    gc.query = MagicMock(side_effect=_query)
    return gc


def _patch_gc(gc):
    """Create a patch for GraphClient that returns the mock as context manager."""
    mock_cls = MagicMock()
    mock_cls.return_value.__enter__ = MagicMock(return_value=gc)
    mock_cls.return_value.__exit__ = MagicMock(return_value=False)
    return patch(_GC_PATCH, mock_cls)


class TestOriginNoOp:
    """No-op import (identical fields) preserves origin=pipeline."""

    def test_noop_preserves_pipeline_origin(self, tmp_path: Path) -> None:
        isnc = _make_isnc(tmp_path, [_ENTRY_BASE])

        # Graph has same values as catalog
        graph_state = {
            "electron_temperature": {
                "description": "Electron temperature",
                "documentation": "The electron temperature Te.",
                "kind": "scalar",
                "links": [],
                "status": "draft",
                "deprecates": None,
                "superseded_by": None,
                "validity_domain": "core plasma",
                "constraints": ["T_e > 0"],
                "origin": "pipeline",
            }
        }

        gc = _mock_gc(graph_state)

        with _patch_gc(gc):
            from imas_codex.standard_names.catalog_import import run_import

            report = run_import(isnc)

        # Check that _origin was set to 'pipeline' (no diff)
        assert report.imported > 0
        written_entries = [
            e for e in report.entries if e["id"] == "electron_temperature"
        ]
        assert len(written_entries) == 1
        assert written_entries[0]["_origin"] == "pipeline"
        assert report.skipped == 1
        assert report.updated == 0


class TestOriginFlip:
    """Edit in catalog flips origin to catalog_edit."""

    def test_description_change_flips_origin(self, tmp_path: Path) -> None:
        modified = {**_ENTRY_BASE, "description": "Improved electron temperature desc"}
        isnc = _make_isnc(tmp_path, [modified])

        graph_state = {
            "electron_temperature": {
                "description": "Electron temperature",
                "documentation": "The electron temperature Te.",
                "kind": "scalar",
                "links": [],
                "status": "draft",
                "deprecates": None,
                "superseded_by": None,
                "validity_domain": "core plasma",
                "constraints": ["T_e > 0"],
                "origin": "pipeline",
            }
        }

        gc = _mock_gc(graph_state)

        with _patch_gc(gc):
            from imas_codex.standard_names.catalog_import import run_import

            report = run_import(isnc)

        written = [e for e in report.entries if e["id"] == "electron_temperature"]
        assert len(written) == 1
        assert written[0]["_origin"] == "catalog_edit"
        assert report.updated == 1

    def test_new_entry_stamps_catalog_edit(self, tmp_path: Path) -> None:
        """Brand new entries (not in graph) get origin=catalog_edit."""
        isnc = _make_isnc(tmp_path, [_ENTRY_BASE])

        gc = _mock_gc(graph_state={})  # empty graph

        with _patch_gc(gc):
            from imas_codex.standard_names.catalog_import import run_import

            report = run_import(isnc)

        written = [e for e in report.entries if e["id"] == "electron_temperature"]
        assert written[0]["_origin"] == "catalog_edit"
        assert report.created == 1
