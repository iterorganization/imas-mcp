"""Tests for import watermark+lock concurrency control.

Two concurrent imports: second aborts cleanly without corrupting state.
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
}


def _make_isnc(tmp_path: Path) -> Path:
    root = tmp_path / "isnc"
    sn_dir = root / "standard_names"
    sn_dir.mkdir(parents=True)
    (sn_dir / "kinetics.yml").write_text(yaml.safe_dump([_ENTRY]))
    return root


def _patch_gc(gc):
    mock_cls = MagicMock()
    mock_cls.return_value.__enter__ = MagicMock(return_value=gc)
    mock_cls.return_value.__exit__ = MagicMock(return_value=False)
    return patch(_GC_PATCH, mock_cls)


class TestWatermarkLock:
    """Lock acquisition and watermark CAS semantics."""

    def test_second_import_aborts_when_lock_held(self, tmp_path: Path) -> None:
        """If lock is held by another process, import aborts cleanly."""
        isnc = _make_isnc(tmp_path)

        gc = MagicMock()

        def _query(cypher, **params):
            if "ImportLock" in cypher and "holder IS NULL" in cypher:
                # Lock is held — cannot acquire
                return []
            if "ImportLock" in cypher and "RETURN" in cypher:
                return [
                    {
                        "holder": "other-host:12345",
                        "acquired_at": "2026-01-01T00:00:00Z",
                    }
                ]
            if "stale" in cypher.lower() or "duration" in cypher:
                return []
            return []

        gc.query = MagicMock(side_effect=_query)

        with _patch_gc(gc):
            from imas_codex.standard_names.catalog_import import run_import

            report = run_import(isnc)

        assert len(report.errors) >= 1
        error_text = " ".join(report.errors)
        assert "lock" in error_text.lower()
        assert report.imported == 0

    def test_watermark_cas_failure_reported(self, tmp_path: Path) -> None:
        """If watermark CAS fails, error is reported but entries are written."""
        isnc = _make_isnc(tmp_path)

        gc = MagicMock()

        def _query(cypher, **params):
            if "ImportLock" in cypher and "holder IS NULL" in cypher:
                return [{"acquired": True}]
            if "ImportLock" in cypher and "holder = $holder" in cypher:
                return []  # release succeeds
            if "ImportLock" in cypher:
                return [{"holder": None, "acquired_at": None}]
            if (
                "ImportWatermark" in cypher
                and "last_commit_sha = $expected" in cypher
                and "SET" in cypher
            ):
                # CAS fails — another import moved the watermark
                return []
            if (
                "ImportWatermark" in cypher
                and "last_commit_sha IS NULL" in cypher
                and "SET" in cypher
            ):
                # CAS for first import fails too
                return []
            if "ImportWatermark" in cypher:
                return [
                    {
                        "last_commit_sha": "old-sha-before",
                        "last_import_at": None,
                        "source_repo": None,
                    }
                ]
            if "HAS_UNIT" in cypher and "RETURN" in cypher:
                return []
            if "cocos_transformation_type" in cypher and "RETURN" in cypher:
                return []
            if "StandardName" in cypher and "origin" in cypher:
                return []
            return []

        gc.query = MagicMock(side_effect=_query)

        with (
            _patch_gc(gc),
            patch(
                "imas_codex.standard_names.catalog_import._resolve_catalog_sha",
                return_value="abc123def456",
            ),
            patch(
                "imas_codex.standard_names.catalog_import._is_git_repo",
                return_value=True,
            ),
        ):
            from imas_codex.standard_names.catalog_import import run_import

            report = run_import(isnc)

        # Entries should be written despite CAS failure
        assert report.imported > 0
        assert report.watermark_advanced is False
        # CAS error should be reported
        cas_errors = [e for e in report.errors if "watermark" in e.lower()]
        assert len(cas_errors) >= 1
