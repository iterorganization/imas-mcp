"""Tests for pipeline_version.py — hash stability, diff, and clear gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.pipeline_version import (
    PIPELINE_CODE,
    PIPELINE_PROMPTS,
    compute_pipeline_hash,
    diff_pipeline_hashes,
)

# ---------------------------------------------------------------------------
# compute_pipeline_hash
# ---------------------------------------------------------------------------


def test_pipeline_hash_returns_composite_key() -> None:
    """compute_pipeline_hash must always return a '_composite' key."""
    h = compute_pipeline_hash()
    assert "_composite" in h
    assert len(h["_composite"]) == 16, "composite must be 16 hex chars"


def test_pipeline_hash_stable_across_calls() -> None:
    """Calling compute_pipeline_hash twice must return the same composite."""
    h1 = compute_pipeline_hash()
    h2 = compute_pipeline_hash()
    assert h1 == h2


def test_pipeline_hash_includes_isn_version() -> None:
    """The hash dict must include an 'isn_version' key."""
    h = compute_pipeline_hash()
    assert "isn_version" in h


def test_pipeline_hash_changes_when_prompt_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Modifying a tracked prompt file must change the composite hash."""
    # Use a real prompt file that exists
    project_root = Path(__file__).parents[2]
    existing_prompts = [p for p in PIPELINE_PROMPTS if (project_root / p).exists()]
    if not existing_prompts:
        pytest.skip("No tracked prompt files exist in this checkout")

    target = project_root / existing_prompts[0]
    original_bytes = target.read_bytes()

    try:
        h_before = compute_pipeline_hash()
        # Append a whitespace change
        target.write_bytes(original_bytes + b"\n# test\n")
        h_after = compute_pipeline_hash()
        assert h_before["_composite"] != h_after["_composite"], (
            "composite hash should change when a prompt file is modified"
        )
    finally:
        # Always restore
        target.write_bytes(original_bytes)


def test_pipeline_hash_graceful_when_isn_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When imas-standard-names is not installed, 'isn_version' is 'unknown'."""
    with patch("imas_codex.standard_names.pipeline_version.hashlib"):
        pass  # just verify the import patch approach below works

    # Patch importlib.metadata.version to raise
    with patch(
        "importlib.metadata.version",
        side_effect=Exception("package not found"),
    ):
        h = compute_pipeline_hash()
        assert h.get("isn_version") == "unknown"


# ---------------------------------------------------------------------------
# diff_pipeline_hashes
# ---------------------------------------------------------------------------


def test_diff_detects_changed_key() -> None:
    old = {"a": "aaa", "b": "bbb", "_composite": "xxx"}
    new = {"a": "aaa", "b": "ccc", "_composite": "yyy"}
    changed = diff_pipeline_hashes(old, new)
    assert "b" in changed
    assert "a" not in changed
    assert "_composite" not in changed


def test_diff_detects_added_key() -> None:
    old = {"a": "aaa"}
    new = {"a": "aaa", "c": "ccc"}
    changed = diff_pipeline_hashes(old, new)
    assert "c" in changed


def test_diff_detects_removed_key() -> None:
    old = {"a": "aaa", "b": "bbb"}
    new = {"a": "aaa"}
    changed = diff_pipeline_hashes(old, new)
    assert "b" in changed


def test_diff_empty_when_no_changes() -> None:
    h = {"a": "111", "b": "222", "_composite": "abc"}
    assert diff_pipeline_hashes(h, h) == []


# ---------------------------------------------------------------------------
# _check_pipeline_clear_gate (via CLI)
# ---------------------------------------------------------------------------


def _make_gc_mock(composite: str, detail_dict: dict, name_count: int) -> MagicMock:
    """Build a mock GraphClient that returns given hash/count values."""
    gc = MagicMock()
    # Context manager support
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    def _query(cypher: str, **kwargs):
        if "SNRun" in cypher and "pipeline_hash" in cypher:
            return [
                {
                    "composite": composite,
                    "detail": json.dumps(detail_dict),
                    "started_at": "2026-01-01T00:00:00",
                    "run_id": "test-run-id",
                }
            ]
        if "StandardName" in cypher and "count" in cypher:
            return [{"n": name_count}]
        return []

    gc.query = _query
    return gc


def test_clear_gate_passes_when_hashes_match() -> None:
    """Gate must not raise when current and stored hashes match."""
    from imas_codex.cli.sn import _check_pipeline_clear_gate

    current = compute_pipeline_hash()
    gc_mock = _make_gc_mock(current["_composite"], {}, 100)

    with (
        patch("imas_codex.graph.client.GraphClient", return_value=gc_mock),
        patch(
            "imas_codex.standard_names.pipeline_version.compute_pipeline_hash",
            return_value=current,
        ),
    ):
        _check_pipeline_clear_gate()  # must not raise


def test_clear_gate_passes_when_no_prior_run() -> None:
    """Gate must not raise on a fresh graph (no SNRun with pipeline_hash)."""
    from imas_codex.cli.sn import _check_pipeline_clear_gate

    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    gc.query = MagicMock(return_value=[])  # no rows

    with patch("imas_codex.graph.client.GraphClient", return_value=gc):
        _check_pipeline_clear_gate()  # must not raise


def test_clear_gate_blocks_when_hash_differs() -> None:
    """Gate must exit(1) when hash changed and there are generated names."""
    from imas_codex.cli.sn import _check_pipeline_clear_gate

    current = compute_pipeline_hash()
    # Store a deliberately different composite
    gc_mock = _make_gc_mock("deadbeef12345678", {}, 50)

    with (
        patch("imas_codex.graph.client.GraphClient", return_value=gc_mock),
        patch(
            "imas_codex.standard_names.pipeline_version.compute_pipeline_hash",
            return_value=current,
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        _check_pipeline_clear_gate()

    assert exc_info.value.code == 1


def test_clear_gate_passes_when_no_names_exist() -> None:
    """Gate must not block when hashes differ but graph has zero names."""
    from imas_codex.cli.sn import _check_pipeline_clear_gate

    current = compute_pipeline_hash()
    gc_mock = _make_gc_mock("deadbeef12345678", {}, 0)

    with (
        patch("imas_codex.graph.client.GraphClient", return_value=gc_mock),
        patch(
            "imas_codex.standard_names.pipeline_version.compute_pipeline_hash",
            return_value=current,
        ),
    ):
        _check_pipeline_clear_gate()  # must not raise (empty graph)


def test_clear_gate_skips_silently_on_graph_error() -> None:
    """Gate must not raise when the graph is unreachable."""
    from imas_codex.cli.sn import _check_pipeline_clear_gate

    with patch(
        "imas_codex.graph.client.GraphClient",
        side_effect=Exception("connection refused"),
    ):
        _check_pipeline_clear_gate()  # must not raise
