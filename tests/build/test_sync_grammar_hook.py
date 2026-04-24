"""Tests for the sync-grammar build hook.

The hook lives inside ``CustomBuildHook._sync_grammar_best_effort`` in
``hatch_build_hooks.py``.  These tests exercise the soft-fail behaviour
so that a missing graph or missing ISN package never blocks a build.

``hatchling`` is a build-time dependency only and is not installed in
the test environment.  We stub it out in ``sys.modules`` so the module
can be imported.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_hatchling() -> None:
    """Insert a minimal hatchling stub so hatch_build_hooks.py can import."""
    if "hatchling" not in sys.modules:
        hatchling = types.ModuleType("hatchling")
        builders = types.ModuleType("hatchling.builders")
        hooks = types.ModuleType("hatchling.builders.hooks")
        plugin = types.ModuleType("hatchling.builders.hooks.plugin")
        interface = types.ModuleType("hatchling.builders.hooks.plugin.interface")

        class BuildHookInterface:  # minimal stub
            def __init__(
                self,
                root,
                config,
                build_config=None,
                metadata=None,
                directory=None,
                target_name=None,
                app=None,
            ):
                self.root = root
                self.config = config or {}

        interface.BuildHookInterface = BuildHookInterface  # type: ignore[attr-defined]
        sys.modules["hatchling"] = hatchling
        sys.modules["hatchling.builders"] = builders
        sys.modules["hatchling.builders.hooks"] = hooks
        sys.modules["hatchling.builders.hooks.plugin"] = plugin
        sys.modules["hatchling.builders.hooks.plugin.interface"] = interface


def _make_hook(tmp_path: Path) -> Any:
    """Import and instantiate ``CustomBuildHook`` from the project root."""
    _stub_hatchling()

    project_root = Path(__file__).parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import fresh each time so monkeypatching works cleanly
    if "hatch_build_hooks" in sys.modules:
        del sys.modules["hatch_build_hooks"]

    import hatch_build_hooks  # noqa: PLC0415

    return hatch_build_hooks.CustomBuildHook(
        root=str(tmp_path),
        config={},
        build_config=MagicMock(),
        metadata=MagicMock(),
        directory=str(tmp_path),
        target_name="wheel",
        app=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sync_grammar_skipped_via_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IMAS_CODEX_SKIP_GRAMMAR_SYNC=1 must suppress the grammar sync."""
    monkeypatch.setenv("IMAS_CODEX_SKIP_GRAMMAR_SYNC", "1")
    hook = _make_hook(tmp_path)

    with patch(
        "imas_codex.standard_names.grammar_sync.sync_isn_grammar_to_graph"
    ) as mock_sync:
        hook._sync_grammar_best_effort(tmp_path.parent)
        assert not mock_sync.called, (
            "sync should be skipped when IMAS_CODEX_SKIP_GRAMMAR_SYNC=1"
        )


def test_sync_grammar_skips_silently_on_import_error(tmp_path: Path) -> None:
    """When imas_codex is not importable (bootstrap), the hook must not raise."""
    hook = _make_hook(tmp_path)

    with patch.dict(sys.modules, {"imas_codex.standard_names.grammar_sync": None}):
        try:
            hook._sync_grammar_best_effort(tmp_path.parent)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"_sync_grammar_best_effort raised unexpectedly: {exc}")


def test_sync_grammar_skips_silently_on_graph_error(tmp_path: Path) -> None:
    """When the graph is unreachable, the hook must not raise."""
    hook = _make_hook(tmp_path)
    project_root = Path(__file__).parents[2]

    fake_sync = MagicMock(
        side_effect=RuntimeError("bolt://nonexistent:7687 — connection refused")
    )
    with patch(
        "imas_codex.standard_names.grammar_sync.sync_isn_grammar_to_graph",
        fake_sync,
    ):
        try:
            hook._sync_grammar_best_effort(project_root)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"hook raised on graph error: {exc}")


def test_sync_grammar_reports_on_success(tmp_path: Path) -> None:
    """When sync succeeds the hook should print a summary (no exception)."""
    hook = _make_hook(tmp_path)
    project_root = Path(__file__).parents[2]

    mock_report = MagicMock()
    mock_report.isn_version = "0.9.0"
    mock_report.segments = 10
    mock_report.templates = 5

    with patch(
        "imas_codex.standard_names.grammar_sync.sync_isn_grammar_to_graph",
        return_value=mock_report,
    ):
        try:
            hook._sync_grammar_best_effort(project_root)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"hook raised on success path: {exc}")
