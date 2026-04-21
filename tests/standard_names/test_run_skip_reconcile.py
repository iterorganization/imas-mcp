"""Test ``--skip-reconcile`` cleanly bypasses reconcile; other phases run.

Verifies that setting ``skip_reconcile=True`` on TurnConfig skips the
reconcile phase while other phases execute normally.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from imas_codex.standard_names.turn import PhaseResult, TurnConfig, run_turn


@pytest.mark.asyncio
class TestSkipReconcile:
    """--skip-reconcile should bypass reconcile; other phases run."""

    async def test_reconcile_skipped(self):
        """reconcile phase marked as skipped when skip_reconcile=True."""
        cfg = TurnConfig(
            domain="equilibrium",
            dry_run=True,
            skip_reconcile=True,
        )

        results = await run_turn(cfg)

        reconcile_result = next(r for r in results if r.name == "reconcile")
        assert reconcile_result.skipped is True

    async def test_other_phases_run_normally(self):
        """Other phases should not be skipped when only skip_reconcile=True."""
        cfg = TurnConfig(
            domain="equilibrium",
            dry_run=True,
            skip_reconcile=True,
        )

        results = await run_turn(cfg)

        # generate, enrich, review should not be skipped in dry_run mode
        for r in results:
            if r.name in ("generate", "enrich", "review"):
                assert r.skipped is False, f"{r.name} should not be skipped"

    async def test_reconcile_runs_by_default(self):
        """reconcile phase runs by default (skip_reconcile=False)."""
        cfg = TurnConfig(
            domain="equilibrium",
            dry_run=True,
            skip_reconcile=False,
        )

        results = await run_turn(cfg)

        reconcile_result = next(r for r in results if r.name == "reconcile")
        # In dry_run, reconcile returns count=0 but is NOT skipped
        assert reconcile_result.skipped is False

    async def test_reconcile_calls_graph_ops(self):
        """When not skipped, reconcile calls reconcile_standard_name_sources."""
        cfg = TurnConfig(
            domain="equilibrium",
            skip_reconcile=False,
            # Skip everything else to keep the test fast
            skip_generate=True,
            skip_enrich=True,
            skip_resolve_links=True,
            skip_review=True,
            skip_regen=True,
            source="dd",
        )

        with patch(
            "imas_codex.standard_names.graph_ops.reconcile_standard_name_sources",
            return_value={"stale_marked": 1, "revived": 2, "relinked": 3},
        ) as mock_reconcile:
            results = await run_turn(cfg)

        mock_reconcile.assert_called_once_with("dd")
        reconcile_result = next(r for r in results if r.name == "reconcile")
        assert reconcile_result.count == 6  # sum of all values
        assert reconcile_result.skipped is False
