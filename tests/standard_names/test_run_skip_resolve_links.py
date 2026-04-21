"""Test ``--skip-resolve-links`` cleanly bypasses resolve-links.

Symmetric to test_run_skip_reconcile: verifies that
``skip_resolve_links=True`` on TurnConfig skips the resolve-links
phase while all other phases run normally.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.turn import TurnConfig, run_turn


@pytest.mark.asyncio
class TestSkipResolveLinks:
    """--skip-resolve-links should bypass resolve-links; others run."""

    async def test_resolve_links_skipped(self):
        """resolve-links phase is marked as skipped."""
        cfg = TurnConfig(
            domain="equilibrium",
            dry_run=True,
            skip_resolve_links=True,
        )

        results = await run_turn(cfg)

        rl_result = next(r for r in results if r.name == "resolve-links")
        assert rl_result.skipped is True

    async def test_other_phases_not_skipped(self):
        """Other phases should NOT be skipped when only resolve-links is."""
        cfg = TurnConfig(
            domain="equilibrium",
            dry_run=True,
            skip_resolve_links=True,
        )

        results = await run_turn(cfg)

        for r in results:
            if r.name in ("reconcile", "generate", "enrich", "review"):
                assert r.skipped is False, f"{r.name} should not be skipped"

    async def test_resolve_links_runs_by_default(self):
        """resolve-links runs when skip_resolve_links=False."""
        cfg = TurnConfig(
            domain="equilibrium",
            dry_run=True,
            skip_resolve_links=False,
        )

        results = await run_turn(cfg)

        rl_result = next(r for r in results if r.name == "resolve-links")
        assert rl_result.skipped is False
