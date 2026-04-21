"""Test ``sn run --only resolve-links`` invokes resolve-links and no other phase.

Verifies that when ``--only resolve-links`` is set, only the resolve-links
phase runs; all others (reconcile, generate, enrich, review, regen) are
skipped.  Also verifies scoping: when touched names are available from a
prior generate phase, resolve-links limits its sweep to those names.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from imas_codex.standard_names.turn import (
    PhaseResult,
    TurnConfig,
    run_turn,
    skip_flags_from_only,
)


class TestSkipFlagsFromOnly:
    """Unit tests for the --only → skip_* mapping helper."""

    def test_resolve_links_only(self):
        flags = skip_flags_from_only("resolve-links")
        assert flags["skip_generate"] is True
        assert flags["skip_enrich"] is True
        assert flags["skip_review"] is True
        assert flags["skip_regen"] is True
        # reconcile and resolve-links are no longer in skip_flags_from_only
        assert "skip_reconcile" not in flags
        assert "skip_resolve_links" not in flags

    def test_none_returns_empty(self):
        assert skip_flags_from_only(None) == {}

    def test_reconcile_only(self):
        flags = skip_flags_from_only("reconcile")
        assert flags["skip_generate"] is True
        assert "skip_reconcile" not in flags

    def test_extract_maps_to_generate(self):
        flags = skip_flags_from_only("extract")
        assert flags["skip_generate"] is False
        assert flags["skip_review"] is True


@pytest.mark.asyncio
class TestOnlyResolveLinks:
    """run_turn with resolve-links-only config."""

    async def test_only_resolve_links_runs_resolve_phase(self):
        """Only the resolve-links phase should execute; others skipped."""
        flags = skip_flags_from_only("resolve-links")
        cfg = TurnConfig(
            domain="equilibrium", dry_run=True, only="resolve-links", **flags
        )

        results = await run_turn(cfg)

        phase_names = [r.name for r in results]
        assert "resolve-links" in phase_names

        # All other phases must be skipped
        for r in results:
            if r.name != "resolve-links":
                assert r.skipped, f"Phase {r.name} should be skipped"

    async def test_resolve_links_scoped_to_touched_names(self):
        """When no names are touched (--only), resolve-links does global sweep."""
        flags = skip_flags_from_only("resolve-links")
        cfg = TurnConfig(domain="equilibrium", only="resolve-links", **flags)

        with patch(
            "imas_codex.standard_names.turn._fetch_unresolved_links",
            return_value=[],
        ) as mock_fetch:
            await run_turn(cfg)

        # Should have been called with None (global sweep) since no names touched
        mock_fetch.assert_called()
        call_args = mock_fetch.call_args
        assert call_args[0][0] is None  # name_ids is None → global sweep

    async def test_resolve_links_receives_touched_names_from_generate(self):
        """When generate produces names, resolve-links is scoped to them."""
        # Run generate + resolve-links only (no --only, so reconcile/resolve run)
        cfg = TurnConfig(
            domain="equilibrium",
            skip_enrich=True,
            skip_review=True,
            skip_regen=True,
        )

        gen_result = PhaseResult(
            name="generate",
            count=2,
            touched_names=["electron_temperature", "ion_temperature"],
        )

        with (
            patch(
                "imas_codex.standard_names.turn._run_reconcile_phase",
                new_callable=AsyncMock,
                return_value=PhaseResult(name="reconcile", count=0),
            ),
            patch(
                "imas_codex.standard_names.turn._run_generate_phase",
                new_callable=AsyncMock,
                return_value=gen_result,
            ),
            patch(
                "imas_codex.standard_names.turn._fetch_unresolved_links",
                return_value=[],
            ) as mock_fetch,
        ):
            await run_turn(cfg)

        # resolve-links should receive the touched names
        mock_fetch.assert_called()
        call_args = mock_fetch.call_args
        name_ids = call_args[0][0]
        assert name_ids is not None
        assert "electron_temperature" in name_ids
        assert "ion_temperature" in name_ids
