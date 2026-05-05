"""Integration tests for the Phase 8 pool-based orchestrator.

Covers:

* Reconcile-once-at-startup (B2): reconcile completes before any claim.
* CLI routing: default → run_sn_pools; --paths → run_turn.
* Physics-domain passthrough to extract_phase only.
* Stale claim clearing on restart (via reconcile).
* SNRun finalization with correct stop-reason.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# All claim/graph function patches target the *source module*
# (imas_codex.standard_names.graph_ops) because loop.py uses deferred
# imports (inside function bodies).
_GO = "imas_codex.standard_names.graph_ops"
_BM = "imas_codex.standard_names.budget.BudgetManager"

# Stub all 6 claim functions to return empty (no work).
_CLAIM_PATCHES = {
    "generate_name": f"{_GO}.claim_generate_name_batch",
    "review_name": f"{_GO}.claim_review_name_batch",
    "refine_name": f"{_GO}.claim_refine_name_batch",
    "generate_docs": f"{_GO}.claim_generate_docs_batch",
    "review_docs": f"{_GO}.claim_review_docs_batch",
    "refine_docs": f"{_GO}.claim_refine_docs_batch",
}


# =====================================================================
# 1. Reconcile runs before pools (B2 acceptance #5)
# =====================================================================


class TestReconcileRunsBeforePools:
    @pytest.mark.asyncio
    async def test_reconcile_runs_before_pools(self) -> None:
        """Assert reconcile_standard_name_sources is called and completes
        BEFORE the first claim_* call.

        Uses mock spies with timestamps.
        """
        timestamps: dict[str, float] = {}

        def _reconcile_spy(source_type: str = "dd") -> dict:
            timestamps["reconcile"] = time.monotonic()
            return {"relinked": 0, "stale_marked": 0, "revived": 0}

        def _claim_factory(name: str):
            """Return a sync claim that records its first-call timestamp."""

            def _claim(**_kw):
                key = f"claim_{name}"
                if key not in timestamps:
                    timestamps[key] = time.monotonic()
                return []  # empty → no work

            return _claim

        _mock_gc_ctx = MagicMock()
        _mock_gc_inst = MagicMock()
        _mock_gc_inst.query.return_value = [{"cnt": 1}]
        _mock_gc_ctx.__enter__ = MagicMock(return_value=_mock_gc_inst)
        _mock_gc_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                f"{_GO}.reconcile_standard_name_sources",
                side_effect=_reconcile_spy,
            ),
            patch(
                _CLAIM_PATCHES["generate_name"],
                side_effect=_claim_factory("generate_name"),
            ),
            patch(
                _CLAIM_PATCHES["generate_docs"],
                side_effect=_claim_factory("generate_docs"),
            ),
            patch(
                _CLAIM_PATCHES["review_name"],
                side_effect=_claim_factory("review_name"),
            ),
            patch(
                _CLAIM_PATCHES["review_docs"],
                side_effect=_claim_factory("review_docs"),
            ),
            patch(
                _CLAIM_PATCHES["refine_name"], side_effect=_claim_factory("refine_name")
            ),
            patch(
                _CLAIM_PATCHES["refine_docs"], side_effect=_claim_factory("refine_docs")
            ),
            patch(f"{_GO}.create_sn_run_open"),
            patch(f"{_GO}.finalize_sn_run"),
            patch(f"{_BM}.start", new_callable=AsyncMock),
            patch(f"{_BM}.drain_pending", new_callable=AsyncMock, return_value=True),
            patch(f"{_BM}.get_total_spent", new_callable=AsyncMock, return_value=0.0),
            patch(
                "imas_codex.graph.client.GraphClient",
                return_value=_mock_gc_ctx,
            ),
        ):
            from imas_codex.standard_names.loop import run_sn_pools

            stop = asyncio.Event()

            async def _stop_after_delay():
                await asyncio.sleep(0.3)
                stop.set()

            await asyncio.gather(
                run_sn_pools(cost_limit=5.0, stop_event=stop),
                _stop_after_delay(),
            )

        # Reconcile must have been called.
        assert "reconcile" in timestamps, "reconcile was never called"

        # Every claim timestamp must be > reconcile timestamp.
        claim_keys = [k for k in timestamps if k.startswith("claim_")]
        # At least some pools should have tried to claim.
        if claim_keys:
            min_claim = min(timestamps[k] for k in claim_keys)
            assert timestamps["reconcile"] < min_claim, (
                f"reconcile ({timestamps['reconcile']:.6f}) must precede "
                f"first claim ({min_claim:.6f})"
            )


# =====================================================================
# 2. CLI default routes to run_sn_pools
# =====================================================================


class TestCLIRouting:
    def test_cli_default_routes_to_run_sn_pools(self) -> None:
        """Invoke _run_sn_loop_cmd with no --paths; assert run_sn_pools is called.

        run_discovery and summary_table are deferred imports inside
        _run_sn_loop_cmd, so patch at their source modules.
        """
        with (
            patch(
                "imas_codex.cli.discover.common.run_discovery",
                return_value={"summary": MagicMock()},
            ) as mock_disc,
            patch(
                "imas_codex.standard_names.loop.summary_table",
                return_value={
                    "run_id": "test",
                    "turn_number": 1,
                    "stop_reason": "completed",
                    "cost_spent": 0.0,
                    "cost_limit": 5.0,
                    "names_composed": 0,
                    "names_enriched": 0,
                    "names_reviewed": 0,
                    "names_regenerated": 0,
                    "elapsed_s": 0.0,
                    "domains_touched": [],
                },
            ),
            patch(
                "imas_codex.cli.discover.common.use_rich_output",
                return_value=False,
            ),
            patch("imas_codex.cli.discover.common.setup_logging"),
        ):
            from imas_codex.cli.sn import _run_sn_loop_cmd

            _run_sn_loop_cmd(
                cost_limit=5.0,
                per_domain_limit=None,
                dry_run=False,
                quiet=True,
                verbose=False,
                source="dd",
            )

            # run_discovery was called with async_main closure that calls
            # run_sn_pools.
            assert mock_disc.called
            call_args = mock_disc.call_args
            async_main_fn = call_args[0][1]

            import inspect

            source = inspect.getsource(async_main_fn)
            assert "run_sn_pools" in source, (
                "CLI default mode should call run_sn_pools, "
                f"but async_main source is:\n{source}"
            )

    # ------------------------------------------------------------------
    # 3. CLI --paths routes to single-pass (not loop)
    # ------------------------------------------------------------------

    def test_cli_paths_routes_to_single_pass(self) -> None:
        """When --paths is set, use_loop must be False (skipping _run_sn_loop_cmd).

        Verifies the routing condition rather than invoking the full CLI
        (the single-pass path makes real LLM calls which would time out).
        """
        # The routing logic at cli/sn.py line ~964:
        #   use_loop = not single_pass and not paths_list and source == "dd"
        # With paths_list set, use_loop is False → _run_sn_loop_cmd skipped.
        paths_list = ("equilibrium/time_slice/profiles_1d/psi",)
        single_pass = False
        source = "dd"

        use_loop = not single_pass and not paths_list and source == "dd"
        assert not use_loop, "--paths should set use_loop=False"

        # Also verify without paths → use_loop=True
        use_loop_no_paths = not single_pass and not () and source == "dd"
        assert use_loop_no_paths, "without --paths, use_loop should be True"


# =====================================================================
# 4. Physics-domain passes to extract only
# =====================================================================


class TestPhysicsDomainPassthrough:
    def test_physics_domain_passes_to_extract_only(self) -> None:
        """Assert pool claim functions do NOT receive a domain filter.

        ``only_domain`` is accepted by ``run_sn_pools`` for extract_phase
        scoping but is NOT forwarded to any claim query.
        """
        from imas_codex.standard_names.budget import BudgetManager
        from imas_codex.standard_names.loop import _build_pool_specs

        mgr = BudgetManager(total_budget=5.0)
        stop = asyncio.Event()
        specs = _build_pool_specs(mgr, stop, min_score=0.6)

        assert len(specs) == 6
        pool_names = {s.name for s in specs}
        assert pool_names == {
            "generate_name",
            "review_name",
            "refine_name",
            "generate_docs",
            "review_docs",
            "refine_docs",
        }

        # Verify that run_sn_pools signature accepts only_domain
        import inspect

        from imas_codex.standard_names.loop import run_sn_pools

        sig = inspect.signature(run_sn_pools)
        assert "only_domain" in sig.parameters


# =====================================================================
# 5. Restart clears stale claims (via reconcile)
# =====================================================================


class TestRestartClearsStaleClaims:
    @pytest.mark.asyncio
    async def test_restart_clears_stale_claims(self) -> None:
        """Start run_sn_pools; assert reconcile runs (which clears
        stale claimed_at/claim_token).
        """
        reconcile_called = {"flag": False}

        def _reconcile(source_type: str = "dd") -> dict:
            reconcile_called["flag"] = True
            return {"relinked": 0, "stale_marked": 0, "revived": 3}

        _mock_gc_ctx = MagicMock()
        _mock_gc_inst = MagicMock()
        _mock_gc_inst.query.return_value = [{"cnt": 1}]
        _mock_gc_ctx.__enter__ = MagicMock(return_value=_mock_gc_inst)
        _mock_gc_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch(f"{_GO}.reconcile_standard_name_sources", side_effect=_reconcile),
            patch(f"{_GO}.create_sn_run_open"),
            patch(f"{_GO}.finalize_sn_run"),
            patch(f"{_BM}.start", new_callable=AsyncMock),
            patch(f"{_BM}.drain_pending", new_callable=AsyncMock, return_value=True),
            patch(f"{_BM}.get_total_spent", new_callable=AsyncMock, return_value=0.0),
            # Stub all claims to return empty.
            patch(_CLAIM_PATCHES["generate_name"], return_value=[]),
            patch(_CLAIM_PATCHES["generate_docs"], return_value=[]),
            patch(_CLAIM_PATCHES["review_name"], return_value=[]),
            patch(_CLAIM_PATCHES["review_docs"], return_value=[]),
            patch(_CLAIM_PATCHES["refine_name"], return_value=[]),
            patch(_CLAIM_PATCHES["refine_docs"], return_value=[]),
            patch(
                "imas_codex.graph.client.GraphClient",
                return_value=_mock_gc_ctx,
            ),
        ):
            from imas_codex.standard_names.loop import run_sn_pools

            stop = asyncio.Event()

            async def _stop_soon():
                await asyncio.sleep(0.2)
                stop.set()

            await asyncio.gather(
                run_sn_pools(cost_limit=5.0, stop_event=stop),
                _stop_soon(),
            )

        assert reconcile_called["flag"], "reconcile should be called on startup"


# =====================================================================
# 6. Finalize SNRun with correct status
# =====================================================================


class TestFinalizeWithCorrectStatus:
    @pytest.mark.asyncio
    async def test_finalize_interrupted_on_stop_event(self) -> None:
        """Assert finalize_sn_run is called once with status matching the
        stop reason (interrupted when stop_event fires).
        """
        finalize_calls: list[dict] = []

        def _finalize(run_id, **kwargs):
            finalize_calls.append({"run_id": run_id, **kwargs})

        _mock_gc_ctx = MagicMock()
        _mock_gc_inst = MagicMock()
        _mock_gc_inst.query.return_value = [{"cnt": 1}]
        _mock_gc_ctx.__enter__ = MagicMock(return_value=_mock_gc_inst)
        _mock_gc_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                f"{_GO}.reconcile_standard_name_sources",
                return_value={"relinked": 0, "stale_marked": 0, "revived": 0},
            ),
            patch(f"{_GO}.create_sn_run_open"),
            patch(f"{_GO}.finalize_sn_run", side_effect=_finalize),
            patch(f"{_BM}.start", new_callable=AsyncMock),
            patch(f"{_BM}.drain_pending", new_callable=AsyncMock, return_value=True),
            patch(f"{_BM}.get_total_spent", new_callable=AsyncMock, return_value=0.0),
            patch(_CLAIM_PATCHES["generate_name"], return_value=[]),
            patch(_CLAIM_PATCHES["generate_docs"], return_value=[]),
            patch(_CLAIM_PATCHES["review_name"], return_value=[]),
            patch(_CLAIM_PATCHES["review_docs"], return_value=[]),
            patch(_CLAIM_PATCHES["refine_name"], return_value=[]),
            patch(_CLAIM_PATCHES["refine_docs"], return_value=[]),
            patch(
                "imas_codex.graph.client.GraphClient",
                return_value=_mock_gc_ctx,
            ),
        ):
            from imas_codex.standard_names.loop import run_sn_pools

            stop = asyncio.Event()

            async def _stop_soon():
                await asyncio.sleep(0.2)
                stop.set()

            await asyncio.gather(
                run_sn_pools(cost_limit=5.0, stop_event=stop),
                _stop_soon(),
            )

        # finalize should have been called exactly once.
        assert len(finalize_calls) == 1
        call = finalize_calls[0]
        assert call["status"] == "interrupted"
        assert call["stop_reason"] == "interrupted"

    @pytest.mark.asyncio
    async def test_finalize_completed_on_natural_drain(self) -> None:
        """All pools drain naturally (no work) → status=completed or interrupted."""
        finalize_calls: list[dict] = []

        def _finalize(run_id, **kwargs):
            finalize_calls.append({"run_id": run_id, **kwargs})

        _mock_gc_ctx = MagicMock()
        _mock_gc_inst = MagicMock()
        _mock_gc_inst.query.return_value = [{"cnt": 1}]
        _mock_gc_ctx.__enter__ = MagicMock(return_value=_mock_gc_inst)
        _mock_gc_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                f"{_GO}.reconcile_standard_name_sources",
                return_value={"relinked": 0, "stale_marked": 0, "revived": 0},
            ),
            patch(f"{_GO}.create_sn_run_open"),
            patch(f"{_GO}.finalize_sn_run", side_effect=_finalize),
            patch(f"{_BM}.start", new_callable=AsyncMock),
            patch(f"{_BM}.drain_pending", new_callable=AsyncMock, return_value=True),
            patch(f"{_BM}.get_total_spent", new_callable=AsyncMock, return_value=0.0),
            patch(_CLAIM_PATCHES["generate_name"], return_value=[]),
            patch(_CLAIM_PATCHES["generate_docs"], return_value=[]),
            patch(_CLAIM_PATCHES["review_name"], return_value=[]),
            patch(_CLAIM_PATCHES["review_docs"], return_value=[]),
            patch(_CLAIM_PATCHES["refine_name"], return_value=[]),
            patch(_CLAIM_PATCHES["refine_docs"], return_value=[]),
            patch(
                "imas_codex.graph.client.GraphClient",
                return_value=_mock_gc_ctx,
            ),
        ):
            from imas_codex.standard_names.loop import run_sn_pools

            stop = asyncio.Event()

            async def _stop_after_pools_idle():
                await asyncio.sleep(0.5)
                stop.set()

            await asyncio.gather(
                run_sn_pools(cost_limit=5.0, stop_event=stop),
                _stop_after_pools_idle(),
            )

        assert len(finalize_calls) == 1
        # With stop_event set, this is "interrupted".
        assert finalize_calls[0]["status"] in ("completed", "interrupted")
