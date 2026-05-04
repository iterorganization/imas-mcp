"""Tests for the idle-exhaustion watchdog and budget detection.

These tests guard the fixes for the long-running idle-loop hang described
in ``plans/`` (SN pipeline workers don't exit when work is exhausted):

1. ``_budget_watchdog`` sets ``stop_event`` on hard exhaustion.
2. ``_budget_saturation_watchdog`` sets ``stop_event`` when all pools
   exceed consecutive reserve-failure threshold.
3. ``_idle_exhaustion_watchdog`` sets ``stop_event`` after a sustained
   window of zero pending counts and zero progress.
4. ``run_pools`` exits with the supplied ``idle_exhausted_event`` set
   when the idle watchdog fires.
"""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.standard_names.budget import BudgetManager
from imas_codex.standard_names.pools import PoolSpec, run_pools

# ---------------------------------------------------------------------------
# Budget: MIN_VIABLE_TURN and near_exhausted removed (Phase 2D)
# ---------------------------------------------------------------------------


class TestMinViableTurnRemoved:
    def test_min_viable_turn_constant_removed(self) -> None:
        """MIN_VIABLE_TURN must no longer exist in the budget module."""
        import imas_codex.standard_names.budget as budget_mod

        assert not hasattr(budget_mod, "MIN_VIABLE_TURN")

    def test_near_exhausted_method_removed(self) -> None:
        """near_exhausted must no longer exist on BudgetManager."""
        assert not hasattr(BudgetManager, "near_exhausted")


# ---------------------------------------------------------------------------
# Helpers — idle pool with controllable pending_count
# ---------------------------------------------------------------------------


def _make_idle_spec(name: str, *, pending: int = 0) -> PoolSpec:
    """Pool that always returns no work."""

    async def claim() -> None:
        await asyncio.sleep(0.01)
        return None

    async def process(batch: object) -> int:  # pragma: no cover
        return 0

    spec = PoolSpec(name=name, claim=claim, process=process)
    spec.health.pending_count = pending
    spec.backoff.base = 0.05
    spec.backoff.cap = 0.1
    spec.backoff.reset()
    return spec


# ---------------------------------------------------------------------------
# Idle-exhaustion watchdog (Fix 2)
# ---------------------------------------------------------------------------


class TestIdleExhaustionWatchdog:
    """Verify the watchdog exits run_pools when all pools are idle."""

    @pytest.mark.asyncio
    async def test_run_pools_exits_after_idle_threshold(self) -> None:
        """All pools have pending_count==0 and never make progress —
        run_pools must exit with idle_exhausted_event set."""
        mgr = BudgetManager(total_budget=5.0)
        stop_event = asyncio.Event()
        idle_exhausted = asyncio.Event()

        pools = [_make_idle_spec("generate", pending=0)]

        # Tight idle params so the test runs in <1s.
        await asyncio.wait_for(
            run_pools(
                pools,
                mgr,
                stop_event,
                grace_period=0.5,
                weights={"generate": 1.0},
                idle_exhausted_event=idle_exhausted,
                idle_exhaustion_poll=0.05,
                idle_exhaustion_polls=3,
            ),
            timeout=5.0,
        )

        assert idle_exhausted.is_set(), (
            "idle_exhausted_event must be set when all pools sit idle"
        )
        assert stop_event.is_set()

    @pytest.mark.asyncio
    async def test_idle_watchdog_does_not_fire_with_pending_work(self) -> None:
        """Pools with pending_count > 0 must NOT be considered idle —
        external stop_event is the only exit path here."""
        mgr = BudgetManager(total_budget=5.0)
        stop_event = asyncio.Event()
        idle_exhausted = asyncio.Event()

        pools = [_make_idle_spec("generate", pending=5)]

        async def external_stop() -> None:
            # Long enough that >3 poll cycles elapse — ensures the
            # watchdog had ample opportunity to misfire if its
            # pending_count gate were broken.
            await asyncio.sleep(0.4)
            stop_event.set()

        stopper = asyncio.create_task(external_stop())
        try:
            await asyncio.wait_for(
                run_pools(
                    pools,
                    mgr,
                    stop_event,
                    grace_period=0.5,
                    weights={"generate": 1.0},
                    idle_exhausted_event=idle_exhausted,
                    idle_exhaustion_poll=0.05,
                    idle_exhaustion_polls=3,
                ),
                timeout=5.0,
            )
        finally:
            stopper.cancel()
            await asyncio.gather(stopper, return_exceptions=True)

        assert not idle_exhausted.is_set(), (
            "idle watchdog must not fire while any pool has pending work"
        )

    @pytest.mark.asyncio
    async def test_idle_watchdog_resets_on_progress(self) -> None:
        """A single forward step in ``total_processed`` must reset the
        idle counter so a transient lull does not stop the run."""
        mgr = BudgetManager(total_budget=5.0)
        stop_event = asyncio.Event()
        idle_exhausted = asyncio.Event()

        spec = _make_idle_spec("generate", pending=0)
        pools = [spec]

        async def bump_progress() -> None:
            # Bump progress repeatedly so the watchdog never accumulates
            # 3 consecutive idle polls.
            for _ in range(8):
                await asyncio.sleep(0.05)
                spec.health.total_processed += 1
            stop_event.set()  # exit cleanly via external signal

        bumper = asyncio.create_task(bump_progress())
        try:
            await asyncio.wait_for(
                run_pools(
                    pools,
                    mgr,
                    stop_event,
                    grace_period=0.5,
                    weights={"generate": 1.0},
                    idle_exhausted_event=idle_exhausted,
                    idle_exhaustion_poll=0.05,
                    idle_exhaustion_polls=3,
                ),
                timeout=5.0,
            )
        finally:
            bumper.cancel()
            await asyncio.gather(bumper, return_exceptions=True)

        assert not idle_exhausted.is_set(), (
            "idle watchdog must reset on progress and never fire here"
        )


# ---------------------------------------------------------------------------
# Budget saturation watchdog (Phase C replacement for near-exhausted)
# ---------------------------------------------------------------------------


class TestBudgetSaturationWatchdog:
    @pytest.mark.asyncio
    async def test_watchdog_fires_on_budget_saturation(self) -> None:
        """When all pools exceed the consecutive reserve-failure threshold,
        the saturation watchdog must set stop_event."""
        mgr = BudgetManager(total_budget=3.0)
        stop_event = asyncio.Event()
        budget_saturated = asyncio.Event()

        pools = [_make_idle_spec("generate", pending=1)]

        async def saturate_budget() -> None:
            await asyncio.sleep(0.1)
            # The saturation watchdog checks pool names derived from the
            # actual PoolSpec list — here just "generate".
            mgr._consecutive_reserve_failures["generate"] = mgr.SATURATION_THRESHOLD

        saturator = asyncio.create_task(saturate_budget())
        try:
            await asyncio.wait_for(
                run_pools(
                    pools,
                    mgr,
                    stop_event,
                    grace_period=0.5,
                    weights={"generate": 1.0},
                    idle_exhausted_event=asyncio.Event(),
                    budget_saturated_event=budget_saturated,
                    # Disable idle watchdog by raising threshold high.
                    idle_exhaustion_poll=10.0,
                    idle_exhaustion_polls=1000,
                ),
                timeout=10.0,
            )
        finally:
            saturator.cancel()
            await asyncio.gather(saturator, return_exceptions=True)

        assert stop_event.is_set()
        assert budget_saturated.is_set()
        assert not mgr.hard_exhausted(), (
            "saturation watchdog should fire before hard exhaustion"
        )
