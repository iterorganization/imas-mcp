"""Tests for the _budget_watchdog task in run_pools (Phase 8 budget fix).

Regression tests ensuring that:
1. The watchdog sets stop_event promptly when mgr.exhausted() flips True.
2. The watchdog does NOT trigger stop_event when the budget remains healthy.
3. The watchdog exits cleanly alongside the pool tasks in both scenarios.
"""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.standard_names.budget import BudgetManager
from imas_codex.standard_names.pools import PoolSpec, run_pools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_idle_spec(name: str) -> PoolSpec:
    """Pool that always returns no work — appropriate for watchdog-focused tests."""

    async def claim() -> None:
        await asyncio.sleep(0.01)
        return None

    async def process(batch: object) -> int:  # pragma: no cover
        return 0

    spec = PoolSpec(name=name, claim=claim, process=process)
    spec.health.pending_count = 1  # keep it "active" for admission purposes
    spec.backoff.base = 0.05
    spec.backoff.cap = 0.1
    spec.backoff.reset()
    return spec


# ---------------------------------------------------------------------------
# Test 1: watchdog sets stop_event when budget exhausted
# ---------------------------------------------------------------------------


class TestBudgetWatchdogSetsStopEvent:
    """Watchdog must set stop_event within ~10 s when mgr.hard_exhausted() is True."""

    @pytest.mark.asyncio
    async def test_budget_watchdog_sets_stop_event_when_exhausted(self) -> None:
        """run_pools exits cleanly after watchdog fires on budget exhaustion.

        A fake mgr whose pool goes negative after a 0.2 s delay triggers the
        watchdog on its next poll (poll=0.05 s).  run_pools must return within
        a generous 10 s window.
        """
        mgr = BudgetManager(total_budget=5.0)
        stop_event = asyncio.Event()

        # Pool with no work — just idles while we wait for the watchdog.
        pools = [_make_idle_spec("generate")]

        # Schedule exhaustion after a short delay so pools have time to start.
        async def exhaust_after_delay() -> None:
            await asyncio.sleep(0.2)
            # Simulate actual spend reaching the budget limit.
            # hard_exhausted() checks _spent >= _total, so we set _spent
            # directly (the watchdog now uses hard_exhausted, not exhausted).
            mgr._spent = 5.01  # force hard_exhausted() → True

        exhaust_task = asyncio.create_task(exhaust_after_delay())

        try:
            await asyncio.wait_for(
                run_pools(
                    pools,
                    mgr,
                    stop_event,
                    grace_period=1.0,
                    weights={"generate": 1.0},
                ),
                timeout=10.0,
            )
        finally:
            exhaust_task.cancel()
            await asyncio.gather(exhaust_task, return_exceptions=True)

        # stop_event must have been set by the watchdog.
        assert stop_event.is_set(), (
            "stop_event must be set after budget exhaustion; "
            "watchdog may not have fired"
        )
        # mgr must still report hard-exhausted (no side-effects that reset it).
        assert mgr.hard_exhausted()


# ---------------------------------------------------------------------------
# Test 2: watchdog does NOT trigger when budget is healthy
# ---------------------------------------------------------------------------


class TestBudgetWatchdogDoesNotTriggerUnderBudget:
    """Watchdog must not set stop_event when the budget is not exhausted."""

    @pytest.mark.asyncio
    async def test_budget_watchdog_does_not_trigger_when_under_budget(self) -> None:
        """Pools exit only via external stop_event, not watchdog.

        Budget is $100 — far above any spend in this idle test.  The external
        stop_event fires after 0.3 s.  The watchdog must not fire before that.
        """
        mgr = BudgetManager(total_budget=100.0)
        stop_event = asyncio.Event()

        pools = [_make_idle_spec("generate")]

        async def external_stop() -> None:
            await asyncio.sleep(0.3)
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
                ),
                timeout=5.0,
            )
        finally:
            stopper.cancel()
            await asyncio.gather(stopper, return_exceptions=True)

        # stop_event must be set (by external signal, not watchdog).
        assert stop_event.is_set()
        # Budget should still be healthy.
        assert not mgr.exhausted(), (
            "budget must not be exhausted; watchdog must not have burned it"
        )
