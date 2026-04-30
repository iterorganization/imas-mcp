"""Tests for BudgetManager.hard_exhausted() predicate.

Ensures the watchdog uses committed spend (not pool depletion) to decide
global shutdown, preventing premature termination when large reservations
transiently drain the pool.
"""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.standard_names.budget import BudgetManager, LLMCostEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(**kwargs) -> LLMCostEvent:
    """Create a minimal LLMCostEvent for charge_event calls."""
    defaults = {
        "model": "test-model",
        "tokens_in": 100,
        "tokens_out": 50,
        "phase": "generate_name",
    }
    defaults.update(kwargs)
    return LLMCostEvent(**defaults)


# ---------------------------------------------------------------------------
# (a) Large lease drains pool but committed=0 → exhausted True, hard_exhausted False
# ---------------------------------------------------------------------------


class TestReservationDoesNotTriggerHardExhausted:
    """A large reservation should NOT cause hard_exhausted to fire."""

    def test_large_lease_exhausts_pool_not_hard(self) -> None:
        """Reserve entire budget → exhausted() True, hard_exhausted() False."""
        mgr = BudgetManager(total_budget=5.0)

        # Reserve the full budget — pool drops to 0
        lease = mgr.reserve(5.0, phase="generate_name")
        assert lease is not None

        # Pool is drained → exhausted() fires
        assert mgr.exhausted()

        # But nothing has been *spent* → hard_exhausted() must NOT fire
        assert not mgr.hard_exhausted()

    def test_partial_lease_exhausts_pool_not_hard(self) -> None:
        """Two leases totalling budget → exhausted True, hard_exhausted False."""
        mgr = BudgetManager(total_budget=2.0)

        lease1 = mgr.reserve(1.5, phase="generate_name")
        lease2 = mgr.reserve(0.5, phase="review_name")
        assert lease1 is not None
        assert lease2 is not None

        assert mgr.exhausted()
        assert not mgr.hard_exhausted()

        # Release both — pool restored, both predicates clear
        lease1.release_unused()
        lease2.release_unused()
        assert not mgr.exhausted()
        assert not mgr.hard_exhausted()


# ---------------------------------------------------------------------------
# (b) Committed spend reaches limit → hard_exhausted True
# ---------------------------------------------------------------------------


class TestSpendTriggersHardExhausted:
    """When actual spend reaches the budget, hard_exhausted must fire."""

    def test_full_spend_triggers_hard_exhausted(self) -> None:
        """Charge full budget → hard_exhausted() True."""
        mgr = BudgetManager(total_budget=1.0)

        lease = mgr.reserve(1.0, phase="generate_name")
        assert lease is not None

        event = _make_event()
        lease.charge_event(1.0, event)

        assert mgr.hard_exhausted()
        # exhausted() should also be True (pool is 0)
        assert mgr.exhausted()

    def test_partial_spend_not_hard_exhausted(self) -> None:
        """Spend less than budget → hard_exhausted False."""
        mgr = BudgetManager(total_budget=5.0)

        lease = mgr.reserve(2.0, phase="generate_name")
        assert lease is not None

        event = _make_event()
        lease.charge_event(1.5, event)

        assert not mgr.hard_exhausted()

    def test_incremental_spend_triggers_hard_exhausted(self) -> None:
        """Multiple charges summing to budget → hard_exhausted True."""
        mgr = BudgetManager(total_budget=1.0)

        lease = mgr.reserve(1.0, phase="generate_name")
        assert lease is not None

        event = _make_event()
        lease.charge_event(0.3, event)
        assert not mgr.hard_exhausted()

        lease.charge_event(0.3, event)
        assert not mgr.hard_exhausted()

        lease.charge_event(0.4, event)
        assert mgr.hard_exhausted()


# ---------------------------------------------------------------------------
# (c) Existing admission/compose retry behavior unchanged (uses exhausted)
# ---------------------------------------------------------------------------


class TestAdmissionStillUsesExhausted:
    """pool_admit uses exhausted() — verify it still gates on pool depletion."""

    def test_pool_admit_blocks_when_pool_exhausted(self) -> None:
        """Admission must be blocked when pool is empty (reservation-based)."""
        mgr = BudgetManager(total_budget=1.0)
        weights = {"generate_name": 1.0}

        # Reserve everything — pool empty
        lease = mgr.reserve(1.0, phase="generate_name")
        assert lease is not None
        assert mgr.exhausted()

        # Admission gate uses exhausted() → rejected
        assert not mgr.pool_admit("generate_name", weights, {"generate_name"})

    def test_pool_admit_allows_when_pool_available(self) -> None:
        """Admission is allowed when pool has funds, even if some is reserved."""
        mgr = BudgetManager(total_budget=5.0)
        weights = {"generate_name": 0.5, "review_name": 0.5}

        lease = mgr.reserve(2.0, phase="generate_name")
        assert lease is not None
        assert not mgr.exhausted()

        # Should still admit — pool has $3 left
        assert mgr.pool_admit("review_name", weights, {"generate_name", "review_name"})


# ---------------------------------------------------------------------------
# (d) Watchdog integration — hard_exhausted controls shutdown
# ---------------------------------------------------------------------------


class TestWatchdogUsesHardExhausted:
    """Watchdog must NOT fire on pool depletion alone (reservation spike)."""

    @pytest.mark.asyncio
    async def test_watchdog_skips_reservation_spike(self) -> None:
        """Large reservation drains pool → watchdog does NOT fire."""
        from imas_codex.standard_names.pools import PoolSpec, run_pools

        mgr = BudgetManager(total_budget=5.0)
        stop_event = asyncio.Event()

        # Reserve all 5.0 — pool empty, exhausted() True
        _lease = mgr.reserve(5.0, phase="generate_name")
        assert mgr.exhausted()
        assert not mgr.hard_exhausted()

        async def _idle_claim():
            await asyncio.sleep(0.02)
            return None

        async def _idle_process(batch):
            return 0

        spec = PoolSpec(name="generate_name", claim=_idle_claim, process=_idle_process)
        spec.health.pending_count = 1
        spec.backoff.base = 0.05
        spec.backoff.cap = 0.1
        spec.backoff.reset()

        # External stop after 0.5s — watchdog should NOT have fired before this
        async def _external_stop():
            await asyncio.sleep(0.5)
            stop_event.set()

        stopper = asyncio.create_task(_external_stop())

        try:
            await asyncio.wait_for(
                run_pools(
                    [spec],
                    mgr,
                    stop_event,
                    grace_period=0.5,
                    weights={"generate_name": 1.0},
                ),
                timeout=5.0,
            )
        finally:
            stopper.cancel()
            await asyncio.gather(stopper, return_exceptions=True)

        # Budget NOT hard-exhausted — watchdog should not have killed it
        assert not mgr.hard_exhausted()
        assert stop_event.is_set()  # external stop fired
