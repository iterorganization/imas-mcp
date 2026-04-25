"""Tests for the lease-style BudgetManager API.

Covers: reserve, charge, release, context manager, invariant,
concurrency, and edge cases.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from imas_codex.standard_names.budget import (
    BudgetExceeded,
    BudgetLease,
    BudgetManager,
)

# =====================================================================
# Basic reserve / pool deduction
# =====================================================================


def test_reserve_deducts_from_pool():
    """Reserving deducts from the available pool."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.3)
    assert lease is not None
    assert abs(mgr.remaining - 0.7) < 1e-9
    assert mgr.check_invariant()


def test_reserve_returns_none_if_insufficient():
    """Two reserves that exceed total budget — second returns None."""
    mgr = BudgetManager(total_budget=1.0)

    lease1 = mgr.reserve(0.6)
    assert lease1 is not None

    lease2 = mgr.reserve(0.6)
    assert lease2 is None

    # Pool only reduced by first reservation
    assert abs(mgr.remaining - 0.4) < 1e-9
    assert mgr.check_invariant()


def test_reserve_exact_amount():
    """Reserving exactly the remaining pool succeeds."""
    mgr = BudgetManager(total_budget=0.5)
    lease = mgr.reserve(0.5)
    assert lease is not None
    assert mgr.remaining < 1e-9
    assert mgr.exhausted()
    assert mgr.check_invariant()


# =====================================================================
# Charge
# =====================================================================


def test_charge_deducts_from_lease():
    """Charging updates both lease and manager spend."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(1.0)
    assert lease is not None

    lease.charge(0.3)
    assert abs(lease.charged - 0.3) < 1e-9
    assert abs(lease.remaining - 0.7) < 1e-9
    assert abs(mgr.spent - 0.3) < 1e-9
    assert mgr.check_invariant()


def test_charge_multiple_times():
    """Multiple charges accumulate correctly."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(1.0)
    assert lease is not None

    lease.charge(0.2)
    lease.charge(0.3)
    lease.charge(0.1)
    assert abs(lease.charged - 0.6) < 1e-9
    assert abs(mgr.spent - 0.6) < 1e-9
    assert mgr.check_invariant()


def test_charge_raises_on_overshoot():
    """Charging more than reserved raises BudgetExceeded."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(1.0)
    assert lease is not None

    with pytest.raises(BudgetExceeded, match="would exceed"):
        lease.charge(1.5)

    # Original state unchanged (charge was atomic — failed before recording)
    assert abs(lease.charged - 0.0) < 1e-9
    assert mgr.check_invariant()


def test_charge_raises_on_cumulative_overshoot():
    """Cumulative charges that exceed reserved raise BudgetExceeded."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.5)
    assert lease is not None

    lease.charge(0.3)
    with pytest.raises(BudgetExceeded):
        lease.charge(0.3)  # 0.3 + 0.3 = 0.6 > 0.5

    assert abs(lease.charged - 0.3) < 1e-9
    assert mgr.check_invariant()


def test_charge_negative_raises():
    """Negative charges are rejected."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(1.0)
    assert lease is not None

    with pytest.raises(ValueError, match="non-negative"):
        lease.charge(-0.1)


def test_charge_zero_is_noop():
    """Charging zero is valid and has no effect."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(1.0)
    assert lease is not None

    lease.charge(0.0)
    assert abs(lease.charged - 0.0) < 1e-9
    assert mgr.check_invariant()


# =====================================================================
# Release
# =====================================================================


def test_release_unused_returns_to_pool():
    """Unused portion is returned to the pool on release."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.5)
    assert lease is not None

    lease.charge(0.2)
    unused = lease.release_unused()

    assert abs(unused - 0.3) < 1e-9
    assert abs(mgr.remaining - 0.8) < 1e-9  # 0.5 pool + 0.3 released
    assert abs(mgr.spent - 0.2) < 1e-9
    assert mgr.check_invariant()


def test_release_is_idempotent():
    """Calling release_unused twice doesn't double-count."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.5)
    assert lease is not None

    lease.charge(0.2)
    first = lease.release_unused()
    second = lease.release_unused()

    assert abs(first - 0.3) < 1e-9
    assert abs(second - 0.0) < 1e-9
    assert abs(mgr.remaining - 0.8) < 1e-9
    assert mgr.check_invariant()


def test_release_no_charge():
    """Releasing without any charges returns the full reservation."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.4)
    assert lease is not None

    unused = lease.release_unused()
    assert abs(unused - 0.4) < 1e-9
    assert abs(mgr.remaining - 1.0) < 1e-9
    assert abs(mgr.spent - 0.0) < 1e-9
    assert mgr.check_invariant()


# =====================================================================
# Context manager
# =====================================================================


def test_context_manager_auto_release():
    """Using `with lease:` auto-releases on exit."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.5)
    assert lease is not None

    with lease:
        lease.charge(0.1)

    # Remaining 0.4 auto-released
    assert abs(mgr.remaining - 0.9) < 1e-9
    assert abs(mgr.spent - 0.1) < 1e-9
    assert mgr.check_invariant()


def test_context_manager_on_exception():
    """Context manager releases even when an exception occurs."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.5)
    assert lease is not None

    with pytest.raises(RuntimeError):
        with lease:
            lease.charge(0.2)
            raise RuntimeError("boom")

    # 0.3 unused returned
    assert abs(mgr.remaining - 0.8) < 1e-9
    assert abs(mgr.spent - 0.2) < 1e-9
    assert mgr.check_invariant()


# =====================================================================
# Invariant
# =====================================================================


def test_invariant_pool_plus_reserved_plus_spent_equals_total():
    """Property-style: random operations maintain the invariant."""
    import random

    mgr = BudgetManager(total_budget=10.0)
    leases = []
    rng = random.Random(42)

    for _ in range(50):
        op = rng.choice(["reserve", "charge", "release"])

        if op == "reserve":
            amount = rng.uniform(0.01, 2.0)
            lease = mgr.reserve(amount)
            if lease is not None:
                leases.append(lease)

        elif op == "charge" and leases:
            lease = rng.choice(leases)
            if not lease._released:
                amount = rng.uniform(0.001, 0.1)
                if lease.charged + amount <= lease.reserved:
                    lease.charge(amount)

        elif op == "release" and leases:
            lease = rng.choice(leases)
            lease.release_unused()

        assert mgr.check_invariant(), f"Invariant violated after op={op}"

    # Clean up all remaining leases
    for lease in leases:
        lease.release_unused()
    assert mgr.check_invariant()


# =====================================================================
# Concurrency
# =====================================================================


def test_concurrent_reserves():
    """Many threads racing for reserve() don't over-commit."""
    mgr = BudgetManager(total_budget=1.0)
    results: list[BudgetLease | None] = []
    lock = threading.Lock()

    def _reserve():
        lease = mgr.reserve(0.1)
        with lock:
            results.append(lease)

    threads = [threading.Thread(target=_reserve) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successful = [r for r in results if r is not None]
    assert len(successful) == 10  # 1.0 / 0.1 = 10 max
    assert all(r is None for r in results if r not in successful)
    assert mgr.check_invariant()


def test_concurrent_reserves_async():
    """Async tasks racing for reserve() don't over-commit."""

    async def _run():
        mgr = BudgetManager(total_budget=1.0)

        async def _reserve():
            return mgr.reserve(0.1)

        tasks = [_reserve() for _ in range(20)]
        results = await asyncio.gather(*tasks)

        successful = [r for r in results if r is not None]
        assert len(successful) == 10
        assert mgr.check_invariant()

    asyncio.run(_run())


# =====================================================================
# Multiple leases
# =====================================================================


def test_multiple_leases_independent():
    """Multiple leases from the same manager are independent."""
    mgr = BudgetManager(total_budget=1.0)

    lease1 = mgr.reserve(0.3)
    lease2 = mgr.reserve(0.3)
    assert lease1 is not None
    assert lease2 is not None
    assert abs(mgr.remaining - 0.4) < 1e-9

    lease1.charge(0.1)
    lease2.charge(0.2)

    assert abs(mgr.spent - 0.3) < 1e-9
    assert mgr.check_invariant()

    lease1.release_unused()  # returns 0.2
    assert abs(mgr.remaining - 0.6) < 1e-9

    lease2.release_unused()  # returns 0.1
    assert abs(mgr.remaining - 0.7) < 1e-9
    assert abs(mgr.spent - 0.3) < 1e-9
    assert mgr.check_invariant()


# =====================================================================
# Summary / exhausted
# =====================================================================


def test_exhausted_when_pool_drained():
    """Manager reports exhausted when pool is zero."""
    mgr = BudgetManager(total_budget=0.5)
    assert not mgr.exhausted()

    lease = mgr.reserve(0.5)
    assert lease is not None
    assert mgr.exhausted()

    lease.release_unused()
    assert not mgr.exhausted()


def test_summary_reflects_state():
    """Summary dict reflects current state."""
    mgr = BudgetManager(total_budget=2.0)
    lease = mgr.reserve(0.5)
    assert lease is not None
    lease.charge(0.3)

    s = mgr.summary
    assert s["total_budget"] == 2.0
    assert abs(s["remaining"] - 1.5) < 1e-9
    assert abs(s["total_spent"] - 0.3) < 1e-9
    assert s["active_reservations"] == 1
    assert s["batch_count"] == 1


def test_repr():
    """BudgetLease has a useful repr."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.5)
    assert lease is not None
    r = repr(lease)
    assert "BudgetLease" in r
    assert "0.5000" in r


# =====================================================================
# charge_or_extend — soft overrun handling
# =====================================================================


def test_charge_or_extend_within_reservation():
    """charge_or_extend behaves identically to charge when under reservation."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.5)
    assert lease is not None

    lease.charge_or_extend(0.3)
    assert abs(lease.charged - 0.3) < 1e-9
    assert abs(mgr.spent - 0.3) < 1e-9
    assert mgr.check_invariant()


def test_charge_or_extend_borrows_from_pool():
    """charge_or_extend extends the reservation when actual cost > reserved."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.10)  # Reserve a small amount; pool has 0.90 remaining

    # Actual cost is 0.15 — 0.05 over the 0.10 reservation
    lease.charge_or_extend(0.15)

    assert abs(lease.charged - 0.15) < 1e-9
    assert abs(mgr.spent - 0.15) < 1e-9
    # Pool should have decreased by the extension (0.05 borrowed)
    assert abs(mgr.remaining - 0.85) < 1e-9
    assert mgr.check_invariant()


def test_charge_or_extend_raises_when_pool_exhausted():
    """charge_or_extend raises BudgetExceeded when pool is also empty."""
    mgr = BudgetManager(total_budget=0.5)
    lease = mgr.reserve(0.5)  # Pool is now empty
    assert lease is not None

    with pytest.raises(BudgetExceeded):
        lease.charge_or_extend(0.6)  # Overrun; pool is 0

    # Invariant still holds; nothing was charged
    assert abs(lease.charged - 0.0) < 1e-9
    assert mgr.check_invariant()


def test_charge_or_extend_multiple_overruns():
    """Multiple charge_or_extend calls accumulate correctly."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.10)  # Pool: 0.90

    lease.charge_or_extend(0.08)  # Within reservation
    lease.charge_or_extend(0.09)  # 0.08+0.09=0.17 > 0.10 → borrows 0.07

    assert abs(lease.charged - 0.17) < 1e-9
    assert abs(mgr.spent - 0.17) < 1e-9
    assert mgr.check_invariant()


def test_charge_or_extend_invariant_maintained():
    """Invariant holds after a mix of reserve, charge_or_extend, and release."""
    mgr = BudgetManager(total_budget=2.0)

    lease1 = mgr.reserve(0.3)
    lease2 = mgr.reserve(0.2)
    assert lease1 is not None
    assert lease2 is not None

    lease1.charge_or_extend(0.4)  # Overrun by 0.1, borrowed from pool
    lease2.charge_or_extend(0.1)  # Within reservation

    assert mgr.check_invariant()

    lease1.release_unused()
    lease2.release_unused()

    assert mgr.check_invariant()
    assert abs(mgr.spent - 0.5) < 1e-9


def test_extend_reservation_returns_zero_when_pool_empty():
    """_extend_reservation returns 0 when pool is empty."""
    mgr = BudgetManager(total_budget=0.5)
    lease = mgr.reserve(0.5)
    assert lease is not None

    extended = mgr._extend_reservation(lease.lease_id, 0.2)
    assert extended == 0.0
    assert mgr.check_invariant()


# =====================================================================
# Backward compat: review/budget.py re-export
# =====================================================================


def test_review_budget_reexport():
    """ReviewBudgetManager is re-exported as an alias for BudgetManager."""
    from imas_codex.standard_names.review.budget import ReviewBudgetManager

    mgr = ReviewBudgetManager(total_budget=1.0)
    assert isinstance(mgr, BudgetManager)
    lease = mgr.reserve(0.5)
    assert lease is not None


# =====================================================================
# charge_soft: LLM already paid — spend MUST be recorded
# =====================================================================


def test_charge_soft_within_reservation_records_spend():
    """charge_soft within reservation records spend normally, returns 0 overspend."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.5)
    assert lease is not None

    overspend = lease.charge_soft(0.2)
    assert overspend == 0.0
    assert abs(lease.charged - 0.2) < 1e-9
    assert abs(mgr.spent - 0.2) < 1e-9


def test_charge_soft_extends_from_pool_when_possible():
    """charge_soft borrows from pool when reservation insufficient."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.3)
    assert lease is not None
    # Pool has 0.7 remaining; charge 0.5 overruns reservation by 0.2
    overspend = lease.charge_soft(0.5)
    assert overspend == 0.0  # extension covered it
    assert abs(mgr.spent - 0.5) < 1e-9


def test_charge_soft_records_spend_even_when_pool_exhausted():
    """When pool cannot cover overrun, spend is STILL recorded (no raise)."""
    mgr = BudgetManager(total_budget=0.3)
    lease = mgr.reserve(0.3)  # pool now 0
    assert lease is not None
    # Actual LLM cost was 0.5 — already paid, MUST be recorded
    overspend = lease.charge_soft(0.5)
    assert overspend > 0.0
    assert abs(overspend - 0.2) < 1e-9  # 0.5 charged - 0.3 reserved
    assert abs(mgr.spent - 0.5) < 1e-9  # spend recorded as reported


def test_charge_soft_never_raises_budget_exceeded():
    """charge_soft is total — never raises BudgetExceeded."""
    mgr = BudgetManager(total_budget=0.1)
    lease = mgr.reserve(0.1)
    assert lease is not None
    # Massive overspend — must not raise
    lease.charge_soft(10.0)
    assert abs(mgr.spent - 10.0) < 1e-9


def test_charge_soft_negative_amount_raises():
    """Negative charge amount is a programming error."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.5)
    assert lease is not None
    with pytest.raises(ValueError):
        lease.charge_soft(-0.1)


# =====================================================================
# Bug 3 regression: summary key is total_spent, not total_actual
# =====================================================================


def test_summary_key_is_total_spent_not_total_actual():
    """BudgetManager.summary uses 'total_spent', not the incorrect 'total_actual'.

    Regression guard for the KeyError: 'total_actual' bug in sn.py:3502.
    """
    mgr = BudgetManager(total_budget=2.0)
    lease = mgr.reserve(0.5)
    assert lease is not None
    lease.charge_or_extend(0.3)
    lease.release_unused()

    s = mgr.summary
    # The canonical key is 'total_spent'
    assert "total_spent" in s, "summary must contain 'total_spent'"
    assert "total_actual" not in s, "'total_actual' must NOT appear in summary"
    assert abs(s["total_spent"] - 0.3) < 1e-9
