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
# Backward compat: review/budget.py re-export
# =====================================================================


def test_review_budget_reexport():
    """ReviewBudgetManager is re-exported as an alias for BudgetManager."""
    from imas_codex.standard_names.review.budget import ReviewBudgetManager

    mgr = ReviewBudgetManager(total_budget=1.0)
    assert isinstance(mgr, BudgetManager)
    lease = mgr.reserve(0.5)
    assert lease is not None
