"""Tests for the compose-worker budget gate.

Verifies that compose_worker respects the BudgetManager and doesn't
overshoot beyond concurrency × batch_cost.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.standard_names.budget import BudgetManager

# =====================================================================
# Minimal state stub
# =====================================================================


@dataclass
class _ComposeStats:
    total: int = 0
    processed: int = 0
    cost: float = 0.0
    errors: int = 0
    stream_queue: Any = field(default_factory=lambda: MagicMock())

    def record_batch(self, n: int) -> None:
        pass

    def freeze_rate(self) -> None:
        pass


@dataclass
class _Phase:
    done: bool = False

    def mark_done(self) -> None:
        self.done = True


# =====================================================================
# Budget gate unit tests (no workers, just BudgetManager with compose logic)
# =====================================================================


class TestComposeBudgetGate:
    """Test the budget-gate logic that compose_worker uses."""

    def test_reserve_before_batch(self):
        """Batches should reserve budget before doing LLM work."""
        mgr = BudgetManager(total_budget=0.30)

        # Simulate 10 batches, each needing $0.10
        completed = 0
        for _ in range(10):
            lease = mgr.reserve(0.10)
            if lease is None:
                break
            with lease:
                # Simulate LLM call costing exactly $0.08
                lease.charge(0.08)
            completed += 1

        assert completed == 3  # $0.30 / $0.10 = 3 reservations
        assert mgr.check_invariant()

    def test_compose_no_overshoot_beyond_reserved(self):
        """Total spend never exceeds original budget."""
        mgr = BudgetManager(total_budget=0.30)
        total_spend = 0.0

        for _ in range(10):
            lease = mgr.reserve(0.10)
            if lease is None:
                break
            with lease:
                cost = 0.08
                lease.charge(cost)
                total_spend += cost

        # Spend is bounded by what was reserved (3 × $0.08 = $0.24)
        assert total_spend <= 0.30 + 1e-9
        assert mgr.check_invariant()

    def test_compose_charge_matches_actual_cost(self):
        """Each lease tracks its actual charged amount."""
        mgr = BudgetManager(total_budget=1.0)
        costs = [0.05, 0.12, 0.03, 0.08]

        for cost in costs:
            lease = mgr.reserve(0.15)  # Reserve with headroom
            assert lease is not None
            with lease:
                lease.charge(cost)
                assert abs(lease.charged - cost) < 1e-9

        assert abs(mgr.spent - sum(costs)) < 1e-9
        assert mgr.check_invariant()


class TestComposeBudgetConcurrency:
    """Test the budget gate under concurrent batch execution."""

    def test_concurrent_batches_respect_budget(self):
        """With 5 concurrent batches and limited budget, early batches
        claim budget; later ones get None and skip."""

        async def _run():
            mgr = BudgetManager(total_budget=0.30)
            sem = asyncio.Semaphore(5)
            completed = 0
            skipped = 0

            async def _batch(batch_cost: float):
                nonlocal completed, skipped
                async with sem:
                    lease = mgr.reserve(batch_cost * 1.3)
                    if lease is None:
                        skipped += 1
                        return

                    try:
                        # Simulate LLM call
                        await asyncio.sleep(0.001)
                        lease.charge(batch_cost)
                        completed += 1
                    finally:
                        lease.release_unused()

            # 10 batches at $0.10 each
            tasks = [_batch(0.10) for _ in range(10)]
            await asyncio.gather(*tasks)

            # Budget = $0.30, reserve per batch = $0.13, so 2 fit ($0.26)
            # Third would need $0.13 but only $0.04 remains → None
            assert completed <= 3
            assert skipped >= 7
            assert mgr.check_invariant()

        asyncio.run(_run())

    def test_in_flight_batches_complete_after_budget_exhausted(self):
        """Batches that already reserved budget complete even if budget
        is exhausted for new batches."""

        async def _run():
            mgr = BudgetManager(total_budget=0.50)
            completed_ids = []

            async def _batch(batch_id: int):
                lease = mgr.reserve(0.20)
                if lease is None:
                    return
                try:
                    # Simulate some work
                    await asyncio.sleep(0.001)
                    lease.charge(0.15)
                    completed_ids.append(batch_id)
                finally:
                    lease.release_unused()

            tasks = [_batch(i) for i in range(10)]
            await asyncio.gather(*tasks)

            # $0.50 / $0.20 = 2 reservations; rest get None
            assert len(completed_ids) == 2
            assert mgr.spent < 0.50 + 1e-9
            assert mgr.check_invariant()

        asyncio.run(_run())

    def test_budget_gate_with_retries(self):
        """Budget reservation accounts for retry headroom."""

        async def _run():
            mgr = BudgetManager(total_budget=1.0)
            max_retries = 2
            per_item_cost = 0.01
            items_per_batch = 5
            # Reserve: items × cost × (retries+1) × 1.3 headroom
            reserve_amount = items_per_batch * per_item_cost * (max_retries + 1) * 1.3

            leases = []
            for _ in range(20):
                lease = mgr.reserve(reserve_amount)
                if lease is None:
                    break
                # Simulate: actual cost is just 1 attempt (no retries needed)
                lease.charge(items_per_batch * per_item_cost)
                leases.append(lease)

            # Clean up
            for lease in leases:
                lease.release_unused()

            # Reserve amount = 5 × 0.01 × 3 × 1.3 = 0.195
            # 1.0 / 0.195 ≈ 5 batches fit
            assert len(leases) == 5
            # But actual spend is only 5 × 0.05 = 0.25
            assert abs(mgr.spent - 0.25) < 1e-9
            assert mgr.check_invariant()

        asyncio.run(_run())
