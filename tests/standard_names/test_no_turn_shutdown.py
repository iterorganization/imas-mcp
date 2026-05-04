"""Verify SN pipeline shutdown signals after turn-driven shutdown removal.

Phase C removed the ``near_exhausted()`` / ``MIN_VIABLE_TURN`` shutdown
gate and replaced it with a signal-driven ``budget_saturated`` watchdog.
These tests verify:

1. ``budget_saturated`` fires correctly when all pools exceed the
   consecutive reserve-failure threshold.
2. ``all_pools_idle`` fires correctly with finite work and ample budget.
3. ``MIN_VIABLE_TURN`` is no longer referenced in runtime shutdown paths.
4. ``BudgetManager.summary()`` includes the ``pending_writes`` metric.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from imas_codex.standard_names.budget import BudgetManager

# ---------------------------------------------------------------------------
# Test 1: budget_saturated shutdown
# ---------------------------------------------------------------------------


class TestBudgetSaturatedShutdown:
    def test_all_pools_budget_saturated_fires_at_threshold(self) -> None:
        """When every pool exceeds SATURATION_THRESHOLD consecutive reserve
        failures, ``all_pools_budget_saturated()`` returns True."""
        mgr = BudgetManager(total_budget=10.0)
        pool_names = (
            "generate_name",
            "review_name",
            "refine_name",
            "generate_docs",
            "review_docs",
            "refine_docs",
        )
        # Initially no failures — not saturated.
        assert not mgr.all_pools_budget_saturated(pool_names)

        # Set all but one to threshold — still not saturated.
        for name in pool_names[:-1]:
            mgr._consecutive_reserve_failures[name] = mgr.SATURATION_THRESHOLD
        assert not mgr.all_pools_budget_saturated(pool_names)

        # Set the last one — now saturated.
        mgr._consecutive_reserve_failures[pool_names[-1]] = mgr.SATURATION_THRESHOLD
        assert mgr.all_pools_budget_saturated(pool_names)

    def test_reserve_failure_increments_counter(self) -> None:
        """``reserve()`` returning None increments the failure counter."""
        mgr = BudgetManager(total_budget=0.01)  # tiny budget
        # Attempt a reserve that exceeds the pool.
        result = mgr.reserve(1.0, phase="generate_name")
        assert result is None
        assert mgr._consecutive_reserve_failures["generate_name"] == 1

        # Second failure.
        mgr.reserve(1.0, phase="generate_name")
        assert mgr._consecutive_reserve_failures["generate_name"] == 2

    def test_successful_reserve_resets_counter(self) -> None:
        """A successful ``reserve()`` resets the failure counter to 0."""
        mgr = BudgetManager(total_budget=10.0)
        # Fail a few times first by exhausting the pool temporarily.
        mgr._consecutive_reserve_failures["generate_name"] = 5
        # Now a successful reserve.
        lease = mgr.reserve(1.0, phase="generate_name")
        assert lease is not None
        assert mgr._consecutive_reserve_failures["generate_name"] == 0


# ---------------------------------------------------------------------------
# Test 2: all_pools_idle shutdown with finite work
# ---------------------------------------------------------------------------


class TestAllPoolsIdleShutdown:
    @pytest.mark.asyncio
    async def test_idle_shutdown_with_finite_work(self) -> None:
        """With ample budget and idle pools, shutdown must be via idle
        watchdog — not budget_saturated."""
        from imas_codex.standard_names.pools import PoolSpec, run_pools

        async def empty_claim() -> None:
            await asyncio.sleep(0.01)
            return None

        async def noop_process(batch: object) -> int:  # pragma: no cover
            return 0

        spec = PoolSpec(name="generate_name", claim=empty_claim, process=noop_process)
        spec.health.pending_count = 0
        spec.backoff.base = 0.05
        spec.backoff.cap = 0.1
        spec.backoff.reset()

        mgr = BudgetManager(total_budget=1000.0)  # effectively unlimited
        stop_event = asyncio.Event()
        idle_exhausted = asyncio.Event()
        budget_saturated = asyncio.Event()

        await asyncio.wait_for(
            run_pools(
                [spec],
                mgr,
                stop_event,
                grace_period=0.5,
                weights={"generate_name": 1.0},
                idle_exhausted_event=idle_exhausted,
                budget_saturated_event=budget_saturated,
                idle_exhaustion_poll=0.05,
                idle_exhaustion_polls=3,
            ),
            timeout=5.0,
        )

        assert idle_exhausted.is_set(), (
            "Should shut down via idle exhaustion, not budget saturation"
        )
        assert not budget_saturated.is_set(), (
            "Budget saturation must not fire with ample budget"
        )


# ---------------------------------------------------------------------------
# Test 3: no MIN_VIABLE_TURN in shutdown code paths
# ---------------------------------------------------------------------------

# Directories containing runtime shutdown logic.
_SN_DIR = (
    Path(__file__).resolve().parent.parent.parent / "imas_codex" / "standard_names"
)
_SHUTDOWN_FILES = ["pools.py", "loop.py", "budget.py"]


class TestNoMinViableTurnInShutdown:
    def test_min_viable_turn_fully_removed(self) -> None:
        """MIN_VIABLE_TURN must not exist anywhere in runtime code."""
        for fname in _SHUTDOWN_FILES:
            fpath = _SN_DIR / fname
            if not fpath.exists():
                continue
            tree = ast.parse(fpath.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == "MIN_VIABLE_TURN":
                    pytest.fail(
                        f"MIN_VIABLE_TURN referenced in {fname} "
                        f"(line {node.lineno}) — fully removed in Phase 2D"
                    )

    def test_near_exhausted_removed(self) -> None:
        """near_exhausted method must be removed from BudgetManager."""
        assert not hasattr(BudgetManager, "near_exhausted")


# ---------------------------------------------------------------------------
# Test 4: pending_writes metric in BudgetManager.summary()
# ---------------------------------------------------------------------------


class TestPendingWritesMetric:
    def test_pending_writes_in_summary(self) -> None:
        """``BudgetManager.summary`` must include ``pending_writes``."""
        mgr = BudgetManager(total_budget=5.0)
        s = mgr.summary
        assert "pending_writes" in s
        assert isinstance(s["pending_writes"], int)
        assert s["pending_writes"] == 0

    @pytest.mark.asyncio
    async def test_pending_writes_reflects_queue_depth(self) -> None:
        """After enqueuing writes, ``pending_writes`` must reflect depth."""
        mgr = BudgetManager(total_budget=5.0, run_id="test-run")
        # Don't start the writer — enqueued items stay in queue.
        from imas_codex.standard_names.budget import LLMCostEvent

        event = LLMCostEvent(
            model="test",
            tokens_in=10,
            tokens_out=5,
            phase="generate_name",
        )
        mgr._enqueue_write(0.01, event, 0.0)
        s = mgr.summary
        assert s["pending_writes"] >= 1
