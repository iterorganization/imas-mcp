"""Tests for the Phase 8 worker-pool orchestrator scaffolding.

Covers:

* Soft-fairness admission control (``BudgetManager.pool_admit``).
* Idle-pool weight reflow (idle pools forfeit share to active ones).
* Pool loop exits cleanly on ``stop_event``.
* Empty-claim exponential backoff.
* Graceful shutdown waits for in-flight batches.
* Per-pool health (wedge detection, error count).
"""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.standard_names.budget import BudgetManager
from imas_codex.standard_names.pools import (
    POOL_WEIGHTS,
    PoolHealth,
    PoolSpec,
    pool_loop,
    run_pools,
)

# =====================================================================
# BudgetManager.pool_admit
# =====================================================================


class TestPoolAdmit:
    def _mgr(self, total: float = 5.0) -> BudgetManager:
        return BudgetManager(total_budget=total)

    def test_admits_when_no_spend_yet(self) -> None:
        mgr = self._mgr()
        # Before anything has been spent, every active pool is admitted.
        assert mgr.pool_admit("generate", POOL_WEIGHTS, {"generate", "enrich"})

    def test_rejects_pool_not_in_active_set(self) -> None:
        mgr = self._mgr()
        # A pool whose queue is empty is not in active_pools and is
        # therefore not admitted (forfeits its share).
        assert not mgr.pool_admit("regen", POOL_WEIGHTS, {"generate"})

    def test_under_share_admitted(self) -> None:
        mgr = self._mgr()
        # Inject phase-spent so generate is well under its 0.30 share.
        mgr._phase_spent["generate"] = 0.10
        mgr._phase_spent["enrich"] = 0.90
        active = {"generate", "enrich"}
        # generate share = 0.10/1.00 = 0.10; effective_weight =
        # 0.30 / (0.30 + 0.25) ≈ 0.545.  Admit.
        assert mgr.pool_admit("generate", POOL_WEIGHTS, active)

    def test_over_share_rejected(self) -> None:
        mgr = self._mgr()
        # generate has consumed almost all spend — should be rejected
        # so other active pools can catch up.
        mgr._phase_spent["generate"] = 0.95
        mgr._phase_spent["enrich"] = 0.05
        active = {"generate", "enrich"}
        # generate share = 0.95/1.00 = 0.95 > 0.545 → reject.
        assert not mgr.pool_admit("generate", POOL_WEIGHTS, active)

    def test_idle_pool_share_reflows_to_active(self) -> None:
        """When some pools are idle, their weight reflows to active pools.

        With active = {generate} only, generate's effective weight is
        1.0 (it owns the whole pie), so it is always admitted regardless
        of share.
        """
        mgr = self._mgr()
        mgr._phase_spent["generate"] = 4.99
        # generate is the only active pool — full share forfeit.
        assert mgr.pool_admit("generate", POOL_WEIGHTS, {"generate"})

    def test_solo_active_pool_always_admitted(self) -> None:
        mgr = self._mgr()
        mgr._phase_spent["generate"] = 0.50
        # When generate is the only active pool, plan.md's "no other
        # pool is active" branch admits unconditionally regardless of
        # share or weight.
        assert mgr.pool_admit("generate", {"generate": 1.0}, {"generate"})

    def test_unknown_pool_rejected(self) -> None:
        mgr = self._mgr()
        assert not mgr.pool_admit("nonexistent", POOL_WEIGHTS, {"generate"})


# =====================================================================
# Pool loop semantics
# =====================================================================


class TestPoolLoop:
    @pytest.mark.asyncio
    async def test_exits_cleanly_on_stop_event(self) -> None:
        mgr = BudgetManager(total_budget=5.0)
        stop = asyncio.Event()

        async def claim() -> None:
            return None  # always empty

        async def process(batch: object) -> int:
            return 0

        spec = PoolSpec(name="generate", claim=claim, process=process)
        spec.health.pending_count = 1  # admit
        # Use a tiny backoff base for test speed.
        spec.backoff.base = 0.05
        spec.backoff.cap = 0.1
        spec.backoff.reset()

        async def active_pools_fn() -> set[str]:
            return {"generate"}

        task = asyncio.create_task(
            pool_loop(
                spec,
                mgr,
                stop,
                active_pools_fn=lambda: {"generate"},
                admission_poll=0.05,
            )
        )
        await asyncio.sleep(0.2)  # let it spin a few iterations
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
        # Pool exited; no errors, no in-flight.
        assert spec.health.error_count == 0
        assert spec.health.in_flight == 0

    @pytest.mark.asyncio
    async def test_processes_claimed_batch(self) -> None:
        mgr = BudgetManager(total_budget=5.0)
        stop = asyncio.Event()
        processed = []

        claims_remaining = [{"items": ["a", "b", "c"]}, None]

        async def claim() -> dict | None:
            return claims_remaining.pop(0) if claims_remaining else None

        async def process(batch: dict) -> int:
            processed.extend(batch["items"])
            return len(batch["items"])

        spec = PoolSpec(name="generate", claim=claim, process=process)
        spec.health.pending_count = 1
        spec.backoff.base = 0.05
        spec.backoff.cap = 0.1
        spec.backoff.reset()

        task = asyncio.create_task(
            pool_loop(
                spec,
                mgr,
                stop,
                active_pools_fn=lambda: {"generate"},
                admission_poll=0.05,
            )
        )
        await asyncio.sleep(0.3)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
        assert processed == ["a", "b", "c"]
        assert spec.health.last_progress_at > 0

    @pytest.mark.asyncio
    async def test_process_exception_does_not_crash_pool(self) -> None:
        mgr = BudgetManager(total_budget=5.0)
        stop = asyncio.Event()
        attempts = {"n": 0}

        async def claim() -> dict | None:
            attempts["n"] += 1
            return {"items": ["x"]} if attempts["n"] <= 2 else None

        async def process(batch: dict) -> int:
            raise RuntimeError("boom")

        spec = PoolSpec(name="generate", claim=claim, process=process)
        spec.health.pending_count = 1
        spec.backoff.base = 0.05
        spec.backoff.cap = 0.1
        spec.backoff.reset()

        task = asyncio.create_task(
            pool_loop(
                spec,
                mgr,
                stop,
                active_pools_fn=lambda: {"generate"},
                admission_poll=0.05,
            )
        )
        await asyncio.sleep(0.4)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
        # Two process exceptions raised, pool still running until stop.
        assert spec.health.error_count >= 2
        assert spec.health.last_error is not None
        assert "boom" in spec.health.last_error


# =====================================================================
# run_pools orchestration
# =====================================================================


class TestRunPools:
    @pytest.mark.asyncio
    async def test_concurrent_progress_under_load(self) -> None:
        """All non-empty pools make progress concurrently (acceptance #1)."""
        mgr = BudgetManager(total_budget=10.0)
        stop = asyncio.Event()
        progress = {"generate": 0, "enrich": 0, "review_names": 0}

        def make_spec(name: str) -> PoolSpec:
            async def claim() -> dict | None:
                return {"items": [name]}

            async def process(batch: dict) -> int:
                progress[name] += 1
                # Tiny work to allow other pools to run.
                await asyncio.sleep(0.01)
                return 1

            spec = PoolSpec(name=name, claim=claim, process=process)
            spec.health.pending_count = 100  # admit
            spec.backoff.base = 0.01
            spec.backoff.cap = 0.05
            spec.backoff.reset()
            return spec

        pools = [make_spec(n) for n in ["generate", "enrich", "review_names"]]

        async def driver() -> None:
            await asyncio.sleep(0.3)
            stop.set()

        await asyncio.gather(
            run_pools(pools, mgr, stop, grace_period=0.5),
            driver(),
        )
        # Each pool processed at least one batch.
        assert all(v >= 1 for v in progress.values()), progress

    @pytest.mark.asyncio
    async def test_idle_pool_does_not_spin(self) -> None:
        """Acceptance #2: idle pool backs off and does not hammer claims."""
        mgr = BudgetManager(total_budget=5.0)
        stop = asyncio.Event()
        claim_calls = {"n": 0}

        async def claim() -> None:
            claim_calls["n"] += 1
            return None  # always empty

        async def process(batch: object) -> int:
            return 0

        spec = PoolSpec(name="generate", claim=claim, process=process)
        spec.health.pending_count = 1  # active for admission test
        # Use realistic backoff so we measure ramp-up.
        spec.backoff.base = 0.1
        spec.backoff.cap = 0.5

        async def driver() -> None:
            await asyncio.sleep(1.0)
            stop.set()

        await asyncio.gather(
            run_pools([spec], mgr, stop, grace_period=0.5),
            driver(),
        )
        # Over 1s, with 0.1→0.2→0.4→0.5 cap backoff, expect ≤ 8 calls.
        # If backoff were broken we'd see ≫ 50.
        assert claim_calls["n"] < 15, claim_calls

    @pytest.mark.asyncio
    async def test_drain_pending_called_after_pools_exit(self) -> None:
        """Acceptance #3: BudgetManager.drain_pending runs before return."""
        mgr = BudgetManager(total_budget=5.0)
        stop = asyncio.Event()
        drained = {"flag": False}

        original_drain = mgr.drain_pending

        async def spy_drain(*a, **kw) -> bool:
            drained["flag"] = True
            return await original_drain(*a, **kw)

        mgr.drain_pending = spy_drain  # type: ignore[method-assign]

        async def claim() -> None:
            return None

        async def process(batch: object) -> int:
            return 0

        spec = PoolSpec(name="generate", claim=claim, process=process)
        spec.health.pending_count = 1
        spec.backoff.base = 0.01
        spec.backoff.cap = 0.05
        spec.backoff.reset()

        async def driver() -> None:
            await asyncio.sleep(0.1)
            stop.set()

        await asyncio.gather(
            run_pools([spec], mgr, stop, grace_period=0.5),
            driver(),
        )
        assert drained["flag"]


# =====================================================================
# PoolHealth
# =====================================================================


class TestPoolHealth:
    def test_wedge_detection(self) -> None:
        h = PoolHealth(pool="enrich", pending_count=10)
        h.last_progress_at = 100.0
        # Recent progress: not wedged.
        assert not h.is_wedged(poll_interval=3.0, now=101.0)
        # 7s without progress at 3s poll: wedged (>= 2× poll).
        assert h.is_wedged(poll_interval=3.0, now=107.0)

    def test_wedge_requires_pending_work(self) -> None:
        h = PoolHealth(pool="regen", pending_count=0)
        h.last_progress_at = 0.0  # very old
        # Empty queue: never wedged.
        assert not h.is_wedged(poll_interval=3.0, now=1000.0)
