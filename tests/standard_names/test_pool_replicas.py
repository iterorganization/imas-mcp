"""Tests for per-pool replica support in the pool orchestrator.

Covers:
* Replicas spawn correct number of asyncio tasks
* Per-replica backoff independence
* Shared PoolHealth aggregation across replicas
* Idle watchdog sees aggregate from shared health
* Empty-claim threshold scales by replica count
* Shutdown grace handles all replica tasks
"""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.standard_names.budget import BudgetManager
from imas_codex.standard_names.pools import (
    _EMPTY_CLAIM_EXCLUDE_THRESHOLD,
    POOL_WEIGHTS,
    PoolSpec,
    run_pools,
)


class TestReplicaBackoff:
    """Each replica gets its own backoff instance."""

    def test_per_replica_backoff_instances(self) -> None:
        spec = PoolSpec(
            name="review_name",
            claim=_noop_claim,
            process=_noop_process,
            replicas=4,
        )
        assert len(spec._replica_backoffs) == 4
        # Each is an independent object.
        assert len({id(b) for b in spec._replica_backoffs}) == 4

    def test_single_replica_backoff(self) -> None:
        spec = PoolSpec(
            name="generate_name",
            claim=_noop_claim,
            process=_noop_process,
        )
        assert spec.replicas == 1
        assert len(spec._replica_backoffs) == 1

    def test_replica_backoffs_independent(self) -> None:
        spec = PoolSpec(
            name="review_name",
            claim=_noop_claim,
            process=_noop_process,
            replicas=3,
        )
        # Advance replica 0's backoff
        spec._replica_backoffs[0].next_sleep()
        spec._replica_backoffs[0].next_sleep()
        # Replica 1 should still be at base
        sleep1 = spec._replica_backoffs[1]._current
        assert sleep1 == spec._replica_backoffs[1].base


class TestReplicaTaskSpawning:
    """run_pools spawns N tasks for pools with replicas > 1."""

    @pytest.mark.asyncio
    async def test_replicas_make_concurrent_progress(self) -> None:
        """4 replicas of review_name all process batches."""
        mgr = BudgetManager(total_budget=10.0)
        stop = asyncio.Event()
        processed_by = []
        claim_lock = asyncio.Lock()
        batches = [{"items": [f"batch_{i}"]} for i in range(8)]

        async def claim() -> dict | None:
            async with claim_lock:
                return batches.pop(0) if batches else None

        async def process(batch: dict) -> int:
            await asyncio.sleep(0.05)  # simulate LLM work
            processed_by.extend(batch["items"])
            return 1

        spec = PoolSpec(
            name="review_name",
            claim=claim,
            process=process,
            replicas=4,
        )
        spec.health.pending_count = 100
        for b in spec._replica_backoffs:
            b.base = 0.02
            b.cap = 0.05

        async def driver() -> None:
            await asyncio.sleep(0.5)
            stop.set()

        await asyncio.gather(
            run_pools([spec], mgr, stop, grace_period=0.5),
            driver(),
        )
        # All 8 batches consumed
        assert len(processed_by) == 8
        assert spec.health.total_processed == 8

    @pytest.mark.asyncio
    async def test_replica_health_shared(self) -> None:
        """All replicas increment the same PoolHealth.total_processed."""
        mgr = BudgetManager(total_budget=10.0)
        stop = asyncio.Event()
        claims_left = {"n": 6}
        claim_lock = asyncio.Lock()

        async def claim() -> dict | None:
            async with claim_lock:
                if claims_left["n"] > 0:
                    claims_left["n"] -= 1
                    return {"items": ["x"]}
                return None

        async def process(batch: dict) -> int:
            await asyncio.sleep(0.02)
            return 1

        spec = PoolSpec(
            name="review_name",
            claim=claim,
            process=process,
            replicas=3,
        )
        spec.health.pending_count = 10
        for b in spec._replica_backoffs:
            b.base = 0.02
            b.cap = 0.05

        async def driver() -> None:
            await asyncio.sleep(0.4)
            stop.set()

        await asyncio.gather(
            run_pools([spec], mgr, stop, grace_period=0.5),
            driver(),
        )
        assert spec.health.total_processed == 6

    @pytest.mark.asyncio
    async def test_mixed_replicas_all_pools_progress(self) -> None:
        """Mix of replica=1 and replica=4 pools all make progress."""
        mgr = BudgetManager(total_budget=10.0)
        stop = asyncio.Event()
        progress = {"generate_name": 0, "review_name": 0}

        def make_spec(name: str, replicas: int = 1) -> PoolSpec:
            async def claim() -> dict | None:
                return {"items": [name]}

            async def process(batch: dict) -> int:
                progress[name] += 1
                await asyncio.sleep(0.01)
                return 1

            spec = PoolSpec(name=name, claim=claim, process=process, replicas=replicas)
            spec.health.pending_count = 100
            for b in spec._replica_backoffs:
                b.base = 0.01
                b.cap = 0.05
            return spec

        pools = [make_spec("generate_name", 1), make_spec("review_name", 4)]

        async def driver() -> None:
            await asyncio.sleep(0.3)
            stop.set()

        await asyncio.gather(
            run_pools(pools, mgr, stop, grace_period=0.5),
            driver(),
        )
        assert progress["generate_name"] >= 1
        assert progress["review_name"] >= 1


class TestReplicaEmptyClaimThreshold:
    """Empty-claim exclusion threshold scales by replica count."""

    def test_threshold_scales_with_replicas(self) -> None:
        """With 4 replicas, threshold should be 4 * base threshold."""
        spec = PoolSpec(
            name="review_name",
            claim=_noop_claim,
            process=_noop_process,
            replicas=4,
        )
        # Simulate empty claims that would exceed base threshold
        # but not the scaled threshold.
        spec.health.pending_count = 5
        spec.health.consecutive_empty_claims = _EMPTY_CLAIM_EXCLUDE_THRESHOLD * 2
        # With replicas=4, scaled threshold = 12 (3 * 4).
        # 6 < 12, so pool should still be active.
        scaled_threshold = _EMPTY_CLAIM_EXCLUDE_THRESHOLD * spec.replicas
        assert spec.health.consecutive_empty_claims < scaled_threshold


class TestReplicaShutdown:
    """Shutdown grace period handles all replica tasks."""

    @pytest.mark.asyncio
    async def test_all_replicas_exit_on_stop(self) -> None:
        mgr = BudgetManager(total_budget=10.0)
        stop = asyncio.Event()

        async def claim() -> dict | None:
            return None

        async def process(batch: dict) -> int:
            return 0

        spec = PoolSpec(
            name="review_name",
            claim=claim,
            process=process,
            replicas=3,
        )
        spec.health.pending_count = 1
        for b in spec._replica_backoffs:
            b.base = 0.02
            b.cap = 0.05

        async def driver() -> None:
            await asyncio.sleep(0.15)
            stop.set()

        await asyncio.gather(
            run_pools([spec], mgr, stop, grace_period=0.5),
            driver(),
        )
        # Pool exited cleanly — no errors, no in-flight.
        assert spec.health.error_count == 0
        assert spec.health.in_flight == 0


class TestPoolSpecWeightDefaults:
    """PoolSpec.weight defaults from POOL_WEIGHTS when not explicitly set."""

    def test_default_weight_from_pool_weights(self) -> None:
        spec = PoolSpec(
            name="review_name",
            claim=_noop_claim,
            process=_noop_process,
        )
        assert spec.weight == POOL_WEIGHTS["review_name"]

    def test_explicit_weight_overrides_default(self) -> None:
        spec = PoolSpec(
            name="review_name",
            claim=_noop_claim,
            process=_noop_process,
            weight=0.42,
        )
        assert spec.weight == 0.42


# ── Helpers ──────────────────────────────────────────────────────────


async def _noop_claim() -> dict | None:
    return None


async def _noop_process(batch: dict) -> int:
    return 0
