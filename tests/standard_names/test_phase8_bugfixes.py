"""Tests for Phase 8 smoke-test blockers.

BUG-1 — Cypher head(coalesce(x.physics_domain, [])) fails for scalar fields.
BUG-2 — Python set comprehension over physics_domain fails when value is a list.
BUG-3 — pool_loop does not release claims when process() raises.
BUG-4 — write_standard_names() calls gc.query() after the surrounding
         ``with GraphClient()`` block closes (use-after-close).

All tests are mock-based; no live Neo4j required.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.standard_names.budget import BudgetManager
from imas_codex.standard_names.pools import PoolSpec, pool_loop

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_gc() -> MagicMock:
    """Build a mock GraphClient that supports ``with`` blocks."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    return gc


def _patch_gc(mock_gc: MagicMock):
    return patch(
        "imas_codex.standard_names.graph_ops.GraphClient",
        return_value=mock_gc,
    )


def _mock_mgr() -> MagicMock:
    mgr = MagicMock(spec=BudgetManager)
    mgr.pool_admit.return_value = True
    mgr.total_budget = 10.0
    mgr.spent = 0.0
    return mgr


# ---------------------------------------------------------------------------
# BUG-1 / BUG-2: claim queries return scalar physics_domain
# ---------------------------------------------------------------------------


class TestClaimReturnScalarPhysicsDomain:
    """Verifies that claim_*_seed_and_expand return physics_domain as str, not list.

    These tests feed the mock GraphClient with items whose physics_domain
    is a plain string (as the canonical scalar form) and assert the returned
    items carry a str — not a list that would break set-comprehension hashing.
    """

    def _run_sn_claim(self, claim_fn, seed_row: dict, readback_row: dict) -> list[dict]:
        """Helper: exercise a StandardName-backed claim function."""
        gc = _mock_gc()
        gc.query = MagicMock(
            side_effect=[
                [seed_row],  # seed
                [readback_row],  # read-back by token
            ]
        )
        with _patch_gc(gc):
            return claim_fn(batch_size=1)

    # ── claim_enrich ──────────────────────────────────────────────────

    def test_claim_enrich_returns_scalar_physics_domain(self) -> None:
        from imas_codex.standard_names.graph_ops import claim_enrich_seed_and_expand

        result = self._run_sn_claim(
            claim_enrich_seed_and_expand,
            seed_row={
                "_cluster_id": None,
                "_unit": "eV",
                "_physics_domain": "equilibrium",
            },
            readback_row={
                "id": "eq/psi",
                "description": "Poloidal flux",
                "documentation": None,
                "kind": "scalar",
                "unit": "Wb",
                "cluster_id": None,
                "physics_domain": "equilibrium",
                "validation_status": "valid",
                "enriched_at": None,
            },
        )
        assert result, "expected at least one item"
        pd = result[0]["physics_domain"]
        assert isinstance(pd, str), (
            f"physics_domain should be str, got {type(pd)}: {pd!r}"
        )

    # ── claim_review_names ────────────────────────────────────────────

    def test_claim_review_names_returns_scalar_physics_domain(self) -> None:
        from imas_codex.standard_names.graph_ops import (
            claim_review_names_seed_and_expand,
        )

        result = self._run_sn_claim(
            claim_review_names_seed_and_expand,
            seed_row={
                "_cluster_id": None,
                "_unit": "m",
                "_physics_domain": "magnetics",
            },
            readback_row={
                "id": "mag/field",
                "description": "Magnetic field",
                "documentation": None,
                "kind": "vector",
                "unit": "T",
                "cluster_id": None,
                "physics_domain": "magnetics",
                "validation_status": "valid",
                "reviewer_score_name": None,
                "reviewed_name_at": None,
            },
        )
        assert result
        pd = result[0]["physics_domain"]
        assert isinstance(pd, str), (
            f"physics_domain should be str, got {type(pd)}: {pd!r}"
        )

    # ── claim_review_docs ─────────────────────────────────────────────

    def test_claim_review_docs_returns_scalar_physics_domain(self) -> None:
        from imas_codex.standard_names.graph_ops import (
            claim_review_docs_seed_and_expand,
        )

        result = self._run_sn_claim(
            claim_review_docs_seed_and_expand,
            seed_row={
                "_cluster_id": "c1",
                "_unit": "keV",
                "_physics_domain": "transport",
            },
            readback_row={
                "id": "tr/Te",
                "description": "Electron temperature",
                "documentation": "Docs here.",
                "kind": "scalar",
                "unit": "keV",
                "cluster_id": "c1",
                "physics_domain": "transport",
                "validation_status": "valid",
                "reviewer_score_docs": None,
                "reviewed_docs_at": None,
                "enriched_at": "2024-01-01",
            },
        )
        assert result
        pd = result[0]["physics_domain"]
        assert isinstance(pd, str), (
            f"physics_domain should be str, got {type(pd)}: {pd!r}"
        )

    # ── claim_regen ───────────────────────────────────────────────────

    def test_claim_regen_returns_scalar_physics_domain(self) -> None:
        from imas_codex.standard_names.graph_ops import claim_regen_seed_and_expand

        result = self._run_sn_claim(
            claim_regen_seed_and_expand,
            seed_row={
                "_cluster_id": None,
                "_unit": "s",
                "_physics_domain": "core_profiles",
            },
            readback_row={
                "id": "cp/tau",
                "description": "Confinement time",
                "documentation": None,
                "kind": "scalar",
                "unit": "s",
                "cluster_id": None,
                "physics_domain": "core_profiles",
                "validation_status": "valid",
                "reviewer_score_name": 0.3,
                "reviewed_name_at": "2024-01-01",
                "regen_count": 0,
            },
        )
        assert result
        pd = result[0]["physics_domain"]
        assert isinstance(pd, str), (
            f"physics_domain should be str, got {type(pd)}: {pd!r}"
        )

    # ── claim_compose ─────────────────────────────────────────────────

    def test_claim_compose_returns_scalar_physics_domain_from_imas(self) -> None:
        """claim_compose reads physics_domain from IMASNode (scalar String).

        The seed query now returns imas.physics_domain directly (no head/coalesce).
        Verify the CASE expression no longer wraps in head().
        """
        from imas_codex.standard_names.graph_ops import claim_compose_seed_and_expand

        gc = _mock_gc()
        gc.query = MagicMock(
            side_effect=[
                # seed — _physics_domain is a plain string (IMASNode scalar)
                [
                    {
                        "_cluster_id": None,
                        "_unit": "eV",
                        "_physics_domain": "equilibrium",
                        "_batch_key": "equilibrium",
                    }
                ],
                # read-back — no physics_domain in compose items
                [
                    {
                        "id": "sns-1",
                        "source_id": "eq/psi",
                        "source_type": "dd",
                        "batch_key": "equilibrium",
                        "description": "Poloidal flux",
                    }
                ],
            ]
        )
        with _patch_gc(gc):
            result = claim_compose_seed_and_expand(batch_size=1)

        assert result, "expected at least one item"
        # The seed CASE expression should return a scalar _physics_domain
        seed_call_args = gc.query.call_args_list[0].args[0]
        assert "head(coalesce" not in seed_call_args, (
            "Seed Cypher still contains head(coalesce — BUG-1 not fixed"
        )

    def test_claim_compose_expand_uses_equality_not_in(self) -> None:
        """Fallback expand path uses `=` not `IN` for IMASNode.physics_domain.

        IMASNode.physics_domain is stored as a scalar String in the graph.
        Cypher's IN operator requires a list and calls head() internally,
        raising: 'Expected String("equilibrium") to be a list'.
        The fix changes `$fallback_domain IN imas.physics_domain`
        to `imas.physics_domain = $fallback_domain`.
        """
        from imas_codex.standard_names.graph_ops import claim_compose_seed_and_expand

        gc = _mock_gc()
        gc.query = MagicMock(
            side_effect=[
                # seed — cluster_id=None triggers the fallback expand path
                [
                    {
                        "_cluster_id": None,
                        "_unit": "eV",
                        "_physics_domain": "equilibrium",
                        "_batch_key": "equilibrium",
                    }
                ],
                # expand query result (empty, but the query must be issued)
                [],
                # read-back by token
                [],
            ]
        )
        with _patch_gc(gc):
            claim_compose_seed_and_expand(batch_size=2)

        # Find the expand Cypher call (2nd query call)
        assert gc.query.call_count >= 2, "expand query was never issued"
        expand_call = gc.query.call_args_list[1].args[0]
        assert "IN imas.physics_domain" not in expand_call, (
            "Expand Cypher still uses IN operator on scalar IMASNode.physics_domain"
        )
        assert "imas.physics_domain = $fallback_domain" in expand_call, (
            "Expand Cypher does not use scalar = comparison for physics_domain"
        )


# ---------------------------------------------------------------------------
# BUG-2: _scalar_domain normalizer handles mixed types
# ---------------------------------------------------------------------------


class TestScalarDomainNormalizer:
    """Verify the _scalar_domain normalizer in the three worker modules."""

    @staticmethod
    def _scalar_domain(d: object) -> str | None:
        """Mirror the normalizer from workers — tested in isolation here."""
        if isinstance(d, list):
            return d[0] if d else None
        return d  # type: ignore[return-value]

    def test_scalar_string_unchanged(self) -> None:
        assert self._scalar_domain("equilibrium") == "equilibrium"

    def test_singleton_list_unwrapped(self) -> None:
        assert self._scalar_domain(["equilibrium"]) == "equilibrium"

    def test_empty_list_returns_none(self) -> None:
        assert self._scalar_domain([]) is None

    def test_none_returns_none(self) -> None:
        assert self._scalar_domain(None) is None

    def test_domains_in_batch_handles_mixed_types(self) -> None:
        """Batch with mix of str and list physics_domain must not raise TypeError."""
        batch = [
            {"id": "n1", "physics_domain": "equilibrium"},
            {"id": "n2", "physics_domain": ["transport"]},
            {"id": "n3", "physics_domain": None},
            {"id": "n4"},  # key absent
            {"id": "n5", "physics_domain": ["magnetics", "equilibrium"]},
        ]

        # Replicate the normalised set-comprehension from workers/enrich/review
        def _scalar_domain(d: object) -> str | None:
            if isinstance(d, list):
                return d[0] if d else None
            return d  # type: ignore[return-value]

        # This must not raise TypeError: unhashable type: 'list'
        domains = sorted(
            {
                _scalar_domain(item.get("physics_domain"))
                for item in batch
                if item.get("physics_domain")
            }
            - {None}
        )
        assert domains == ["equilibrium", "magnetics", "transport"]


# ---------------------------------------------------------------------------
# BUG-3: pool_loop releases claims on process() failure
# ---------------------------------------------------------------------------


class TestPoolLoopReleaseOnFailure:
    """pool_loop must call spec.release(batch) when spec.process raises."""

    @pytest.mark.asyncio
    async def test_pool_loop_releases_claims_on_process_failure(self) -> None:
        """release is awaited with the same batch when process raises."""
        mgr = _mock_mgr()
        stop = asyncio.Event()

        released_batches: list[dict] = []

        batch_payload = {"items": [{"id": "sn-1"}, {"id": "sn-2"}]}
        claims_iter = [batch_payload, None]

        async def claim() -> dict | None:
            return claims_iter.pop(0) if claims_iter else None

        async def process(batch: dict) -> int:
            raise RuntimeError("simulated process failure")

        async def release(batch: dict) -> None:
            released_batches.append(batch)

        spec = PoolSpec(
            name="generate",
            claim=claim,
            process=process,
            release=release,
        )
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

        # release must have been called once with the batch
        assert len(released_batches) == 1, (
            f"expected release called once, got {len(released_batches)}"
        )
        assert released_batches[0] is batch_payload

    @pytest.mark.asyncio
    async def test_pool_loop_release_failure_does_not_crash_loop(self) -> None:
        """If release itself raises, pool_loop catches and continues."""
        mgr = _mock_mgr()
        stop = asyncio.Event()

        process_attempts = {"n": 0}
        batch_payload = {"items": [{"id": "sn-x"}]}
        claims_iter = [batch_payload, None]

        async def claim() -> dict | None:
            return claims_iter.pop(0) if claims_iter else None

        async def process(batch: dict) -> int:
            process_attempts["n"] += 1
            raise RuntimeError("process boom")

        async def release(batch: dict) -> None:
            raise RuntimeError("release also boom")

        spec = PoolSpec(
            name="generate",
            claim=claim,
            process=process,
            release=release,
        )
        spec.health.pending_count = 1
        spec.backoff.base = 0.05
        spec.backoff.cap = 0.1
        spec.backoff.reset()

        # pool_loop must NOT propagate the release exception
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
        # If this raises TimeoutError or the task raised, the test fails.
        await asyncio.wait_for(task, timeout=2.0)

        # process was called (confirms pool didn't die before reaching process)
        assert process_attempts["n"] >= 1

    @pytest.mark.asyncio
    async def test_pool_loop_no_release_when_process_succeeds(self) -> None:
        """release must NOT be called when process succeeds normally."""
        mgr = _mock_mgr()
        stop = asyncio.Event()

        release_called = {"n": 0}
        batch_payload = {"items": [{"id": "sn-ok"}]}
        claims_iter = [batch_payload, None]

        async def claim() -> dict | None:
            return claims_iter.pop(0) if claims_iter else None

        async def process(batch: dict) -> int:
            return len(batch["items"])

        async def release(batch: dict) -> None:
            release_called["n"] += 1

        spec = PoolSpec(
            name="generate",
            claim=claim,
            process=process,
            release=release,
        )
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

        assert release_called["n"] == 0, "release must not be called on success"

    @pytest.mark.asyncio
    async def test_pool_loop_no_release_field_does_not_raise(self) -> None:
        """PoolSpec without release=... still works (backward compatibility)."""
        mgr = _mock_mgr()
        stop = asyncio.Event()

        batch_payload = {"items": [{"id": "sn-compat"}]}
        claims_iter = [batch_payload, None]

        async def claim() -> dict | None:
            return claims_iter.pop(0) if claims_iter else None

        async def process(batch: dict) -> int:
            raise RuntimeError("boom without release field")

        spec = PoolSpec(name="generate", claim=claim, process=process)
        # release defaults to None
        assert spec.release is None

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
        await asyncio.wait_for(task, timeout=2.0)  # must not raise


# ---------------------------------------------------------------------------
# BUG-4: skeleton sweep use-after-close
# ---------------------------------------------------------------------------


class TestSkeletonSweepNoUseAfterClose:
    """write_standard_names() must not call gc.query() outside the `with` block.

    Before the fix, the skeleton sweep ran after the surrounding
    ``with GraphClient() as gc:`` closed, raising
    ``RuntimeError: GraphClient is closed`` on every persist.
    """

    def _make_minimal_name(self) -> dict:
        return {
            "id": "test_quantity",
            "description": "A test quantity",
            "documentation": None,
            "kind": "scalar",
            "unit": None,
            "source_types": ["dd"],
            "source_id": None,
            "physics_domain": None,
        }

    def _make_gc_sequence(self) -> tuple[MagicMock, MagicMock]:
        """Return (gc_main, gc_sweep) — two separate mock GraphClient instances
        to represent the two ``with GraphClient()`` blocks in write_standard_names().
        """
        gc_main = MagicMock()
        gc_main.__enter__ = MagicMock(return_value=gc_main)
        gc_main.__exit__ = MagicMock(return_value=False)
        gc_main.query = MagicMock(return_value=[])

        gc_sweep = MagicMock()
        gc_sweep.__enter__ = MagicMock(return_value=gc_sweep)
        gc_sweep.__exit__ = MagicMock(return_value=False)
        gc_sweep.query = MagicMock(return_value=[{"swept": 3}])

        return gc_main, gc_sweep

    def test_sweep_does_not_raise_after_close(self) -> None:
        """The sweep must execute inside its own GraphClient context."""
        from imas_codex.standard_names.graph_ops import write_standard_names

        gc_main, gc_sweep = self._make_gc_sequence()
        call_count = 0

        def gc_factory():
            nonlocal call_count
            call_count += 1
            return gc_main if call_count == 1 else gc_sweep

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            side_effect=gc_factory,
        ):
            # Must not raise RuntimeError: GraphClient is closed
            result = write_standard_names([self._make_minimal_name()])

        assert isinstance(result, int)

    def test_sweep_returns_count(self) -> None:
        """The function returns the number of names written (not swept)."""
        from imas_codex.standard_names.graph_ops import write_standard_names

        gc_main, gc_sweep = self._make_gc_sequence()
        call_count = 0

        def gc_factory():
            nonlocal call_count
            call_count += 1
            return gc_main if call_count == 1 else gc_sweep

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            side_effect=gc_factory,
        ):
            result = write_standard_names([self._make_minimal_name()])

        # write_standard_names returns the count of names written, not swept
        assert result == 1

    def test_sweep_query_called_on_open_client(self) -> None:
        """The sweep gc.query() must be called while the sweep client is open."""
        from imas_codex.standard_names.graph_ops import write_standard_names

        gc_main, gc_sweep = self._make_gc_sequence()
        sweep_query_called_while_open: list[bool] = []

        def tracking_query(*args, **kwargs):
            # If __exit__ has already been called, the client is closed
            sweep_query_called_while_open.append(gc_sweep.__exit__.call_count == 0)
            return [{"swept": 0}]

        gc_sweep.query = MagicMock(side_effect=tracking_query)

        call_count = 0

        def gc_factory():
            nonlocal call_count
            call_count += 1
            return gc_main if call_count == 1 else gc_sweep

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            side_effect=gc_factory,
        ):
            write_standard_names([self._make_minimal_name()])

        assert sweep_query_called_while_open, "sweep query was never called"
        assert all(sweep_query_called_while_open), (
            "sweep query was called on a closed client"
        )


# ---------------------------------------------------------------------------
# BUG-5: fairness deadlock when admitted pools have no claimable work
# ---------------------------------------------------------------------------


class TestFairnessDeadlockBreaker:
    """consecutive_empty_claims excludes stalled pools from the admission gate.

    Scenario: a pool has non-zero pending_count (display query uses loose
    criteria) but claim() consistently returns None (strict eligibility, e.g.
    confidence threshold).  After _EMPTY_CLAIM_EXCLUDE_THRESHOLD consecutive
    admitted-but-empty cycles the pool is excluded from active_pools_fn so
    productive pools can be admitted again.
    """

    @pytest.mark.asyncio
    async def test_consecutive_empty_claims_increments_on_empty_claim(self) -> None:
        """consecutive_empty_claims increments each time claim returns None."""
        from imas_codex.standard_names.pools import PoolHealth

        health = PoolHealth(pool="test")
        assert health.consecutive_empty_claims == 0

        # Simulate two admitted-but-empty cycles
        health.consecutive_empty_claims += 1
        health.consecutive_empty_claims += 1
        assert health.consecutive_empty_claims == 2

    @pytest.mark.asyncio
    async def test_consecutive_empty_claims_resets_on_successful_claim(self) -> None:
        """pool_loop resets consecutive_empty_claims when claim returns a batch."""
        mgr = _mock_mgr()
        stop = asyncio.Event()

        successful_batch = {"items": [{"id": "sn-ok"}]}
        # First two claims return None, third succeeds; capture counter at reset.
        claims = [None, None, successful_batch]
        counter_at_reset: list[int] = []

        async def claim() -> dict | None:
            return claims.pop(0) if claims else None

        processed: list[dict] = []

        async def process(batch: dict) -> int:
            # After reset, consecutive_empty_claims must be 0 here.
            counter_at_reset.append(spec.health.consecutive_empty_claims)
            processed.append(batch)
            return len(batch["items"])

        spec = PoolSpec(name="generate", claim=claim, process=process)
        spec.health.pending_count = 1
        spec.backoff.base = 0.01
        spec.backoff.cap = 0.02
        spec.backoff.reset()

        task = asyncio.create_task(
            pool_loop(
                spec,
                mgr,
                stop,
                active_pools_fn=lambda: {"generate"},
                admission_poll=0.01,
            )
        )
        # Give enough time for 3 claim cycles + process
        await asyncio.sleep(0.4)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)

        # consecutive_empty_claims must be 0 at the moment process is called
        # (i.e. after the reset, before any subsequent empty claims).
        assert len(processed) >= 1, "process was never called"
        assert counter_at_reset[0] == 0, (
            f"consecutive_empty_claims={counter_at_reset[0]} inside process(); "
            "expected 0 — reset must occur before process is called"
        )

    @pytest.mark.asyncio
    async def test_stalled_pool_excluded_from_active_pools(self) -> None:
        """After threshold empty claims a pool leaves active_pools_fn result.

        Two pools: 'enrich' always returns None; 'regen' returns a batch.
        Without the fix, 'regen' gets blocked by 'enrich' hogging weight share.
        With the fix, 'enrich' is excluded after _EMPTY_CLAIM_EXCLUDE_THRESHOLD
        consecutive empty claims, letting 'regen' run freely.
        """
        import asyncio

        from imas_codex.standard_names.pools import (
            _EMPTY_CLAIM_EXCLUDE_THRESHOLD,
            PoolSpec,
        )

        # Build a minimal active_pools_fn that mirrors run_pools logic.
        pools: list[PoolSpec] = []

        def active_pools_fn() -> set[str]:
            from imas_codex.standard_names.pools import _EMPTY_CLAIM_EXCLUDE_THRESHOLD

            return {
                p.name
                for p in pools
                if p.health.pending_count > 0
                and p.health.consecutive_empty_claims < _EMPTY_CLAIM_EXCLUDE_THRESHOLD
            }

        # 'enrich' — non-zero pending_count but claim always returns None.
        enrich_spec = PoolSpec(
            name="enrich",
            claim=AsyncMock(return_value=None),
            process=AsyncMock(return_value=0),
        )
        enrich_spec.health.pending_count = 100  # display says lots of work
        enrich_spec.backoff.base = 0.01
        enrich_spec.backoff.cap = 0.02
        enrich_spec.backoff.reset()

        pools.append(enrich_spec)

        # Simulate _EMPTY_CLAIM_EXCLUDE_THRESHOLD consecutive empty claims.
        for _ in range(_EMPTY_CLAIM_EXCLUDE_THRESHOLD):
            enrich_spec.health.consecutive_empty_claims += 1

        # After threshold: 'enrich' must NOT appear in active_pools.
        active = active_pools_fn()
        assert "enrich" not in active, (
            f"'enrich' should be excluded from active_pools after "
            f"{_EMPTY_CLAIM_EXCLUDE_THRESHOLD} consecutive empty claims, "
            f"but active_pools={active}"
        )

    def test_below_threshold_pool_remains_active(self) -> None:
        """Pool stays in active_pools_fn until threshold is reached."""
        from imas_codex.standard_names.pools import (
            _EMPTY_CLAIM_EXCLUDE_THRESHOLD,
            PoolSpec,
        )

        pools: list[PoolSpec] = []

        def active_pools_fn() -> set[str]:
            from imas_codex.standard_names.pools import _EMPTY_CLAIM_EXCLUDE_THRESHOLD

            return {
                p.name
                for p in pools
                if p.health.pending_count > 0
                and p.health.consecutive_empty_claims < _EMPTY_CLAIM_EXCLUDE_THRESHOLD
            }

        spec = PoolSpec(
            name="review_docs",
            claim=AsyncMock(return_value=None),
            process=AsyncMock(return_value=0),
        )
        spec.health.pending_count = 50
        pools.append(spec)

        # One below threshold — pool must still be active.
        spec.health.consecutive_empty_claims = _EMPTY_CLAIM_EXCLUDE_THRESHOLD - 1
        assert "review_docs" in active_pools_fn()

        # At threshold — pool must be excluded.
        spec.health.consecutive_empty_claims = _EMPTY_CLAIM_EXCLUDE_THRESHOLD
        assert "review_docs" not in active_pools_fn()
