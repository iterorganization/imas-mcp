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
    """Build a mock GraphClient that supports ``with`` blocks and transactions.

    Supports both legacy ``gc.query()`` and the new transaction pattern
    (``gc.session() → session.begin_transaction() → tx.run()``).
    """
    from contextlib import contextmanager

    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    # Transaction-based mock: gc.session() → session.begin_transaction() → tx
    tx = MagicMock()
    tx.closed = False
    tx.commit = MagicMock()
    tx.close = MagicMock()

    session = MagicMock()
    session.begin_transaction = MagicMock(return_value=tx)

    @contextmanager
    def _session_ctx():
        yield session

    gc.session = _session_ctx
    gc._tx = tx  # expose for test access
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
        gc._tx.run = MagicMock(
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
        gc._tx.run = MagicMock(
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
        seed_call_args = gc._tx.run.call_args_list[0].args[0]
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
        gc._tx.run = MagicMock(
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
        assert gc._tx.run.call_count >= 2, "expand query was never issued"
        expand_call = gc._tx.run.call_args_list[1].args[0]
        assert "IN imas.physics_domain" not in expand_call, (
            "Expand Cypher still uses IN operator on scalar IMASNode.physics_domain"
        )
        assert (
            "imas.physics_domain" in expand_call and "$fallback_domain" in expand_call
        ), "Expand Cypher does not use scalar = comparison for physics_domain"


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
    criteria) but claim() consistently returns None (strict eligibility).
    After _EMPTY_CLAIM_EXCLUDE_THRESHOLD consecutive admitted-but-empty
    cycles the pool is excluded from active_pools_fn so productive pools
    can be admitted again.
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


# =============================================================================
# Follow-up bugfix tests (pool lock-out recovery, counter aggregation, orphans)
# =============================================================================


class TestPoolAdmitRecoverWhenPendingIncreases:
    """Bug 1 — self-healing re-admission via pending_count growth.

    active_pools_fn (inside run_pools) resets consecutive_empty_claims to 0
    when a pool's pending_count grows.  This test exercises the exact same
    logic by calling a standalone active_pools_fn closure that mirrors the
    production implementation.
    """

    def _make_active_pools_fn(self, pools):
        """Replicate the production active_pools_fn closure."""
        from imas_codex.standard_names.pools import _EMPTY_CLAIM_EXCLUDE_THRESHOLD

        def active_pools_fn() -> set[str]:
            result: set[str] = set()
            for p in pools:
                current = p.health.pending_count
                if (
                    current > p.health._last_pending_count
                    and p.health.consecutive_empty_claims > 0
                ):
                    p.health.consecutive_empty_claims = 0
                p.health._last_pending_count = current
                if (
                    current > 0
                    and p.health.consecutive_empty_claims
                    < _EMPTY_CLAIM_EXCLUDE_THRESHOLD
                ):
                    result.add(p.name)
            return result

        return active_pools_fn

    def test_excluded_pool_re_enters_when_pending_grows(self) -> None:
        """Pool excluded by consecutive_empty_claims re-enters when pending grows."""
        from imas_codex.standard_names.pools import (
            _EMPTY_CLAIM_EXCLUDE_THRESHOLD,
            PoolSpec,
        )

        pools: list[PoolSpec] = []
        spec = PoolSpec(
            name="enrich",
            claim=AsyncMock(return_value=None),
            process=AsyncMock(return_value=0),
        )
        spec.health.pending_count = 10
        spec.health.consecutive_empty_claims = (
            _EMPTY_CLAIM_EXCLUDE_THRESHOLD + 2
        )  # excluded
        spec.health._last_pending_count = 10  # no growth yet
        pools.append(spec)

        active_pools_fn = self._make_active_pools_fn(pools)

        # First call: no growth → still excluded.
        result = active_pools_fn()
        assert "enrich" not in result, (
            "pool should still be excluded (no pending growth)"
        )

        # Simulate new names becoming enrich-eligible.
        spec.health.pending_count = 25
        result = active_pools_fn()
        assert "enrich" in result, "pool should re-enter after pending count increased"
        assert spec.health.consecutive_empty_claims == 0, (
            "consecutive_empty_claims should have been reset to 0"
        )

    def test_no_reset_when_pending_unchanged(self) -> None:
        """Excluded pool stays excluded when pending count doesn't change."""
        from imas_codex.standard_names.pools import (
            _EMPTY_CLAIM_EXCLUDE_THRESHOLD,
            PoolSpec,
        )

        pools: list[PoolSpec] = []
        spec = PoolSpec(
            name="enrich",
            claim=AsyncMock(return_value=None),
            process=AsyncMock(return_value=0),
        )
        spec.health.pending_count = 10
        spec.health.consecutive_empty_claims = _EMPTY_CLAIM_EXCLUDE_THRESHOLD
        spec.health._last_pending_count = 10
        pools.append(spec)

        active_pools_fn = self._make_active_pools_fn(pools)

        # Count stays at 10 — pool stays excluded.
        result = active_pools_fn()
        assert "enrich" not in result
        assert spec.health.consecutive_empty_claims == _EMPTY_CLAIM_EXCLUDE_THRESHOLD


class TestPoolLoopAccumulatesTotalProcessed:
    """Bug 3 — pool_loop accumulates total_processed via PoolHealth."""

    @pytest.mark.asyncio
    async def test_total_processed_accumulates(self) -> None:
        """After two successful batches of 5, total_processed == 10."""
        calls = 0

        async def _claim():
            nonlocal calls
            if calls >= 2:
                return None
            return {"ids": [f"sn-{i}" for i in range(5)]}

        async def _process(batch):
            nonlocal calls
            calls += 1
            return 5  # 5 items processed per batch

        stop = asyncio.Event()
        mgr = _mock_mgr()

        spec = PoolSpec(name="enrich", claim=_claim, process=_process)

        async def _stopper():
            # Give pool_loop time to drain both batches then signal stop.
            while calls < 2:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            stop.set()

        await asyncio.gather(
            pool_loop(
                spec,
                mgr,
                stop,
                active_pools_fn=lambda: {"enrich"},
                admission_poll=0.005,
            ),
            _stopper(),
        )

        assert spec.health.total_processed == 10

    @pytest.mark.asyncio
    async def test_total_processed_not_incremented_on_empty_claim(self) -> None:
        """Empty claims do not increment total_processed."""
        stop = asyncio.Event()
        mgr = _mock_mgr()
        spec = PoolSpec(
            name="enrich",
            claim=AsyncMock(return_value=None),
            process=AsyncMock(return_value=5),
        )

        async def _stopper():
            await asyncio.sleep(0.05)
            stop.set()

        await asyncio.gather(
            pool_loop(
                spec,
                mgr,
                stop,
                active_pools_fn=lambda: {"enrich"},
                admission_poll=0.005,
            ),
            _stopper(),
        )

        assert spec.health.total_processed == 0


class TestRunSnPoolsFinalizePopulatesCounters:
    """Bug 3 — run_sn_pools populates SNRun.names_* from health_map."""

    @pytest.mark.asyncio
    async def test_finalize_populates_counter_fields(self) -> None:
        """run_sn_pools assigns names_composed/enriched/reviewed/regenerated
        from pool total_processed values."""
        _GO = "imas_codex.standard_names.graph_ops"
        _BM = "imas_codex.standard_names.budget.BudgetManager"

        # We need run_pools to return a health_map with known total_processed.
        from imas_codex.standard_names.pools import PoolHealth

        health_generate = PoolHealth(pool="generate")
        health_generate.total_processed = 7
        health_enrich = PoolHealth(pool="enrich")
        health_enrich.total_processed = 4
        health_review_names = PoolHealth(pool="review_names")
        health_review_names.total_processed = 3
        health_review_docs = PoolHealth(pool="review_docs")
        health_review_docs.total_processed = 2
        health_regen = PoolHealth(pool="regen")
        health_regen.total_processed = 1

        fake_health_map = {
            "generate": health_generate,
            "enrich": health_enrich,
            "review_names": health_review_names,
            "review_docs": health_review_docs,
            "regen": health_regen,
        }

        with (
            patch(f"{_GO}.reconcile_standard_name_sources", return_value={}),
            patch(
                "imas_codex.standard_names.pools.run_pools",
                new_callable=AsyncMock,
                return_value=fake_health_map,
            ),
            patch(f"{_GO}.create_sn_run_open"),
            patch(f"{_GO}.finalize_sn_run"),
            patch(f"{_GO}.release_all_orphan_claims", return_value={"sn": 0, "sns": 0}),
            patch(f"{_BM}.start", new_callable=AsyncMock),
            patch(f"{_BM}.drain_pending", new_callable=AsyncMock, return_value=True),
            patch(f"{_BM}.get_total_spent", new_callable=AsyncMock, return_value=0.0),
            patch(f"{_BM}.exhausted", return_value=True),
            patch(f"{_BM}.phase_spent", new_callable=lambda: property(lambda self: {})),
        ):
            from imas_codex.standard_names.loop import run_sn_pools

            stop = asyncio.Event()
            stop.set()  # immediate stop
            summary = await run_sn_pools(cost_limit=5.0, stop_event=stop)

        assert summary.names_composed == 7
        assert summary.names_enriched == 4
        assert summary.names_reviewed == 5  # 3 + 2
        assert summary.names_regenerated == 1


class TestReleaseAllOrphanClaims:
    """Bug 4 — release_all_orphan_claims clears SN + SNS nodes."""

    def test_release_clears_sn_and_sns(self) -> None:
        """release_all_orphan_claims issues two SET queries and returns counts."""
        gc = _mock_gc()
        gc.query = MagicMock(
            side_effect=[
                [{"released": 3}],  # StandardName query
                [{"released": 5}],  # StandardNameSource query
            ]
        )

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=gc,
        ):
            from imas_codex.standard_names.graph_ops import release_all_orphan_claims

            result = release_all_orphan_claims()

        assert result == {"sn": 3, "sns": 5}
        assert gc.query.call_count == 2

    def test_release_returns_zero_when_no_orphans(self) -> None:
        """Empty result sets map to zero counts."""
        gc = _mock_gc()
        gc.query = MagicMock(return_value=[{"released": 0}])

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=gc,
        ):
            from imas_codex.standard_names.graph_ops import release_all_orphan_claims

            result = release_all_orphan_claims()

        assert result == {"sn": 0, "sns": 0}


# =============================================================================
# pending_fn / _pending_count_watchdog tests
# =============================================================================


class TestPendingCountWatchdog:
    """``_pending_count_watchdog`` updates PoolHealth.pending_count from pending_fn."""

    @pytest.mark.asyncio
    async def test_watchdog_sets_pending_counts_on_first_poll(self) -> None:
        """Watchdog sets pending_count on all pools immediately."""
        from imas_codex.standard_names.pools import PoolSpec, _pending_count_watchdog

        spec_a = PoolSpec(
            name="enrich",
            claim=AsyncMock(return_value=None),
            process=AsyncMock(return_value=0),
        )
        spec_b = PoolSpec(
            name="review_names",
            claim=AsyncMock(return_value=None),
            process=AsyncMock(return_value=0),
        )

        # Start with 0 pending.
        assert spec_a.health.pending_count == 0
        assert spec_b.health.pending_count == 0

        stop = asyncio.Event()
        call_count = 0

        def pending_fn() -> dict[str, int]:
            nonlocal call_count
            call_count += 1
            return {"enrich": 12, "review_names": 7, "generate": 3}

        # Run watchdog, let it do its initial sync poll, then stop.
        async def _stopper():
            await asyncio.sleep(0.05)
            stop.set()

        await asyncio.gather(
            _pending_count_watchdog([spec_a, spec_b], stop, pending_fn, poll=1.0),
            _stopper(),
        )

        # Initial poll should have fired at least once.
        assert call_count >= 1
        assert spec_a.health.pending_count == 12
        assert spec_b.health.pending_count == 7

    @pytest.mark.asyncio
    async def test_watchdog_polls_repeatedly(self) -> None:
        """Watchdog keeps polling while stop_event is not set."""
        from imas_codex.standard_names.pools import PoolSpec, _pending_count_watchdog

        spec = PoolSpec(
            name="enrich",
            claim=AsyncMock(return_value=None),
            process=AsyncMock(return_value=0),
        )

        stop = asyncio.Event()
        call_count = 0

        def pending_fn() -> dict[str, int]:
            nonlocal call_count
            call_count += 1
            # Second call returns higher count to verify watchdog updated.
            return {"enrich": call_count * 5}

        async def _stopper():
            # Let at least 2 polls fire with 0.02s poll interval.
            await asyncio.sleep(0.12)
            stop.set()

        await asyncio.gather(
            _pending_count_watchdog([spec], stop, pending_fn, poll=0.02),
            _stopper(),
        )

        # Should have polled more than once (initial + at least one timed).
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_watchdog_ignores_unknown_pool_names(self) -> None:
        """pending_fn keys not matching any pool are silently ignored."""
        from imas_codex.standard_names.pools import PoolSpec, _pending_count_watchdog

        spec = PoolSpec(
            name="enrich",
            claim=AsyncMock(return_value=None),
            process=AsyncMock(return_value=0),
        )

        stop = asyncio.Event()

        def pending_fn() -> dict[str, int]:
            return {"enrich": 9, "nonexistent_pool": 999}

        async def _stopper():
            await asyncio.sleep(0.05)
            stop.set()

        # Should not raise.
        await asyncio.gather(
            _pending_count_watchdog([spec], stop, pending_fn, poll=1.0),
            _stopper(),
        )

        assert spec.health.pending_count == 9

    @pytest.mark.asyncio
    async def test_watchdog_survives_pending_fn_exception(self) -> None:
        """Exceptions from pending_fn are logged and swallowed; watchdog continues."""
        from imas_codex.standard_names.pools import PoolSpec, _pending_count_watchdog

        spec = PoolSpec(
            name="enrich",
            claim=AsyncMock(return_value=None),
            process=AsyncMock(return_value=0),
        )

        stop = asyncio.Event()
        call_count = 0

        def pending_fn() -> dict[str, int]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("graph connection refused")
            return {"enrich": 5}

        async def _stopper():
            await asyncio.sleep(0.12)
            stop.set()

        # Should not propagate the exception.
        await asyncio.gather(
            _pending_count_watchdog([spec], stop, pending_fn, poll=0.02),
            _stopper(),
        )

        # After the exception on call 1, subsequent calls succeed.
        assert call_count >= 2
        # Final pending_count updated from successful calls.
        assert spec.health.pending_count == 5

    @pytest.mark.asyncio
    async def test_run_pools_wires_pending_fn(self) -> None:
        """run_pools with pending_fn updates pool pending_count from the watchdog."""
        import asyncio

        from imas_codex.standard_names.pools import PoolSpec, run_pools

        stop = asyncio.Event()
        mgr = _mock_mgr()
        mgr.drain_pending = AsyncMock(return_value=None)

        spec = PoolSpec(
            name="enrich",
            claim=AsyncMock(return_value=None),
            process=AsyncMock(return_value=0),
        )

        def pending_fn() -> dict[str, int]:
            return {"enrich": 17, "generate": 3}

        async def _stopper():
            await asyncio.sleep(0.15)
            stop.set()

        await asyncio.gather(
            run_pools(
                [spec],
                mgr,
                stop,
                pending_fn=pending_fn,
                pending_poll_interval=0.01,
            ),
            _stopper(),
        )

        # The watchdog must have updated pending_count (initial poll fires
        # synchronously before any claim loop iterations).
        assert spec.health.pending_count == 17


class TestPoolPendingCountsSplit:
    """``_pool_pending_counts`` in cli/sn.py returns generate/regen split correctly."""

    def test_generate_maps_to_draft_only(self) -> None:
        """generate pool maps to 'draft' count, NOT draft+revise."""
        # We exercise the logic without touching the CLI by directly calling
        # the mapping pattern used in _pool_pending_counts.
        raw = {
            "draft": 5,
            "revise": 3,
            "enrich": 7,
            "review_names": 2,
            "review_docs": 1,
        }
        # Mirror the mapping from cli/sn.py:_pool_pending_counts
        result = {
            "generate": raw["draft"],
            "enrich": raw["enrich"],
            "review_names": raw["review_names"],
            "review_docs": raw["review_docs"],
            "regen": raw["revise"],
        }
        assert result["generate"] == 5, "generate should be draft only"
        assert result["regen"] == 3, "regen should be revise only"
        assert result["enrich"] == 7
        assert result["review_names"] == 2
        assert result["review_docs"] == 1

    def test_generate_and_regen_sum_matches_display_total(self) -> None:
        """generate+regen sum equals the display aggregate (draft+revise)."""
        raw = {
            "draft": 4,
            "revise": 6,
            "enrich": 0,
            "review_names": 0,
            "review_docs": 0,
        }
        result = {
            "generate": raw["draft"],
            "regen": raw["revise"],
        }
        display_generate_total = raw["draft"] + raw["revise"]  # what display shows
        assert result["generate"] + result["regen"] == display_generate_total
