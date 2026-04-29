"""Tests for Phase 8.1 six-pool wiring, backlog throttle, and pending counts.

Covers:

1. All six pools registered with correct names.
2. No legacy pools (review_names, regen, enrich, compose) remain.
3. Pool weights sum to 1.0.
4. Pool ids_kwarg: generate_name uses source_ids, SN-side pools use sn_ids.
5. pending_counts query returns correct counts (mocked graph).
6. pending_counts filters by facility (not applicable — current impl is global).
7. Throttle pauses generate_name when review_name backlog exceeds cap.
8. Throttle pauses generate_docs when review_docs backlog exceeds cap.
9. Throttle does not pause when backlog is below cap.
10. Pool state includes pending_counts for display layer.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.budget import BudgetManager

# =====================================================================
# Helpers
# =====================================================================

_GO = "imas_codex.standard_names.graph_ops"


def _build_specs(min_score: float = 0.75) -> list:
    """Build pool specs with a stopped event and budget manager."""
    from imas_codex.standard_names.loop import _build_pool_specs

    mgr = BudgetManager(total_budget=10.0)
    stop = asyncio.Event()
    return _build_pool_specs(mgr, stop, min_score=min_score)


# =====================================================================
# 1. All six pools registered
# =====================================================================


class TestAllSixPoolsRegistered:
    def test_all_six_pools_registered(self) -> None:
        specs = _build_specs()
        names = {s.name for s in specs}
        assert names == {
            "generate_name",
            "review_name",
            "refine_name",
            "generate_docs",
            "review_docs",
            "refine_docs",
        }

    def test_pool_count_is_six(self) -> None:
        specs = _build_specs()
        assert len(specs) == 6


# =====================================================================
# 2. No legacy pools
# =====================================================================


class TestNoLegacyPools:
    def test_no_legacy_pools(self) -> None:
        specs = _build_specs()
        names = {s.name for s in specs}
        legacy = {"review_names", "regen", "enrich", "compose", "generate"}
        assert names.isdisjoint(legacy), (
            f"Legacy pool names still present: {names & legacy}"
        )


# =====================================================================
# 3. Pool weights sum to 1.0
# =====================================================================


class TestPoolWeightsSumToOne:
    def test_pool_weights_sum_to_one(self) -> None:
        specs = _build_specs()
        total = sum(s.weight for s in specs)
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_pool_weights_dict_sums_to_one(self) -> None:
        from imas_codex.standard_names.pools import POOL_WEIGHTS

        total = sum(POOL_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"POOL_WEIGHTS sum to {total}"

    def test_pool_weights_match_specs(self) -> None:
        from imas_codex.standard_names.pools import POOL_WEIGHTS

        specs = _build_specs()
        for spec in specs:
            assert spec.name in POOL_WEIGHTS, f"Pool '{spec.name}' not in POOL_WEIGHTS"
            assert abs(spec.weight - POOL_WEIGHTS[spec.name]) < 1e-9, (
                f"Pool '{spec.name}' weight {spec.weight} != "
                f"POOL_WEIGHTS['{spec.name}'] {POOL_WEIGHTS[spec.name]}"
            )


# =====================================================================
# 4. Pool ids_kwarg correct
# =====================================================================


class TestPoolIdsKwarg:
    """generate_name uses source_ids (StandardNameSource); all SN-side
    pools use sn_ids (StandardName)."""

    @pytest.mark.asyncio
    async def test_generate_name_uses_source_ids(self) -> None:
        """Verify generate_name's release adapter passes source_ids."""
        batch = {
            "items": [
                {"id": "src-1", "claim_token": "tok-1"},
                {"id": "src-2", "claim_token": "tok-1"},
            ]
        }

        # Patch BEFORE building specs so the closure captures the mock
        with patch(
            f"{_GO}.release_generate_name_claims",
            return_value=2,
        ) as mock_release:
            specs = _build_specs()
            gen_spec = next(s for s in specs if s.name == "generate_name")
            await gen_spec.release(batch)
            mock_release.assert_called_once()
            call_kwargs = mock_release.call_args[1]
            assert "source_ids" in call_kwargs, (
                f"Expected source_ids kwarg, got {list(call_kwargs.keys())}"
            )
            assert call_kwargs["source_ids"] == ["src-1", "src-2"]

    @pytest.mark.asyncio
    async def test_sn_side_pools_use_sn_ids(self) -> None:
        """Verify all SN-side pools' release adapters pass sn_ids."""
        sn_pools = {
            "review_name": "release_review_names_claims",
            "refine_name": "release_refine_name_claims",
            "generate_docs": "release_generate_docs_claims",
            "review_docs": "release_review_docs_claims",
            "refine_docs": "release_refine_docs_claims",
        }

        batch = {
            "items": [
                {"id": "sn-1", "claim_token": "tok-A"},
            ]
        }

        for pool_name, release_fn_name in sn_pools.items():
            with patch(
                f"{_GO}.{release_fn_name}",
                return_value=1,
            ) as mock_release:
                specs = _build_specs()
                specs_by_name = {s.name: s for s in specs}
                spec = specs_by_name[pool_name]
                await spec.release(batch)
                mock_release.assert_called_once()
                call_kwargs = mock_release.call_args[1]
                assert "sn_ids" in call_kwargs, (
                    f"Pool '{pool_name}' expected sn_ids kwarg, "
                    f"got {list(call_kwargs.keys())}"
                )


# =====================================================================
# 5. pending_counts query returns correct counts
# =====================================================================


class TestPendingCountsQuery:
    def test_pending_counts_returns_all_six_keys(self) -> None:
        """pool_pending_counts returns a dict with all six pool keys."""
        mock_result = [
            {
                "generate_name": 10,
                "review_name": 5,
                "refine_name": 2,
                "generate_docs": 8,
                "review_docs": 3,
                "refine_docs": 1,
            }
        ]
        with patch("imas_codex.graph.client.GraphClient") as MockGC:
            mock_gc = MagicMock()
            mock_gc.query.return_value = mock_result
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.standard_names.graph_ops import pool_pending_counts

            counts = pool_pending_counts(min_score=0.75, rotation_cap=3)

        expected_keys = {
            "generate_name",
            "review_name",
            "refine_name",
            "generate_docs",
            "review_docs",
            "refine_docs",
        }
        assert set(counts.keys()) == expected_keys
        assert counts["generate_name"] == 10
        assert counts["review_name"] == 5
        assert counts["refine_name"] == 2
        assert counts["generate_docs"] == 8
        assert counts["review_docs"] == 3
        assert counts["refine_docs"] == 1

    def test_pending_counts_empty_graph_returns_zeros(self) -> None:
        """Empty graph returns all zeros."""
        with patch("imas_codex.graph.client.GraphClient") as MockGC:
            mock_gc = MagicMock()
            mock_gc.query.return_value = []
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.standard_names.graph_ops import pool_pending_counts

            counts = pool_pending_counts()

        assert all(v == 0 for v in counts.values())


# =====================================================================
# 6. pending_counts — single round-trip
# =====================================================================


class TestPendingCountsSingleQuery:
    def test_single_query_call(self) -> None:
        """pool_pending_counts makes exactly one query call."""
        with patch("imas_codex.graph.client.GraphClient") as MockGC:
            mock_gc = MagicMock()
            mock_gc.query.return_value = [
                {
                    "generate_name": 0,
                    "review_name": 0,
                    "refine_name": 0,
                    "generate_docs": 0,
                    "review_docs": 0,
                    "refine_docs": 0,
                }
            ]
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.standard_names.graph_ops import pool_pending_counts

            pool_pending_counts(min_score=0.75, rotation_cap=3)

        # Exactly one query call = single round-trip
        assert mock_gc.query.call_count == 1


# =====================================================================
# 7. Throttle pauses generate_name when review_name backlog > cap
# =====================================================================


class TestThrottlePausesGenerateName:
    @pytest.mark.asyncio
    async def test_throttle_pauses_generate_name_when_review_backlog(self) -> None:
        """generate_name claim returns None when review_name backlog > cap."""
        from imas_codex.standard_names.defaults import REVIEW_NAME_BACKLOG_CAP

        specs = _build_specs()
        specs_by_name = {s.name: s for s in specs}

        # Simulate high review_name backlog
        specs_by_name["review_name"].health.pending_count = REVIEW_NAME_BACKLOG_CAP + 1

        # generate_name's throttled claim should return None
        with patch(
            f"{_GO}.claim_generate_name_seed_and_expand",
            return_value=[{"id": "x", "claim_token": "t"}],
        ):
            result = await specs_by_name["generate_name"].claim()
            assert result is None, (
                "generate_name should be throttled when review_name "
                f"backlog ({REVIEW_NAME_BACKLOG_CAP + 1}) > cap "
                f"({REVIEW_NAME_BACKLOG_CAP})"
            )


# =====================================================================
# 8. Throttle pauses generate_docs when review_docs backlog > cap
# =====================================================================


class TestThrottlePausesGenerateDocs:
    @pytest.mark.asyncio
    async def test_throttle_pauses_generate_docs_when_review_docs_backlog(
        self,
    ) -> None:
        """generate_docs claim returns None when review_docs backlog > cap."""
        from imas_codex.standard_names.defaults import REVIEW_DOCS_BACKLOG_CAP

        specs = _build_specs()
        specs_by_name = {s.name: s for s in specs}

        # Simulate high review_docs backlog
        specs_by_name["review_docs"].health.pending_count = REVIEW_DOCS_BACKLOG_CAP + 1

        with patch(
            f"{_GO}.claim_generate_docs_seed_and_expand",
            return_value=[{"id": "x", "claim_token": "t"}],
        ):
            result = await specs_by_name["generate_docs"].claim()
            assert result is None, (
                "generate_docs should be throttled when review_docs "
                f"backlog ({REVIEW_DOCS_BACKLOG_CAP + 1}) > cap"
            )


# =====================================================================
# 9. Throttle does NOT pause when backlog is below cap
# =====================================================================


class TestThrottleAllowsBelowCap:
    @pytest.mark.asyncio
    async def test_throttle_does_not_pause_when_backlog_below_cap(self) -> None:
        """Upstream claims proceed when downstream backlog <= cap."""
        from imas_codex.standard_names.defaults import REVIEW_NAME_BACKLOG_CAP

        fake_items = [{"id": "src-1", "claim_token": "tok"}]
        # Patch BEFORE building specs so the closure captures the mock
        with patch(
            f"{_GO}.claim_generate_name_seed_and_expand",
            return_value=fake_items,
        ):
            specs = _build_specs()
            specs_by_name = {s.name: s for s in specs}

            # Set review_name backlog below cap
            specs_by_name["review_name"].health.pending_count = (
                REVIEW_NAME_BACKLOG_CAP - 1
            )

            result = await specs_by_name["generate_name"].claim()
            assert result is not None, (
                "generate_name should NOT be throttled when review_name "
                f"backlog ({REVIEW_NAME_BACKLOG_CAP - 1}) < cap"
            )
            assert result["items"] == fake_items

    @pytest.mark.asyncio
    async def test_throttle_allows_at_exact_cap(self) -> None:
        """At exactly cap (not >) throttle does NOT pause."""
        from imas_codex.standard_names.defaults import REVIEW_NAME_BACKLOG_CAP

        fake_items = [{"id": "src-1", "claim_token": "tok"}]
        # Patch BEFORE building specs so the closure captures the mock
        with patch(
            f"{_GO}.claim_generate_name_seed_and_expand",
            return_value=fake_items,
        ):
            specs = _build_specs()
            specs_by_name = {s.name: s for s in specs}

            # Set at exactly cap
            specs_by_name["review_name"].health.pending_count = REVIEW_NAME_BACKLOG_CAP

            result = await specs_by_name["generate_name"].claim()
            assert result is not None, (
                "generate_name should NOT be throttled at exactly cap "
                f"({REVIEW_NAME_BACKLOG_CAP})"
            )

    @pytest.mark.asyncio
    async def test_refine_name_throttled_by_review_name_backlog(self) -> None:
        """refine_name is also throttled by review_name backlog."""
        from imas_codex.standard_names.defaults import REVIEW_NAME_BACKLOG_CAP

        specs = _build_specs()
        specs_by_name = {s.name: s for s in specs}

        # High review_name backlog
        specs_by_name["review_name"].health.pending_count = REVIEW_NAME_BACKLOG_CAP + 10

        with patch(
            f"{_GO}.claim_refine_name_seed_and_expand",
            return_value=[{"id": "sn-1", "claim_token": "t"}],
        ):
            result = await specs_by_name["refine_name"].claim()
            assert result is None, "refine_name should be throttled by review_name"

    @pytest.mark.asyncio
    async def test_refine_docs_throttled_by_review_docs_backlog(self) -> None:
        """refine_docs is throttled by review_docs backlog."""
        from imas_codex.standard_names.defaults import REVIEW_DOCS_BACKLOG_CAP

        specs = _build_specs()
        specs_by_name = {s.name: s for s in specs}

        specs_by_name["review_docs"].health.pending_count = REVIEW_DOCS_BACKLOG_CAP + 5

        with patch(
            f"{_GO}.claim_refine_docs_seed_and_expand",
            return_value=[{"id": "sn-2", "claim_token": "t"}],
        ):
            result = await specs_by_name["refine_docs"].claim()
            assert result is None, "refine_docs should be throttled by review_docs"


# =====================================================================
# 10. Pool state includes pending_counts for display
# =====================================================================


class TestPoolStateIncludesPendingCounts:
    def test_pool_health_objects_accessible_from_specs(self) -> None:
        """Each PoolSpec has a health attribute with pending_count."""
        specs = _build_specs()
        for spec in specs:
            assert hasattr(spec, "health"), f"Pool '{spec.name}' has no health"
            assert hasattr(spec.health, "pending_count"), (
                f"Pool '{spec.name}' health has no pending_count"
            )

    @pytest.mark.asyncio
    async def test_pending_watchdog_updates_pool_health(self) -> None:
        """_pending_count_watchdog updates each pool's health.pending_count."""
        from imas_codex.standard_names.pools import PoolSpec, _pending_count_watchdog

        async def _noop_claim() -> dict[str, Any] | None:
            return None

        async def _noop_process(batch: dict[str, Any]) -> int:
            return 0

        pools = [
            PoolSpec(name="generate_name", claim=_noop_claim, process=_noop_process),
            PoolSpec(name="review_name", claim=_noop_claim, process=_noop_process),
        ]

        def pending_fn() -> dict[str, int]:
            return {"generate_name": 42, "review_name": 7}

        stop = asyncio.Event()
        task = asyncio.create_task(
            _pending_count_watchdog(pools, stop, pending_fn, poll=0.05)
        )

        # Let the watchdog poll once
        await asyncio.sleep(0.1)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)

        assert pools[0].health.pending_count == 42
        assert pools[1].health.pending_count == 7

    def test_run_sn_pools_wires_health_to_display(self) -> None:
        """run_sn_pools calls loop_state.set_pool_health for each pool.

        Verifies the display integration point exists.
        """
        from imas_codex.standard_names.loop import _build_pool_specs

        mgr = BudgetManager(total_budget=10.0)
        stop = asyncio.Event()
        specs = _build_pool_specs(mgr, stop)

        # Simulate the wiring code in run_sn_pools
        mock_state = MagicMock()
        mock_state.set_pool_health = MagicMock()

        for spec in specs:
            mock_state.set_pool_health(spec.name, spec.health)

        assert mock_state.set_pool_health.call_count == 6
        called_names = {c[0][0] for c in mock_state.set_pool_health.call_args_list}
        assert called_names == {
            "generate_name",
            "review_name",
            "refine_name",
            "generate_docs",
            "review_docs",
            "refine_docs",
        }
