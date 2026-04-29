"""Tests for SN6PoolDisplay wiring into the live run loop.

Verifies:
- Worker ``on_event`` callback populates display pool state
- ``on_event=None`` (default) does not crash workers
- Events from multiple pools accumulate correctly
- ``refresh_pending`` seeds baseline from ``*_done`` keys
- ``begin_shutdown`` alias exists for shutdown handler
"""

from __future__ import annotations

import time

import pytest

from imas_codex.standard_names.display import (
    POOL_ORDER,
    PoolDisplayState,
    SN6PoolDisplay,
)


class TestOnEventPopulatesPoolState:
    """on_event pushes items into the correct pool and increments counters."""

    def test_generate_name_event(self) -> None:
        display = SN6PoolDisplay(cost_limit=5.0)
        ev = {
            "pool": "generate_name",
            "name": "electron_temperature",
            "source": "core_profiles/profiles_1d/electrons/temperature",
            "dd_path": "core_profiles/profiles_1d/electrons/temperature",
            "model": "gpt-4.1-mini",
            "cost": 0.002,
        }
        display.on_event(ev)

        state = display.pools["generate_name"]
        assert state.completed == 1
        assert state.cost == pytest.approx(0.002)
        assert len(state.items) == 1
        assert state.items[0]["name"] == "electron_temperature"

    def test_review_name_event(self) -> None:
        display = SN6PoolDisplay(cost_limit=5.0)
        ev = {
            "pool": "review_name",
            "name": "plasma_current",
            "score": 0.85,
            "comment": "Good name",
            "stage": "name",
            "model": "gpt-4.1",
            "cost": 0.01,
        }
        display.on_event(ev)

        state = display.pools["review_name"]
        assert state.completed == 1
        assert state.cost == pytest.approx(0.01)
        assert state.items[0]["score"] == 0.85

    def test_refine_name_event(self) -> None:
        display = SN6PoolDisplay(cost_limit=5.0)
        ev = {
            "pool": "refine_name",
            "name": "b_field_toroidal",
            "old_name": "b_tor",
            "new_name": "b_field_toroidal",
            "chain_length": 2,
            "escalated": False,
            "model": "gpt-4.1",
            "cost": 0.015,
        }
        display.on_event(ev)

        state = display.pools["refine_name"]
        assert state.completed == 1
        assert state.items[0]["chain_length"] == 2

    def test_unknown_pool_ignored(self) -> None:
        """Events with unrecognised pool names are silently dropped."""
        display = SN6PoolDisplay(cost_limit=5.0)
        ev = {"pool": "nonexistent_pool", "name": "foo"}
        display.on_event(ev)  # Should not raise
        for state in display.pools.values():
            assert state.completed == 0

    def test_zero_cost_event(self) -> None:
        """Events without ``cost`` key default to 0."""
        display = SN6PoolDisplay(cost_limit=5.0)
        ev = {"pool": "generate_docs", "name": "e_temp"}
        display.on_event(ev)
        state = display.pools["generate_docs"]
        assert state.completed == 1
        assert state.cost == 0.0


class TestOnEventNoneDoesNotCrash:
    """Workers must not crash when on_event=None (default)."""

    def test_default_none_is_safe(self) -> None:
        """Verify that worker functions accept on_event=None without error.

        This tests the API contract: all 6 pool workers and
        ``_compose_batch_core`` accept ``on_event`` as a keyword-only arg
        with ``None`` as default.
        """
        import inspect

        from imas_codex.standard_names.workers import (
            _compose_batch_core,
            process_generate_docs_batch,
            process_generate_name_batch,
            process_refine_docs_batch,
            process_refine_name_batch,
            process_review_docs_batch,
            process_review_name_batch,
        )

        for fn in (
            _compose_batch_core,
            process_generate_name_batch,
            process_refine_name_batch,
            process_review_name_batch,
            process_generate_docs_batch,
            process_review_docs_batch,
            process_refine_docs_batch,
        ):
            sig = inspect.signature(fn)
            param = sig.parameters.get("on_event")
            assert param is not None, f"{fn.__name__} missing on_event param"
            assert param.default is None, f"{fn.__name__} on_event default is not None"
            assert param.kind in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ), f"{fn.__name__} on_event should be keyword-accessible"


class TestMultiPoolAccumulation:
    """Events from multiple pools accumulate independently."""

    def test_multi_pool_isolation(self) -> None:
        display = SN6PoolDisplay(cost_limit=10.0)

        # Feed events to 3 different pools
        for i in range(3):
            display.on_event(
                {
                    "pool": "generate_name",
                    "name": f"name_{i}",
                    "cost": 0.001,
                }
            )
        for i in range(2):
            display.on_event(
                {
                    "pool": "review_name",
                    "name": f"rev_{i}",
                    "score": 0.9,
                    "cost": 0.005,
                }
            )
        display.on_event(
            {
                "pool": "generate_docs",
                "name": "doc_0",
                "cost": 0.01,
            }
        )

        assert display.pools["generate_name"].completed == 3
        assert display.pools["generate_name"].cost == pytest.approx(0.003)
        assert display.pools["review_name"].completed == 2
        assert display.pools["review_name"].cost == pytest.approx(0.01)
        assert display.pools["generate_docs"].completed == 1
        assert display.pools["generate_docs"].cost == pytest.approx(0.01)

        # Untouched pools remain at 0
        assert display.pools["refine_name"].completed == 0
        assert display.pools["review_docs"].completed == 0
        assert display.pools["refine_docs"].completed == 0


class TestRefreshPendingBaseline:
    """refresh_pending seeds completed counts from *_done keys."""

    def test_seeds_from_done_counts(self) -> None:
        pending = {
            "draft": 10,
            "draft_done": 50,
            "review_names": 5,
            "review_names_done": 30,
            "enrich": 3,
            "enrich_done": 20,
            "review_docs": 2,
            "review_docs_done": 15,
            "revise": 4,
        }
        display = SN6PoolDisplay(
            cost_limit=5.0,
            pending_fn=lambda: pending,
        )
        display.refresh_pending()

        # Baselines seeded from *_done keys
        assert display.pools["generate_name"].completed == 50
        assert display.pools["generate_name"].total == 60  # 50 + 10

        assert display.pools["review_name"].completed == 30
        assert display.pools["review_name"].total == 35  # 30 + 5

        assert display.pools["generate_docs"].completed == 20
        assert display.pools["generate_docs"].total == 23  # 20 + 3

        assert display.pools["review_docs"].completed == 15
        assert display.pools["review_docs"].total == 17  # 15 + 2

        # refine_name has no done key — stays at 0 completed but total = pending
        assert display.pools["refine_name"].completed == 0
        assert display.pools["refine_name"].total == 4

    def test_on_event_after_baseline_seed(self) -> None:
        """on_event adds to completed count on top of seeded baseline."""
        pending = {"draft": 5, "draft_done": 10}
        display = SN6PoolDisplay(
            cost_limit=5.0,
            pending_fn=lambda: pending,
        )
        display.refresh_pending()
        assert display.pools["generate_name"].completed == 10

        # Worker completes an item
        display.on_event({"pool": "generate_name", "name": "foo", "cost": 0.001})
        assert display.pools["generate_name"].completed == 11

    def test_pending_fn_exception_swallowed(self) -> None:
        """refresh_pending does not crash if pending_fn raises."""

        def boom() -> dict[str, int]:
            raise RuntimeError("graph down")

        display = SN6PoolDisplay(cost_limit=5.0, pending_fn=boom)
        display.refresh_pending()  # Should not raise
        for state in display.pools.values():
            assert state.completed == 0

    def test_no_pending_fn(self) -> None:
        """refresh_pending is a no-op when pending_fn is None."""
        display = SN6PoolDisplay(cost_limit=5.0)
        display.refresh_pending()  # Should not raise


class TestShutdownAlias:
    """begin_shutdown must exist for the shutdown handler."""

    def test_begin_shutdown_exists(self) -> None:
        display = SN6PoolDisplay(cost_limit=5.0)
        assert hasattr(display, "begin_shutdown")
        assert callable(display.begin_shutdown)

    def test_begin_shutdown_sets_flag(self) -> None:
        display = SN6PoolDisplay(cost_limit=5.0)
        assert not display._shutting_down
        display.begin_shutdown()
        assert display._shutting_down
        assert display._shutdown_start is not None


class TestDisplayPoolsInitialised:
    """All 6 pools from POOL_ORDER are initialised in the display."""

    def test_all_pools_present(self) -> None:
        display = SN6PoolDisplay(cost_limit=5.0)
        for pool_name in POOL_ORDER:
            assert pool_name in display.pools
            state = display.pools[pool_name]
            assert isinstance(state, PoolDisplayState)
            assert state.name == pool_name
            assert state.completed == 0
            assert state.total == 0
            assert state.cost == 0.0
            assert len(state.items) == 0


class TestEventsThisRun:
    """_events_this_run tracks only events in the current session."""

    def test_events_this_run_starts_zero(self) -> None:
        state = PoolDisplayState(name="generate_name")
        assert state._events_this_run == 0

    def test_events_this_run_incremented_by_on_event(self) -> None:
        display = SN6PoolDisplay(cost_limit=5.0)
        display.on_event({"pool": "generate_name", "name": "foo", "cost": 0.001})
        display.on_event({"pool": "generate_name", "name": "bar", "cost": 0.001})
        assert display.pools["generate_name"]._events_this_run == 2

    def test_events_this_run_not_affected_by_baseline(self) -> None:
        """Graph baseline via refresh_pending does NOT inflate _events_this_run."""
        pending = {"draft": 5, "draft_done": 100}
        display = SN6PoolDisplay(cost_limit=5.0, pending_fn=lambda: pending)
        display.refresh_pending()

        state = display.pools["generate_name"]
        # Baseline seeded completed, but _events_this_run untouched
        assert state.completed == 100
        assert state._events_this_run == 0

    def test_rate_uses_events_this_run(self) -> None:
        """Rate is computed from _events_this_run, not completed (baseline)."""
        pending = {"draft": 5, "draft_done": 1000}
        display = SN6PoolDisplay(cost_limit=5.0, pending_fn=lambda: pending)
        display.refresh_pending()

        state = display.pools["generate_name"]
        # Before any event, rate is None (not 1000 / elapsed)
        assert state.rate is None

        # Simulate that pool started 10 seconds ago
        state.start_time = time.time() - 10.0

        # After one event, rate should be ~1/10 = 0.1
        display.on_event({"pool": "generate_name", "name": "x", "cost": 0.001})
        assert state._events_this_run == 1
        assert state.rate is not None
        # Rate ≈ 0.1 (1 event / ~10s), NOT 100.1 (1001 / ~10s)
        assert state.rate < 1.0

    def test_events_isolated_across_pools(self) -> None:
        """_events_this_run tracks per-pool, not global."""
        display = SN6PoolDisplay(cost_limit=5.0)
        display.on_event({"pool": "generate_name", "name": "a", "cost": 0.001})
        display.on_event({"pool": "generate_name", "name": "b", "cost": 0.001})
        display.on_event({"pool": "review_name", "name": "c", "cost": 0.005})

        assert display.pools["generate_name"]._events_this_run == 2
        assert display.pools["review_name"]._events_this_run == 1
        assert display.pools["refine_name"]._events_this_run == 0


class TestContextManager:
    """SN6PoolDisplay can be used as a context manager."""

    def test_enter_exit(self) -> None:
        display = SN6PoolDisplay(cost_limit=5.0)
        # Use it as a context manager — should not crash
        with display as d:
            assert d is display
            # _live should be set
            assert display._live is not None
        # After exit, _live should be stopped
