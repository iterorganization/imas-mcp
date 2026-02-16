"""Tests for the run_supervised_loop common supervision infrastructure."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, call

import pytest

from imas_codex.discovery.base.supervision import (
    OrphanRecoverySpec,
    SupervisedWorkerGroup,
    make_orphan_recovery_tick,
    run_supervised_loop,
)


@pytest.fixture
def worker_group() -> SupervisedWorkerGroup:
    """Create a worker group with one dummy worker."""
    wg = SupervisedWorkerGroup()
    wg.create_status("test_worker_0", group="test")
    return wg


class TestRunSupervisedLoop:
    """Tests for run_supervised_loop."""

    @pytest.mark.asyncio
    async def test_exits_when_should_stop_returns_true(self, worker_group):
        """Loop exits immediately when should_stop returns True."""
        await run_supervised_loop(
            worker_group,
            lambda: True,
            shutdown_timeout=1.0,
        )
        # Should complete without hanging

    @pytest.mark.asyncio
    async def test_calls_on_worker_status_initially(self, worker_group):
        """on_worker_status is called once at startup."""
        callback = MagicMock()

        await run_supervised_loop(
            worker_group,
            lambda: True,
            on_worker_status=callback,
            shutdown_timeout=1.0,
        )

        # Should have been called at least once (initial update)
        assert callback.call_count >= 1
        callback.assert_called_with(worker_group)

    @pytest.mark.asyncio
    async def test_polls_until_should_stop(self, worker_group):
        """Loop polls should_stop and exits when it returns True."""
        call_count = 0

        def should_stop():
            nonlocal call_count
            call_count += 1
            return call_count > 3

        await run_supervised_loop(
            worker_group,
            should_stop,
            shutdown_timeout=1.0,
            poll_interval=0.01,
        )

        assert call_count > 3

    @pytest.mark.asyncio
    async def test_calls_on_tick(self, worker_group):
        """on_tick callback is called during polling."""
        tick_count = 0
        poll_count = 0

        def on_tick():
            nonlocal tick_count
            tick_count += 1

        def should_stop():
            nonlocal poll_count
            poll_count += 1
            return poll_count > 5

        await run_supervised_loop(
            worker_group,
            should_stop,
            on_tick=on_tick,
            shutdown_timeout=1.0,
            poll_interval=0.01,
        )

        assert tick_count > 0

    @pytest.mark.asyncio
    async def test_calls_async_on_tick(self, worker_group):
        """on_tick can be an async callable."""
        tick_count = 0
        poll_count = 0

        async def on_tick():
            nonlocal tick_count
            tick_count += 1

        def should_stop():
            nonlocal poll_count
            poll_count += 1
            return poll_count > 5

        await run_supervised_loop(
            worker_group,
            should_stop,
            on_tick=on_tick,
            shutdown_timeout=1.0,
            poll_interval=0.01,
        )

        assert tick_count > 0

    @pytest.mark.asyncio
    async def test_handles_cancelled_error(self, worker_group):
        """CancelledError during polling is handled gracefully."""
        poll_count = 0

        def should_stop():
            nonlocal poll_count
            poll_count += 1
            return False

        # Run in a task that we can cancel
        task = asyncio.create_task(
            run_supervised_loop(
                worker_group,
                should_stop,
                shutdown_timeout=1.0,
                poll_interval=0.01,
            )
        )

        await asyncio.sleep(0.05)
        task.cancel()

        # CancelledError from cancel_all propagates but the loop itself
        # catches it internally and proceeds to cleanup
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected — cancel_all may re-raise

        # Loop ran at least a few iterations before cancel
        assert poll_count > 0

    @pytest.mark.asyncio
    async def test_cancels_worker_tasks(self, worker_group):
        """Worker group tasks are cancelled on shutdown."""
        completed = asyncio.Event()

        async def dummy_worker():
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                completed.set()
                raise

        worker_group.add_task(asyncio.create_task(dummy_worker()))
        poll_count = 0

        def should_stop():
            nonlocal poll_count
            poll_count += 1
            return poll_count > 2

        await run_supervised_loop(
            worker_group,
            should_stop,
            shutdown_timeout=2.0,
            poll_interval=0.01,
        )

        # Worker should have been cancelled
        assert completed.is_set()

    @pytest.mark.asyncio
    async def test_status_callback_exception_handled(self, worker_group):
        """Exceptions in on_worker_status callback don't crash the loop."""
        call_count = 0

        def bad_callback(wg):
            nonlocal call_count
            call_count += 1
            raise ValueError("callback error")

        poll_count = 0

        def should_stop():
            nonlocal poll_count
            poll_count += 1
            return poll_count > 5

        # Should complete without raising
        await run_supervised_loop(
            worker_group,
            should_stop,
            on_worker_status=bad_callback,
            shutdown_timeout=1.0,
            poll_interval=0.01,
            status_interval=0.01,
        )

        # Callback was called (initial + periodic)
        assert call_count >= 1


# =============================================================================
# Orphan Recovery Tick
# =============================================================================


class TestOrphanRecoverySpec:
    """Tests for OrphanRecoverySpec dataclass."""

    def test_defaults(self):
        """Default values are sensible."""
        spec = OrphanRecoverySpec("FacilityPath")
        assert spec.label == "FacilityPath"
        assert spec.facility_field == "facility_id"
        assert spec.timeout_seconds == 300

    def test_custom_values(self):
        """Custom values override defaults."""
        spec = OrphanRecoverySpec(
            "WikiPage", facility_field="facility", timeout_seconds=600
        )
        assert spec.label == "WikiPage"
        assert spec.facility_field == "facility"
        assert spec.timeout_seconds == 600


class TestMakeOrphanRecoveryTick:
    """Tests for the make_orphan_recovery_tick factory."""

    def test_returns_callable(self):
        """Factory returns a callable."""
        tick = make_orphan_recovery_tick(
            "test_facility",
            [OrphanRecoverySpec("FacilityPath")],
        )
        assert callable(tick)

    def test_respects_interval(self, monkeypatch):
        """Tick is a no-op within the interval window."""
        # Mock reset_stale_claims to track calls
        calls = []

        def mock_reset(label, facility, **kwargs):
            calls.append((label, facility))
            return 0

        monkeypatch.setattr(
            "imas_codex.discovery.base.supervision.reset_stale_claims",
            mock_reset,
            raising=False,
        )
        # Use lazy import path
        import imas_codex.discovery.base.claims as claims_mod

        monkeypatch.setattr(claims_mod, "reset_stale_claims", mock_reset)

        tick = make_orphan_recovery_tick(
            "test",
            [OrphanRecoverySpec("FacilityPath")],
            interval=1000.0,  # Very long interval
        )

        # First call should not trigger (interval hasn't elapsed)
        tick()
        assert len(calls) == 0

    def test_triggers_after_interval(self, monkeypatch):
        """Tick triggers recovery after interval elapses."""
        import time as time_mod

        calls = []

        def mock_reset(label, facility, **kwargs):
            calls.append((label, facility))
            return 0

        monkeypatch.setattr(
            "imas_codex.discovery.base.claims.reset_stale_claims",
            mock_reset,
        )

        tick = make_orphan_recovery_tick(
            "test",
            [OrphanRecoverySpec("FacilityPath")],
            interval=0.0,  # Always trigger
        )

        tick()
        assert len(calls) == 1
        assert calls[0] == ("FacilityPath", "test")

    def test_multiple_specs(self, monkeypatch):
        """Tick processes all specs in order."""
        calls = []

        def mock_reset(label, facility, **kwargs):
            calls.append((label, facility, kwargs.get("timeout_seconds")))
            return 0

        monkeypatch.setattr(
            "imas_codex.discovery.base.claims.reset_stale_claims",
            mock_reset,
        )

        tick = make_orphan_recovery_tick(
            "iter",
            [
                OrphanRecoverySpec("WikiPage", timeout_seconds=300),
                OrphanRecoverySpec("WikiArtifact", timeout_seconds=600),
            ],
            interval=0.0,
        )

        tick()
        assert len(calls) == 2
        assert calls[0] == ("WikiPage", "iter", 300)
        assert calls[1] == ("WikiArtifact", "iter", 600)

    def test_handles_exception(self, monkeypatch):
        """Exceptions in reset_stale_claims don't crash the tick."""

        def mock_reset(label, facility, **kwargs):
            raise RuntimeError("Neo4j down")

        monkeypatch.setattr(
            "imas_codex.discovery.base.claims.reset_stale_claims",
            mock_reset,
        )

        tick = make_orphan_recovery_tick(
            "test",
            [OrphanRecoverySpec("FacilityPath")],
            interval=0.0,
        )

        # Should not raise
        tick()

    @pytest.mark.asyncio
    async def test_integration_with_supervised_loop(self, worker_group, monkeypatch):
        """Orphan tick integrates with run_supervised_loop."""
        calls = []

        def mock_reset(label, facility, **kwargs):
            calls.append(label)
            return 0

        monkeypatch.setattr(
            "imas_codex.discovery.base.claims.reset_stale_claims",
            mock_reset,
        )

        tick = make_orphan_recovery_tick(
            "test",
            [OrphanRecoverySpec("FacilityPath")],
            interval=0.0,  # Always trigger
        )

        poll_count = 0

        def should_stop():
            nonlocal poll_count
            poll_count += 1
            return poll_count > 5

        await run_supervised_loop(
            worker_group,
            should_stop,
            on_tick=tick,
            shutdown_timeout=1.0,
            poll_interval=0.01,
        )

        # Tick was called multiple times during the loop
        assert len(calls) > 0
