"""Tests for PipelinePhase completion tracking."""

import asyncio

import pytest

from imas_codex.discovery.base.supervision import PipelinePhase


class TestPipelinePhase:
    def test_initial_state(self):
        phase = PipelinePhase("scan")
        assert not phase.idle
        assert not phase.done
        assert not phase.is_idle_or_done
        assert phase.idle_count == 0
        assert phase.total_processed == 0

    def test_record_activity_resets_idle(self):
        phase = PipelinePhase("scan")
        phase.record_idle()
        phase.record_idle()
        assert phase.idle_count == 2
        phase.record_activity(5)
        assert phase.idle_count == 0
        assert phase.total_processed == 5

    def test_idle_after_threshold(self):
        phase = PipelinePhase("scan", idle_threshold=3)
        phase.record_idle()
        phase.record_idle()
        assert not phase.idle
        phase.record_idle()
        assert phase.idle

    def test_done_without_has_work_fn(self):
        phase = PipelinePhase("scan", idle_threshold=2)
        phase.record_idle()
        phase.record_idle()
        assert phase.idle
        assert phase.done

    def test_done_with_has_work_fn_returns_true(self):
        """When graph says work remains, phase is NOT done even if idle."""
        phase = PipelinePhase("scan", has_work_fn=lambda: True, idle_threshold=2)
        phase.record_idle()
        phase.record_idle()
        assert not phase.done
        # idle_count should have been reset by done check
        assert phase.idle_count == 0

    def test_done_with_has_work_fn_returns_false(self):
        phase = PipelinePhase("scan", has_work_fn=lambda: False, idle_threshold=2)
        phase.record_idle()
        phase.record_idle()
        assert phase.done

    def test_mark_done_forces_completion(self):
        phase = PipelinePhase("scan")
        assert not phase.done
        phase.mark_done()
        assert phase.done
        assert phase.idle  # mark_done implies idle

    def test_is_idle_or_done(self):
        phase = PipelinePhase("scan", idle_threshold=2)
        assert not phase.is_idle_or_done
        phase.record_idle()
        assert not phase.is_idle_or_done
        phase.record_idle()
        assert phase.is_idle_or_done

    def test_is_idle_or_done_after_mark_done(self):
        phase = PipelinePhase("scan")
        phase.mark_done()
        assert phase.is_idle_or_done

    def test_aliases(self):
        phase = PipelinePhase("scan", idle_threshold=1)
        phase.record_idle()
        assert phase.idle == phase.is_idle
        assert phase.done == phase.is_done

    def test_reset(self):
        phase = PipelinePhase("scan", idle_threshold=2)
        phase.record_idle()
        phase.record_idle()
        assert phase.idle
        phase.reset()
        assert not phase.idle

    @pytest.mark.asyncio
    async def test_wait_until_done_with_mark_done(self):
        phase = PipelinePhase("scan")

        async def mark_later():
            await asyncio.sleep(0.01)
            phase.mark_done()

        asyncio.create_task(mark_later())
        result = await phase.wait_until_done(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_until_done_timeout(self):
        phase = PipelinePhase("scan")
        result = await phase.wait_until_done(timeout=0.01)
        assert result is False

    def test_repr(self):
        phase = PipelinePhase("scan")
        r = repr(phase)
        assert "scan" in r
        assert "PipelinePhase" in r
