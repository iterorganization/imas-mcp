"""Tests for core/progress_monitor.py - Progress monitoring."""

import logging
from unittest.mock import patch

import pytest

from imas_codex.core.progress_monitor import (
    BuildProgressMonitor,
    PhaseTracker,
    ProgressMonitor,
    _resolve_rich,
    create_build_monitor,
    create_progress_monitor,
)


class TestResolveRich:
    """Tests for _resolve_rich helper."""

    def test_explicit_true_with_rich(self):
        assert _resolve_rich(True) is True

    def test_explicit_false(self):
        assert _resolve_rich(False) is False

    def test_none_detects_tty(self):
        # Just ensure it returns a bool without crashing
        result = _resolve_rich(None)
        assert isinstance(result, bool)


class TestProgressMonitor:
    """Tests for single-phase ProgressMonitor."""

    def test_init_defaults(self):
        pm = ProgressMonitor(use_rich=False)
        assert pm._current_total == 0
        assert pm._current_completed == 0

    def test_start_processing_sets_total(self):
        pm = ProgressMonitor(use_rich=False)
        pm.start_processing(["a", "b", "c"], description="Testing")
        assert pm._current_total == 3
        assert pm._current_completed == 0

    def test_update_progress_increments(self):
        pm = ProgressMonitor(use_rich=False)
        pm.start_processing(["a", "b"], description="Testing")
        pm.update_progress("a")
        assert pm._current_completed == 1
        pm.update_progress("b")
        assert pm._current_completed == 2

    def test_finish_processing_resets(self):
        pm = ProgressMonitor(use_rich=False)
        pm.start_processing(["a"], description="Testing")
        pm.update_progress("a")
        pm.finish_processing()
        # After finish, progress bar is stopped (no crash)

    def test_log_methods_dont_crash(self):
        pm = ProgressMonitor(use_rich=False)
        pm.log_info("info message")
        pm.log_error("error message")
        pm.log_warning("warning message")

    def test_set_current_item_logging_fallback(self):
        pm = ProgressMonitor(use_rich=False)
        pm.start_processing(["x"], description="Test")
        pm.set_current_item("x")  # Should not crash

    def test_update_progress_with_error(self):
        pm = ProgressMonitor(use_rich=False)
        pm.start_processing(["a"], description="Test")
        pm.update_progress("a", error="something broke")
        assert pm._current_completed == 1


class TestBuildProgressMonitor:
    """Tests for multi-phase BuildProgressMonitor."""

    def test_init(self):
        bpm = BuildProgressMonitor(use_rich=False)
        assert bpm._phases == []

    def test_managed_build_context(self):
        bpm = BuildProgressMonitor(use_rich=False)
        with bpm.managed_build("Test Build"):
            pass  # No crash

    def test_phase_creates_tracker(self):
        bpm = BuildProgressMonitor(use_rich=False)
        with bpm.managed_build():
            tracker = bpm.phase("Extract", items=["a", "b"], total=2)
            assert isinstance(tracker, PhaseTracker)
            assert len(bpm._phases) == 1
            assert bpm._phases[0]["name"] == "Extract"

    def test_phase_tracker_records_completion(self):
        bpm = BuildProgressMonitor(use_rich=False)
        with bpm.managed_build():
            with bpm.phase("Step", items=["x", "y"]) as phase:
                phase.update("x")
                phase.update("y")
                phase.set_detail("2 items")
        assert bpm._phases[0]["ok"] is True
        assert bpm._phases[0]["count"] == 2
        assert bpm._phases[0]["detail"] == "2 items"

    def test_phase_tracker_records_failure(self):
        bpm = BuildProgressMonitor(use_rich=False)
        with bpm.managed_build():
            try:
                with bpm.phase("Fail"):
                    raise ValueError("boom")
            except ValueError:
                pass
        assert bpm._phases[0]["ok"] is False

    def test_phase_tracker_log(self):
        bpm = BuildProgressMonitor(use_rich=False)
        with bpm.managed_build():
            with bpm.phase("Step", total=1) as phase:
                phase.log("something happened")
                phase.update("item")

    def test_phase_tracker_set_description(self):
        bpm = BuildProgressMonitor(use_rich=False)
        with bpm.managed_build():
            with bpm.phase("Step", total=1) as phase:
                phase.set_description("new desc")
                phase.update("item")

    def test_suppress_and_restore_logging(self):
        """Logging suppression/restoration doesn't crash with Rich enabled."""
        bpm = BuildProgressMonitor(use_rich=True)
        logger = logging.getLogger("test_suppress")
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            with bpm.managed_build("Test"):
                pass
        finally:
            logger.removeHandler(handler)


class TestCreateHelpers:
    """Tests for factory functions."""

    def test_create_progress_monitor_returns_pm(self):
        pm = create_progress_monitor(use_rich=False)
        assert isinstance(pm, ProgressMonitor)

    def test_create_build_monitor_returns_bpm(self):
        bpm = create_build_monitor(use_rich=False)
        assert isinstance(bpm, BuildProgressMonitor)
