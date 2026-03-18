"""Tests for scanner diagnostics infrastructure.

Covers all phases of the signal scanner diagnostics plan:
- Phase 1: Scanner progress tracking, SSH connection timing, scanner timing
- Phase 2: Error rate tracking, backoff visibility, SSH health summary
- Phase 4: Structured log fields, consistent error logging, graph state snapshots
"""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock

import pytest

from imas_codex.discovery.base.progress import (
    ResourceConfig,
    ScannerProgress,
    WorkerStats,
    build_resource_section,
    build_servers_section,
    build_worker_status_section,
)
from imas_codex.discovery.base.supervision import (
    SupervisedWorkerGroup,
    WorkerState,
    WorkerStatus,
    make_snapshot_logger,
)

# =============================================================================
# Phase 1: Scanner Progress Tracking
# =============================================================================


class TestScannerProgress:
    """Tests for ScannerProgress dataclass."""

    def test_initial_state(self):
        sp = ScannerProgress(name="wiki")
        assert sp.name == "wiki"
        assert sp.status == "pending"
        assert sp.items_discovered == 0

    def test_mark_running(self):
        sp = ScannerProgress(name="ppf")
        sp.mark_running("subsystem DA 3/26")
        assert sp.status == "running"
        assert sp.detail == "subsystem DA 3/26"

    def test_mark_done(self):
        sp = ScannerProgress(name="wiki")
        sp.mark_running()
        sp.mark_done(items=5204)
        assert sp.status == "done"
        assert sp.items_discovered == 5204
        assert sp.end_time is not None

    def test_mark_failed(self):
        sp = ScannerProgress(name="mdsplus")
        sp.mark_running()
        sp.mark_failed("SSH timeout")
        assert sp.status == "failed"
        assert sp.error == "SSH timeout"
        assert sp.end_time is not None

    def test_elapsed_running(self):
        sp = ScannerProgress(name="jpf")
        sp.mark_running()
        time.sleep(0.01)
        assert sp.elapsed > 0

    def test_elapsed_done(self):
        sp = ScannerProgress(name="wiki")
        sp.mark_running()
        time.sleep(0.01)
        sp.mark_done()
        elapsed = sp.elapsed
        time.sleep(0.01)
        # Elapsed should not change after done
        assert sp.elapsed == elapsed

    def test_format_status_pending(self):
        sp = ScannerProgress(name="wiki")
        assert sp.format_status() == "wiki"

    def test_format_status_running(self):
        sp = ScannerProgress(name="jpf")
        sp.mark_running("subsystem DA 3/26")
        assert sp.format_status() == "jpf: subsystem DA 3/26"

    def test_format_status_running_no_detail(self):
        sp = ScannerProgress(name="ppf")
        sp.mark_running()
        assert sp.format_status() == "ppf…"

    def test_format_status_done(self):
        sp = ScannerProgress(name="wiki")
        sp.mark_done()
        assert sp.format_status() == "wiki✓"

    def test_format_status_done_with_count(self):
        sp = ScannerProgress(name="ppf")
        sp.mark_done(items=5204)
        assert sp.format_status() == "ppf✓ 5.2K"

    def test_format_status_failed(self):
        sp = ScannerProgress(name="mdsplus")
        sp.mark_failed("error")
        assert sp.format_status() == "mdsplus✗"


class TestWorkerStatsScanner:
    """Tests for scanner tracking on WorkerStats."""

    def test_start_scanner(self):
        stats = WorkerStats()
        sp = stats.start_scanner("wiki")
        assert "wiki" in stats.scanner_progress
        assert sp.status == "running"

    def test_finish_scanner(self):
        stats = WorkerStats()
        stats.start_scanner("ppf")
        stats.finish_scanner("ppf", items=1000)
        assert stats.scanner_progress["ppf"].status == "done"
        assert stats.scanner_progress["ppf"].items_discovered == 1000

    def test_fail_scanner(self):
        stats = WorkerStats()
        stats.start_scanner("mdsplus")
        stats.fail_scanner("mdsplus", "SSH timeout")
        assert stats.scanner_progress["mdsplus"].status == "failed"
        assert stats.scanner_progress["mdsplus"].error == "SSH timeout"

    def test_format_scanner_status(self):
        stats = WorkerStats()
        stats.start_scanner("wiki")
        stats.finish_scanner("wiki")
        stats.start_scanner("ppf")
        stats.finish_scanner("ppf", items=5204)
        status = stats.format_scanner_status()
        assert "wiki✓" in status
        assert "ppf✓ 5.2K" in status

    def test_format_scanner_status_empty(self):
        stats = WorkerStats()
        assert stats.format_scanner_status() == ""

    def test_format_scanner_timing(self):
        stats = WorkerStats()
        sp = stats.start_scanner("wiki")
        sp.mark_done()
        timing = stats.format_scanner_timing()
        assert "wiki:" in timing
        assert "s" in timing

    def test_format_scanner_timing_empty(self):
        stats = WorkerStats()
        assert stats.format_scanner_timing() == ""


# =============================================================================
# Phase 1.2: SSH Connection Timing
# =============================================================================


class TestConnectionTiming:
    """Tests for SSH connection timing on WorkerStats."""

    def test_not_connecting(self):
        stats = WorkerStats()
        assert stats.connection_elapsed is None
        assert stats.format_connection_status() == ""

    def test_connecting(self):
        stats = WorkerStats()
        stats.mark_connecting("jet")
        time.sleep(0.01)
        assert stats.connection_elapsed > 0
        status = stats.format_connection_status()
        assert "connecting" in status
        assert "jet" in status

    def test_connected(self):
        stats = WorkerStats()
        stats.mark_connecting("tcv")
        stats.mark_connected()
        assert stats.connection_elapsed is None
        assert stats.format_connection_status() == ""

    def test_long_connection_shows_warning(self):
        stats = WorkerStats()
        stats.mark_connecting("jet")
        # Simulate old start time
        stats._connection_start = time.time() - 45
        status = stats.format_connection_status()
        assert "⚡" in status
        assert "45s" in status


# =============================================================================
# Phase 2.1: Error Rate Tracking
# =============================================================================


class TestErrorRateTracking:
    """Tests for rolling error rate tracking on WorkerStats."""

    def test_initial_error_rate(self):
        stats = WorkerStats()
        assert stats.error_rate_1m == 0.0
        assert stats.consecutive_errors == 0
        assert stats.error_rate_pct == 0.0

    def test_record_error(self):
        stats = WorkerStats()
        stats.record_error()
        assert stats.errors == 1
        assert stats.consecutive_errors == 1
        assert stats.error_rate_1m > 0

    def test_record_success_resets_consecutive(self):
        stats = WorkerStats()
        stats.record_error()
        stats.record_error()
        assert stats.consecutive_errors == 2
        stats.record_success()
        assert stats.consecutive_errors == 0
        # Total errors should still be tracked
        assert stats.errors == 2

    def test_error_rate_pct(self):
        stats = WorkerStats()
        stats.processed = 80
        stats.errors = 20
        assert abs(stats.error_rate_pct - 20.0) < 0.01

    def test_error_rate_pct_zero_items(self):
        stats = WorkerStats()
        assert stats.error_rate_pct == 0.0

    def test_error_health_style_green(self):
        stats = WorkerStats()
        stats.processed = 100
        stats.errors = 2
        assert stats.error_health_style == "green"

    def test_error_health_style_yellow(self):
        stats = WorkerStats()
        stats.processed = 80
        stats.errors = 20
        assert stats.error_health_style == "yellow"

    def test_error_health_style_red(self):
        stats = WorkerStats()
        stats.processed = 50
        stats.errors = 50
        assert stats.error_health_style == "red"

    def test_rolling_window_error_rate(self):
        stats = WorkerStats()
        # Add timestamps outside the window
        old_time = time.time() - 120  # 2 minutes ago
        stats._error_timestamps = [old_time, old_time + 1]
        stats.errors = 2
        # These should be pruned when recording a new error
        stats.record_error()
        # Only 1 recent error in the window
        assert len(stats._error_timestamps) == 1


# =============================================================================
# Phase 2.2: Backoff Visibility
# =============================================================================


class TestBackoffVisibility:
    """Tests for backoff remaining time on WorkerStatus."""

    def test_no_backoff(self):
        ws = WorkerStatus(name="check_worker_0")
        ws.state = WorkerState.running
        assert ws.backoff_remaining == 0.0

    def test_backoff_remaining(self):
        ws = WorkerStatus(name="check_worker_0")
        ws.state = WorkerState.backoff
        ws.backoff_until = time.time() + 30
        remaining = ws.backoff_remaining
        assert 29 <= remaining <= 31

    def test_backoff_expired(self):
        ws = WorkerStatus(name="check_worker_0")
        ws.state = WorkerState.backoff
        ws.backoff_until = time.time() - 5  # Already expired
        assert ws.backoff_remaining == 0.0

    def test_backoff_in_to_dict(self):
        ws = WorkerStatus(name="check_worker_0")
        ws.state = WorkerState.backoff
        ws.backoff_until = time.time() + 10
        d = ws.to_dict()
        assert "backoff_remaining" in d
        assert d["backoff_remaining"] > 0

    def test_worker_section_shows_backoff_time(self):
        """build_worker_status_section shows backoff remaining time."""
        wg = SupervisedWorkerGroup()
        ws = wg.create_status("check_worker_0", group="check")
        ws.state = WorkerState.backoff
        ws.backoff_until = time.time() + 23

        text = build_worker_status_section(wg)
        plain = text.plain
        assert "backoff" in plain
        assert "23s" in plain or "22s" in plain  # Allow 1s timing variance


# =============================================================================
# Phase 2.3: SSH Health Summary
# =============================================================================


class TestServiceHealthSummary:
    """Tests for ServiceStatus health tracking."""

    def test_record_check_healthy(self):
        from imas_codex.discovery.base.services import ServiceStatus

        s = ServiceStatus(name="ssh")
        s.record_check(True, 2300.0)
        assert s.total_checks == 1
        assert s.total_failures == 0
        assert s.avg_latency_ms == 2300.0

    def test_record_check_failure(self):
        from imas_codex.discovery.base.services import ServiceStatus

        s = ServiceStatus(name="ssh")
        s.record_check(False, 0.0)
        assert s.total_failures == 1
        assert s.failure_ratio == "1/1"

    def test_avg_latency(self):
        from imas_codex.discovery.base.services import ServiceStatus

        s = ServiceStatus(name="ssh")
        s.record_check(True, 1000.0)
        s.record_check(True, 3000.0)
        assert abs(s.avg_latency_ms - 2000.0) < 0.01

    def test_format_health_summary_with_stats(self):
        from imas_codex.discovery.base.services import ServiceStatus

        s = ServiceStatus(name="ssh", detail="jet")
        s.record_check(True, 2300.0)
        s.record_check(True, 2500.0)
        s.record_check(False, 0.0)
        summary = s.format_health_summary()
        assert "avg" in summary
        assert "fail" in summary
        assert "1/3" in summary

    def test_format_health_summary_no_failures(self):
        from imas_codex.discovery.base.services import ServiceStatus

        s = ServiceStatus(name="ssh", detail="jet")
        s.record_check(True, 1000.0)
        summary = s.format_health_summary()
        assert "avg" in summary
        assert "fail" not in summary

    def test_format_health_summary_no_checks(self):
        from imas_codex.discovery.base.services import ServiceStatus

        s = ServiceStatus(name="ssh")
        summary = s.format_health_summary()
        assert summary == ""


# =============================================================================
# Phase 1.3: Scanner Timing in Resource Section
# =============================================================================


class TestScannerTimingDisplay:
    """Tests for scanner timing in resource section."""

    def test_resource_section_with_scanner_timing(self):
        config = ResourceConfig(
            elapsed=120.0,
            stats=[("total", "100", "blue")],
            scanner_timing="wiki:2.1s  ppf:8.4s  mdsplus:0.3s",
        )
        text = build_resource_section(config, gauge_width=20)
        plain = text.plain
        assert "SCANNERS" in plain
        assert "wiki:2.1s" in plain
        assert "ppf:8.4s" in plain

    def test_resource_section_without_scanner_timing(self):
        config = ResourceConfig(
            elapsed=120.0,
            stats=[("total", "100", "blue")],
        )
        text = build_resource_section(config, gauge_width=20)
        plain = text.plain
        assert "SCANNERS" not in plain


# =============================================================================
# Phase 4.3: Graph State Snapshots
# =============================================================================


class TestGraphStateSnapshots:
    """Tests for periodic graph state snapshot logging."""

    def test_snapshot_logger_creation(self):
        state = MagicMock()
        state.total_signals = 1000
        state.signals_enriched = 500
        state.signals_checked = 200
        state.total_cost = 5.0

        tick = make_snapshot_logger("jet", state, interval=0.01)
        assert callable(tick)

    def test_snapshot_logger_logs(self, caplog):
        state = MagicMock()
        state.total_signals = 1000
        state.signals_enriched = 500
        state.signals_checked = 200
        state.pending_enrich = 300
        state.pending_check = 300
        state.total_cost = 5.0
        # Remove attributes that shouldn't auto-detect
        del state.total_pages
        del state.pages_scored
        del state.pages_ingested
        del state.total_paths
        del state.paths_scanned
        del state.paths_scored
        del state.signals_discovered

        tick = make_snapshot_logger("jet", state, interval=0.0)
        with caplog.at_level(logging.INFO):
            tick()

        assert any("SNAPSHOT jet" in r.message for r in caplog.records)
        assert any("total=1000" in r.message for r in caplog.records)

    def test_snapshot_logger_respects_interval(self, caplog):
        state = MagicMock()
        state.total_signals = 100
        state.total_cost = 0.0
        del state.total_pages
        del state.pages_scored
        del state.pages_ingested
        del state.total_paths
        del state.paths_scanned
        del state.paths_scored
        del state.signals_discovered
        del state.signals_enriched
        del state.signals_checked
        del state.pending_enrich
        del state.pending_check

        tick = make_snapshot_logger("tcv", state, interval=300.0)
        with caplog.at_level(logging.INFO):
            tick()
            tick()  # Should not log — interval not elapsed

        snapshot_logs = [r for r in caplog.records if "SNAPSHOT" in r.message]
        assert len(snapshot_logs) <= 1

    def test_snapshot_with_custom_fields(self, caplog):
        state = MagicMock()
        state.my_total = 42
        # MagicMock returns MagicMock for any attr by default,
        # so explicitly null out total_cost to avoid comparison errors
        state.total_cost = 0.0

        tick = make_snapshot_logger(
            "tcv",
            state,
            interval=0.0,
            count_fields={"items": "my_total"},
        )
        with caplog.at_level(logging.INFO):
            tick()

        assert any("items=42" in r.message for r in caplog.records)
