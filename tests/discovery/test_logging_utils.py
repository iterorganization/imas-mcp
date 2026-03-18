"""Tests for CLI logging utilities (Phase 3 + 4).

Covers:
- WorkerLogAdapter structured logging
- log_worker_error consistent error format
- list_log_files, read_log, tail_log file reading
- Time parsing for _parse_since
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from imas_codex.cli.logging import (
    StructuredFormatter,
    WorkerLogAdapter,
    _parse_since,
    get_log_dir,
    get_log_file,
    list_log_files,
    log_worker_error,
    read_log,
    tail_log,
)

# =============================================================================
# Phase 4.1: Structured Worker Logging
# =============================================================================


class TestWorkerLogAdapter:
    """Tests for WorkerLogAdapter."""

    def test_message_includes_worker_name(self, caplog):
        logger = logging.getLogger("test_worker_adapter")
        adapter = WorkerLogAdapter(logger, worker_name="check_worker_2")

        with caplog.at_level(logging.INFO):
            adapter.info("checked 20 signals")

        assert any("check_worker_2" in r.message for r in caplog.records)
        assert any("checked 20 signals" in r.message for r in caplog.records)

    def test_message_includes_batch_id(self, caplog):
        logger = logging.getLogger("test_worker_adapter_batch")
        adapter = WorkerLogAdapter(
            logger, worker_name="enrich_worker_0", batch_id="abc123"
        )

        with caplog.at_level(logging.INFO):
            adapter.info("processing batch")

        assert any("batch=abc123" in r.message for r in caplog.records)

    def test_set_batch(self, caplog):
        logger = logging.getLogger("test_set_batch")
        adapter = WorkerLogAdapter(logger, worker_name="w1")
        adapter.set_batch("batch_xyz")

        with caplog.at_level(logging.INFO):
            adapter.info("test")

        assert any("batch=batch_xyz" in r.message for r in caplog.records)

    def test_no_batch_id(self, caplog):
        logger = logging.getLogger("test_no_batch")
        adapter = WorkerLogAdapter(logger, worker_name="w1")

        with caplog.at_level(logging.INFO):
            adapter.info("test")

        # Should not contain "batch="
        assert not any("batch=" in r.message for r in caplog.records)


# =============================================================================
# Phase 4.1b: StructuredFormatter
# =============================================================================


class TestStructuredFormatter:
    """Tests for StructuredFormatter — extras-aware file log formatting."""

    def test_standard_format_without_extras(self):
        """Messages without structured extras use standard format."""
        fmt = StructuredFormatter()
        record = logging.LogRecord(
            name="imas_codex.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="regular message",
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
        assert "regular message" in output
        assert "[" not in output  # No extras bracket

    def test_worker_name_in_output(self):
        """Messages with worker_name extra include it in brackets."""
        fmt = StructuredFormatter()
        record = logging.LogRecord(
            name="imas_codex.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="checked 20 signals",
            args=(),
            exc_info=None,
        )
        record.worker_name = "check_worker_2"
        output = fmt.format(record)
        assert "[check_worker_2]" in output
        assert "checked 20 signals" in output

    def test_worker_and_batch_in_output(self):
        """Messages with both worker and batch extras."""
        fmt = StructuredFormatter()
        record = logging.LogRecord(
            name="imas_codex.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="processing",
            args=(),
            exc_info=None,
        )
        record.worker_name = "enrich_worker"
        record.batch_id = "abc123"
        output = fmt.format(record)
        assert "enrich_worker" in output
        assert "batch=abc123" in output

    def test_signal_id_in_output(self):
        """Messages with signal_id extra include it."""
        fmt = StructuredFormatter()
        record = logging.LogRecord(
            name="imas_codex.test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="failed",
            args=(),
            exc_info=None,
        )
        record.signal_id = "jet:ip/measured"
        output = fmt.format(record)
        assert "signal=jet:ip/measured" in output


class TestSupervisionReexports:
    """Verify WorkerLogAdapter and log_worker_error are importable from supervision."""

    def test_worker_log_adapter_from_supervision(self):
        from imas_codex.discovery.base.supervision import WorkerLogAdapter as WLA

        assert WLA is WorkerLogAdapter

    def test_log_worker_error_from_supervision(self):
        from imas_codex.discovery.base.supervision import (
            log_worker_error as lwe,
        )

        assert lwe is log_worker_error


# =============================================================================
# Phase 4.2: Consistent Error Logging
# =============================================================================


class TestLogWorkerError:
    """Tests for log_worker_error."""

    def test_infrastructure_error(self, caplog):
        logger = logging.getLogger("test_infra_error")

        with caplog.at_level(logging.WARNING):
            log_worker_error(
                logger,
                worker_name="check_worker_2",
                signal_id="jet:ip/measured",
                error=ConnectionError("SSH timeout"),
                error_type="infrastructure",
                retry_count=1,
                max_retries=3,
            )

        assert any("check_worker_2" in r.message for r in caplog.records)
        assert any("type=infrastructure" in r.message for r in caplog.records)
        assert any("signal=jet:ip/measured" in r.message for r in caplog.records)
        assert any("retry=1/3" in r.message for r in caplog.records)
        # Infrastructure errors are WARNING level
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_application_error(self, caplog):
        logger = logging.getLogger("test_app_error")

        with caplog.at_level(logging.ERROR):
            log_worker_error(
                logger,
                worker_name="enrich_worker_0",
                error=ValueError("Invalid signal"),
                error_type="application",
                batch_id="batch_abc",
            )

        assert any("enrich_worker_0" in r.message for r in caplog.records)
        assert any("type=application" in r.message for r in caplog.records)
        assert any("batch=batch_abc" in r.message for r in caplog.records)
        # Application errors are ERROR level
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_no_optional_fields(self, caplog):
        logger = logging.getLogger("test_minimal_error")

        with caplog.at_level(logging.ERROR):
            log_worker_error(
                logger,
                worker_name="seed_worker",
                error=RuntimeError("fail"),
                error_type="application",
            )

        assert any("seed_worker" in r.message for r in caplog.records)


# =============================================================================
# Phase 3: Log Reading Utilities
# =============================================================================


@pytest.fixture
def temp_log_dir(tmp_path, monkeypatch):
    """Create a temporary log directory with test log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Patch LOG_DIR and force local routing (avoid SSH to remote)
    monkeypatch.setattr("imas_codex.cli.logging.LOG_DIR", log_dir)
    monkeypatch.setattr("imas_codex.cli.logging._get_log_ssh_host", lambda: None)

    # Create test log files
    signals_log = log_dir / "signals_jet.log"
    signals_log.write_text(
        "2026-03-13 10:00:00 DEBUG    imas_codex.discovery: starting scan\n"
        "2026-03-13 10:00:01 INFO     imas_codex.discovery: seeded wiki\n"
        "2026-03-13 10:00:02 INFO     imas_codex.discovery: seeded ppf: 5204 signals\n"
        "2026-03-13 10:00:03 WARNING  imas_codex.discovery: SSH timeout to jet\n"
        "2026-03-13 10:00:04 ERROR    imas_codex.discovery: check_worker_2: signal failed\n"
        "2026-03-13 10:00:05 INFO     imas_codex.discovery: enriched 20 signals\n"
        "2026-03-13 10:00:06 WARNING  imas_codex.discovery: rate limited\n"
        "2026-03-13 10:00:07 ERROR    imas_codex.discovery: connection refused\n"
    )

    wiki_log = log_dir / "wiki_tcv.log"
    wiki_log.write_text(
        "2026-03-13 09:00:00 INFO     imas_codex.wiki: starting wiki scan\n"
        "2026-03-13 09:01:00 WARNING  imas_codex.wiki: page not found\n"
    )

    return log_dir


class TestListLogFiles:
    """Tests for list_log_files."""

    def test_lists_files(self, temp_log_dir):
        files = list_log_files()
        assert len(files) >= 2
        names = [f["name"] for f in files]
        assert "signals_jet.log" in names
        assert "wiki_tcv.log" in names

    def test_file_metadata(self, temp_log_dir):
        files = list_log_files()
        for f in files:
            assert "name" in f
            assert "path" in f
            assert "size_bytes" in f
            assert "modified_iso" in f
            assert "age_hours" in f
            assert f["size_bytes"] > 0

    def test_empty_dir(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty_logs"
        empty.mkdir()
        monkeypatch.setattr("imas_codex.cli.logging.LOG_DIR", empty)
        monkeypatch.setattr("imas_codex.cli.logging._get_log_ssh_host", lambda: None)
        files = list_log_files()
        assert files == []


class TestReadLog:
    """Tests for read_log with filtering."""

    def test_read_warnings_and_above(self, temp_log_dir):
        result = read_log(command="signals", facility="jet", level="WARNING")
        assert "SSH timeout" in result
        assert "connection refused" in result
        assert "starting scan" not in result  # DEBUG excluded
        assert "seeded wiki" not in result  # INFO excluded

    def test_read_all_levels(self, temp_log_dir):
        result = read_log(
            command="signals", facility="jet", level="DEBUG", lines=100
        )
        assert "starting scan" in result
        assert "SSH timeout" in result

    def test_grep_filter(self, temp_log_dir):
        result = read_log(
            command="signals", facility="jet", level="DEBUG", grep="ssh"
        )
        assert "SSH timeout" in result
        assert "seeded wiki" not in result

    def test_lines_limit(self, temp_log_dir):
        result = read_log(
            command="signals", facility="jet", level="DEBUG", lines=2
        )
        lines = result.strip().split("\n")
        assert len(lines) <= 2

    def test_missing_log(self, temp_log_dir):
        result = read_log(command="signals", facility="nonexistent")
        assert "not found" in result.lower()

    def test_errors_only(self, temp_log_dir):
        result = read_log(command="signals", facility="jet", level="ERROR")
        lines = result.strip().split("\n")
        for line in lines:
            if line.strip():
                assert "ERROR" in line


class TestTailLog:
    """Tests for tail_log."""

    def test_tail_returns_last_lines(self, temp_log_dir):
        result = tail_log(command="signals", facility="jet", lines=3)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        # Last line should be the connection refused error
        assert "connection refused" in lines[-1]

    def test_tail_all_lines(self, temp_log_dir):
        result = tail_log(command="signals", facility="jet", lines=100)
        assert "starting scan" in result
        assert "connection refused" in result

    def test_tail_missing_log(self, temp_log_dir):
        result = tail_log(command="nonexistent", facility="missing")
        assert "not found" in result.lower()


# =============================================================================
# Time Parsing
# =============================================================================


class TestParseSince:
    """Tests for _parse_since time string parser."""

    def test_hours(self):
        dt = _parse_since("1h")
        assert dt is not None
        # Should be ~1 hour ago
        age = (time.time() - dt.timestamp())
        assert 3500 < age < 3700

    def test_minutes(self):
        dt = _parse_since("30m")
        assert dt is not None
        age = (time.time() - dt.timestamp())
        assert 1700 < age < 1900

    def test_days(self):
        dt = _parse_since("2d")
        assert dt is not None
        age = (time.time() - dt.timestamp())
        assert 170000 < age < 175000

    def test_seconds(self):
        dt = _parse_since("45s")
        assert dt is not None
        age = (time.time() - dt.timestamp())
        assert 44 < age < 47

    def test_iso_datetime(self):
        dt = _parse_since("2026-03-13T10:00")
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 3
        assert dt.day == 13

    def test_iso_date(self):
        dt = _parse_since("2026-03-13")
        assert dt is not None
        assert dt.year == 2026

    def test_invalid(self):
        dt = _parse_since("not a time")
        assert dt is None
