"""Unit tests for imas_codex.graph.neo4j_ops module.

Tests for Neo4j operation infrastructure extracted from graph_cli.py.
All subprocess calls and filesystem access are mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from imas_codex.graph.neo4j_ops import (
    RECOVERY_DIR,
    backup_existing_data,
    check_stale_neo4j_process,
    parse_dump_error,
)


# ============================================================================
# check_stale_neo4j_process
# ============================================================================


class TestCheckStaleNeo4j:
    """Tests for check_stale_neo4j_process()."""

    def test_no_pid_file(self, tmp_path):
        """No PID file returns (False, None)."""
        is_stale, info = check_stale_neo4j_process(tmp_path)
        assert is_stale is False
        assert info is None

    def test_stale_pid(self, tmp_path, monkeypatch):
        """PID file for dead process is cleaned up and returns (False, None)."""
        pid_file = tmp_path / "neo4j.pid"
        pid_file.write_text("999999")

        # Mock os.kill to raise ProcessLookupError (process doesn't exist)
        def mock_kill(pid, sig):
            raise ProcessLookupError()

        monkeypatch.setattr("os.kill", mock_kill)
        is_stale, info = check_stale_neo4j_process(tmp_path)
        assert is_stale is False
        assert info is None
        # PID file should have been cleaned up
        assert not pid_file.exists()

    def test_process_owned_by_another_user(self, tmp_path, monkeypatch):
        """Process owned by another user is flagged as stale."""
        pid_file = tmp_path / "neo4j.pid"
        pid_file.write_text("999999")

        def mock_kill(pid, sig):
            raise PermissionError()

        monkeypatch.setattr("os.kill", mock_kill)
        is_stale, info = check_stale_neo4j_process(tmp_path)
        assert is_stale is True
        assert "999999" in info


# ============================================================================
# backup_existing_data
# ============================================================================


class TestBackupExistingData:
    """Tests for backup_existing_data()."""

    def test_creates_recovery_dir(self, tmp_path, monkeypatch):
        """Backup creates a timestamped recovery directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "databases").mkdir()
        (data_dir / "databases" / "store.db").write_text("test")

        # Point RECOVERY_DIR to temp
        monkeypatch.setattr(
            "imas_codex.graph.neo4j_ops.RECOVERY_DIR",
            tmp_path / "recovery",
        )

        result = backup_existing_data("test-backup", data_dir=data_dir)
        assert result is not None
        assert result.exists()


# ============================================================================
# parse_dump_error
# ============================================================================


class TestParseDumpError:
    """Tests for parse_dump_error()."""

    def test_lock_detected(self):
        msg, is_lock = parse_dump_error("Error: database is in use by another process")
        assert is_lock is True

    def test_filelockexception(self):
        msg, is_lock = parse_dump_error(
            "org.neo4j.kernel.FileLockException: lock on store"
        )
        assert is_lock is True

    def test_generic_error(self):
        msg, is_lock = parse_dump_error("Caused by: java.io.IOException: disk full")
        assert is_lock is False
        assert "Caused by" in msg

    def test_dump_failed_for_databases(self):
        msg, is_lock = parse_dump_error("Dump failed for databases: 'neo4j'")
        assert is_lock is True

    def test_unable_to_find_store_id(self):
        msg, is_lock = parse_dump_error("Unable to find store id")
        assert is_lock is False
        assert "store id" in msg

    def test_empty_stderr(self):
        msg, is_lock = parse_dump_error("")
        assert is_lock is False
        assert msg == "Unknown error"
