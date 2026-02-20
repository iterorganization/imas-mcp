"""Tests for Neo4j error detection and recovery in GraphClient."""

from __future__ import annotations

import pytest
from neo4j.exceptions import DatabaseError, ServiceUnavailable

from imas_codex.discovery.base.supervision import is_infrastructure_error
from imas_codex.graph.client import GraphClient


class TestIsConnectionError:
    """Tests for GraphClient._is_connection_error."""

    def test_service_unavailable(self):
        exc = ServiceUnavailable("connection refused")
        assert GraphClient._is_connection_error(exc) is True

    def test_connection_error(self):
        exc = ConnectionError("connection reset")
        assert GraphClient._is_connection_error(exc) is True

    def test_os_error(self):
        exc = OSError("network unreachable")
        assert GraphClient._is_connection_error(exc) is True

    def test_database_critical_error(self):
        exc = DatabaseError(
            "The database has encountered a critical error, and needs to be restarted."
        )
        assert GraphClient._is_connection_error(exc) is True

    def test_database_non_critical_error(self):
        """Non-critical DatabaseErrors should NOT be treated as connection errors."""
        exc = DatabaseError("some other database error")
        assert GraphClient._is_connection_error(exc) is False

    def test_value_error_not_connection(self):
        exc = ValueError("bad value")
        assert GraphClient._is_connection_error(exc) is False

    def test_nested_connection_error(self):
        inner = ConnectionError("reset")
        outer = RuntimeError("wrapper")
        outer.__cause__ = inner
        assert GraphClient._is_connection_error(outer) is True


class TestIsDatabaseCriticalError:
    """Tests for GraphClient._is_database_critical_error."""

    def test_critical_error_detected(self):
        exc = DatabaseError(
            "The database has encountered a critical error, and needs to be restarted."
        )
        assert GraphClient._is_database_critical_error(exc) is True

    def test_non_critical_database_error(self):
        exc = DatabaseError("constraint violation")
        assert GraphClient._is_database_critical_error(exc) is False

    def test_non_database_error(self):
        exc = ServiceUnavailable("connection refused")
        assert GraphClient._is_database_critical_error(exc) is False


class TestIsInfrastructureError:
    """Tests for is_infrastructure_error in supervision module."""

    def test_service_unavailable(self):
        exc = ServiceUnavailable("connection refused")
        assert is_infrastructure_error(exc) is True

    def test_database_critical_error(self):
        exc = DatabaseError(
            "The database has encountered a critical error, and needs to be restarted."
        )
        assert is_infrastructure_error(exc) is True

    def test_database_non_critical_error(self):
        exc = DatabaseError("constraint violation")
        assert is_infrastructure_error(exc) is False

    def test_connection_refused_string(self):
        exc = Exception("connection refused by server")
        assert is_infrastructure_error(exc) is True

    def test_timeout_string(self):
        exc = Exception("operation timed out")
        assert is_infrastructure_error(exc) is True

    def test_unrelated_error(self):
        exc = ValueError("bad argument")
        assert is_infrastructure_error(exc) is False
