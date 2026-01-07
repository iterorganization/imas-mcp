"""Tests for Neo4j value serialization in agents server."""

from datetime import UTC, datetime

import pytest

from imas_codex.agents.server import _serialize_neo4j_value


class MockNeo4jDateTime:
    """Mock neo4j.time.DateTime for testing without neo4j dependency."""

    def __init__(self, iso_string: str):
        self._iso = iso_string
        self.tzinfo = UTC

    def isoformat(self) -> str:
        return self._iso


class TestSerializeNeo4jValue:
    """Test the Neo4j value serialization function."""

    def test_none_passthrough(self):
        """None values should pass through unchanged."""
        assert _serialize_neo4j_value(None) is None

    def test_primitive_passthrough(self):
        """Primitive types should pass through unchanged."""
        assert _serialize_neo4j_value(42) == 42
        assert _serialize_neo4j_value(3.14) == 3.14
        assert _serialize_neo4j_value("hello") == "hello"
        assert _serialize_neo4j_value(True) is True

    def test_python_datetime_serialized(self):
        """Python datetime objects should be serialized to ISO format."""
        dt = datetime(2026, 1, 7, 12, 0, 0, tzinfo=UTC)
        result = _serialize_neo4j_value(dt)
        assert isinstance(result, str)
        assert "2026-01-07" in result

    def test_neo4j_datetime_serialized(self):
        """Neo4j DateTime objects should be serialized to ISO format."""
        mock_dt = MockNeo4jDateTime("2026-01-07T12:00:00+00:00")
        result = _serialize_neo4j_value(mock_dt)
        assert result == "2026-01-07T12:00:00+00:00"

    def test_dict_recursive_serialization(self):
        """Dicts should have their values recursively serialized."""
        mock_dt = MockNeo4jDateTime("2026-01-07T12:00:00+00:00")
        data = {
            "name": "test",
            "count": 42,
            "timestamp": mock_dt,
            "nested": {"inner_time": mock_dt},
        }
        result = _serialize_neo4j_value(data)
        assert result["name"] == "test"
        assert result["count"] == 42
        assert result["timestamp"] == "2026-01-07T12:00:00+00:00"
        assert result["nested"]["inner_time"] == "2026-01-07T12:00:00+00:00"

    def test_list_recursive_serialization(self):
        """Lists should have their items recursively serialized."""
        mock_dt = MockNeo4jDateTime("2026-01-07T12:00:00+00:00")
        data = ["hello", 42, mock_dt, {"time": mock_dt}]
        result = _serialize_neo4j_value(data)
        assert result[0] == "hello"
        assert result[1] == 42
        assert result[2] == "2026-01-07T12:00:00+00:00"
        assert result[3]["time"] == "2026-01-07T12:00:00+00:00"

    def test_empty_containers(self):
        """Empty containers should be handled correctly."""
        assert _serialize_neo4j_value({}) == {}
        assert _serialize_neo4j_value([]) == []
