"""Tests for result formatters."""

from imas_codex.graph.formatters import as_summary, as_table, pick


class TestPick:
    """Test pick() field projection."""

    def test_pick_fields(self):
        data = [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]
        result = pick(data, "a", "c")
        assert result == [{"a": 1, "c": 3}, {"a": 4, "c": 6}]

    def test_pick_missing_field_returns_none(self):
        data = [{"a": 1, "b": 2}]
        result = pick(data, "a", "z")
        assert result == [{"a": 1, "z": None}]

    def test_pick_empty_list(self):
        assert pick([], "a") == []


class TestAsTable:
    """Test as_table() markdown table formatter."""

    def test_basic_table(self):
        data = [{"name": "ip", "unit": "A"}]
        result = as_table(data)
        assert "name" in result
        assert "ip" in result
        assert "|" in result

    def test_custom_columns(self):
        data = [{"a": 1, "b": 2, "c": 3}]
        result = as_table(data, columns=["a", "b"])
        assert "a" in result
        assert "b" in result
        assert "c" not in result

    def test_empty_list(self):
        result = as_table([])
        assert result == ""

    def test_header_separator(self):
        data = [{"x": 1}]
        result = as_table(data)
        lines = result.strip().split("\n")
        assert len(lines) >= 3  # header, separator, data
        assert "---" in lines[1]


class TestAsSummary:
    """Test as_summary() formatter."""

    def test_count_summary(self):
        data = [{"type": "A"}, {"type": "A"}, {"type": "B"}]
        result = as_summary(data)
        assert "3" in result  # total count

    def test_group_by(self):
        data = [
            {"domain": "magnetics", "name": "ip"},
            {"domain": "magnetics", "name": "btor"},
            {"domain": "kinetic", "name": "te"},
        ]
        result = as_summary(data, group_by="domain")
        assert "magnetics" in result
        assert "kinetic" in result

    def test_empty_list(self):
        result = as_summary([])
        assert "0" in result
