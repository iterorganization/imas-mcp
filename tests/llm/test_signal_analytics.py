"""Tests for the ``signal_analytics`` MCP tool implementation.

``_signal_analytics`` turns a facility + group_by + filters request into one
of two Cypher shapes (with or without the ``CHECKED_WITH`` join) and renders
the counts as a markdown table. These tests pin input validation, query
routing, parameterisation, error handling, and the formatter — all against a
mocked GraphClient, no live Neo4j.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from neo4j.exceptions import ServiceUnavailable

from imas_codex.llm.search_tools import (
    _ALLOWED_GROUP_BY,
    NEO4J_NOT_RUNNING_MSG,
    _format_analytics,
    _signal_analytics,
)


@pytest.fixture
def gc() -> MagicMock:
    client = MagicMock()
    client.query.return_value = []
    return client


def _issued(gc: MagicMock) -> tuple[str, dict]:
    """Return (cypher, params) of the single query the tool issued."""
    gc.query.assert_called_once()
    (cypher,), params = gc.query.call_args
    return cypher, params


# ============================================================================
# group_by validation
# ============================================================================


class TestGroupByValidation:
    def test_default_group_by_is_status(self, gc):
        _signal_analytics("tcv", gc=gc)

        cypher, params = _issued(gc)
        assert "s.status AS status" in cypher
        assert params == {"facility": "tcv"}

    def test_invalid_dimension_rejected_before_querying(self, gc):
        result = _signal_analytics("tcv", group_by=["status", "bogus"], gc=gc)

        assert result.startswith("Invalid group_by dimensions")
        assert "bogus" in result
        assert "status" in result  # allowed list is echoed back
        gc.query.assert_not_called()

    @pytest.mark.parametrize("dim", sorted(_ALLOWED_GROUP_BY))
    def test_every_allowed_dimension_is_accepted(self, gc, dim):
        result = _signal_analytics("tcv", group_by=[dim], gc=gc)

        gc.query.assert_called_once()
        assert not result.startswith("Invalid")


# ============================================================================
# Query routing: simple vs CHECKED_WITH join
# ============================================================================


class TestQueryRouting:
    def test_plain_dimensions_use_simple_query(self, gc):
        _signal_analytics("tcv", group_by=["physics_domain", "diagnostic"], gc=gc)

        cypher, _ = _issued(gc)
        assert "CHECKED_WITH" not in cypher
        assert (
            "s.physics_domain AS physics_domain, s.diagnostic AS diagnostic" in cypher
        )
        assert "count(s) AS count" in cypher
        assert "ORDER BY count DESC" in cypher

    @pytest.mark.parametrize("dim", ["check_status", "error_type"])
    def test_check_dimensions_join_checked_with(self, gc, dim):
        _signal_analytics("tcv", group_by=[dim], gc=gc)

        cypher, _ = _issued(gc)
        assert "OPTIONAL MATCH (s)-[c:CHECKED_WITH]->()" in cypher
        assert "count(DISTINCT s) AS count" in cypher

    def test_check_status_is_derived_from_success_flag(self, gc):
        _signal_analytics("tcv", group_by=["check_status"], gc=gc)

        cypher, _ = _issued(gc)
        assert "WHEN c IS NULL THEN 'unchecked'" in cypher
        assert "WHEN c.success = true THEN 'passed'" in cypher
        assert "ELSE 'failed' END AS check_status" in cypher

    def test_error_type_dimension_reads_relationship_property(self, gc):
        _signal_analytics("tcv", group_by=["error_type"], gc=gc)

        cypher, _ = _issued(gc)
        assert "c.error_type AS error_type" in cypher

    def test_check_filter_alone_routes_to_join_query(self, gc):
        _signal_analytics(
            "tcv", group_by=["status"], filters={"check_status": "failed"}, gc=gc
        )

        cypher, _ = _issued(gc)
        assert "CHECKED_WITH" in cypher
        assert "c.success = false" in cypher

    @pytest.mark.parametrize(
        ("value", "clause"),
        [
            ("passed", "c.success = true"),
            ("failed", "c.success = false"),
            ("unchecked", "c IS NULL"),
        ],
    )
    def test_check_status_filter_values(self, gc, value, clause):
        _signal_analytics("tcv", filters={"check_status": value}, gc=gc)

        cypher, _ = _issued(gc)
        assert clause in cypher

    def test_unknown_check_status_value_adds_no_clause(self, gc):
        _signal_analytics("tcv", filters={"check_status": "maybe"}, gc=gc)

        cypher, _ = _issued(gc)
        assert "CHECKED_WITH" in cypher
        assert "c.success" not in cypher
        assert "c IS NULL" not in cypher

    def test_error_type_filter_is_parameterised(self, gc):
        _signal_analytics("tcv", filters={"error_type": "timeout"}, gc=gc)

        cypher, params = _issued(gc)
        assert "c.error_type = $f_error_type" in cypher
        assert params["f_error_type"] == "timeout"
        assert "timeout" not in cypher


# ============================================================================
# Node-property filters
# ============================================================================


class TestFilters:
    def test_allowed_filter_becomes_where_clause_and_parameter(self, gc):
        _signal_analytics("tcv", filters={"physics_domain": "magnetics"}, gc=gc)

        cypher, params = _issued(gc)
        assert "s.physics_domain = $f_physics_domain" in cypher
        assert params == {"facility": "tcv", "f_physics_domain": "magnetics"}

    def test_filter_values_are_never_interpolated(self, gc):
        hostile = "x' OR 1=1 //"
        _signal_analytics("tcv", filters={"status": hostile}, gc=gc)

        cypher, params = _issued(gc)
        assert hostile not in cypher
        assert params["f_status"] == hostile

    @pytest.mark.parametrize("group_by", [["status"], ["check_status"]])
    def test_unknown_filter_key_is_silently_dropped(self, gc, group_by):
        """Pins current behaviour: unknown filter keys are ignored, not rejected.

        This is asymmetric with ``group_by`` (which is validated). If that is
        changed to an error, this test should flip to assert the message and
        ``gc.query.assert_not_called()``.
        """
        _signal_analytics("tcv", group_by=group_by, filters={"bogus": "x"}, gc=gc)

        cypher, params = _issued(gc)
        assert "bogus" not in cypher
        assert "f_bogus" not in params

    def test_node_filters_also_apply_in_join_query(self, gc):
        _signal_analytics(
            "tcv",
            group_by=["check_status"],
            filters={"diagnostic": "magnetics", "check_status": "passed"},
            gc=gc,
        )

        cypher, params = _issued(gc)
        assert "s.diagnostic = $f_diagnostic" in cypher
        assert params["f_diagnostic"] == "magnetics"
        assert "c.success = true" in cypher


# ============================================================================
# Error handling
# ============================================================================


class TestErrorHandling:
    def test_service_unavailable_returns_setup_hint(self, gc):
        gc.query.side_effect = ServiceUnavailable("connection refused")

        assert _signal_analytics("tcv", gc=gc) == NEO4J_NOT_RUNNING_MSG

    def test_other_errors_are_reported_not_raised(self, gc):
        gc.query.side_effect = RuntimeError("kaboom")

        result = _signal_analytics("tcv", gc=gc)

        assert result.startswith("Analytics error:")


# ============================================================================
# Formatting
# ============================================================================


class TestFormatAnalytics:
    def test_empty_results_message(self):
        assert (
            _format_analytics(["status"], [], "tcv")
            == "No signals found for facility 'tcv'."
        )

    def test_table_with_total_and_percentages(self):
        results = [
            {"status": "checked", "count": 3},
            {"status": "discovered", "count": 1},
        ]

        report = _format_analytics(["status"], results, "tcv")

        assert report.startswith("## Signal Analytics for tcv")
        assert "Total: 4 signals" in report
        assert "| status | count | % |" in report
        assert "| checked | 3 | 75.0 |" in report
        assert "| discovered | 1 | 25.0 |" in report

    def test_cross_tabulation_columns_follow_group_by_order(self):
        results = [{"physics_domain": "magnetics", "status": "checked", "count": 2}]

        report = _format_analytics(["physics_domain", "status"], results, "tcv")

        assert "| physics_domain | status | count | % |" in report
        assert "| magnetics | checked | 2 | 100.0 |" in report

    def test_missing_dimension_value_rendered_as_dash(self):
        report = _format_analytics(["physics_domain"], [{"count": 2}], "tcv")

        assert "| — | 2 | 100.0 |" in report

    def test_total_uses_thousands_separator(self):
        report = _format_analytics(["status"], [{"status": "a", "count": 12345}], "tcv")

        assert "Total: 12,345 signals" in report

    def test_end_to_end_through_tool(self, gc):
        gc.query.return_value = [{"status": "checked", "count": 2}]

        report = _signal_analytics("tcv", gc=gc)

        assert "## Signal Analytics for tcv" in report
        assert "| checked | 2 | 100.0 |" in report
