"""Tests for DD version validity filtering and cluster scope query construction.

Validates that _dd_version_clause uses correct "valid in major N" semantics:
- A path is valid in DD major N if introduced in major ≤ N and not deprecated in major ≤ N.
- Paths introduced in DD3 and never deprecated are valid in both DD3 and DD4.
- Paths deprecated in DD4 are valid in DD3 but not DD4.
- Paths deprecated in DD3 are invalid in both DD3 and DD4.

Also validates that cluster scope queries produce valid Cypher.
"""

from unittest.mock import MagicMock, call

import pytest

from imas_codex.tools.graph_search import (
    GraphClustersTool,
    GraphPathTool,
    _dd_version_clause,
)


# ============================================================================
# _dd_version_clause unit tests
# ============================================================================


class TestDDVersionClause:
    """Test the version filtering helper directly."""

    def test_none_returns_empty(self):
        assert _dd_version_clause("p", dd_version=None) == ""

    def test_none_does_not_modify_params(self):
        params = {}
        _dd_version_clause("p", dd_version=None, params=params)
        assert params == {}

    def test_returns_and_clause(self):
        clause = _dd_version_clause("p", dd_version=4)
        assert clause.startswith("AND EXISTS")

    def test_sets_dd_version_param(self):
        params = {}
        _dd_version_clause("p", dd_version=4, params=params)
        assert params["dd_version"] == 4

    def test_uses_correct_alias(self):
        clause = _dd_version_clause("node", dd_version=3)
        assert "(node)-[:INTRODUCED_IN]->" in clause
        assert "(node)-[:DEPRECATED_IN]->" in clause

    def test_uses_integer_comparison(self):
        """The clause must use toInteger(split()) for major version comparison,
        not STARTS WITH prefix matching."""
        clause = _dd_version_clause("p", dd_version=4)
        assert "toInteger(split(" in clause
        assert "$dd_version" in clause
        assert "STARTS WITH" not in clause

    def test_checks_introduced_lte(self):
        """Must check introduced_in major <= N (not ==)."""
        clause = _dd_version_clause("p", dd_version=4)
        assert "<= $dd_version" in clause

    def test_checks_not_deprecated_lte(self):
        """Must exclude paths deprecated in any version with major <= N."""
        clause = _dd_version_clause("p", dd_version=4)
        assert "NOT EXISTS" in clause
        assert "DEPRECATED_IN" in clause


# ============================================================================
# Version filtering integration tests (mock graph)
# ============================================================================


class TestVersionFilteringSemantics:
    """Test that check_imas_paths applies version filtering correctly via mocked graph."""

    @pytest.mark.asyncio
    async def test_dd3_path_found_with_dd_version_4(self):
        """A path introduced in DD3 and never deprecated must be found with dd_version=4."""
        gc = MagicMock()
        # First call: the main query with dd_version clause
        gc.query.return_value = [
            {"id": "equilibrium/time_slice/profiles_1d/psi", "ids": "equilibrium",
             "data_type": "FLT_1D", "units": "Wb"}
        ]
        tool = GraphPathTool(gc)
        result = await tool.check_imas_paths(
            "equilibrium/time_slice/profiles_1d/psi", dd_version=4
        )
        assert result.results[0].exists is True

        # Verify the Cypher includes the correct version semantics
        cypher = gc.query.call_args_list[0][0][0]
        assert "toInteger(split(" in cypher
        assert "STARTS WITH" not in cypher

    @pytest.mark.asyncio
    async def test_dd_version_param_is_integer(self):
        """The dd_version parameter passed to Cypher must be an integer, not a string prefix."""
        gc = MagicMock()
        gc.query.return_value = [
            {"id": "test/path", "ids": "test", "data_type": "FLT_0D", "units": ""}
        ]
        tool = GraphPathTool(gc)
        await tool.check_imas_paths("test/path", dd_version=4)

        # Check that dd_version=4 (int) was passed, not dd_version_prefix="4."
        kwargs = gc.query.call_args_list[0][1]
        assert "dd_version" in kwargs
        assert kwargs["dd_version"] == 4
        assert "dd_version_prefix" not in kwargs

    @pytest.mark.asyncio
    async def test_no_dd_version_skips_filter(self):
        """When dd_version is None, no version filter clause should be in the query."""
        gc = MagicMock()
        gc.query.return_value = [
            {"id": "test/path", "ids": "test", "data_type": "FLT_0D", "units": ""}
        ]
        tool = GraphPathTool(gc)
        await tool.check_imas_paths("test/path", dd_version=None)

        cypher = gc.query.call_args_list[0][0][0]
        assert "INTRODUCED_IN" not in cypher
        assert "DEPRECATED_IN" not in cypher


# ============================================================================
# Cluster scope syntax tests
# ============================================================================


class TestClusterScopeQuery:
    """Test that cluster _search_by_path produces valid Cypher with scope parameter."""

    def test_search_by_path_with_scope_has_where(self):
        """When scope is provided, the query must have WHERE before the AND clause."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphClustersTool(gc)

        tool._search_by_path("equilibrium/time_slice/profiles_1d/psi", scope="global")

        cypher = gc.query.call_args[0][0]
        # The scope filter must come after a WHERE, not bare after MATCH
        assert "WHERE true AND c.scope = $scope" in cypher
        # Verify scope param was passed
        kwargs = gc.query.call_args[1]
        assert kwargs["scope"] == "global"

    def test_search_by_path_without_scope(self):
        """When scope is None, no scope filter should appear in WHERE."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphClustersTool(gc)

        tool._search_by_path("equilibrium/time_slice/profiles_1d/psi", scope=None)

        cypher = gc.query.call_args[0][0]
        assert "c.scope = $scope" not in cypher

    def test_search_by_path_with_scope_and_dd_version(self):
        """Both scope and dd_version must produce valid Cypher."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphClustersTool(gc)

        tool._search_by_path(
            "equilibrium/time_slice/profiles_1d/psi",
            scope="ids",
            dd_version=4,
        )

        cypher = gc.query.call_args[0][0]
        # Both filters present and valid
        assert "WHERE true AND c.scope = $scope" in cypher
        assert "INTRODUCED_IN" in cypher
        kwargs = gc.query.call_args[1]
        assert kwargs["scope"] == "ids"
        assert kwargs["dd_version"] == 4

    @pytest.mark.asyncio
    async def test_search_imas_clusters_path_with_scope(self):
        """Full tool call with scope must not raise."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphClustersTool(gc)

        result = await tool.search_imas_clusters(
            query="equilibrium/time_slice/profiles_1d/psi",
            scope="global",
        )

        assert result["query_type"] == "path"
        assert result["clusters_found"] == 0
