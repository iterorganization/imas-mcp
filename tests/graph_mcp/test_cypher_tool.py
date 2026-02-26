"""Tests for the read-only Cypher tool.

Tests mutation blocking, result handling, truncation, and edge cases.
"""

import pytest

from imas_codex.tools.cypher_tool import CypherTool, _is_read_only

pytestmark = pytest.mark.graph_mcp


class TestCypherReadOnlyCheck:
    """Test the _is_read_only function without requiring Neo4j."""

    def test_match_return_allowed(self):
        assert _is_read_only("MATCH (n) RETURN n")

    def test_match_with_where(self):
        assert _is_read_only(
            "MATCH (n:IMASPath) WHERE n.ids_name = 'equilibrium' RETURN n.id"
        )

    def test_call_procedure_allowed(self):
        assert _is_read_only(
            "CALL db.index.vector.queryNodes('idx', 5, $vec) YIELD node RETURN node"
        )

    def test_with_order_limit(self):
        assert _is_read_only("MATCH (n) WITH n ORDER BY n.id LIMIT 10 RETURN n.id")

    def test_create_blocked(self):
        assert not _is_read_only("CREATE (n:Test {id: 'x'})")

    def test_merge_blocked(self):
        assert not _is_read_only("MERGE (n:Test {id: 'x'})")

    def test_delete_blocked(self):
        assert not _is_read_only("MATCH (n) DELETE n")

    def test_detach_delete_blocked(self):
        assert not _is_read_only("MATCH (n) DETACH DELETE n")

    def test_set_blocked(self):
        assert not _is_read_only("MATCH (n) SET n.name = 'x'")

    def test_remove_blocked(self):
        assert not _is_read_only("MATCH (n) REMOVE n.name")

    def test_drop_blocked(self):
        assert not _is_read_only("DROP INDEX my_index")

    def test_load_csv_blocked(self):
        assert not _is_read_only("LOAD CSV FROM 'file:///x.csv' AS row")

    def test_case_insensitive_block(self):
        assert not _is_read_only("match (n) create (m)")

    def test_mutation_in_string_literal_allowed(self):
        """Mutation keywords inside string literals should not trigger blocking."""
        assert _is_read_only("MATCH (n) WHERE n.description CONTAINS 'CREATE' RETURN n")

    def test_mutation_in_comment_allowed(self):
        """Mutation keywords in comments should not trigger blocking."""
        assert _is_read_only("// This would CREATE a node\nMATCH (n) RETURN n")

    def test_foreach_blocked(self):
        assert not _is_read_only("MATCH (n) FOREACH (x IN [1,2] | SET n.val = x)")

    def test_call_subquery_blocked(self):
        """CALL { ... } subqueries with writes should be blocked."""
        assert not _is_read_only("CALL { CREATE (n:Test) } RETURN 1")

    def test_empty_query(self):
        assert _is_read_only("")


class TestCypherToolExecution:
    """Test CypherTool execution against the fixture graph."""

    @pytest.fixture
    def cypher_tool(self, graph_client):
        return CypherTool(graph_client)

    @pytest.mark.anyio
    async def test_simple_query(self, cypher_tool):
        result = await cypher_tool.query_imas_graph(
            "MATCH (d:DDVersion) RETURN d.id AS id ORDER BY d.id"
        )
        assert "error" not in result
        assert result["row_count"] == 3
        ids = [r["id"] for r in result["rows"]]
        assert "4.1.0" in ids

    @pytest.mark.anyio
    async def test_parameterless_traversal(self, cypher_tool):
        result = await cypher_tool.query_imas_graph(
            "MATCH (p:IMASPath)-[:IN_IDS]->(i:IDS {id: 'equilibrium'}) "
            "RETURN p.id AS path ORDER BY p.id"
        )
        assert result["row_count"] == 5

    @pytest.mark.anyio
    async def test_mutation_rejected(self, cypher_tool):
        result = await cypher_tool.query_imas_graph(
            "CREATE (n:Test {id: 'should_fail'})"
        )
        assert "error" in result
        assert "rejected" in result["error"].lower()

    @pytest.mark.anyio
    async def test_empty_query_returns_error(self, cypher_tool):
        result = await cypher_tool.query_imas_graph("")
        assert "error" in result

    @pytest.mark.anyio
    async def test_empty_result_set(self, cypher_tool):
        result = await cypher_tool.query_imas_graph(
            "MATCH (n:NonExistentLabel) RETURN n"
        )
        assert result["rows"] == []
        assert result["truncated"] is False

    @pytest.mark.anyio
    async def test_invalid_cypher_returns_error(self, cypher_tool):
        result = await cypher_tool.query_imas_graph("NOT VALID CYPHER")
        assert "error" in result

    @pytest.mark.anyio
    async def test_columns_returned(self, cypher_tool):
        result = await cypher_tool.query_imas_graph(
            "MATCH (d:DDVersion) RETURN d.id AS version_id, d.is_current AS current LIMIT 1"
        )
        assert "version_id" in result["columns"]
        assert "current" in result["columns"]

    @pytest.mark.anyio
    async def test_version_evolution_query(self, cypher_tool):
        """Test the kind of query agents would compose for version evolution."""
        result = await cypher_tool.query_imas_graph(
            "MATCH (c:IMASPathChange)-[:FOR_IMAS_PATH]->(p:IMASPath) "
            "RETURN c.change_type AS change, p.id AS path, c.to_version AS version"
        )
        assert result["row_count"] >= 1
        row = result["rows"][0]
        assert "change" in row
        assert "path" in row
