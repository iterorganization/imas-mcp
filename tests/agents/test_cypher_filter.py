"""Tests for Cypher mutation detection in agents server."""

import pytest

from imas_codex.agents.server import _is_cypher_mutation


class TestCypherMutationFilter:
    """Test the Cypher mutation detection function."""

    @pytest.mark.parametrize(
        "query,description",
        [
            # Keywords inside string literals should pass
            (
                'MATCH (n) WHERE n.path CONTAINS "SETTINGS" RETURN n',
                "SETTINGS in string",
            ),
            ('MATCH (n) WHERE n.path =~ ".*:SET.*" RETURN n', "SET in regex string"),
            ('MATCH (n {dataset: "test"}) RETURN n', "dataset property"),
            ("RETURN 123 AS offset", "offset AS alias"),
            (
                'MATCH (n) WHERE n.notes CONTAINS "DETACHED" RETURN n',
                "DETACHED in string",
            ),
            ('MATCH (n) WHERE n.name = "MERGE_TOOL" RETURN n', "MERGE_TOOL in string"),
            ("MATCH (n) WHERE n.created_at IS NOT NULL RETURN n", "created_at field"),
            (
                'MATCH (n) WHERE n.path = "\\\\RESULTS::TOP.SET_POINT" RETURN n',
                "MDSplus SET_POINT path",
            ),
            ('MATCH (n) WHERE n.path CONTAINS "RESET" RETURN n', "RESET in path"),
            ('MATCH (n) WHERE n.path CONTAINS "OFFSET" RETURN n', "OFFSET in path"),
            (
                'MATCH (n) WHERE n.description CONTAINS "CREATE" RETURN n',
                "CREATE in search",
            ),
            ('MATCH (n) WHERE n.notes CONTAINS "DELETE" RETURN n', "DELETE in search"),
            ('MATCH (n) WHERE n.action = "REMOVE" RETURN n', "REMOVE in string"),
            ("MATCH (n) WHERE n.name = 'MERGE_DATA' RETURN n", "Single-quoted MERGE"),
            # Standard read queries
            ("MATCH (n:Facility) RETURN n.id, n.name", "Simple MATCH"),
            ("MATCH (n)-[:REL]->(m) RETURN n, m", "Relationship pattern"),
            ("MATCH (n) WITH n.name AS name RETURN name", "WITH clause"),
            ("MATCH (n) OPTIONAL MATCH (n)-[:REL]->(m) RETURN n, m", "OPTIONAL MATCH"),
            ("MATCH (n) RETURN count(*) AS cnt", "Aggregate function"),
        ],
    )
    def test_read_queries_pass(self, query: str, description: str):
        """Read-only queries should not be blocked."""
        assert not _is_cypher_mutation(query), f"Query blocked: {description}"

    @pytest.mark.parametrize(
        "query,description",
        [
            ('CREATE (n:Test {name: "test"}) RETURN n', "CREATE statement"),
            ('MATCH (n) SET n.prop = "value" RETURN n', "SET statement"),
            ("MATCH (n) MERGE (m:Test) RETURN m", "MERGE statement"),
            ("MATCH (n) DELETE n", "DELETE statement"),
            ("MATCH (n) DETACH DELETE n", "DETACH DELETE statement"),
            ("MATCH (n) REMOVE n.prop RETURN n", "REMOVE statement"),
            ("MATCH (n) set n.x = 1", "lowercase set"),
            ("MATCH (n) Set n.x = 1", "mixed case Set"),
            ("CREATE INDEX FOR (n:Label) ON (n.prop)", "CREATE INDEX"),
            ("MERGE (n:Test) ON CREATE SET n.created = true", "MERGE ON CREATE"),
        ],
    )
    def test_mutation_queries_blocked(self, query: str, description: str):
        """Mutation queries should be blocked."""
        assert _is_cypher_mutation(query), f"Query not blocked: {description}"

    def test_mixed_case_keywords_blocked(self):
        """Mutation keywords are detected regardless of case."""
        assert _is_cypher_mutation("match (n) SET n.x = 1")
        assert _is_cypher_mutation("match (n) set n.x = 1")
        assert _is_cypher_mutation("match (n) Set n.x = 1")
        assert _is_cypher_mutation("MATCH (n) create (m)")
        assert _is_cypher_mutation("MATCH (n) Create (m)")

    def test_empty_query(self):
        """Empty query is not a mutation."""
        assert not _is_cypher_mutation("")

    def test_whitespace_only_query(self):
        """Whitespace-only query is not a mutation."""
        assert not _is_cypher_mutation("   \n\t  ")
