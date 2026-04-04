"""Tests for the SEARCH clause builder."""

import pytest

from imas_codex.graph.vector_search import build_vector_search


class TestBuildVectorSearch:
    """Tests for build_vector_search()."""

    def test_basic_search_no_filters(self):
        """Generates valid Cypher 25 SEARCH clause without filters."""
        result = build_vector_search("imas_node_embedding", "IMASNode")
        assert "CYPHER 25" in result
        assert "MATCH (n:IMASNode)" in result
        assert "SEARCH n IN (" in result
        assert "VECTOR INDEX imas_node_embedding" in result
        assert "FOR $embedding" in result
        assert "LIMIT $k" in result
        assert ") SCORE AS score" in result
        # No WHERE clause
        assert "WHERE" not in result

    def test_single_where_clause(self):
        """Single property filter is included as post-filter WHERE."""
        result = build_vector_search(
            "facility_signal_desc_embedding",
            "FacilitySignal",
            where_clauses=["n.facility_id = $facility"],
        )
        assert "WHERE n.facility_id = $facility" in result
        assert "MATCH (n:FacilitySignal)" in result

    def test_multiple_where_clauses(self):
        """Multiple filters are AND-joined."""
        result = build_vector_search(
            "facility_signal_desc_embedding",
            "FacilitySignal",
            where_clauses=[
                "n.facility_id = $facility",
                "n.physics_domain = $domain",
            ],
        )
        assert (
            "WHERE n.facility_id = $facility AND n.physics_domain = $domain" in result
        )

    def test_custom_node_alias(self):
        """Custom node alias is used throughout."""
        result = build_vector_search(
            "wiki_chunk_embedding",
            "WikiChunk",
            node_alias="chunk",
            score_alias="sim",
        )
        assert "MATCH (chunk:WikiChunk)" in result
        assert "SEARCH chunk IN (" in result
        assert ") SCORE AS sim" in result

    def test_custom_k_expression(self):
        """Custom k expression (literal or parameter)."""
        result = build_vector_search("imas_node_embedding", "IMASNode", k="20")
        assert "LIMIT 20" in result

        result2 = build_vector_search("imas_node_embedding", "IMASNode", k="$limit")
        assert "LIMIT $limit" in result2

    def test_custom_embedding_param(self):
        """Custom embedding parameter name."""
        result = build_vector_search(
            "code_chunk_embedding",
            "CodeChunk",
            embedding_param="$query_vec",
        )
        assert "FOR $query_vec" in result

    def test_empty_where_clauses(self):
        """Empty list of where clauses produces no WHERE."""
        result = build_vector_search(
            "imas_node_embedding", "IMASNode", where_clauses=[]
        )
        assert "WHERE" not in result

    def test_output_is_composable(self):
        """Output can be concatenated with additional Cypher."""
        search = build_vector_search("imas_node_embedding", "IMASNode")
        full_query = f"{search}\nRETURN n.id AS id, score"
        assert "RETURN n.id AS id, score" in full_query
        assert "CYPHER 25" in full_query

    def test_relationship_filters_in_where(self):
        """Relationship pattern predicates are valid in where_clauses."""
        result = build_vector_search(
            "imas_node_embedding",
            "IMASNode",
            where_clauses=[
                "n.node_category = 'data'",
                "NOT (n)-[:DEPRECATED_IN]->(:DDVersion)",
            ],
        )
        assert (
            "WHERE n.node_category = 'data' AND NOT (n)-[:DEPRECATED_IN]->(:DDVersion)"
            in result
        )

    def test_all_vector_indexes_from_schema(self):
        """Verify builder works with all known vector indexes."""
        from imas_codex.graph.schema_context_data import VECTOR_INDEXES

        for index_name, (label, _prop) in VECTOR_INDEXES.items():
            result = build_vector_search(index_name, label)
            assert f"MATCH (n:{label})" in result
            assert f"VECTOR INDEX {index_name}" in result
            assert "CYPHER 25" in result
