"""Tests for the SEARCH clause builder."""

import pytest

from imas_codex.graph.vector_search import build_vector_search


class TestBuildVectorSearch:
    """Tests for build_vector_search()."""

    def test_basic_search_no_filters(self):
        """Generates valid SEARCH clause without filters."""
        result = build_vector_search("imas_node_embedding", "IMASNode")
        assert "SEARCH n:IMASNode" in result
        assert "USING VECTOR INDEX imas_node_embedding" in result
        assert "vector.similarity.cosine(n.embedding, $embedding)" in result
        assert "ORDER BY score DESC" in result
        assert "LIMIT $k" in result
        assert "CALL () {" in result
        assert result.endswith("}")
        # No WHERE clause
        assert "WHERE" not in result

    def test_single_where_clause(self):
        """Single property filter is included in SEARCH WHERE."""
        result = build_vector_search(
            "facility_signal_desc_embedding",
            "FacilitySignal",
            where_clauses=["n.facility_id = $facility"],
        )
        assert "WHERE n.facility_id = $facility" in result
        assert "SEARCH n:FacilitySignal" in result

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
        assert "SEARCH chunk:WikiChunk" in result
        assert "vector.similarity.cosine(chunk.embedding, $embedding) AS sim" in result
        assert "ORDER BY sim DESC" in result

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
        assert "vector.similarity.cosine(n.embedding, $query_vec)" in result

    def test_custom_embedding_property(self):
        """Non-default embedding property name."""
        result = build_vector_search(
            "cluster_embedding",
            "IMASSemanticCluster",
            embedding_property="label_embedding",
        )
        assert "vector.similarity.cosine(n.label_embedding, $embedding)" in result

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
        assert "CALL () {" in full_query

    def test_all_vector_indexes_from_schema(self):
        """Verify builder works with all known vector indexes."""
        # VECTOR_INDEXES values are (label, property) tuples
        from imas_codex.graph.schema_context_data import VECTOR_INDEXES

        for index_name, (label, prop) in VECTOR_INDEXES.items():
            result = build_vector_search(index_name, label, embedding_property=prop)
            assert f"SEARCH n:{label}" in result
            assert f"USING VECTOR INDEX {index_name}" in result
            assert f"vector.similarity.cosine(n.{prop}, $embedding)" in result
