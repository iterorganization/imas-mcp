"""Structural graph tests.

Verifies the overall structure and health of the graph: expected
node types present, vector indexes online, build hash integrity,
and cluster structure.
"""

import pytest

from imas_codex.graph.client import EXPECTED_VECTOR_INDEXES

pytestmark = pytest.mark.graph


class TestGraphPresence:
    """Verify expected top-level structure exists."""

    def test_facility_nodes_exist(self, label_counts):
        """At least one Facility node must exist."""
        assert label_counts.get("Facility", 0) >= 1, "No Facility nodes in graph"

    def test_dd_version_exists(self, label_counts):
        """At least one DDVersion node must exist."""
        assert label_counts.get("DDVersion", 0) >= 1, "No DDVersion nodes in graph"

    def test_ids_nodes_exist(self, label_counts):
        """IDS nodes should exist (populated by DD build)."""
        assert label_counts.get("IDS", 0) >= 1, "No IDS nodes in graph"

    def test_imas_path_nodes_exist(self, label_counts):
        """IMASPath nodes should exist (core of DD graph)."""
        count = label_counts.get("IMASPath", 0)
        assert count >= 100, (
            f"Only {count} IMASPath nodes (expected many more). "
            f"DD graph may not be built."
        )

    def test_graph_not_empty(self, graph_stats):
        """Graph should have a meaningful number of nodes."""
        assert graph_stats["nodes"] > 0, "Graph is empty"
        assert graph_stats["relationships"] > 0, "Graph has no relationships"


class TestBuildHash:
    """Verify build hash is stored on current DDVersion."""

    def test_current_dd_version_has_build_hash(self, graph_client):
        """Current DDVersion should have a build_hash (indicates clean build)."""
        result = graph_client.query(
            "MATCH (d:DDVersion {is_current: true}) RETURN d.build_hash AS hash"
        )
        if not result:
            pytest.skip("No current DDVersion node")
        assert result[0]["hash"], "DDVersion.build_hash is null/empty"


class TestVectorIndexes:
    """Verify vector indexes are present and online."""

    def test_vector_indexes_exist(self, graph_indexes):
        """All expected vector indexes should exist."""
        expected_names = {idx[0] for idx in EXPECTED_VECTOR_INDEXES}
        vector_indexes = {
            idx["name"] for idx in graph_indexes if idx["type"] == "VECTOR"
        }
        missing = expected_names - vector_indexes
        assert not missing, (
            f"Missing vector indexes: {missing}. Run GraphClient().initialize_schema()."
        )

    def test_vector_indexes_online(self, graph_indexes):
        """All vector indexes should be in ONLINE state."""
        offline = []
        for idx in graph_indexes:
            if idx["type"] == "VECTOR" and idx["state"] != "ONLINE":
                offline.append(f"{idx['name']}: {idx['state']}")

        assert not offline, "Vector indexes not ONLINE:\n  " + "\n  ".join(offline)

    def test_vector_index_dimensions(self, graph_client, embedding_dimension):
        """Vector indexes should be configured with the correct dimension."""
        result = graph_client.query(
            "SHOW INDEXES YIELD name, type, options "
            "WHERE type = 'VECTOR' "
            "RETURN name, options"
        )
        wrong_dim = []
        for row in result:
            config = row.get("options", {})
            index_config = config.get("indexConfig", {})
            dim = index_config.get("vector.dimensions")
            if dim is not None and dim != embedding_dimension:
                wrong_dim.append(
                    f"{row['name']}: dim={dim} (expected {embedding_dimension})"
                )

        assert not wrong_dim, "Vector indexes with wrong dimension:\n  " + "\n  ".join(
            wrong_dim
        )


class TestClusterIntegrity:
    """Verify semantic cluster structure."""

    def test_clusters_have_centroids(self, graph_client, label_counts):
        """IMASSemanticCluster nodes should have centroid vectors."""
        if not label_counts.get("IMASSemanticCluster"):
            pytest.skip("No IMASSemanticCluster nodes in graph")

        result = graph_client.query(
            "MATCH (c:IMASSemanticCluster) "
            "WHERE c.centroid IS NULL "
            "RETURN count(c) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} IMASSemanticCluster nodes without centroid vector"

    def test_clusters_have_members(self, graph_client, label_counts):
        """Clusters should have at least one IN_CLUSTER member."""
        if not label_counts.get("IMASSemanticCluster"):
            pytest.skip("No IMASSemanticCluster nodes in graph")

        result = graph_client.query(
            "MATCH (c:IMASSemanticCluster) "
            "WHERE NOT (:IMASPath)-[:IN_CLUSTER]->(c) "
            "RETURN count(c) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} IMASSemanticCluster nodes with no IN_CLUSTER members"
        )

    def test_cluster_centroid_dimensions(
        self, graph_client, label_counts, embedding_dimension
    ):
        """Cluster centroids must match the embedding dimension."""
        if not label_counts.get("IMASSemanticCluster"):
            pytest.skip("No IMASSemanticCluster nodes in graph")

        result = graph_client.query(
            "MATCH (c:IMASSemanticCluster) "
            "WHERE c.centroid IS NOT NULL "
            "RETURN DISTINCT size(c.centroid) AS dim "
            "LIMIT 10"
        )
        wrong = [r for r in result if r["dim"] != embedding_dimension]
        assert not wrong, (
            f"Cluster centroids have wrong dimension: "
            f"{[r['dim'] for r in wrong]} (expected {embedding_dimension})"
        )
