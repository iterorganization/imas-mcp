"""Tests validating the fixture graph is correctly loaded.

These tests verify the hand-crafted DD graph fixture data is present
and structurally correct in Neo4j. They serve as the foundation for
all subsequent graph-native MCP tests.
"""

import pytest

pytestmark = pytest.mark.graph_mcp


class TestFixtureGraphStructure:
    """Verify the fixture graph has the expected DD node types and counts."""

    def test_dd_versions_exist(self, graph_client):
        """DDVersion nodes are created with correct count."""
        result = graph_client.query("MATCH (d:DDVersion) RETURN count(d) AS c")
        assert result[0]["c"] == 3

    def test_current_version(self, graph_client):
        """Exactly one DDVersion is marked as current."""
        result = graph_client.query(
            "MATCH (d:DDVersion {is_current: true}) RETURN d.id AS id"
        )
        assert len(result) == 1
        assert result[0]["id"] == "4.1.0"

    def test_version_chain(self, graph_client):
        """DDVersion nodes are chained via PREDECESSOR."""
        result = graph_client.query(
            "MATCH (a:DDVersion)-[:PREDECESSOR]->(b:DDVersion) "
            "RETURN a.id AS from_v, b.id AS to_v ORDER BY a.id"
        )
        assert len(result) == 2
        pairs = [(r["from_v"], r["to_v"]) for r in result]
        assert ("4.0.0", "3.42.0") in pairs
        assert ("4.1.0", "4.0.0") in pairs

    def test_ids_nodes_exist(self, graph_client):
        """IDS nodes are created."""
        result = graph_client.query("MATCH (i:IDS) RETURN i.id AS id ORDER BY i.id")
        ids_names = [r["id"] for r in result]
        assert "equilibrium" in ids_names
        assert "core_profiles" in ids_names

    def test_imas_paths_exist(self, graph_client):
        """IMASPath nodes are created with expected count."""
        result = graph_client.query("MATCH (p:IMASPath) RETURN count(p) AS c")
        assert result[0]["c"] == 9

    def test_paths_linked_to_ids(self, graph_client):
        """Every IMASPath is linked to its IDS via IN_IDS."""
        result = graph_client.query(
            "MATCH (p:IMASPath) WHERE NOT (p)-[:IN_IDS]->(:IDS) RETURN p.id AS id"
        )
        assert len(result) == 0, f"Paths not linked to IDS: {[r['id'] for r in result]}"

    def test_paths_linked_to_version(self, graph_client):
        """Every IMASPath is linked to its introduction version."""
        result = graph_client.query(
            "MATCH (p:IMASPath) WHERE NOT (p)-[:INTRODUCED_IN]->(:DDVersion) "
            "RETURN p.id AS id"
        )
        assert len(result) == 0, (
            f"Paths missing INTRODUCED_IN: {[r['id'] for r in result]}"
        )

    def test_units_exist(self, graph_client):
        """Unit nodes are created."""
        result = graph_client.query("MATCH (u:Unit) RETURN count(u) AS c")
        assert result[0]["c"] == 6

    def test_clusters_exist(self, graph_client):
        """IMASSemanticCluster nodes are created."""
        result = graph_client.query(
            "MATCH (c:IMASSemanticCluster) RETURN count(c) AS c"
        )
        assert result[0]["c"] == 2

    def test_cluster_memberships(self, graph_client):
        """Paths are linked to clusters via IN_CLUSTER."""
        result = graph_client.query(
            "MATCH (p:IMASPath)-[:IN_CLUSTER]->(c:IMASSemanticCluster) "
            "RETURN c.id AS cluster, count(p) AS path_count ORDER BY c.id"
        )
        counts = {r["cluster"]: r["path_count"] for r in result}
        assert counts["cluster_temperature"] == 3
        assert counts["cluster_equilibrium_boundary"] == 3

    def test_path_changes_exist(self, graph_client):
        """IMASPathChange nodes are created."""
        result = graph_client.query("MATCH (c:IMASPathChange) RETURN count(c) AS c")
        assert result[0]["c"] == 1

    def test_path_change_linked(self, graph_client):
        """IMASPathChange is linked to path and version."""
        result = graph_client.query(
            "MATCH (c:IMASPathChange)-[:FOR_IMAS_PATH]->(p:IMASPath), "
            "(c)-[:IN_VERSION]->(v:DDVersion) "
            "RETURN p.id AS path, v.id AS version"
        )
        assert len(result) == 1
        assert result[0]["path"] == "core_profiles/profiles_1d/electrons/pressure"
        assert result[0]["version"] == "4.0.0"

    def test_identifier_schemas_exist(self, graph_client):
        """IdentifierSchema nodes are created and linked."""
        result = graph_client.query(
            "MATCH (p:IMASPath)-[:HAS_IDENTIFIER_SCHEMA]->(s:IdentifierSchema) "
            "RETURN p.id AS path, s.name AS name"
        )
        assert len(result) == 1
        assert result[0]["name"] == "boundary_type"


class TestFixtureGraphQueries:
    """Test common query patterns against the fixture graph."""

    def test_paths_by_ids(self, graph_client):
        """Query paths filtered by IDS name."""
        result = graph_client.query(
            "MATCH (p:IMASPath)-[:IN_IDS]->(i:IDS {id: $ids}) "
            "RETURN p.id AS id ORDER BY p.id",
            ids="equilibrium",
        )
        assert len(result) == 5
        assert all("equilibrium/" in r["id"] for r in result)

    def test_paths_by_version(self, graph_client):
        """Query paths introduced in a specific version."""
        result = graph_client.query(
            "MATCH (p:IMASPath)-[:INTRODUCED_IN]->(v:DDVersion {id: $version}) "
            "RETURN p.id AS id",
            version="4.0.0",
        )
        paths = [r["id"] for r in result]
        assert "core_profiles/profiles_1d/electrons/pressure" in paths

    def test_version_range(self, graph_client):
        """Query min and max DD versions."""
        result = graph_client.query(
            "MATCH (d:DDVersion) "
            "WITH min(d.id) AS min_v, max(d.id) AS max_v "
            "RETURN min_v, max_v"
        )
        assert result[0]["min_v"] == "3.42.0"
        assert result[0]["max_v"] == "4.1.0"

    def test_version_count(self, graph_client):
        """Query total number of DD versions."""
        result = graph_client.query("MATCH (d:DDVersion) RETURN count(d) AS count")
        assert result[0]["count"] == 3

    def test_cluster_path_traversal(self, graph_client):
        """Traverse from cluster to paths."""
        result = graph_client.query(
            "MATCH (c:IMASSemanticCluster {id: $cluster_id})<-[:IN_CLUSTER]-(p:IMASPath) "
            "RETURN p.id AS path ORDER BY p.id",
            cluster_id="cluster_temperature",
        )
        paths = [r["path"] for r in result]
        assert "core_profiles/profiles_1d/electrons/temperature" in paths
        assert "core_profiles/profiles_1d/ion/temperature" in paths

    def test_path_evolution_query(self, graph_client):
        """Query version evolution for a path."""
        result = graph_client.query(
            "MATCH (c:IMASPathChange)-[:FOR_IMAS_PATH]->(p:IMASPath {id: $path}) "
            "RETURN c.change_type AS change, c.from_version AS from_v, "
            "c.to_version AS to_v",
            path="core_profiles/profiles_1d/electrons/pressure",
        )
        assert len(result) == 1
        assert result[0]["change"] == "added"
        assert result[0]["from_v"] == "3.42.0"

    def test_ids_listing(self, graph_client):
        """List all IDS with path counts."""
        result = graph_client.query(
            "MATCH (i:IDS) "
            "OPTIONAL MATCH (p:IMASPath)-[:IN_IDS]->(i) "
            "RETURN i.id AS ids_name, i.documentation AS doc, "
            "count(p) AS actual_paths ORDER BY i.id"
        )
        assert len(result) == 2
        by_name = {r["ids_name"]: r for r in result}
        assert by_name["equilibrium"]["actual_paths"] == 5
        assert by_name["core_profiles"]["actual_paths"] == 4
