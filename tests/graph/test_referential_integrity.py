"""Referential integrity tests.

Verifies that relationships in the graph point to existing nodes
and that required structural links are present.
"""

import pytest

pytestmark = pytest.mark.graph


class TestFacilityOwnership:
    """Facility-owned nodes must link to a valid Facility."""

    def test_facility_id_edges_exist(self, graph_client, schema, graph_labels):
        """Every node with required facility_id must have an AT_FACILITY edge."""
        if "Facility" not in graph_labels:
            pytest.skip("No Facility nodes — DD-only graph")
        violations = []
        for label in sorted(graph_labels):
            if label.startswith("_"):
                continue
            if label not in schema.node_labels:
                continue

            required = schema.get_required_fields(label)
            if "facility_id" not in required:
                continue

            result = graph_client.query(
                f"MATCH (n:{label}) "
                f"WHERE n.facility_id IS NOT NULL "
                f"AND NOT (n)-[:AT_FACILITY]->(:Facility) "
                f"RETURN count(n) AS cnt"
            )
            count = result[0]["cnt"] if result else 0
            if count > 0:
                violations.append(f"{label}: {count} nodes without AT_FACILITY edge")

        assert not violations, "Missing AT_FACILITY relationships:\n  " + "\n  ".join(
            violations
        )

    def test_facility_id_property_matches_edge(
        self, graph_client, schema, graph_labels
    ):
        """The facility_id property must match the linked Facility node's id."""
        violations = []
        for label in sorted(graph_labels):
            if label.startswith("_"):
                continue
            if label not in schema.node_labels:
                continue

            required = schema.get_required_fields(label)
            if "facility_id" not in required:
                continue

            result = graph_client.query(
                f"MATCH (n:{label})-[:AT_FACILITY]->(f:Facility) "
                f"WHERE n.facility_id <> f.id "
                f"RETURN count(n) AS cnt"
            )
            count = result[0]["cnt"] if result else 0
            if count > 0:
                violations.append(f"{label}: {count} nodes with mismatched facility_id")

        assert not violations, (
            "facility_id property doesn't match linked Facility:\n  "
            + "\n  ".join(violations)
        )


class TestWikiHierarchy:
    """Wiki content hierarchy integrity."""

    def test_wiki_chunks_have_parent(self, graph_client, label_counts):
        """Every WikiChunk must have a parent via HAS_CHUNK (WikiPage or Document)."""
        if not label_counts.get("WikiChunk"):
            pytest.skip("No WikiChunk nodes in graph")

        result = graph_client.query(
            "MATCH (c:WikiChunk) WHERE NOT ()-[:HAS_CHUNK]->(c) RETURN count(c) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} WikiChunk nodes without parent (WikiPage or Document)"
        )

    def test_wiki_documents_have_parent_page(self, graph_client, label_counts):
        """Documents discovered via page crawling should link to a parent WikiPage.

        Bulk-discovered documents (bulk_discovered=true) may not have parent
        page links until page crawling enriches them. Only non-bulk documents
        are expected to have HAS_DOCUMENT relationships.
        """
        if not label_counts.get("Document"):
            pytest.skip("No Document nodes in graph")

        # Check non-bulk documents (should always have parent)
        result = graph_client.query(
            "MATCH (a:Document) "
            "WHERE (a.bulk_discovered IS NULL OR a.bulk_discovered = false) "
            "AND NOT (:WikiPage)-[:HAS_DOCUMENT]->(a) "
            "RETURN count(a) AS cnt"
        )
        non_bulk_orphans = result[0]["cnt"] if result else 0

        # Check overall linkage rate for all documents (informational)
        result = graph_client.query(
            "MATCH (a:Document) "
            "RETURN count(a) AS total, "
            "count(CASE WHEN (:WikiPage)-[:HAS_DOCUMENT]->(a) THEN 1 END) AS linked"
        )
        total = result[0]["total"] if result else 0
        linked = result[0]["linked"] if result else 0

        assert non_bulk_orphans == 0, (
            f"{non_bulk_orphans} non-bulk Document nodes without parent WikiPage. "
            f"Overall: {linked}/{total} documents linked to pages."
        )


class TestCodeHierarchy:
    """Source code hierarchy integrity."""

    def test_code_chunks_have_code_example(self, graph_client, label_counts):
        """Every CodeChunk must belong to a CodeExample via HAS_CHUNK."""
        if not label_counts.get("CodeChunk"):
            pytest.skip("No CodeChunk nodes in graph")

        result = graph_client.query(
            "MATCH (c:CodeChunk) "
            "WHERE NOT (:CodeExample)-[:HAS_CHUNK]->(c) "
            "RETURN count(c) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} CodeChunk nodes without parent CodeExample"


class TestIMASLinks:
    """IMAS data dictionary structural integrity."""

    def test_imas_paths_have_ids_link(self, graph_client, label_counts):
        """Every IMASNode must link to an IDS node."""
        if not label_counts.get("IMASNode"):
            pytest.skip("No IMASNode nodes in graph")

        result = graph_client.query(
            "MATCH (p:IMASNode) WHERE NOT (p)-[:IN_IDS]->(:IDS) RETURN count(p) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} IMASNode nodes without IDS link"

    def test_imas_paths_have_dd_version(self, graph_client, label_counts):
        """Every IMASNode must have an INTRODUCED_IN relationship to DDVersion."""
        if not label_counts.get("IMASNode"):
            pytest.skip("No IMASNode nodes in graph")

        result = graph_client.query(
            "MATCH (p:IMASNode) "
            "WHERE NOT (p)-[:INTRODUCED_IN]->(:DDVersion) "
            "RETURN count(p) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} IMASNode nodes without INTRODUCED_IN link"

    def test_dd_version_chain_has_no_cycles(self, graph_client, label_counts):
        """DDVersion PREDECESSOR chain must be acyclic."""
        if not label_counts.get("DDVersion"):
            pytest.skip("No DDVersion nodes in graph")

        # A cycle would mean a version is its own ancestor
        result = graph_client.query("MATCH (v:DDVersion) RETURN count(v) AS total")
        total = result[0]["total"] if result else 0

        # Count distinct versions reachable from chain traversal
        result = graph_client.query(
            "MATCH path = (v:DDVersion)-[:HAS_PREDECESSOR*0..100]->(root:DDVersion) "
            "WHERE NOT (root)-[:HAS_PREDECESSOR]->() "
            "WITH v, length(path) AS depth "
            "RETURN max(depth) AS max_depth, count(DISTINCT v) AS chain_count"
        )
        if result:
            chain_count = result[0]["chain_count"]
            assert chain_count == total, (
                f"Version chain covers {chain_count}/{total} DDVersion nodes. "
                f"Possible cycle or disconnected versions."
            )

    def test_ids_nodes_have_dd_version(self, graph_client, label_counts):
        """Every IDS node should reference a DDVersion."""
        if not label_counts.get("IDS"):
            pytest.skip("No IDS nodes in graph")

        result = graph_client.query(
            "MATCH (i:IDS) WHERE i.dd_version IS NULL RETURN count(i) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} IDS nodes without dd_version"


class TestRelationshipDirection:
    """Verify relationships go in the correct direction."""

    def test_facility_id_direction(self, graph_client, graph_relationship_types):
        """AT_FACILITY edges point FROM entity TO Facility, never reversed."""
        if "AT_FACILITY" not in graph_relationship_types:
            pytest.skip("No AT_FACILITY relationships in graph")

        result = graph_client.query(
            "MATCH (f:Facility)-[:AT_FACILITY]->(n) RETURN count(f) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} reversed AT_FACILITY edges (Facility->something). "
            f"Should always be entity->Facility."
        )

    def test_has_chunk_direction(self, graph_client, graph_relationship_types):
        """HAS_CHUNK edges point FROM parent TO chunk."""
        if "HAS_CHUNK" not in graph_relationship_types:
            pytest.skip("No HAS_CHUNK relationships in graph")

        # Chunks should not be the source of HAS_CHUNK
        result = graph_client.query(
            "MATCH (c:WikiChunk)-[:HAS_CHUNK]->(n) RETURN count(c) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} reversed HAS_CHUNK edges from WikiChunk"
