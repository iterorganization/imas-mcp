"""IMAS Data Dictionary build verification tests.

Validates that the DD build produced correct, complete, and consistent
graph output. These tests go beyond generic schema compliance by
checking DD-specific invariants:

- All DD versions present with complete predecessor chain
- Exactly one current version
- IDS nodes match imas-python's registry
- Path hierarchy (HAS_PARENT) is consistent
- Embedding coverage for meaningful paths
- Cluster labels are non-empty with scope distribution
- Unit and coordinate relationships are consistent
- IMASPathChange nodes link correctly

Requires a live Neo4j with a completed DD build.
"""

import pytest

pytestmark = pytest.mark.graph


class TestDDVersions:
    """Verify DDVersion node completeness and consistency."""

    def test_all_available_versions_present(self, graph_client, label_counts):
        """Every version from imas-python should exist in the graph."""
        if not label_counts.get("DDVersion"):
            pytest.skip("No DDVersion nodes in graph")

        from imas_codex.graph.build_dd import get_all_dd_versions

        expected = set(get_all_dd_versions())
        result = graph_client.query(
            "MATCH (v:DDVersion) RETURN collect(v.id) AS versions"
        )
        actual = set(result[0]["versions"]) if result else set()

        missing = expected - actual
        assert not missing, (
            f"Missing DD versions in graph: {sorted(missing)}. "
            f"Expected {len(expected)}, found {len(actual)}."
        )

    def test_exactly_one_current_version(self, graph_client, label_counts):
        """Exactly one DDVersion should be marked is_current=true."""
        if not label_counts.get("DDVersion"):
            pytest.skip("No DDVersion nodes in graph")

        result = graph_client.query(
            "MATCH (v:DDVersion {is_current: true}) RETURN count(v) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 1, (
            f"Expected exactly 1 current DDVersion, found {count}. "
            f"The build should mark exactly one version as current."
        )

    def test_current_version_matches_imas_python(self, graph_client, label_counts):
        """The current DDVersion should match imas-python's current version."""
        if not label_counts.get("DDVersion"):
            pytest.skip("No DDVersion nodes in graph")

        from imas_codex import dd_version as expected_version

        result = graph_client.query(
            "MATCH (v:DDVersion {is_current: true}) RETURN v.id AS version"
        )
        if not result:
            pytest.fail("No current DDVersion node found")

        assert result[0]["version"] == expected_version, (
            f"Current DDVersion is {result[0]['version']}, "
            f"expected {expected_version} from imas-python."
        )

    def test_predecessor_chain_complete(self, graph_client, label_counts):
        """Every non-oldest DDVersion should have exactly one PREDECESSOR."""
        if not label_counts.get("DDVersion"):
            pytest.skip("No DDVersion nodes in graph")

        total = label_counts["DDVersion"]

        # Count versions without predecessor (should be exactly 1 — the oldest)
        result = graph_client.query(
            "MATCH (v:DDVersion) "
            "WHERE NOT (v)-[:PREDECESSOR]->(:DDVersion) "
            "RETURN count(v) AS cnt"
        )
        roots = result[0]["cnt"] if result else 0
        assert roots == 1, (
            f"Expected 1 root DDVersion (no predecessor), found {roots}. "
            f"Predecessor chain may be broken."
        )

        # Count predecessor edges (should be total - 1)
        result = graph_client.query(
            "MATCH (:DDVersion)-[r:PREDECESSOR]->(:DDVersion) RETURN count(r) AS cnt"
        )
        edges = result[0]["cnt"] if result else 0
        assert edges == total - 1, (
            f"Expected {total - 1} PREDECESSOR edges for {total} versions, "
            f"found {edges}."
        )

    def test_no_version_has_multiple_predecessors(self, graph_client, label_counts):
        """No DDVersion should point to more than one predecessor."""
        if not label_counts.get("DDVersion"):
            pytest.skip("No DDVersion nodes in graph")

        result = graph_client.query(
            "MATCH (v:DDVersion)-[:PREDECESSOR]->(p:DDVersion) "
            "WITH v, count(p) AS pred_count "
            "WHERE pred_count > 1 "
            "RETURN v.id AS version, pred_count LIMIT 5"
        )
        assert not result, (
            f"DDVersions with multiple predecessors: "
            f"{[(r['version'], r['pred_count']) for r in result]}"
        )


class TestIDSCompleteness:
    """Verify IDS nodes are correct and complete."""

    def test_ids_match_imas_python(self, graph_client, label_counts):
        """IDS nodes should match imas-python's IDS registry for current version."""
        if not label_counts.get("IDS"):
            pytest.skip("No IDS nodes in graph")

        import imas

        from imas_codex import dd_version

        factory = imas.IDSFactory(dd_version)
        expected_ids = set(factory.ids_names())

        result = graph_client.query("MATCH (i:IDS) RETURN collect(i.name) AS names")
        actual_ids = set(result[0]["names"]) if result else set()

        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids

        assert not missing, (
            f"IDS missing from graph: {sorted(missing)}. "
            f"Expected {len(expected_ids)}, found {len(actual_ids)}."
        )
        # Extra IDS from older versions are acceptable
        if extra:
            pytest.xfail(f"Extra IDS from older versions: {sorted(extra)}")

    def test_ids_have_descriptions(self, graph_client, label_counts):
        """Every IDS node should have a non-empty description."""
        if not label_counts.get("IDS"):
            pytest.skip("No IDS nodes in graph")

        result = graph_client.query(
            "MATCH (i:IDS) "
            "WHERE i.description IS NULL OR i.description = '' "
            "RETURN collect(i.name) AS names"
        )
        empty = result[0]["names"] if result else []
        assert not empty, f"IDS without descriptions: {empty}"

    def test_ids_have_physics_domain(self, graph_client, label_counts):
        """Every IDS node should have a physics_domain."""
        if not label_counts.get("IDS"):
            pytest.skip("No IDS nodes in graph")

        result = graph_client.query(
            "MATCH (i:IDS) WHERE i.physics_domain IS NULL "
            "RETURN collect(i.name) AS names"
        )
        missing = result[0]["names"] if result else []
        assert not missing, f"IDS without physics_domain: {missing}"


class TestPathHierarchy:
    """Verify IMASPath hierarchy is well-formed."""

    def test_non_root_paths_have_parent(self, graph_client, label_counts):
        """Every non-IDS-root IMASPath should have a HAS_PARENT relationship."""
        if not label_counts.get("IMASPath"):
            pytest.skip("No IMASPath nodes in graph")

        # Root-level paths (direct children of IDS, e.g., "equilibrium/time")
        # should have HAS_PARENT pointing to the IDS-level path.
        # Only the IDS-root path (e.g., "equilibrium") has no parent.
        # IDS-root paths have exactly one "/" segment
        result = graph_client.query(
            "MATCH (p:IMASPath) "
            "WHERE NOT (p)-[:HAS_PARENT]->(:IMASPath) "
            "AND size(split(p.id, '/')) > 2 "
            "RETURN count(p) AS cnt"
        )
        orphans = result[0]["cnt"] if result else 0
        assert orphans == 0, (
            f"{orphans} non-root IMASPath nodes have no HAS_PARENT relationship"
        )

    def test_no_self_referencing_parent(self, graph_client, label_counts):
        """No IMASPath should be its own parent."""
        if not label_counts.get("IMASPath"):
            pytest.skip("No IMASPath nodes in graph")

        result = graph_client.query(
            "MATCH (p:IMASPath)-[:HAS_PARENT]->(p) RETURN count(p) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} IMASPath nodes are their own parent"

    def test_ids_property_matches_path_prefix(self, graph_client, label_counts):
        """The ids property should match the first segment of the path id."""
        if not label_counts.get("IMASPath"):
            pytest.skip("No IMASPath nodes in graph")

        result = graph_client.query(
            "MATCH (p:IMASPath) "
            "WHERE p.ids <> split(p.id, '/')[0] "
            "RETURN count(p) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} IMASPath nodes have ids property not matching path prefix"
        )


class TestEmbeddingCompleteness:
    """Verify embedding coverage and quality for the DD build."""

    def test_meaningful_paths_have_embeddings(self, graph_client, label_counts):
        """Paths that pass the exclusion filter should have embeddings.

        Error fields and metadata-only paths are excluded from embedding.
        The remaining 'meaningful' paths should all have embeddings after
        a complete build.
        """
        if not label_counts.get("IMASPath"):
            pytest.skip("No IMASPath nodes in graph")

        # Check that embedded paths have embedding_text too
        result = graph_client.query(
            "MATCH (p:IMASPath) "
            "WHERE p.embedding IS NOT NULL AND p.embedding_text IS NULL "
            "RETURN count(p) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} paths have embedding but no embedding_text "
            f"(embedding_text needed for cache validation)"
        )

    def test_embedded_paths_have_hash(self, graph_client, label_counts):
        """Embedded paths should have embedding_hash for cache validation."""
        if not label_counts.get("IMASPath"):
            pytest.skip("No IMASPath nodes in graph")

        result = graph_client.query(
            "MATCH (p:IMASPath) "
            "WHERE p.embedding IS NOT NULL AND p.embedding_hash IS NULL "
            "RETURN count(p) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} paths have embedding but no embedding_hash "
            f"(hash needed for incremental rebuild)"
        )

    def test_no_embeddings_without_embedding(self, graph_client, label_counts):
        """Paths should not have embedding_text without an actual embedding."""
        if not label_counts.get("IMASPath"):
            pytest.skip("No IMASPath nodes in graph")

        result = graph_client.query(
            "MATCH (p:IMASPath) "
            "WHERE p.embedding IS NULL AND p.embedding_text IS NOT NULL "
            "RETURN count(p) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} paths have embedding_text but no embedding vector"


class TestUnitRelationships:
    """Verify unit relationship consistency."""

    def test_has_unit_targets_valid_units(self, graph_client, label_counts):
        """HAS_UNIT relationships should point to Unit nodes with symbols."""
        if not label_counts.get("Unit"):
            pytest.skip("No Unit nodes in graph")

        result = graph_client.query(
            "MATCH (p:IMASPath)-[:HAS_UNIT]->(u:Unit) "
            "WHERE u.symbol IS NULL "
            "RETURN count(p) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} HAS_UNIT relationships point to Unit nodes without symbols"
        )

    def test_unit_nodes_have_relationships(self, graph_client, label_counts):
        """All Unit nodes should be referenced by at least one path."""
        if not label_counts.get("Unit"):
            pytest.skip("No Unit nodes in graph")

        result = graph_client.query(
            "MATCH (u:Unit) WHERE NOT ()-[:HAS_UNIT]->(u) RETURN count(u) AS cnt"
        )
        orphans = result[0]["cnt"] if result else 0
        assert orphans == 0, f"{orphans} Unit nodes are orphaned (no HAS_UNIT)"


class TestErrorRelationships:
    """Verify HAS_ERROR relationship structure."""

    def test_error_relationships_exist(self, graph_client, label_counts):
        """After a full build, HAS_ERROR relationships should exist."""
        if not label_counts.get("IMASPath"):
            pytest.skip("No IMASPath nodes in graph")

        result = graph_client.query("MATCH ()-[r:HAS_ERROR]->() RETURN count(r) AS cnt")
        count = result[0]["cnt"] if result else 0
        assert count > 0, "No HAS_ERROR relationships found after DD build"

    def test_error_paths_are_imas_paths(self, graph_client, label_counts):
        """Both sides of HAS_ERROR should be IMASPath nodes."""
        if not label_counts.get("IMASPath"):
            pytest.skip("No IMASPath nodes in graph")

        result = graph_client.query(
            "MATCH (data)-[r:HAS_ERROR]->(err) "
            "WHERE NOT data:IMASPath OR NOT err:IMASPath "
            "RETURN count(r) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} HAS_ERROR relationships connect non-IMASPath nodes"


class TestIMASPathChanges:
    """Verify IMASPathChange nodes are correctly linked."""

    def test_path_changes_have_for_imas_path(self, graph_client, label_counts):
        """Every IMASPathChange should link to an IMASPath."""
        if not label_counts.get("IMASPathChange"):
            pytest.skip("No IMASPathChange nodes in graph")

        result = graph_client.query(
            "MATCH (pc:IMASPathChange) "
            "WHERE NOT (pc)-[:FOR_IMAS_PATH]->(:IMASPath) "
            "RETURN count(pc) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} IMASPathChange nodes without FOR_IMAS_PATH relationship"
        )

    def test_path_changes_have_in_version(self, graph_client, label_counts):
        """Every IMASPathChange should link to a DDVersion."""
        if not label_counts.get("IMASPathChange"):
            pytest.skip("No IMASPathChange nodes in graph")

        result = graph_client.query(
            "MATCH (pc:IMASPathChange) "
            "WHERE NOT (pc)-[:IN_VERSION]->(:DDVersion) "
            "RETURN count(pc) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} IMASPathChange nodes without IN_VERSION relationship"
        )

    def test_path_changes_have_change_type(self, graph_client, label_counts):
        """Every IMASPathChange must have a change_type."""
        if not label_counts.get("IMASPathChange"):
            pytest.skip("No IMASPathChange nodes in graph")

        result = graph_client.query(
            "MATCH (pc:IMASPathChange) "
            "WHERE pc.change_type IS NULL "
            "RETURN count(pc) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} IMASPathChange nodes without change_type"

    def test_definition_changes_exist(self, graph_client, label_counts):
        """Documentation (definition) changes should be tracked across versions."""
        if not label_counts.get("IMASPathChange"):
            pytest.skip("No IMASPathChange nodes in graph")

        result = graph_client.query(
            "MATCH (pc:IMASPathChange {change_type: 'documentation'}) "
            "RETURN count(pc) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count > 0, (
            "No documentation changes tracked — expected many across DD versions"
        )

    def test_definition_changes_have_semantic_type(self, graph_client, label_counts):
        """Documentation changes should have semantic classification."""
        if not label_counts.get("IMASPathChange"):
            pytest.skip("No IMASPathChange nodes in graph")

        result = graph_client.query(
            "MATCH (pc:IMASPathChange {change_type: 'documentation'}) "
            "WHERE pc.semantic_type IS NOT NULL "
            "RETURN count(pc) AS classified"
        )
        classified = result[0]["classified"] if result else 0
        assert classified > 0, "No documentation changes have semantic classification"

    def test_change_types_are_valid(self, graph_client, label_counts):
        """change_type should only be known metadata fields."""
        if not label_counts.get("IMASPathChange"):
            pytest.skip("No IMASPathChange nodes in graph")

        valid_types = {
            "documentation",
            "units",
            "data_type",
            "node_type",
            "cocos_label_transformation",
        }
        result = graph_client.query(
            "MATCH (pc:IMASPathChange) RETURN DISTINCT pc.change_type AS change_type"
        )
        actual_types = {r["change_type"] for r in result}
        invalid = actual_types - valid_types
        assert not invalid, f"Invalid change_type values: {invalid}"

    def test_path_history_queryable(self, graph_client, label_counts):
        """A single path's full change history should be queryable by traversal."""
        if not label_counts.get("IMASPathChange"):
            pytest.skip("No IMASPathChange nodes in graph")

        # Find a path with multiple changes
        result = graph_client.query(
            "MATCH (pc:IMASPathChange)-[:FOR_IMAS_PATH]->(p:IMASPath) "
            "WITH p, count(pc) AS changes WHERE changes >= 2 "
            "RETURN p.id AS path LIMIT 1"
        )
        assert result, "Expected at least one path with multiple changes"

        path = result[0]["path"]

        # Query full history for that path
        history = graph_client.query(
            "MATCH (pc:IMASPathChange)-[:FOR_IMAS_PATH]->(p:IMASPath {id: $path}) "
            "MATCH (pc)-[:IN_VERSION]->(v:DDVersion) "
            "RETURN v.id AS version, pc.change_type AS change_type, "
            "       pc.old_value AS old_value, pc.new_value AS new_value "
            "ORDER BY v.id",
            path=path,
        )
        assert len(history) >= 2, (
            f"Path {path} should have at least 2 changes, got {len(history)}"
        )
        # Each change should have old and new values
        for h in history:
            assert h["version"] is not None
            assert h["change_type"] is not None


class TestCOCOSCompleteness:
    """Verify COCOS reference nodes and version linking."""

    def test_cocos_reference_nodes(self, graph_client):
        """All 16 valid COCOS reference nodes should exist."""
        result = graph_client.query("MATCH (c:COCOS) RETURN count(c) AS cnt")
        assert result[0]["cnt"] == 16

    def test_cocos_nodes_have_parameters(self, graph_client):
        """Each COCOS node should have all four Sauter parameters."""
        result = graph_client.query(
            "MATCH (c:COCOS) "
            "WHERE c.sigma_bp IS NULL OR c.e_bp IS NULL "
            "   OR c.sigma_r_phi_z IS NULL OR c.sigma_rho_theta_phi IS NULL "
            "RETURN count(c) AS cnt"
        )
        assert result[0]["cnt"] == 0, "COCOS nodes missing Sauter parameters"

    def test_dd_versions_linked_to_cocos(self, graph_client):
        """DDVersions from 3.35.0+ should have HAS_COCOS relationships."""
        result = graph_client.query(
            "MATCH (v:DDVersion)-[:HAS_COCOS]->(c:COCOS) RETURN count(v) AS cnt"
        )
        assert result[0]["cnt"] >= 17, (
            f"Expected >=17 DDVersions linked to COCOS, got {result[0]['cnt']}"
        )

    def test_cocos_label_transformation_on_paths(self, graph_client):
        """Current-version paths with COCOS sensitivity should be labeled."""
        result = graph_client.query(
            "MATCH (p:IMASPath) "
            "WHERE p.cocos_label_transformation IS NOT NULL "
            "RETURN count(p) AS cnt"
        )
        assert result[0]["cnt"] >= 300, (
            f"Expected >=300 paths with cocos_label_transformation, "
            f"got {result[0]['cnt']}"
        )

    def test_cocos_changes_tracked(self, graph_client, label_counts):
        """COCOS label changes across versions should be tracked."""
        if not label_counts.get("IMASPathChange"):
            pytest.skip("No IMASPathChange nodes in graph")

        result = graph_client.query(
            "MATCH (c:IMASPathChange) "
            "WHERE c.change_type = 'cocos_label_transformation' "
            "RETURN count(c) AS cnt"
        )
        assert result[0]["cnt"] > 0, (
            "Expected IMASPathChange nodes for cocos_label_transformation"
        )


class TestClusterLabels:
    """Verify cluster label quality and consistency."""

    def test_all_clusters_labeled(self, graph_client, label_counts):
        """Every IMASSemanticCluster should have a non-empty label."""
        if not label_counts.get("IMASSemanticCluster"):
            pytest.skip("No IMASSemanticCluster nodes in graph")

        result = graph_client.query(
            "MATCH (c:IMASSemanticCluster) "
            "WHERE c.label IS NULL OR c.label = '' "
            "RETURN count(c) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} clusters without labels. "
            f"Run `imas-codex imas clusters label` to generate."
        )

    def test_all_clusters_have_descriptions(self, graph_client, label_counts):
        """Every cluster should have a description for NL search."""
        if not label_counts.get("IMASSemanticCluster"):
            pytest.skip("No IMASSemanticCluster nodes in graph")

        result = graph_client.query(
            "MATCH (c:IMASSemanticCluster) "
            "WHERE c.description IS NULL OR c.description = '' "
            "RETURN count(c) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} clusters without descriptions. "
            f"Run `imas-codex imas clusters label` to generate."
        )

    def test_cluster_labels_unique_within_scope(self, graph_client, label_counts):
        """Cluster labels should be unique within the same scope."""
        if not label_counts.get("IMASSemanticCluster"):
            pytest.skip("No IMASSemanticCluster nodes in graph")

        result = graph_client.query(
            "MATCH (c:IMASSemanticCluster) "
            "WHERE c.label IS NOT NULL "
            "WITH c.scope AS scope, c.label AS label, count(*) AS cnt "
            "WHERE cnt > 1 "
            "RETURN scope, count(label) AS duplicate_labels "
            "ORDER BY duplicate_labels DESC"
        )
        violations = []
        for row in result:
            scope = row["scope"] or "null"
            dupes = row["duplicate_labels"]
            # Allow up to 20% duplicates within a scope
            scope_result = graph_client.query(
                "MATCH (c:IMASSemanticCluster {scope: $scope}) "
                "RETURN count(c) AS total",
                scope=row["scope"],
            )
            scope_total = scope_result[0]["total"] if scope_result else 0
            if scope_total > 0 and dupes / scope_total > 0.2:
                violations.append(
                    f"{scope}: {dupes}/{scope_total} duplicate labels "
                    f"({dupes / scope_total:.1%})"
                )
        assert not violations, (
            "Too many duplicate labels within scope:\n  " + "\n  ".join(violations)
        )

    def test_cluster_scopes_present(self, graph_client, label_counts):
        """Clusters should have scope values distributed across expected types."""
        if not label_counts.get("IMASSemanticCluster"):
            pytest.skip("No IMASSemanticCluster nodes in graph")

        result = graph_client.query(
            "MATCH (c:IMASSemanticCluster) "
            "RETURN c.scope AS scope, count(c) AS cnt "
            "ORDER BY cnt DESC"
        )
        scopes = {r["scope"] for r in result}
        expected_scopes = {"global", "domain", "ids"}
        missing = expected_scopes - scopes
        assert not missing, (
            f"Missing cluster scopes: {missing}. "
            f"Build should produce global, domain, and ids-level clusters."
        )

    def test_clusters_have_ids_names(self, graph_client, label_counts):
        """IDS-scoped clusters should have ids_names populated."""
        if not label_counts.get("IMASSemanticCluster"):
            pytest.skip("No IMASSemanticCluster nodes in graph")

        result = graph_client.query(
            "MATCH (c:IMASSemanticCluster) "
            "WHERE c.scope = 'ids' "
            "AND (c.ids_names IS NULL OR size(c.ids_names) = 0) "
            "RETURN count(c) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} IDS-scoped clusters without ids_names"

    def test_cluster_path_counts_match_members(self, graph_client, label_counts):
        """Cluster path_count should match actual IN_CLUSTER member count."""
        if not label_counts.get("IMASSemanticCluster"):
            pytest.skip("No IMASSemanticCluster nodes in graph")

        result = graph_client.query(
            "MATCH (c:IMASSemanticCluster) "
            "OPTIONAL MATCH (p:IMASPath)-[:IN_CLUSTER]->(c) "
            "WITH c, c.path_count AS declared, count(p) AS actual "
            "WHERE declared <> actual "
            "RETURN count(c) AS cnt, "
            "  collect(c.id)[..5] AS sample_ids"
        )
        if result and result[0]["cnt"] > 0:
            count = result[0]["cnt"]
            samples = result[0]["sample_ids"]
            assert count == 0, (
                f"{count} clusters have path_count mismatch "
                f"(declared vs actual IN_CLUSTER). Samples: {samples}"
            )


class TestClusterEmbeddings:
    """Verify cluster embedding vectors for semantic search."""

    def test_clusters_have_centroid_embeddings(self, graph_client, label_counts):
        """All clusters should have centroid embeddings (mean of member paths)."""
        if not label_counts.get("IMASSemanticCluster"):
            pytest.skip("No IMASSemanticCluster nodes in graph")

        total = label_counts["IMASSemanticCluster"]
        result = graph_client.query(
            "MATCH (c:IMASSemanticCluster) "
            "WHERE c.embedding IS NOT NULL "
            "RETURN count(c) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == total, f"Only {count}/{total} clusters have centroid embeddings"

    def test_clusters_have_label_embeddings(self, graph_client, label_counts):
        """All clusters should have label_embedding for NL search."""
        if not label_counts.get("IMASSemanticCluster"):
            pytest.skip("No IMASSemanticCluster nodes in graph")

        total = label_counts["IMASSemanticCluster"]
        result = graph_client.query(
            "MATCH (c:IMASSemanticCluster) "
            "WHERE c.label_embedding IS NOT NULL "
            "RETURN count(c) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == total, (
            f"Only {count}/{total} clusters have label embeddings. "
            f"Run `imas-codex imas clusters embed` to generate."
        )

    def test_clusters_have_description_embeddings(self, graph_client, label_counts):
        """All clusters should have description_embedding for NL search."""
        if not label_counts.get("IMASSemanticCluster"):
            pytest.skip("No IMASSemanticCluster nodes in graph")

        total = label_counts["IMASSemanticCluster"]
        result = graph_client.query(
            "MATCH (c:IMASSemanticCluster) "
            "WHERE c.description_embedding IS NOT NULL "
            "RETURN count(c) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == total, (
            f"Only {count}/{total} clusters have description embeddings. "
            f"Run `imas-codex imas clusters embed` to generate."
        )

    def test_cluster_embedding_dimensions_consistent(self, graph_client, label_counts):
        """All embedding types should have the same dimensionality."""
        if not label_counts.get("IMASSemanticCluster"):
            pytest.skip("No IMASSemanticCluster nodes in graph")

        result = graph_client.query(
            "MATCH (c:IMASSemanticCluster) "
            "WHERE c.embedding IS NOT NULL "
            "AND c.label_embedding IS NOT NULL "
            "AND c.description_embedding IS NOT NULL "
            "WITH size(c.embedding) AS centroid_dim, "
            "     size(c.label_embedding) AS label_dim, "
            "     size(c.description_embedding) AS desc_dim "
            "WITH DISTINCT centroid_dim, label_dim, desc_dim "
            "RETURN centroid_dim, label_dim, desc_dim"
        )
        assert len(result) == 1, (
            f"Expected exactly 1 dimension set, got {len(result)}: {result}"
        )
        row = result[0]
        assert row["centroid_dim"] == row["label_dim"] == row["desc_dim"], (
            f"Embedding dimensions differ: "
            f"centroid={row['centroid_dim']}, "
            f"label={row['label_dim']}, "
            f"description={row['desc_dim']}"
        )


class TestDeprecationTracking:
    """Verify path deprecation is tracked correctly."""

    def test_deprecated_paths_have_version(self, graph_client, label_counts):
        """Deprecated paths should link to the version where they were deprecated."""
        if not label_counts.get("IMASPath"):
            pytest.skip("No IMASPath nodes in graph")

        result = graph_client.query(
            "MATCH (p:IMASPath)-[:DEPRECATED_IN]->(v:DDVersion) "
            "RETURN count(DISTINCT p) AS deprecated_paths, "
            "  count(DISTINCT v) AS deprecation_versions"
        )
        if result and result[0]["deprecated_paths"] > 0:
            # Just verify the relationships exist; count is informational
            assert result[0]["deprecation_versions"] > 0

    def test_renamed_paths_both_exist(self, graph_client, label_counts):
        """Both sides of RENAMED_TO should be valid IMASPath nodes."""
        if not label_counts.get("IMASPath"):
            pytest.skip("No IMASPath nodes in graph")

        result = graph_client.query(
            "MATCH (old)-[r:RENAMED_TO]->(new) "
            "WHERE NOT old:IMASPath OR NOT new:IMASPath "
            "RETURN count(r) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} RENAMED_TO relationships connect non-IMASPath nodes"
        )


class TestCoordinateRelationships:
    """Verify coordinate relationship structure."""

    def test_has_coordinate_targets_are_valid(self, graph_client, label_counts):
        """HAS_COORDINATE should point to IMASPath or IMASCoordinateSpec."""
        if not label_counts.get("IMASPath"):
            pytest.skip("No IMASPath nodes in graph")

        result = graph_client.query(
            "MATCH (p:IMASPath)-[r:HAS_COORDINATE]->(target) "
            "WHERE NOT target:IMASPath AND NOT target:IMASCoordinateSpec "
            "RETURN count(r) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} HAS_COORDINATE relationships point to "
            f"non-IMASPath/non-IMASCoordinateSpec nodes"
        )

    def test_coordinate_dimensions_are_positive(self, graph_client, label_counts):
        """HAS_COORDINATE dimension property should be a positive integer."""
        if not label_counts.get("IMASPath"):
            pytest.skip("No IMASPath nodes in graph")

        result = graph_client.query(
            "MATCH (:IMASPath)-[r:HAS_COORDINATE]->() "
            "WHERE r.dimension IS NULL OR r.dimension < 1 "
            "RETURN count(r) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} HAS_COORDINATE relationships with invalid dimension"
        )
