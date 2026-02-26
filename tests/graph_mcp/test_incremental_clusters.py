"""Tests for Phase 4: Incremental Cluster Sync.

Validates that:
- _import_clusters uses MERGE (not DETACH DELETE + CREATE)
- Labels persist in graph across clustering runs
- _sync_labels_from_cache is no longer called during import
- clusters label writes labels to graph nodes
- Stale clusters are removed incrementally
"""

import hashlib
import inspect
import re

import pytest


class TestImportClustersIncremental:
    """Validate that _import_clusters uses incremental MERGE pattern."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        from imas_codex.graph.build_dd import _import_clusters

        self.source = inspect.getsource(_import_clusters)

    def test_uses_merge_not_create(self):
        """Cluster nodes created via MERGE, not CREATE."""
        assert "MERGE (n:IMASSemanticCluster" in self.source
        # Should not use CREATE for cluster nodes
        assert "CREATE (n:IMASSemanticCluster" not in self.source

    def test_no_bulk_detach_delete(self):
        """No DETACH DELETE of all clusters — only stale ones."""
        # Should NOT have a pattern like MATCH (c:IMASSemanticCluster) DETACH DELETE c
        # without a WHERE or UNWIND filter
        lines = self.source.split("\n")
        for i, line in enumerate(lines):
            if "DETACH DELETE" in line:
                # Check context: must have UNWIND $ids or be stale-specific
                context = "\n".join(lines[max(0, i - 5) : i + 1])
                assert "UNWIND $ids" in context or "stale" in context.lower(), (
                    f"Found unguarded DETACH DELETE at line {i}: {line}"
                )

    def test_no_sync_labels_from_cache_call(self):
        """_sync_labels_from_cache is no longer called in _import_clusters."""
        assert "_sync_labels_from_cache" not in self.source

    def test_merge_preserves_labels(self):
        """MERGE SET does not overwrite label or description properties."""
        # Find the MERGE...SET block for cluster nodes
        merge_pattern = re.search(
            r"MERGE \(n:IMASSemanticCluster.*?\n(.*?SET.*?)(?=\n\s*\"\"\")",
            self.source,
            re.DOTALL,
        )
        assert merge_pattern, "Could not find MERGE...SET block"
        set_block = merge_pattern.group(1)
        # label and description should NOT appear in SET
        assert "n.label" not in set_block, "MERGE should not SET label"
        assert "n.description" not in set_block, "MERGE should not SET description"

    def test_content_hash_ids(self):
        """Clusters use content-hash-based IDs for stable identity."""
        assert "_compute_cluster_content_hash" in self.source

    def test_stale_detection(self):
        """Stale clusters detected by comparing existing vs new IDs."""
        assert "existing_ids - new_cluster_ids" in self.source

    def test_membership_refresh(self):
        """IN_CLUSTER relationships are refreshed for updated clusters."""
        assert "IN_CLUSTER" in self.source
        # Should delete old relationships for updated clusters
        assert "DELETE r" in self.source


class TestContentHashDeterminism:
    """Validate content hash produces stable, deterministic IDs."""

    def test_same_paths_same_hash(self):
        from imas_codex.graph.build_dd import _compute_cluster_content_hash

        paths = [
            "equilibrium/time_slice/profiles_1d/psi",
            "equilibrium/time_slice/profiles_1d/q",
        ]
        h1 = _compute_cluster_content_hash(paths, "global")
        h2 = _compute_cluster_content_hash(paths, "global")
        assert h1 == h2

    def test_different_scope_different_hash(self):
        from imas_codex.graph.build_dd import _compute_cluster_content_hash

        paths = ["equilibrium/time_slice/profiles_1d/psi"]
        h_global = _compute_cluster_content_hash(paths, "global")
        h_ids = _compute_cluster_content_hash(paths, "ids")
        assert h_global != h_ids

    def test_order_independent(self):
        from imas_codex.graph.build_dd import _compute_cluster_content_hash

        paths1 = ["a/b", "c/d"]
        paths2 = ["c/d", "a/b"]
        # Function expects sorted paths — caller must sort
        h1 = _compute_cluster_content_hash(sorted(paths1), "global")
        h2 = _compute_cluster_content_hash(sorted(paths2), "global")
        assert h1 == h2

    def test_hash_length(self):
        from imas_codex.graph.build_dd import _compute_cluster_content_hash

        h = _compute_cluster_content_hash(["a/b"], "global")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestSyncLabelsFromCacheRemoved:
    """Validate _sync_labels_from_cache is no longer used in import pipeline."""

    def test_function_still_exists_but_not_called(self):
        """The function may still exist but should not be called from _import_clusters."""
        from imas_codex.graph.build_dd import _import_clusters

        source = inspect.getsource(_import_clusters)
        assert "_sync_labels_from_cache" not in source

    def test_labels_persist_via_merge(self):
        """Documentation confirms labels persist via MERGE."""
        from imas_codex.graph.build_dd import _import_clusters

        source = inspect.getsource(_import_clusters)
        # Should have a comment about labels persisting
        assert "persist" in source.lower() or "preserve" in source.lower()


class TestClustersLabelGraphSync:
    """Validate clusters label command writes to graph."""

    def test_sync_labels_to_graph_function_exists(self):
        """_sync_labels_to_graph helper is available."""
        from imas_codex.cli.clusters import _sync_labels_to_graph

        assert callable(_sync_labels_to_graph)

    def test_sync_labels_to_graph_uses_cypher(self):
        """Function writes labels via Cypher MATCH/SET."""
        from imas_codex.cli.clusters import _sync_labels_to_graph

        source = inspect.getsource(_sync_labels_to_graph)
        assert "MATCH (c:IMASSemanticCluster" in source
        assert "SET c.label" in source
        assert "c.description" in source

    def test_sync_labels_to_graph_batched(self):
        """Labels are written in batches to avoid query size limits."""
        from imas_codex.cli.clusters import _sync_labels_to_graph

        source = inspect.getsource(_sync_labels_to_graph)
        assert "UNWIND $batch" in source

    def test_sync_labels_to_graph_handles_no_graph(self):
        """Gracefully handles case when graph is not available."""
        from imas_codex.cli.clusters import _sync_labels_to_graph

        source = inspect.getsource(_sync_labels_to_graph)
        assert "not available" in source.lower()
