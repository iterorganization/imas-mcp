"""Tests for build_dd hash-based idempotency helpers."""

from unittest.mock import MagicMock

from imas_codex.graph.build_dd import _check_graph_up_to_date, _compute_build_hash


class TestComputeBuildHash:
    """Tests for _compute_build_hash."""

    def test_deterministic(self):
        """Same inputs → same hash."""
        h1 = _compute_build_hash(["4.0.0", "4.1.0"], None, "model-a", True, True)
        h2 = _compute_build_hash(["4.0.0", "4.1.0"], None, "model-a", True, True)
        assert h1 == h2

    def test_order_independent_versions(self):
        """Version order doesn't change hash (sorted internally)."""
        h1 = _compute_build_hash(["4.1.0", "4.0.0"], None, "m", True, True)
        h2 = _compute_build_hash(["4.0.0", "4.1.0"], None, "m", True, True)
        assert h1 == h2

    def test_different_versions_different_hash(self):
        h1 = _compute_build_hash(["4.0.0"], None, "m", True, True)
        h2 = _compute_build_hash(["4.1.0"], None, "m", True, True)
        assert h1 != h2

    def test_different_ids_filter_different_hash(self):
        h1 = _compute_build_hash(["4.0.0"], {"equilibrium"}, "m", True, True)
        h2 = _compute_build_hash(["4.0.0"], {"core_profiles"}, "m", True, True)
        assert h1 != h2

    def test_none_vs_empty_ids_filter(self):
        h1 = _compute_build_hash(["4.0.0"], None, "m", True, True)
        h2 = _compute_build_hash(["4.0.0"], set(), "m", True, True)
        # Both produce empty string for ids_filter → same hash
        assert h1 == h2

    def test_clusters_flag_changes_hash(self):
        h1 = _compute_build_hash(["4.0.0"], None, "m", True, True)
        h2 = _compute_build_hash(["4.0.0"], None, "m", False, True)
        assert h1 != h2

    def test_embeddings_flag_changes_hash(self):
        h1 = _compute_build_hash(["4.0.0"], None, "m", True, True)
        h2 = _compute_build_hash(["4.0.0"], None, "m", True, False)
        assert h1 != h2

    def test_returns_16_char_hex(self):
        h = _compute_build_hash(["4.0.0"], None, "m", True, True)
        assert len(h) == 16
        int(h, 16)  # Should be valid hex


class TestCheckGraphUpToDate:
    """Tests for _check_graph_up_to_date."""

    def _make_client(
        self,
        hash_response=None,
        ver_response=None,
        emb_response=None,
        cluster_response=None,
    ):
        """Build a mock GraphClient with canned query responses."""
        client = MagicMock()

        def query_side_effect(cypher, **kwargs):
            if "is_current" in cypher:
                return hash_response if hash_response is not None else []
            if "collect(d.id)" in cypher:
                return ver_response if ver_response is not None else []
            if "p.embedding" in cypher:
                return emb_response or []
            if "IMASSemanticCluster" in cypher:
                return cluster_response or []
            return []

        client.query.side_effect = query_side_effect
        return client

    def test_no_meta_returns_false(self):
        client = self._make_client(hash_response=[])
        assert _check_graph_up_to_date(client, "abc", ["4.0.0"], True, True) is False

    def test_hash_mismatch_returns_false(self):
        client = self._make_client(
            hash_response=[{"hash": "wrong"}],
        )
        assert _check_graph_up_to_date(client, "abc", ["4.0.0"], True, True) is False

    def test_version_mismatch_returns_false(self):
        client = self._make_client(
            hash_response=[{"hash": "abc"}],
            ver_response=[{"versions": ["3.0.0"]}],
        )
        assert _check_graph_up_to_date(client, "abc", ["4.0.0"], True, True) is False

    def test_matching_no_embeddings_no_clusters(self):
        """Hash + versions match, no embeddings/clusters requested → True."""
        client = self._make_client(
            hash_response=[{"hash": "abc"}],
            ver_response=[{"versions": ["4.0.0"]}],
        )
        assert _check_graph_up_to_date(client, "abc", ["4.0.0"], False, False) is True

    def test_matching_with_embeddings_present(self):
        client = self._make_client(
            hash_response=[{"hash": "abc"}],
            ver_response=[{"versions": ["4.0.0"]}],
            emb_response=[{"total": 100, "with_emb": 100}],
        )
        assert _check_graph_up_to_date(client, "abc", ["4.0.0"], True, False) is True

    def test_matching_with_insufficient_embeddings(self):
        client = self._make_client(
            hash_response=[{"hash": "abc"}],
            ver_response=[{"versions": ["4.0.0"]}],
            emb_response=[{"total": 100, "with_emb": 10}],
        )
        assert _check_graph_up_to_date(client, "abc", ["4.0.0"], True, False) is False

    def test_matching_with_clusters_present(self):
        client = self._make_client(
            hash_response=[{"hash": "abc"}],
            ver_response=[{"versions": ["4.0.0"]}],
            cluster_response=[{"cnt": 50}],
        )
        assert _check_graph_up_to_date(client, "abc", ["4.0.0"], False, True) is True

    def test_matching_with_no_clusters(self):
        client = self._make_client(
            hash_response=[{"hash": "abc"}],
            ver_response=[{"versions": ["4.0.0"]}],
            cluster_response=[{"cnt": 0}],
        )
        assert _check_graph_up_to_date(client, "abc", ["4.0.0"], False, True) is False

    def test_query_exception_returns_false(self):
        client = MagicMock()
        client.query.side_effect = Exception("boom")
        assert _check_graph_up_to_date(client, "abc", ["4.0.0"], True, True) is False
