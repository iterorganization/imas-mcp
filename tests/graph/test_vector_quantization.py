"""Tests for vector index quantization and dimension-mismatch detection."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from imas_codex.graph.client import EXPECTED_VECTOR_INDEXES, GraphClient


class TestVectorQuantization:
    """Verify quantization flag is present in CREATE VECTOR INDEX queries."""

    def test_ensure_vector_indexes_includes_quantization(self):
        """ensure_vector_indexes() must include vector.quantization.enabled: true."""
        # Capture all Cypher strings sent to sess.run()
        captured_queries: list[str] = []

        mock_sess = MagicMock()

        def fake_run(query, **kwargs):
            if isinstance(query, str):
                captured_queries.append(query)
            result = MagicMock()
            result.__iter__ = lambda self: iter([])
            return result

        mock_sess.run.side_effect = fake_run
        mock_sess.__enter__ = lambda s: mock_sess
        mock_sess.__exit__ = MagicMock(return_value=False)

        client = MagicMock(spec=GraphClient)
        client.session.return_value = mock_sess

        with patch(
            "imas_codex.graph.client.get_embedding_dimension", return_value=1024
        ):
            # Call the real method with our mocked client
            GraphClient.ensure_vector_indexes(client)

        create_queries = [q for q in captured_queries if "CREATE VECTOR INDEX" in q]
        assert create_queries, "No CREATE VECTOR INDEX queries were executed"

        for q in create_queries:
            assert "vector.quantization.enabled" in q, (
                f"Missing quantization flag in:\n{q}"
            )
            assert "true" in q.lower(), f"Quantization not set to true in:\n{q}"

    def test_ensure_vector_indexes_includes_cosine_similarity(self):
        """Quantization must not replace the similarity function."""
        captured_queries: list[str] = []

        mock_sess = MagicMock()

        def fake_run(query, **kwargs):
            if isinstance(query, str):
                captured_queries.append(query)
            result = MagicMock()
            result.__iter__ = lambda self: iter([])
            return result

        mock_sess.run.side_effect = fake_run
        mock_sess.__enter__ = lambda s: mock_sess
        mock_sess.__exit__ = MagicMock(return_value=False)

        client = MagicMock(spec=GraphClient)
        client.session.return_value = mock_sess

        with patch(
            "imas_codex.graph.client.get_embedding_dimension", return_value=1024
        ):
            GraphClient.ensure_vector_indexes(client)

        create_queries = [q for q in captured_queries if "CREATE VECTOR INDEX" in q]
        for q in create_queries:
            assert "cosine" in q, f"Missing cosine similarity in:\n{q}"


class TestDimensionMismatchDetection:
    """Verify stale indexes are dropped when dimension changes."""

    def _make_client_and_session(self):
        """Return (client, mock_sess, captured_queries) ready for testing."""
        captured_queries: list[str] = []
        dropped: list[str] = []

        mock_sess = MagicMock()

        def fake_run(query, **kwargs):
            if isinstance(query, str):
                captured_queries.append(query)
                if "DROP INDEX" in query:
                    # Extract index name between backticks
                    import re

                    m = re.search(r"DROP INDEX `([^`]+)`", query)
                    if m:
                        dropped.append(m.group(1))
            result = MagicMock()
            result.__iter__ = lambda self: iter([])
            return result

        mock_sess.run.side_effect = fake_run
        mock_sess.__enter__ = lambda s: mock_sess
        mock_sess.__exit__ = MagicMock(return_value=False)

        client = MagicMock(spec=GraphClient)
        client.session.return_value = mock_sess

        return client, mock_sess, captured_queries, dropped

    def test_drops_index_with_wrong_dimension(self):
        """Indexes whose stored dimension differs from config must be dropped."""
        import re

        captured_queries: list[str] = []
        dropped: list[str] = []

        mock_sess = MagicMock()
        stale_index = {"name": "imas_node_embedding", "dim": 256}

        def fake_run(query, **kwargs):
            if isinstance(query, str):
                captured_queries.append(query)
                if "DROP INDEX" in query:
                    m = re.search(r"DROP INDEX `([^`]+)`", query)
                    if m:
                        dropped.append(m.group(1))
            if "SHOW INDEXES" in query and "VECTOR" in query:
                result = MagicMock()
                result.__iter__ = lambda self: iter([stale_index])
                return result
            # Second SHOW INDEXES (existing names check) returns empty
            result = MagicMock()
            result.__iter__ = lambda self: iter([])
            return result

        mock_sess.run.side_effect = fake_run
        mock_sess.__enter__ = lambda s: mock_sess
        mock_sess.__exit__ = MagicMock(return_value=False)

        client = MagicMock(spec=GraphClient)
        client.session.return_value = mock_sess

        with patch(
            "imas_codex.graph.client.get_embedding_dimension", return_value=1024
        ):
            GraphClient.ensure_vector_indexes(client)

        assert "imas_node_embedding" in dropped, (
            "Stale index with wrong dimension should have been dropped"
        )

    def test_does_not_drop_index_with_correct_dimension(self):
        """Indexes already at the configured dimension must NOT be dropped."""
        client, mock_sess, captured_queries, dropped = self._make_client_and_session()

        correct_index = {"name": "imas_node_embedding", "dim": 1024}

        def fake_run(query, **kwargs):
            if isinstance(query, str):
                captured_queries.append(query)
            if "SHOW INDEXES" in query and "VECTOR" in query:
                result = MagicMock()
                result.__iter__ = lambda self: iter([correct_index])
                return result
            result = MagicMock()
            result.__iter__ = lambda self: iter([])
            return result

        mock_sess.run.side_effect = fake_run

        with patch(
            "imas_codex.graph.client.get_embedding_dimension", return_value=1024
        ):
            GraphClient.ensure_vector_indexes(client)

        assert "imas_node_embedding" not in dropped, (
            "Index with correct dimension must not be dropped"
        )
        drop_queries = [q for q in captured_queries if "DROP INDEX" in q]
        assert not drop_queries, "No DROP INDEX should have been issued"

    def test_drops_multiple_mismatched_indexes(self):
        """All mismatched indexes across all index names should be dropped."""
        client, mock_sess, captured_queries, dropped = self._make_client_and_session()

        stale_indexes = [
            {"name": "imas_node_embedding", "dim": 256},
            {"name": "wiki_chunk_embedding", "dim": 256},
        ]

        def fake_run(query, **kwargs):
            if isinstance(query, str):
                captured_queries.append(query)
                if "DROP INDEX" in query:
                    import re

                    m = re.search(r"DROP INDEX `([^`]+)`", query)
                    if m:
                        dropped.append(m.group(1))
            if "SHOW INDEXES" in query and "VECTOR" in query:
                result = MagicMock()
                result.__iter__ = lambda self: iter(stale_indexes)
                return result
            result = MagicMock()
            result.__iter__ = lambda self: iter([])
            return result

        mock_sess.run.side_effect = fake_run

        with patch(
            "imas_codex.graph.client.get_embedding_dimension", return_value=1024
        ):
            GraphClient.ensure_vector_indexes(client)

        assert "imas_node_embedding" in dropped
        assert "wiki_chunk_embedding" in dropped
