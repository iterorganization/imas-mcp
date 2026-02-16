"""Tests for clear_dd_graph and the imas clear CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


class TestClearDdGraph:
    """Test clear_dd_graph function with mocked GraphClient."""

    def _make_client(self, node_counts: dict[str, int] | None = None):
        """Create a mock GraphClient that simulates batch deletion."""
        counts = node_counts or {
            "IMASPath": 100,
            "DDVersion": 5,
            "IDS": 10,
            "IMASSemanticCluster": 50,
            "IMASPathChange": 20,
            "EmbeddingChange": 0,
            "IdentifierSchema": 3,
            "IMASCoordinateSpec": 8,
        }

        client = MagicMock()

        def query_side_effect(cypher, **kwargs):
            batch_size = kwargs.get("batch_size", 5000)

            # Handle DETACH DELETE queries
            if "DETACH DELETE" in cypher:
                for label, count in counts.items():
                    if f":{label}" in cypher and f"(n:{label})" in cypher:
                        if count > 0:
                            deleted = min(count, batch_size)
                            counts[label] -= deleted
                            return [{"deleted": deleted}]
                        return [{"deleted": 0}]

                # Orphaned Unit cleanup
                if "Unit" in cypher and "HAS_UNIT" in cypher:
                    return [{"deleted": 5}]

            # Handle REMOVE (DDVersion build_hash cleanup)
            if "REMOVE" in cypher:
                return []

            # Handle DROP INDEX
            if "DROP INDEX" in cypher:
                return []

            return [{"deleted": 0}]

        client.query.side_effect = query_side_effect
        return client

    def test_clears_all_node_types(self):
        """All DD node types are deleted."""
        from imas_codex.graph.build_dd import clear_dd_graph

        client = self._make_client()
        results = clear_dd_graph(client)

        assert results["paths"] == 100
        assert results["versions"] == 5
        assert results["ids_nodes"] == 10
        assert results["clusters"] == 50
        assert results["path_changes"] == 20
        assert results["coordinate_specs"] == 8
        assert results["identifier_schemas"] == 3
        assert results["orphaned_units"] == 5

    def test_empty_graph_returns_zeros(self):
        """Clearing an empty graph returns all zeros."""
        from imas_codex.graph.build_dd import clear_dd_graph

        client = self._make_client(
            dict.fromkeys(
                [
                    "IMASPath",
                    "DDVersion",
                    "IDS",
                    "IMASSemanticCluster",
                    "IMASPathChange",
                    "EmbeddingChange",
                    "IdentifierSchema",
                    "IMASCoordinateSpec",
                ],
                0,
            )
        )
        results = clear_dd_graph(client)

        assert results["paths"] == 0
        assert results["versions"] == 0
        assert results["clusters"] == 0

    def test_drops_vector_indexes(self):
        """DD vector indexes are dropped."""
        from imas_codex.graph.build_dd import clear_dd_graph

        client = self._make_client()
        clear_dd_graph(client)

        # Check that DROP INDEX was called for DD-specific indexes (schema-derived)
        drop_calls = [
            call for call in client.query.call_args_list if "DROP INDEX" in str(call)
        ]
        drop_text = " ".join(str(c) for c in drop_calls)
        assert "imas_path_embedding" in drop_text
        assert "cluster_embedding" in drop_text

    def test_clears_dd_versions(self):
        """DDVersion nodes are deleted (build_hash cleared with them)."""
        from imas_codex.graph.build_dd import clear_dd_graph

        client = self._make_client()
        results = clear_dd_graph(client)

        # DDVersion nodes are deleted entirely, which removes build_hash
        assert results["versions"] == 5

    def test_batch_deletion(self):
        """Large node sets are deleted in batches."""
        from imas_codex.graph.build_dd import clear_dd_graph

        # 12000 paths requires 3 batches at batch_size=5000
        client = self._make_client({"IMASPath": 12000})
        # Override other types to 0
        for _label in [
            "DDVersion",
            "IDS",
            "IMASSemanticCluster",
            "IMASPathChange",
            "EmbeddingChange",
            "IdentifierSchema",
            "IMASCoordinateSpec",
        ]:
            pass  # Already 0 from default in mock

        results = clear_dd_graph(client, batch_size=5000)
        assert results["paths"] == 12000


class TestImasClearCLI:
    """Test the imas clear CLI command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_clear_help(self, runner):
        """imas clear has help text."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["imas", "clear", "--help"])
        assert result.exit_code == 0
        assert "Delete all IMAS" in result.output

    def test_clear_registered(self, runner):
        """imas clear is registered in the group."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["imas", "--help"])
        assert result.exit_code == 0
        assert "clear" in result.output

    @patch("imas_codex.graph.GraphClient")
    def test_clear_empty_graph(self, mock_gc_class, runner):
        """clear on empty graph shows message and exits."""
        from imas_codex.cli import main

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [
            {"paths": 0, "versions": 0, "ids": 0, "clusters": 0, "changes": 0}
        ]
        mock_gc_class.return_value = mock_gc

        result = runner.invoke(main, ["imas", "clear", "--force"])
        assert result.exit_code == 0
        assert "No DD nodes" in result.output

    @patch("imas_codex.graph.build_dd.clear_dd_graph")
    @patch("imas_codex.graph.GraphClient")
    def test_clear_force_skips_confirmation(self, mock_gc_class, mock_clear, runner):
        """--force flag skips confirmation."""
        from imas_codex.cli import main

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [
            {"paths": 100, "versions": 5, "ids": 10, "clusters": 50, "changes": 20}
        ]
        mock_gc_class.return_value = mock_gc
        mock_clear.return_value = {
            "paths": 100,
            "versions": 5,
            "ids_nodes": 10,
            "clusters": 50,
            "path_changes": 20,
            "embedding_changes": 0,
            "identifier_schemas": 0,
            "coordinate_specs": 8,
            "orphaned_units": 3,
        }

        result = runner.invoke(main, ["imas", "clear", "--force"])
        assert result.exit_code == 0
        assert "100" in result.output  # IMASPath count
        mock_clear.assert_called_once()

    @patch("imas_codex.graph.GraphClient")
    def test_clear_aborts_without_force(self, mock_gc_class, runner):
        """Without --force, user can abort."""
        from imas_codex.cli import main

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [
            {"paths": 100, "versions": 5, "ids": 10, "clusters": 50, "changes": 20}
        ]
        mock_gc_class.return_value = mock_gc

        result = runner.invoke(main, ["imas", "clear"], input="n\n")
        assert result.exit_code != 0  # Aborted
