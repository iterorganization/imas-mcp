"""Tests for unified MCP search tools.

Each test class covers one search_* MCP tool. Tests mock the GraphClient
and Encoder to avoid requiring a running Neo4j/embedding server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.agentic.search_formatters import format_signals_report
from imas_codex.agentic.search_tools import _search_signals

# ---------------------------------------------------------------------------
# search_signals
# ---------------------------------------------------------------------------


class TestSearchSignals:
    """Unit tests for search_signals tool."""

    @pytest.fixture()
    def mock_gc(self):
        """GraphClient mock that returns canned data."""
        gc = MagicMock()
        gc.query = MagicMock(return_value=[])
        return gc

    @pytest.fixture()
    def mock_encoder(self):
        """Encoder mock that returns a fixed embedding."""
        enc = MagicMock()
        enc.embed_texts = MagicMock(return_value=[[0.1] * 1024])
        return enc

    def test_empty_results(self, mock_gc, mock_encoder):
        """Empty vector + enrichment results produce a descriptive message."""
        mock_gc.query.side_effect = [[], []]  # vector search, tree node search
        result = _search_signals(
            query="plasma current",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "No signals found" in result

    def test_signal_results_formatted(self, mock_gc, mock_encoder):
        """Signal results are formatted into a readable report."""
        # First call: vector search returns signal IDs + scores
        vector_results = [
            {"id": "tcv:magnetics/ip", "score": 0.92},
            {"id": "tcv:magnetics/bpol", "score": 0.85},
        ]
        # Second call: enrichment query returns full details
        enrichment_results = [
            {
                "id": "tcv:magnetics/ip",
                "name": "ip",
                "description": "Plasma current",
                "physics_domain": "magnetics",
                "unit": "A",
                "checked": True,
                "example_shot": 84000,
                "diagnostic_name": "magnetics",
                "diagnostic_category": "magnetics",
                "access_template": "t = MDSplus.Tree('tcv_shot', shot)\nnode = t.getNode('\\\\results::i_p')",
                "access_type": "mdsplus",
                "imports_template": "import MDSplus",
                "connection_template": None,
                "tree_path": "\\RESULTS::I_P",
                "tree_name": "tcv_shot",
                "imas_path": "magnetics.ip.0d[:].value",
                "imas_docs": "Plasma current positive sign",
                "imas_unit": "A",
            },
            {
                "id": "tcv:magnetics/bpol",
                "name": "bpol",
                "description": "Poloidal magnetic field",
                "physics_domain": "magnetics",
                "unit": "T",
                "checked": False,
                "example_shot": None,
                "diagnostic_name": "magnetics",
                "diagnostic_category": "magnetics",
                "access_template": None,
                "access_type": None,
                "imports_template": None,
                "connection_template": None,
                "tree_path": None,
                "tree_name": None,
                "imas_path": None,
                "imas_docs": None,
                "imas_unit": None,
            },
        ]
        # Third call: tree node vector search
        tree_results = [
            {
                "id": "tcv:\\RESULTS::I_P",
                "path": "\\RESULTS::I_P",
                "tree_name": "tcv_shot",
                "description": "Plasma current",
                "unit": "A",
            }
        ]
        mock_gc.query.side_effect = [vector_results, enrichment_results, tree_results]

        result = _search_signals(
            query="plasma current",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )

        assert "## Signals" in result
        assert "tcv:magnetics/ip" in result
        assert "Plasma current" in result
        assert "magnetics" in result

    def test_diagnostic_filter_passed(self, mock_gc, mock_encoder):
        """diagnostic parameter is passed to the vector search query."""
        mock_gc.query.side_effect = [[], []]  # vector search, tree nodes
        _search_signals(
            query="current",
            facility="tcv",
            diagnostic="magnetics",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        # Check first call (vector search) includes diagnostic filter
        first_call = mock_gc.query.call_args_list[0]
        cypher = first_call[0][0]
        assert "diagnostic" in cypher

    def test_physics_domain_filter_passed(self, mock_gc, mock_encoder):
        """physics_domain parameter is passed to the vector search query."""
        mock_gc.query.side_effect = [[], []]  # vector search, tree nodes
        _search_signals(
            query="current",
            facility="tcv",
            physics_domain="magnetics",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        first_call = mock_gc.query.call_args_list[0]
        cypher = first_call[0][0]
        assert "physics_domain" in cypher

    def test_no_tree_nodes_section_omitted(self, mock_gc, mock_encoder):
        """When tree node search returns empty, section is omitted."""
        vector_results = [{"id": "tcv:magnetics/ip", "score": 0.92}]
        enrichment_results = [
            {
                "id": "tcv:magnetics/ip",
                "name": "ip",
                "description": "Plasma current",
                "physics_domain": "magnetics",
                "unit": "A",
                "checked": True,
                "example_shot": 84000,
                "diagnostic_name": "magnetics",
                "diagnostic_category": None,
                "access_template": None,
                "access_type": None,
                "imports_template": None,
                "connection_template": None,
                "tree_path": None,
                "tree_name": None,
                "imas_path": None,
                "imas_docs": None,
                "imas_unit": None,
            },
        ]
        mock_gc.query.side_effect = [vector_results, enrichment_results, []]

        result = _search_signals(
            query="plasma current",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "Related Tree Nodes" not in result

    def test_embedding_unavailable(self, mock_gc):
        """When encoder raises, return helpful error message."""
        from imas_codex.embeddings.encoder import EmbeddingBackendError

        bad_encoder = MagicMock()
        bad_encoder.embed_texts.side_effect = EmbeddingBackendError("unavailable")

        result = _search_signals(
            query="current",
            facility="tcv",
            gc=mock_gc,
            encoder=bad_encoder,
        )
        assert "Embedding" in result or "unavailable" in result.lower()

    def test_neo4j_unavailable(self, mock_encoder):
        """When Neo4j is down, return helpful error message."""
        from neo4j.exceptions import ServiceUnavailable

        bad_gc = MagicMock()
        bad_gc.query.side_effect = ServiceUnavailable("Connection refused")

        result = _search_signals(
            query="current",
            facility="tcv",
            gc=bad_gc,
            encoder=mock_encoder,
        )
        assert "not running" in result.lower() or "neo4j" in result.lower()


# ---------------------------------------------------------------------------
# Formatter unit tests
# ---------------------------------------------------------------------------


class TestFormatSignalsReport:
    """Unit tests for the signals report formatter."""

    def test_empty_signals(self):
        """Empty signal list produces empty-result message."""
        result = format_signals_report([], [], {})
        assert "No signals found" in result

    def test_basic_signal_formatting(self):
        """A single signal is formatted with all sections."""
        signals = [
            {
                "id": "tcv:magnetics/ip",
                "name": "ip",
                "description": "Plasma current",
                "physics_domain": "magnetics",
                "unit": "A",
                "checked": True,
                "example_shot": 84000,
                "diagnostic_name": "magnetics",
                "diagnostic_category": "magnetics",
                "access_template": "tree.getNode('\\\\results::i_p')",
                "access_type": "mdsplus",
                "imports_template": "import MDSplus",
                "connection_template": None,
                "tree_path": "\\RESULTS::I_P",
                "tree_name": "tcv_shot",
                "imas_path": "magnetics.ip.0d[:].value",
                "imas_docs": "Plasma current positive sign",
                "imas_unit": "A",
            }
        ]
        scores = {"tcv:magnetics/ip": 0.92}

        result = format_signals_report(signals, [], scores)

        assert "tcv:magnetics/ip" in result
        assert "0.92" in result
        assert "Plasma current" in result
        assert "magnetics" in result
        assert "Data access" in result
        assert "IMAS mapping" in result
        assert "Tree node" in result

    def test_tree_nodes_section(self):
        """Tree nodes appear in Related Tree Nodes section."""
        tree_nodes = [
            {
                "id": "tcv:\\RESULTS::I_P",
                "path": "\\RESULTS::I_P",
                "tree_name": "tcv_shot",
                "description": "Plasma current",
                "unit": "A",
            }
        ]
        result = format_signals_report([], tree_nodes, {})
        assert "Related Tree Nodes" in result
        assert "\\RESULTS::I_P" in result

    def test_access_template_truncation(self):
        """Long access templates are truncated."""
        long_template = "x = 1\n" * 100
        signals = [
            {
                "id": "tcv:test/sig",
                "name": "sig",
                "description": "Test signal",
                "physics_domain": None,
                "unit": None,
                "checked": False,
                "example_shot": None,
                "diagnostic_name": None,
                "diagnostic_category": None,
                "access_template": long_template,
                "access_type": "mdsplus",
                "imports_template": None,
                "connection_template": None,
                "tree_path": None,
                "tree_name": None,
                "imas_path": None,
                "imas_docs": None,
                "imas_unit": None,
            }
        ]
        result = format_signals_report(signals, [], {"tcv:test/sig": 0.5})
        # Template should be truncated - total result should be reasonable
        assert len(result) < 5000
