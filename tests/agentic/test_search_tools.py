"""Tests for unified MCP search tools.

Each test class covers one search_* MCP tool. Tests mock the GraphClient
and Encoder to avoid requiring a running Neo4j/embedding server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.agentic.search_formatters import (
    format_docs_report,
    format_signals_report,
)
from imas_codex.agentic.search_tools import _search_docs, _search_signals

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


# ---------------------------------------------------------------------------
# search_docs
# ---------------------------------------------------------------------------


class TestSearchDocs:
    """Unit tests for search_docs tool."""

    @pytest.fixture()
    def mock_gc(self):
        gc = MagicMock()
        gc.query = MagicMock(return_value=[])
        return gc

    @pytest.fixture()
    def mock_encoder(self):
        enc = MagicMock()
        enc.embed_texts = MagicMock(return_value=[[0.1] * 1024])
        return enc

    def test_empty_results(self, mock_gc, mock_encoder):
        """Empty results produce a descriptive message."""
        # wiki chunks, artifacts, images — all return empty
        mock_gc.query.side_effect = [[], [], []]
        result = _search_docs(
            query="equilibrium",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "No documentation found" in result

    def test_wiki_chunks_formatted(self, mock_gc, mock_encoder):
        """Wiki chunks are grouped by page and formatted."""
        chunk_vector = [
            {"id": "jet:wiki:chunk:1", "score": 0.90},
            {"id": "jet:wiki:chunk:2", "score": 0.85},
        ]
        enrichment = [
            {
                "id": "jet:wiki:chunk:1",
                "text": "Fishbone instabilities are observed when...",
                "section": "Overview",
                "page_title": "Fishbone instabilities",
                "page_url": "https://wiki.jet.efda.org/fishbone",
                "linked_signals": ["jet:mhd/fishbone_amplitude"],
                "linked_tree_nodes": [],
                "imas_refs": ["mhd_linear.time_slice[:].toroidal_mode[:].n_tor"],
            },
            {
                "id": "jet:wiki:chunk:2",
                "text": "Detection methods include Mirnov coil analysis...",
                "section": "Detection Methods",
                "page_title": "Fishbone instabilities",
                "page_url": "https://wiki.jet.efda.org/fishbone",
                "linked_signals": [],
                "linked_tree_nodes": [],
                "imas_refs": [],
            },
        ]
        # Calls: wiki chunks vector, artifact vector, image vector, enrichment
        mock_gc.query.side_effect = [chunk_vector, [], [], enrichment]

        result = _search_docs(
            query="fishbone instabilities",
            facility="jet",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "Wiki Documentation" in result
        assert "Fishbone instabilities" in result
        assert "Overview" in result
        assert "Detection Methods" in result

    def test_page_grouping(self, mock_gc, mock_encoder):
        """Chunks from the same page are grouped together."""
        chunk_vector = [
            {"id": "c1", "score": 0.9},
            {"id": "c2", "score": 0.8},
        ]
        enrichment = [
            {
                "id": "c1",
                "text": "Section 1 content",
                "section": "Sec1",
                "page_title": "Same Page",
                "page_url": "http://wiki/same",
                "linked_signals": [],
                "linked_tree_nodes": [],
                "imas_refs": [],
            },
            {
                "id": "c2",
                "text": "Section 2 content",
                "section": "Sec2",
                "page_title": "Same Page",
                "page_url": "http://wiki/same",
                "linked_signals": [],
                "linked_tree_nodes": [],
                "imas_refs": [],
            },
        ]
        mock_gc.query.side_effect = [chunk_vector, [], [], enrichment]

        result = _search_docs(
            query="test", facility="tcv", gc=mock_gc, encoder=mock_encoder
        )
        # Page title should appear once as header, not twice
        assert result.count('### Page: "Same Page"') == 1

    def test_cross_links_shown(self, mock_gc, mock_encoder):
        """Cross-links to signals and IMAS paths are shown."""
        chunk_vector = [{"id": "c1", "score": 0.9}]
        enrichment = [
            {
                "id": "c1",
                "text": "Content about plasma current",
                "section": "Signals",
                "page_title": "Magnetics",
                "page_url": None,
                "linked_signals": ["tcv:magnetics/ip"],
                "linked_tree_nodes": ["\\RESULTS::I_P"],
                "imas_refs": ["magnetics.ip.0d[:].value"],
            },
        ]
        mock_gc.query.side_effect = [chunk_vector, [], [], enrichment]

        result = _search_docs(
            query="plasma current", facility="tcv", gc=mock_gc, encoder=mock_encoder
        )
        assert "tcv:magnetics/ip" in result
        assert "magnetics.ip.0d[:].value" in result

    def test_embedding_unavailable(self, mock_gc):
        """When encoder is unavailable, return helpful message."""
        from imas_codex.embeddings.encoder import EmbeddingBackendError

        bad_encoder = MagicMock()
        bad_encoder.embed_texts.side_effect = EmbeddingBackendError("unavailable")

        result = _search_docs(
            query="test", facility="tcv", gc=mock_gc, encoder=bad_encoder
        )
        assert "Embedding" in result or "unavailable" in result.lower()


class TestFormatDocsReport:
    """Unit tests for the docs report formatter."""

    def test_empty_docs(self):
        result = format_docs_report([], [], {})
        assert "No documentation found" in result

    def test_artifacts_section(self):
        """Artifacts appear in Related Documents section."""
        artifacts = [
            {
                "id": "art1",
                "title": "Analysis Report.pdf",
                "description": "Detailed analysis",
                "page_title": "MHD diagnostics",
            },
        ]
        result = format_docs_report([], artifacts, {})
        assert "Related Documents" in result
        assert "Analysis Report.pdf" in result
        assert "MHD diagnostics" in result
