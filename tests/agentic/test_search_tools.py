"""Tests for unified MCP search tools.

Each test class covers one search_* MCP tool. Tests mock the GraphClient
and Encoder to avoid requiring a running Neo4j/embedding server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.agentic.search_formatters import (
    format_code_report,
    format_docs_report,
    format_imas_report,
    format_signals_report,
)
from imas_codex.agentic.search_tools import (
    _search_code,
    _search_docs,
    _search_imas,
    _search_signals,
)

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


# ---------------------------------------------------------------------------
# search_code
# ---------------------------------------------------------------------------


class TestSearchCode:
    """Unit tests for search_code tool."""

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
        mock_gc.query.side_effect = [[]]
        result = _search_code(
            query="equilibrium reconstruction",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "No code examples found" in result

    def test_code_results_formatted(self, mock_gc, mock_encoder):
        """Code results are formatted with data references."""
        vector_results = [{"id": "chunk:1", "score": 0.89}]
        enrichment = [
            {
                "id": "chunk:1",
                "text": "def read_equilibrium(shot):\n    tree = MDSplus.Tree('tcv_shot', shot)\n    psi = tree.getNode('\\\\results::psi').data()",
                "function_name": "read_equilibrium",
                "source_file": "/home/codes/liuqe/liuqe_reader.py",
                "facility_id": "tcv",
                "data_refs": [
                    {
                        "type": "mdsplus",
                        "raw": "\\RESULTS::PSI",
                        "tree": "\\RESULTS::PSI",
                        "imas": "equilibrium.time_slice[:].profiles_2d[:].psi",
                        "tdi": None,
                    }
                ],
                "directory": "/home/codes/liuqe",
                "dir_description": "LIUQE equilibrium reconstruction code",
            },
        ]
        mock_gc.query.side_effect = [vector_results, enrichment]

        result = _search_code(
            query="equilibrium",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "Code Examples" in result
        assert "read_equilibrium" in result
        assert "liuqe_reader.py" in result
        assert "Data references" in result

    def test_facility_filter(self, mock_gc, mock_encoder):
        """facility parameter filters code chunks."""
        mock_gc.query.side_effect = [[]]
        _search_code(
            query="test",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        first_call = mock_gc.query.call_args_list[0]
        cypher = first_call[0][0]
        assert "facility" in cypher.lower()

    def test_no_facility_filter(self, mock_gc, mock_encoder):
        """Without facility, no facility filter in query."""
        mock_gc.query.side_effect = [[]]
        _search_code(
            query="test",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        first_call = mock_gc.query.call_args_list[0]
        assert "facility" not in first_call[1]  # kwargs

    def test_embedding_unavailable(self, mock_gc):
        """When encoder is unavailable, return helpful message."""
        from imas_codex.embeddings.encoder import EmbeddingBackendError

        bad_encoder = MagicMock()
        bad_encoder.embed_texts.side_effect = EmbeddingBackendError("unavailable")

        result = _search_code(query="test", gc=mock_gc, encoder=bad_encoder)
        assert "Embedding" in result or "unavailable" in result.lower()


class TestFormatCodeReport:
    """Unit tests for the code report formatter."""

    def test_empty_code(self):
        result = format_code_report([], {})
        assert "No code examples found" in result

    def test_data_refs_shown(self):
        """Data references appear in output."""
        code_results = [
            {
                "id": "c1",
                "text": "tree.getNode('psi').data()",
                "function_name": "read_psi",
                "source_file": "/code/reader.py",
                "facility_id": "tcv",
                "data_refs": [
                    {
                        "type": "mdsplus",
                        "raw": "\\PSI",
                        "tree": "\\PSI",
                        "imas": None,
                        "tdi": None,
                    }
                ],
                "directory": "/code",
                "dir_description": "Analysis code",
            }
        ]
        result = format_code_report(code_results, {"c1": 0.85})
        assert "read_psi" in result
        assert "\\PSI" in result
        assert "Data references" in result


# ---------------------------------------------------------------------------
# search_imas
# ---------------------------------------------------------------------------


class TestSearchImas:
    """Unit tests for search_imas tool."""

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
        """Empty vector results produce a descriptive message."""
        # path vector search returns empty, cluster search returns empty
        mock_gc.query.side_effect = [[], []]
        result = _search_imas(
            query="electron temperature",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "No IMAS paths found" in result

    def test_path_results_formatted(self, mock_gc, mock_encoder):
        """Path results are enriched and formatted into a report."""
        path_vector = [
            {"id": "core_profiles.profiles_1d[:].electrons.temperature", "score": 0.95},
        ]
        cluster_vector = [
            {
                "id": "cluster:electron_temp",
                "label": "Electron Temperature Profiles",
                "scope": "core_profiles",
                "path_count": 12,
                "sample_paths": ["core_profiles.profiles_1d[:].electrons.temperature"],
                "score": 0.88,
            },
        ]
        enrichment = [
            {
                "id": "core_profiles.profiles_1d[:].electrons.temperature",
                "name": "temperature",
                "ids": "core_profiles",
                "documentation": "Electron temperature profile",
                "data_type": "FLT_1D",
                "physics_domain": "core_transport",
                "cocos_label_transformation": None,
                "unit": "eV",
                "clusters": ["Electron Temperature Profiles"],
                "coordinates": ["core_profiles.profiles_1d[:].grid.rho_tor_norm"],
                "introduced_in": "3.0.0",
            },
        ]
        # Calls: path vector, cluster vector, enrichment
        mock_gc.query.side_effect = [path_vector, cluster_vector, enrichment]

        result = _search_imas(
            query="electron temperature",
            gc=mock_gc,
            encoder=mock_encoder,
        )

        assert "IMAS Paths" in result
        assert "core_profiles.profiles_1d[:].electrons.temperature" in result
        assert "eV" in result
        assert "Electron Temperature Profiles" in result

    def test_ids_filter_passed(self, mock_gc, mock_encoder):
        """ids_filter parameter is passed to the vector search query."""
        mock_gc.query.side_effect = [[], []]
        _search_imas(
            query="temperature",
            ids_filter="core_profiles",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        first_call = mock_gc.query.call_args_list[0]
        cypher = first_call[0][0]
        assert "ids_filter" in cypher or "ids" in cypher

    def test_facility_crossrefs(self, mock_gc, mock_encoder):
        """Facility cross-references are fetched and formatted."""
        path_vector = [
            {"id": "magnetics.ip.0d[:].value", "score": 0.92},
        ]
        cluster_vector = []
        enrichment = [
            {
                "id": "magnetics.ip.0d[:].value",
                "name": "value",
                "ids": "magnetics",
                "documentation": "Plasma current",
                "data_type": "FLT_0D",
                "physics_domain": "magnetics",
                "cocos_label_transformation": "ip_like",
                "unit": "A",
                "clusters": [],
                "coordinates": [],
                "introduced_in": "3.0.0",
            },
        ]
        crossrefs = [
            {
                "id": "magnetics.ip.0d[:].value",
                "facility_signals": ["tcv:magnetics/ip"],
                "wiki_mentions": ["Plasma Current"],
                "code_files": ["/home/codes/ip_analysis.py"],
            },
        ]
        # Calls: path vector, cluster vector, enrichment, facility crossrefs
        mock_gc.query.side_effect = [path_vector, cluster_vector, enrichment, crossrefs]

        result = _search_imas(
            query="plasma current",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )

        assert "tcv:magnetics/ip" in result
        assert "Plasma Current" in result
        assert "ip_analysis.py" in result

    def test_version_context(self, mock_gc, mock_encoder):
        """Version context is fetched and formatted when requested."""
        path_vector = [
            {"id": "magnetics.ip.0d[:].value", "score": 0.92},
        ]
        cluster_vector = []
        enrichment = [
            {
                "id": "magnetics.ip.0d[:].value",
                "name": "value",
                "ids": "magnetics",
                "documentation": "Plasma current",
                "data_type": "FLT_0D",
                "physics_domain": "magnetics",
                "cocos_label_transformation": None,
                "unit": "A",
                "clusters": [],
                "coordinates": [],
                "introduced_in": "3.0.0",
            },
        ]
        version_ctx = [
            {
                "id": "magnetics.ip.0d[:].value",
                "change_count": 2,
                "notable_changes": [
                    {
                        "version": "3.20.0",
                        "type": "sign_convention",
                        "summary": "Sign convention clarified for COCOS",
                    },
                ],
            },
        ]
        # Calls: path vector, cluster vector, enrichment, version context
        mock_gc.query.side_effect = [
            path_vector,
            cluster_vector,
            enrichment,
            version_ctx,
        ]

        result = _search_imas(
            query="plasma current",
            include_version_context=True,
            gc=mock_gc,
            encoder=mock_encoder,
        )

        assert "Version history" in result
        assert "sign_convention" in result

    def test_embedding_unavailable(self, mock_gc):
        """When encoder raises, return helpful error message."""
        from imas_codex.embeddings.encoder import EmbeddingBackendError

        bad_encoder = MagicMock()
        bad_encoder.embed_texts.side_effect = EmbeddingBackendError("unavailable")

        result = _search_imas(query="test", gc=mock_gc, encoder=bad_encoder)
        assert "Embedding" in result or "unavailable" in result.lower()

    def test_neo4j_unavailable(self, mock_encoder):
        """When Neo4j is down, return helpful error message."""
        from neo4j.exceptions import ServiceUnavailable

        bad_gc = MagicMock()
        bad_gc.query.side_effect = ServiceUnavailable("Connection refused")

        result = _search_imas(query="test", gc=bad_gc, encoder=mock_encoder)
        assert "not running" in result.lower() or "neo4j" in result.lower()


class TestFormatImasReport:
    """Unit tests for the IMAS report formatter."""

    def test_empty_results(self):
        """Empty results produce a descriptive message."""
        result = format_imas_report([], [], {}, {}, {})
        assert "No IMAS paths found" in result

    def test_path_formatting(self):
        """A single IMAS path is formatted with metadata."""
        paths = [
            {
                "id": "core_profiles.profiles_1d[:].electrons.temperature",
                "name": "temperature",
                "ids": "core_profiles",
                "documentation": "Electron temperature as function of rho_tor_norm",
                "data_type": "FLT_1D",
                "physics_domain": "core_transport",
                "cocos_label_transformation": None,
                "unit": "eV",
                "clusters": ["Electron Temperature Profiles"],
                "coordinates": ["core_profiles.profiles_1d[:].grid.rho_tor_norm"],
                "introduced_in": "3.0.0",
            },
        ]
        scores = {"core_profiles.profiles_1d[:].electrons.temperature": 0.95}

        result = format_imas_report(paths, [], {}, {}, scores)

        assert "core_profiles.profiles_1d[:].electrons.temperature" in result
        assert "0.95" in result
        assert "eV" in result
        assert "core_transport" in result
        assert "Electron Temperature Profiles" in result
        assert "3.0.0" in result

    def test_cluster_formatting(self):
        """Clusters appear in Related Clusters section."""
        clusters = [
            {
                "id": "cluster:plasma_current",
                "label": "Plasma Current Measurements",
                "scope": "magnetics",
                "path_count": 8,
                "sample_paths": [
                    "magnetics.ip.0d[:].value",
                    "magnetics.ip.measure[:].value",
                ],
                "score": 0.88,
            },
        ]
        scores = {"cluster:plasma_current": 0.88}

        result = format_imas_report([], clusters, {}, {}, scores)

        assert "Related Clusters" in result
        assert "Plasma Current Measurements" in result
        assert "magnetics" in result

    def test_facility_xrefs_shown(self):
        """Facility cross-references appear in path output."""
        paths = [
            {
                "id": "magnetics.ip.0d[:].value",
                "name": "value",
                "ids": "magnetics",
                "documentation": "Plasma current",
                "data_type": "FLT_0D",
                "physics_domain": None,
                "cocos_label_transformation": None,
                "unit": "A",
                "clusters": [],
                "coordinates": [],
                "introduced_in": None,
            },
        ]
        xrefs = {
            "magnetics.ip.0d[:].value": {
                "id": "magnetics.ip.0d[:].value",
                "facility_signals": ["tcv:magnetics/ip"],
                "wiki_mentions": ["Current Measurements"],
                "code_files": ["/codes/ip.py"],
            },
        }
        scores = {"magnetics.ip.0d[:].value": 0.9}

        result = format_imas_report(paths, [], xrefs, {}, scores)

        assert "tcv:magnetics/ip" in result
        assert "Current Measurements" in result
        assert "/codes/ip.py" in result
