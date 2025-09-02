"""
Test configuration and fixtures for the new MCP-based architecture.

This module provides test fixtures for the composition-based server architecture,
focusing on MCP protocol testing and feature validation.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import Client

from imas_mcp.search.document_store import Document, DocumentMetadata, DocumentStore
from imas_mcp.search.engines.base_engine import MockSearchEngine
from imas_mcp.server import Server

# Standard test IDS set for consistency across all tests
# This avoids re-embedding and ensures consistent performance
STANDARD_TEST_IDS_SET = {"equilibrium", "core_profiles"}


def create_mock_document(path_id: str, ids_name: str = "core_profiles") -> Document:
    """Create a mock document for testing."""
    metadata = DocumentMetadata(
        path_id=path_id,
        ids_name=ids_name,
        path_name=path_id.split("/")[-1],
        units="m",
        data_type="float",
        coordinates=("rho_tor_norm",),
        physics_domain="transport",
        physics_phenomena=("transport", "plasma"),
    )

    return Document(
        metadata=metadata,
        documentation=f"Mock documentation for {path_id}",
        physics_context={"domain": "transport", "phenomena": ["transport", "plasma"]},
        relationships={},
        raw_data={"data_type": "float", "units": "m"},
    )


def create_mock_documents() -> list[Document]:
    """Create a set of mock documents for testing."""
    return [
        create_mock_document("core_profiles/profiles_1d/electrons/temperature"),
        create_mock_document("core_profiles/profiles_1d/electrons/density"),
        create_mock_document("equilibrium/time_slice/profiles_1d/psi", "equilibrium"),
        create_mock_document(
            "equilibrium/time_slice/profiles_2d/b_field_r", "equilibrium"
        ),
    ]


@pytest.fixture(autouse=True)
def disable_caching():
    """Automatically disable caching for all tests by making cache always miss."""
    # Patch the cache get method to always return None (cache miss)
    with patch("imas_mcp.search.decorators.cache._cache.get", return_value=None):
        # Also patch the set method to do nothing
        with patch("imas_mcp.search.decorators.cache._cache.set"):
            yield


@pytest.fixture(autouse=True)
def mock_heavy_operations():
    """Mock all heavy operations that slow down tests."""
    mock_documents = create_mock_documents()

    with patch.multiple(
        DocumentStore,
        # Mock document loading
        _ensure_loaded=MagicMock(),
        _ensure_ids_loaded=MagicMock(),
        _load_ids_documents=MagicMock(),
        _load_identifier_catalog_documents=MagicMock(),
        load_all_documents=MagicMock(),
        # Mock index building
        _build_sqlite_fts_index=MagicMock(),
        _should_rebuild_fts_index=MagicMock(return_value=False),
        # Mock data access with test data
        get_all_documents=MagicMock(return_value=mock_documents),
        get_document=MagicMock(
            side_effect=lambda path_id: next(
                (doc for doc in mock_documents if doc.metadata.path_id == path_id), None
            )
        ),
        get_documents_by_ids=MagicMock(
            side_effect=lambda ids_name: [
                doc for doc in mock_documents if doc.metadata.ids_name == ids_name
            ]
        ),
        get_available_ids=MagicMock(return_value=list(STANDARD_TEST_IDS_SET)),
        __len__=MagicMock(return_value=len(mock_documents)),
        # Mock search methods
        search_full_text=MagicMock(return_value=mock_documents[:2]),
        search_by_keywords=MagicMock(return_value=mock_documents[:2]),
        search_by_physics_domain=MagicMock(return_value=mock_documents[:2]),
        search_by_units=MagicMock(return_value=mock_documents[:2]),
        # Mock statistics
        get_statistics=MagicMock(
            return_value={
                "total_documents": len(mock_documents),
                "total_ids": len(STANDARD_TEST_IDS_SET),
                "physics_domains": 2,
                "unique_units": 1,
                "coordinate_systems": 1,
                "documentation_terms": 100,
                "path_segments": 50,
            }
        ),
        # Mock identifier methods
        get_identifier_schemas=MagicMock(return_value=[]),
        get_identifier_paths=MagicMock(return_value=[]),
        get_identifier_schema_by_name=MagicMock(return_value=None),
    ):
        # Mock semantic search initialization to prevent embedding generation
        with patch("imas_mcp.server.SemanticSearch") as mock_semantic:
            mock_semantic_instance = MagicMock()
            mock_semantic_instance._initialize.return_value = None
            mock_semantic.return_value = mock_semantic_instance

            # Mock unit accessor to prevent heavy physics integration
            with patch(
                "imas_mcp.search.document_store.UnitAccessor"
            ) as mock_unit_accessor:
                mock_unit_accessor.return_value.get_all_unit_contexts.return_value = {}
                mock_unit_accessor.return_value.get_unit_context.return_value = (
                    "test context"
                )
                mock_unit_accessor.return_value.get_category_for_unit.return_value = (
                    "test_category"
                )
                mock_unit_accessor.return_value.get_domains_for_unit.return_value = [
                    "transport"
                ]

                # Mock search engines to prevent heavy initialization
                with patch(
                    "imas_mcp.search.engines.semantic_engine.SemanticSearchEngine"
                ) as mock_semantic_engine:
                    mock_semantic_engine.return_value = MockSearchEngine()

                    with patch(
                        "imas_mcp.search.engines.lexical_engine.LexicalSearchEngine"
                    ) as mock_lexical_engine:
                        mock_lexical_engine.return_value = MockSearchEngine()

                        with patch(
                            "imas_mcp.search.engines.hybrid_engine.HybridSearchEngine"
                        ) as mock_hybrid_engine:
                            mock_hybrid_engine.return_value = MockSearchEngine()

                            yield


@pytest.fixture(scope="session")
def server() -> Server:
    """Session-scoped server fixture for performance."""
    return Server(ids_set=STANDARD_TEST_IDS_SET)


@pytest.fixture(scope="session")
def client(server):
    """Session-scoped MCP client fixture."""
    return Client(server.mcp)


@pytest.fixture(scope="session")
def tools(server):
    """Session-scoped tools composition fixture."""
    return server.tools


@pytest.fixture(scope="session")
def resources(server):
    """Session-scoped resources composition fixture."""
    return server.resources


@pytest.fixture
def sample_search_results() -> dict[str, Any]:
    """Sample search results for testing."""
    return {
        "results": [
            {
                "path": "core_profiles/profiles_1d/electrons/temperature",
                "ids_name": "core_profiles",
                "score": 0.95,
                "documentation": "Electron temperature profile",
            },
            {
                "path": "equilibrium/time_slice/profiles_1d/psi",
                "ids_name": "equilibrium",
                "score": 0.88,
                "documentation": "Poloidal flux profile",
            },
        ],
        "total_results": 2,
    }


@pytest.fixture
def mcp_test_context():
    """Test context for MCP protocol testing."""
    return {
        "test_query": "plasma temperature",
        "test_ids": "core_profiles",
        "expected_tools": [
            "search_imas",
            "explain_concept",
            "get_overview",
            "analyze_ids_structure",
            "explore_relationships",
            "explore_identifiers",
            "export_ids",
            "export_physics_domain",
        ],
    }


@pytest.fixture
def workflow_test_data():
    """Test data for workflow testing."""
    return {
        "search_query": "core plasma transport",
        "analysis_target": "core_profiles",
        "export_domain": "transport",
        "concept_to_explain": "equilibrium",
    }
