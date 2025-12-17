"""
Test configuration and fixtures for the new MCP-based architecture.

This module provides test fixtures for the composition-based server architecture,
focusing on MCP protocol testing and feature validation.
"""

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from dotenv import load_dotenv
from fastmcp import Client

from imas_codex.clusters.search import ClusterSearchResult
from imas_codex.embeddings.encoder import Encoder
from imas_codex.search.document_store import Document, DocumentMetadata, DocumentStore
from imas_codex.search.engines.base_engine import MockSearchEngine
from imas_codex.server import Server

# Load .env file with override=True to ensure test environment uses .env values
# This fixes issues where empty or stale shell environment variables persist
load_dotenv(override=True)


def pytest_addoption(parser):
    parser.addoption(
        "--embedding-model",
        action="store",
        default=None,
        help="Embedding model to use for tests (default: all-MiniLM-L6-v2 or from env)",
    )


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "api_embedding: mark test as requiring API embedding model"
    )


@pytest.fixture(scope="session")
def embedding_model_name(request):
    """Get the embedding model name from command line option."""
    return request.config.getoption("--embedding-model")


@pytest.fixture(scope="session", autouse=True)
def configure_embedding_model(embedding_model_name):
    """Configure the embedding model environment variable."""
    # Store original value from .env/shell for tests that need the real API config
    if "IMAS_CODEX_EMBEDDING_MODEL" in os.environ:
        os.environ["IMAS_CODEX_ORIGINAL_EMBEDDING_MODEL"] = os.environ[
            "IMAS_CODEX_EMBEDDING_MODEL"
        ]

    if embedding_model_name:
        os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = embedding_model_name
    else:
        # If OPENAI_API_KEY is set, use API model; otherwise use local
        # This allows tests to work with .env configuration
        if not os.environ.get("IMAS_CODEX_EMBEDDING_MODEL"):
            if os.environ.get("OPENAI_API_KEY"):
                os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = (
                    "openai/text-embedding-3-small"
                )
            else:
                os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"


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
        create_mock_document("equilibrium/time_slice/boundary/psi", "equilibrium"),
        create_mock_document("equilibrium/time_slice/boundary/psi_norm", "equilibrium"),
        create_mock_document("equilibrium/time_slice/boundary/type", "equilibrium"),
    ]


def create_mock_clusters() -> list[dict]:
    """Create mock cluster data for testing."""
    return [
        {
            "id": 0,
            "label": "Electron Temperature Profiles",
            "description": "Temperature measurements for electrons",
            "is_cross_ids": False,
            "ids_names": ["core_profiles"],
            "paths": [
                "core_profiles/profiles_1d/electrons/temperature",
                "core_profiles/profiles_1d/electrons/temperature_fit",
            ],
            "similarity_score": 0.95,
            "cluster_similarity": 0.87,
        },
        {
            "id": 1,
            "label": "Magnetic Field Components",
            "description": "Magnetic field measurements and derived quantities",
            "is_cross_ids": True,
            "ids_names": ["equilibrium", "core_profiles"],
            "paths": [
                "equilibrium/time_slice/profiles_2d/b_field_r",
                "equilibrium/time_slice/profiles_2d/b_field_z",
            ],
            "similarity_score": 0.88,
            "cluster_similarity": 0.82,
        },
        {
            "id": 2,
            "label": "Boundary Conditions",
            "description": "Plasma boundary and separatrix data",
            "is_cross_ids": False,
            "ids_names": ["equilibrium"],
            "paths": [
                "equilibrium/time_slice/boundary/psi",
                "equilibrium/time_slice/boundary/psi_norm",
                "equilibrium/time_slice/boundary/type",
            ],
            "similarity_score": 0.92,
            "cluster_similarity": 0.79,
        },
    ]


def create_mock_cluster_search_results(query: str) -> list[ClusterSearchResult]:
    """Create mock cluster search results for testing."""
    mock_clusters = create_mock_clusters()
    return [
        ClusterSearchResult(
            cluster_id=c["id"],
            label=c["label"],
            description=c["description"],
            is_cross_ids=c["is_cross_ids"],
            ids_names=c["ids_names"],
            paths=c["paths"],
            similarity_score=c["similarity_score"],
            cluster_similarity=c["cluster_similarity"],
        )
        for c in mock_clusters[:2]  # Return first 2 clusters
    ]


@pytest.fixture(autouse=True)
def temporary_embedding_cache_dir(tmp_path_factory, monkeypatch):
    """Keep embedding cache files isolated per test."""
    temp_dir = tmp_path_factory.mktemp("embedding_cache")
    monkeypatch.setattr(
        Encoder,
        "_get_cache_directory",
        lambda self, _temp_dir=temp_dir: _temp_dir,
    )
    yield


@pytest.fixture(autouse=True)
def disable_caching():
    """Automatically disable caching for all tests by making cache always miss."""
    # Patch the cache get method to always return None (cache miss)
    with patch("imas_codex.search.decorators.cache._cache.get", return_value=None):
        # Also patch the set method to do nothing
        with patch("imas_codex.search.decorators.cache._cache.set"):
            yield


@pytest.fixture(scope="session", autouse=True)
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
        with patch("imas_codex.server.SemanticSearch") as mock_semantic:
            mock_semantic_instance = MagicMock()
            mock_semantic_instance._initialize.return_value = None
            mock_semantic.return_value = mock_semantic_instance

            # Mock search engine methods to prevent heavy execution
            mock_engine = MockSearchEngine()
            with (
                patch(
                    "imas_codex.search.engines.semantic_engine.SemanticSearchEngine.search",
                    side_effect=mock_engine.search,
                ),
                patch(
                    "imas_codex.search.engines.lexical_engine.LexicalSearchEngine.search",
                    side_effect=mock_engine.search,
                ),
                patch(
                    "imas_codex.search.engines.hybrid_engine.HybridSearchEngine.search",
                    side_effect=mock_engine.search,
                ),
            ):
                # Mock Clusters class to prevent loading cluster files
                mock_clusters = create_mock_clusters()
                with patch("imas_codex.core.clusters.Clusters") as mock_clusters_class:
                    mock_clusters_instance = MagicMock()
                    mock_clusters_instance.is_available.return_value = True
                    mock_clusters_instance.get_clusters.return_value = mock_clusters
                    mock_clusters_class.return_value = mock_clusters_instance

                    # Also patch in the tools module where it's imported
                    with patch(
                        "imas_codex.tools.clusters_tool.Clusters"
                    ) as mock_clusters_tool:
                        mock_clusters_tool.return_value = mock_clusters_instance

                        # Mock ClusterSearcher to return mock results
                        with patch(
                            "imas_codex.tools.clusters_tool.ClusterSearcher"
                        ) as mock_searcher_class:
                            mock_searcher = MagicMock()
                            mock_searcher.search_by_path.return_value = (
                                create_mock_cluster_search_results("path")
                            )
                            mock_searcher.search_by_text.return_value = (
                                create_mock_cluster_search_results("text")
                            )
                            mock_searcher_class.return_value = mock_searcher

                            # Mock Encoder to prevent model loading
                            with patch(
                                "imas_codex.tools.clusters_tool.Encoder"
                            ) as mock_encoder_class:
                                mock_encoder = MagicMock()
                                mock_encoder.encode.return_value = np.zeros(
                                    (1, 384), dtype=np.float32
                                )
                                mock_encoder_class.return_value = mock_encoder

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
            "check_imas_paths",
            "fetch_imas_paths",
            "get_imas_overview",
            "get_imas_identifiers",
            "list_imas_paths",
            "search_imas_clusters",
            "search_imas_paths",
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
