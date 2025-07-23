import json
from typing import Any, Dict

import pytest
from fastmcp import Client

from imas_mcp.graph_analyzer import IMASGraphAnalyzer
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.semantic_search import SemanticSearch
from imas_mcp.server import Server


# Standard test IDS set for consistency across all tests
# This avoids re-embedding and ensures consistent performance
STANDARD_TEST_IDS_SET = {"equilibrium", "core_profiles"}


def extract_result(result):
    """Extract JSON from FastMCP TextContent result."""
    if hasattr(result, "__iter__") and len(result) > 0:
        if hasattr(result[0], "text"):
            return json.loads(result[0].text)
    return result


@pytest.fixture(scope="session")
def server() -> Server:
    """Session-scoped server fixture for performance."""
    return Server(ids_set=STANDARD_TEST_IDS_SET)


@pytest.fixture(scope="session")
def client(server):
    """Session-scoped client fixture."""
    return Client(server.mcp)


@pytest.fixture(scope="session")
def tools(server):
    """Session-scoped tools fixture."""
    return server.tools


@pytest.fixture(scope="session")
def document_store(tools) -> DocumentStore:
    """Session-scoped document store fixture."""
    return tools.document_store


@pytest.fixture(scope="session")
def semantic_search(tools) -> SemanticSearch:
    """Session-scoped semantic search fixture."""
    return tools.semantic_search


@pytest.fixture(scope="session")
def graph_analyzer(tools) -> IMASGraphAnalyzer:
    """Session-scoped graph analyzer fixture."""
    return tools.graph_analyzer


@pytest.fixture(scope="session")
def search_cache(tools):
    """Session-scoped search cache fixture."""
    return tools.search_cache


@pytest.fixture(scope="session")
def search_composer(tools):
    """Session-scoped search composer fixture."""
    return tools.search_composer


@pytest.fixture
def sample_search_results() -> Dict[str, Any]:
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
