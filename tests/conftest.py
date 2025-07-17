import json
from functools import cached_property
from typing import Any, Dict

import pytest
from fastmcp import Client

from imas_mcp.graph_analyzer import IMASGraphAnalyzer
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.semantic_search import SemanticSearch
from imas_mcp.server import Server


def extract_result(result):
    """Extract JSON from FastMCP TextContent result."""
    if hasattr(result, "__iter__") and len(result) > 0:
        if hasattr(result[0], "text"):
            return json.loads(result[0].text)
    return result


class TestServerFixture:
    """Long-lived server fixture for expensive operations using cached_property."""

    def __init__(self):
        self.server = Server(ids_set={"equilibrium", "core_profiles"})
        self.client = Client(self.server.mcp)

    @cached_property
    def document_store(self) -> DocumentStore:
        """Get or create document store - cached for performance."""
        return self.server.document_store

    @cached_property
    def semantic_search(self) -> SemanticSearch:
        """Get or create semantic search - cached for performance."""
        return self.server.semantic_search

    @cached_property
    def graph_analyzer(self) -> IMASGraphAnalyzer:
        """Get or create graph analyzer - cached for performance."""
        return self.server.graph_analyzer


# Session-scoped fixture for expensive server initialization
@pytest.fixture(scope="session")
def test_server() -> TestServerFixture:
    """Session-scoped server fixture for performance."""
    return TestServerFixture()


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
