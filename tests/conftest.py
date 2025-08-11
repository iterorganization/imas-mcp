"""
Test configuration and fixtures for the new MCP-based architecture.

This module provides test fixtures for the composition-based server architecture,
focusing on MCP protocol testing and feature validation.
"""

from typing import Any
from unittest.mock import patch

import pytest
from fastmcp import Client

from imas_mcp.server import Server

# Standard test IDS set for consistency across all tests
# This avoids re-embedding and ensures consistent performance
STANDARD_TEST_IDS_SET = {"equilibrium", "core_profiles"}


@pytest.fixture(autouse=True)
def disable_caching():
    """Automatically disable caching for all tests by making cache always miss."""
    # Patch the cache get method to always return None (cache miss)
    with patch("imas_mcp.search.decorators.cache._cache.get", return_value=None):
        # Also patch the set method to do nothing
        with patch("imas_mcp.search.decorators.cache._cache.set"):
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
