#!/usr/bin/env python
"""
Unit tests for the keyword search functionality in Whoosh-based PathIndex.
"""

import pytest
import tempfile
from pathlib import Path
from imas_mcp_server.path_index import PathIndex


@pytest.fixture
def path_index():
    """Fixture to create a test PathIndex with sample paths and documentation."""
    # Create a memory-based index without Whoosh for faster testing
    index = PathIndex(version="3.42.0")

    # Add some sample paths with documentation
    index.add_path(
        "equilibrium/time_slice/profiles_1d/q_safety_factor",
        "Safety factor q profile as a function of normalized poloidal flux.",
    )
    index.add_path(
        "core_profiles/time_slice/electrons/density",
        "Electron density profile as a function of normalized poloidal flux.",
    )
    index.add_path(
        "core_profiles/time_slice/ions/1/density",
        "Ion density profile for the first ion species as a function of normalized poloidal flux.",
    )
    index.add_path(
        "magnetics/time_trace/ip",
        "Plasma current time trace measured by magnetic diagnostics.",
    )

    return index


@pytest.fixture
def whoosh_path_index():
    """Fixture to create a test PathIndex with Whoosh indexing and sample paths."""
    # Create a temporary directory for the Whoosh index
    with tempfile.TemporaryDirectory() as temp_dir:
        index_dir = Path(temp_dir)
        index = PathIndex(version="3.42.0", index_dir=index_dir)

        # Add the same sample paths as the in-memory index
        index.add_path(
            "equilibrium/time_slice/profiles_1d/q_safety_factor",
            "Safety factor q profile as a function of normalized poloidal flux.",
        )
        index.add_path(
            "core_profiles/time_slice/electrons/density",
            "Electron density profile as a function of normalized poloidal flux.",
        )
        index.add_path(
            "core_profiles/time_slice/ions/1/density",
            "Ion density profile for the first ion species as a function of normalized poloidal flux.",
        )
        index.add_path(
            "magnetics/time_trace/ip",
            "Plasma current time trace measured by magnetic diagnostics.",
        )

        yield index


def test_search_by_keywords(path_index):
    """Test the search_by_keywords method using in-memory fallback."""
    # Test search for safety factor
    results = path_index.search_by_keywords("safety factor profile")
    assert len(results) >= 1
    paths = [r["path"] for r in results]
    assert "equilibrium/time_slice/profiles_1d/q_safety_factor" in paths

    # Test search for electron density
    results = path_index.search_by_keywords("electron density profile")
    assert len(results) >= 1
    paths = [r["path"] for r in results]
    assert "core_profiles/time_slice/electrons/density" in paths

    # Test search for plasma current
    results = path_index.search_by_keywords("plasma current measurement")
    assert len(results) >= 1
    paths = [r["path"] for r in results]
    assert "magnetics/time_trace/ip" in paths


def test_search_with_partial_matches(path_index):
    """Test search with partial keyword matches."""
    # Search should match both electron and ion density profiles
    results = path_index.search_by_keywords("density profiles")
    assert len(results) >= 2
    paths = [result["path"] for result in results]
    assert "core_profiles/time_slice/electrons/density" in paths
    assert "core_profiles/time_slice/ions/1/density" in paths


def test_search_with_no_matches(path_index):
    """Test search with no matches."""
    results = path_index.search_by_keywords("something that does not exist")
    assert len(results) == 0


def test_search_result_structure(path_index):
    """Test that search results have the expected structure."""
    results = path_index.search_by_keywords("safety factor")
    assert len(results) >= 1

    # Check that each result has the expected fields
    result = results[0]
    assert "path" in result
    assert "score" in result
    assert "doc" in result

    # Check that the score is a positive number
    assert result["score"] > 0


def test_whoosh_search_by_keywords(whoosh_path_index):
    """Test the search_by_keywords method using the Whoosh index."""
    # Test search for safety factor
    results = whoosh_path_index.search_by_keywords("safety factor profile")
    assert len(results) >= 1
    paths = [r["path"] for r in results]
    assert "equilibrium/time_slice/profiles_1d/q_safety_factor" in paths

    # Test search for electron density
    results = whoosh_path_index.search_by_keywords("electron density profile")
    assert len(results) >= 1
    paths = [r["path"] for r in results]
    assert "core_profiles/time_slice/electrons/density" in paths


def test_whoosh_search_result_structure(whoosh_path_index):
    """Test that Whoosh search results have the expected structure."""
    results = whoosh_path_index.search_by_keywords("safety factor")
    assert len(results) >= 1

    # Check that each result has the expected fields
    result = results[0]
    assert "path" in result
    assert "score" in result
    assert "doc" in result

    # Check that the score is a positive number
    assert result["score"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
