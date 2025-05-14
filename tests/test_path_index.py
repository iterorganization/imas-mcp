# Standard library imports
import logging
import tempfile
from pathlib import Path

# Third-party imports
import pytest

# Local imports
from imas_mcp_server.path_index import PathIndex
from imas_mcp_server.path_index_cache import PathIndexCache

# Disable logging for test
logging.disable(logging.CRITICAL)


@pytest.fixture(scope="session")
def path_index_factory(tmp_path_factory):
    """Fixture for PathIndexCache instance with a temporary cache directory.

    Args:
        tmp_path_factory: pytest fixture that provides a temporary directory factory

    Keyword Args:
        version (str): The IMAS version to use. Defaults to "3.42.0".
        ids_set (set): A set of IDS names to index. Defaults to an empty set.
    """

    def _factory(version=None, ids_set={"pf_passive"}):
        tmp_path = tmp_path_factory.getbasetemp()
        cache_dir = tmp_path
        return PathIndexCache(
            version=version, ids_set=ids_set, cache_dir=cache_dir
        ).path_index

    return _factory


def test_path_index_sets(path_index_factory):
    """Test that the index paths are created correctly."""
    path_index = path_index_factory()
    path_subset = {"pf_passive/ids_properties/homogeneous_time", "pf_passive/loop/name"}
    assert path_index.paths.issuperset(path_subset)

    assert not path_index.paths.issuperset(
        {"pf_passive/ids_properties", "pf_passive/loop"}
    )


def test_docs(path_index_factory):
    """Test that the index documentation is created correctly."""
    path_index = path_index_factory(version="4.0.0", ids_set={"pf_active"})
    # Check that documentation contains both parts, but don't enforce exact format
    doc = path_index.docs["pf_active/coil/name"]
    assert "Active PF coils" in doc
    assert "Short string identifier (unique for a given device)" in doc


@pytest.fixture
def whoosh_path_index():
    """Fixture to create a test PathIndex with Whoosh indexing and sample paths and documentation."""
    # Create a temporary directory for the index
    with tempfile.TemporaryDirectory() as temp_dir:
        index_dir = Path(temp_dir)
        index = PathIndex(version="3.42.0", index_dir=index_dir)

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

        yield index


def test_whoosh_search_by_keywords(whoosh_path_index):
    """Test the search_by_keywords method using Whoosh."""
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

    # Test search for plasma current
    results = whoosh_path_index.search_by_keywords("plasma current measurement")
    assert len(results) >= 1
    paths = [r["path"] for r in results]
    assert "magnetics/time_trace/ip" in paths


if __name__ == "__main__":
    pytest.main([__file__])
