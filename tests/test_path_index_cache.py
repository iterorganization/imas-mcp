import logging
import pytest

import imas

from imas_mcp_server.path_index_cache import PathIndexCache

# Disable logging for test
logging.disable(logging.CRITICAL)


@pytest.fixture(scope="session")
def path_index_cache(tmp_path_factory):
    """Fixture for PathIndexCache instance with a temporary cache directory.

    Args:
        tmp_path_factory: pytest fixture that provides a temporary directory factory

    Keyword Args:
        version (str): The IMAS version to use. Defaults to "3.42.0".
        ids_set (set): A set of IDS names to index. Defaults to an empty set.
    """

    def _factory(version=None, ids_set={"pf_passive", "wall"}, filename=None):
        tmp_path = tmp_path_factory.getbasetemp()
        cache_dir = tmp_path
        return PathIndexCache(
            version=version, ids_set=ids_set, cache_dir=cache_dir, filename=filename
        )

    return _factory


def test_default_dd_version(path_index_cache):
    """Test the dd_version property."""
    version = None
    path_index = path_index_cache(version=version).path_index
    assert path_index.version == imas.ids_factory.IDSFactory().version


def test_dd_version(path_index_cache):
    """Test the dd_version property."""
    version = "3.42.0"
    path_index = path_index_cache(version=version).path_index
    assert path_index.version == version


def test_ids_set(path_index_cache):
    """Test the ids_set parameter."""
    custom_ids = {"pf_active"}
    cache = path_index_cache(ids_set=custom_ids)
    # The cache.path_index.ids should match our custom_ids
    assert cache.ids == custom_ids


def test_cache_dir(path_index_cache):
    """Test that the cache directory is created correctly."""
    cache = path_index_cache()
    # Check that the cache directory exists and is a directory
    assert cache.cache_dir.exists()
    assert cache.cache_dir.is_dir()


def test_cache_file_clear(path_index_cache):
    """Test that the cache file is created and cleared correctly."""
    cache = path_index_cache(filename="test_cache.pkl")
    assert cache.filepath.exists()
    cache.clear()
    assert not cache.filepath.exists()


if __name__ == "__main__":
    pytest.main([__file__])
