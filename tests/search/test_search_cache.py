"""
Tests for search caching functionality.
"""

import pytest
import time
from unittest.mock import patch

from imas_mcp.search.cache import SearchCache


class TestSearchCache:
    """Test suite for SearchCache class."""

    def test_cache_initialization(self):
        """Test cache initialization with default and custom parameters."""
        # Default initialization
        cache = SearchCache()
        assert cache.cache.maxsize == 1000
        assert cache.cache.ttl == 3600
        assert cache.stats == {"hits": 0, "misses": 0, "sets": 0}

        # Custom initialization
        cache = SearchCache(maxsize=500, ttl=1800)
        assert cache.cache.maxsize == 500
        assert cache.cache.ttl == 1800

    def test_generate_key_consistency(self):
        """Test that cache key generation is consistent."""
        cache = SearchCache()

        # Same parameters should generate same key
        key1 = cache._generate_key("test query", "core_profiles", 10, "auto")
        key2 = cache._generate_key("test query", "core_profiles", 10, "auto")
        assert key1 == key2

        # Different parameters should generate different keys
        key3 = cache._generate_key("different query", "core_profiles", 10, "auto")
        assert key1 != key3

        key4 = cache._generate_key("test query", "equilibrium", 10, "auto")
        assert key1 != key4

    def test_generate_key_with_list_query(self):
        """Test cache key generation with list queries."""
        cache = SearchCache()

        # List queries should be normalized (sorted)
        key1 = cache._generate_key(["plasma", "temperature"], None, 10, "auto")
        key2 = cache._generate_key(["temperature", "plasma"], None, 10, "auto")
        assert key1 == key2

        # Different lists should generate different keys
        key3 = cache._generate_key(["plasma", "density"], None, 10, "auto")
        assert key1 != key3

    def test_cache_miss(self):
        """Test cache miss behavior."""
        cache = SearchCache()

        result = cache.get("nonexistent query")
        assert result is None
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 0

    def test_cache_set_and_hit(self):
        """Test setting and retrieving cached results."""
        cache = SearchCache()

        # Set a result
        test_result = {
            "results": [{"path": "test/path", "score": 0.9}],
            "total_results": 1,
            "search_strategy": "semantic",
        }

        cache.set("test query", test_result)
        assert cache.stats["sets"] == 1

        # Retrieve the result
        cached_result = cache.get("test query")
        assert cached_result is not None
        assert cached_result["cache_hit"] is True
        assert cached_result["results"] == test_result["results"]
        assert cache.stats["hits"] == 1

    def test_cache_does_not_store_errors(self):
        """Test that error results are not cached."""
        cache = SearchCache()

        # Error result should not be cached
        error_result = {
            "results": [],
            "error": "Some error occurred",
            "total_results": 0,
        }

        cache.set("error query", error_result)
        assert cache.stats["sets"] == 0  # Should not increment

        # Empty results should not be cached
        empty_result = {
            "results": [],
            "total_results": 0,
            "search_strategy": "semantic",
        }

        cache.set("empty query", empty_result)
        assert cache.stats["sets"] == 0  # Should not increment

    def test_cache_clears_cache_hit_flag(self):
        """Test that cache_hit flag is removed before storing."""
        cache = SearchCache()

        # Result with cache_hit flag
        test_result = {
            "results": [{"path": "test/path"}],
            "total_results": 1,
            "cache_hit": True,  # This should be removed
        }

        cache.set("test query", test_result)

        # Retrieve and check that original didn't have cache_hit removed
        cached_result = cache.get("test query")
        assert "cache_hit" in cached_result  # Added by get()
        assert cached_result["cache_hit"] is True

        # But the stored version shouldn't have had it originally
        assert cache.cache[cache._generate_key("test query")].get("cache_hit") is None

    def test_cache_with_all_parameters(self):
        """Test caching with all search parameters."""
        cache = SearchCache()

        test_result = {"results": [{"path": "test/path"}], "total_results": 1}

        # Set with all parameters
        cache.set(
            query="test query",
            result=test_result,
            ids_name="core_profiles",
            max_results=20,
            search_mode="semantic",
        )

        # Get with same parameters
        cached_result = cache.get(
            query="test query",
            ids_name="core_profiles",
            max_results=20,
            search_mode="semantic",
        )

        assert cached_result is not None
        assert cached_result["cache_hit"] is True

        # Get with different parameters should miss
        different_result = cache.get(
            query="test query",
            ids_name="equilibrium",  # Different IDS
            max_results=20,
            search_mode="semantic",
        )

        assert different_result is None

    def test_cache_clear(self):
        """Test cache clearing functionality."""
        cache = SearchCache()

        # Add some data
        test_result = {"results": [{"path": "test"}], "total_results": 1}
        cache.set("query1", test_result)
        cache.set("query2", test_result)

        assert len(cache.cache) == 2
        assert cache.stats["sets"] == 2

        # Clear cache
        cache.clear()

        assert len(cache.cache) == 0
        # Stats should remain
        assert cache.stats["sets"] == 2

    def test_cache_stats(self):
        """Test cache statistics calculation."""
        cache = SearchCache()

        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0
        assert stats["total_requests"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["cache_size"] == 0
        assert stats["max_size"] == 1000
        assert stats["ttl"] == 3600

        # Add some activity
        test_result = {"results": [{"path": "test"}], "total_results": 1}
        cache.set("query1", test_result)
        cache.get("query1")  # hit
        cache.get("query2")  # miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == 0.5
        assert stats["cache_size"] == 1

    def test_cache_error_handling(self):
        """Test cache behavior with errors."""
        cache = SearchCache()

        # Mock cachetools to raise exceptions
        with patch.object(cache.cache, "get", side_effect=Exception("Cache error")):
            result = cache.get("test query")
            assert result is None
            assert cache.stats["misses"] == 1

        with patch.object(
            cache.cache, "__setitem__", side_effect=Exception("Set error")
        ):
            test_result = {"results": [{"path": "test"}], "total_results": 1}
            cache.set("test query", test_result)
            # Should not crash, just log warning

    @pytest.mark.parametrize(
        "query_input,expected_normalized",
        [
            ("Simple Query", "simple query"),
            (["plasma", "temperature"], "plasma temperature"),
            (
                ["Temperature", "Plasma"],
                "plasma temperature",
            ),  # Should be sorted and lowercased
            ("  Spaced Query  ", "spaced query"),
        ],
    )
    def test_query_normalization(self, query_input, expected_normalized):
        """Test various query input formats are normalized consistently."""
        cache = SearchCache()

        key = cache._generate_key(query_input)

        # Generate key with expected normalized form
        expected_key = cache._generate_key(expected_normalized)

        assert key == expected_key

    def test_cache_maxsize_limit(self):
        """Test that cache respects maxsize limit."""
        # Small cache for testing
        cache = SearchCache(maxsize=2, ttl=3600)

        # Add items up to limit
        test_result = {"results": [{"path": "test"}], "total_results": 1}
        cache.set("query1", test_result)
        cache.set("query2", test_result)

        assert len(cache.cache) == 2

        # Adding third item should evict oldest (cachetools handles this)
        cache.set("query3", test_result)

        # Cache size should still be 2 (at most)
        assert len(cache.cache) <= 2

    def test_ttl_expiration(self):
        """Test TTL expiration behavior."""
        # Very short TTL for testing
        cache = SearchCache(maxsize=10, ttl=1)

        test_result = {"results": [{"path": "test"}], "total_results": 1}
        cache.set("query1", test_result)

        # Immediate retrieval should work
        result = cache.get("query1")
        assert result is not None
        assert cache.stats["hits"] == 1

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired now
        result = cache.get("query1")
        assert result is None
        assert cache.stats["misses"] == 1
