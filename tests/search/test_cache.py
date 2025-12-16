"""Tests for search/cache.py module."""

from imas_codex.search.cache import SearchCache


class TestSearchCache:
    """Tests for the SearchCache class."""

    def test_initialization_with_custom_params(self):
        """Cache initializes with custom maxsize and ttl."""
        cache = SearchCache(maxsize=100, ttl=1800)

        assert cache.cache.maxsize == 100
        assert cache.cache.ttl == 1800

    def test_key_generation_deterministic(self):
        """Same parameters produce same cache key."""
        cache = SearchCache()
        key1 = cache._generate_key("test query", "core_profiles", 10, "auto")
        key2 = cache._generate_key("test query", "core_profiles", 10, "auto")
        key3 = cache._generate_key("different query", "core_profiles", 10, "auto")

        assert key1 == key2
        assert key1 != key3

    def test_key_generation_with_list_query(self):
        """List queries produce valid cache keys."""
        cache = SearchCache()
        key = cache._generate_key(["term1", "term2"], "ids", 5, "semantic")

        assert isinstance(key, str)
        assert len(key) == 16

    def test_set_and_get_round_trip(self):
        """Cache set and get work correctly."""
        cache = SearchCache()
        result = {"results": [{"path": "test/path"}], "total": 1}

        cache.set("test query", result, "ids", 10, "auto")
        cached = cache.get("test query", "ids", 10, "auto")

        assert cached is not None
        assert cached["cache_hit"] is True
        assert cached["results"] == result["results"]

    def test_cache_miss_returns_none(self):
        """Cache miss returns None and increments miss counter."""
        cache = SearchCache()
        result = cache.get("nonexistent query")

        assert result is None
        assert cache.stats["misses"] == 1

    def test_error_results_not_cached(self):
        """Error results are not stored in cache."""
        cache = SearchCache()
        error_result = {"error": "Something went wrong", "results": []}

        cache.set("error query", error_result)
        cached = cache.get("error query")

        assert cached is None

    def test_empty_results_not_cached(self):
        """Empty results are not stored in cache."""
        cache = SearchCache()
        empty_result = {"results": []}

        cache.set("empty query", empty_result)
        cached = cache.get("empty query")

        assert cached is None

    def test_clear_removes_all_entries(self):
        """Clear removes all cached entries."""
        cache = SearchCache()
        cache.set("query", {"results": [{"test": 1}]})
        cache.clear()

        assert cache.get("query") is None
        assert len(cache.cache) == 0

    def test_stats_track_hits_misses_and_sets(self):
        """Statistics track cache operations correctly."""
        cache = SearchCache()
        cache.set("q1", {"results": [{"path": "test"}]})
        cache.get("q1")  # hit
        cache.get("q2")  # miss
        cache.get("q1")  # hit

        stats = cache.get_stats()

        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["hit_rate"] == 2 / 3
        assert stats["total_requests"] == 3
