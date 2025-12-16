"""
Search caching using cachetools for performance optimization.

This module provides a simple wrapper around cachetools for search result caching
with TTL and size limits.
"""

import hashlib
import json
import logging
from typing import Any

from cachetools import TTLCache

logger = logging.getLogger(__name__)


class SearchCache:
    """Search result cache using cachetools with TTL and size limits."""

    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        """
        Initialize cache with size and TTL limits.

        Args:
            maxsize: Maximum number of cached items
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.stats = {"hits": 0, "misses": 0, "sets": 0}

    def _generate_key(
        self,
        query: str | list[str],
        ids_name: str | None = None,
        max_results: int = 10,
        search_mode: str = "auto",
    ) -> str:
        """Generate a cache key from search parameters."""
        # Normalize query
        if isinstance(query, list):
            query_str = " ".join(sorted(query))
        else:
            query_str = str(query)

        # Create cache key data
        key_data = {
            "query": query_str.lower().strip(),
            "ids_name": ids_name,
            "max_results": max_results,
            "search_mode": search_mode,
        }

        # Generate stable hash
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get(
        self,
        query: str | list[str],
        ids_name: str | None = None,
        max_results: int = 10,
        search_mode: str = "auto",
    ) -> dict[str, Any] | None:
        """Get cached search result."""
        key = self._generate_key(query, ids_name, max_results, search_mode)

        try:
            result = self.cache.get(key)
            if result is not None:
                self.stats["hits"] += 1
                # Mark as cache hit
                if isinstance(result, dict):
                    result = result.copy()
                    result["cache_hit"] = True
                logger.debug(f"Cache hit for key: {key}")
                return result
            else:
                self.stats["misses"] += 1
                logger.debug(f"Cache miss for key: {key}")
                return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            self.stats["misses"] += 1
            return None

    def set(
        self,
        query: str | list[str],
        result: dict[str, Any],
        ids_name: str | None = None,
        max_results: int = 10,
        search_mode: str = "auto",
    ) -> None:
        """Cache a search result."""
        key = self._generate_key(query, ids_name, max_results, search_mode)

        try:
            # Don't cache error results or empty results
            if result.get("error") or not result.get("results"):
                return

            # Remove cache_hit flag before storing
            result_to_cache = result.copy()
            result_to_cache.pop("cache_hit", None)

            self.cache[key] = result_to_cache
            self.stats["sets"] += 1
            logger.debug(f"Cached result for key: {key}")
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        logger.info("Search cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_size": self.cache.maxsize,
            "ttl": self.cache.ttl,
        }
