"""
Cache decorator for search results.

Provides result caching with configurable TTL and cache key strategies.
"""

import functools
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class CacheEntry:
    """Cache entry with expiration time."""

    def __init__(self, value: Any, ttl: int):
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return time.time() > self.expires_at


class SimpleCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if key in self._cache:
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            return entry.value

        return None

    def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in cache with TTL."""
        # Remove oldest entries if cache is full
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = CacheEntry(value, ttl)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


# Global cache instance
_cache = SimpleCache()


def build_cache_key(args: tuple, kwargs: dict, strategy: str = "semantic") -> str:
    """
    Build cache key based on function arguments and strategy.

    Args:
        args: Positional arguments
        kwargs: Keyword arguments
        strategy: Cache key strategy ("semantic", "exact", "query_only")

    Returns:
        Cache key string
    """
    # Determine if first arg is 'self' (method) or not (function)
    # If args[0] has methods like get_tool_name, it's likely a 'self'
    skip_first = (
        len(args) > 0
        and hasattr(args[0], "__dict__")
        and (
            hasattr(args[0], "get_tool_name")
            or hasattr(args[0], "_create_error_response")
        )
    )

    effective_args = args[1:] if skip_first else args

    if strategy == "query_only":
        # Only use the query for caching
        query = kwargs.get("query", effective_args[0] if effective_args else "")
        key_data = {"query": query}
    elif strategy == "exact":
        # Use all arguments for exact matching
        key_data = {
            "args": [str(arg) for arg in effective_args],
            "kwargs": {k: v for k, v in kwargs.items() if k != "ctx"},
        }
    else:  # semantic - default strategy
        # Use positional args and important kwargs
        # Include all arguments except 'self' and 'ctx'
        key_data = {
            "args": [str(arg) for arg in effective_args],
            "kwargs": {k: v for k, v in kwargs.items() if k != "ctx"},
        }

    # Create deterministic hash
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def cache_results(ttl: int = 300, key_strategy: str = "semantic") -> Callable[[F], F]:
    """
    Decorator to cache function results.

    Args:
        ttl: Time to live in seconds (default: 300 = 5 minutes)
        key_strategy: Cache key strategy ("semantic", "exact", "query_only")

    Returns:
        Decorated function with caching
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            cache_key = build_cache_key(args, kwargs, strategy=key_strategy)

            # Try to get from cache
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                # Add cache hit indicator
                if isinstance(cached_result, dict):
                    cached_result = cached_result.copy()
                    cached_result["_cache_hit"] = True
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache the result if it's successful
            if isinstance(result, dict) and "error" not in result:
                # Add cache miss indicator
                if isinstance(result, dict):
                    result = result.copy()
                    result["_cache_hit"] = False

                _cache.set(cache_key, result, ttl)

            return result

        return wrapper  # type: ignore

    return decorator


def no_cache_results(
    ttl: int = 300, key_strategy: str = "semantic"
) -> Callable[[F], F]:
    """
    Decorator that bypasses caching (for testing).

    This decorator has the same signature as cache_results but never caches,
    making it perfect for replacing cache_results during testing.

    Args:
        ttl: Ignored (for signature compatibility)
        key_strategy: Ignored (for signature compatibility)

    Returns:
        Decorated function without caching
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute function directly without caching
            return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def clear_cache() -> None:
    """Clear the global cache."""
    _cache.clear()


def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics."""
    return {
        "size": _cache.size(),
        "max_size": _cache.max_size,
    }
