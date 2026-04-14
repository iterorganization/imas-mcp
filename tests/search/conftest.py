"""Search test fixtures — expected path cache and graph helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import pytest

from tests.search.benchmark_data import ALL_QUERIES

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).parent / ".expected_paths_cache.json"
_CACHE_MAX_AGE_DAYS = 7


def _cache_key(gc) -> str:
    """Build cache key from DD version + node count hash."""
    try:
        info = gc.query(
            """
            MATCH (v:DDVersion)
            WITH count(v) AS vc
            OPTIONAL MATCH (v2:DDVersion) WHERE v2.is_current = true
            WITH vc, v2.version AS current_version
            MATCH (n:IMASNode) WHERE n.node_category = 'data'
            WITH vc, current_version, count(n) AS nc
            MATCH (c:IMASSemanticCluster)
            WITH vc, current_version, nc, count(c) AS cc
            RETURN vc, current_version, nc, cc
            """
        )
        if info:
            row = info[0]
            raw = (
                f"dd_versions={row['vc']}"
                f"_current={row.get('current_version', 'unknown')}"
                f"_nodes={row['nc']}"
                f"_clusters={row['cc']}"
            )
            return hashlib.sha256(raw.encode()).hexdigest()[:16]
    except Exception:
        pass
    return "unknown"


def _load_cache(gc=None) -> dict | None:
    """Load cached expected paths if fresh.

    Parameters
    ----------
    gc : GraphClient | None
        When provided, the stored cache key is compared against the current
        graph state and the cache is invalidated on mismatch.
    """
    if not _CACHE_FILE.exists():
        return None
    try:
        data = json.loads(_CACHE_FILE.read_text())
        # Check age
        cached_at = data.get("cached_at", 0)
        age_days = (time.time() - cached_at) / 86400
        if age_days > _CACHE_MAX_AGE_DAYS:
            logger.info("Expected paths cache is %.1f days old, regenerating", age_days)
            return None
        # Check graph version key if gc is available
        if gc is not None:
            current_key = _cache_key(gc)
            cached_key = data.get("cache_key", "")
            if current_key != "unknown" and cached_key != current_key:
                logger.info(
                    "Cache key mismatch (cached=%s, current=%s), regenerating",
                    cached_key,
                    current_key,
                )
                return None
        return data
    except Exception:
        return None


def _save_cache(paths: dict[str, list[str]], cache_key: str) -> None:
    """Save expanded expected paths to cache file."""
    data = {
        "cache_key": cache_key,
        "cached_at": time.time(),
        "paths": paths,
    }
    _CACHE_FILE.write_text(json.dumps(data, indent=2, sort_keys=True))
    logger.info("Saved expected paths cache (%d queries)", len(paths))


@pytest.fixture(scope="session")
def expanded_expected_paths(request):
    """Session-scoped fixture providing auto-expanded expected paths.

    On first run or when cache is stale:
    1. Generates expected paths for all benchmark queries from the graph
    2. Caches to .expected_paths_cache.json

    On subsequent runs: loads from cache.

    Use --regenerate-expected flag to force regeneration.

    Returns dict[str, set[str]] mapping query_text -> expanded path set.
    Falls back to hand-curated paths if graph is unavailable.
    """
    force_regen = request.config.getoption("--regenerate-expected", default=False)

    # Try loading cache (without gc first — fast path for age-only check)
    if not force_regen:
        cached = _load_cache()
        if cached and "paths" in cached:
            return {qt: set(paths) for qt, paths in cached["paths"].items()}

    # Need graph to generate
    try:
        from imas_codex.graph.client import GraphClient

        gc = GraphClient()
        gc.get_stats()
    except Exception:
        logger.info("Graph unavailable — using hand-curated expected paths only")
        return {q.query_text: set(q.expected_paths) for q in ALL_QUERIES}

    try:
        # Re-validate cache with graph connection for key-based freshness check
        if not force_regen:
            cached = _load_cache(gc)
            if cached and "paths" in cached:
                return {qt: set(paths) for qt, paths in cached["paths"].items()}

        from tests.search.generate_expected_paths import generate_all_expected_paths

        key = _cache_key(gc)
        all_paths = generate_all_expected_paths(ALL_QUERIES, gc)

        # Convert sets to lists for JSON serialization
        serializable = {qt: sorted(paths) for qt, paths in all_paths.items()}
        _save_cache(serializable, key)

        return all_paths
    except Exception:
        logger.warning("Failed to generate expected paths", exc_info=True)
        return {q.query_text: set(q.expected_paths) for q in ALL_QUERIES}
    finally:
        gc.close()


def pytest_addoption(parser):
    """Add --regenerate-expected CLI option."""
    parser.addoption(
        "--regenerate-expected",
        action="store_true",
        default=False,
        help="Force regeneration of expected paths cache",
    )
