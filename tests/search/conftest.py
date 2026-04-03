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
    """Build cache key from DD version + cluster count."""
    try:
        info = gc.query(
            """
            MATCH (v:DDVersion) WITH count(v) AS vc
            MATCH (c:IMASSemanticCluster) WITH vc, count(c) AS cc
            RETURN vc, cc
            """
        )
        if info:
            raw = f"dd_versions={info[0]['vc']}_clusters={info[0]['cc']}"
            return hashlib.sha256(raw.encode()).hexdigest()[:16]
    except Exception:
        pass
    return "unknown"


def _load_cache() -> dict | None:
    """Load cached expected paths if fresh."""
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

    # Try loading cache
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
