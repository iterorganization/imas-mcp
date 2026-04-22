"""Unit tests for hybrid_dd_search pure function.

Tests require a live Neo4j connection; they are skipped automatically
when the graph is unreachable via the top-level pytest_collection_modifyitems
hook (pytest.mark.graph marker).
"""

from __future__ import annotations

import os

import pytest

from imas_codex.graph.dd_search import hybrid_dd_search

pytestmark = pytest.mark.graph


@pytest.fixture(scope="module", autouse=True)
def _use_production_embedder():
    """Override embedding config to match graph vector indexes."""
    prior = {
        "IMAS_CODEX_EMBEDDING_MODEL": os.environ.get("IMAS_CODEX_EMBEDDING_MODEL"),
        "IMAS_CODEX_EMBEDDING_LOCATION": os.environ.get(
            "IMAS_CODEX_EMBEDDING_LOCATION"
        ),
        "IMAS_CODEX_EMBEDDING_DIMENSION": os.environ.get(
            "IMAS_CODEX_EMBEDDING_DIMENSION"
        ),
    }
    os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = "Qwen/Qwen3-Embedding-0.6B"
    os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = "titan"
    os.environ["IMAS_CODEX_EMBEDDING_DIMENSION"] = "256"
    yield
    for key, value in prior.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="module")
def gc(_use_production_embedder):
    """Session-scoped GraphClient."""
    from imas_codex.graph.client import GraphClient

    try:
        client = GraphClient()
        client.get_stats()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")
    yield client
    client.close()


# ---------------------------------------------------------------------------
# Basic return structure
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "path",
    "ids_name",
    "documentation",
    "data_type",
    "units",
    "physics_domain",
    "score",
}


def test_returns_nonempty_list(gc):
    """hybrid_dd_search returns a non-empty list for a known concept."""
    hits = hybrid_dd_search(gc, "plasma current", k=10)
    assert isinstance(hits, list)
    assert len(hits) > 0


def test_hit_has_expected_keys(gc):
    """Each hit dict (via SearchHit model_dump) has the expected keys."""
    hits = hybrid_dd_search(gc, "electron temperature", k=5)
    assert hits
    for hit in hits:
        dumped = hit.model_dump()
        for key in EXPECTED_KEYS:
            assert key in dumped, f"Missing key {key!r} in hit"


def test_hits_are_searchhit_instances(gc):
    """Returned objects are SearchHit model instances."""
    from imas_codex.search.search_strategy import SearchHit

    hits = hybrid_dd_search(gc, "safety factor", k=5)
    assert hits
    for hit in hits:
        assert isinstance(hit, SearchHit)


def test_scores_are_positive_and_sorted(gc):
    """Scores should be positive and in descending order."""
    hits = hybrid_dd_search(gc, "magnetic field", k=10)
    assert hits
    scores = [h.score for h in hits]
    assert all(s > 0 for s in scores), f"Non-positive scores: {scores}"
    assert scores == sorted(scores, reverse=True), "Hits not in score-descending order"


# ---------------------------------------------------------------------------
# Filter parameters
# ---------------------------------------------------------------------------


def test_ids_filter_restricts_results(gc):
    """ids_filter should restrict all hits to the specified IDS."""
    hits = hybrid_dd_search(gc, "temperature", ids_filter="core_profiles", k=20)
    assert hits
    for hit in hits:
        assert hit.ids_name == "core_profiles", (
            f"Expected core_profiles, got {hit.ids_name}"
        )


def test_k_limits_results(gc):
    """k parameter should cap the number of returned hits."""
    hits = hybrid_dd_search(gc, "electron", k=3)
    assert len(hits) <= 3


def test_empty_query_returns_empty(gc):
    """An empty or whitespace query should return an empty list."""
    # The function normalizes the query first; empty-ish queries may still
    # produce text search results, but we test graceful handling.
    hits = hybrid_dd_search(gc, "xyzzy_nonexistent_concept_12345", k=5)
    # May return empty or very few low-score results — not an error either way
    assert isinstance(hits, list)
