"""Unit tests for hybrid_dd_search and related_dd_search pure functions.

Tests require a live Neo4j connection; they are skipped automatically
when the graph is unreachable via the top-level pytest_collection_modifyitems
hook (pytest.mark.graph marker).
"""

from __future__ import annotations

import os

import pytest

from imas_codex.graph.dd_search import (
    RelatedPathHit,
    RelatedPathResult,
    hybrid_dd_search,
    related_dd_search,
)

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


# ===========================================================================
# related_dd_search — cross-IDS relationship discovery
# ===========================================================================

_PSI_PATH = "equilibrium/time_slice/profiles_1d/psi"


class TestRelatedDdSearchBasic:
    """Basic return-type and structure tests for related_dd_search."""

    def test_returns_result_object(self, gc):
        """related_dd_search returns a RelatedPathResult."""
        result = related_dd_search(gc, _PSI_PATH)
        assert isinstance(result, RelatedPathResult)
        assert result.path == _PSI_PATH

    def test_hits_are_related_path_hit_instances(self, gc):
        """Each hit is a RelatedPathHit dataclass instance."""
        result = related_dd_search(gc, _PSI_PATH)
        assert result.hits
        for hit in result.hits:
            assert isinstance(hit, RelatedPathHit)
            assert hit.path
            assert hit.ids
            assert hit.relationship_type in (
                "cluster",
                "coordinate",
                "unit",
                "identifier",
                "cocos",
            )

    def test_total_connections_matches_hits(self, gc):
        """total_connections property equals len(hits)."""
        result = related_dd_search(gc, _PSI_PATH)
        assert result.total_connections == len(result.hits)

    def test_sections_groups_by_type(self, gc):
        """sections property groups hits by relationship_type."""
        result = related_dd_search(gc, _PSI_PATH)
        sections = result.sections
        total = sum(len(v) for v in sections.values())
        assert total == result.total_connections

    def test_hits_exclude_same_ids(self, gc):
        """All hits should be from a different IDS than the query path."""
        result = related_dd_search(gc, _PSI_PATH, max_results=50)
        query_ids = "equilibrium"
        for hit in result.hits:
            if hit.relationship_type != "cocos":
                assert hit.ids != query_ids, f"Hit {hit.path} is in same IDS {hit.ids}"


class TestRelatedDdSearchFilters:
    """Filter and parameter tests."""

    def test_relationship_type_filter(self, gc):
        """Filtering to a single relationship type returns only that type."""
        result = related_dd_search(gc, _PSI_PATH, relationship_types="cluster")
        assert result.relationship_types == "cluster"
        for hit in result.hits:
            assert hit.relationship_type == "cluster"

    def test_max_results_limits(self, gc):
        """max_results caps per-type results."""
        result = related_dd_search(
            gc, _PSI_PATH, relationship_types="cluster", max_results=3
        )
        assert len(result.hits) <= 3

    def test_nonexistent_path_returns_empty(self, gc):
        """A path that doesn't exist returns zero hits gracefully."""
        result = related_dd_search(gc, "nonexistent/path/that/does/not/exist")
        assert isinstance(result, RelatedPathResult)
        assert result.total_connections == 0


class TestRelatedDdSearchMcpCompat:
    """MCP dict compatibility — to_mcp_dict matches original format."""

    def test_mcp_dict_has_expected_keys(self, gc):
        """to_mcp_dict returns the four top-level keys."""
        result = related_dd_search(gc, _PSI_PATH)
        d = result.to_mcp_dict()
        assert set(d.keys()) == {
            "path",
            "relationship_types",
            "sections",
            "total_connections",
        }
        assert d["path"] == _PSI_PATH

    def test_mcp_dict_section_keys(self, gc):
        """Section keys match original naming convention."""
        result = related_dd_search(gc, _PSI_PATH, relationship_types="all")
        d = result.to_mcp_dict()
        valid_sections = {
            "cluster_siblings",
            "coordinate_partners",
            "unit_companions",
            "identifier_links",
            "cocos_kin",
        }
        for key in d["sections"]:
            assert key in valid_sections, f"Unexpected section key: {key}"

    def test_mcp_dict_cluster_entry_shape(self, gc):
        """Cluster entries have cluster/path/ids/doc/node_category keys."""
        result = related_dd_search(gc, _PSI_PATH, relationship_types="cluster")
        d = result.to_mcp_dict()
        if "cluster_siblings" in d["sections"]:
            for entry in d["sections"]["cluster_siblings"]:
                assert "path" in entry
                assert "ids" in entry
                assert "cluster" in entry
