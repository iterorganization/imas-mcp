"""Regression tests for IMAS search remediation (Phases 1-4).

Validates:
- Phase 1: Hybrid vector+text scoring in IMAS DD server, full SearchHit metadata
- Phase 2: BM25 fulltext scoring with CONTAINS fallback
- Phase 3: CodeExample embedding/vector search integration
- Phase 4: Generic metadata path filtering

Tests use mock GraphClient and Encoder—no running Neo4j/embedding server needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _route_query(routes: dict[str, list[dict[str, Any]]]) -> Any:
    """gc.query side_effect that dispatches by Cypher substring."""

    def handler(cypher: str, **kwargs: Any) -> list[dict[str, Any]]:
        for pattern, result in routes.items():
            if pattern in cypher:
                return result
        return []

    return handler


# ---------------------------------------------------------------------------
# Phase 4: Node category classification (replaces _is_generic_metadata_path)
# ---------------------------------------------------------------------------


class TestNodeCategoryFiltering:
    """Tests that ExclusionChecker correctly categorizes metadata paths.

    These paths were previously filtered by _is_generic_metadata_path().
    Now they are classified at build time via ExclusionChecker and
    filtered at query time via node_category index.
    """

    @pytest.fixture()
    def classify(self):
        """Node classification via ExclusionChecker (same logic as _classify_node)."""
        from imas_codex.core.exclusions import ExclusionChecker

        checker = ExclusionChecker()

        def _classify(path_id: str, name: str) -> str:
            if checker._is_error_field(name):
                return "error"
            if checker._is_metadata_path(path_id):
                return "metadata"
            return "data"

        return _classify

    def test_filters_description_tail(self, classify):
        assert (
            classify("equilibrium/time_slice/profiles_1d/description", "description")
            == "metadata"
        )

    def test_filters_name_tail(self, classify):
        assert (
            classify("core_profiles/profiles_1d/electrons/name", "name") == "metadata"
        )

    def test_filters_comment_tail(self, classify):
        assert classify("magnetics/flux_loop/comment", "comment") == "metadata"

    def test_filters_source_tail(self, classify):
        assert classify("equilibrium/time_slice/source", "source") == "metadata"

    def test_filters_provider_tail(self, classify):
        assert (
            classify("core_profiles/global_quantities/provider", "provider")
            == "metadata"
        )

    def test_filters_identifier_name(self, classify):
        assert (
            classify("equilibrium/time_slice/boundary/identifier/name", "name")
            == "metadata"
        )

    def test_filters_identifier_description(self, classify):
        assert (
            classify(
                "equilibrium/time_slice/boundary/identifier/description", "description"
            )
            == "metadata"
        )

    def test_keeps_physics_leaf(self, classify):
        assert classify("equilibrium/time_slice/profiles_1d/psi", "psi") == "data"

    def test_keeps_temperature(self, classify):
        assert (
            classify("core_profiles/profiles_1d/electrons/temperature", "temperature")
            == "data"
        )

    def test_keeps_short_path(self, classify):
        """Paths with fewer than 3 segments are never metadata."""
        assert classify("equilibrium/name", "name") == "data"

    def test_keeps_value_leaf(self, classify):
        assert classify("magnetics/ip/0d/value", "value") == "data"


# ---------------------------------------------------------------------------
# Phase 2: Text search - DD server (_text_search_dd_paths)
# ---------------------------------------------------------------------------


class TestTextSearchImasPathsDDServer:
    """Tests for _text_search_dd_paths in graph_search.py.

    Verifies BM25 fulltext path with CONTAINS fallback.
    """

    @pytest.fixture()
    def mock_gc(self):
        gc = MagicMock()
        gc.query = MagicMock(side_effect=_route_query({}))
        return gc

    def test_fulltext_index_path_returns_normalized_scores(self, mock_gc):
        """When fulltext index returns results, scores are normalized 0-1."""
        from imas_codex.tools.graph_search import _text_search_dd_paths

        mock_gc.query.side_effect = _route_query(
            {
                "db.index.fulltext.queryNodes": [
                    {"id": "magnetics/ip/0d/value", "score": 10.0},
                    {"id": "magnetics/flux_loop/flux/data", "score": 5.0},
                ],
            }
        )

        results = _text_search_dd_paths(mock_gc, "plasma current", 50, None)

        assert len(results) == 2
        # Top result should be normalized to 1.0 (max)
        top = next(r for r in results if r["id"] == "magnetics/ip/0d/value")
        assert top["score"] == 1.0

        # Lower result should be normalized proportionally (no floor)
        lower = next(r for r in results if r["id"] == "magnetics/flux_loop/flux/data")
        assert 0.0 < lower["score"] < 1.0

    def test_fulltext_includes_node_category_filter(self, mock_gc):
        """Fulltext query includes node_category='data' in WHERE clause."""
        from imas_codex.tools.graph_search import _text_search_dd_paths

        mock_gc.query.side_effect = _route_query(
            {
                "db.index.fulltext.queryNodes": [
                    {"id": "magnetics/ip/0d/value", "score": 10.0},
                ],
            }
        )

        _text_search_dd_paths(mock_gc, "plasma current", 50, None)

        # Verify the fulltext query was called with node_category filter
        calls = mock_gc.query.call_args_list
        ft_call = next(c for c in calls if "db.index.fulltext.queryNodes" in c[0][0])
        assert "node_category = 'data'" in ft_call[0][0]

    def test_contains_fallback_when_fulltext_fails(self, mock_gc):
        """When fulltext index raises, falls back to CONTAINS matching."""
        from imas_codex.tools.graph_search import _text_search_dd_paths

        call_count = 0

        def side_effect(cypher: str, **kwargs):
            nonlocal call_count
            call_count += 1
            if "db.index.fulltext.queryNodes" in cypher:
                raise Exception("Index not found")
            if "CONTAINS" in cypher:
                return [{"id": "magnetics/ip/0d/value", "score": 0.95}]
            return []

        mock_gc.query.side_effect = side_effect

        results = _text_search_dd_paths(mock_gc, "plasma current", 50, None)

        assert len(results) >= 1
        assert results[0]["id"] == "magnetics/ip/0d/value"

    def test_ids_filter_applied(self, mock_gc):
        """ids_filter is passed to fulltext search WHERE clause."""
        from imas_codex.tools.graph_search import _text_search_dd_paths

        mock_gc.query.side_effect = _route_query(
            {
                "db.index.fulltext.queryNodes": [
                    {"id": "magnetics/ip/0d/value", "score": 5.0},
                ],
            }
        )

        _text_search_dd_paths(mock_gc, "current", 50, "magnetics")

        # Check fulltext call includes ids_filter
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "db.index.fulltext.queryNodes" in cypher:
                assert "ids_filter" in cypher or "ids_filter" in call[1]
                break


# ---------------------------------------------------------------------------
# Phase 2: Text search - Codex server (_text_search_dd_paths_by_query)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="_text_search_dd_paths_by_query removed; unified in GraphSearchTool"
)
class TestTextSearchImasPathsCodexServer:
    """Tests for _text_search_dd_paths_by_query in search_tools.py.

    Verifies BM25 fulltext path with CONTAINS fallback.
    """

    @pytest.fixture()
    def mock_gc(self):
        gc = MagicMock()
        gc.query = MagicMock(side_effect=_route_query({}))
        return gc

    def test_fulltext_returns_normalized_scores(self, mock_gc):
        """BM25 scores are normalized 0-1 with 0.7 floor."""
        from imas_codex.llm.search_tools import _text_search_dd_paths_by_query

        mock_gc.query.side_effect = _route_query(
            {
                "db.index.fulltext.queryNodes": [
                    {
                        "id": "core_profiles.profiles_1d[:].electrons.temperature",
                        "score": 8.0,
                    },
                    {
                        "id": "core_profiles.profiles_1d[:].ion.temperature",
                        "score": 4.0,
                    },
                ],
            }
        )

        results = _text_search_dd_paths_by_query(
            mock_gc, "electron temperature", 20, None
        )

        assert len(results) == 2
        top = next(r for r in results if "electrons" in r["id"])
        assert top["score"] == 1.0  # normalized max

        lower = next(r for r in results if "ion" in r["id"])
        assert 0.7 <= lower["score"] <= 1.0

    def test_fulltext_filters_generic_paths(self, mock_gc):
        """Generic metadata paths are filtered from fulltext results."""
        from imas_codex.llm.search_tools import _text_search_dd_paths_by_query

        mock_gc.query.side_effect = _route_query(
            {
                "db.index.fulltext.queryNodes": [
                    {"id": "equilibrium/time_slice/profiles_1d/psi", "score": 5.0},
                    {
                        "id": "equilibrium/time_slice/profiles_1d/description",
                        "score": 4.5,
                    },
                ],
            }
        )

        results = _text_search_dd_paths_by_query(mock_gc, "psi", 20, None)
        ids = {r["id"] for r in results}

        assert "equilibrium/time_slice/profiles_1d/psi" in ids
        assert "equilibrium/time_slice/profiles_1d/description" not in ids

    def test_contains_fallback(self, mock_gc):
        """Falls back to CONTAINS when fulltext raises."""
        from imas_codex.llm.search_tools import _text_search_dd_paths_by_query

        def side_effect(cypher: str, **kwargs):
            if "db.index.fulltext.queryNodes" in cypher:
                raise Exception("Index not found")
            if "CONTAINS" in cypher:
                return [{"id": "magnetics.ip.0d[:].value", "score": 0.95}]
            return []

        mock_gc.query.side_effect = side_effect

        results = _text_search_dd_paths_by_query(mock_gc, "plasma current", 20, None)

        assert len(results) >= 1
        assert any(r["id"] == "magnetics.ip.0d[:].value" for r in results)


# ---------------------------------------------------------------------------
# Phase 1: Hybrid search + full SearchHit metadata (IMAS DD server)
# ---------------------------------------------------------------------------


class TestGraphSearchToolHybrid:
    """Tests for GraphSearchTool.search_dd_paths hybrid scoring and metadata.

    Mocks the Encoder and GraphClient to avoid requiring Neo4j/embedding.
    """

    @pytest.fixture()
    def mock_gc(self):
        gc = MagicMock()
        gc.query = MagicMock(side_effect=_route_query({}))
        return gc

    def _make_tool(self, mock_gc):
        from imas_codex.tools.graph_search import GraphSearchTool

        tool = GraphSearchTool(mock_gc)
        tool._embed_query = MagicMock(return_value=[0.1] * 256)
        return tool

    @pytest.mark.asyncio
    async def test_hybrid_boost_dual_match(self, mock_gc):
        """Paths found by both vector AND text get RRF-merged score."""
        tool = self._make_tool(mock_gc)

        mock_gc.query.side_effect = _route_query(
            {
                "imas_node_embedding": [
                    {"id": "magnetics/ip/0d/value", "score": 0.85},
                ],
                "db.index.fulltext.queryNodes": [
                    {"id": "magnetics/ip/0d/value", "score": 10.0},
                ],
                "UNWIND $path_ids": [
                    {
                        "id": "magnetics/ip/0d/value",
                        "name": "value",
                        "ids": "magnetics",
                        "documentation": "Plasma current",
                        "data_type": "FLT_0D",
                        "physics_domain": "magnetics",
                        "units": "A",
                        "node_type": "leaf",
                        "lifecycle_status": "active",
                        "lifecycle_version": None,
                        "timebase": None,
                        "structure_reference": None,
                        "coordinate1": None,
                        "coordinate2": None,
                        "coordinates": [],
                        "has_identifier_schema": False,
                        "introduced_after_version": "3.0.0",
                        "cocos_label": None,
                        "cocos_expression": None,
                        "description": None,
                        "keywords": None,
                        "enrichment_source": None,
                    }
                ],
            }
        )

        result = await tool.search_dd_paths(query="plasma current")

        assert len(result.hits) == 1
        hit = result.hits[0]
        # Dual match via RRF: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.033
        assert hit.score > 0

    @pytest.mark.asyncio
    async def test_text_only_match_included(self, mock_gc):
        """Paths found only by text search are included in results."""
        tool = self._make_tool(mock_gc)

        mock_gc.query.side_effect = _route_query(
            {
                "imas_node_embedding": [],  # no vector results
                "db.index.fulltext.queryNodes": [
                    {"id": "magnetics/ip/0d/value", "score": 8.0},
                ],
                "UNWIND $path_ids": [
                    {
                        "id": "magnetics/ip/0d/value",
                        "name": "value",
                        "ids": "magnetics",
                        "documentation": "Plasma current",
                        "data_type": "FLT_0D",
                        "physics_domain": "magnetics",
                        "units": "A",
                        "node_type": "leaf",
                        "lifecycle_status": "active",
                        "lifecycle_version": None,
                        "timebase": None,
                        "structure_reference": None,
                        "coordinate1": None,
                        "coordinate2": None,
                        "coordinates": [],
                        "has_identifier_schema": False,
                        "introduced_after_version": "3.0.0",
                        "cocos_label": None,
                        "cocos_expression": None,
                        "description": None,
                        "keywords": None,
                        "enrichment_source": None,
                    }
                ],
            }
        )

        result = await tool.search_dd_paths(query="plasma current")

        assert len(result.hits) >= 1
        assert result.hits[0].path == "magnetics/ip/0d/value"

    @pytest.mark.asyncio
    async def test_full_searchhit_metadata(self, mock_gc):
        """All 18 SearchHit fields are populated from enrichment query."""
        tool = self._make_tool(mock_gc)

        mock_gc.query.side_effect = _route_query(
            {
                "imas_node_embedding": [
                    {"id": "equilibrium/time_slice/profiles_1d/psi", "score": 0.92},
                ],
                "UNWIND $path_ids": [
                    {
                        "id": "equilibrium/time_slice/profiles_1d/psi",
                        "name": "psi",
                        "ids": "equilibrium",
                        "documentation": "Poloidal magnetic flux profile",
                        "data_type": "FLT_1D",
                        "physics_domain": "magnetics",
                        "units": "T.m^2",
                        "node_type": "leaf",
                        "lifecycle_status": "active",
                        "lifecycle_version": "4.0.0",
                        "timebase": "equilibrium/time",
                        "structure_reference": "https://imas.io/docs/equilibrium",
                        "coordinate1": "rho_tor_norm",
                        "coordinate2": None,
                        "coordinates": [
                            "equilibrium/time_slice/profiles_1d/grid/rho_tor_norm"
                        ],
                        "has_identifier_schema": False,
                        "introduced_after_version": "3.42.0",
                        "cocos_label": None,
                        "cocos_expression": None,
                        "description": None,
                        "keywords": None,
                        "enrichment_source": None,
                    }
                ],
            }
        )

        result = await tool.search_dd_paths(query="poloidal flux")

        assert len(result.hits) == 1
        hit = result.hits[0]

        # Core fields
        assert hit.path == "equilibrium/time_slice/profiles_1d/psi"
        assert hit.ids_name == "equilibrium"
        assert hit.documentation == "Poloidal magnetic flux profile"
        assert hit.data_type == "FLT_1D"
        assert hit.units == "T.m^2"
        assert hit.physics_domain == "magnetics"
        assert hit.node_type == "leaf"

        # Extended fields (Phase 1 additions)
        assert hit.lifecycle_status == "active"
        assert hit.lifecycle_version == "4.0.0"
        assert hit.timebase == "equilibrium/time"
        assert hit.structure_reference == "https://imas.io/docs/equilibrium"
        assert hit.coordinate1 == "rho_tor_norm"
        assert hit.coordinate2 is None
        assert "equilibrium/time_slice/profiles_1d/grid/rho_tor_norm" in hit.coordinates
        assert hit.has_identifier_schema is False
        assert hit.introduced_after_version == "3.42.0"

        # Score and rank
        assert hit.score > 0
        assert hit.rank == 1

    @pytest.mark.asyncio
    async def test_vector_query_includes_node_category_filter(self, mock_gc):
        """Vector search query includes node_category='data' WHERE clause."""
        tool = self._make_tool(mock_gc)

        mock_gc.query.side_effect = _route_query(
            {
                "imas_node_embedding": [
                    {"id": "equilibrium/time_slice/profiles_1d/psi", "score": 0.90},
                ],
                "UNWIND $path_ids": [
                    {
                        "id": "equilibrium/time_slice/profiles_1d/psi",
                        "name": "psi",
                        "ids": "equilibrium",
                        "documentation": "Poloidal magnetic flux",
                        "data_type": "FLT_1D",
                        "physics_domain": "magnetics",
                        "units": "T.m^2",
                        "node_type": "leaf",
                        "lifecycle_status": "active",
                        "lifecycle_version": None,
                        "timebase": None,
                        "structure_reference": None,
                        "coordinate1": None,
                        "coordinate2": None,
                        "coordinates": [],
                        "has_identifier_schema": False,
                        "introduced_after_version": "3.42.0",
                        "cocos_label": None,
                        "cocos_expression": None,
                        "description": None,
                        "keywords": None,
                        "enrichment_source": None,
                    }
                ],
            }
        )

        await tool.search_dd_paths(query="profiles")

        # Verify the vector query includes node_category filter
        calls = mock_gc.query.call_args_list
        vector_call = next(c for c in calls if "imas_node_embedding" in c[0][0])
        assert "node_category = 'data'" in vector_call[0][0]

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self, mock_gc):
        """Empty query returns structured error."""
        tool = self._make_tool(mock_gc)

        result = await tool.search_dd_paths(query="")

        assert result.hits == []
        assert "error" in result.summary

    @pytest.mark.asyncio
    async def test_no_results_returns_empty(self, mock_gc):
        """No matches from either vector or text produce empty result."""
        tool = self._make_tool(mock_gc)

        mock_gc.query.side_effect = _route_query({})

        result = await tool.search_dd_paths(query="zzz_nonexistent_zzz")

        assert result.hits == []
        assert result.summary["hits_returned"] == 0

    @pytest.mark.asyncio
    async def test_ids_coverage_in_summary(self, mock_gc):
        """Summary includes IDS coverage from results."""
        tool = self._make_tool(mock_gc)

        mock_gc.query.side_effect = _route_query(
            {
                "imas_node_embedding": [
                    {"id": "equilibrium/time_slice/profiles_1d/psi", "score": 0.90},
                    {
                        "id": "core_profiles/profiles_1d/electrons/temperature",
                        "score": 0.88,
                    },
                ],
                "UNWIND $path_ids": [
                    {
                        "id": "equilibrium/time_slice/profiles_1d/psi",
                        "name": "psi",
                        "ids": "equilibrium",
                        "documentation": "Poloidal flux",
                        "data_type": "FLT_1D",
                        "physics_domain": "magnetics",
                        "units": "T.m^2",
                        "node_type": "leaf",
                        "lifecycle_status": None,
                        "lifecycle_version": None,
                        "timebase": None,
                        "structure_reference": None,
                        "coordinate1": None,
                        "coordinate2": None,
                        "coordinates": [],
                        "has_identifier_schema": False,
                        "introduced_after_version": None,
                        "cocos_label": None,
                        "cocos_expression": None,
                        "description": None,
                        "keywords": None,
                        "enrichment_source": None,
                    },
                    {
                        "id": "core_profiles/profiles_1d/electrons/temperature",
                        "name": "temperature",
                        "ids": "core_profiles",
                        "documentation": "Electron temperature",
                        "data_type": "FLT_1D",
                        "physics_domain": "transport",
                        "units": "eV",
                        "node_type": "leaf",
                        "lifecycle_status": None,
                        "lifecycle_version": None,
                        "timebase": None,
                        "structure_reference": None,
                        "coordinate1": None,
                        "coordinate2": None,
                        "coordinates": [],
                        "has_identifier_schema": False,
                        "introduced_after_version": None,
                        "cocos_label": None,
                        "cocos_expression": None,
                        "description": None,
                        "keywords": None,
                        "enrichment_source": None,
                    },
                ],
            }
        )

        result = await tool.search_dd_paths(query="profiles")

        assert "core_profiles" in result.summary["ids_coverage"]
        assert "equilibrium" in result.summary["ids_coverage"]
        assert len(result.physics_domains) >= 1


# ---------------------------------------------------------------------------
# Phase 2: Hybrid text+vector scoring in Codex server (_search_imas)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="_search_imas removed; unified in GraphSearchTool")
class TestSearchImasHybridCodex:
    """Tests for _search_imas hybrid text+vector merge in search_tools.py.

    Extends the existing TestSearchImas to cover BM25 specifics.
    """

    @pytest.fixture()
    def mock_gc(self):
        gc = MagicMock()
        gc.query = MagicMock(side_effect=_route_query({}))
        return gc

    @pytest.fixture()
    def mock_encoder(self):
        enc = MagicMock()
        enc.embed_texts = MagicMock(return_value=[[0.1] * 1024])
        return enc

    def test_text_only_match_uses_text_score(self, mock_gc, mock_encoder):
        """A path found only by text (not vector) uses text score directly."""
        from imas_codex.llm.search_tools import _search_imas

        mock_gc.query.side_effect = _route_query(
            {
                "imas_node_embedding": [],  # no vector results
                "db.index.fulltext.queryNodes('imas_node_text": [
                    {"id": "magnetics.ip.0d[:].value", "score": 8.0},
                ],
                "UNWIND $path_ids": [
                    {
                        "id": "magnetics.ip.0d[:].value",
                        "name": "value",
                        "ids": "magnetics",
                        "documentation": "Plasma current",
                        "data_type": "FLT_0D",
                        "ndim": 0,
                        "node_type": "leaf",
                        "physics_domain": "magnetics",
                        "cocos_transformation_type": None,
                        "lifecycle_status": "active",
                        "lifecycle_version": None,
                        "structure_reference": None,
                        "unit": "A",
                        "clusters": [],
                        "coordinates": [],
                        "introduced_in": "3.0.0",
                    }
                ],
            }
        )

        result = _search_imas("plasma current", gc=mock_gc, encoder=mock_encoder)

        assert "magnetics.ip.0d[:].value" in result
        assert "Plasma current" in result

    def test_dual_match_boosted(self, mock_gc, mock_encoder):
        """Path found in both vector AND text gets +0.05 boost."""
        from imas_codex.llm.search_tools import _search_imas

        mock_gc.query.side_effect = _route_query(
            {
                "imas_node_embedding": [
                    {"id": "magnetics.ip.0d[:].value", "score": 0.80},
                ],
                "db.index.fulltext.queryNodes('imas_node_text": [
                    {"id": "magnetics.ip.0d[:].value", "score": 10.0},
                ],
                "UNWIND $path_ids": [
                    {
                        "id": "magnetics.ip.0d[:].value",
                        "name": "value",
                        "ids": "magnetics",
                        "documentation": "Plasma current",
                        "data_type": "FLT_0D",
                        "ndim": 0,
                        "node_type": "leaf",
                        "physics_domain": "magnetics",
                        "cocos_transformation_type": None,
                        "lifecycle_status": "active",
                        "lifecycle_version": None,
                        "structure_reference": None,
                        "unit": "A",
                        "clusters": [],
                        "coordinates": [],
                        "introduced_in": "3.0.0",
                    }
                ],
            }
        )

        result = _search_imas("plasma current", gc=mock_gc, encoder=mock_encoder)

        # The path should appear but score details are in the formatted report
        assert "magnetics.ip.0d[:].value" in result

    def test_enrichment_includes_lifecycle(self, mock_gc, mock_encoder):
        """Enriched results include lifecycle_status, structure_reference."""
        from imas_codex.llm.search_tools import _search_imas

        mock_gc.query.side_effect = _route_query(
            {
                "imas_node_embedding": [
                    {"id": "magnetics.ip.0d[:].value", "score": 0.92},
                ],
                "UNWIND $path_ids": [
                    {
                        "id": "magnetics.ip.0d[:].value",
                        "name": "value",
                        "ids": "magnetics",
                        "documentation": "Plasma current",
                        "data_type": "FLT_0D",
                        "ndim": 0,
                        "node_type": "leaf",
                        "physics_domain": "magnetics",
                        "cocos_transformation_type": None,
                        "lifecycle_status": "obsolescent",
                        "lifecycle_version": "4.0.0",
                        "structure_reference": "https://imas.io/magnetics/ip",
                        "unit": "A",
                        "clusters": [],
                        "coordinates": [],
                        "introduced_in": "3.0.0",
                    }
                ],
            }
        )

        result = _search_imas("plasma current", gc=mock_gc, encoder=mock_encoder)

        assert "obsolescent" in result
        assert "https://imas.io/magnetics/ip" in result


# ---------------------------------------------------------------------------
# Phase 3: CodeExample vector search (_vector_search_code_examples)
# ---------------------------------------------------------------------------


class TestCodeExampleVectorSearch:
    """Tests for _vector_search_code_examples in search_tools.py."""

    @pytest.fixture()
    def mock_gc(self):
        gc = MagicMock()
        gc.query = MagicMock(side_effect=_route_query({}))
        return gc

    def test_returns_chunk_ids_from_examples(self, mock_gc):
        """Vector search on CodeExamples traverses to CodeChunks."""
        from imas_codex.llm.search_tools import _vector_search_code_examples

        mock_gc.query.side_effect = _route_query(
            {
                "code_example_desc_embedding": [
                    {"id": "chunk:1", "score": 0.90},
                    {"id": "chunk:2", "score": 0.85},
                ],
            }
        )

        results = _vector_search_code_examples(mock_gc, [0.1] * 256, "tcv", 10)

        assert len(results) == 2
        assert results[0]["id"] == "chunk:1"
        assert results[0]["score"] == 0.90

    def test_facility_filter_applied(self, mock_gc):
        """Facility filter is included in Cypher WHERE clause."""
        from imas_codex.llm.search_tools import _vector_search_code_examples

        mock_gc.query.side_effect = _route_query(
            {
                "code_example_desc_embedding": [],
            }
        )

        _vector_search_code_examples(mock_gc, [0.1] * 256, "tcv", 10)

        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "code_example_desc_embedding" in cypher:
                assert "facility" in cypher.lower()
                assert call[1].get("facility") == "tcv"
                break
        else:
            pytest.fail("No vector search call found")

    def test_no_facility_no_filter(self, mock_gc):
        """Without facility, no facility filter in query."""
        from imas_codex.llm.search_tools import _vector_search_code_examples

        mock_gc.query.side_effect = _route_query(
            {
                "code_example_desc_embedding": [],
            }
        )

        _vector_search_code_examples(mock_gc, [0.1] * 256, None, 10)

        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "code_example_desc_embedding" in cypher:
                assert "facility" not in call[1]
                break

    def test_graceful_on_missing_index(self, mock_gc):
        """Gracefully returns empty when index doesn't exist."""
        from imas_codex.llm.search_tools import _vector_search_code_examples

        mock_gc.query.side_effect = Exception("Index not found")

        results = _vector_search_code_examples(mock_gc, [0.1] * 256, "tcv", 10)

        assert results == []


class TestSearchCodeWithExamples:
    """Tests for _search_code integration with CodeExample vector search."""

    @pytest.fixture()
    def mock_gc(self):
        gc = MagicMock()
        gc.query = MagicMock(side_effect=_route_query({}))
        return gc

    @pytest.fixture()
    def mock_encoder(self):
        enc = MagicMock()
        enc.embed_texts = MagicMock(return_value=[[0.1] * 1024])
        return enc

    def test_code_example_chunks_merged(self, mock_gc, mock_encoder):
        """CodeExample chunks are merged into results with boost."""
        from imas_codex.llm.search_tools import _search_code

        enrichment = [
            {
                "id": "chunk:1",
                "text": "def read_eq(shot): pass",
                "function_name": "read_eq",
                "source_file": "/code/eq.py",
                "source_file_id": "tcv:/code/eq.py",
                "facility_id": "tcv",
                "data_refs": [],
                "directory": "/code",
                "dir_description": None,
            }
        ]

        mock_gc.query.side_effect = _route_query(
            {
                "code_chunk_embedding": [],  # no direct chunk matches
                "code_example_desc_embedding": [
                    {"id": "chunk:1", "score": 0.88},
                ],
                "CodeChunk {id: cid}": enrichment,
            }
        )

        result = _search_code(
            query="equilibrium reconstruction",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )

        assert "read_eq" in result
        assert "eq.py" in result

    def test_code_example_boost_on_overlap(self, mock_gc, mock_encoder):
        """Chunks found by both code_chunk and code_example get boosted."""
        from imas_codex.llm.search_tools import _search_code

        enrichment = [
            {
                "id": "chunk:1",
                "text": "def read_eq(shot): pass",
                "function_name": "read_eq",
                "source_file": "/code/eq.py",
                "source_file_id": "tcv:/code/eq.py",
                "facility_id": "tcv",
                "data_refs": [],
                "directory": "/code",
                "dir_description": None,
            }
        ]

        mock_gc.query.side_effect = _route_query(
            {
                # Found by direct chunk vector search
                "code_chunk_embedding": [
                    {"id": "chunk:1", "score": 0.75},
                ],
                # Also found by CodeExample vector search
                "code_example_desc_embedding": [
                    {"id": "chunk:1", "score": 0.82},
                ],
                "CodeChunk {id: cid}": enrichment,
            }
        )

        result = _search_code(
            query="equilibrium",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )

        # Should appear with boosted score (not the raw 0.75)
        assert "read_eq" in result
        assert "0.75" not in result
