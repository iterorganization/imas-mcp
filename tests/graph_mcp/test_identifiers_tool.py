"""
Test suite for GraphIdentifiersTool (graph-backed).

Tests keyword tokenization, scoring, and integration with the graph.
"""

import pytest

from imas_codex.models.result_models import GetIdentifiersResult
from imas_codex.tools.graph_search import GraphIdentifiersTool

pytestmark = pytest.mark.graph_mcp


class TestGraphIdentifiersTokenization:
    """Tests for GraphIdentifiersTool keyword tokenization and scoring."""

    def test_tokenize_underscores(self):
        """Underscores split into separate keywords."""
        tokens = GraphIdentifiersTool._tokenize_query("grid_type")
        assert tokens == ["grid", "type"]

    def test_tokenize_commas(self):
        """Comma-separated terms split correctly."""
        tokens = GraphIdentifiersTool._tokenize_query("coordinate,transport")
        assert tokens == ["coordinate", "transport"]

    def test_tokenize_spaces(self):
        """Space-separated terms split correctly."""
        tokens = GraphIdentifiersTool._tokenize_query("coordinate transport")
        assert tokens == ["coordinate", "transport"]

    def test_tokenize_mixed_separators(self):
        """Mixed separators all work."""
        tokens = GraphIdentifiersTool._tokenize_query("grid_type, coordinate")
        assert tokens == ["grid", "type", "coordinate"]

    def test_tokenize_empty(self):
        """Empty/whitespace input returns empty list."""
        assert GraphIdentifiersTool._tokenize_query("   ") == []
        assert GraphIdentifiersTool._tokenize_query("") == []

    def test_score_exact_name_match_highest(self):
        """Exact original query in name scores highest."""
        score = GraphIdentifiersTool._score_keyword_match(
            keywords=["coordinate"],
            original_query="coordinate",
            name="coordinate_identifier",
            description="",
            parsed_options=[],
            schema_keywords=[],
        )
        # Should get tier 6 (exact) + tier 3 (keyword in name)
        assert score >= 9

    def test_score_underscore_query_matches_name(self):
        """Underscore query normalized to spaces matches name."""
        score = GraphIdentifiersTool._score_keyword_match(
            keywords=["grid", "type"],
            original_query="grid_type",
            name="magnetics_probe_type_identifier",
            description="",
            parsed_options=[],
            schema_keywords=[],
        )
        # "type" matches in name → at least tier 3
        assert score >= 3

    def test_score_all_keywords_in_name_bonus(self):
        """All keywords in name gets tier 4 bonus."""
        score_both = GraphIdentifiersTool._score_keyword_match(
            keywords=["grid", "emission"],
            original_query="grid emission",
            name="emission_grid_identifier",
            description="",
            parsed_options=[],
            schema_keywords=[],
        )
        score_one = GraphIdentifiersTool._score_keyword_match(
            keywords=["grid", "emission"],
            original_query="grid emission",
            name="emission_only_identifier",
            description="",
            parsed_options=[],
            schema_keywords=[],
        )
        assert score_both > score_one

    def test_score_option_content_match(self):
        """Keywords matching option content score > 0."""
        score = GraphIdentifiersTool._score_keyword_match(
            keywords=["tungsten"],
            original_query="tungsten",
            name="materials_identifier",
            description="Physical materials",
            parsed_options=[
                {"name": "tungsten", "description": "Tungsten material"},
            ],
            schema_keywords=[],
        )
        assert score >= 1

    def test_score_no_match_returns_zero(self):
        """Completely unrelated query returns 0."""
        score = GraphIdentifiersTool._score_keyword_match(
            keywords=["banana"],
            original_query="banana",
            name="coordinate_identifier",
            description="Coordinate system types",
            parsed_options=[{"name": "x", "description": "cartesian"}],
            schema_keywords=["coordinates"],
        )
        assert score == 0


@pytest.mark.asyncio
class TestGraphIdentifiersSearch:
    """Integration tests for GraphIdentifiersTool search (requires graph).

    Tests work against both the minimal CI fixture (1 IdentifierSchema:
    boundary_type) and the full production graph (50+ schemas).
    """

    @pytest.fixture
    def tool(self, graph_client):
        return GraphIdentifiersTool(graph_client)

    async def test_no_query_returns_all(self, tool):
        """No query returns all schemas (at least the fixture one)."""
        result = await tool.get_dd_identifiers()
        assert isinstance(result, GetIdentifiersResult)
        assert len(result.schemas) >= 1

    async def test_query_returns_results(self, tool):
        """A query that matches fixture data returns results."""
        result = await tool.get_dd_identifiers(query="boundary_type")
        assert isinstance(result, GetIdentifiersResult)
        assert len(result.schemas) >= 1, "boundary_type should match schemas"

    async def test_query_filters_results(self, tool):
        """Query returns fewer results than unfiltered list."""
        all_result = await tool.get_dd_identifiers()
        filtered = await tool.get_dd_identifiers(query="boundary_type")
        assert len(filtered.schemas) <= len(all_result.schemas)

    async def test_option_content_search(self, tool):
        """Search matches content inside schema options."""
        result = await tool.get_dd_identifiers(query="limiter")
        assert isinstance(result, GetIdentifiersResult)
        assert len(result.schemas) >= 1, "limiter is an option in boundary_type"

    async def test_no_match_returns_empty(self, tool):
        """Unrelated query returns no results."""
        result = await tool.get_dd_identifiers(query="xyznonexistent")
        assert isinstance(result, GetIdentifiersResult)
        assert len(result.schemas) == 0

    async def test_no_dd_version_parameter(self, tool):
        """get_dd_identifiers does not accept dd_version."""
        import inspect

        sig = inspect.signature(tool.get_dd_identifiers)
        assert "dd_version" not in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
