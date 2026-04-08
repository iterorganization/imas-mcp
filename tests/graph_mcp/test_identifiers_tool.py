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
    """Integration tests for GraphIdentifiersTool search (requires graph)."""

    @pytest.fixture
    def tool(self):
        from imas_codex.graph.client import GraphClient

        gc = GraphClient()
        return GraphIdentifiersTool(gc)

    async def test_no_query_returns_all(self, tool):
        """No query returns all schemas."""
        result = await tool.get_dd_identifiers()
        assert isinstance(result, GetIdentifiersResult)
        assert len(result.schemas) > 50

    async def test_grid_type_returns_results(self, tool):
        """The original bug: 'grid_type' must return results."""
        result = await tool.get_dd_identifiers(query="grid_type")
        assert isinstance(result, GetIdentifiersResult)
        assert len(result.schemas) > 0, "grid_type should match grid-related schemas"
        names = [s["path"] for s in result.schemas]
        assert any("grid" in n for n in names), "Should include grid-related schemas"

    async def test_comma_separated_or(self, tool):
        """Comma-separated queries use OR logic."""
        result = await tool.get_dd_identifiers(query="coordinate,transport")
        assert isinstance(result, GetIdentifiersResult)
        names = [s["path"] for s in result.schemas]
        has_coord = any("coordinate" in n for n in names)
        has_transport = any("transport" in n for n in names)
        assert has_coord and has_transport, "Should match both terms"

    async def test_option_content_search(self, tool):
        """Search matches content inside schema options."""
        result = await tool.get_dd_identifiers(query="tungsten")
        assert isinstance(result, GetIdentifiersResult)
        assert len(result.schemas) >= 1
        assert any("material" in s["path"] for s in result.schemas)

    async def test_no_dd_version_parameter(self, tool):
        """get_dd_identifiers does not accept dd_version."""
        import inspect

        sig = inspect.signature(tool.get_dd_identifiers)
        assert "dd_version" not in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
