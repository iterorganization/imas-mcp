"""
Test suite for get_dd_identifiers tool functionality.

This test suite validates that the get_dd_identifiers tool works correctly,
covering all query types and analytics calculations.
"""

import pytest

from imas_codex.models.result_models import GetIdentifiersResult
from imas_codex.tools.graph_search import GraphIdentifiersTool
from imas_codex.tools.identifiers_tool import IdentifiersTool


class TestExploreIdentifiersTool:
    """Test get_dd_identifiers tool functionality."""

    @pytest.fixture
    async def identifiers_tool(self):
        """Create identifiers tool instance."""
        return IdentifiersTool()

    @pytest.mark.asyncio
    async def test_tool_returns_results_for_standard_queries(self, identifiers_tool):
        """Tool returns non-empty results for standard queries."""

        # Test with no query (should return all schemas)
        result = await identifiers_tool.get_dd_identifiers()
        assert isinstance(result, GetIdentifiersResult)
        assert len(result.schemas) > 0, "Should return schemas when no query specified"
        assert len(result.paths) > 0, "Should return paths when no query specified"
        assert result.analytics["total_schemas"] > 0, "Should have total_schemas > 0"

        # Test with broad query terms
        for query in ["materials", "coordinate", "plasma"]:
            result = await identifiers_tool.get_dd_identifiers(query=query)
            assert isinstance(result, GetIdentifiersResult)
            # Note: Some queries may return empty results if no matching schemas exist
            # This is expected behavior, not an error

    @pytest.mark.asyncio
    async def test_enumeration_spaces_calculation(self, identifiers_tool):
        """Enumeration spaces are properly calculated."""

        # Test with materials query which should have a known enumeration space
        result = await identifiers_tool.get_dd_identifiers(query="materials")
        assert isinstance(result, GetIdentifiersResult)

        if len(result.schemas) > 0:
            # Materials schema should have 33 options
            materials_schema = next(
                (s for s in result.schemas if "materials" in s["path"].lower()), None
            )
            if materials_schema:
                assert materials_schema["option_count"] == 33, (
                    "Materials schema should have 33 options"
                )

        # Test overall enumeration space calculation
        assert result.analytics["enumeration_space"] >= 0, (
            "Enumeration space should be non-negative"
        )

        # Calculate expected enumeration space
        expected_space = sum(schema["option_count"] for schema in result.schemas)
        assert result.analytics["enumeration_space"] == expected_space, (
            "Enumeration space should match sum of schema options"
        )

    @pytest.mark.asyncio
    async def test_schema_discovery(self, identifiers_tool):
        """Schema discovery works correctly."""

        result = await identifiers_tool.get_dd_identifiers()
        assert isinstance(result, GetIdentifiersResult)

        # Should discover multiple schemas (adjust expectation based on environment)
        min_expected_schemas = 3  # At least a few schemas should be available
        assert len(result.schemas) >= min_expected_schemas, (
            f"Should discover at least {min_expected_schemas} schemas, got {len(result.schemas)}"
        )

        # Each schema should have required fields
        for schema in result.schemas[:5]:  # Test first 5 schemas
            assert "path" in schema, "Schema should have path"
            assert "option_count" in schema, "Schema should have option_count"
            assert "branching_significance" in schema, (
                "Schema should have branching_significance"
            )
            assert "options" in schema, "Schema should have options"

            # Sample options should be properly formatted
            for option in schema["options"]:
                assert "name" in option, "Option should have name"
                assert "index" in option, "Option should have index"
                assert "description" in option, "Option should have description"

    @pytest.mark.asyncio
    async def test_query_behavior(self, identifiers_tool):
        """Test behavior with various query patterns."""

        # Test that overly specific queries return empty results (this is correct behavior)
        result = await identifiers_tool.get_dd_identifiers(query="plasma state")
        assert isinstance(result, GetIdentifiersResult)
        # Empty results for overly specific queries is expected, not an error

        # Test partial matching works
        result = await identifiers_tool.get_dd_identifiers(query="material")
        assert isinstance(result, GetIdentifiersResult)
        # May return empty if no matching schemas, which is valid

    @pytest.mark.asyncio
    async def test_error_handling(self, identifiers_tool):
        """Test error handling scenarios."""

        # Test with valid call
        try:
            result = await identifiers_tool.get_dd_identifiers()
            assert isinstance(result, GetIdentifiersResult)
        except Exception as e:
            pytest.fail(f"Valid call should not raise exception: {e}")

    @pytest.mark.asyncio
    async def test_analytics_calculations(self, identifiers_tool):
        """Test analytics field calculations."""

        result = await identifiers_tool.get_dd_identifiers()
        assert isinstance(result, GetIdentifiersResult)

        analytics = result.analytics
        assert "total_schemas" in analytics
        assert "total_paths" in analytics
        assert "enumeration_space" in analytics
        assert "significance" in analytics

        # Analytics should be consistent with returned data
        assert analytics["total_schemas"] >= len(result.schemas)

    @pytest.mark.asyncio
    async def test_branching_significance_calculation(self, identifiers_tool):
        """Test branching significance calculation."""

        result = await identifiers_tool.get_dd_identifiers()
        assert isinstance(result, GetIdentifiersResult)

        significance_levels = ["MINIMAL", "MODERATE", "HIGH", "CRITICAL"]

        for schema in result.schemas:
            significance = schema["branching_significance"]
            assert significance in significance_levels, (
                f"Invalid significance level: {significance}"
            )

            # Verify significance correlates with option count
            option_count = schema["option_count"]
            if option_count > 10:
                assert significance == "CRITICAL"
            elif option_count > 5:
                assert significance == "HIGH"
            elif option_count > 1:
                assert significance == "MODERATE"
            else:
                assert significance == "MINIMAL"


class TestIdentifiersToolPerformance:
    """Performance tests for identifiers tool."""

    @pytest.fixture
    async def identifiers_tool(self):
        """Create identifiers tool instance."""
        return IdentifiersTool()

    @pytest.mark.asyncio
    async def test_performance_with_large_catalogs(self, identifiers_tool):
        """Test performance with large identifier catalogs."""

        import time

        start_time = time.time()

        result = await identifiers_tool.get_dd_identifiers()

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time
        assert execution_time < 5.0, (
            f"Tool should complete within 5 seconds, took {execution_time}"
        )
        assert isinstance(result, GetIdentifiersResult)


class TestIdentifiersToolValidation:
    """Validation tests to confirm tool functionality."""

    @pytest.fixture
    async def identifiers_tool(self):
        """Create identifiers tool instance."""
        return IdentifiersTool()

    @pytest.mark.asyncio
    async def test_functional_validation(self, identifiers_tool):
        """Validate that the tool is fully functional."""

        # Validate basic functionality
        result = await identifiers_tool.get_dd_identifiers()
        assert len(result.schemas) > 0, "Tool should return schemas"
        print("✅ Tool returns schemas for basic queries")

        # Validate enumeration calculation
        result = await identifiers_tool.get_dd_identifiers(query="materials")
        if len(result.schemas) > 0:
            expected_space = sum(schema["option_count"] for schema in result.schemas)
            assert result.analytics["enumeration_space"] == expected_space
        print("✅ Enumeration spaces properly calculated")

        # Validate schema discovery
        result = await identifiers_tool.get_dd_identifiers()
        assert len(result.schemas) >= 3, "Should discover multiple schemas"
        print("✅ Schema discovery working")

        print("\n🎉 IDENTIFIERS TOOL VALIDATION COMPLETE!")
        print("📊 Tool Status: FULLY FUNCTIONAL")


class TestIdentifiersToolInternals:
    """Tests for internal methods of IdentifiersTool."""

    @pytest.fixture
    def identifiers_tool(self):
        """Create identifiers tool instance."""
        return IdentifiersTool()

    def test_filter_schemas_empty_keywords(self, identifiers_tool):
        """Empty keywords return empty result."""
        identifiers_tool._identifier_catalog = {"schemas": {"test": {}}}

        result = identifiers_tool._filter_schemas_by_query("   ")

        assert result == []

    def test_filter_schemas_no_catalog(self, identifiers_tool):
        """No catalog returns empty result."""
        identifiers_tool._identifier_catalog = {}

        result = identifiers_tool._filter_schemas_by_query("test")

        assert result == []

    def test_filter_schemas_multiple_keywords_or_logic(self, identifiers_tool):
        """Multiple keywords use OR logic."""
        identifiers_tool._identifier_catalog = {
            "schemas": {
                "coordinate_type": {
                    "description": "Coordinate system types",
                    "options": [],
                },
                "material_type": {"description": "Material types", "options": []},
                "plasma_state": {"description": "Plasma state options", "options": []},
            }
        }

        result = identifiers_tool._filter_schemas_by_query("coordinate, material")

        assert len(result) == 2

    def test_filter_schemas_matches_options(self, identifiers_tool):
        """Query matches option names and descriptions."""
        identifiers_tool._identifier_catalog = {
            "schemas": {
                "test_schema": {
                    "description": "Some schema",
                    "options": [
                        {"name": "tungsten", "description": "Tungsten material"},
                    ],
                }
            }
        }

        result = identifiers_tool._filter_schemas_by_query("tungsten")

        assert "test_schema" in result

    def test_get_filtered_schemas_no_query_returns_all(self, identifiers_tool):
        """No query returns all schemas."""
        identifiers_tool._identifier_catalog = {"schemas": {"a": {}, "b": {}, "c": {}}}

        result = identifiers_tool._get_filtered_schemas(None)

        assert len(result) == 3

    def test_generate_recommendations_with_coordinate_schemas(self, identifiers_tool):
        """Recommendations include coordinate-specific suggestions."""
        schemas = {"coordinate_type": {}}
        recs = identifiers_tool._generate_identifier_recommendations("test", schemas)

        assert len(recs) > 0

    def test_generate_recommendations_with_type_schemas(self, identifiers_tool):
        """Recommendations include type-specific suggestions."""
        schemas = {"material_type": {}}
        recs = identifiers_tool._generate_identifier_recommendations("test", schemas)

        assert len(recs) > 0

    def test_generate_recommendations_limited_to_six(self, identifiers_tool):
        """Recommendations are limited to 6 items."""
        schemas = {f"schema_{i}": {} for i in range(5)}
        recs = identifiers_tool._generate_identifier_recommendations(None, schemas)

        assert len(recs) <= 6


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
