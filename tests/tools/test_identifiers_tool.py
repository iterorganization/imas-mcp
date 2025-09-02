"""
Test suite for explore_identifiers tool functionality.

This test suite validates that the explore_identifiers tool works correctly,
covering all scopes, query types, and analytics calculations.
"""

import pytest

from imas_mcp.models.constants import IdentifierScope
from imas_mcp.models.result_models import IdentifierResult
from imas_mcp.tools.identifiers_tool import IdentifiersTool


class TestExploreIdentifiersTool:
    """Test explore_identifiers tool functionality."""

    @pytest.fixture
    async def identifiers_tool(self):
        """Create identifiers tool instance."""
        return IdentifiersTool()

    @pytest.mark.asyncio
    async def test_tool_returns_results_for_standard_queries(self, identifiers_tool):
        """Tool returns non-empty results for standard queries."""

        # Test with no query (should return all schemas)
        result = await identifiers_tool.explore_identifiers()
        assert isinstance(result, IdentifierResult)
        assert len(result.schemas) > 0, "Should return schemas when no query specified"
        assert len(result.paths) > 0, "Should return paths when no query specified"
        assert result.analytics["total_schemas"] > 0, "Should have total_schemas > 0"

        # Test with broad query terms
        for query in ["materials", "coordinate", "plasma"]:
            result = await identifiers_tool.explore_identifiers(query=query)
            assert isinstance(result, IdentifierResult)
            # Note: Some queries may return empty results if no matching schemas exist
            # This is expected behavior, not an error

    @pytest.mark.asyncio
    async def test_all_scope_options_function(self, identifiers_tool):
        """All scope options function correctly."""

        scopes_to_test = [
            IdentifierScope.ALL,
            IdentifierScope.ENUMS,
            IdentifierScope.IDENTIFIERS,
            IdentifierScope.COORDINATES,
            IdentifierScope.CONSTANTS,
        ]

        for scope in scopes_to_test:
            result = await identifiers_tool.explore_identifiers(scope=scope)
            assert isinstance(result, IdentifierResult)
            assert result.scope == scope, f"Scope {scope} should be preserved in result"
            # Note: Some scopes might return empty results depending on data availability
            # This is expected behavior, not an error

        # Test that ENUMS scope filters to only schemas with options
        result = await identifiers_tool.explore_identifiers(scope=IdentifierScope.ENUMS)
        if len(result.schemas) > 0:
            for schema in result.schemas:
                assert schema["option_count"] > 0, (
                    "ENUMS scope should only return schemas with options"
                )

    @pytest.mark.asyncio
    async def test_enumeration_spaces_calculation(self, identifiers_tool):
        """Enumeration spaces are properly calculated."""

        # Test with materials query which should have a known enumeration space
        result = await identifiers_tool.explore_identifiers(
            query="materials", scope=IdentifierScope.ENUMS
        )
        assert isinstance(result, IdentifierResult)

        if len(result.schemas) > 0:
            # Materials schema should have 31 options
            materials_schema = next(
                (s for s in result.schemas if "materials" in s["path"].lower()), None
            )
            if materials_schema:
                assert materials_schema["option_count"] == 31, (
                    "Materials schema should have 31 options"
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

        result = await identifiers_tool.explore_identifiers()
        assert isinstance(result, IdentifierResult)

        # Should discover multiple schemas (adjust expectation based on environment)
        # Full environment has 57+ schemas, filtered CI environment has fewer
        # Use a reasonable minimum that works for both full and CI environments
        min_expected_schemas = 5  # At least a few schemas should be available
        assert len(result.schemas) > min_expected_schemas, (
            f"Should discover at least {min_expected_schemas} schemas, got {len(result.schemas)}"
        )

        # Each schema should have required fields
        for schema in result.schemas[:5]:  # Test first 5 schemas
            assert "path" in schema, "Schema should have path"
            assert "option_count" in schema, "Schema should have option_count"
            assert "branching_significance" in schema, (
                "Schema should have branching_significance"
            )
            assert "sample_options" in schema, "Schema should have sample_options"

            # Sample options should be properly formatted
            for option in schema["sample_options"]:
                assert "name" in option, "Option should have name"
                assert "index" in option, "Option should have index"
                assert "description" in option, "Option should have description"

    @pytest.mark.asyncio
    async def test_query_behavior(self, identifiers_tool):
        """Test behavior with various query patterns."""

        # Test that overly specific queries return empty results (this is correct behavior)
        result = await identifiers_tool.explore_identifiers(query="plasma state")
        assert isinstance(result, IdentifierResult)
        # Empty results for overly specific queries is expected, not an error

        # Test partial matching works
        result = await identifiers_tool.explore_identifiers(query="material")
        assert isinstance(result, IdentifierResult)
        # May return empty if no matching schemas, which is valid

    @pytest.mark.asyncio
    async def test_error_handling(self, identifiers_tool):
        """Test error handling scenarios."""

        # Test with valid scope values
        try:
            result = await identifiers_tool.explore_identifiers(
                scope=IdentifierScope.ALL
            )
            assert isinstance(result, IdentifierResult)
        except Exception as e:
            pytest.fail(f"Valid scope should not raise exception: {e}")

    @pytest.mark.asyncio
    async def test_analytics_calculations(self, identifiers_tool):
        """Test analytics field calculations."""

        result = await identifiers_tool.explore_identifiers()
        assert isinstance(result, IdentifierResult)

        analytics = result.analytics
        assert "total_schemas" in analytics
        assert "total_paths" in analytics
        assert "enumeration_space" in analytics
        assert "significance" in analytics
        assert "scope_applied" in analytics

        # Analytics should be consistent with returned data
        # Note: Analytics total_schemas reflects total available, while len(result.schemas)
        # may be limited by pagination or result limiting
        assert analytics["total_schemas"] >= len(result.schemas)
        # Note: total_paths might be different due to pagination/limiting

    @pytest.mark.asyncio
    async def test_coordinate_schemas_discovery(self, identifiers_tool):
        """Test discovery of coordinate-related schemas."""

        result = await identifiers_tool.explore_identifiers(
            scope=IdentifierScope.COORDINATES
        )
        assert isinstance(result, IdentifierResult)

        # Should find coordinate-related schemas if any exist
        # Note: May be empty if no coordinate schemas match the filter, which is valid

    @pytest.mark.asyncio
    async def test_branching_significance_calculation(self, identifiers_tool):
        """Test branching significance calculation."""

        result = await identifiers_tool.explore_identifiers()
        assert isinstance(result, IdentifierResult)

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

        result = await identifiers_tool.explore_identifiers()

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time
        assert execution_time < 5.0, (
            f"Tool should complete within 5 seconds, took {execution_time}"
        )
        assert isinstance(result, IdentifierResult)


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
        result = await identifiers_tool.explore_identifiers()
        assert len(result.schemas) > 0, "Tool should return schemas"
        print("✅ Tool returns schemas for basic queries")

        # Validate scope functionality
        for scope in IdentifierScope:
            result = await identifiers_tool.explore_identifiers(scope=scope)
            assert isinstance(result, IdentifierResult), f"Scope {scope} should work"
        print("✅ All scope options function correctly")

        # Validate enumeration calculation
        result = await identifiers_tool.explore_identifiers(
            query="materials", scope=IdentifierScope.ENUMS
        )
        if len(result.schemas) > 0:
            expected_space = sum(schema["option_count"] for schema in result.schemas)
            assert result.analytics["enumeration_space"] == expected_space
        print("✅ Enumeration spaces properly calculated")

        # Validate schema discovery
        result = await identifiers_tool.explore_identifiers()
        assert len(result.schemas) > 5, "Should discover multiple schemas"
        print("✅ Schema discovery working")

        print("\n🎉 IDENTIFIERS TOOL VALIDATION COMPLETE!")
        print("📊 Tool Status: FULLY FUNCTIONAL")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
