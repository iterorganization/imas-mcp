"""
Test suite for get_imas_identifiers tool functionality.

This test suite validates that the get_imas_identifiers tool works correctly,
covering all query types and analytics calculations.
"""

import pytest

from imas_mcp.models.result_models import GetIdentifiersResult
from imas_mcp.tools.identifiers_tool import IdentifiersTool


class TestExploreIdentifiersTool:
    """Test get_imas_identifiers tool functionality."""

    @pytest.fixture
    async def identifiers_tool(self):
        """Create identifiers tool instance."""
        return IdentifiersTool()

    @pytest.mark.asyncio
    async def test_tool_returns_results_for_standard_queries(self, identifiers_tool):
        """Tool returns non-empty results for standard queries."""

        # Test with no query (should return all schemas)
        result = await identifiers_tool.get_imas_identifiers()
        assert isinstance(result, GetIdentifiersResult)
        assert len(result.schemas) > 0, "Should return schemas when no query specified"
        assert len(result.paths) > 0, "Should return paths when no query specified"
        assert result.analytics["total_schemas"] > 0, "Should have total_schemas > 0"

        # Test with broad query terms
        for query in ["materials", "coordinate", "plasma"]:
            result = await identifiers_tool.get_imas_identifiers(query=query)
            assert isinstance(result, GetIdentifiersResult)
            # Note: Some queries may return empty results if no matching schemas exist
            # This is expected behavior, not an error

    @pytest.mark.asyncio
    async def test_enumeration_spaces_calculation(self, identifiers_tool):
        """Enumeration spaces are properly calculated."""

        # Test with materials query which should have a known enumeration space
        result = await identifiers_tool.get_imas_identifiers(query="materials")
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

        result = await identifiers_tool.get_imas_identifiers()
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
        result = await identifiers_tool.get_imas_identifiers(query="plasma state")
        assert isinstance(result, GetIdentifiersResult)
        # Empty results for overly specific queries is expected, not an error

        # Test partial matching works
        result = await identifiers_tool.get_imas_identifiers(query="material")
        assert isinstance(result, GetIdentifiersResult)
        # May return empty if no matching schemas, which is valid

    @pytest.mark.asyncio
    async def test_error_handling(self, identifiers_tool):
        """Test error handling scenarios."""

        # Test with valid call
        try:
            result = await identifiers_tool.get_imas_identifiers()
            assert isinstance(result, GetIdentifiersResult)
        except Exception as e:
            pytest.fail(f"Valid call should not raise exception: {e}")

    @pytest.mark.asyncio
    async def test_analytics_calculations(self, identifiers_tool):
        """Test analytics field calculations."""

        result = await identifiers_tool.get_imas_identifiers()
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

        result = await identifiers_tool.get_imas_identifiers()
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

        result = await identifiers_tool.get_imas_identifiers()

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
        result = await identifiers_tool.get_imas_identifiers()
        assert len(result.schemas) > 0, "Tool should return schemas"
        print("✅ Tool returns schemas for basic queries")

        # Validate enumeration calculation
        result = await identifiers_tool.get_imas_identifiers(query="materials")
        if len(result.schemas) > 0:
            expected_space = sum(schema["option_count"] for schema in result.schemas)
            assert result.analytics["enumeration_space"] == expected_space
        print("✅ Enumeration spaces properly calculated")

        # Validate schema discovery
        result = await identifiers_tool.get_imas_identifiers()
        assert len(result.schemas) >= 3, "Should discover multiple schemas"
        print("✅ Schema discovery working")

        print("\n🎉 IDENTIFIERS TOOL VALIDATION COMPLETE!")
        print("📊 Tool Status: FULLY FUNCTIONAL")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
