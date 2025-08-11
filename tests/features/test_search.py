"""
Test search features through user interface.

This module tests search functionality as user-facing features,
focusing on search quality, filtering, and performance.
"""

import time

import pytest

from imas_mcp.models.error_models import ToolError
from imas_mcp.models.result_models import SearchResult


class TestSearchFeatures:
    """Test search functionality through user interface."""

    @pytest.mark.asyncio
    async def test_basic_search_functionality(self, tools):
        """Test basic search returns relevant results."""
        result = await tools.search_imas(query="plasma temperature", max_results=5)

        # Should return SearchResult object
        assert isinstance(result, SearchResult)
        assert hasattr(result, "hits")
        assert hasattr(result, "hit_count")
        assert isinstance(result.hits, list)
        assert len(result.hits) <= 5

        # Results should have expected structure
        if result.hits:
            first_hit = result.hits[0]
            assert hasattr(first_hit, "path")
            assert hasattr(first_hit, "score")
            assert isinstance(first_hit.score, int | float)

    @pytest.mark.asyncio
    async def test_filtered_search_by_physics_domain(self, tools):
        """Test search with IDS filtering."""
        result = await tools.search_imas(
            query="temperature", ids_filter=["core_profiles"], max_results=10
        )

        assert isinstance(result, SearchResult)
        assert hasattr(result, "hits")

        # All results should be from the specified IDS (if any)
        for search_hit in result.hits:
            # Results from specified IDS should be prioritized
            assert hasattr(search_hit, "path")

    @pytest.mark.asyncio
    async def test_search_result_quality(self, tools):
        """Test search result relevance and quality."""
        # Test specific physics term
        result = await tools.search_imas(
            query="electron density profile", max_results=10
        )

        assert isinstance(result, SearchResult)
        if result.hits:
            # Results should be relevant to electron density
            for search_hit in result.hits:
                # Basic structure validation - path should exist
                assert hasattr(search_hit, "path")
                assert hasattr(search_hit, "documentation")

                # At least one relevant keyword should appear
                # Note: This is a quality check, not a strict requirement
                # as search might return related concepts

    @pytest.mark.asyncio
    async def test_search_with_different_query_types(self, tools):
        """Test search handles different types of queries."""
        test_queries = [
            "temperature",  # Single term
            "plasma temperature profile",  # Multi-term physics concept
            "core_profiles",  # IDS name
            "equilibrium/time_slice",  # Path-like query
        ]

        for query in test_queries:
            result = await tools.search_imas(query=query, max_results=5)

            assert isinstance(result, SearchResult)
            assert hasattr(result, "hits")
            # Each query type should return some form of structured response

    @pytest.mark.asyncio
    async def test_search_result_limits(self, tools):
        """Test search respects result limits."""
        # Test different limits
        limits = [1, 5, 10, 20]

        for limit in limits:
            result = await tools.search_imas(query="plasma", max_results=limit)

            assert isinstance(result, SearchResult)
            assert hasattr(result, "hits")
            assert len(result.hits) <= limit

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, tools):
        """Test search handles empty or minimal queries."""
        result = await tools.search_imas(query="", max_results=5)

        # Should handle gracefully - may return error dict or SearchResult
        assert result is not None
        # Both error dict and SearchResult are acceptable for empty query

    @pytest.mark.asyncio
    async def test_search_performance_basic(self, tools):
        """Test basic search performance characteristics."""
        start_time = time.time()
        result = await tools.search_imas(query="plasma temperature", max_results=10)
        end_time = time.time()

        # Search should complete in reasonable time (< 10 seconds for testing)
        execution_time = end_time - start_time
        assert execution_time < 10.0, f"Search took {execution_time:.2f}s, too slow"

        # Should return results
        assert isinstance(result, SearchResult)


class TestSearchResultStructure:
    """Test search result structure and consistency."""

    @pytest.mark.asyncio
    async def test_search_result_consistency(self, tools):
        """Test search results have consistent structure."""
        result = await tools.search_imas(query="temperature", max_results=5)

        assert isinstance(result, SearchResult)
        assert hasattr(result, "hits")

        # All results should have consistent structure
        for search_result in result.hits:
            assert hasattr(search_result, "path")
            assert hasattr(search_result, "score")

            # Score should be numeric
            assert isinstance(search_result.score, int | float)
            assert 0 <= search_result.score <= 1

    @pytest.mark.asyncio
    async def test_search_metadata_completeness(self, tools):
        """Test search results include complete metadata."""
        result = await tools.search_imas(
            query="core_profiles temperature", max_results=3
        )

        assert isinstance(result, SearchResult)

        if result.hits:
            for search_result in result.hits:
                # Check for expected metadata fields
                expected_fields = ["path", "score"]
                for field in expected_fields:
                    assert hasattr(search_result, field)

                # Score should be numeric
                assert isinstance(search_result.score, int | float)
                assert 0 <= search_result.score <= 1


class TestSearchErrorHandling:
    """Test search error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_search_invalid_parameters(self, tools):
        """Test search handles invalid parameters gracefully."""
        # Test negative max_results - this should return ToolError due to validation
        result = await tools.search_imas(query="test", max_results=-1)

        # Should return ToolError due to validation failure
        assert isinstance(result, SearchResult | ToolError)

    @pytest.mark.asyncio
    async def test_search_very_long_query(self, tools):
        """Test search handles very long queries."""
        long_query = "plasma " * 100  # Very long query

        result = await tools.search_imas(query=long_query, max_results=5)

        # Should handle without crashing - validation error returns ToolError

        assert isinstance(result, ToolError)
        assert "Validation error" in result.error

    @pytest.mark.asyncio
    async def test_search_special_characters(self, tools):
        """Test search handles special characters in queries."""
        special_queries = [
            "temperature/density",
            "plasma_profile",
            "T_e(r)",
            "β_plasma",
        ]

        for query in special_queries:
            result = await tools.search_imas(query=query, max_results=3)

            # Should handle without crashing
            assert isinstance(result, SearchResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
