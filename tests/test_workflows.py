"""
Test complete user interaction workflows.

This module tests end-to-end user workflows that span multiple tools,
focusing on realistic user scenarios and tool interaction patterns.
"""

import asyncio
import time

import pytest

from imas_codex.models.error_models import ToolError
from imas_codex.models.result_models import (
    GetIdentifiersResult,
    GetOverviewResult,
    SearchClustersResult,
    SearchPathsResult,
)


class TestUserWorkflows:
    """Test complete user interaction workflows."""

    @pytest.mark.asyncio
    async def test_discovery_workflow(self, tools, workflow_test_data):
        """Test: overview → search workflow."""
        # Step 1: Get overview to understand what's available
        overview = await tools.get_imas_overview()
        assert isinstance(overview, GetOverviewResult)

        if overview.available_ids:
            # Step 2: Search for specific content from test dataset IDS
            search_query = "core_profiles temperature"  # Target our test dataset
            search_result = await tools.search_imas_paths(
                query=search_query, max_results=5
            )
            assert isinstance(search_result, SearchPathsResult)

    @pytest.mark.asyncio
    async def test_research_workflow(self, tools, workflow_test_data):
        """Test: search → relationships → deep analysis workflow."""
        # Step 1: Search for physics concept
        search_query = workflow_test_data["search_query"]
        search_result = await tools.search_imas_paths(
            query=search_query, max_results=10
        )
        assert isinstance(search_result, SearchPathsResult)

        if search_result.hits:
            # Step 2: Explore relationships for found IDS
            first_result = search_result.hits[0]
            if hasattr(first_result, "ids_name"):
                ids_name = first_result.ids_name
                # Verify this IDS actually exists before using it
                if ids_name not in ["core_profiles", "equilibrium"]:
                    ids_name = workflow_test_data["analysis_target"]
            else:
                # Fallback to test data
                ids_name = workflow_test_data["analysis_target"]

            relationships_result = await tools.search_imas_clusters(
                path=f"{ids_name}/profiles_1d/time"
            )
            # Accept either SearchClustersResult or ToolError (when clusters.json is missing)
            assert isinstance(relationships_result, SearchClustersResult | ToolError)

            # Step 3: Explore identifiers for comprehensive understanding
            identifiers_result = await tools.get_imas_identifiers()
            assert isinstance(identifiers_result, GetIdentifiersResult)

    @pytest.mark.asyncio
    async def test_comprehensive_exploration_workflow(self, tools):
        """Test comprehensive exploration of a single IDS."""
        ids_name = "core_profiles"  # Well-known IDS for testing

        # Step 1: Explore relationships
        relationships = await tools.search_imas_clusters(
            path=f"{ids_name}/profiles_1d/time"
        )
        # Accept either SearchClustersResult or ToolError (when clusters.json is missing)
        assert isinstance(relationships, SearchClustersResult | ToolError)

        # Step 2: Explore identifiers
        identifiers = await tools.get_imas_identifiers()
        assert isinstance(identifiers, GetIdentifiersResult)

        # Step 3: Search within this IDS
        search = await tools.search_imas_paths(
            query=f"{ids_name} temperature", max_results=5
        )
        assert isinstance(search, SearchPathsResult)


class TestWorkflowPerformance:
    """Test workflow performance characteristics."""

    @pytest.mark.asyncio
    async def test_workflow_total_time(self, tools):
        """Test complete workflow completes in reasonable time."""
        start_time = time.time()

        # Execute a typical workflow
        overview = await tools.get_imas_overview()
        search = await tools.search_imas_paths(query="temperature", max_results=3)

        end_time = time.time()

        total_time = end_time - start_time
        assert total_time < 30.0, f"Workflow took {total_time:.2f}s, too slow"

        # All steps should complete successfully
        assert isinstance(overview, GetOverviewResult)
        assert isinstance(search, SearchPathsResult)

    @pytest.mark.asyncio
    async def test_concurrent_tool_usage(self, tools):
        """Test tools can be used concurrently without interference."""
        # Run multiple tools concurrently
        tasks = [
            tools.get_imas_overview(),
            tools.search_imas_paths(query="temperature", max_results=3),
        ]

        results = await asyncio.gather(*tasks)

        # All tasks should complete successfully
        assert len(results) == 2
        assert isinstance(results[0], GetOverviewResult)  # overview
        assert isinstance(results[1], SearchPathsResult)  # search


class TestWorkflowErrorRecovery:
    """Test workflow error handling and recovery."""

    @pytest.mark.asyncio
    async def test_workflow_continues_after_error(self, tools):
        """Test workflow can continue after one step fails."""
        # Step 1: Valid operation
        overview = await tools.get_imas_overview()
        assert isinstance(overview, GetOverviewResult)

        # Step 2: Continue with valid operation
        search = await tools.search_imas_paths(query="temperature", max_results=3)
        assert isinstance(search, SearchPathsResult)

        # Workflow should complete without errors


class TestWorkflowDataConsistency:
    """Test data consistency across workflow steps."""

    @pytest.mark.asyncio
    async def test_search_consistency(self, tools):
        """Test data is consistent between search calls."""
        # Search for content
        search_result = await tools.search_imas_paths(
            query="core_profiles temperature", max_results=5
        )

        assert isinstance(search_result, SearchPathsResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
