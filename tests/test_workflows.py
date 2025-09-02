"""
Test complete user interaction workflows.

This module tests end-to-end user workflows that span multiple tools,
focusing on realistic user scenarios and tool interaction patterns.
"""

import asyncio
import time

import pytest

from imas_mcp.models.error_models import ToolError
from imas_mcp.models.result_models import (
    ConceptResult,
    DomainExport,
    IdentifierResult,
    IDSExport,
    OverviewResult,
    RelationshipResult,
    SearchResult,
    StructureResult,
)


class TestUserWorkflows:
    """Test complete user interaction workflows."""

    @pytest.mark.asyncio
    async def test_discovery_workflow(self, tools, workflow_test_data):
        """Test: overview → search → explain → analyze workflow."""
        # Step 1: Get overview to understand what's available
        overview = await tools.get_overview()
        assert isinstance(overview, OverviewResult)

        if overview.available_ids:
            # Step 2: Search for specific content from test dataset IDS
            search_query = "core_profiles temperature"  # Target our test dataset
            search_result = await tools.search_imas(query=search_query, max_results=5)
            assert isinstance(search_result, SearchResult)

            if search_result.hits:
                # Step 3: Explain a concept found in search
                first_result = search_result.hits[0]
                if hasattr(first_result, "path"):
                    # Extract IDS name from the search result path
                    path_parts = first_result.path.split("/")
                    concept = path_parts[0] if path_parts else "core_profiles"

                    explain_result = await tools.explain_concept(concept=concept)
                    assert isinstance(explain_result, ConceptResult)

                    # Step 4: Analyze structure of the IDS
                    if concept in overview.available_ids:
                        analysis_result = await tools.analyze_ids_structure(
                            ids_name=concept
                        )
                        assert isinstance(analysis_result, StructureResult)

    @pytest.mark.asyncio
    async def test_research_workflow(self, tools, workflow_test_data):
        """Test: search → relationships → deep analysis workflow."""
        # Step 1: Search for physics concept
        search_query = workflow_test_data["search_query"]
        search_result = await tools.search_imas(query=search_query, max_results=10)
        assert isinstance(search_result, SearchResult)

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

            relationships_result = await tools.explore_relationships(
                path=f"{ids_name}/profiles_1d/time"
            )
            assert isinstance(relationships_result, RelationshipResult)

            # Step 3: Deep analysis of structure
            analysis_result = await tools.analyze_ids_structure(ids_name=ids_name)
            assert isinstance(analysis_result, StructureResult)

            # Step 4: Explore identifiers for comprehensive understanding
            identifiers_result = await tools.explore_identifiers()
            assert isinstance(identifiers_result, IdentifierResult)

    @pytest.mark.asyncio
    async def test_export_workflow(self, tools, workflow_test_data):
        """Test: search → filter → export workflow."""
        # Step 1: Search for content in specific domain
        export_domain = workflow_test_data["export_domain"]
        search_result = await tools.search_imas(
            query="temperature profile", max_results=5
        )
        assert isinstance(search_result, SearchResult)

        # Step 2: Export the physics domain based on search
        domain_export = await tools.export_physics_domain(domain=export_domain)
        assert isinstance(domain_export, DomainExport)

        # Step 3: Export specific IDS if found in search
        if search_result.hits:
            for result in search_result.hits:
                if hasattr(result, "ids_name"):
                    ids_name = result.ids_name
                    ids_export = await tools.export_ids(ids_list=[ids_name])
                    assert isinstance(ids_export, IDSExport)
                    break

    @pytest.mark.asyncio
    async def test_physics_workflow(self, tools, workflow_test_data):
        """Test physics domain exploration workflow."""
        concept = workflow_test_data["concept_to_explain"]

        # Step 1: Explain physics concept
        explanation = await tools.explain_concept(concept=concept)
        assert isinstance(explanation, ConceptResult)

        # Step 2: Search for related content
        search_result = await tools.search_imas(query=concept, max_results=10)
        assert isinstance(search_result, SearchResult)

        # Step 3: Analyze structure of relevant IDS
        if search_result.hits:
            for result in search_result.hits:
                if hasattr(result, "ids_name"):
                    ids_name = result.ids_name
                    analysis = await tools.analyze_ids_structure(ids_name=ids_name)
                    assert isinstance(analysis, StructureResult)
                    break

    @pytest.mark.asyncio
    async def test_comprehensive_exploration_workflow(self, tools):
        """Test comprehensive exploration of a single IDS."""
        ids_name = "core_profiles"  # Well-known IDS for testing

        # Step 1: Explain the IDS concept
        explanation = await tools.explain_concept(concept=ids_name)
        assert isinstance(explanation, ConceptResult)

        # Step 2: Analyze its structure
        structure = await tools.analyze_ids_structure(ids_name=ids_name)
        assert isinstance(structure, StructureResult)

        # Step 3: Explore relationships
        relationships = await tools.explore_relationships(
            path=f"{ids_name}/profiles_1d/time"
        )
        assert isinstance(relationships, RelationshipResult)

        # Step 4: Explore identifiers
        identifiers = await tools.explore_identifiers()
        assert isinstance(identifiers, IdentifierResult)

        # Step 5: Export the data
        export = await tools.export_ids(ids_list=[ids_name])
        assert isinstance(export, IDSExport)

        # Step 6: Search within this IDS
        search = await tools.search_imas(query=f"{ids_name} temperature", max_results=5)
        assert isinstance(search, SearchResult)


class TestWorkflowPerformance:
    """Test workflow performance characteristics."""

    @pytest.mark.asyncio
    async def test_workflow_total_time(self, tools):
        """Test complete workflow completes in reasonable time."""
        start_time = time.time()

        # Execute a typical workflow
        overview = await tools.get_overview()
        search = await tools.search_imas(query="temperature", max_results=3)
        explanation = await tools.explain_concept(concept="core_profiles")

        end_time = time.time()

        total_time = end_time - start_time
        assert total_time < 30.0, f"Workflow took {total_time:.2f}s, too slow"

        # All steps should complete successfully
        assert isinstance(overview, OverviewResult)
        assert isinstance(search, SearchResult)
        assert isinstance(explanation, ConceptResult)

    @pytest.mark.asyncio
    async def test_concurrent_tool_usage(self, tools):
        """Test tools can be used concurrently without interference."""
        # Run multiple tools concurrently
        tasks = [
            tools.get_overview(),
            tools.search_imas(query="temperature", max_results=3),
            tools.explain_concept(concept="equilibrium"),
        ]

        results = await asyncio.gather(*tasks)

        # All tasks should complete successfully
        assert len(results) == 3
        assert isinstance(results[0], OverviewResult)  # overview
        assert isinstance(results[1], SearchResult)  # search
        assert isinstance(results[2], ConceptResult)  # explanation


class TestWorkflowErrorRecovery:
    """Test workflow error handling and recovery."""

    @pytest.mark.asyncio
    async def test_workflow_continues_after_error(self, tools):
        """Test workflow can continue after one step fails."""
        # Step 1: Valid operation
        overview = await tools.get_overview()
        assert isinstance(overview, OverviewResult)

        # Step 2: Operation that might fail
        invalid_analysis = await tools.analyze_ids_structure(ids_name="invalid_ids")
        assert isinstance(invalid_analysis, ToolError)
        # This should return an ToolError for invalid IDS

        # Step 3: Continue with valid operation
        search = await tools.search_imas(query="temperature", max_results=3)
        assert isinstance(search, SearchResult)

        # Workflow should complete despite the error in step 2

    @pytest.mark.asyncio
    async def test_workflow_error_information_quality(self, tools):
        """Test workflow errors provide useful information for users."""
        # Try analysis with invalid IDS
        result = await tools.analyze_ids_structure(ids_name="invalid_ids")

        assert isinstance(result, ToolError)
        # Error should be informative
        assert isinstance(result.error, str)
        assert len(result.error) > 0

        # Should provide suggestions for recovery
        assert hasattr(result, "suggestions")
        assert isinstance(result.suggestions, list)


class TestWorkflowDataConsistency:
    """Test data consistency across workflow steps."""

    @pytest.mark.asyncio
    async def test_search_to_analysis_consistency(self, tools):
        """Test data is consistent between search and analysis."""
        # Search for content
        search_result = await tools.search_imas(
            query="core_profiles temperature", max_results=5
        )

        if search_result.hits:
            for result in search_result.hits:
                if hasattr(result, "ids_name"):
                    ids_name = result.ids_name

                    # Analyze the same IDS
                    analysis = await tools.analyze_ids_structure(ids_name=ids_name)

                    # IDS name should be consistent
                    if isinstance(analysis, StructureResult):
                        assert analysis.ids_name == ids_name
                    break

    @pytest.mark.asyncio
    async def test_overview_to_export_consistency(self, tools):
        """Test consistency between overview and export."""
        # Get overview
        overview = await tools.get_overview()

        if "available_ids" in overview and overview["available_ids"]:
            # Pick first available IDS
            ids_name = overview["available_ids"][0]

            # Export should work for available IDS
            export = await tools.export_ids(ids_name=ids_name)

            if "error" not in export:
                assert export["ids_name"] == ids_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
