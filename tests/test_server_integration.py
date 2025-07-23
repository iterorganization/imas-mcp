import pytest

from tests.conftest import extract_result


class TestIntegration:
    """Integration tests for MCP tool interactions."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.fast
    async def test_search_to_explain_workflow(self, server, client):
        """Test workflow: search -> explain concept."""
        async with client:
            # Step 1: Search for concepts
            search_result = await client.call_tool(
                "search_imas",
                {"query": "plasma temperature", "max_results": 5},
            )

            search_result = extract_result(search_result)
            assert "results" in search_result

            # Step 2: Explain a concept found in search
            if search_result["results"]:
                explain_result = await client.call_tool(
                    "explain_concept",
                    {"concept": "plasma temperature"},
                )

                explain_result = extract_result(explain_result)
                assert "concept" in explain_result

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.fast
    async def test_search_to_structure_analysis_workflow(self, server, client):
        """Test workflow: search -> analyze IDS structure."""
        async with client:
            # Step 1: Search for paths
            search_result = await client.call_tool(
                "search_imas",
                {
                    "query": "equilibrium profiles",
                    "max_results": 10,
                },
            )

            search_result = extract_result(search_result)
            assert "results" in search_result

            # Step 2: Analyze structure of found IDS
            if search_result["results"]:
                ids_name = search_result["results"][0]["ids_name"]
                structure_result = await client.call_tool(
                    "analyze_ids_structure", {"ids_name": ids_name}
                )

                structure_result = extract_result(structure_result)
                assert "ids_name" in structure_result

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_bulk_export_workflow(self, server, client):
        """Test bulk export workflow."""
        async with client:
            # Get overview first
            overview_result = await client.call_tool("get_overview", {})

            overview_result = extract_result(overview_result)
            assert "available_ids" in overview_result

            # Export subset of available IDS
            if overview_result["available_ids"]:
                ids_subset = overview_result["available_ids"][:3]
                export_result = await client.call_tool(
                    "export_ids", {"ids_list": ids_subset}
                )

                export_result = extract_result(export_result)
                assert "requested_ids" in export_result["export_data"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_analyze_to_explore_workflow(self, server, client):
        """Test workflow: analyze structure -> explore relationships."""
        async with client:
            # Step 1: Analyze IDS structure
            structure_result = await client.call_tool(
                "analyze_ids_structure", {"ids_name": "core_profiles"}
            )

            structure_result = extract_result(structure_result)

            # Step 2: Explore relationships for sample paths
            if "sample_paths" in structure_result and structure_result["sample_paths"]:
                sample_path = structure_result["sample_paths"][0]
                relationship_result = await client.call_tool(
                    "explore_relationships", {"path": sample_path, "max_depth": 1}
                )

                relationship_result = extract_result(relationship_result)
                assert "path" in relationship_result

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_export_workflow(self, server, client):
        """Test workflow: overview -> bulk export -> domain export."""
        async with client:
            # Step 1: Get overview
            overview_result = await client.call_tool("get_overview", {})
            overview_result = extract_result(overview_result)

            if overview_result["available_ids"]:
                # Step 2: Bulk export
                ids_subset = overview_result["available_ids"][:2]
                bulk_result = await client.call_tool(
                    "export_ids",
                    {"ids_list": ids_subset, "output_format": "minimal"},
                )
                bulk_result = extract_result(bulk_result)

                # Step 3: Domain export
                domain_result = await client.call_tool(
                    "export_physics_domain", {"domain": ids_subset[0], "max_paths": 3}
                )
                domain_result = extract_result(domain_result)
                assert "domain" in domain_result

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.fast
    async def test_identifier_exploration_workflow(self, server, client):
        """Test workflow: explore identifiers -> search -> analyze."""
        async with client:
            # Step 1: Explore identifiers
            identifier_result = await client.call_tool(
                "explore_identifiers", {"scope": "summary"}
            )
            identifier_result = extract_result(identifier_result)

            # Step 2: Search for identifier-related concepts
            search_result = await client.call_tool(
                "search_imas", {"query": "identifier branching", "max_results": 3}
            )
            search_result = extract_result(search_result)

            # Step 3: Analyze structure if results found
            if search_result["results"]:
                ids_name = search_result["results"][0]["ids_name"]
                structure_result = await client.call_tool(
                    "analyze_ids_structure", {"ids_name": ids_name}
                )
                structure_result = extract_result(structure_result)
                assert (
                    "identifier_analysis" in structure_result
                    or "ids_name" in structure_result
                )
