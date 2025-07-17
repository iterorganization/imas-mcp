import pytest

from tests.conftest import extract_result


class TestIntegration:
    """Integration tests for MCP tool interactions."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.fast
    async def test_search_to_explain_workflow(self, test_server):
        """Test workflow: search -> explain concept."""
        async with test_server.client:
            # Step 1: Search for concepts
            search_result = await test_server.client.call_tool(
                "search_imas",
                {"query": "plasma temperature", "max_results": 5},
            )

            search_result = extract_result(search_result)
            assert "results" in search_result

            # Step 2: Explain a concept found in search
            if search_result["results"]:
                explain_result = await test_server.client.call_tool(
                    "explain_concept",
                    {"concept": "plasma temperature"},
                )

                explain_result = extract_result(explain_result)
                assert "concept" in explain_result

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.fast
    async def test_search_to_structure_analysis_workflow(self, test_server):
        """Test workflow: search -> analyze IDS structure."""
        async with test_server.client:
            # Step 1: Search for paths
            search_result = await test_server.client.call_tool(
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
                structure_result = await test_server.client.call_tool(
                    "analyze_ids_structure", {"ids_name": ids_name}
                )

                structure_result = extract_result(structure_result)
                assert "ids_name" in structure_result

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_bulk_export_workflow(self, test_server):
        """Test bulk export workflow."""
        async with test_server.client:
            # Get overview first
            overview_result = await test_server.client.call_tool("get_overview", {})

            overview_result = extract_result(overview_result)
            assert "available_ids" in overview_result

            # Export subset of available IDS
            if overview_result["available_ids"]:
                ids_subset = overview_result["available_ids"][:3]
                export_result = await test_server.client.call_tool(
                    "export_ids_bulk", {"ids_list": ids_subset}
                )

                export_result = extract_result(export_result)
                assert "requested_ids" in export_result
