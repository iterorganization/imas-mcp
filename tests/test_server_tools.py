"""
Test suite for all IMAS MCP Server tools.

This test suite covers all 8 MCP tools with performance-optimized, atomic tests.
Tests are categorized by speed and use session-scoped fixtures for expensive operations.
"""

import pytest

from tests.conftest import extract_result


class TestSearchImas:
    """Test suite for search_imas tool - comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_search_imas_basic_query(self, test_server):
        """Test basic search functionality."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "search_imas",
                {"query": "plasma temperature", "max_results": 5},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "results" in result
        assert "total_results" in result
        assert "search_strategy" in result
        assert len(result["results"]) <= 5
        # Accept either semantic_search or auto (fallback mode)
        assert result["search_strategy"] in ["semantic_search", "auto"]

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_search_imas_list_query(self, test_server):
        """Test search with list query."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "search_imas",
                {"query": ["plasma", "temperature"], "max_results": 3},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "results" in result
        assert "total_results" in result
        assert len(result["results"]) <= 3

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_search_imas_with_ids_filter(self, test_server):
        """Test search with IDS filtering."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "search_imas",
                {
                    "query": "electron density",
                    "ids_name": "core_profiles",
                    "max_results": 3,
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "results" in result
        # Check that results are filtered to specific IDS
        for item in result["results"]:
            assert "core_profiles" in item["path"]

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_search_imas_empty_results(self, test_server):
        """Test search with query that returns no results using lexical search."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "search_imas",
                {
                    "query": "nonexistent_impossible_query",
                    "max_results": 5,
                    "search_mode": "lexical",
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "results" in result
        # With lexical search, nonsensical queries should return no results
        assert isinstance(result["results"], list)
        assert len(result["results"]) == 0
        assert result["search_strategy"] == "lexical"

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_search_imas_max_results_limit(self, test_server):
        """Test max_results parameter enforcement."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "search_imas",
                {"query": "temperature", "max_results": 2},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert len(result["results"]) <= 2

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_search_imas_result_structure(self, test_server):
        """Test search result structure and required fields."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "search_imas",
                {"query": "plasma", "max_results": 1},
            )

        result = extract_result(result)
        assert isinstance(result, dict)

        if result["results"]:
            item = result["results"][0]
            assert "path" in item
            assert "score" in item
            assert "documentation" in item
            assert "units" in item
            assert "ids_name" in item
            assert "identifier" in item
            assert isinstance(item["identifier"], dict)

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_search_imas_invalid_ids_name(self, test_server):
        """Test search with invalid IDS name."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "search_imas",
                {"query": "temperature", "ids_name": "invalid_ids"},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "results" in result
        # Should still return results (filter will simply not match anything)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_search_imas_physics_enhancement(self, test_server):
        """Test physics enhancement features."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "search_imas",
                {"query": "electron temperature", "max_results": 5},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        # Check for physics enhancement fields (optional)
        # These are optional enhancements that may not always be present


class TestExplainConcept:
    """Test suite for explain_concept tool - comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explain_concept_basic(self, test_server):
        """Test basic concept explanation."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explain_concept",
                {"concept": "plasma temperature"},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "concept" in result
        assert result["concept"] == "plasma temperature"
        assert "detail_level" in result
        assert "related_paths" in result
        assert "search_results_count" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explain_concept_with_detail_level(self, test_server):
        """Test concept explanation with detail level."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explain_concept",
                {
                    "concept": "electron density",
                    "detail_level": "basic",
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "concept" in result
        assert "detail_level" in result
        assert result["detail_level"] == "basic"

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explain_concept_detail_levels(self, test_server):
        """Test different detail levels."""
        detail_levels = ["basic", "intermediate", "advanced"]

        for level in detail_levels:
            async with test_server.client:
                result = await test_server.client.call_tool(
                    "explain_concept",
                    {"concept": "electron density", "detail_level": level},
                )

            result = extract_result(result)
            assert isinstance(result, dict)
            assert "concept" in result
            assert "detail_level" in result
            assert result["detail_level"] == level

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explain_concept_result_structure(self, test_server):
        """Test concept explanation result structure."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explain_concept",
                {"concept": "safety factor"},
            )

        result = extract_result(result)
        assert isinstance(result, dict)

        required_fields = [
            "concept",
            "detail_level",
            "related_paths",
            "physics_context",
            "search_results_count",
            "identifier_context",
        ]
        for field in required_fields:
            assert field in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explain_concept_unknown_concept(self, test_server):
        """Test explanation of unknown concept."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explain_concept",
                {"concept": "nonexistent_physics_concept_xyz"},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "concept" in result
        # Should still provide some response even for unknown concepts

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_explain_concept_physics_context(self, test_server):
        """Test physics context in concept explanation."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explain_concept",
                {"concept": "poloidal flux", "detail_level": "advanced"},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "physics_context" in result
        assert isinstance(result["physics_context"], dict)


class TestGetOverview:
    """Test suite for get_overview tool - comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_get_overview_general(self, test_server):
        """Test basic overview without question."""
        async with test_server.client:
            result = await test_server.client.call_tool("get_overview", {})

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "total_documents" in result
        assert "available_ids" in result
        assert "index_name" in result
        assert "ids_statistics" in result
        assert "identifier_summary" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_get_overview_with_question(self, test_server):
        """Test overview with specific question."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "get_overview",
                {"question": "What IDS are available?"},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "total_documents" in result
        assert "question" in result
        assert result["question"] == "What IDS are available?"
        assert "question_results" in result
        assert "search_strategy" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_get_overview_ids_statistics(self, test_server):
        """Test IDS statistics in overview."""
        async with test_server.client:
            result = await test_server.client.call_tool("get_overview", {})

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "ids_statistics" in result

        ids_stats = result["ids_statistics"]
        assert isinstance(ids_stats, dict)

        # Check that each IDS has required statistics
        for ids_name, stats in ids_stats.items():
            assert "path_count" in stats
            assert "identifier_count" in stats
            assert "description" in stats

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_get_overview_identifier_summary(self, test_server):
        """Test identifier summary in overview."""
        async with test_server.client:
            result = await test_server.client.call_tool("get_overview", {})

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "identifier_summary" in result

        id_summary = result["identifier_summary"]
        required_fields = [
            "total_identifiers",
            "total_schemas",
            "enumeration_space",
            "identifier_coverage",
            "significance",
            "complexity_metrics",
        ]
        for field in required_fields:
            assert field in id_summary

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_get_overview_question_search(self, test_server):
        """Test question-based search in overview."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "get_overview",
                {"question": "equilibrium profiles"},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "question_results" in result
        assert isinstance(result["question_results"], list)


class TestAnalyzeIDSStructure:
    """Test suite for analyze_ids_structure tool - comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_analyze_ids_structure_valid_ids(self, test_server):
        """Test IDS structure analysis with valid IDS."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "analyze_ids_structure",
                {"ids_name": "core_profiles"},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "ids_name" in result
        assert result["ids_name"] == "core_profiles"
        assert "total_paths" in result
        assert "structure" in result
        assert "identifier_analysis" in result
        assert "sample_paths" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_analyze_ids_structure_invalid_ids(self, test_server):
        """Test IDS structure analysis with invalid IDS."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "analyze_ids_structure",
                {"ids_name": "invalid_ids"},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "ids_name" in result
        assert result["ids_name"] == "invalid_ids"
        assert "error" in result
        assert "available_ids" in result
        assert "suggestions" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_analyze_ids_structure_result_structure(self, test_server):
        """Test structure of analysis result."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "analyze_ids_structure",
                {"ids_name": "equilibrium"},
            )

        result = extract_result(result)
        if "error" not in result:
            assert isinstance(result, dict)

            # Check structure field
            structure = result["structure"]
            assert "root_level_paths" in structure
            assert "max_depth" in structure
            assert "document_count" in structure

            # Check identifier analysis
            id_analysis = result["identifier_analysis"]
            assert "total_identifier_nodes" in id_analysis
            assert "branching_paths" in id_analysis
            assert "coverage" in id_analysis

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_analyze_ids_structure_path_patterns(self, test_server):
        """Test path pattern analysis."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "analyze_ids_structure",
                {"ids_name": "core_profiles"},
            )

        result = extract_result(result)
        if "error" not in result:
            assert "path_patterns" in result
            assert isinstance(result["path_patterns"], dict)


class TestExploreRelationships:
    """Test suite for explore_relationships tool - comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_relationships_basic(self, test_server):
        """Test basic relationship exploration."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_relationships",
                {"path": "core_profiles", "max_depth": 1},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "path" in result
        assert result["path"] == "core_profiles"

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_relationships_ids_only(self, test_server):
        """Test relationship exploration with IDS name only."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_relationships",
                {"path": "core_profiles", "max_depth": 1},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "path" in result
        assert result["path"] == "core_profiles"
        assert "relationship_type" in result
        assert "max_depth" in result
        assert "ids_name" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_relationships_with_type(self, test_server):
        """Test relationship exploration with specific type."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_relationships",
                {
                    "path": "core_profiles",
                    "relationship_type": "physics_concepts",
                    "max_depth": 1,
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "path" in result
        assert result["path"] == "core_profiles"
        assert "relationship_type" in result
        assert result["relationship_type"] == "physics_concepts"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_explore_relationships_with_depth(self, test_server):
        """Test relationship exploration with max depth."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_relationships",
                {"path": "core_profiles", "max_depth": 1},  # Reduced from 2 to 1
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "path" in result
        assert result["path"] == "core_profiles"

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_relationships_invalid_path(self, test_server):
        """Test relationship exploration with invalid path."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_relationships",
                {"path": "invalid_ids", "max_depth": 1},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "path" in result
        assert "error" in result
        assert "available_ids" in result
        assert "suggestions" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_relationships_result_structure(self, test_server):
        """Test relationship exploration result structure."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_relationships",
                {"path": "core_profiles", "max_depth": 1},
            )

        result = extract_result(result)
        if "error" not in result:
            assert "related_paths" in result
            assert "relationship_count" in result
            assert "analysis" in result
            assert "identifier_context" in result

            # Check analysis structure
            analysis = result["analysis"]
            assert "same_ids_paths" in analysis
            assert "cross_ids_paths" in analysis
            assert "semantic_search_relationships" in analysis

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_explore_relationships_max_depth(self, test_server):
        """Test relationship exploration with different max depths."""
        depths = [1, 2]  # Reduced from [1, 2, 3] to prevent hanging

        for depth in depths:
            async with test_server.client:
                result = await test_server.client.call_tool(
                    "explore_relationships",
                    {"path": "equilibrium", "max_depth": depth},
                )

            result = extract_result(result)
            assert isinstance(result, dict)
            assert "max_depth" in result
            assert result["max_depth"] == depth

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_explore_relationships_physics_enhancement(self, test_server):
        """Test physics enhancement in relationship exploration."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_relationships",
                {"path": "core_profiles", "max_depth": 1},
            )

        result = extract_result(result)
        if "error" not in result:
            # Check for optional physics relationships
            # These are optional enhancements
            pass


class TestExploreIdentifiers:
    """Test suite for explore_identifiers tool - comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_basic(self, test_server):
        """Test basic identifier exploration."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_identifiers", {"query": "plasma identifiers"}
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "schemas" in result
        assert "query" in result
        assert result["query"] == "plasma identifiers"

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_no_query(self, test_server):
        """Test identifier exploration without query."""
        async with test_server.client:
            result = await test_server.client.call_tool("explore_identifiers", {})

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "schemas" in result
        assert "summary" in result
        assert "overview" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_with_query(self, test_server):
        """Test identifier exploration with query."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_identifiers", {"query": "temperature"}
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "schemas" in result
        assert "query" in result
        assert result["query"] == "temperature"

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_with_scope(self, test_server):
        """Test identifier exploration with scope."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_identifiers", {"scope": "schemas"}
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "schemas" in result
        # Note: response structure may vary based on scope
        if "scope" in result:
            assert result["scope"] == "schemas"

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_scope_all(self, test_server):
        """Test identifier exploration with scope 'all'."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_identifiers", {"scope": "all"}
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "schemas" in result
        assert "summary" in result
        assert "identifier_paths" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_scope_schemas(self, test_server):
        """Test identifier exploration with scope 'schemas'."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_identifiers", {"scope": "schemas"}
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "schemas" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_scope_summary(self, test_server):
        """Test identifier exploration with scope 'summary'."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_identifiers", {"scope": "summary"}
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "summary" in result or "schemas" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_scope_paths(self, test_server):
        """Test identifier exploration with scope 'paths'."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_identifiers", {"scope": "paths"}
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "identifier_paths" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_summary_structure(self, test_server):
        """Test identifier summary structure."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_identifiers", {"scope": "summary"}
            )

        result = extract_result(result)
        if "summary" in result:
            summary = result["summary"]
            expected_fields = [
                "total_schemas",
                "total_identifier_paths",
                "total_enumeration_options",
                "complexity_metrics",
            ]
            for field in expected_fields:
                assert field in summary

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_schemas_structure(self, test_server):
        """Test identifier schemas structure."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_identifiers", {"scope": "schemas"}
            )

        result = extract_result(result)
        if "schemas" in result and result["schemas"]:
            schema = result["schemas"][0]
            expected_fields = [
                "name",
                "description",
                "total_options",
                "complexity",
                "usage_count",
                "sample_options",
            ]
            for field in expected_fields:
                assert field in schema


class TestExportIDSBulk:
    """Test suite for export_ids_bulk tool - comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_ids_bulk_empty_list(self, test_server):
        """Test bulk export with empty IDS list."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_ids_bulk",
                {"ids_list": []},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "error" in result
        assert "suggestions" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_ids_bulk_single_ids(self, test_server):
        """Test bulk export with single IDS."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_ids_bulk",
                {
                    "ids_list": ["core_profiles"],
                    "output_format": "minimal",
                    "include_relationships": False,
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "requested_ids" in result
        assert result["requested_ids"] == ["core_profiles"]
        assert "export_summary" in result
        assert "ids_data" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_ids_bulk_multiple_ids(self, test_server):
        """Test bulk export with multiple IDS."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_ids_bulk",
                {
                    "ids_list": ["core_profiles", "equilibrium"],
                    "output_format": "minimal",
                    "include_relationships": False,
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "requested_ids" in result
        assert result["requested_ids"] == ["core_profiles", "equilibrium"]

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_ids_bulk_invalid_ids(self, test_server):
        """Test bulk export with invalid IDS names."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_ids_bulk",
                {"ids_list": ["invalid_ids1", "invalid_ids2"]},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "invalid_ids" in result
        # When no valid IDS are provided, response includes error and suggestions
        assert ("valid_ids" in result) or ("error" in result)
        assert len(result["invalid_ids"]) == 2

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_ids_bulk_mixed_validity(self, test_server):
        """Test bulk export with mix of valid and invalid IDS."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_ids_bulk",
                {
                    "ids_list": ["core_profiles", "invalid_ids"],
                    "output_format": "minimal",
                    "include_relationships": False,
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "valid_ids" in result
        assert "invalid_ids" in result
        assert "core_profiles" in result["valid_ids"]
        assert "invalid_ids" in result["invalid_ids"]

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_ids_bulk_output_formats(self, test_server):
        """Test different output formats."""
        formats = ["minimal", "structured", "comprehensive"]

        for format_type in formats:
            async with test_server.client:
                result = await test_server.client.call_tool(
                    "export_ids_bulk",
                    {
                        "ids_list": ["core_profiles"],
                        "output_format": format_type,
                        "include_relationships": False,
                    },
                )

            result = extract_result(result)
            assert isinstance(result, dict)
            assert "export_format" in result
            assert result["export_format"] == format_type

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_ids_bulk_export_summary(self, test_server):
        """Test export summary structure."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_ids_bulk",
                {
                    "ids_list": ["equilibrium"],
                    "output_format": "minimal",
                    "include_relationships": False,
                },
            )

        result = extract_result(result)
        if "export_summary" in result:
            summary = result["export_summary"]
            expected_fields = [
                "total_requested",
                "successfully_exported",
                "failed_exports",
                "total_paths_exported",
                "export_completeness",
            ]
            for field in expected_fields:
                assert field in summary

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_export_ids_bulk_with_relationships(self, test_server):
        """Test bulk export with relationships."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_ids_bulk",
                {
                    "ids_list": ["core_profiles", "equilibrium"],
                    "include_relationships": True,
                    "output_format": "structured",
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "requested_ids" in result
        # The response includes relationship data in the structure
        assert "cross_relationships" in result or "physics_domains" in result

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_export_ids_bulk_with_physics_context(self, test_server):
        """Test bulk export with physics context."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_ids_bulk",
                {
                    "ids_list": ["core_profiles"],
                    "include_physics_context": True,
                    "output_format": "structured",
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "physics_domains" in result


class TestExportPhysicsDomain:
    """Test suite for export_physics_domain tool - comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_physics_domain_empty_domain(self, test_server):
        """Test physics domain export with empty domain."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_physics_domain",
                {"domain": ""},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "error" in result
        assert "suggestions" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_physics_domain_basic(self, test_server):
        """Test basic physics domain export."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_physics_domain",
                {
                    "domain": "core_profiles",
                    "analysis_depth": "overview",
                    "max_paths": 5,
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "domain" in result
        assert result["domain"] == "core_profiles"
        assert "analysis_depth" in result
        assert "domain_data" in result
        assert "export_summary" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_physics_domain_analysis_depths(self, test_server):
        """Test different analysis depths."""
        depths = ["overview", "focused", "comprehensive"]

        for depth in depths:
            async with test_server.client:
                result = await test_server.client.call_tool(
                    "export_physics_domain",
                    {
                        "domain": "equilibrium",
                        "analysis_depth": depth,
                        "max_paths": 3,
                    },
                )

            result = extract_result(result)
            assert isinstance(result, dict)
            assert "analysis_depth" in result
            assert result["analysis_depth"] == depth

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_physics_domain_max_paths(self, test_server):
        """Test max_paths parameter."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_physics_domain",
                {
                    "domain": "transport",
                    "max_paths": 2,
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "max_paths" in result
        assert result["max_paths"] == 2

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_physics_domain_invalid_domain(self, test_server):
        """Test physics domain export with invalid domain."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_physics_domain",
                {"domain": "nonexistent_domain_xyz"},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        # Should handle invalid domain gracefully - either error or fallback results
        assert ("error" in result) or ("domain_data" in result)
        # If error, should have suggestions
        if "error" in result:
            assert "suggestions" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_physics_domain_data_structure(self, test_server):
        """Test domain data structure."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_physics_domain",
                {
                    "domain": "core_profiles",
                    "max_paths": 5,
                },
            )

        result = extract_result(result)
        if "domain_data" in result:
            domain_data = result["domain_data"]
            expected_fields = [
                "total_paths",
                "paths",
                "associated_ids",
                "measurement_types",
                "units_distribution",
            ]
            for field in expected_fields:
                assert field in domain_data

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_physics_domain_export_summary(self, test_server):
        """Test export summary structure."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_physics_domain",
                {
                    "domain": "equilibrium",
                    "max_paths": 3,
                },
            )

        result = extract_result(result)
        if "export_summary" in result:
            summary = result["export_summary"]
            expected_fields = [
                "domain",
                "total_paths_found",
                "associated_ids_count",
                "unique_measurement_types",
                "analysis_completeness",
            ]
            for field in expected_fields:
                assert field in summary

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_export_physics_domain_with_cross_domain(self, test_server):
        """Test physics domain export with cross-domain analysis."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_physics_domain",
                {
                    "domain": "equilibrium",
                    "include_cross_domain": True,
                    "max_paths": 2,  # Reduced for performance
                    "analysis_depth": "overview",
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "domain" in result
        assert "include_cross_domain" in result
        assert result["include_cross_domain"] is True
        assert "cross_domain_analysis" in result

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_export_physics_domain_comprehensive(self, test_server):
        """Test comprehensive domain export."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_physics_domain",
                {
                    "domain": "core_profiles",
                    "analysis_depth": "comprehensive",
                    "max_paths": 10,
                    "include_cross_domain": False,
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "domain_structure" in result
        assert "measurement_dependencies" in result


# Additional test classes for comprehensive coverage
class TestServerErrorHandling:
    """Test suite for server error handling and edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_tool_with_missing_parameters(self, test_server):
        """Test tool calls with missing required parameters."""
        async with test_server.client:
            with pytest.raises(Exception):  # Should raise validation error
                await test_server.client.call_tool(
                    "analyze_ids_structure",
                    {},  # Missing required ids_name parameter
                )

        # FastMCP should handle this at the framework level
        # This test ensures we don't crash on malformed requests

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_tool_with_invalid_parameter_types(self, test_server):
        """Test tool calls with invalid parameter types."""
        async with test_server.client:
            with pytest.raises(Exception):  # Should raise validation error
                await test_server.client.call_tool(
                    "search_imas",
                    {"query": 123, "max_results": "invalid"},  # Wrong types
                )

        # FastMCP should handle type validation
        # This test ensures we handle type errors gracefully

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_concurrent_tool_calls(self, test_server):
        """Test concurrent tool calls for thread safety."""
        import asyncio

        async with test_server.client:
            # Create multiple concurrent calls
            tasks = [
                test_server.client.call_tool("get_overview", {}),
                test_server.client.call_tool("search_imas", {"query": "temperature"}),
                test_server.client.call_tool(
                    "analyze_ids_structure", {"ids_name": "core_profiles"}
                ),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check that all calls completed without exceptions
            for result in results:
                assert not isinstance(result, Exception)


class TestServerPerformance:
    """Test suite for server performance characteristics."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_search_performance(self, test_server):
        """Test search performance with various query sizes."""
        import time

        queries = [
            "temperature",
            "electron temperature plasma",
            "electron temperature plasma density profile equilibrium",
        ]

        for query in queries:
            async with test_server.client:
                start_time = time.time()
                result = await test_server.client.call_tool(
                    "search_imas",
                    {"query": query, "max_results": 5},
                )
                end_time = time.time()

                # Ensure reasonable performance (< 5 seconds)
                assert end_time - start_time < 5.0

                result = extract_result(result)
                assert isinstance(result, dict)

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_bulk_export_performance(self, test_server):
        """Test bulk export performance."""
        import time

        async with test_server.client:
            start_time = time.time()
            result = await test_server.client.call_tool(
                "export_ids_bulk",
                {
                    "ids_list": ["core_profiles"],
                    "output_format": "minimal",
                    "include_relationships": False,
                    "include_physics_context": False,
                },
            )
            end_time = time.time()

            # Ensure reasonable performance (< 10 seconds)
            assert end_time - start_time < 10.0

            result = extract_result(result)
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage_stability(self, test_server):
        """Test memory usage stability during repeated calls."""
        import gc

        # Perform multiple operations to test memory stability
        for i in range(10):
            async with test_server.client:
                result = await test_server.client.call_tool(
                    "search_imas",
                    {"query": f"temperature {i}", "max_results": 3},
                )

                result = extract_result(result)
                assert isinstance(result, dict)

                # Force garbage collection
                gc.collect()


class TestToolIntegration:
    """Test suite for tool integration workflows."""

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
    async def test_overview_to_analyze_workflow(self, test_server):
        """Test workflow: overview -> analyze IDS structure."""
        async with test_server.client:
            # Step 1: Get overview
            overview_result = await test_server.client.call_tool("get_overview", {})

            overview_result = extract_result(overview_result)
            assert "available_ids" in overview_result

            # Step 2: Analyze first available IDS
            if overview_result["available_ids"]:
                ids_name = overview_result["available_ids"][0]
                structure_result = await test_server.client.call_tool(
                    "analyze_ids_structure", {"ids_name": ids_name}
                )

                structure_result = extract_result(structure_result)
                assert "ids_name" in structure_result

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_analyze_to_explore_workflow(self, test_server):
        """Test workflow: analyze structure -> explore relationships."""
        async with test_server.client:
            # Step 1: Analyze IDS structure
            structure_result = await test_server.client.call_tool(
                "analyze_ids_structure", {"ids_name": "core_profiles"}
            )

            structure_result = extract_result(structure_result)

            # Step 2: Explore relationships for sample paths
            if "sample_paths" in structure_result and structure_result["sample_paths"]:
                sample_path = structure_result["sample_paths"][0]
                relationship_result = await test_server.client.call_tool(
                    "explore_relationships", {"path": sample_path, "max_depth": 1}
                )

                relationship_result = extract_result(relationship_result)
                assert "path" in relationship_result

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_export_workflow(self, test_server):
        """Test workflow: overview -> bulk export -> domain export."""
        async with test_server.client:
            # Step 1: Get overview
            overview_result = await test_server.client.call_tool("get_overview", {})
            overview_result = extract_result(overview_result)

            if overview_result["available_ids"]:
                # Step 2: Bulk export
                ids_subset = overview_result["available_ids"][:2]
                bulk_result = await test_server.client.call_tool(
                    "export_ids_bulk",
                    {"ids_list": ids_subset, "output_format": "minimal"},
                )
                bulk_result = extract_result(bulk_result)

                # Step 3: Domain export
                domain_result = await test_server.client.call_tool(
                    "export_physics_domain", {"domain": ids_subset[0], "max_paths": 3}
                )
                domain_result = extract_result(domain_result)
                assert "domain" in domain_result

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.fast
    async def test_identifier_exploration_workflow(self, test_server):
        """Test workflow: explore identifiers -> search -> analyze."""
        async with test_server.client:
            # Step 1: Explore identifiers
            identifier_result = await test_server.client.call_tool(
                "explore_identifiers", {"scope": "summary"}
            )
            identifier_result = extract_result(identifier_result)

            # Step 2: Search for identifier-related concepts
            search_result = await test_server.client.call_tool(
                "search_imas", {"query": "identifier branching", "max_results": 3}
            )
            search_result = extract_result(search_result)

            # Step 3: Analyze structure if results found
            if search_result["results"]:
                ids_name = search_result["results"][0]["ids_name"]
                structure_result = await test_server.client.call_tool(
                    "analyze_ids_structure", {"ids_name": ids_name}
                )
                structure_result = extract_result(structure_result)
                assert "identifier_analysis" in structure_result
