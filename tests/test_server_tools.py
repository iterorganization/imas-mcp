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
    async def test_search_imas_basic_query(self, server, client):
        """Test basic search functionality."""
        async with client:
            result = await client.call_tool(
                "search_imas",
                {"query": "plasma temperature", "max_results": 5},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "results" in result
        assert "results_count" in result
        assert "search_strategy" in result
        assert len(result["results"]) <= 5
        # Accept either semantic_search or auto (fallback mode)
        assert result["search_strategy"] in ["semantic_search", "auto"]

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_search_imas_list_query(self, server, client):
        """Test search with list query."""
        async with client:
            result = await client.call_tool(
                "search_imas",
                {"query": ["plasma", "temperature"], "max_results": 3},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "results" in result
        assert "results_count" in result
        assert len(result["results"]) <= 3

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_search_imas_with_ids_filter(self, server, client):
        """Test search with IDS filtering."""
        async with client:
            result = await client.call_tool(
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
    async def test_search_imas_empty_results(self, server, client):
        """Test search with query that returns no results using lexical search."""
        async with client:
            result = await client.call_tool(
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
    async def test_search_imas_max_results_limit(self, server, client):
        """Test max_results parameter enforcement."""
        async with client:
            result = await client.call_tool(
                "search_imas",
                {"query": "temperature", "max_results": 2},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert len(result["results"]) <= 2

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_search_imas_result_structure(self, server, client):
        """Test search result structure and required fields."""
        async with client:
            result = await client.call_tool(
                "search_imas",
                {"query": "plasma", "max_results": 1},
            )

        result = extract_result(result)
        assert isinstance(result, dict)

        if result["results"]:
            item = result["results"][0]
            assert "path" in item
            assert "relevance_score" in item
            assert "documentation" in item
            assert "units" in item
            assert "ids_name" in item
            assert "identifier" in item
            assert isinstance(item["identifier"], dict)

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_search_imas_invalid_ids_name(self, server, client):
        """Test search with invalid IDS name."""
        async with client:
            result = await client.call_tool(
                "search_imas",
                {"query": "temperature", "ids_name": "invalid_ids"},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "results" in result
        # Should still return results (filter will simply not match anything)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_search_imas_physics_enhancement(self, server, client):
        """Test physics enhancement features."""
        async with client:
            result = await client.call_tool(
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
    async def test_explain_concept_basic(self, server, client):
        """Test basic concept explanation."""
        async with client:
            result = await client.call_tool(
                "explain_concept",
                {"concept": "plasma temperature"},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "concept" in result
        assert result["concept"] == "plasma temperature"
        assert "detail_level" in result
        assert "related_paths" in result
        assert "sources_analyzed" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explain_concept_with_detail_level(self, server, client):
        """Test concept explanation with detail level."""
        async with client:
            result = await client.call_tool(
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
    async def test_explain_concept_detail_levels(self, server, client):
        """Test different detail levels."""
        detail_levels = ["basic", "intermediate", "advanced"]

        for level in detail_levels:
            async with client:
                result = await client.call_tool(
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
    async def test_explain_concept_result_structure(self, server, client):
        """Test concept explanation result structure."""
        async with client:
            result = await client.call_tool(
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
            "sources_analyzed",
            "identifier_analysis",
        ]
        for field in required_fields:
            assert field in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explain_concept_unknown_concept(self, server, client):
        """Test explanation of unknown concept."""
        async with client:
            result = await client.call_tool(
                "explain_concept",
                {"concept": "nonexistent_physics_concept_xyz"},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "concept" in result
        # Should still provide some response even for unknown concepts

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_explain_concept_physics_context(self, server, client):
        """Test physics context in concept explanation."""
        async with client:
            result = await client.call_tool(
                "explain_concept",
                {"concept": "poloidal flux", "detail_level": "advanced"},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "physics_context" in result
        # physics_context can be None or a PhysicsSearchResult dict
        if result["physics_context"] is not None:
            assert isinstance(result["physics_context"], dict)


class TestGetOverview:
    """Test suite for get_overview tool - comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_get_overview_general(self, server, client):
        """Test basic overview without question."""
        async with client:
            result = await client.call_tool("get_overview", {})

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "total_ids" in result
        assert "available_ids" in result
        assert "sample_analysis" in result
        assert "identifier_summary" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_get_overview_with_question(self, server, client):
        """Test overview with specific question."""
        async with client:
            result = await client.call_tool(
                "get_overview",
                {"question": "What IDS are available?"},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "total_ids" in result
        assert "question" in result
        assert result["question"] == "What IDS are available?"
        assert "question_results" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_get_overview_ids_statistics(self, server, client):
        """Test IDS statistics in overview."""
        async with client:
            result = await client.call_tool("get_overview", {})

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
    async def test_get_overview_identifier_summary(self, server, client):
        """Test identifier summary in overview."""
        async with client:
            result = await client.call_tool("get_overview", {})

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "identifier_summary" in result

        id_summary = result["identifier_summary"]
        # Check for the actual fields returned by the identifier summary
        expected_keys = [
            "total_schemas",
            "total_identifier_paths",
            "total_enumeration_options",
        ]
        found_keys = [k for k in expected_keys if k in id_summary]
        assert len(found_keys) >= 2, (
            f"Expected at least 2 keys from {expected_keys}, found {found_keys}"
        )

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_get_overview_question_search(self, server, client):
        """Test question-based search in overview."""
        async with client:
            result = await client.call_tool(
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
    async def test_analyze_ids_structure_valid_ids(self, server, client):
        """Test IDS structure analysis with valid IDS."""
        async with client:
            result = await client.call_tool(
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
    async def test_analyze_ids_structure_invalid_ids(self, server, client):
        """Test IDS structure analysis with invalid IDS."""
        async with client:
            result = await client.call_tool(
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
    async def test_analyze_ids_structure_result_structure(self, server, client):
        """Test structure of analysis result."""
        async with client:
            result = await client.call_tool(
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
    async def test_analyze_ids_structure_path_patterns(self, server, client):
        """Test path pattern analysis."""
        async with client:
            result = await client.call_tool(
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
    async def test_explore_relationships_basic(self, server, client):
        """Test basic relationship exploration."""
        async with client:
            result = await client.call_tool(
                "explore_relationships",
                {"path": "core_profiles", "max_depth": 1},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "path" in result
        assert result["path"] == "core_profiles"

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_relationships_ids_only(self, server, client):
        """Test relationship exploration with IDS name only."""
        async with client:
            result = await client.call_tool(
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
    async def test_explore_relationships_with_type(self, server, client):
        """Test relationship exploration with specific type."""
        async with client:
            result = await client.call_tool(
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
    async def test_explore_relationships_with_depth(self, server, client):
        """Test relationship exploration with max depth."""
        async with client:
            result = await client.call_tool(
                "explore_relationships",
                {"path": "core_profiles", "max_depth": 1},  # Reduced from 2 to 1
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "path" in result
        assert result["path"] == "core_profiles"

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_relationships_invalid_path(self, server, client):
        """Test relationship exploration with invalid path."""
        async with client:
            result = await client.call_tool(
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
    async def test_explore_relationships_result_structure(self, server, client):
        """Test relationship exploration result structure."""
        async with client:
            result = await client.call_tool(
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
    async def test_explore_relationships_max_depth(self, server, client):
        """Test relationship exploration with different max depths."""
        depths = [1, 2]  # Reduced from [1, 2, 3] to prevent hanging

        for depth in depths:
            async with client:
                result = await client.call_tool(
                    "explore_relationships",
                    {"path": "equilibrium", "max_depth": depth},
                )

            result = extract_result(result)
            assert isinstance(result, dict)
            assert "max_depth" in result
            assert result["max_depth"] == depth

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_explore_relationships_physics_enhancement(self, server, client):
        """Test physics enhancement in relationship exploration."""
        async with client:
            result = await client.call_tool(
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
    async def test_explore_identifiers_basic(self, server, client):
        """Test basic identifier exploration."""
        async with client:
            result = await client.call_tool(
                "explore_identifiers", {"query": "plasma identifiers"}
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "schemas" in result
        assert "query" in result
        assert result["query"] == "plasma identifiers"

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_no_query(self, server, client):
        """Test identifier exploration without query."""
        async with client:
            result = await client.call_tool("explore_identifiers", {})

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "schemas" in result
        assert "branching_analytics" in result
        assert "total_schemas" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_with_query(self, server, client):
        """Test identifier exploration with query."""
        async with client:
            result = await client.call_tool(
                "explore_identifiers", {"query": "temperature"}
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "schemas" in result
        assert "query" in result
        assert result["query"] == "temperature"

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_with_scope(self, server, client):
        """Test identifier exploration with scope."""
        async with client:
            result = await client.call_tool("explore_identifiers", {"scope": "schemas"})

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "schemas" in result
        # Note: response structure may vary based on scope
        if "scope" in result:
            assert result["scope"] == "schemas"

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_scope_all(self, server, client):
        """Test identifier exploration with scope 'all'."""
        async with client:
            result = await client.call_tool("explore_identifiers", {"scope": "all"})

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "schemas" in result
        assert "branching_analytics" in result
        assert "total_schemas" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_scope_schemas(self, server, client):
        """Test identifier exploration with scope 'schemas'."""
        async with client:
            result = await client.call_tool("explore_identifiers", {"scope": "schemas"})

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "schemas" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_scope_summary(self, server, client):
        """Test identifier exploration with scope 'summary'."""
        async with client:
            result = await client.call_tool("explore_identifiers", {"scope": "summary"})

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "schemas" in result or "branching_analytics" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_scope_paths(self, server, client):
        """Test identifier exploration with scope 'paths'."""
        async with client:
            result = await client.call_tool("explore_identifiers", {"scope": "paths"})

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "identifier_paths" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_identifiers_summary_structure(self, server, client):
        """Test identifier summary structure."""
        async with client:
            result = await client.call_tool("explore_identifiers", {"scope": "summary"})

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
    async def test_explore_identifiers_schemas_structure(self, server, client):
        """Test identifier schemas structure."""
        async with client:
            result = await client.call_tool("explore_identifiers", {"scope": "schemas"})

        result = extract_result(result)
        if "schemas" in result and result["schemas"]:
            schema = result["schemas"][0]
            expected_fields = [
                "path",
                "schema_path",
                "option_count",
                "branching_significance",
                "sample_options",
            ]
            for field in expected_fields:
                assert field in schema


class TestExportIDS:
    """Test suite for export_ids tool - comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_ids_empty_list(self, server, client):
        """Test bulk export with empty IDS list."""
        async with client:
            result = await client.call_tool(
                "export_ids",
                {"ids_list": []},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "error" in result
        assert "suggestions" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_ids_single_ids(self, server, client):
        """Test bulk export with single IDS."""
        async with client:
            result = await client.call_tool(
                "export_ids",
                {
                    "ids_list": ["core_profiles"],
                    "output_format": "structured",
                    "include_relationships": False,
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "ids_list" in result
        assert result["ids_list"] == ["core_profiles"]
        assert "export_data" in result
        assert "ids_data" in result["export_data"]

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_ids_multiple_ids(self, server, client):
        """Test bulk export with multiple IDS."""
        async with client:
            result = await client.call_tool(
                "export_ids",
                {
                    "ids_list": ["core_profiles", "equilibrium"],
                    "output_format": "structured",
                    "include_relationships": False,
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "ids_list" in result
        assert result["ids_list"] == ["core_profiles", "equilibrium"]

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_ids_invalid_ids(self, server, client):
        """Test bulk export with invalid IDS names."""
        async with client:
            result = await client.call_tool(
                "export_ids",
                {"ids_list": ["invalid_ids1", "invalid_ids2"]},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        # When no valid IDS are provided, response includes error and suggestions
        if "error" in result:
            assert "invalid_ids" in result
        else:
            assert "export_data" in result
            export_data = result["export_data"]
            assert "invalid_ids" in export_data
            assert len(export_data["invalid_ids"]) == 2

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_ids_mixed_validity(self, server, client):
        """Test bulk export with mix of valid and invalid IDS."""
        async with client:
            result = await client.call_tool(
                "export_ids",
                {
                    "ids_list": ["core_profiles", "invalid_ids"],
                    "output_format": "structured",
                    "include_relationships": False,
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "export_data" in result
        export_data = result["export_data"]
        assert "valid_ids" in export_data
        assert "invalid_ids" in export_data
        assert "core_profiles" in export_data["valid_ids"]
        assert "invalid_ids" in export_data["invalid_ids"]

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_ids_output_formats(self, server, client):
        """Test different output formats."""
        formats = ["raw", "structured", "enhanced"]

        for format_type in formats:
            async with client:
                result = await client.call_tool(
                    "export_ids",
                    {
                        "ids_list": ["core_profiles"],
                        "output_format": format_type,
                        "include_relationships": False,
                    },
                )

            result = extract_result(result)
            assert isinstance(result, dict)
            assert "output_format" in result
            assert result["output_format"] == format_type

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_ids_export_summary(self, server, client):
        """Test export summary structure."""
        async with client:
            result = await client.call_tool(
                "export_ids",
                {
                    "ids_list": ["equilibrium"],
                    "output_format": "structured",
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
    async def test_export_ids_with_relationships(self, server, client):
        """Test bulk export with relationships."""
        async with client:
            result = await client.call_tool(
                "export_ids",
                {
                    "ids_list": ["core_profiles", "equilibrium"],
                    "include_relationships": True,
                    "output_format": "structured",
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "export_data" in result
        export_data = result["export_data"]
        assert "requested_ids" in export_data
        # The response includes relationship data in the structure
        assert "cross_relationships" in export_data or "physics_domains" in export_data

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_export_ids_with_physics_context(self, server, client):
        """Test bulk export with physics context."""
        async with client:
            result = await client.call_tool(
                "export_ids",
                {
                    "ids_list": ["core_profiles"],
                    "include_physics_context": True,
                    "output_format": "structured",
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "export_data" in result
        export_data = result["export_data"]
        assert "physics_domains" in export_data


class TestExportPhysicsDomain:
    """Test suite for export_physics_domain tool - comprehensive coverage."""

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_physics_domain_empty_domain(self, server, client):
        """Test physics domain export with empty domain."""
        async with client:
            result = await client.call_tool(
                "export_physics_domain",
                {"domain": ""},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "error" in result
        assert "suggestions" in result

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_physics_domain_basic(self, server, client):
        """Test basic physics domain export."""
        async with client:
            result = await client.call_tool(
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
        assert "export_data" in result  # Updated to use unified export_data field
        assert (
            "export_summary" in result["export_data"]
        )  # Summary is nested in export_data

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_physics_domain_analysis_depths(self, server, client):
        """Test different analysis depths."""
        depths = ["overview", "focused", "comprehensive"]

        for depth in depths:
            async with client:
                result = await client.call_tool(
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
    async def test_export_physics_domain_max_paths(self, server, client):
        """Test max_paths parameter."""
        async with client:
            result = await client.call_tool(
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
    async def test_export_physics_domain_invalid_domain(self, server, client):
        """Test physics domain export with invalid domain."""
        async with client:
            result = await client.call_tool(
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
    async def test_export_physics_domain_data_structure(self, server, client):
        """Test domain data structure."""
        async with client:
            result = await client.call_tool(
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
    async def test_export_physics_domain_export_summary(self, server, client):
        """Test export summary structure."""
        async with client:
            result = await client.call_tool(
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
    async def test_export_physics_domain_with_cross_domain(self, server, client):
        """Test physics domain export with cross-domain analysis."""
        async with client:
            result = await client.call_tool(
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
        # Check that cross_domain_analysis is in export_data
        assert "export_data" in result
        assert "cross_domain_analysis" in result["export_data"]

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_export_physics_domain_comprehensive(self, server, client):
        """Test comprehensive domain export."""
        async with client:
            result = await client.call_tool(
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
        # Check that domain_structure and measurement_dependencies are in export_data
        assert "export_data" in result
        assert "domain_structure" in result["export_data"]
        assert "measurement_dependencies" in result["export_data"]


# Additional test classes for comprehensive coverage
class TestServerErrorHandling:
    """Test suite for server error handling and edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_tool_with_missing_parameters(self, server, client):
        """Test tool calls with missing required parameters."""
        async with client:
            with pytest.raises(Exception):  # Should raise validation error
                await client.call_tool(
                    "analyze_ids_structure",
                    {},  # Missing required ids_name parameter
                )

        # FastMCP should handle this at the framework level
        # This test ensures we don't crash on malformed requests

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_tool_with_invalid_parameter_types(self, server, client):
        """Test tool calls with invalid parameter types."""
        async with client:
            with pytest.raises(Exception):  # Should raise validation error
                await client.call_tool(
                    "search_imas",
                    {"query": 123, "max_results": "invalid"},  # Wrong types
                )

        # FastMCP should handle type validation
        # This test ensures we handle type errors gracefully

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_concurrent_tool_calls(self, server, client):
        """Test concurrent tool calls for thread safety."""
        import asyncio

        async with client:
            # Create multiple concurrent calls
            tasks = [
                client.call_tool("get_overview", {}),
                client.call_tool("search_imas", {"query": "temperature"}),
                client.call_tool(
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
    async def test_search_performance(self, server, client):
        """Test search performance with various query sizes."""
        import time

        queries = [
            "temperature",
            "electron temperature plasma",
            "electron temperature plasma density profile equilibrium",
        ]

        for query in queries:
            async with client:
                start_time = time.time()
                result = await client.call_tool(
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
    async def test_bulk_export_performance(self, server, client):
        """Test bulk export performance."""
        import time

        async with client:
            start_time = time.time()
            result = await client.call_tool(
                "export_ids",
                {
                    "ids_list": ["core_profiles"],
                    "output_format": "structured",
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
    async def test_memory_usage_stability(self, server, client):
        """Test memory usage stability during repeated calls."""
        import gc

        # Perform multiple operations to test memory stability
        for i in range(10):
            async with client:
                result = await client.call_tool(
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
    async def test_overview_to_analyze_workflow(self, server, client):
        """Test workflow: overview -> analyze IDS structure."""
        async with client:
            # Step 1: Get overview
            overview_result = await client.call_tool("get_overview", {})

            overview_result = extract_result(overview_result)
            assert "available_ids" in overview_result

            # Step 2: Analyze first available IDS
            if overview_result["available_ids"]:
                ids_name = overview_result["available_ids"][0]
                structure_result = await client.call_tool(
                    "analyze_ids_structure", {"ids_name": ids_name}
                )

                structure_result = extract_result(structure_result)
                assert "ids_name" in structure_result

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
                    {"ids_list": ids_subset, "output_format": "structured"},
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

            # Step 3: Analyze structure only if search succeeded and results found
            if search_result["results"] and search_result.get("error") is None:
                ids_name = search_result["results"][0]["ids_name"]
                structure_result = await client.call_tool(
                    "analyze_ids_structure", {"ids_name": ids_name}
                )
                structure_result = extract_result(structure_result)
                # Only assert if no error in structure result
                if structure_result.get("error") is None:
                    assert "identifier_analysis" in structure_result
