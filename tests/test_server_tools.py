import pytest

from tests.conftest import extract_result


class TestSearchImas:
    """Test suite for search_imas implementation."""

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
        assert len(result["results"]) <= 5

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
    async def test_search_imas_list_query(self, test_server):
        """Test search with list query."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "search_imas",
                {
                    "query": ["plasma", "temperature"],
                    "max_results": 3,
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "results" in result
        assert "total_results" in result


class TestExplainConcept:
    """Test suite for explain_concept implementation."""

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


class TestGetOverview:
    """Test suite for get_overview implementation."""

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_get_overview_general(self, test_server):
        """Test general overview."""
        async with test_server.client:
            result = await test_server.client.call_tool("get_overview", {})

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "total_documents" in result
        assert "available_ids" in result

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


class TestAnalyzeIDSStructure:
    """Test suite for analyze_ids_structure implementation."""

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


class TestExploreRelationships:
    """Test suite for explore_relationships implementation."""

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_explore_relationships_basic(self, test_server):
        """Test basic relationship exploration."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_relationships",
                {"path": "core_profiles/profiles_1d", "max_depth": 1},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "path" in result
        assert result["path"] == "core_profiles/profiles_1d"

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

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_explore_relationships_with_depth(self, test_server):
        """Test relationship exploration with max depth."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "explore_relationships",
                {"path": "core_profiles", "max_depth": 2},
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "path" in result
        assert result["path"] == "core_profiles"


class TestExploreIdentifiers:
    """Test suite for explore_identifiers implementation."""

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


class TestExportIDSBulk:
    """Test suite for export_ids_bulk implementation."""

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
    @pytest.mark.slow
    async def test_export_ids_bulk_with_relationships(self, test_server):
        """Test bulk export with relationships."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_ids_bulk",
                {
                    "ids_list": ["core_profiles"],
                    "include_relationships": True,
                    "output_format": "structured",
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "requested_ids" in result
        # The response includes relationship data in the structure
        assert "cross_relationships" in result or "physics_domains" in result


class TestExportPhysicsDomain:
    """Test suite for export_physics_domain implementation."""

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
                    "max_paths": 3,  # Reduced for performance
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "domain" in result
        assert result["domain"] == "core_profiles"

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

    @pytest.mark.asyncio
    @pytest.mark.fast
    async def test_export_physics_domain_with_analysis_depth(self, test_server):
        """Test physics domain export with analysis depth."""
        async with test_server.client:
            result = await test_server.client.call_tool(
                "export_physics_domain",
                {
                    "domain": "transport",
                    "analysis_depth": "overview",
                    "max_paths": 3,  # Reduced for performance
                },
            )

        result = extract_result(result)
        assert isinstance(result, dict)
        assert "domain" in result
        assert "analysis_depth" in result
        assert result["analysis_depth"] == "overview"
