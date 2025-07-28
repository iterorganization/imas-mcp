"""
Integration tests for IMAS MCP tools with API-matching mocked data.

Tests tool interactions, data flow, and business logic using mocked data
that matches the actual API schemas and return types.
"""

import pytest
import asyncio
from unittest.mock import Mock
from fastmcp import Context

from imas_mcp.tools.overview_tool import OverviewTool
from imas_mcp.tools.analysis_tool import AnalysisTool
from imas_mcp.tools.relationships_tool import RelationshipsTool
from imas_mcp.tools.identifiers_tool import IdentifiersTool
from imas_mcp.tools.export_tool import ExportTool


class TestApiIntegration:
    """Test tool integration with proper API-matching data."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock MCP context."""
        context = Mock(spec=Context)
        context.session = Mock()
        context.meta = Mock()
        return context

    @pytest.fixture
    def mock_document_store(self):
        """Create mock document store with properly structured data."""
        mock_store = Mock()

        # Available IDS names
        mock_store.get_available_ids.return_value = [
            "core_profiles",
            "equilibrium",
            "disruptions",
            "transport",
        ]

        # Mock document with proper structure matching our data model
        mock_doc = Mock()
        mock_doc.metadata.path_name = "core_profiles/profiles_1d/electrons/temperature"
        mock_doc.metadata.data_type = "float"
        mock_doc.metadata.physics_domain = "core_plasma"
        mock_doc.metadata.units = "eV"
        mock_doc.documentation = "Electron temperature profile"
        mock_doc.raw_data = {
            "identifier_schema": {
                "schema_path": "core_profiles/profiles_1d/species",
                "options": [
                    {"name": "electron", "index": 0},
                    {"name": "deuterium", "index": 1},
                    {"name": "tritium", "index": 2},
                ],
            }
        }

        mock_store.get_documents_by_ids.return_value = [mock_doc]

        # Identifier branching summary
        mock_store.get_identifier_branching_summary.return_value = {
            "total_schemas": 15,
            "total_identifier_paths": 342,
            "total_enumeration_options": 1247,
        }

        # Identifier schemas
        mock_store.get_identifier_schemas.return_value = [mock_doc]

        return mock_store

    @pytest.fixture
    def mock_search_composer(self):
        """Create mock search composer with proper search results."""
        mock_composer = Mock()

        # Search results matching SearchHit structure
        mock_composer.search_with_params.return_value = {
            "results": [
                {
                    "path": "equilibrium/time_slice/boundary/outline/r",
                    "documentation": "R coordinates of plasma boundary",
                    "physics_domain": "equilibrium",
                    "data_type": "float",
                    "units": "m",
                    "relevance_score": 0.9,
                    "ids_name": "equilibrium",
                    "identifier": {"time_slice": 0},
                },
                {
                    "path": "core_profiles/profiles_1d/electrons/density",
                    "documentation": "Electron density profile",
                    "physics_domain": "core_plasma",
                    "data_type": "float",
                    "units": "m^-3",
                    "relevance_score": 0.85,
                    "ids_name": "core_profiles",
                    "identifier": {"species": 0},
                },
            ]
        }

        return mock_composer

    @pytest.mark.asyncio
    async def test_analysis_tool_structure_analysis(
        self, mock_document_store, mock_context
    ):
        """Test that analysis tool returns proper StructureResult format."""
        tool = AnalysisTool(document_store=mock_document_store)

        result = await tool.analyze_ids_structure("core_profiles", ctx=mock_context)

        # Should return dict matching StructureResult schema
        assert isinstance(result, dict)
        assert "ids_name" in result
        assert "description" in result
        assert "structure" in result
        assert "sample_paths" in result
        assert "max_depth" in result
        assert "physics_domains" in result

        # Values should match our mock data
        assert result["ids_name"] == "core_profiles"
        assert isinstance(result["structure"], dict)
        assert isinstance(result["sample_paths"], list)
        assert isinstance(result["max_depth"], int)
        assert isinstance(result["physics_domains"], list)

    @pytest.mark.asyncio
    async def test_relationships_tool_exploration(
        self, mock_document_store, mock_search_composer, mock_context
    ):
        """Test relationships tool returns proper RelationshipResult format."""
        tool = RelationshipsTool(
            document_store=mock_document_store, search_composer=mock_search_composer
        )

        result = await tool.explore_relationships(
            "core_profiles/profiles_1d/electrons/temperature",
            relationship_type="physics",
            max_depth=2,
            ctx=mock_context,
        )

        # Should return dict matching RelationshipResult schema
        assert isinstance(result, dict)
        assert "path" in result
        assert "relationship_type" in result
        assert "max_depth" in result
        assert "connections" in result
        assert "paths" in result
        assert "count" in result

        # Values should match our input and mock data
        assert result["path"] == "core_profiles/profiles_1d/electrons/temperature"
        assert result["max_depth"] == 2
        assert isinstance(result["connections"], dict)

    @pytest.mark.asyncio
    async def test_identifiers_tool_exploration(
        self, mock_document_store, mock_context
    ):
        """Test identifiers tool returns proper IdentifierResult format."""
        tool = IdentifiersTool(document_store=mock_document_store)

        result = await tool.explore_identifiers(scope="all", ctx=mock_context)

        # Should return dict matching IdentifierResult schema
        assert isinstance(result, dict)
        assert "scope" in result
        assert "schemas" in result
        assert "paths" in result
        assert "analytics" in result

        # Values should match our mock data
        assert result["scope"] == "all"
        assert isinstance(result["schemas"], list)
        assert isinstance(result["paths"], list)
        assert isinstance(result["analytics"], dict)

    @pytest.mark.asyncio
    async def test_export_tool_ids_export(
        self, mock_document_store, mock_search_composer, mock_context
    ):
        """Test export tool returns proper IDSExport format."""
        tool = ExportTool(
            document_store=mock_document_store, search_composer=mock_search_composer
        )

        result = await tool.export_ids(
            ["core_profiles", "equilibrium"],
            include_relationships=True,
            include_physics_context=True,
            ctx=mock_context,
        )

        # Should return dict matching IDSExport schema
        assert isinstance(result, dict)
        assert "ids_names" in result
        assert "include_relationships" in result
        assert "include_physics_context" in result
        assert "data" in result
        assert "metadata" in result

        # Values should match our input
        assert result["ids_names"] == ["core_profiles", "equilibrium"]
        assert result["include_relationships"] is True
        assert result["include_physics_context"] is True
        assert isinstance(result["data"], dict)
        assert isinstance(result["metadata"], dict)

    @pytest.mark.asyncio
    async def test_export_tool_domain_export(
        self, mock_document_store, mock_search_composer, mock_context
    ):
        """Test export tool physics domain export."""
        tool = ExportTool(
            document_store=mock_document_store, search_composer=mock_search_composer
        )

        result = await tool.export_physics_domain(
            "core_plasma", include_cross_domain=True, max_paths=20, ctx=mock_context
        )

        # Should return dict matching DomainExport schema
        assert isinstance(result, dict)
        assert "domain" in result
        assert "include_cross_domain" in result
        assert "max_paths" in result
        assert "data" in result
        assert "metadata" in result

        # Values should match our input
        assert result["domain"] == "core_plasma"
        assert result["include_cross_domain"] is True
        assert result["max_paths"] == 20

    @pytest.mark.asyncio
    async def test_overview_tool_response(self, mock_context):
        """Test overview tool returns proper OverviewResult format."""
        tool = OverviewTool()

        result = await tool.get_overview(ctx=mock_context)

        # Should return dict matching OverviewResult schema
        assert isinstance(result, dict)
        assert "content" in result
        assert "available_ids" in result
        assert "physics_domains" in result

        # Content should be informative
        assert isinstance(result["content"], str)
        assert len(result["content"]) > 100  # Should have substantial content
        assert isinstance(result["available_ids"], list)
        assert isinstance(result["physics_domains"], list)

    @pytest.mark.asyncio
    async def test_error_handling_with_invalid_ids(
        self, mock_document_store, mock_context
    ):
        """Test error handling when invalid IDS is provided."""
        # Configure mock to return empty list for invalid IDS
        mock_document_store.get_available_ids.return_value = [
            "core_profiles",
            "equilibrium",
        ]

        tool = AnalysisTool(document_store=mock_document_store)

        result = await tool.analyze_ids_structure("invalid_ids", ctx=mock_context)

        # Should return error response format
        assert isinstance(result, dict)
        assert "error" in result
        assert "suggestions" in result
        assert "available_ids" in result

        # Error message should mention the invalid IDS
        assert "invalid_ids" in result["error"]
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_workflow_data_consistency(
        self, mock_document_store, mock_search_composer, mock_context
    ):
        """Test data consistency across tool workflow."""
        # Create tools
        analysis_tool = AnalysisTool(document_store=mock_document_store)
        relationships_tool = RelationshipsTool(
            document_store=mock_document_store, search_composer=mock_search_composer
        )
        export_tool = ExportTool(
            document_store=mock_document_store, search_composer=mock_search_composer
        )

        # Execute workflow: analyze → explore relationships → export
        analysis_result = await analysis_tool.analyze_ids_structure("core_profiles")

        # Use path from analysis for relationship exploration
        test_path = "core_profiles/profiles_1d/electrons/temperature"
        relationship_result = await relationships_tool.explore_relationships(test_path)

        # Export the analyzed IDS
        export_result = await export_tool.export_ids(["core_profiles"])

        # Verify data consistency
        assert analysis_result["ids_name"] == "core_profiles"
        assert relationship_result["path"] == test_path
        assert "core_profiles" in export_result["ids_names"]

        # All tools should return valid structure
        for result in [analysis_result, relationship_result, export_result]:
            assert isinstance(result, dict)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_concurrent_tool_operations(
        self, mock_document_store, mock_search_composer, mock_context
    ):
        """Test tools work correctly under concurrent access."""
        # Create tools
        analysis_tool = AnalysisTool(document_store=mock_document_store)
        relationships_tool = RelationshipsTool(
            document_store=mock_document_store, search_composer=mock_search_composer
        )
        identifiers_tool = IdentifiersTool(document_store=mock_document_store)

        # Run operations concurrently
        tasks = [
            analysis_tool.analyze_ids_structure("core_profiles"),
            relationships_tool.explore_relationships("core_profiles/profiles_1d"),
            identifiers_tool.explore_identifiers(scope="all"),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All operations should succeed
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Task {i} failed: {result}"
            assert isinstance(result, dict)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_invalid_enum_handling(
        self, mock_document_store, mock_search_composer, mock_context
    ):
        """Test handling of invalid enum values."""
        relationships_tool = RelationshipsTool(
            document_store=mock_document_store, search_composer=mock_search_composer
        )

        # Test with invalid relationship_type
        result = await relationships_tool.explore_relationships(
            "core_profiles/temperature",
            relationship_type="invalid_type",
            ctx=mock_context,
        )

        # Should handle gracefully (either through validation or error handling)
        assert isinstance(result, dict)
        # Should either have valid data or error information
        assert len(result) > 0
