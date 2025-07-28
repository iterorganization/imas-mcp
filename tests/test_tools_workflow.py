"""
Integration tests for IMAS MCP tools workflow.

This module tests the integration and workflow between different tools,
verifying they work together correctly and produce expected outputs.
"""

import pytest
from unittest.mock import Mock

from imas_mcp.tools.overview_tool import OverviewTool
from imas_mcp.tools.analysis_tool import AnalysisTool
from imas_mcp.tools.relationships_tool import RelationshipsTool
from imas_mcp.tools.identifiers_tool import IdentifiersTool
from imas_mcp.tools.export_tool import ExportTool
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.search_strategy import SearchComposer


class TestToolsWorkflow:
    """Test the complete workflow between tools."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a comprehensive mock document store."""
        mock_store = Mock(spec=DocumentStore)

        # Mock available IDS
        mock_store.get_available_ids.return_value = [
            "core_profiles",
            "equilibrium",
            "transport",
            "heating",
            "wall",
        ]

        # Mock document data for core_profiles IDS
        mock_doc = Mock()
        mock_doc.metadata.path_name = "core_profiles/temperature"
        mock_doc.metadata.data_type = "float"
        mock_doc.metadata.physics_domain = "core_plasma"
        mock_doc.metadata.units = "eV"
        mock_doc.documentation = (
            "Core plasma temperature measurements for electrons and ions"
        )
        mock_doc.raw_data = {
            "identifier_schema": {
                "schema_path": "core_profiles/species",
                "options": [
                    {"name": "electron", "index": 0, "description": "Electron species"},
                    {"name": "ion", "index": 1, "description": "Ion species"},
                ],
            }
        }

        mock_doc2 = Mock()
        mock_doc2.metadata.path_name = "core_profiles/density"
        mock_doc2.metadata.data_type = "float"
        mock_doc2.metadata.physics_domain = "core_plasma"
        mock_doc2.metadata.units = "m^-3"
        mock_doc2.documentation = "Particle density profiles"
        mock_doc2.raw_data = {}

        mock_store.get_documents_by_ids.return_value = [mock_doc, mock_doc2]

        # Mock identifier data
        mock_store.get_identifier_branching_summary.return_value = {
            "total_schemas": 5,
            "total_identifier_paths": 25,
            "total_enumeration_options": 100,
        }

        mock_schema_doc = Mock()
        mock_schema_doc.metadata.path_name = "core_profiles/species"
        mock_schema_doc.raw_data = {
            "schema_path": "core_profiles/species/schema",
            "total_options": 2,
            "options": [
                {"name": "electron", "index": 0, "description": "Electron species"},
                {"name": "ion", "index": 1, "description": "Ion species"},
            ],
        }
        mock_store.get_identifier_schemas.return_value = [mock_schema_doc]
        mock_store.search_identifier_schemas.return_value = [mock_schema_doc]

        return mock_store

    @pytest.fixture
    def mock_search_composer(self):
        """Create a mock search composer with realistic results."""
        mock_composer = Mock(spec=SearchComposer)
        mock_composer.search_with_params.return_value = {
            "results": [
                {
                    "path": "equilibrium/magnetic_field",
                    "documentation": "Equilibrium magnetic field data related to core plasma physics",
                    "physics_domain": "equilibrium",
                    "data_type": "float",
                    "units": "T",
                    "relevance_score": 0.85,
                },
                {
                    "path": "transport/heat_flux",
                    "documentation": "Heat transport flux calculations based on temperature gradients",
                    "physics_domain": "transport",
                    "data_type": "float",
                    "units": "MW/m^2",
                    "relevance_score": 0.78,
                },
                {
                    "path": "heating/power_density",
                    "documentation": "Heating power density profile affecting temperature",
                    "physics_domain": "heating",
                    "data_type": "float",
                    "units": "MW/m^3",
                    "relevance_score": 0.72,
                },
            ]
        }
        return mock_composer

    @pytest.fixture
    def tools_suite(self, mock_document_store, mock_search_composer):
        """Create a complete suite of tools for integration testing."""
        return {
            "overview": OverviewTool(),
            "analysis": AnalysisTool(document_store=mock_document_store),
            "relationships": RelationshipsTool(
                document_store=mock_document_store, search_composer=mock_search_composer
            ),
            "identifiers": IdentifiersTool(document_store=mock_document_store),
            "export": ExportTool(
                document_store=mock_document_store, search_composer=mock_search_composer
            ),
        }

    @pytest.mark.asyncio
    async def test_discovery_to_analysis_workflow(self, tools_suite):
        """Test workflow: Overview → Analysis → Detailed exploration."""
        # Step 1: Get overview to discover available IDS
        overview_result = await tools_suite["overview"].get_overview()

        assert isinstance(overview_result, dict)
        assert "available_ids" in overview_result
        assert len(overview_result["available_ids"]) > 0

        # Step 2: Analyze a specific IDS found in overview
        available_ids = overview_result["available_ids"]
        test_ids = available_ids[0]  # Use first available IDS

        analysis_result = await tools_suite["analysis"].analyze_ids_structure(test_ids)

        assert isinstance(analysis_result, dict)
        assert analysis_result["ids_name"] == test_ids
        assert "structure" in analysis_result
        assert "sample_paths" in analysis_result
        assert "identifier_analysis" in analysis_result

        # Verify analysis provides useful data
        structure = analysis_result["structure"]
        assert structure["document_count"] > 0
        assert len(analysis_result["sample_paths"]) > 0

    @pytest.mark.asyncio
    async def test_analysis_to_relationships_workflow(self, tools_suite):
        """Test workflow: Analysis → Relationship exploration."""
        # Step 1: Analyze an IDS to get paths
        analysis_result = await tools_suite["analysis"].analyze_ids_structure(
            "core_profiles"
        )

        assert "sample_paths" in analysis_result
        sample_paths = analysis_result["sample_paths"]
        assert len(sample_paths) > 0

        # Step 2: Explore relationships for one of the discovered paths
        test_path = sample_paths[0]
        relationships_result = await tools_suite["relationships"].explore_relationships(
            test_path
        )

        assert isinstance(relationships_result, dict)
        assert relationships_result["path"] == test_path
        assert "connections" in relationships_result
        assert "paths" in relationships_result
        assert "physics_domains" in relationships_result

        # Verify relationships found meaningful connections
        connections = relationships_result["connections"]
        assert "total_relationships" in connections
        assert "physics_connections" in connections

    @pytest.mark.asyncio
    async def test_identifiers_to_analysis_workflow(self, tools_suite):
        """Test workflow: Identifier exploration → Detailed analysis."""
        # Step 1: Explore identifier schemas to understand branching
        identifiers_result = await tools_suite["identifiers"].explore_identifiers(
            scope="schemas"
        )

        assert isinstance(identifiers_result, dict)
        assert "schemas" in identifiers_result
        assert "analytics" in identifiers_result

        schemas = identifiers_result["schemas"]
        if len(schemas) > 0:
            # Step 2: Analyze the IDS that contains identifier schemas
            schema_path = schemas[0]["path"]
            ids_name = schema_path.split("/")[0] if "/" in schema_path else schema_path

            analysis_result = await tools_suite["analysis"].analyze_ids_structure(
                ids_name
            )

            assert isinstance(analysis_result, dict)
            assert "identifier_analysis" in analysis_result

            # Verify the analysis includes identifier information
            identifier_analysis = analysis_result["identifier_analysis"]
            assert "total_identifier_nodes" in identifier_analysis
            assert "branching_paths" in identifier_analysis

    @pytest.mark.asyncio
    async def test_overview_to_export_workflow(self, tools_suite):
        """Test workflow: Overview → Bulk export."""
        # Step 1: Get overview to find available IDS
        overview_result = await tools_suite["overview"].get_overview()
        available_ids = overview_result["available_ids"]

        # Step 2: Export multiple IDS found in overview
        export_ids = available_ids[:2]  # Export first 2 IDS
        export_result = await tools_suite["export"].export_ids(
            ids_list=export_ids, include_relationships=True, output_format="structured"
        )

        assert isinstance(export_result, dict)
        assert "data" in export_result

        export_data = export_result["data"]
        assert export_data["valid_ids"] == export_ids
        assert "ids_data" in export_data
        assert "export_summary" in export_data

        # Verify export includes data for each IDS
        for ids_name in export_ids:
            assert ids_name in export_data["ids_data"]

    @pytest.mark.asyncio
    async def test_relationships_to_domain_export_workflow(self, tools_suite):
        """Test workflow: Relationship exploration → Domain export."""
        # Step 1: Explore relationships to discover physics domains
        relationships_result = await tools_suite["relationships"].explore_relationships(
            "core_profiles/temperature"
        )

        physics_domains = relationships_result["physics_domains"]

        if len(physics_domains) > 0:
            # Step 2: Export data for one of the discovered domains
            test_domain = physics_domains[0]
            domain_export_result = await tools_suite["export"].export_physics_domain(
                domain=test_domain, analysis_depth="focused"
            )

            assert isinstance(domain_export_result, dict)
            assert domain_export_result["domain"] == test_domain
            assert "domain_info" in domain_export_result

    @pytest.mark.asyncio
    async def test_cross_tool_data_consistency(self, tools_suite):
        """Test that data is consistent across different tools."""
        # Get overview data
        overview_result = await tools_suite["overview"].get_overview()
        overview_ids = set(overview_result["available_ids"])

        # Get analysis data for first IDS
        test_ids = list(overview_ids)[0]
        analysis_result = await tools_suite["analysis"].analyze_ids_structure(test_ids)

        # Get identifier data
        identifiers_result = await tools_suite["identifiers"].explore_identifiers(
            scope="paths"
        )

        # Verify consistency: IDS names should match across tools
        assert test_ids in overview_ids
        assert analysis_result["ids_name"] == test_ids

        # Check that paths in identifier results relate to available IDS
        identifier_paths = identifiers_result["paths"]
        for path_info in identifier_paths:
            path_ids = path_info["ids_name"]
            # Path IDS should be from available IDS (or unknown for mocked data)
            assert path_ids in overview_ids or path_ids == "unknown"

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, tools_suite):
        """Test that tools handle errors gracefully in integrated workflows."""
        # Test with invalid IDS name across multiple tools
        invalid_ids = "nonexistent_ids"

        # Analysis tool should handle invalid IDS
        analysis_result = await tools_suite["analysis"].analyze_ids_structure(
            invalid_ids
        )
        assert "error" in analysis_result
        assert "suggestions" in analysis_result

        # Relationships tool should handle invalid path
        relationships_result = await tools_suite["relationships"].explore_relationships(
            invalid_ids
        )
        assert "error" in relationships_result
        assert "suggestions" in relationships_result

        # Export tool should handle invalid IDS
        export_result = await tools_suite["export"].export_ids([invalid_ids])
        assert "error" in export_result
        assert "suggestions" in export_result

    @pytest.mark.asyncio
    async def test_tool_recommendations_workflow(self, tools_suite):
        """Test that tools can suggest follow-up actions that other tools can fulfill."""
        # Get overview with suggestions
        overview_result = await tools_suite["overview"].get_overview()

        # Check for usage guidance that mentions other tools
        usage_guidance = overview_result.get("usage_guidance", {})
        tools_available = usage_guidance.get("tools_available", [])

        # Verify that suggested tools actually exist in our suite
        suggested_tool_names = {
            "search_imas",
            "explain_concept",
            "analyze_ids_structure",
            "explore_relationships",
            "explore_identifiers",
            "export_ids",
        }

        for tool_entry in tools_available:
            tool_name = tool_entry.split(" - ")[0]
            if tool_name in suggested_tool_names:
                # Verify we have a corresponding tool in our suite
                if tool_name == "analyze_ids_structure":
                    assert "analysis" in tools_suite
                elif tool_name == "explore_relationships":
                    assert "relationships" in tools_suite
                elif tool_name == "explore_identifiers":
                    assert "identifiers" in tools_suite
                elif tool_name == "export_ids":
                    assert "export" in tools_suite

    @pytest.mark.asyncio
    async def test_data_flow_completeness(self, tools_suite):
        """Test complete data flow from discovery to export."""
        # Complete workflow: Overview → Analysis → Relationships → Export

        # 1. Discover available data
        overview = await tools_suite["overview"].get_overview()
        assert len(overview["available_ids"]) > 0

        # 2. Analyze structure of discovered data
        test_ids = overview["available_ids"][0]
        analysis = await tools_suite["analysis"].analyze_ids_structure(test_ids)
        assert analysis["structure"]["document_count"] > 0

        # 3. Explore relationships of analyzed paths
        sample_path = analysis["sample_paths"][0]
        relationships = await tools_suite["relationships"].explore_relationships(
            sample_path
        )
        assert relationships["count"] >= 0  # May be 0 for mocked data

        # 4. Export the complete dataset
        export = await tools_suite["export"].export_ids(
            [test_ids], include_relationships=True, output_format="enhanced"
        )
        export_data = export["data"]
        assert export_data["export_summary"]["export_completeness"] == "complete"
        assert len(export_data["ids_data"]) == 1
        assert test_ids in export_data["ids_data"]
