"""
Tests for ExportTool implementation.

This module tests the export_ids and export_physics_domain tools with all decorators applied.
"""

import pytest
from unittest.mock import Mock
from imas_mcp.tools.export_tool import ExportTool
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.search_strategy import SearchComposer


class TestExportTool:
    """Test cases for ExportTool."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store."""
        mock_store = Mock(spec=DocumentStore)
        mock_store.get_available_ids.return_value = [
            "core_profiles",
            "equilibrium",
            "transport",
        ]

        # Mock document data
        mock_doc = Mock()
        mock_doc.metadata.path_name = "core_profiles/temperature"
        mock_doc.metadata.data_type = "float"
        mock_doc.metadata.physics_domain = "core_plasma"
        mock_doc.metadata.units = "eV"
        mock_doc.documentation = "Temperature measurement data"
        mock_doc.raw_data = {
            "identifier_schema": {
                "schema_path": "core_profiles/schema",
                "options": [{"name": "electron", "index": 0}],
            }
        }

        mock_store.get_documents_by_ids.return_value = [mock_doc]
        return mock_store

    @pytest.fixture
    def mock_search_composer(self):
        """Create a mock search composer."""
        mock_composer = Mock(spec=SearchComposer)
        mock_composer.search_with_params.return_value = {
            "results": [
                {
                    "path": "core_profiles/temperature",
                    "documentation": "Core plasma temperature measurements",
                    "physics_domain": "core_plasma",
                    "data_type": "float",
                    "units": "eV",
                    "relevance_score": 0.9,
                },
                {
                    "path": "equilibrium/magnetic_field",
                    "documentation": "Magnetic field equilibrium data",
                    "physics_domain": "equilibrium",
                    "data_type": "float",
                    "units": "T",
                    "relevance_score": 0.8,
                },
            ]
        }
        return mock_composer

    @pytest.fixture
    def export_tool(self, mock_document_store, mock_search_composer):
        """Create ExportTool instance with mocked dependencies."""
        return ExportTool(
            document_store=mock_document_store, search_composer=mock_search_composer
        )

    def test_get_tool_name(self, export_tool):
        """Test that tool returns correct name."""
        assert export_tool.get_tool_name() == "export_tools"

    @pytest.mark.asyncio
    async def test_export_ids_valid_ids(self, export_tool):
        """Test bulk export with valid IDS names."""
        result = await export_tool.export_ids(
            ids_list=["core_profiles", "equilibrium"], output_format="structured"
        )

        assert isinstance(result, dict)
        assert "ids_names" in result
        assert "data" in result
        assert "metadata" in result

        export_data = result["data"]
        assert "requested_ids" in export_data
        assert "valid_ids" in export_data
        assert "invalid_ids" in export_data
        assert "ids_data" in export_data
        assert "export_summary" in export_data

        # Check that valid IDS were processed
        assert export_data["valid_ids"] == ["core_profiles", "equilibrium"]
        assert len(export_data["invalid_ids"]) == 0

    @pytest.mark.asyncio
    async def test_export_ids_invalid_ids(self, export_tool):
        """Test bulk export with invalid IDS names."""
        result = await export_tool.export_ids(
            ids_list=["invalid_ids", "another_invalid"], output_format="structured"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "No valid IDS names provided" in result["error"]
        assert "invalid_ids" in result
        assert "available_ids" in result

    @pytest.mark.asyncio
    async def test_export_ids_mixed_validity(self, export_tool):
        """Test bulk export with mix of valid and invalid IDS names."""
        result = await export_tool.export_ids(
            ids_list=["core_profiles", "invalid_ids"], output_format="structured"
        )

        export_data = result["data"]
        assert export_data["valid_ids"] == ["core_profiles"]
        assert export_data["invalid_ids"] == ["invalid_ids"]
        assert export_data["export_summary"]["export_completeness"] == "partial"

    @pytest.mark.asyncio
    async def test_export_ids_different_formats(self, export_tool):
        """Test export with different output formats."""
        formats = ["raw", "structured", "enhanced"]

        for fmt in formats:
            result = await export_tool.export_ids(
                ids_list=["core_profiles"], output_format=fmt
            )

            assert isinstance(result, dict)
            export_data = result["data"]
            assert export_data["export_format"] == fmt

    @pytest.mark.asyncio
    async def test_export_ids_invalid_format(self, export_tool):
        """Test export with invalid output format."""
        result = await export_tool.export_ids(
            ids_list=["core_profiles"], output_format="invalid_format"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "Invalid format" in result["error"]

    @pytest.mark.asyncio
    async def test_export_ids_empty_list(self, export_tool):
        """Test export with empty IDS list."""
        result = await export_tool.export_ids(ids_list=[])

        assert isinstance(result, dict)
        assert "error" in result
        assert "No IDS specified" in result["error"]

    @pytest.mark.asyncio
    async def test_export_ids_with_relationships(self, export_tool):
        """Test export with relationship analysis enabled."""
        result = await export_tool.export_ids(
            ids_list=["core_profiles", "equilibrium"], include_relationships=True
        )

        export_data = result["data"]
        assert "cross_relationships" in export_data
        # Should have relationship analysis for the pair
        assert len(export_data["cross_relationships"]) >= 0

    @pytest.mark.asyncio
    async def test_export_ids_error_handling(self, export_tool):
        """Test error handling in export_ids."""
        # Make document store raise an exception
        export_tool.document_store.get_available_ids.side_effect = Exception(
            "Database error"
        )

        result = await export_tool.export_ids(ids_list=["core_profiles"])

        assert isinstance(result, dict)
        assert "error" in result
        assert "suggestions" in result

    @pytest.mark.asyncio
    async def test_export_physics_domain_valid(self, export_tool):
        """Test domain export with valid domain."""
        result = await export_tool.export_physics_domain(
            domain="core_plasma", analysis_depth="focused"
        )

        assert isinstance(result, dict)
        assert result["domain"] == "core_plasma"
        assert "domain_info" in result
        assert "metadata" in result

        domain_info = result["domain_info"]
        assert "analysis_depth" in domain_info
        assert "paths" in domain_info
        assert "related_ids" in domain_info

    @pytest.mark.asyncio
    async def test_export_physics_domain_empty_domain(self, export_tool):
        """Test domain export with empty domain."""
        result = await export_tool.export_physics_domain(domain="")

        assert isinstance(result, dict)
        assert "error" in result
        assert "No domain specified" in result["error"]

    @pytest.mark.asyncio
    async def test_export_physics_domain_no_results(self, export_tool):
        """Test domain export when no data found."""
        # Make search return empty results
        export_tool.search_composer.search_with_params.return_value = {"results": []}

        result = await export_tool.export_physics_domain(domain="nonexistent_domain")

        assert isinstance(result, dict)
        assert "error" in result
        assert "No data found" in result["error"]

    @pytest.mark.asyncio
    async def test_export_physics_domain_max_paths_limit(self, export_tool):
        """Test that max_paths is properly limited."""
        result = await export_tool.export_physics_domain(
            domain="core_plasma",
            max_paths=100,  # Should be limited to 50
        )

        assert result["max_paths"] == 50

    @pytest.mark.asyncio
    async def test_export_physics_domain_error_handling(self, export_tool):
        """Test error handling in export_physics_domain."""
        # Make search composer raise an exception
        export_tool.search_composer.search_with_params.side_effect = Exception(
            "Search error"
        )

        result = await export_tool.export_physics_domain(domain="core_plasma")

        assert isinstance(result, dict)
        assert "error" in result
        assert "suggestions" in result

    def test_extract_identifier_info(self, export_tool):
        """Test identifier information extraction."""
        mock_doc = Mock()
        mock_doc.raw_data = {
            "identifier_schema": {
                "schema_path": "test/schema",
                "options": [
                    {"name": "option1"},
                    {"name": "option2"},
                    {"name": "option3"},
                ],
            }
        }

        info = export_tool._extract_identifier_info(mock_doc)

        assert info["schema_path"] == "test/schema"
        assert info["options_count"] == 3
        assert len(info["sample_options"]) == 3

    def test_build_export_sample_prompt(self, export_tool):
        """Test export sample prompt building."""
        prompt = export_tool._build_export_sample_prompt(
            ["core_profiles", "equilibrium"], "structured"
        )

        assert "core_profiles" in prompt
        assert "equilibrium" in prompt
        assert "structured" in prompt
        assert "IMAS Bulk Export Analysis Request" in prompt

    def test_build_domain_export_sample_prompt(self, export_tool):
        """Test domain export sample prompt building."""
        prompt = export_tool._build_domain_export_sample_prompt(
            "core_plasma", "comprehensive"
        )

        assert "core_plasma" in prompt
        assert "comprehensive" in prompt
        assert "IMAS Physics Domain Export Request" in prompt

    @pytest.mark.asyncio
    async def test_decorator_integration(self, export_tool):
        """Test that decorators are properly applied."""
        # The tools should have the _mcp_tool attribute from the decorator
        assert hasattr(export_tool.export_ids, "_mcp_tool")
        assert export_tool.export_ids._mcp_tool is True
        assert hasattr(export_tool.export_physics_domain, "_mcp_tool")
        assert export_tool.export_physics_domain._mcp_tool is True

        # Test that the methods can be called (decorators don't break them)
        result1 = await export_tool.export_ids(ids_list=["core_profiles"])
        assert isinstance(result1, dict)

        result2 = await export_tool.export_physics_domain(domain="core_plasma")
        assert isinstance(result2, dict)
