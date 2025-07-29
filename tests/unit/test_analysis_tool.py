"""
Tests for AnalysisTool implementation.

This module tests the analyze_ids_structure tool with all decorators applied.
"""

import pytest
from unittest.mock import Mock
from imas_mcp.tools.analysis_tool import AnalysisTool
from imas_mcp.search.document_store import DocumentStore


class TestAnalysisTool:
    """Test cases for AnalysisTool."""

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
        mock_doc.documentation = "Temperature measurement"
        mock_doc.raw_data = {
            "identifier_schema": {
                "schema_path": "core_profiles/schema",
                "options": [
                    {"name": "electron", "index": 0},
                    {"name": "ion", "index": 1},
                ],
            }
        }

        mock_store.get_documents_by_ids.return_value = [mock_doc]
        return mock_store

    @pytest.fixture
    def analysis_tool(self, mock_document_store):
        """Create AnalysisTool instance with mocked dependencies."""
        return AnalysisTool(document_store=mock_document_store)

    def test_get_tool_name(self, analysis_tool):
        """Test that tool returns correct name."""
        assert analysis_tool.get_tool_name() == "analyze_ids_structure"

    @pytest.mark.asyncio
    async def test_analyze_ids_structure_valid_ids(self, analysis_tool):
        """Test analysis with valid IDS name."""
        result = await analysis_tool.analyze_ids_structure("core_profiles")

        assert isinstance(result, dict)
        assert result["ids_name"] == "core_profiles"
        assert "description" in result
        assert "structure" in result
        assert "sample_paths" in result
        assert "max_depth" in result
        assert "physics_domains" in result
        assert "identifier_analysis" in result

        # Check structure data
        structure = result["structure"]
        assert "document_count" in structure
        assert "max_depth" in structure
        assert structure["document_count"] == 1

        # Check identifier analysis
        identifier_analysis = result["identifier_analysis"]
        assert "total_identifier_nodes" in identifier_analysis
        assert "branching_paths" in identifier_analysis
        assert "coverage" in identifier_analysis

    @pytest.mark.asyncio
    async def test_analyze_ids_structure_invalid_ids(self, analysis_tool):
        """Test analysis with invalid IDS name."""
        result = await analysis_tool.analyze_ids_structure("invalid_ids")

        assert isinstance(result, dict)
        assert "error" in result
        assert "invalid_ids" in result["error"]
        assert "available_ids" in result
        assert "suggestions" in result

    @pytest.mark.asyncio
    async def test_analyze_ids_structure_empty_ids(self, analysis_tool):
        """Test analysis with empty IDS documents."""
        # Mock empty document list
        analysis_tool.document_store.get_documents_by_ids.return_value = []

        result = await analysis_tool.analyze_ids_structure("core_profiles")

        assert isinstance(result, dict)
        assert result["ids_name"] == "core_profiles"
        # With empty documents, structure should reflect that
        assert result["structure"]["document_count"] == 0
        assert result["max_depth"] == 0

    @pytest.mark.asyncio
    async def test_analyze_ids_structure_error_handling(self, analysis_tool):
        """Test error handling in analyze_ids_structure."""
        # Create a fresh tool instance with an error-prone document store
        error_mock_store = Mock()
        error_mock_store.get_available_ids.side_effect = Exception("Database error")
        error_tool = AnalysisTool(document_store=error_mock_store)

        result = await error_tool.analyze_ids_structure("core_profiles")

        assert isinstance(result, dict)
        # The error handling decorator should catch the exception and return structured error
        assert "error" in result
        assert "suggestions" in result

    def test_build_analysis_sample_prompt(self, analysis_tool):
        """Test sample prompt building."""
        prompt = analysis_tool._build_analysis_sample_prompt("core_profiles")

        assert "core_profiles" in prompt
        assert "IMAS IDS Structure Analysis Request" in prompt
        assert "Architecture Overview" in prompt
        assert "Data Hierarchy" in prompt

    @pytest.mark.asyncio
    async def test_decorator_integration(self, analysis_tool):
        """Test that decorators are properly applied."""
        # The tool should have the _mcp_tool attribute from the decorator
        assert hasattr(analysis_tool.analyze_ids_structure, "_mcp_tool")
        assert analysis_tool.analyze_ids_structure._mcp_tool is True
        assert hasattr(analysis_tool.analyze_ids_structure, "_mcp_description")

        # Test that the method can be called (decorators don't break it)
        result = await analysis_tool.analyze_ids_structure("core_profiles")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_identifier_schema_processing(self, analysis_tool):
        """Test identifier schema processing."""
        result = await analysis_tool.analyze_ids_structure("core_profiles")

        identifier_analysis = result["identifier_analysis"]
        assert identifier_analysis["total_identifier_nodes"] == 1

        branching_paths = identifier_analysis["branching_paths"]
        assert len(branching_paths) == 1

        branch = branching_paths[0]
        assert branch["path"] == "core_profiles/temperature"
        assert branch["option_count"] == 2
        assert branch["branching_significance"] == "MODERATE"
        assert len(branch["sample_options"]) == 2

    @pytest.mark.asyncio
    async def test_physics_domains_detection(self, analysis_tool):
        """Test physics domain detection."""
        result = await analysis_tool.analyze_ids_structure("core_profiles")

        physics_domains = result["physics_domains"]
        assert "core_profiles" in physics_domains

    @pytest.mark.asyncio
    async def test_path_analysis(self, analysis_tool):
        """Test path analysis functionality."""
        # Create fresh mock documents
        mock_doc1 = Mock()
        mock_doc1.metadata.path_name = "core_profiles/temperature"
        mock_doc1.metadata.data_type = "float"
        mock_doc1.metadata.physics_domain = "core_plasma"
        mock_doc1.metadata.units = "eV"
        mock_doc1.documentation = "Temperature measurement"
        mock_doc1.raw_data = {}

        mock_doc2 = Mock()
        mock_doc2.metadata.path_name = "core_profiles/density/electron"
        mock_doc2.metadata.data_type = "float"
        mock_doc2.metadata.physics_domain = "core_plasma"
        mock_doc2.metadata.units = "m^-3"
        mock_doc2.documentation = "Electron density"
        mock_doc2.raw_data = {}

        # Set up the mock to return both documents
        analysis_tool.document_store.get_documents_by_ids.return_value = [
            mock_doc1,
            mock_doc2,
        ]

        result = await analysis_tool.analyze_ids_structure("core_profiles")

        assert result["structure"]["document_count"] == 2
        assert len(result["sample_paths"]) >= 2
        assert len(result["sample_paths"]) == 2
        assert result["max_depth"] > 1  # Should detect nested paths
