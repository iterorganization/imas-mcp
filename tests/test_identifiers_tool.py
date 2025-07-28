"""
Tests for IdentifiersTool implementation.

This module tests the explore_identifiers tool with all decorators applied.
"""

import pytest
from unittest.mock import Mock
from imas_mcp.tools.identifiers_tool import IdentifiersTool
from imas_mcp.search.document_store import DocumentStore


class TestIdentifiersTool:
    """Test cases for IdentifiersTool."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store."""
        mock_store = Mock(spec=DocumentStore)

        # Mock identifier branching summary
        mock_store.get_identifier_branching_summary.return_value = {
            "total_schemas": 10,
            "total_identifier_paths": 50,
            "total_enumeration_options": 200,
        }

        # Mock identifier schemas
        mock_schema_doc = Mock()
        mock_schema_doc.metadata.path_name = "core_profiles/species"
        mock_schema_doc.raw_data = {
            "schema_path": "core_profiles/species/schema",
            "total_options": 3,
            "options": [
                {"name": "electron", "index": 0, "description": "Electron species"},
                {"name": "deuterium", "index": 1, "description": "Deuterium ion"},
                {"name": "tritium", "index": 2, "description": "Tritium ion"},
            ],
        }

        mock_store.get_identifier_schemas.return_value = [mock_schema_doc]
        mock_store.search_identifier_schemas.return_value = [mock_schema_doc]

        # Mock available IDS
        mock_store.get_available_ids.return_value = ["core_profiles", "equilibrium"]

        # Mock documents with identifier schemas
        mock_doc = Mock()
        mock_doc.metadata.path_name = "core_profiles/species/density"
        mock_doc.metadata.documentation = "Species density measurements"
        mock_doc.raw_data = {"identifier_schema": {"type": "species"}}

        mock_store.get_documents_by_ids.return_value = [mock_doc]

        return mock_store

    @pytest.fixture
    def identifiers_tool(self, mock_document_store):
        """Create IdentifiersTool instance with mocked dependencies."""
        return IdentifiersTool(document_store=mock_document_store)

    def test_get_tool_name(self, identifiers_tool):
        """Test that tool returns correct name."""
        assert identifiers_tool.get_tool_name() == "explore_identifiers"

    @pytest.mark.asyncio
    async def test_explore_identifiers_all_scope(self, identifiers_tool):
        """Test identifier exploration with 'all' scope."""
        result = await identifiers_tool.explore_identifiers(scope="all")

        assert isinstance(result, dict)
        assert result["scope"] == "all"
        assert "schemas" in result
        assert "paths" in result
        assert "analytics" in result

        # Check analytics
        analytics = result["analytics"]
        assert analytics["total_schemas"] == 10
        assert analytics["total_paths"] == 50
        assert analytics["enumeration_space"] == 200
        assert "significance" in analytics

    @pytest.mark.asyncio
    async def test_explore_identifiers_schemas_scope(self, identifiers_tool):
        """Test identifier exploration with 'schemas' scope."""
        result = await identifiers_tool.explore_identifiers(scope="schemas")

        assert isinstance(result, dict)
        assert result["scope"] == "schemas"
        assert len(result["schemas"]) > 0

        # Check schema structure
        schema = result["schemas"][0]
        assert "path" in schema
        assert "schema_path" in schema
        assert "option_count" in schema
        assert "branching_significance" in schema
        assert "sample_options" in schema

        # Check options structure
        assert len(schema["sample_options"]) > 0
        option = schema["sample_options"][0]
        assert "name" in option
        assert "index" in option
        assert "description" in option

    @pytest.mark.asyncio
    async def test_explore_identifiers_paths_scope(self, identifiers_tool):
        """Test identifier exploration with 'paths' scope."""
        result = await identifiers_tool.explore_identifiers(scope="paths")

        assert isinstance(result, dict)
        assert result["scope"].value == "PATHS"
        assert len(result["paths"]) > 0

        # Check path structure
        path = result["paths"][0]
        assert "path" in path
        assert "ids_name" in path
        assert "has_identifier" in path
        assert "documentation" in path

    @pytest.mark.asyncio
    async def test_explore_identifiers_with_query(self, identifiers_tool):
        """Test identifier exploration with query."""
        result = await identifiers_tool.explore_identifiers(
            query="species", scope="schemas"
        )

        assert isinstance(result, dict)
        # Should call search_identifier_schemas instead of get_identifier_schemas
        identifiers_tool.document_store.search_identifier_schemas.assert_called_with(
            "species"
        )

    @pytest.mark.asyncio
    async def test_explore_identifiers_error_handling(self, identifiers_tool):
        """Test error handling in identifier exploration."""
        # Make document store raise an exception
        identifiers_tool.document_store.get_identifier_branching_summary.side_effect = (
            Exception("Database error")
        )

        result = await identifiers_tool.explore_identifiers()

        assert isinstance(result, dict)
        assert "error" in result
        assert "suggestions" in result

    @pytest.mark.asyncio
    async def test_explore_identifiers_paths_error(self, identifiers_tool):
        """Test error handling when getting identifier paths fails."""
        # Make get_documents_by_ids raise an exception
        identifiers_tool.document_store.get_documents_by_ids.side_effect = Exception(
            "IDS error"
        )

        result = await identifiers_tool.explore_identifiers(scope="paths")

        # Should handle the error gracefully and continue
        assert isinstance(result, dict)
        assert "paths" in result
        # Paths might be empty due to error, but shouldn't crash

    def test_build_identifiers_sample_prompt(self, identifiers_tool):
        """Test sample prompt building."""
        prompt = identifiers_tool._build_identifiers_sample_prompt(
            query="species", scope="schemas"
        )

        assert "species" in prompt
        assert "schemas" in prompt
        assert "IMAS Identifier Schema Exploration Request" in prompt
        assert "Significance" in prompt
        assert "Key Schemas" in prompt

    @pytest.mark.asyncio
    async def test_decorator_integration(self, identifiers_tool):
        """Test that decorators are properly applied."""
        # The tool should have the _mcp_tool attribute from the decorator
        assert hasattr(identifiers_tool.explore_identifiers, "_mcp_tool")
        assert identifiers_tool.explore_identifiers._mcp_tool is True
        assert hasattr(identifiers_tool.explore_identifiers, "_mcp_description")

        # Test that the method can be called (decorators don't break it)
        result = await identifiers_tool.explore_identifiers()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_branching_significance_classification(self, identifiers_tool):
        """Test branching significance classification."""
        result = await identifiers_tool.explore_identifiers(scope="schemas")

        schema = result["schemas"][0]
        # With 3 options, should be classified as "MODERATE"
        assert schema["branching_significance"] == "MODERATE"

    @pytest.mark.asyncio
    async def test_schema_limits(self, identifiers_tool):
        """Test that schema results are limited appropriately."""
        # Create multiple mock schemas
        mock_schemas = []
        for i in range(15):
            mock_schema = Mock()
            mock_schema.metadata.path_name = f"test_path_{i}"
            mock_schema.raw_data = {
                "schema_path": f"test/path/{i}",
                "total_options": 2,
                "options": [],
            }
            mock_schemas.append(mock_schema)

        identifiers_tool.document_store.get_identifier_schemas.return_value = (
            mock_schemas
        )

        result = await identifiers_tool.explore_identifiers(scope="schemas")

        # Should limit to 10 schemas
        assert len(result["schemas"]) == 10

    @pytest.mark.asyncio
    async def test_path_limits(self, identifiers_tool):
        """Test that path results are limited appropriately."""
        # Create multiple mock documents
        mock_docs = []
        for i in range(25):
            mock_doc = Mock()
            mock_doc.metadata.path_name = f"test_ids/path_{i}"
            mock_doc.metadata.documentation = f"Test documentation {i}"
            mock_doc.raw_data = {"identifier_schema": {"type": "test"}}
            mock_docs.append(mock_doc)

        identifiers_tool.document_store.get_documents_by_ids.return_value = mock_docs

        result = await identifiers_tool.explore_identifiers(scope="paths")

        # Should limit to 20 paths
        assert len(result["paths"]) == 20
