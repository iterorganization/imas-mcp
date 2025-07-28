"""
Integration tests verifying that all tools work with positional arguments.

These tests ensure LLMs can call our tools with simple positional arguments,
which is the standard way LLMs interact with tools.
"""

import pytest
import asyncio
from unittest.mock import Mock

from imas_mcp.tools.overview_tool import OverviewTool
from imas_mcp.tools.analysis_tool import AnalysisTool
from imas_mcp.tools.relationships_tool import RelationshipsTool
from imas_mcp.tools.identifiers_tool import IdentifiersTool
from imas_mcp.tools.export_tool import ExportTool


class TestToolsPositionalArguments:
    """Test that tools accept positional arguments correctly."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store with basic responses."""
        mock_store = Mock()
        mock_store.get_available_ids.return_value = ["core_profiles", "equilibrium"]

        # Mock document with basic structure
        mock_doc = Mock()
        mock_doc.metadata.path_name = "core_profiles/temperature"
        mock_doc.metadata.data_type = "float"
        mock_doc.metadata.physics_domain = "core_plasma"
        mock_doc.metadata.units = "eV"
        mock_doc.documentation = "Temperature data"
        mock_doc.raw_data = {}

        mock_store.get_documents_by_ids.return_value = [mock_doc]
        mock_store.get_identifier_branching_summary.return_value = {
            "total_schemas": 1,
            "total_identifier_paths": 5,
            "total_enumeration_options": 10,
        }
        mock_store.get_identifier_schemas.return_value = [mock_doc]

        return mock_store

    @pytest.fixture
    def mock_search_composer(self):
        """Create a mock search composer with basic responses."""
        mock_composer = Mock()
        mock_composer.search_with_params.return_value = {
            "results": [
                {
                    "path": "equilibrium/magnetic_field",
                    "documentation": "Magnetic field data",
                    "physics_domain": "equilibrium",
                    "data_type": "float",
                    "units": "T",
                    "relevance_score": 0.8,
                }
            ]
        }
        return mock_composer

    @pytest.mark.asyncio
    async def test_overview_tool_no_arguments(self):
        """Test overview tool with no arguments (as LLMs typically call it)."""
        tool = OverviewTool()

        # LLM call: get_overview()
        result = await tool.get_overview()

        assert isinstance(result, dict)
        assert "content" in result
        assert "imas_data_dictionary" in result["content"]

    @pytest.mark.asyncio
    async def test_analysis_tool_single_string_argument(self, mock_document_store):
        """Test analysis tool with single string argument (as LLMs use)."""
        tool = AnalysisTool(document_store=mock_document_store)

        # LLM call: analyze_ids_structure("core_profiles")
        result = await tool.analyze_ids_structure("core_profiles")

        assert isinstance(result, dict)
        assert "ids_name" in result
        assert result["ids_name"] == "core_profiles"

    @pytest.mark.asyncio
    async def test_relationships_tool_single_string_argument(
        self, mock_document_store, mock_search_composer
    ):
        """Test relationships tool with single string argument."""
        tool = RelationshipsTool(
            document_store=mock_document_store, search_composer=mock_search_composer
        )

        # LLM call: explore_relationships("core_profiles")
        result = await tool.explore_relationships("core_profiles")

        assert isinstance(result, dict)
        assert "path" in result
        assert result["path"] == "core_profiles"

    @pytest.mark.asyncio
    async def test_relationships_tool_with_optional_argument(
        self, mock_document_store, mock_search_composer
    ):
        """Test relationships tool with optional second argument."""
        tool = RelationshipsTool(
            document_store=mock_document_store, search_composer=mock_search_composer
        )

        # LLM call: explore_relationships("core_profiles", "functional")
        result = await tool.explore_relationships("core_profiles", "functional")

        assert isinstance(result, dict)
        assert "path" in result
        assert result["path"] == "core_profiles"

    @pytest.mark.asyncio
    async def test_identifiers_tool_no_arguments(self, mock_document_store):
        """Test identifiers tool with no arguments."""
        tool = IdentifiersTool(document_store=mock_document_store)

        # LLM call: explore_identifiers()
        result = await tool.explore_identifiers()

        assert isinstance(result, dict)
        assert "identifier_analysis" in result

    @pytest.mark.asyncio
    async def test_identifiers_tool_with_scope_argument(self, mock_document_store):
        """Test identifiers tool with scope argument."""
        tool = IdentifiersTool(document_store=mock_document_store)

        # LLM call: explore_identifiers("all")
        result = await tool.explore_identifiers("all")

        assert isinstance(result, dict)
        assert "identifier_analysis" in result

    @pytest.mark.asyncio
    async def test_export_tool_list_argument(
        self, mock_document_store, mock_search_composer
    ):
        """Test export tool with list argument."""
        tool = ExportTool(
            document_store=mock_document_store, search_composer=mock_search_composer
        )

        # LLM call: export_ids(["core_profiles"])
        result = await tool.export_ids(["core_profiles"])

        assert isinstance(result, dict)
        assert "ids_names" in result
        assert result["ids_names"] == ["core_profiles"]

    @pytest.mark.asyncio
    async def test_export_tool_domain_string_argument(
        self, mock_document_store, mock_search_composer
    ):
        """Test export tool domain with string argument."""
        tool = ExportTool(
            document_store=mock_document_store, search_composer=mock_search_composer
        )

        # LLM call: export_physics_domain("core_plasma")
        result = await tool.export_physics_domain("core_plasma")

        assert isinstance(result, dict)
        assert "physics_domains" in result
        assert "core_plasma" in result["physics_domains"]

    @pytest.mark.asyncio
    async def test_all_tools_handle_invalid_input_gracefully(
        self, mock_document_store, mock_search_composer
    ):
        """Test that all tools handle invalid inputs gracefully."""
        tools_and_calls = [
            (
                AnalysisTool(document_store=mock_document_store),
                "analyze_ids_structure",
                [""],
            ),
            (
                RelationshipsTool(
                    document_store=mock_document_store,
                    search_composer=mock_search_composer,
                ),
                "explore_relationships",
                [""],
            ),
            (
                IdentifiersTool(document_store=mock_document_store),
                "explore_identifiers",
                ["invalid_scope"],
            ),
            (
                ExportTool(
                    document_store=mock_document_store,
                    search_composer=mock_search_composer,
                ),
                "export_ids",
                [[]],
            ),
            (
                ExportTool(
                    document_store=mock_document_store,
                    search_composer=mock_search_composer,
                ),
                "export_physics_domain",
                [""],
            ),
        ]

        for tool, method_name, args in tools_and_calls:
            method = getattr(tool, method_name)
            result = await method(*args)

            # Should return dict with error handling
            assert isinstance(result, dict)
            # Should either succeed or provide clear error messages
            if "error" in result:
                assert isinstance(result["error"], str)
                assert len(result["error"]) > 0

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls_with_positional_args(
        self, mock_document_store, mock_search_composer
    ):
        """Test that multiple tools can be called concurrently with positional arguments."""
        # Create tools
        overview_tool = OverviewTool()
        analysis_tool = AnalysisTool(document_store=mock_document_store)
        relationships_tool = RelationshipsTool(
            document_store=mock_document_store, search_composer=mock_search_composer
        )

        # Create concurrent calls with positional arguments
        tasks = [
            asyncio.create_task(overview_tool.get_overview()),
            asyncio.create_task(analysis_tool.analyze_ids_structure("core_profiles")),
            asyncio.create_task(
                relationships_tool.explore_relationships("core_profiles")
            ),
        ]

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Task {i} failed: {result}"
            assert isinstance(result, dict), f"Task {i} should return dict"

    @pytest.mark.asyncio
    async def test_validation_accepts_correct_positional_arguments(
        self, mock_document_store
    ):
        """Test that validation decorator correctly processes positional arguments."""
        tool = AnalysisTool(document_store=mock_document_store)

        # These should all work without validation errors
        valid_calls = [
            ("core_profiles",),
            ("equilibrium",),
            ("transport_model",),
            ("mhd_instabilities",),
        ]

        for args in valid_calls:
            result = await tool.analyze_ids_structure(*args)
            assert isinstance(result, dict)
            # Should not have validation errors
            assert "validation_errors" not in result or not result.get(
                "validation_errors"
            )
