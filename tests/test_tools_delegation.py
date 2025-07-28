"""
Test suite for extracted tool implementations.

Tests the extracted tools with decorator composition:
- Explain tool
- Overview tool
- Analysis tool (stub)
- Relationships tool (stub)
- Identifiers tool (stub)
- Export tool (stub)
"""

import pytest
from unittest.mock import AsyncMock, patch

from imas_mcp.tools import Tools
from imas_mcp.tools.explain_tool import ExplainTool
from imas_mcp.tools.overview_tool import OverviewTool


class TestExtractedToolImplementations:
    """Test cases for extracted tool implementations."""

    @pytest.fixture
    def tools_instance(self):
        """Create Tools instance for testing."""
        return Tools()

    @pytest.fixture
    def explain_tool(self):
        """Create explain tool for testing."""
        return ExplainTool()

    @pytest.fixture
    def overview_tool(self):
        """Create overview tool for testing."""
        return OverviewTool()

    @pytest.mark.asyncio
    async def test_tools_delegation_explain(self, tools_instance):
        """Test Tools class delegation to explain tool."""
        with patch.object(
            tools_instance.explain_tool, "explain_concept"
        ) as mock_explain:
            mock_explain.return_value = {
                "concept": "test",
                "explanation": "test explanation",
            }

            result = await tools_instance.explain_concept("test_concept")

            mock_explain.assert_called_once_with("test_concept")
            assert result["concept"] == "test"

    @pytest.mark.asyncio
    async def test_tools_delegation_overview(self, tools_instance):
        """Test Tools class delegation to overview tool."""
        with patch.object(
            tools_instance.overview_tool, "get_overview"
        ) as mock_overview:
            mock_overview.return_value = {
                "content": "test overview",
                "available_ids": [],
            }

            result = await tools_instance.get_overview("test question")

            mock_overview.assert_called_once_with("test question")
            assert result["content"] == "test overview"

    @pytest.mark.asyncio
    async def test_explain_tool_basic_functionality(self, explain_tool):
        """Test explain tool basic functionality."""
        with patch.object(explain_tool, "_search_service") as mock_service:
            mock_service.search = AsyncMock(return_value=[])

            result = await explain_tool.explain_concept(concept="temperature")

            assert isinstance(result, dict)
            assert "concept" in result
            assert result["concept"] == "temperature"

    @pytest.mark.asyncio
    async def test_overview_tool_basic_functionality(self, overview_tool):
        """Test overview tool basic functionality."""
        with patch.object(overview_tool, "_search_service") as mock_service:
            mock_service.search = AsyncMock(return_value=[])

            result = await overview_tool.get_overview()

            assert isinstance(result, dict)
            assert "content" in result
            assert "available_ids" in result

    @pytest.mark.asyncio
    async def test_overview_tool_with_query(self, overview_tool):
        """Test overview tool with specific query."""
        with patch.object(overview_tool, "_search_service") as mock_service:
            mock_service.search = AsyncMock(return_value=[])

            result = await overview_tool.get_overview(
                query="What is plasma temperature?"
            )

            assert isinstance(result, dict)
            assert "query" in result
            assert result["query"] == "What is plasma temperature?"

    def test_tool_names(self, explain_tool, overview_tool):
        """Test tool name retrieval."""
        assert explain_tool.get_tool_name() == "explain_concept"
        assert overview_tool.get_tool_name() == "get_overview"

    def test_tools_initialization(self, tools_instance):
        """Test Tools class initialization with all tool instances."""
        assert hasattr(tools_instance, "explain_tool")
        assert hasattr(tools_instance, "overview_tool")
        assert hasattr(tools_instance, "search_tool")
        assert hasattr(tools_instance, "analysis_tool")
        assert hasattr(tools_instance, "relationships_tool")
        assert hasattr(tools_instance, "identifiers_tool")
        assert hasattr(tools_instance, "export_tool")

    @pytest.mark.asyncio
    async def test_decorator_composition_explain(self, explain_tool):
        """Test that explain tool has decorators properly applied."""
        # Verify the method has the expected decorator attributes
        method = explain_tool.explain_concept
        assert hasattr(method, "_mcp_tool")
        assert hasattr(method, "_mcp_description")
        assert method._mcp_tool is True
        assert "Explain IMAS concepts" in method._mcp_description

    @pytest.mark.asyncio
    async def test_decorator_composition_overview(self, overview_tool):
        """Test that overview tool has decorators properly applied."""
        # Verify the method has the expected decorator attributes
        method = overview_tool.get_overview
        assert hasattr(method, "_mcp_tool")
        assert hasattr(method, "_mcp_description")
        assert method._mcp_tool is True
        assert "Get IMAS overview" in method._mcp_description
