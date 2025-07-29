"""
Tests for BaseTool abstract base class.

Tests the common functionality shared across all IMAS MCP tools.
"""

import pytest

from imas_mcp.tools.base import BaseTool


class ConcreteTool(BaseTool):
    """Concrete implementation for testing BaseTool."""

    def get_tool_name(self) -> str:
        return "test_tool"


class TestBaseTool:
    """Test BaseTool abstract base class functionality."""

    def test_base_tool_is_abstract(self):
        """Test that BaseTool cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTool()  # type: ignore

    def test_concrete_tool_instantiation(self):
        """Test that concrete tool implementations can be instantiated."""
        tool = ConcreteTool()
        assert tool is not None
        assert isinstance(tool, BaseTool)

    def test_get_tool_name_abstract_method(self):
        """Test that get_tool_name is properly implemented in concrete class."""
        tool = ConcreteTool()
        assert tool.get_tool_name() == "test_tool"

    def test_create_error_response_basic(self):
        """Test basic error response creation."""
        tool = ConcreteTool()

        error_response = tool._create_error_response("Test error message")

        assert error_response["error"] == "Test error message"
        assert error_response["query"] == ""
        assert error_response["tool"] == "test_tool"
        assert error_response["status"] == "error"

    def test_create_error_response_with_query(self):
        """Test error response creation with query parameter."""
        tool = ConcreteTool()

        error_response = tool._create_error_response("Test error", "test query")

        assert error_response["error"] == "Test error"
        assert error_response["query"] == "test query"
        assert error_response["tool"] == "test_tool"
        assert error_response["status"] == "error"

    def test_logger_available(self):
        """Test that logger is available on tool instances."""
        tool = ConcreteTool()
        assert hasattr(tool, "logger")
        assert tool.logger is not None

    def test_error_response_structure(self):
        """Test that error response has expected structure."""
        tool = ConcreteTool()

        error_response = tool._create_error_response("test")

        # Check all required keys are present
        required_keys = {"error", "query", "tool", "status"}
        assert set(error_response.keys()) == required_keys

        # Check value types
        assert isinstance(error_response["error"], str)
        assert isinstance(error_response["query"], str)
        assert isinstance(error_response["tool"], str)
        assert isinstance(error_response["status"], str)
