"""
Tests for tool suggestion functionality.

Tests which tools include tool suggestions and which don't, based on the
@tool_suggestions decorator.
"""

import pytest

from imas_mcp.server import Server
from tests.conftest import STANDARD_TEST_IDS_SET


class TestToolSuggestions:
    """Test that tool suggestion functionality works correctly for tools that have it."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server(ids_set=STANDARD_TEST_IDS_SET)

    @pytest.mark.asyncio
    async def test_tools_with_suggestions(self, server):
        """Test that tools with @tool_suggestions decorator include suggested_tools."""
        # Tools that have @tool_suggestions decorator
        tools_with_suggestions = [
            ("search_imas", {"query": "plasma"}),
            ("explain_concept", {"concept": "plasma"}),
            ("get_overview", {}),
            ("analyze_ids_structure", {"ids_name": "core_profiles"}),
            ("explore_relationships", {"path": "core_profiles"}),
        ]

        for tool_name, kwargs in tools_with_suggestions:
            tool_func = getattr(server.tools, tool_name)
            result = await tool_func(**kwargs)

            # Tools with @tool_suggestions decorator should have suggested_tools
            assert "suggested_tools" in result, f"{tool_name} missing suggested_tools"
            assert isinstance(result["suggested_tools"], list), (
                f"{tool_name} suggested_tools should be a list"
            )

    @pytest.mark.asyncio
    async def test_tools_without_suggestions(self, server):
        """Test that tools without @tool_suggestions decorator do not include suggested_tools."""
        # Tools that do NOT have @tool_suggestions decorator
        tools_without_suggestions = [
            ("explore_identifiers", {"scope": "summary"}),
            ("export_ids", {"ids_list": ["core_profiles"]}),
            ("export_physics_domain", {"domain": "core_plasma"}),
        ]

        for tool_name, kwargs in tools_without_suggestions:
            tool_func = getattr(server.tools, tool_name)
            result = await tool_func(**kwargs)

            # Tools without @tool_suggestions decorator should NOT have suggested_tools
            assert "suggested_tools" not in result, (
                f"{tool_name} unexpectedly has suggested_tools"
            )
