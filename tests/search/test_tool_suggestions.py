"""Tests for search/tool_suggestions.py module."""

import pytest

from imas_mcp.search.tool_suggestions import suggest_follow_up_tools, tool_suggestions


class TestSuggestFollowUpTools:
    """Tests for the suggest_follow_up_tools function."""

    def test_search_results_suggest_overview_and_list(self):
        """Search results suggest overview and list tools."""
        results = {
            "results": [
                {"path": "core_profiles/profiles_1d/temperature", "score": 0.95},
                {"path": "equilibrium/boundary/psi", "score": 0.88},
            ]
        }
        suggestions = suggest_follow_up_tools(results, "search_imas_paths")

        assert len(suggestions) > 0
        assert any(s["tool"] == "get_imas_overview" for s in suggestions)
        assert any(s["tool"] == "list_imas_paths" for s in suggestions)

    def test_overview_results_suggest_search(self):
        """Overview results suggest search tool."""
        results = {"concept": "plasma temperature"}
        suggestions = suggest_follow_up_tools(results, "get_imas_overview")

        assert len(suggestions) > 0
        assert any(s["tool"] == "search_imas_paths" for s in suggestions)

    def test_list_paths_results_suggest_clusters(self):
        """List paths results suggest clusters tool."""
        results = {"ids_name": "equilibrium"}
        suggestions = suggest_follow_up_tools(results, "list_imas_paths")

        assert len(suggestions) > 0
        assert any(s["tool"] == "search_imas_clusters" for s in suggestions)

    def test_fetch_export_results(self):
        """Fetch/export results may have suggestions."""
        results = {"exported": True}
        suggestions = suggest_follow_up_tools(results, "fetch_imas_paths")

        assert isinstance(suggestions, list)

    def test_empty_results_returns_list(self):
        """Empty results returns a list with limited suggestions."""
        suggestions = suggest_follow_up_tools({}, "unknown_function")

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3

    def test_none_results_returns_empty_list(self):
        """None results returns empty list (error handling)."""
        suggestions = suggest_follow_up_tools(None, "search_imas_paths")

        assert suggestions == []


class TestToolSuggestionsDecorator:
    """Tests for the tool_suggestions decorator."""

    @pytest.mark.asyncio
    async def test_decorator_adds_suggested_tools(self):
        """Decorator adds suggested_tools to response."""

        @tool_suggestions
        async def mock_tool():
            return {"results": [{"path": "test/path"}]}

        result = await mock_tool()
        assert "suggested_tools" in result

    @pytest.mark.asyncio
    async def test_decorator_preserves_existing_suggestions(self):
        """Decorator doesn't override existing suggestions."""

        @tool_suggestions
        async def mock_tool():
            return {"results": [], "suggested_tools": ["existing"]}

        result = await mock_tool()
        assert result["suggested_tools"] == ["existing"]

    @pytest.mark.asyncio
    async def test_decorator_handles_non_dict_result(self):
        """Decorator handles non-dict results gracefully."""

        @tool_suggestions
        async def mock_tool():
            return "string result"

        result = await mock_tool()
        assert result == "string result"
