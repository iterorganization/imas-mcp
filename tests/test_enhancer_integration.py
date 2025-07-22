"""
Integration tests for Phase 1.2 Selective AI Enhancement.

Tests the integration of selective AI enhancement strategy with the MCP server,
including tool suggestions, multi-format exports, and conditional AI processing.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from imas_mcp.server import Server


class MockContext:
    """Mock MCP context for testing AI enhancement."""

    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    async def sample(self, prompt, **kwargs):
        """Mock AI sampling with configurable behavior."""
        if self.should_fail:
            raise Exception("AI service unavailable")

        return Mock(text='{"insights": "AI enhancement working", "status": "enhanced"}')


class TestSelectiveAIEnhancement:
    """Test selective AI enhancement integration with server."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server()

    @pytest.mark.asyncio
    async def test_search_fast_mode_no_ai(self, server):
        """Test that lexical search mode provides fast, non-AI search."""
        result = await server.search_imas(
            query="plasma temperature",
            search_mode="lexical",  # Use explicit lexical mode
            max_results=5,
            ctx=MockContext(),
        )

        # Should have results but no AI enhancement for lexical mode
        assert "results" in result
        assert "suggested_tools" in result
        assert result["search_strategy"] == "lexical"
        # No AI enhancement for lexical mode - no ai_insights at all
        assert "ai_insights" not in result

    @pytest.mark.asyncio
    async def test_search_comprehensive_mode_with_ai(self, server):
        """Test that semantic search mode applies AI enhancement."""
        result = await server.search_imas(
            query="plasma temperature profiles",
            search_mode="semantic",  # Use explicit semantic mode
            max_results=10,
            ctx=MockContext(),
        )

        # Should have results with AI enhancement for semantic mode
        assert "results" in result
        assert "suggested_tools" in result
        assert result["search_strategy"] == "semantic"

        # AI should be applied for semantic mode
        ai_insights = result.get("ai_insights", {})
        assert ai_insights.get("status") in ["AI enhancement applied", "enhanced"]

    @pytest.mark.asyncio
    async def test_search_without_context_no_ai(self, server):
        """Test that search without context doesn't apply AI enhancement."""
        result = await server.search_imas(
            query="plasma temperature",
            search_mode="semantic",  # Use semantic mode but without context
            max_results=10,
            ctx=None,
        )

        # Should have results but no AI enhancement without context
        assert "results" in result
        assert "suggested_tools" in result

        # No AI enhancement without context
        ai_insights = result.get("ai_insights", {})
        assert ai_insights.get("status") == "AI context not available"

    @pytest.mark.asyncio
    async def test_explain_concept_always_enhanced(self, server):
        """Test that explain_concept always applies AI enhancement when context available."""
        result = await server.explain_concept(
            concept="plasma temperature", detail_level="intermediate", ctx=MockContext()
        )

        # Should have concept explanation with AI enhancement
        assert "concept" in result
        assert result["concept"] == "plasma temperature"

        # AI should always be applied for explain_concept
        ai_insights = result.get("ai_insights", {})
        assert ai_insights.get("status") in ["AI enhancement applied", "enhanced"]

    @pytest.mark.asyncio
    async def test_explore_identifiers_never_enhanced(self, server):
        """Test that explore_identifiers never applies AI enhancement."""
        result = await server.explore_identifiers(
            query="temperature", scope="summary", ctx=MockContext()
        )

        # Should have identifier data but no AI enhancement
        assert "summary" in result or "overview" in result

        # AI should never be applied for explore_identifiers
        ai_insights = result.get("ai_insights", {})
        assert ai_insights.get("status") == "AI enhancement not needed for this request"

    @pytest.mark.asyncio
    async def test_ai_enhancement_failure_graceful(self, server):
        """Test graceful handling of AI enhancement failures."""
        result = await server.search_imas(
            query="plasma temperature profiles",
            search_mode="comprehensive",
            max_results=10,
            ctx=MockContext(should_fail=True),
        )

        # Should have results even when AI fails
        assert "results" in result
        assert "suggested_tools" in result

        # Should have error message for failed AI enhancement
        ai_insights = result.get("ai_insights", {})
        assert "error" in ai_insights


class TestToolSuggestions:
    """Test tool suggestion integration."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server()

    @pytest.mark.asyncio
    async def test_search_includes_tool_suggestions(self, server):
        """Test that search results include tool suggestions."""
        result = await server.search_imas(query="plasma temperature", max_results=5)

        assert "suggested_tools" in result
        assert isinstance(result["suggested_tools"], list)

        # Check suggestion structure if any suggestions exist
        for suggestion in result["suggested_tools"]:
            assert "tool" in suggestion
            assert "reason" in suggestion
            assert "sample_call" in suggestion

    @pytest.mark.asyncio
    async def test_all_tools_include_suggestions(self, server):
        """Test that all tools include suggested_tools in response."""
        tools_to_test = [
            ("search_imas", {"query": "plasma"}),
            ("explain_concept", {"concept": "plasma"}),
            ("get_overview", {}),
            ("analyze_ids_structure", {"ids_name": "core_profiles"}),
            ("explore_relationships", {"path": "core_profiles"}),
            ("explore_identifiers", {"scope": "summary"}),
            ("export_ids_bulk", {"ids_list": ["core_profiles"]}),
            ("export_physics_domain", {"domain": "core_plasma"}),
        ]

        for tool_name, kwargs in tools_to_test:
            tool_func = getattr(server, tool_name)
            result = await tool_func(**kwargs)

            assert "suggested_tools" in result, f"{tool_name} missing suggested_tools"
            assert isinstance(result["suggested_tools"], list)


class TestMultiFormatExport:
    """Test multi-format export functionality."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server()

    @pytest.mark.asyncio
    async def test_raw_export_format(self, server):
        """Test raw export format (minimal processing)."""
        result = await server.export_ids_bulk(
            ids_list=["core_profiles"],
            output_format="raw",
            include_relationships=True,  # Should be ignored for raw format
            include_physics_context=True,  # Should be ignored for raw format
            ctx=MockContext(),
        )

        assert result["export_format"] == "raw"
        assert "ids_data" in result
        assert "export_summary" in result

        # Raw format should not include relationships or physics context
        assert "cross_relationships" not in result or not result["cross_relationships"]
        assert "physics_domains" not in result or not result["physics_domains"]

        # Should not have AI enhancement for raw format
        ai_insights = result.get("ai_insights", {})
        assert ai_insights.get("status") == "AI enhancement not needed for this request"

    @pytest.mark.asyncio
    async def test_structured_export_format(self, server):
        """Test structured export format (organized data with relationships)."""
        result = await server.export_ids_bulk(
            ids_list=["core_profiles"],
            output_format="structured",
            include_relationships=True,
            include_physics_context=True,
            ctx=MockContext(),
        )

        assert result["export_format"] == "structured"
        assert "ids_data" in result
        assert "export_summary" in result

        # Structured format should include relationships and physics context
        if result.get("valid_ids") and len(result["valid_ids"]) > 1:
            assert "cross_relationships" in result
        if result.get("valid_ids"):
            assert "physics_domains" in result

        # Should not have AI enhancement for structured format
        ai_insights = result.get("ai_insights", {})
        assert ai_insights.get("status") == "AI enhancement not needed for this request"

    @pytest.mark.asyncio
    async def test_enhanced_export_format(self, server):
        """Test enhanced export format (AI-enhanced insights)."""
        result = await server.export_ids_bulk(
            ids_list=["core_profiles"],
            output_format="enhanced",
            include_relationships=True,
            include_physics_context=True,
            ctx=MockContext(),
        )

        assert result["export_format"] == "enhanced"
        assert "ids_data" in result
        assert "export_summary" in result

        # Enhanced format should have AI enhancement
        ai_insights = result.get("ai_insights", {})
        assert ai_insights.get("status") in ["AI enhancement applied", "enhanced"]

    @pytest.mark.asyncio
    async def test_invalid_export_format(self, server):
        """Test handling of invalid export format."""
        result = await server.export_ids_bulk(
            ids_list=["core_profiles"],
            output_format="invalid_format",
            ctx=MockContext(),
        )

        assert "error" in result
        assert "invalid_format" in result["error"]
        assert "suggestions" in result

    @pytest.mark.asyncio
    async def test_export_format_performance_optimization(self, server):
        """Test that different formats have appropriate performance characteristics."""
        # This test focuses on the logical structure rather than actual timing

        # Raw format should have minimal data
        raw_result = await server.export_ids_bulk(
            ids_list=["core_profiles"], output_format="raw"
        )

        # Structured format should have more complete data
        structured_result = await server.export_ids_bulk(
            ids_list=["core_profiles"], output_format="structured"
        )

        # Enhanced format should have AI insights
        enhanced_result = await server.export_ids_bulk(
            ids_list=["core_profiles"], output_format="enhanced", ctx=MockContext()
        )

        # Verify format-specific characteristics
        assert raw_result["export_format"] == "raw"
        assert structured_result["export_format"] == "structured"
        assert enhanced_result["export_format"] == "enhanced"

        # Raw should be most minimal
        assert (
            "ai_insights" not in raw_result
            or raw_result["ai_insights"].get("status")
            == "AI enhancement not needed for this request"
        )

        # Enhanced should have AI insights
        assert "ai_insights" in enhanced_result


class TestConditionalAILogic:
    """Test conditional AI logic for various tools."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server()

    @pytest.mark.asyncio
    async def test_structure_analysis_conditional_ai(self, server):
        """Test conditional AI for structure analysis based on IDS complexity."""
        # Test with complex IDS (should enable AI)
        result = await server.analyze_ids_structure(
            ids_name="core_profiles", ctx=MockContext()
        )

        if "error" not in result:  # Only test if IDS exists
            ai_insights = result.get("ai_insights", {})
            # Should apply AI for complex IDS like core_profiles
            assert ai_insights.get("status") in ["AI enhancement applied", "enhanced"]

    @pytest.mark.asyncio
    async def test_relationships_conditional_ai(self, server):
        """Test conditional AI for relationship exploration."""
        # Test with deep analysis (should enable AI)
        result = await server.explore_relationships(
            path="core_profiles", max_depth=3, ctx=MockContext()
        )

        if "error" not in result:  # Only test if path exists
            ai_insights = result.get("ai_insights", {})
            # Should apply AI for deep relationship analysis
            assert ai_insights.get("status") in ["AI enhancement applied", "enhanced"]

    @pytest.mark.asyncio
    async def test_physics_domain_conditional_ai(self, server):
        """Test conditional AI for physics domain export."""
        # Test with comprehensive analysis (should enable AI)
        result = await server.export_physics_domain(
            domain="core_plasma", analysis_depth="comprehensive", ctx=MockContext()
        )

        if "error" not in result:  # Only test if domain has data
            ai_insights = result.get("ai_insights", {})
            # Should apply AI for comprehensive analysis
            assert ai_insights.get("status") in ["AI enhancement applied", "enhanced"]

        # Test with focused analysis (should not enable AI)
        result = await server.export_physics_domain(
            domain="core_plasma", analysis_depth="focused", ctx=MockContext()
        )

        if "error" not in result:  # Only test if domain has data
            ai_insights = result.get("ai_insights", {})
            # Should not apply AI for focused analysis
            assert (
                ai_insights.get("status")
                == "AI enhancement not needed for this request"
            )


class TestErrorHandling:
    """Test error handling in selective AI enhancement."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server()

    @pytest.mark.asyncio
    async def test_ai_failure_graceful_degradation(self, server):
        """Test graceful degradation when AI enhancement fails."""
        result = await server.search_imas(
            query="plasma temperature",
            search_mode="comprehensive",
            ctx=MockContext(should_fail=True),
        )

        # Should still return valid results even when AI fails
        assert "results" in result
        assert "suggested_tools" in result
        assert "search_strategy" in result

        # Should have error in AI insights
        ai_insights = result.get("ai_insights", {})
        assert "error" in ai_insights

    @pytest.mark.asyncio
    async def test_malformed_ai_response_handling(self, server):
        """Test handling of malformed AI responses."""
        # Mock context that returns malformed JSON
        mock_ctx = Mock()
        mock_ctx.sample = AsyncMock(return_value=Mock(text="invalid json {"))

        result = await server.explain_concept(
            concept="plasma temperature", ctx=mock_ctx
        )

        # Should still return valid results
        assert "concept" in result
        assert result["concept"] == "plasma temperature"

        # Should handle malformed AI response gracefully
        ai_insights = result.get("ai_insights", {})
        assert "response" in ai_insights or "error" in ai_insights

    @pytest.mark.asyncio
    async def test_suggestion_generation_error_handling(self, server):
        """Test error handling in tool suggestion generation."""
        # This should not cause the main function to fail
        result = await server.search_imas(query="plasma temperature", max_results=5)

        # Should always include suggested_tools, even if empty
        assert "suggested_tools" in result
        assert isinstance(result["suggested_tools"], list)


@pytest.mark.integration
class TestIntegration:
    """Integration tests for selective AI enhancement."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server()

    @pytest.mark.asyncio
    async def test_complete_workflow_with_suggestions(self, server):
        """Test complete workflow using tool suggestions."""
        # Start with search
        search_result = await server.search_imas(
            query="plasma temperature", search_mode="comprehensive", ctx=MockContext()
        )

        assert "suggested_tools" in search_result
        assert len(search_result["suggested_tools"]) > 0

        # Follow a suggestion if available
        suggestions = search_result["suggested_tools"]
        explain_suggestion = next(
            (s for s in suggestions if "explain_concept" in s["tool"]), None
        )

        if explain_suggestion:
            # Test following the suggestion
            concept_result = await server.explain_concept(
                concept="plasma temperature", ctx=MockContext()
            )

            assert "concept" in concept_result
            assert "suggested_tools" in concept_result

    @pytest.mark.asyncio
    async def test_selective_ai_performance_optimization(self, server):
        """Test that selective AI enhancement provides performance benefits."""
        # Fast operations should not trigger AI
        fast_result = await server.search_imas(
            query="plasma", search_mode="fast", ctx=MockContext()
        )

        assert (
            fast_result.get("ai_insights", {}).get("status")
            == "AI enhancement not needed for this request"
        )

        # Raw exports should not trigger AI
        raw_export = await server.export_ids_bulk(
            ids_list=["core_profiles"], output_format="raw", ctx=MockContext()
        )

        assert (
            raw_export.get("ai_insights", {}).get("status")
            == "AI enhancement not needed for this request"
        )

    @pytest.mark.asyncio
    async def test_format_based_processing(self, server):
        """Test that format-based processing works correctly."""
        formats = ["raw", "structured", "enhanced"]

        for format_type in formats:
            result = await server.export_ids_bulk(
                ids_list=["core_profiles"],
                output_format=format_type,
                ctx=MockContext() if format_type == "enhanced" else None,
            )

            assert result["export_format"] == format_type

            if format_type == "enhanced":
                # Enhanced format should have AI insights when context provided
                assert "ai_insights" in result
            else:
                # Other formats should not have AI enhancement
                ai_insights = result.get("ai_insights", {})
                if ai_insights:
                    assert (
                        ai_insights.get("status")
                        == "AI enhancement not needed for this request"
                    )
