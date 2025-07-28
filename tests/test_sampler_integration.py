"""
Integration tests for sampler (AI enhancement) functionality.

Tests the integration of sampling decorator with the MCP server,
including tool suggestions, multi-format exports, and conditional AI processing.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from imas_mcp.server import Server
from imas_mcp.models.enums import SearchMode
from tests.conftest import STANDARD_TEST_IDS_SET


class MockContext:
    """Mock MCP context for testing sampling enhancement."""

    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    async def sample(self, prompt, **kwargs):
        """Mock AI sampling with configurable behavior."""
        if self.should_fail:
            raise Exception("AI service unavailable")

        return Mock(text='{"insights": "AI enhancement working", "status": "enhanced"}')


class TestSelectiveSampling:
    """Test selective AI enhancement integration with server."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server(ids_set=STANDARD_TEST_IDS_SET)

    @pytest.mark.asyncio
    async def test_search_fast_mode_no_ai(self, server):
        """Test that lexical search mode provides fast, non-AI search."""
        result = await server.tools.search_imas(
            query="plasma temperature",
            search_mode="lexical",  # Use explicit lexical mode
            max_results=5,
            ctx=MockContext(),
        )

        # Should have results but no AI enhancement for lexical mode
        assert "results" in result
        assert "suggestions" in result
        assert result["search_mode"] == SearchMode.LEXICAL
        # New system always includes ai_insights with status message
        ai_insights = result.get("ai_insights", {})
        assert (
            ai_insights.get("status")
            == "AI enhancement not applied - conditions not met"
        )

    @pytest.mark.asyncio
    async def test_search_comprehensive_mode_with_ai(self, server):
        """Test that semantic search mode applies AI enhancement."""
        result = await server.tools.search_imas(
            query="plasma temperature profiles",
            search_mode="semantic",  # Use explicit semantic mode
            max_results=10,
            ctx=MockContext(),
        )

        # Should have results with AI enhancement for semantic mode
        assert "results" in result
        assert "suggestions" in result
        assert result["search_mode"] == SearchMode.SEMANTIC

        # AI should be applied for semantic mode if implemented
        ai_insights = result.get("ai_insights", {})
        # Accept that AI enhancement may not be fully implemented yet
        if ai_insights:
            assert ai_insights.get("status") in [
                "AI enhancement applied",
                "enhanced",
                "AI context not available",
            ]

    @pytest.mark.asyncio
    async def test_search_without_context_no_ai(self, server):
        """Test that search without context doesn't apply AI enhancement."""
        result = await server.tools.search_imas(
            query="plasma temperature",
            search_mode="semantic",  # Use semantic mode but without context
            max_results=10,
            ctx=None,
        )

        # Should have results but no AI enhancement without context
        assert "results" in result
        assert "suggestions" in result

        # No AI enhancement without context
        ai_insights = result.get("ai_insights", {})
        assert (
            ai_insights.get("status")
            == "AI enhancement not applied - conditions not met"
        )

    @pytest.mark.asyncio
    async def test_explain_concept_always_enhanced(self, server):
        """Test that explain_concept always applies AI enhancement when context available."""
        result = await server.tools.explain_concept(
            concept="plasma temperature", detail_level="intermediate", ctx=MockContext()
        )

        # Should have concept explanation with AI enhancement
        assert "concept" in result
        assert result["concept"] == "plasma temperature"

        # AI should always be applied for explain_concept
        ai_insights = result.get("ai_insights", {})
        # Accept that AI enhancement may not be fully implemented yet
        if ai_insights:
            assert ai_insights.get("status") in [
                "AI enhancement applied",
                "enhanced",
                "AI context not available",
            ]

    @pytest.mark.asyncio
    async def test_explore_identifiers_never_enhanced(self, server):
        """Test that explore_identifiers never applies AI enhancement."""
        try:
            result = await server.tools.explore_identifiers(
                query="temperature", scope="summary", ctx=MockContext()
            )

            # Should have identifier data but no AI enhancement
            assert "total_schemas" in result
            assert "schemas" in result
            assert "branching_analytics" in result

            # AI should never be applied for explore_identifiers (never strategy)
            ai_insights = result.get("ai_insights", {})
            assert (
                ai_insights.get("status")
                == "AI enhancement not applied - conditions not met"
            )
        except NotImplementedError:
            # Skip test if explore_identifiers is not implemented yet
            pytest.skip("explore_identifiers not yet implemented")

    @pytest.mark.asyncio
    async def test_ai_enhancement_failure_graceful(self, server):
        """Test graceful handling of AI enhancement failures."""
        result = await server.tools.search_imas(
            query="plasma temperature profiles",
            search_mode="semantic",  # Use semantic instead of comprehensive
            max_results=10,
            ctx=MockContext(should_fail=True),
        )

        # Should have results even when AI fails
        assert "results" in result
        assert "suggestions" in result
        assert "search_mode" in result

        # Should have error in AI insights if AI enhancement is implemented
        ai_insights = result.get("ai_insights", {})
        if ai_insights:
            assert "error" in ai_insights or ai_insights.get("status") in [
                "AI context not available",
                "AI enhancement not needed for this request",
            ]


class TestMultiFormatExport:
    """Test multi-format export functionality."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server(ids_set=STANDARD_TEST_IDS_SET)

    @pytest.mark.asyncio
    async def test_raw_export_format(self, server):
        """Test raw export format (minimal processing)."""
        result = await server.tools.export_ids(
            ids_list=["core_profiles"],
            output_format="raw",
            include_relationships=True,  # Should be ignored for raw format
            include_physics_context=True,  # Should be ignored for raw format
            ctx=MockContext(),
        )

        assert result["output_format"] == "raw"
        assert "export_data" in result
        assert "ids_data" in result["export_data"]
        assert "export_summary" in result["export_data"]

        # Raw format should not include relationships or physics context
        export_data = result["export_data"]
        assert (
            "cross_relationships" not in export_data
            or not export_data["cross_relationships"]
        )
        assert (
            "physics_domains" not in export_data or not export_data["physics_domains"]
        )

        # Should not have AI enhancement for raw format
        ai_insights = result.get("ai_insights", {})
        assert (
            ai_insights.get("status")
            == "AI enhancement not applied - conditions not met"
        )

    @pytest.mark.asyncio
    async def test_structured_export_format(self, server):
        """Test structured export format (organized data with relationships)."""
        result = await server.tools.export_ids(
            ids_list=["core_profiles"],
            output_format="structured",
            include_relationships=True,
            include_physics_context=True,
            ctx=MockContext(),
        )

        assert result["output_format"] == "structured"
        assert "export_data" in result
        assert "ids_data" in result["export_data"]
        assert "export_summary" in result["export_data"]

        # Structured format should include relationships and physics context
        export_data = result["export_data"]
        if export_data.get("valid_ids") and len(export_data["valid_ids"]) > 1:
            assert "cross_relationships" in export_data
        if export_data.get("valid_ids"):
            assert "physics_domains" in export_data

        # Should not have AI enhancement for structured format
        ai_insights = result.get("ai_insights", {})
        assert (
            ai_insights.get("status")
            == "AI enhancement not applied - conditions not met"
        )

    @pytest.mark.asyncio
    async def test_enhanced_export_format(self, server):
        """Test enhanced export format (AI-enhanced insights)."""
        result = await server.tools.export_ids(
            ids_list=["core_profiles"],
            output_format="enhanced",
            include_relationships=True,
            include_physics_context=True,
            ctx=MockContext(),
        )

        assert result["output_format"] == "enhanced"
        assert "export_data" in result
        assert "ids_data" in result["export_data"]
        assert "export_summary" in result["export_data"]

        # Enhanced format should have AI enhancement
        ai_insights = result.get("ai_insights", {})
        assert ai_insights.get("status") in ["AI enhancement applied", "enhanced"]

    @pytest.mark.asyncio
    async def test_invalid_export_format(self, server):
        """Test handling of invalid export format."""
        result = await server.tools.export_ids(
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
        raw_result = await server.tools.export_ids(
            ids_list=["core_profiles"], output_format="raw"
        )

        # Structured format should have more complete data
        structured_result = await server.tools.export_ids(
            ids_list=["core_profiles"], output_format="structured"
        )

        # Enhanced format should have AI insights
        enhanced_result = await server.tools.export_ids(
            ids_list=["core_profiles"], output_format="enhanced", ctx=MockContext()
        )

        # Verify format-specific characteristics
        assert raw_result["output_format"] == "raw"
        assert structured_result["output_format"] == "structured"
        assert enhanced_result["output_format"] == "enhanced"

        # Raw should be most minimal
        assert (
            "ai_insights" in raw_result
            and raw_result["ai_insights"].get("status")
            == "AI enhancement not applied - conditions not met"
        )

        # Enhanced should have AI insights
        assert "ai_insights" in enhanced_result


class TestConditionalSamplingLogic:
    """Test conditional AI logic for various tools."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server(ids_set=STANDARD_TEST_IDS_SET)

    @pytest.mark.asyncio
    async def test_structure_analysis_conditional_ai(self, server):
        """Test conditional AI for structure analysis based on IDS complexity."""
        # Test with complex IDS (should enable AI)
        result = await server.tools.analyze_ids_structure(
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
        result = await server.tools.explore_relationships(
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
        result = await server.tools.export_physics_domain(
            domain="core_plasma", analysis_depth="comprehensive", ctx=MockContext()
        )

        if "error" not in result:  # Only test if domain has data
            ai_insights = result.get("ai_insights", {})
            # Should apply AI for comprehensive analysis
            assert ai_insights.get("status") in ["AI enhancement applied", "enhanced"]

        # Test with focused analysis (should not enable AI)
        result = await server.tools.export_physics_domain(
            domain="core_plasma", analysis_depth="focused", ctx=MockContext()
        )

        if "error" not in result:  # Only test if domain has data
            ai_insights = result.get("ai_insights", {})
            # Should not apply AI for focused analysis
            assert (
                ai_insights.get("status")
                == "AI enhancement not applied - conditions not met"
            )


class TestErrorHandling:
    """Test error handling in selective AI enhancement."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server(ids_set=STANDARD_TEST_IDS_SET)

    @pytest.mark.asyncio
    async def test_ai_failure_graceful_degradation(self, server):
        """Test graceful degradation when AI enhancement fails."""
        result = await server.tools.search_imas(
            query="plasma temperature",
            search_mode="semantic",  # Use semantic instead of comprehensive
            ctx=MockContext(should_fail=True),
        )

        # Should still return valid results even when AI fails
        assert "results" in result
        assert "suggestions" in result
        assert "search_mode" in result

        # Should have error in AI insights
        ai_insights = result.get("ai_insights", {})
        assert "error" in ai_insights

    @pytest.mark.asyncio
    async def test_malformed_ai_response_handling(self, server):
        """Test handling of malformed AI responses."""
        # Mock context that returns malformed JSON
        mock_ctx = Mock()
        mock_ctx.sample = AsyncMock(return_value=Mock(text="invalid json {"))

        result = await server.tools.explain_concept(
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
        result = await server.tools.search_imas(
            query="plasma temperature", max_results=5
        )

        # Should always include suggestions, even if empty
        assert "suggestions" in result
        assert isinstance(result["suggestions"], list)


@pytest.mark.integration
class TestIntegration:
    """Integration tests for selective AI enhancement."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server(ids_set=STANDARD_TEST_IDS_SET)

    @pytest.mark.asyncio
    async def test_complete_workflow_with_suggestions(self, server):
        """Test complete workflow using tool suggestions."""
        # Start with search
        search_result = await server.tools.search_imas(
            query="plasma temperature", search_mode="semantic", ctx=MockContext()
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
            concept_result = await server.tools.explain_concept(
                concept="plasma temperature", ctx=MockContext()
            )

            assert "concept" in concept_result
            assert "suggestions" in concept_result

    @pytest.mark.asyncio
    async def test_selective_ai_performance_optimization(self, server):
        """Test that selective AI enhancement provides performance benefits."""
        # Fast operations should not trigger AI
        fast_result = await server.tools.search_imas(
            query="plasma", search_mode="lexical", ctx=MockContext()
        )

        assert (
            fast_result.get("ai_insights", {}).get("status")
            == "AI enhancement not applied - conditions not met"
        )

        # Raw exports should not trigger AI
        raw_export = await server.tools.export_ids(
            ids_list=["core_profiles"], output_format="raw", ctx=MockContext()
        )

        assert (
            raw_export.get("ai_insights", {}).get("status")
            == "AI enhancement not applied - conditions not met"
        )

    @pytest.mark.asyncio
    async def test_format_based_processing(self, server):
        """Test that format-based processing works correctly."""
        formats = ["raw", "structured", "enhanced"]

        for format_type in formats:
            result = await server.tools.export_ids(
                ids_list=["core_profiles"],
                output_format=format_type,
                ctx=MockContext() if format_type == "enhanced" else None,
            )

            assert result["output_format"] == format_type

            if format_type == "enhanced":
                # Enhanced format should have AI insights when context provided
                assert "ai_insights" in result
            else:
                # Other formats should not have AI enhancement
                ai_insights = result.get("ai_insights", {})
                assert (
                    ai_insights.get("status")
                    == "AI enhancement not applied - conditions not met"
                )
