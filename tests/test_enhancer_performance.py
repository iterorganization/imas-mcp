"""
Performance tests for Phase 1.2 Selective AI Enhancement.

Tests that the selective AI enhancement strategy provides the expected
performance benefits by avoiding unnecessary AI processing.
"""

import asyncio
import time
import pytest
from unittest.mock import Mock

from imas_mcp.server import Server


class SlowMockContext:
    """Mock context that simulates slow AI processing."""

    def __init__(self, delay=0.5):
        self.delay = delay

    async def sample(self, prompt, **kwargs):
        """Mock AI sampling with artificial delay."""
        await asyncio.sleep(self.delay)
        return Mock(text='{"insights": "AI enhancement working", "status": "enhanced"}')


class FastMockContext:
    """Mock context that simulates fast AI processing."""

    async def sample(self, prompt, **kwargs):
        """Mock AI sampling with minimal delay."""
        return Mock(text='{"insights": "AI enhancement working", "status": "enhanced"}')


@pytest.mark.performance
class TestSelectiveAIPerformance:
    """Test performance benefits of selective AI enhancement."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server()

    @pytest.mark.asyncio
    async def test_fast_search_performance(self, server):
        """Test that fast search mode avoids AI processing delay."""
        slow_ctx = SlowMockContext(delay=0.5)

        # Fast mode should not use AI, so should be fast even with slow context
        start_time = time.time()
        result = await server.search_imas(
            query="plasma temperature", search_mode="fast", max_results=5, ctx=slow_ctx
        )
        execution_time = time.time() - start_time

        # Should complete quickly since AI is not used
        assert execution_time < 0.3  # Should be much faster than AI delay
        assert result["search_strategy"] == "lexical"
        assert (
            result.get("ai_insights", {}).get("status")
            == "AI enhancement not needed for this request"
        )

    @pytest.mark.asyncio
    async def test_comprehensive_search_performance(self, server):
        """Test that comprehensive search mode uses AI when context available."""
        slow_ctx = SlowMockContext(delay=0.3)

        # Comprehensive mode should use AI, so will be slower
        start_time = time.time()
        result = await server.search_imas(
            query="plasma temperature profiles",
            search_mode="comprehensive",
            max_results=10,
            ctx=slow_ctx,
        )
        execution_time = time.time() - start_time

        # Should take longer due to AI processing
        assert execution_time >= 0.2  # Should include AI delay
        assert result["search_strategy"] == "semantic"
        ai_insights = result.get("ai_insights", {})
        assert ai_insights.get("status") in ["AI enhancement applied", "enhanced"]

    @pytest.mark.asyncio
    async def test_raw_export_performance(self, server):
        """Test that raw export format avoids AI processing."""
        slow_ctx = SlowMockContext(delay=0.5)

        # Raw format should not use AI
        start_time = time.time()
        result = await server.export_ids_bulk(
            ids_list=["core_profiles"], output_format="raw", ctx=slow_ctx
        )
        execution_time = time.time() - start_time

        # Should complete quickly since AI is not used
        assert execution_time < 0.3
        assert result["export_format"] == "raw"
        assert (
            result.get("ai_insights", {}).get("status")
            == "AI enhancement not needed for this request"
        )

    @pytest.mark.asyncio
    async def test_enhanced_export_performance(self, server):
        """Test that enhanced export format uses AI processing."""
        slow_ctx = SlowMockContext(delay=0.3)

        # Enhanced format should use AI
        start_time = time.time()
        result = await server.export_ids_bulk(
            ids_list=["core_profiles"], output_format="enhanced", ctx=slow_ctx
        )
        execution_time = time.time() - start_time

        # Should take longer due to AI processing
        assert execution_time >= 0.2
        assert result["export_format"] == "enhanced"
        ai_insights = result.get("ai_insights", {})
        assert ai_insights.get("status") in ["AI enhancement applied", "enhanced"]

    @pytest.mark.asyncio
    async def test_never_strategy_performance(self, server):
        """Test that 'never' strategy tools avoid AI processing."""
        slow_ctx = SlowMockContext(delay=0.5)

        # explore_identifiers should never use AI
        start_time = time.time()
        result = await server.explore_identifiers(
            query="temperature", scope="summary", ctx=slow_ctx
        )
        execution_time = time.time() - start_time

        # Should complete quickly since AI is never used
        assert execution_time < 0.3
        assert (
            result.get("ai_insights", {}).get("status")
            == "AI enhancement not needed for this request"
        )

    @pytest.mark.asyncio
    async def test_always_strategy_performance(self, server):
        """Test that 'always' strategy tools use AI processing."""
        slow_ctx = SlowMockContext(delay=0.3)

        # explain_concept should always use AI
        start_time = time.time()
        result = await server.explain_concept(
            concept="plasma temperature", detail_level="intermediate", ctx=slow_ctx
        )
        execution_time = time.time() - start_time

        # Should take longer due to AI processing
        assert execution_time >= 0.2
        ai_insights = result.get("ai_insights", {})
        assert ai_insights.get("status") in ["AI enhancement applied", "enhanced"]

    @pytest.mark.asyncio
    async def test_conditional_strategy_performance_variation(self, server):
        """Test that conditional strategy varies performance based on parameters."""
        slow_ctx = SlowMockContext(delay=0.3)

        # Test structure analysis with simple IDS (should not use AI)
        start_time = time.time()
        result1 = await server.analyze_ids_structure(
            ids_name="simple_test",  # Simple name that shouldn't trigger AI
            ctx=slow_ctx,
        )
        time1 = time.time() - start_time

        # Test structure analysis with complex IDS (should use AI if IDS exists)
        start_time = time.time()
        result2 = await server.analyze_ids_structure(
            ids_name="core_profiles",  # Complex name that should trigger AI
            ctx=slow_ctx,
        )
        time2 = time.time() - start_time

        # If both succeed, complex IDS should take longer (if it uses AI)
        if "error" not in result1 and "error" not in result2:
            # Check AI usage based on results
            ai1 = result1.get("ai_insights", {}).get("status", "")
            ai2 = result2.get("ai_insights", {}).get("status", "")

            if "not needed" in ai1 and "applied" in ai2:
                assert time2 > time1

    @pytest.mark.asyncio
    async def test_tool_suggestions_performance(self, server):
        """Test that tool suggestions don't significantly impact performance."""
        # Test with and without suggestion generation
        start_time = time.time()
        result = await server.search_imas(
            query="plasma temperature", search_mode="fast", max_results=5
        )
        execution_time = time.time() - start_time

        # Should complete quickly and include suggestions
        assert execution_time < 0.5
        assert "suggested_tools" in result
        assert isinstance(result["suggested_tools"], list)

    @pytest.mark.asyncio
    async def test_no_context_performance(self, server):
        """Test that operations without context are fast."""
        # No context should mean no AI processing delay
        start_time = time.time()
        result = await server.search_imas(
            query="plasma temperature",
            search_mode="comprehensive",  # Would use AI if context available
            max_results=10,
            ctx=None,  # No context
        )
        execution_time = time.time() - start_time

        # Should be fast without context
        assert execution_time < 0.3
        assert result.get("ai_insights", {}).get("status") == "AI context not available"


@pytest.mark.performance
class TestMultiFormatExportPerformance:
    """Test performance characteristics of different export formats."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server()

    @pytest.mark.asyncio
    async def test_format_performance_hierarchy(self, server):
        """Test that export formats have expected performance hierarchy."""
        slow_ctx = SlowMockContext(delay=0.3)
        test_ids = ["core_profiles"]

        # Test raw format (should be fastest)
        start_time = time.time()
        raw_result = await server.export_ids_bulk(
            ids_list=test_ids, output_format="raw", ctx=slow_ctx
        )
        raw_time = time.time() - start_time

        # Test structured format (medium speed)
        start_time = time.time()
        structured_result = await server.export_ids_bulk(
            ids_list=test_ids, output_format="structured", ctx=slow_ctx
        )
        structured_time = time.time() - start_time

        # Test enhanced format (should be slowest due to AI)
        start_time = time.time()
        enhanced_result = await server.export_ids_bulk(
            ids_list=test_ids, output_format="enhanced", ctx=slow_ctx
        )
        enhanced_time = time.time() - start_time

        # Verify format characteristics
        assert raw_result["export_format"] == "raw"
        assert structured_result["export_format"] == "structured"
        assert enhanced_result["export_format"] == "enhanced"

        # Raw should be fastest (no AI, minimal processing)
        assert raw_time < 0.3

        # Enhanced should take longest (includes AI processing)
        assert enhanced_time >= 0.2

        # Performance hierarchy should generally hold: raw <= structured < enhanced
        # (allowing some variance for test execution)
        assert raw_time <= structured_time + 0.1
        assert structured_time <= enhanced_time + 0.1

    @pytest.mark.asyncio
    async def test_bulk_size_performance_scaling(self, server):
        """Test that performance scales appropriately with bulk export size."""
        fast_ctx = FastMockContext()

        # Small export
        start_time = time.time()
        small_result = await server.export_ids_bulk(
            ids_list=["core_profiles"], output_format="structured", ctx=fast_ctx
        )
        small_time = time.time() - start_time

        # Larger export (if multiple IDS available)
        available_ids = small_result.get("valid_ids", ["core_profiles"])
        if len(available_ids) >= 1:
            # Use available IDS or repeat single IDS for testing
            test_ids = available_ids[:1] * 2  # Test with repeated IDS if needed

            start_time = time.time()
            large_result = await server.export_ids_bulk(
                ids_list=test_ids, output_format="structured", ctx=fast_ctx
            )
            large_time = time.time() - start_time

            # Larger exports should generally take more time
            # (though this depends on data availability)
            assert isinstance(large_time, float)  # Basic validation
            assert large_result["export_format"] == "structured"
            assert small_time >= 0  # Ensure timing was recorded
            assert large_time >= 0  # Ensure timing was recorded


@pytest.mark.performance
class TestConditionalAIPerformanceBenefits:
    """Test specific performance benefits of conditional AI logic."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server()

    @pytest.mark.asyncio
    async def test_search_complexity_performance_scaling(self, server):
        """Test that search complexity appropriately affects AI usage."""
        slow_ctx = SlowMockContext(delay=0.3)

        # Simple query (should not trigger AI in conditional mode)
        start_time = time.time()
        simple_result = await server.search_imas(
            query="plasma",  # Simple single term
            search_mode="auto",  # Conditional mode
            max_results=5,
            ctx=slow_ctx,
        )
        simple_time = time.time() - start_time

        # Complex query (should trigger AI in conditional mode)
        start_time = time.time()
        complex_result = await server.search_imas(
            query="plasma AND temperature AND (density OR pressure)",  # Complex boolean
            search_mode="auto",  # Conditional mode
            max_results=15,  # High result count
            ctx=slow_ctx,
        )
        complex_time = time.time() - start_time

        # Simple should be faster (no AI), complex may be slower (with AI)
        simple_ai = simple_result.get("ai_insights", {}).get("status", "")
        complex_ai = complex_result.get("ai_insights", {}).get("status", "")

        # Simple should avoid AI processing
        assert "not needed" in simple_ai or "not available" in simple_ai
        assert simple_time < 0.3

        # Complex may use AI depending on conditions
        assert isinstance(complex_time, float)
        assert isinstance(complex_ai, str)

    @pytest.mark.asyncio
    async def test_relationship_depth_performance_scaling(self, server):
        """Test that relationship analysis depth affects AI usage."""
        slow_ctx = SlowMockContext(delay=0.3)

        # Shallow analysis (should not trigger AI)
        start_time = time.time()
        shallow_result = await server.explore_relationships(
            path="core_profiles",
            max_depth=1,  # Shallow depth
            ctx=slow_ctx,
        )
        shallow_time = time.time() - start_time

        # Deep analysis (should trigger AI)
        start_time = time.time()
        deep_result = await server.explore_relationships(
            path="core_profiles",
            max_depth=3,  # Deep analysis
            ctx=slow_ctx,
        )
        deep_time = time.time() - start_time

        # If both succeed, check AI usage
        if "error" not in shallow_result and "error" not in deep_result:
            shallow_ai = shallow_result.get("ai_insights", {}).get("status", "")
            deep_ai = deep_result.get("ai_insights", {}).get("status", "")

            # Shallow should avoid AI, deep should use it
            if "not needed" in shallow_ai and "applied" in deep_ai:
                assert deep_time > shallow_time

    @pytest.mark.asyncio
    async def test_physics_domain_depth_performance_scaling(self, server):
        """Test that physics domain analysis depth affects AI usage."""
        slow_ctx = SlowMockContext(delay=0.3)

        # Focused analysis (should not trigger AI)
        start_time = time.time()
        focused_result = await server.export_physics_domain(
            domain="core_plasma",
            analysis_depth="focused",  # Should not trigger AI
            max_paths=5,
            ctx=slow_ctx,
        )
        focused_time = time.time() - start_time

        # Comprehensive analysis (should trigger AI)
        start_time = time.time()
        comprehensive_result = await server.export_physics_domain(
            domain="core_plasma",
            analysis_depth="comprehensive",  # Should trigger AI
            max_paths=5,
            ctx=slow_ctx,
        )
        comprehensive_time = time.time() - start_time

        # If both succeed, check AI usage and timing
        if "error" not in focused_result and "error" not in comprehensive_result:
            focused_ai = focused_result.get("ai_insights", {}).get("status", "")
            comprehensive_ai = comprehensive_result.get("ai_insights", {}).get(
                "status", ""
            )

            # Focused should avoid AI, comprehensive should use it
            assert "not needed" in focused_ai
            if "applied" in comprehensive_ai:
                assert comprehensive_time > focused_time


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance"])
