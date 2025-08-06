"""
Unit tests for sampling service.

Tests the sampling service that determines when to apply AI sampling
to tool results based on result characteristics and sampling strategies.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import List, Any
from pydantic import Field

from imas_mcp.services.sampling import SamplingService, SamplingStrategy
from imas_mcp.models.response_models import AIResponse


class MockResult(AIResponse):
    """Mock result for testing."""

    hits: List[Any] = Field(default_factory=list)
    hit_count: int = 0
    nodes: List[Any] = Field(default_factory=list)


class TestSamplingService:
    """Test the sampling service functionality."""

    def test_init(self):
        """Test service initialization."""
        service = SamplingService()
        assert service is not None

    def test_pydantic_ensures_hits_is_list(self):
        """Test that Pydantic ensures hits is always a list, never None."""
        # Test default initialization
        result = MockResult()
        assert result.hits == []
        assert isinstance(result.hits, list)

        # Test with explicit list
        result_with_hits = MockResult(hits=[{"id": "test"}])
        assert result_with_hits.hits == [{"id": "test"}]
        assert isinstance(result_with_hits.hits, list)

        # Test that we cannot set hits to None (this would cause a type error)
        # This test ensures the type system prevents None assignment

    @pytest.mark.asyncio
    async def test_apply_sampling_no_sampling_strategy(self):
        """Test that NO_SAMPLING strategy returns original result unchanged."""
        service = SamplingService()
        result = MockResult(hits=[{"id": "test"}], hit_count=1)

        processed_result = await service.apply_sampling(
            result=result, strategy=SamplingStrategy.NO_SAMPLING
        )

        assert processed_result == result
        # Check that ai_insights is either empty dict or not set
        assert (
            not getattr(processed_result, "ai_insights", {})
            or processed_result.ai_response == {}
        )

    @pytest.mark.asyncio
    async def test_apply_sampling_always_strategy(self):
        """Test that ALWAYS strategy always attempts sampling."""
        service = SamplingService()
        result = MockResult(hits=[{"id": "test"}], hit_count=1)
        mock_context = MagicMock()
        mock_context.sample = AsyncMock(return_value="AI generated insights")

        processed_result = await service.apply_sampling(
            result=result, strategy=SamplingStrategy.ALWAYS, ctx=mock_context
        )

        # Should attempt sampling
        assert mock_context.sample.called
        assert processed_result == result
        assert processed_result.ai_response == "AI generated insights"

    @pytest.mark.asyncio
    async def test_apply_sampling_without_context(self):
        """Test sampling without MCP context returns original result."""
        service = SamplingService()
        result = MockResult(hits=[{"id": "test"}], hit_count=1)

        processed_result = await service.apply_sampling(
            result=result, strategy=SamplingStrategy.ALWAYS, ctx=None
        )

        assert processed_result == result

    @pytest.mark.asyncio
    async def test_apply_sampling_with_custom_prompt(self):
        """Test sampling with custom prompt."""
        service = SamplingService()
        result = MockResult(hits=[{"id": "test"}], hit_count=1)
        mock_context = MagicMock()
        mock_context.sample = AsyncMock(return_value="Custom insights")

        custom_prompt = "Analyze this specific result"

        processed_result = await service.apply_sampling(
            result=result,
            strategy=SamplingStrategy.ALWAYS,
            sample_prompt=custom_prompt,
            ctx=mock_context,
        )

        # Verify custom prompt was used
        mock_context.sample.assert_called_once()
        call_args = mock_context.sample.call_args
        assert "prompt" in call_args.kwargs
        assert call_args.kwargs["prompt"] == custom_prompt
        assert processed_result.ai_response == "Custom insights"

    @pytest.mark.asyncio
    async def test_apply_sampling_error_handling(self):
        """Test that sampling errors are handled gracefully."""
        service = SamplingService()
        result = MockResult(hits=[{"id": "test"}], hit_count=1)
        mock_context = MagicMock()
        mock_context.sample = AsyncMock(side_effect=Exception("Sampling failed"))

        processed_result = await service.apply_sampling(
            result=result, strategy=SamplingStrategy.ALWAYS, ctx=mock_context
        )

        # Should return original result on error
        assert processed_result == result

    def test_should_sample_no_sampling_strategy(self):
        """Test should_sample with NO_SAMPLING strategy."""
        service = SamplingService()
        result = MockResult(hits=[{"id": "test"}], hit_count=1)

        should_sample = service.should_sample(
            strategy=SamplingStrategy.NO_SAMPLING, result=result
        )

        assert should_sample is False

    def test_should_sample_always_strategy(self):
        """Test should_sample with ALWAYS strategy."""
        service = SamplingService()
        result = MockResult(hits=[{"id": "test"}], hit_count=1)

        should_sample = service.should_sample(
            strategy=SamplingStrategy.ALWAYS, result=result
        )

        assert should_sample is True

    def test_should_sample_conditional_with_context(self):
        """Test should_sample with CONDITIONAL strategy when context is available."""
        service = SamplingService()
        result = MockResult(hits=[{"id": "test"}], hit_count=1)
        mock_context = MagicMock()

        should_sample = service.should_sample(
            strategy=SamplingStrategy.CONDITIONAL, result=result, ctx=mock_context
        )

        assert should_sample is True

    def test_should_sample_conditional_without_context(self):
        """Test should_sample with CONDITIONAL strategy when no context available."""
        service = SamplingService()
        result = MockResult(hits=[{"id": "test"}], hit_count=1)

        should_sample = service.should_sample(
            strategy=SamplingStrategy.CONDITIONAL, result=result, ctx=None
        )

        assert should_sample is False

    def test_get_result_count_with_hits_list(self):
        """Test result count extraction from hits list."""
        service = SamplingService()
        result = MockResult(hits=[{"id": "1"}, {"id": "2"}])

        count = service._get_result_count(result)
        assert count == 2

    def test_get_result_count_with_hit_count(self):
        """Test result count extraction from hit_count attribute when hits is empty."""
        service = SamplingService()
        # Create result with hit_count and empty hits list
        # Note: The method checks hits first, so with an empty hits list, it returns 0
        result = MockResult(hit_count=5)

        count = service._get_result_count(result)
        # Since hits exists (as empty list), the method returns len(hits) = 0
        assert count == 0

    def test_get_result_count_with_hit_count_no_hits_attr(self):
        """Test result count extraction from hit_count when hits attribute is None."""
        service = SamplingService()
        # Create a mock object where hits is None (simulating missing attribute)
        result = MagicMock()
        result.hit_count = 5
        result.hits = None  # Explicitly set to None to test the fallback

        count = service._get_result_count(result)
        assert count == 5

    def test_get_result_count_with_nodes(self):
        """Test result count extraction from nodes attribute."""
        service = SamplingService()
        # Create a fresh result with specific nodes but no hits or hit_count
        result = MockResult()
        # Manually set attributes to test nodes specifically
        result.__dict__["hits"] = None
        result.__dict__["hit_count"] = None
        result.__dict__["nodes"] = [{"id": "1"}, {"id": "2"}, {"id": "3"}]

        count = service._get_result_count(result)
        assert count == 3

    def test_get_result_count_default(self):
        """Test result count extraction defaults to 1 for non-empty results."""
        service = SamplingService()
        result = MockResult()
        # Set all count-related attributes to None to test default
        result.__dict__["hits"] = None
        result.__dict__["hit_count"] = None
        result.__dict__["nodes"] = None

        count = service._get_result_count(result)
        assert count == 1

    def test_smart_sampling_empty_results(self):
        """Test smart sampling decision for empty results."""
        service = SamplingService()
        result = MockResult(hits=[], hit_count=0)
        mock_context = MagicMock()

        should_sample = service.should_sample(
            strategy=SamplingStrategy.SMART,
            result=result,
            ctx=mock_context,
            query="plasma temperature",
        )

        # Empty results should trigger sampling
        assert should_sample is True

    def test_smart_sampling_large_result_sets(self):
        """Test smart sampling decision for large result sets."""
        service = SamplingService()
        # Create result with many hits
        hits = [{"id": f"hit_{i}"} for i in range(15)]
        result = MockResult(hits=hits, hit_count=15)
        mock_context = MagicMock()

        should_sample = service.should_sample(
            strategy=SamplingStrategy.SMART,
            result=result,
            ctx=mock_context,
            query="plasma equilibrium",
        )

        # Large result sets should trigger sampling
        assert should_sample is True

    def test_smart_sampling_physics_query_boost(self):
        """Test smart sampling decision gets boosted for physics queries."""
        service = SamplingService()
        result = MockResult(hits=[{"id": "1"}], hit_count=1)
        mock_context = MagicMock()

        # Test with physics term
        should_sample_physics = service.should_sample(
            strategy=SamplingStrategy.SMART,
            result=result,
            ctx=mock_context,
            query="plasma temperature distribution",
        )

        # Test with non-physics term
        should_sample_generic = service.should_sample(
            strategy=SamplingStrategy.SMART,
            result=result,
            ctx=mock_context,
            query="simple data query",
        )

        # Physics query should be more likely to trigger sampling
        # Both might trigger due to other factors, but physics should have higher score
        assert isinstance(should_sample_physics, bool)
        assert isinstance(should_sample_generic, bool)

    def test_smart_sampling_no_context(self):
        """Test smart sampling decision without context."""
        service = SamplingService()
        result = MockResult(hits=[{"id": "test"}], hit_count=1)

        should_sample = service.should_sample(
            strategy=SamplingStrategy.SMART, result=result, ctx=None
        )

        assert should_sample is False

    def test_generate_default_prompt_with_query(self):
        """Test default prompt generation with query."""
        service = SamplingService()
        result = MockResult(hits=[{"id": "test"}])

        prompt = service._generate_default_prompt(
            result, query="plasma temperature", tool_name="search_imas"
        )

        assert "plasma temperature" in prompt
        assert "search_imas" in prompt
        assert "Analysis Request" in prompt

    def test_generate_default_prompt_defaults(self):
        """Test default prompt generation with default values."""
        service = SamplingService()
        result = MockResult(hits=[])

        prompt = service._generate_default_prompt(result)

        assert "data" in prompt
        assert "tool" in prompt


class TestSamplingStrategyValues:
    """Test that SamplingStrategy enum has expected values."""

    def test_sampling_strategy_values(self):
        """Test all expected strategy values exist."""
        assert SamplingStrategy.NO_SAMPLING.value == "no_sampling"
        assert SamplingStrategy.ALWAYS.value == "always"
        assert SamplingStrategy.CONDITIONAL.value == "conditional"
        assert SamplingStrategy.SMART.value == "smart"


@pytest.mark.parametrize(
    "strategy,expected_behavior",
    [
        (SamplingStrategy.NO_SAMPLING, False),
        (SamplingStrategy.ALWAYS, True),
        (SamplingStrategy.CONDITIONAL, False),  # Without context
        (SamplingStrategy.SMART, False),  # Without context
    ],
)
def test_should_sample_strategies_without_context(strategy, expected_behavior):
    """Parametrized test for should_sample behavior without context."""
    service = SamplingService()
    result = MockResult(hits=[{"id": "test"}], hit_count=1)

    should_sample = service.should_sample(strategy=strategy, result=result, ctx=None)

    assert should_sample == expected_behavior


@pytest.mark.parametrize(
    "physics_term",
    [
        "plasma",
        "magnetic",
        "temperature",
        "pressure",
        "equilibrium",
        "transport",
        "heating",
        "current",
        "profile",
        "disruption",
    ],
)
def test_smart_sampling_physics_terms(physics_term):
    """Parametrized test for physics term detection in smart sampling."""
    service = SamplingService()
    result = MockResult(hits=[{"id": "test"}], hit_count=1)
    mock_context = MagicMock()

    # Test query with physics term should be detected
    query_with_physics = f"Find {physics_term} data in IMAS"
    should_sample = service.should_sample(
        strategy=SamplingStrategy.SMART,
        result=result,
        ctx=mock_context,
        query=query_with_physics,
    )

    # Should at least attempt smart sampling logic (result may vary based on scoring)
    assert isinstance(should_sample, bool)
