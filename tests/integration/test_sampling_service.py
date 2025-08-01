"""Integration tests for sampling service."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from imas_mcp.services.sampling import SamplingService, SamplingStrategy


class TestSamplingServiceIntegration:
    """Test sampling service integration with MCP context."""

    @pytest.fixture
    def sampling_service(self):
        return SamplingService()

    @pytest.fixture
    def mock_result(self):
        """Mock search result for testing."""
        result = MagicMock()
        result.hits = []
        result.hit_count = 0
        return result

    @pytest.mark.asyncio
    async def test_sampling_with_mcp_context(self, sampling_service, mock_result):
        """Test sampling service uses MCP context."""
        # Mock MCP context
        mock_context = MagicMock()
        mock_context.sample = AsyncMock(return_value="Sampled result")

        # Apply sampling with ALWAYS strategy to ensure sampling occurs
        result = await sampling_service.apply_sampling(
            result=mock_result,
            strategy=SamplingStrategy.ALWAYS,
            ctx=mock_context,
            temperature=0.5,
        )

        # Verify MCP context was used
        assert mock_context.sample.called
        # Result should be the original mock with insights attached
        assert result is mock_result
        assert hasattr(result, "ai_insights")
        assert result.ai_insights == "Sampled result"

    @pytest.mark.asyncio
    async def test_no_sampling_strategy(self, sampling_service, mock_result):
        """Test NO_SAMPLING strategy returns original result."""
        result = await sampling_service.apply_sampling(
            result=mock_result, strategy=SamplingStrategy.NO_SAMPLING
        )

        assert result == mock_result

    @pytest.mark.asyncio
    async def test_sampling_error_handling(self, sampling_service, mock_result):
        """Test sampling handles errors gracefully."""
        # Mock context to raise exception
        mock_context = MagicMock()
        mock_context.sample = AsyncMock(side_effect=Exception("Context error"))

        result = await sampling_service.apply_sampling(
            result=mock_result, strategy=SamplingStrategy.SMART, ctx=mock_context
        )

        # Should return original result on error
        assert result == mock_result

    def test_smart_sampling_decision(self, sampling_service):
        """Test smart sampling decision logic."""
        # Test with empty results (should sample)
        mock_empty_result = MagicMock()
        mock_empty_result.hits = []
        mock_context = MagicMock()

        should_sample = sampling_service.should_sample(
            SamplingStrategy.SMART,
            mock_empty_result,
            mock_context,
            query="plasma temperature",
        )

        assert should_sample

    def test_conditional_sampling_without_context(self, sampling_service):
        """Test conditional sampling without context."""
        mock_result = MagicMock()

        should_sample = sampling_service.should_sample(
            SamplingStrategy.CONDITIONAL, mock_result, None
        )

        assert not should_sample
