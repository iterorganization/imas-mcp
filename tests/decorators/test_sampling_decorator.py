"""
Unit tests for AI Sampling Decorator.

Tests the simplified AI sampling decorator that provides enhanced insights
and contextual information for search results and other tool outputs.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from imas_mcp.search.decorators.sampling import (
    apply_sample,
    sample,
    build_search_sample_prompt,
    build_concept_sample_prompt,
)


class TestSamplingCore:
    """Test core sampling functionality."""

    @pytest.mark.asyncio
    async def test_apply_sample_success(self):
        """Test successful AI sampling."""
        # Mock context with AI capabilities
        ctx = Mock()
        ctx.session = Mock()
        ctx.session.create_message = AsyncMock()

        # Mock AI response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "AI generated insights and analysis"
        ctx.session.create_message.return_value = mock_response

        # Test sampling
        result = await apply_sample(
            "Test prompt for physics analysis", ctx, temperature=0.5, max_tokens=500
        )

        # Verify result
        assert result["status"] == "success"
        assert result["content"] == "AI generated insights and analysis"
        assert result["prompt_used"] == "Test prompt for physics analysis"
        assert result["settings"]["temperature"] == 0.5
        assert result["settings"]["max_tokens"] == 500

        # Verify AI was called correctly
        ctx.session.create_message.assert_called_once_with(
            messages=[{"role": "user", "content": "Test prompt for physics analysis"}],
            temperature=0.5,
            max_tokens=500,
        )

    @pytest.mark.asyncio
    async def test_apply_sample_no_context(self):
        """Test sampling with no AI context."""
        result = await apply_sample("Test prompt", None)

        assert result["status"] == "unavailable"
        assert result["reason"] == "No AI context available"

    @pytest.mark.asyncio
    async def test_apply_sample_no_ai_session(self):
        """Test sampling with context but no AI session."""
        ctx = Mock()
        # Context without session attribute

        result = await apply_sample("Test prompt", ctx)

        assert result["status"] in ["unavailable", "error"]
        assert (
            "No AI context available" in result["reason"] or "Mock" in result["reason"]
        )

    @pytest.mark.asyncio
    async def test_apply_sample_ai_error(self):
        """Test sampling with AI error."""
        ctx = Mock()
        ctx.session = Mock()
        ctx.session.create_message = AsyncMock(
            side_effect=Exception("AI service error")
        )

        result = await apply_sample("Test prompt", ctx)

        assert result["status"] == "error"
        assert "AI service error" in result["reason"]


class TestSamplingDecorator:
    """Test sampling decorator functionality."""

    @pytest.mark.asyncio
    async def test_sample_decorator_success(self):
        """Test sampling decorator with successful AI enhancement."""
        # Mock context
        ctx = Mock()
        ctx.session = Mock()
        ctx.session.create_message = AsyncMock()

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Enhanced analysis with AI insights"
        ctx.session.create_message.return_value = mock_response

        # Create decorated function
        @sample(temperature=0.3, max_tokens=800)
        async def test_function(query: str, ctx=None):
            return {
                "results": ["plasma.temperature", "magnetic_field.b_tor"],
                "sample_prompt": f"Analyze search results for: {query}",
                "query": query,
            }

        # Test decorated function
        result = await test_function("plasma physics", ctx=ctx)

        # Verify sampling was applied
        assert "sample_insights" in result
        assert result["sample_insights"]["status"] == "success"
        assert (
            result["sample_insights"]["content"] == "Enhanced analysis with AI insights"
        )
        assert "sample_prompt" not in result  # Should be removed after sampling

    @pytest.mark.asyncio
    async def test_sample_decorator_no_prompt(self):
        """Test sampling decorator when result has no sample prompt."""

        @sample()
        async def test_function(query: str, ctx=None):
            return {
                "results": ["plasma.temperature"],
                "query": query,
                # No sample_prompt in result
            }

        result = await test_function("test", ctx=Mock())

        # Should return original result without sampling
        assert "sample_insights" not in result
        assert result["results"] == ["plasma.temperature"]

    @pytest.mark.asyncio
    async def test_sample_decorator_error_result(self):
        """Test sampling decorator with error result."""

        @sample()
        async def test_function(query: str, ctx=None):
            return {
                "error": "Something went wrong",
                "sample_prompt": "This should not be processed",
            }

        result = await test_function("test", ctx=Mock())

        # Should return original error without sampling
        assert "sample_insights" not in result
        assert result["error"] == "Something went wrong"


class TestPromptBuilding:
    """Test AI prompt building functions."""

    def test_build_search_sample_prompt_with_results(self):
        """Test building search sample prompt with results."""
        results = [
            {
                "path": "plasma.temperature.t_e",
                "documentation": "Electron temperature measurements in plasma core",
                "relevance_score": 0.95,
            },
            {
                "path": "magnetic_field.b_tor",
                "documentation": "Toroidal magnetic field strength",
                "relevance_score": 0.87,
            },
        ]

        prompt = build_search_sample_prompt("plasma temperature", results)

        assert "plasma temperature" in prompt
        assert "plasma.temperature.t_e" in prompt
        assert "magnetic_field.b_tor" in prompt
        assert "Physics context and significance" in prompt
        assert "Follow-up searches" in prompt or "follow-up searches" in prompt

    def test_build_search_sample_prompt_no_results(self):
        """Test building search sample prompt with no results."""
        prompt = build_search_sample_prompt("nonexistent concept", [])

        assert "nonexistent concept" in prompt
        assert "No results were found" in prompt
        assert "alternative search terms" in prompt
        assert "Related IMAS concepts" in prompt or "related IMAS concepts" in prompt

    def test_build_search_sample_prompt_max_results_limit(self):
        """Test that prompt respects max_results limit."""
        results = [{"path": f"test.path.{i}"} for i in range(10)]

        prompt = build_search_sample_prompt("test query", results, max_results=3)

        # Should only include first 3 results
        assert "test.path.0" in prompt
        assert "test.path.1" in prompt
        assert "test.path.2" in prompt
        assert "test.path.3" not in prompt

    def test_build_concept_sample_prompt_basic(self):
        """Test building concept explanation prompt at basic level."""
        prompt = build_concept_sample_prompt("tokamak", "basic")

        assert "tokamak" in prompt
        assert "basic, accessible explanation" in prompt
        assert "suitable for students" in prompt
        assert "fusion physics" in prompt

    def test_build_concept_sample_prompt_advanced(self):
        """Test building concept explanation prompt at advanced level."""
        prompt = build_concept_sample_prompt("magnetohydrodynamics", "advanced")

        assert "magnetohydrodynamics" in prompt
        assert "in-depth analysis" in prompt
        assert "mathematical details" in prompt
        assert "research context" in prompt

    def test_build_concept_sample_prompt_default_level(self):
        """Test building concept explanation prompt with default level."""
        prompt = build_concept_sample_prompt("plasma equilibrium")

        assert "plasma equilibrium" in prompt
        assert "detailed explanation" in prompt
        assert "technical context" in prompt


class TestSamplingIntegration:
    """Test sampling integration with tool workflows."""

    @pytest.mark.asyncio
    async def test_search_tool_sampling_workflow(self):
        """Test realistic sampling workflow for search tool."""
        # Mock context
        ctx = Mock()
        ctx.session = Mock()
        ctx.session.create_message = AsyncMock()

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = """
        The search results show plasma temperature measurement paths which are fundamental
        for understanding plasma confinement and energy transport in tokamaks.
        
        Key physics insights:
        1. Electron temperature (t_e) is critical for fusion reaction rates
        2. These measurements help validate transport models
        3. Temperature profiles indicate confinement quality
        
        Recommended follow-ups: search for 'temperature profile', 'heat transport'
        """
        ctx.session.create_message.return_value = mock_response

        # Simulate search tool with sampling
        @sample(temperature=0.3, max_tokens=800)
        async def mock_search_tool(query: str, ctx=None):
            # Simulate search results
            results = [
                {
                    "path": "plasma.temperature.t_e",
                    "relevance_score": 0.95,
                    "documentation": "Electron temperature",
                }
            ]

            return {
                "hits": results,
                "total": len(results),
                "query": query,
                "sample_prompt": build_search_sample_prompt(query, results),
            }

        # Execute search with sampling
        result = await mock_search_tool("plasma temperature", ctx=ctx)

        # Verify complete workflow
        assert result["hits"]
        assert result["query"] == "plasma temperature"
        assert "sample_insights" in result
        assert result["sample_insights"]["status"] == "success"
        assert "fusion reaction rates" in result["sample_insights"]["content"]
        assert "sample_prompt" not in result  # Cleaned up


class TestSamplingConfiguration:
    """Test sampling configuration and parameters."""

    @pytest.mark.asyncio
    async def test_temperature_parameter_propagation(self):
        """Test that temperature parameter is properly propagated."""
        ctx = Mock()
        ctx.session = Mock()
        ctx.session.create_message = AsyncMock()

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response"
        ctx.session.create_message.return_value = mock_response

        # Test with custom temperature
        result = await apply_sample(
            "Test prompt", ctx, temperature=0.7, max_tokens=1000
        )

        # Verify parameters were used
        ctx.session.create_message.assert_called_once_with(
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.7,
            max_tokens=1000,
        )

        assert result["settings"]["temperature"] == 0.7
        assert result["settings"]["max_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_decorator_parameter_defaults(self):
        """Test that decorator uses correct default parameters."""
        ctx = Mock()
        ctx.session = Mock()
        ctx.session.create_message = AsyncMock()

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response"
        ctx.session.create_message.return_value = mock_response

        @sample()  # Use defaults
        async def test_function(ctx=None):
            return {"sample_prompt": "Test prompt"}

        await test_function(ctx=ctx)

        # Verify default parameters
        ctx.session.create_message.assert_called_once_with(
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.3,  # Default
            max_tokens=800,  # Default
        )
