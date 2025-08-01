"""Sampling service for LLM client integration."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel
from .base import BaseService
from ..models.response_models import AIResponse


class SamplingStrategy(Enum):
    """Sampling strategy options."""

    NO_SAMPLING = "no_sampling"
    ALWAYS = "always"
    CONDITIONAL = "conditional"
    SMART = "smart"


class SamplingService(BaseService):
    """Service for LLM client sampling operations."""

    async def apply_sampling(
        self,
        result: AIResponse,
        strategy: SamplingStrategy = SamplingStrategy.SMART,
        sample_prompt: Optional[str] = None,
        ctx: Optional[Any] = None,
        temperature: float = 0.3,
        max_tokens: int = 800,
        **kwargs,
    ) -> AIResponse:
        """
        Apply LLM sampling to generate insights.

        Replaces the functionality of @sample decorator with service-based approach.
        """
        if strategy == SamplingStrategy.NO_SAMPLING:
            return result

        if not self.should_sample(strategy, result, ctx, **kwargs):
            return result

        try:
            if not ctx:
                self.logger.debug("No MCP context available for sampling")
                return result

            # Use provided prompt or generate one
            if not sample_prompt:
                sample_prompt = self._generate_default_prompt(result, **kwargs)

            # Sample client
            if hasattr(ctx, "sample"):
                sampled_content = await ctx.sample(
                    prompt=sample_prompt, temperature=temperature, max_tokens=max_tokens
                )
            else:
                self.logger.debug("MCP sampling not available in context")
                return result

            # Attach sampling result to the original result
            result.ai_insights = sampled_content

            self.logger.debug("Sampling applied successfully")
            return result

        except Exception as e:
            self.logger.warning(f"Sampling failed: {e}")
            return result

    def should_sample(
        self,
        strategy: SamplingStrategy,
        result: BaseModel,
        ctx: Optional[Any] = None,
        **kwargs,
    ) -> bool:
        """
        Decide whether to apply sampling based on strategy and context.
        """
        if strategy == SamplingStrategy.NO_SAMPLING:
            return False
        elif strategy == SamplingStrategy.ALWAYS:
            return True
        elif strategy == SamplingStrategy.CONDITIONAL:
            return ctx is not None
        elif strategy == SamplingStrategy.SMART:
            return self._smart_sampling_decision(result, ctx, **kwargs)

        return False

    def _smart_sampling_decision(
        self, result: BaseModel, ctx: Optional[Any], **kwargs
    ) -> bool:
        """Make intelligent sampling decisions based on multiple factors."""
        if not ctx:
            return False

        sampling_score = 0.0

        # Check if sampling should be applied
        result_count = self._get_result_count(result)

        if result_count == 0:
            sampling_score += 0.3  # Empty results need explanation
        elif result_count > 10:
            sampling_score += 0.2  # Large result sets benefit from summarization
        elif 1 <= result_count <= 5:
            sampling_score += 0.4  # Small result sets are good candidates

        # Physics context boost
        query = kwargs.get("query", "")
        if isinstance(query, str):
            physics_terms = [
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
            ]
            if any(term in query.lower() for term in physics_terms):
                sampling_score += 0.2

        return sampling_score >= 0.4

    def _get_result_count(self, result: BaseModel) -> int:
        """Extract result count from various result structures."""
        # Use getattr to safely access dynamic attributes
        hits = getattr(result, "hits", [])
        if hits is not None and hasattr(hits, "__len__"):
            return len(hits)

        hit_count = getattr(result, "hit_count", None)
        if hit_count is not None:
            return hit_count

        nodes = getattr(result, "nodes", [])
        if nodes is not None and hasattr(nodes, "__len__"):
            return len(nodes)

        return 1  # Default to 1 for non-empty results

    def _generate_default_prompt(self, result: BaseModel, **kwargs) -> str:
        """Generate a default sampling prompt based on result type."""
        query = kwargs.get("query", "data")
        tool_name = kwargs.get("tool_name", "tool")

        return f"""Analysis Request for {tool_name} Result

Query: "{query}"

Please provide analysis and insights for this {tool_name} result.

Include:
1. Context and significance of the results
2. Recommended follow-up actions
3. Relevant IMAS data patterns
4. Physics context where applicable

No results were found for this query in the IMAS data dictionary.

Please provide:
1. Suggestions for alternative search terms or queries
2. Possible related IMAS concepts or data paths
3. Common physics contexts where this term might appear
4. Recommended follow-up searches"""
