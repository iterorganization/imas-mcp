"""
Sample decorator for SearchResult enhancement.

Provides AI-powered sampling on search results with configurable prompts.
"""

import functools
import logging
from typing import Any, Callable, TypeVar

from ...models.result_models import SearchResult

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


async def apply_ai_sampling(
    search_result: SearchResult,
    ctx: Any,
    tool_instance: Any = None,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> SearchResult:
    """
    Apply AI sampling to a SearchResult using tool-specific prompt building.

    Args:
        search_result: The SearchResult to enhance
        ctx: MCP context for AI interaction
        tool_instance: Tool instance for prompt building
        temperature: AI temperature setting
        max_tokens: Maximum tokens for AI response

    Returns:
        Enhanced SearchResult with ai_response
    """
    try:
        if not ctx or not hasattr(ctx, "session"):
            logger.debug("No AI context available for sampling")
            return search_result

        # Check if AI is available in the session
        if not hasattr(ctx.session, "create_message"):
            logger.debug("AI not available in session")
            return search_result

        # Try to get prompt from SearchResult's ai_prompt field first
        prompt = search_result.ai_prompt

        # If no ai_prompt and we have a tool instance, build tool-specific prompt
        if not prompt and tool_instance and hasattr(tool_instance, "build_prompt"):
            tool_context = {
                "query": search_result.query,
                "results": search_result.hits,
                "hit_count": search_result.hit_count,
                "search_mode": getattr(search_result, "search_mode", "auto"),
            }

            # Determine prompt type based on results
            prompt_type = (
                "search_analysis" if search_result.hit_count > 0 else "no_results"
            )
            prompt = tool_instance.build_prompt(prompt_type, tool_context)

        if not prompt:
            logger.debug("No AI prompt available for sampling")
            return search_result

        # Create AI message
        response = await ctx.session.create_message(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if hasattr(response, "content") and response.content:
            content = (
                response.content[0].text
                if hasattr(response.content[0], "text")
                else str(response.content[0])
            )

            # Update the SearchResult with AI response
            search_result.ai_response = {
                "status": "success",
                "content": content,
                "prompt_used": prompt,
                "settings": {"temperature": temperature, "max_tokens": max_tokens},
            }
        else:
            search_result.ai_response = {
                "status": "empty",
                "reason": "AI returned empty response",
            }

    except Exception as e:
        logger.warning(f"AI sampling failed: {e}")
        search_result.ai_response = {"status": "error", "reason": str(e)}

    return search_result


def sample(temperature: float = 0.3, max_tokens: int = 800) -> Callable[[F], F]:
    """
    Decorator to add AI sampling to SearchResult.

    Args:
        temperature: AI temperature setting (0.0-1.0)
        max_tokens: Maximum tokens for AI response

    Returns:
        Decorated function with AI sampling applied to SearchResult
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract context
            ctx = kwargs.get("ctx")

            # Get tool instance (self) from args
            tool_instance = args[0] if args else None

            # Execute original function
            result = await func(*args, **kwargs)

            # Apply sampling if result is SearchResult and context is available
            if isinstance(result, SearchResult) and ctx is not None:
                result = await apply_ai_sampling(
                    result, ctx, tool_instance, temperature, max_tokens
                )

            return result

        return wrapper  # type: ignore

    return decorator
