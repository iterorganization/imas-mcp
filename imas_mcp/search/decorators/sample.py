"""
Sample decorator for SearchResult enhancement.

Provides AI-powered sampling on search results with configurable prompts.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from fastmcp import Context
from mcp.types import TextContent

from ...models.result_models import SearchResult


class PromptBuilder(Protocol):
    """Protocol for objects that can build AI prompts."""

    def build_prompt(self, prompt_type: str, tool_context: dict[str, Any]) -> str:
        """Build an AI prompt for the given type and context."""
        ...

    def system_prompt(self) -> str:
        """Get the system prompt for this tool's AI sampling."""
        ...


F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


async def apply_ai_sampling(
    search_result: SearchResult,
    ctx: Context | None,
    tool_instance: PromptBuilder | None = None,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> SearchResult:
    """
    Apply AI sampling to a SearchResult using FastMCP context.

    Args:
        search_result: The SearchResult to enhance
        ctx: FastMCP Context object for AI interaction
        tool_instance: Tool instance implementing PromptBuilder protocol for prompt building
        temperature: AI temperature setting
        max_tokens: Maximum tokens for AI response

    Returns:
        Enhanced SearchResult with ai_response
    """
    try:
        # Check if we have a FastMCP context with sampling capability
        if not ctx or not hasattr(ctx, "sample"):
            logger.debug("No FastMCP context with sampling capability available")
            return search_result

        # Try to get prompt from SearchResult's ai_prompt field first
        prompt = None
        if search_result.ai_prompt:
            # ai_prompt is a Dict[str, str], get any available prompt
            prompt = next(iter(search_result.ai_prompt.values()), None)

        # If no ai_prompt and we have a tool instance, build tool-specific prompt
        if not prompt and tool_instance:
            logger.debug(
                f"No ai_prompt found, attempting to build prompt using tool instance: {type(tool_instance).__name__}"
            )

            # Verify tool instance implements PromptBuilder protocol
            if not (
                hasattr(tool_instance, "build_prompt")
                and callable(tool_instance.build_prompt)
            ):
                logger.warning(
                    f"Tool instance {type(tool_instance).__name__} does not implement build_prompt method"
                )
            else:
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
                logger.debug(
                    f"Building prompt with type: {prompt_type}, context keys: {list(tool_context.keys())}"
                )

                try:
                    prompt = tool_instance.build_prompt(prompt_type, tool_context)
                    logger.debug(
                        f"Successfully built prompt of length: {len(prompt) if prompt else 0}"
                    )
                except Exception as e:
                    logger.error(f"Failed to build prompt: {e}")
                    prompt = None

        # Ensure prompt is a string (not dict)
        if not isinstance(prompt, str):
            prompt = str(prompt)

        if not prompt:
            logger.debug("No AI prompt available for sampling")
            return search_result

        # Use FastMCP's ctx.sample method
        logger.debug(f"Calling ctx.sample with prompt length: {len(prompt)}")

        # Get tool-specific system prompt if available
        system_prompt = "You are an expert assistant analyzing IMAS fusion physics data. Provide detailed, accurate insights."
        if tool_instance and hasattr(tool_instance, "system_prompt"):
            try:
                custom_system_prompt = tool_instance.system_prompt()
                if custom_system_prompt and isinstance(custom_system_prompt, str):
                    system_prompt = custom_system_prompt
                    logger.debug(
                        f"Using custom system prompt from tool instance (length: {len(system_prompt)})"
                    )
                else:
                    logger.debug(
                        "Tool instance system_prompt returned empty or non-string value"
                    )
            except Exception as e:
                logger.warning(f"Failed to get tool system prompt: {e}")
        else:
            logger.debug(
                f"Using default system prompt - tool_instance: {tool_instance is not None}, has_system_prompt: {hasattr(tool_instance, 'system_prompt') if tool_instance else False}"
            )

        response = await ctx.sample(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Handle response - check if it's TextContent with .text attribute
        if isinstance(response, TextContent) and response.text:
            # Update the SearchResult with AI response
            search_result.ai_response = {
                "status": "success",
                "content": response.text,
                "prompt_used": prompt,
                "settings": {"temperature": temperature, "max_tokens": max_tokens},
            }
            logger.debug("AI sampling successful")
        else:
            search_result.ai_response = {
                "status": "empty",
                "reason": "AI returned empty or non-text response",
            }
            logger.debug("AI returned empty or non-text response")

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
            # Debug logging
            logger.debug(f"@sample decorator called for {func.__name__}")
            logger.debug(f"kwargs keys: {list(kwargs.keys())}")
            logger.debug(f"ctx in kwargs: {'ctx' in kwargs}")

            # Extract context parameter
            ctx = kwargs.get("ctx")
            logger.debug(f"ctx value: {ctx}")

            # Get tool instance (self) from args - should be first argument in method call
            tool_instance: PromptBuilder | None = args[0] if args else None
            logger.debug(
                f"Tool instance extracted: {type(tool_instance).__name__ if tool_instance else 'None'}"
            )
            logger.debug(
                f"Tool instance has build_prompt: {hasattr(tool_instance, 'build_prompt') if tool_instance else False}"
            )
            logger.debug(
                f"Tool instance has system_prompt: {hasattr(tool_instance, 'system_prompt') if tool_instance else False}"
            )

            # Execute original function
            result = await func(*args, **kwargs)

            # Apply sampling if result is SearchResult and context is available
            logger.debug(
                f"Checking sampling conditions - isinstance SearchResult: {isinstance(result, SearchResult)}, ctx is not None: {ctx is not None}"
            )
            if isinstance(result, SearchResult) and ctx is not None:
                logger.debug("Applying AI sampling...")
                result = await apply_ai_sampling(
                    result, ctx, tool_instance, temperature, max_tokens
                )
            else:
                logger.debug("Sampling skipped - conditions not met")

            return result

        return wrapper  # type: ignore

    return decorator
