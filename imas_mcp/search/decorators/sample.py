"""
Sample decorator for ToolResult sampling.

Provides AI-powered content sampling on tool results using tool-defined sampling tasks.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from fastmcp import Context
from mcp.types import TextContent

from ...models.result_models import ToolResult


class PromptBuilder(Protocol):
    """Protocol for objects that can build AI prompts."""

    def build_prompt(self, prompt_type: str, tool_context: dict[str, Any]) -> str:
        """Build an AI prompt for the given type and context."""
        ...

    def system_prompt(self) -> str:
        """Get the system prompt for this tool's AI sampling."""
        ...

    def build_sample_tasks(self, tool_result: ToolResult) -> list[dict[str, Any]]:
        """Build sampling tasks for sampling tool result content.

        Returns:
            List of sampling tasks with field, prompt_type, and context
        """
        ...


F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


async def apply_ai_sampling(
    tool_result: ToolResult,  # Any model that inherits from ToolResult
    ctx: Context | None,
    tool_instance: PromptBuilder | None = None,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> ToolResult:
    """
    Apply AI content sampling using tool-defined sampling tasks.

    Args:
        tool_result: The ToolResult object to sample
        ctx: FastMCP Context object for AI interaction
        tool_instance: Tool instance implementing PromptBuilder protocol for prompt building
        temperature: AI temperature setting
        max_tokens: Maximum tokens for AI response

    Returns:
        ToolResult with sampled content
    """
    try:
        # Check if we have a FastMCP context with sampling capability
        if not ctx or not hasattr(ctx, "sample") or not tool_instance:
            logger.debug(
                f"Sampling preconditions not met - ctx: {ctx}, has sample: {hasattr(ctx, 'sample') if ctx else False}, tool_instance: {type(tool_instance).__name__ if tool_instance else None}"
            )
            return tool_result

        # Get tool-specific system prompt
        system_prompt = "You are an expert assistant analyzing IMAS fusion physics data. Provide detailed, accurate insights."
        if hasattr(tool_instance, "system_prompt"):
            try:
                custom_system_prompt = tool_instance.system_prompt()
                if custom_system_prompt and isinstance(custom_system_prompt, str):
                    system_prompt = custom_system_prompt
                    logger.debug(
                        f"Using custom system prompt from tool instance (length: {len(system_prompt)})"
                    )
            except Exception as e:
                logger.warning(f"Failed to get tool system prompt: {e}")

        # Get tool-defined sampling tasks
        sampling_tasks = []
        if hasattr(tool_instance, "build_sample_tasks"):
            try:
                sampling_tasks = tool_instance.build_sample_tasks(tool_result)
                logger.debug(
                    f"Built {len(sampling_tasks)} sampling tasks from tool instance"
                )
            except Exception as e:
                logger.warning(f"Failed to build sample tasks: {e}")

        if not sampling_tasks:
            logger.debug("No sampling tasks defined, skipping AI sampling")
            return tool_result

        # Apply each sampling task
        sampled_fields = []
        for task in sampling_tasks:
            try:
                field_name = task.get("field")
                prompt_type = task.get("prompt_type")
                context = task.get("context", {})

                if not field_name or not prompt_type:
                    logger.warning(f"Invalid sampling task: {task}")
                    continue

                # Build prompt using tool's build_prompt method
                prompt = tool_instance.build_prompt(prompt_type, context)
                if not prompt:
                    logger.warning(f"No prompt generated for task {field_name}")
                    continue

                logger.debug(
                    f"Sampling field '{field_name}' with prompt type '{prompt_type}'"
                )

                # Sample with tool's system prompt
                response = await ctx.sample(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                if isinstance(response, TextContent) and response.text:
                    # Apply sampling using tool-specific field handler
                    handler_name = f"_apply_{field_name}_sampling"
                    if hasattr(tool_instance, handler_name):
                        handler = getattr(tool_instance, handler_name)
                        handler(tool_result, response.text)
                        logger.debug(
                            f"Applied custom sampling handler for {field_name}"
                        )
                        sampled_fields.append(field_name)
                    else:
                        logger.warning(
                            f"No sampling handler found for field '{field_name}' - skipping"
                        )

            except Exception as e:
                logger.warning(f"Failed to sample {task.get('field', 'unknown')}: {e}")

        # Record sampling metadata
        if sampled_fields:
            tool_result.ai_prompt = {
                "type": "content_sampling",
                "sampled_fields": ", ".join(sampled_fields),
                "temperature": str(temperature),
                "max_tokens": str(max_tokens),
                "system_prompt_source": type(tool_instance).__name__,
            }

            tool_result.ai_response = {
                "status": "content_sampled",
                "sampled_fields": sampled_fields,
                "task_count": len(sampling_tasks),
            }
            logger.debug(f"AI sampling successful for fields: {sampled_fields}")
        else:
            logger.debug("No fields were successfully sampled")

    except Exception as e:
        logger.warning(f"AI content sampling failed: {e}")
        tool_result.ai_response = {"status": "sampling_error", "reason": str(e)}

    return tool_result


def sample(temperature: float = 0.3, max_tokens: int = 800) -> Callable[[F], F]:
    """
    Decorator to add AI sampling to any ToolResult.

    Args:
        temperature: AI temperature setting (0.0-1.0)
        max_tokens: Maximum tokens for AI response

    Returns:
        Decorated function with AI sampling applied to ToolResult objects
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
            logger.debug(f"ctx type: {type(ctx)}")
            if ctx:
                logger.debug(f"ctx has sample method: {hasattr(ctx, 'sample')}")

            # Get tool instance (self) from args - should be first argument in method call
            tool_instance: PromptBuilder | None = args[0] if args else None
            logger.debug(
                f"Tool instance extracted: {type(tool_instance).__name__ if tool_instance else 'None'}"
            )
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

            # Apply sampling if result is a ToolResult and context is available
            logger.debug(
                f"Checking sampling conditions - isinstance ToolResult: {isinstance(result, ToolResult)}, ctx is not None: {ctx is not None}"
            )
            logger.debug(f"Result type: {type(result)}")
            logger.debug(
                f"Result MRO: {[cls.__name__ for cls in type(result).__mro__]}"
            )

            if isinstance(result, ToolResult) and ctx is not None:
                logger.debug("Applying AI sampling...")
                result = await apply_ai_sampling(
                    result, ctx, tool_instance, temperature, max_tokens
                )
            else:
                logger.warning("Sampling skipped - conditions not met")

            return result

        return wrapper  # type: ignore

    return decorator
