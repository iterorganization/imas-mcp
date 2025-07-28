"""
Sampling decorator for enriching search results.

Provides AI-powered insights and contextual information sampling.
"""

import functools
import logging
from typing import Any, Callable, Dict, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


async def apply_sample(
    sample_prompt: str, ctx: Any, temperature: float = 0.3, max_tokens: int = 800
) -> Dict[str, Any]:
    """
    Apply sampling to search results.

    Args:
        sample_prompt: The prompt to send to AI
        ctx: MCP context for AI interaction
        temperature: AI temperature setting
        max_tokens: Maximum tokens for AI response

    Returns:
        Sampling result
    """
    try:
        if not ctx or not hasattr(ctx, "session"):
            return {"status": "unavailable", "reason": "No AI context available"}

        # Check if AI is available in the session
        if not hasattr(ctx.session, "create_message"):
            return {"status": "unavailable", "reason": "AI not available in session"}

        # Create AI message
        response = await ctx.session.create_message(
            messages=[{"role": "user", "content": sample_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if hasattr(response, "content") and response.content:
            content = (
                response.content[0].text
                if hasattr(response.content[0], "text")
                else str(response.content[0])
            )
            return {
                "status": "success",
                "content": content,
                "prompt_used": sample_prompt,
                "settings": {"temperature": temperature, "max_tokens": max_tokens},
            }

        return {"status": "empty", "reason": "AI returned empty response"}

    except Exception as e:
        logger.warning(f"Sampling failed: {e}")
        return {"status": "error", "reason": str(e)}


def sample(temperature: float = 0.3, max_tokens: int = 800) -> Callable[[F], F]:
    """
    Decorator to add sampling to function results.

    Args:
        temperature: AI temperature setting (0.0-1.0)
        max_tokens: Maximum tokens for AI response

    Returns:
        Decorated function with sampling
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract context
            ctx = kwargs.get("ctx")

            # Execute original function
            result = await func(*args, **kwargs)

            # Only sample successful results that have sample prompts
            if (
                isinstance(result, dict)
                and "error" not in result
                and "sample_prompt" in result
                and ctx is not None
            ):
                # Apply sampling
                sampling = await apply_sample(
                    result["sample_prompt"], ctx, temperature, max_tokens
                )

                # Add sampling to result
                result["sample_insights"] = sampling

                # Remove the prompt from final result to keep it clean
                result.pop("sample_prompt", None)

            return result

        return wrapper

    return decorator


def build_search_sample_prompt(query: str, results: list, max_results: int = 3) -> str:
    """
    Build sample prompt for search result sampling.

    Args:
        query: Original search query
        results: Search results list
        max_results: Maximum results to include in prompt

    Returns:
        Sample prompt
    """
    if not results:
        return f"""Search Query Analysis: "{query}"

No results were found for this query in the IMAS data dictionary.

Please provide:
1. Suggestions for alternative search terms or queries
2. Possible related IMAS concepts or data paths
3. Common physics contexts where this term might appear
4. Recommended follow-up searches"""

    # Limit results for prompt
    top_results = results[:max_results]

    # Build results summary
    results_summary = []
    for i, result in enumerate(top_results, 1):
        if isinstance(result, dict):
            path = result.get("path", "Unknown path")
            doc = result.get("documentation", "")[:100]
            score = result.get("relevance_score", 0)
            results_summary.append(f"{i}. {path} (score: {score:.2f})")
            if doc:
                results_summary.append(f"   Documentation: {doc}...")
        else:
            results_summary.append(f"{i}. {str(result)[:100]}")

    prompt = f"""Search Results Analysis for: "{query}"

Found {len(results)} relevant paths in IMAS data dictionary.

Top results:
{chr(10).join(results_summary)}

Please provide enhanced analysis including:
1. Physics context and significance of these paths
2. Recommended follow-up searches or related concepts  
3. Data usage patterns and common workflows
4. Validation considerations for these measurements
5. Brief explanation of how these paths relate to the query"""

    return prompt


def build_concept_sample_prompt(
    concept: str, detail_level: str = "intermediate"
) -> str:
    """
    Build sample prompt for concept explanation sampling.

    Args:
        concept: Concept to explain
        detail_level: Level of detail (basic, intermediate, advanced)

    Returns:
        Sample prompt
    """
    level_instructions = {
        "basic": "Provide a basic, accessible explanation suitable for students or newcomers",
        "intermediate": "Provide a detailed explanation with technical context and examples",
        "advanced": "Provide an in-depth analysis with mathematical details and research context",
    }

    instruction = level_instructions.get(
        detail_level, level_instructions["intermediate"]
    )

    return f"""IMAS Concept Explanation: "{concept}"

{instruction}.

Please explain:
1. What this concept means in the context of IMAS and fusion physics
2. How it relates to plasma physics and tokamak operations
3. Where this concept appears in IMAS data structures
4. Practical applications and measurement considerations
5. Related concepts and cross-references in IMAS

Focus on practical understanding for researchers working with IMAS data."""
