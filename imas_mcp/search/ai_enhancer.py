"""
AI Enhancement Module for IMAS MCP Server.

This module provides a decorator for AI-powered enhancement of MCP tool responses
using MCP context sampling when available.
"""

import json
import logging
from functools import wraps
from typing import Any, Callable, Dict, cast

from mcp.types import TextContent

logger = logging.getLogger(__name__)


# AI prompts focused on specific tasks with relationship awareness
SEARCH_EXPERT = """You are an IMAS search expert with deep knowledge of data relationships. Analyze relevance-ranked search results and provide:
1. 5 related search terms for plasma physics research, considering cross-references and physics concepts
2. Brief physics insights about the found data paths and their measurement context
3. Suggestions for complementary searches based on measurement relationships and connected IDS

The search results are ordered by relevance considering exact matches, path position, 
documentation content, path specificity, physics concepts, and cross-reference connectivity. 
Focus on practical physics relationships and measurement considerations that would lead to 
valuable follow-up searches using the rich relationship network."""

EXPLANATION_EXPERT = """You are an IMAS physics expert providing clear explanations of plasma physics concepts. Provide:
1. Physics significance and context from physics concepts database
2. Domain-specific explanations appropriate for the detail level requested  
3. Related IMAS paths and measurement considerations
4. Practical applications and measurement scenarios

Focus on accurate physics explanations that connect theoretical concepts to practical 
IMAS data structures and measurement workflows."""

OVERVIEW_EXPERT = """You are an IMAS data expert providing comprehensive overviews. Focus on:
1. Clear explanations of IMAS structure and physics domains
2. Practical guidance for researchers and data analysts
3. Navigation strategies for finding relevant data
4. Physics context and measurement workflows
Provide actionable insights for effective IMAS usage."""

STRUCTURE_EXPERT = """You are an IMAS structural analysis expert. Analyze IDS organization and provide:
1. Physics domain context and measurement scope
2. Data hierarchy and organization logic
3. Critical measurement paths and their physics significance
4. Practical usage guidance for researchers
5. Integration patterns with other IDS
Focus on actionable insights for data analysis workflows."""

RELATIONSHIP_EXPERT = """You are an IMAS relationship analysis expert. Analyze connection patterns and provide:
1. Physics-based relationship explanations for the discovered connections
2. Measurement workflow insights showing how data paths connect in practice
3. Practical navigation suggestions for related data exploration

Focus on quantitative insights with physics context, emphasizing how the relationship 
network reflects actual plasma measurement dependencies and physics coupling."""


def ai_enhancer(
    system_prompt: str,
    operation_name: str,
    temperature: float = 0.3,
    max_tokens: int = 800,
):
    """
    Decorator for AI enhancement of MCP tool responses.

    Automatically adds AI insights to tool responses when MCP context is available.
    Falls back gracefully when context is unavailable.

    Args:
        system_prompt: The system prompt defining AI behavior for this tool
        operation_name: Name of the operation for logging
        temperature: Sampling temperature for AI response
        max_tokens: Maximum tokens for AI response

    Usage:
        @ai_enhancer(SEARCH_EXPERT, "Search analysis", temperature=0.3)
        async def search_imas(self, query: str, ctx: Optional[Context] = None):
            # Tool implementation returns base response
            results = self.perform_search(query)

            # If ctx is available, decorator will add AI insights
            return {
                "results": results,
                "ai_prompt": f"Analyze search for: {query}\\nResults: {results}"
            }
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # Extract ctx from kwargs
            ctx = kwargs.get("ctx")

            # Call the original tool function
            base_response = await func(*args, **kwargs)

            # If no context or no AI prompt in response, return as-is
            if (
                not ctx
                or not isinstance(base_response, dict)
                or "ai_prompt" not in base_response
            ):
                # Remove ai_prompt from response if present
                if isinstance(base_response, dict) and "ai_prompt" in base_response:
                    ai_insights = {"error": "AI enhancement temporarily unavailable"}
                    base_response["ai_insights"] = ai_insights
                    del base_response["ai_prompt"]
                return base_response

            # Extract AI prompt and remove from response
            ai_prompt = base_response.pop("ai_prompt")

            try:
                ai_response = await ctx.sample(
                    ai_prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                if ai_response:
                    text_content = cast(TextContent, ai_response)
                    try:
                        ai_insights = json.loads(text_content.text)
                    except json.JSONDecodeError:
                        ai_insights = {"response": text_content.text}
                else:
                    ai_insights = {"error": "No AI response received"}

            except Exception as e:
                logger.warning(f"{operation_name} failed: {e}")
                ai_insights = {"error": "AI enhancement temporarily unavailable"}

            # Add AI insights to the response
            base_response["ai_insights"] = ai_insights
            return base_response

        return wrapper

    return decorator
