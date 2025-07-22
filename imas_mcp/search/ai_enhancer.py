"""
AI Enhancement Module for IMAS MCP Server.

This module provides a decorator for AI-powered enhancement of MCP tool responses
using MCP context sampling when available. Implements selective AI enhancement
strategy for optimal performance.
"""

import json
import logging
from functools import wraps
from typing import Any, Callable, Dict, cast

from mcp.types import TextContent

from .ai_enhancement_strategy import (
    should_apply_ai_enhancement,
    suggest_follow_up_tools,
)

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

BULK_EXPORT_EXPERT = """You are an IMAS bulk data export expert. Analyze export selections and provide:
1. Data usage recommendations for the exported datasets
2. Physics context explaining relationships between selected IDS or data paths
3. Suggested analysis workflows utilizing the exported data
4. Integration patterns and measurement dependencies
5. Quality considerations and data validation approaches

Focus on practical guidance for researchers working with bulk IMAS data exports,
emphasizing physics-aware analysis workflows and cross-IDS relationships."""

PHYSICS_DOMAIN_EXPERT = """You are an IMAS physics domain expert. Analyze domain-specific data and provide:
1. Comprehensive physics context for the selected domain
2. Key measurement relationships and dependencies within the domain
3. Suggested analysis approaches for domain-specific research
4. Integration patterns with other physics domains
5. Practical considerations for domain-focused data analysis

Focus on domain-specific physics insights, measurement workflows, and research guidance
that leverages the rich IMAS data structure for comprehensive plasma analysis."""


def ai_enhancer(
    system_prompt: str,
    operation_name: str,
    temperature: float = 0.3,
    max_tokens: int = 800,
):
    """
    Decorator for selective AI enhancement of MCP tool responses.

    Automatically adds AI insights to tool responses when MCP context is available
    and when the selective enhancement strategy determines AI would be valuable.
    Falls back gracefully when context is unavailable or enhancement is not needed.

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

            # Add tool suggestions and AI prompt for conditional enhancement
            base_response = {
                "results": results,
                "suggested_tools": suggest_follow_up_tools(results, "search_imas"),
                "ai_prompt": f"Analyze search for: {query}\\nResults: {results}"
            }

            # Decorator will apply AI enhancement based on selective strategy
            return base_response
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # Extract ctx from kwargs
            ctx = kwargs.get("ctx")

            # Call the original tool function
            base_response = await func(*args, **kwargs)

            # Always add tool suggestions if not present
            if (
                isinstance(base_response, dict)
                and "suggested_tools" not in base_response
            ):
                try:
                    suggestions = suggest_follow_up_tools(base_response, func.__name__)
                    base_response["suggested_tools"] = suggestions
                except Exception as e:
                    logger.warning(f"Failed to generate tool suggestions: {e}")
                    base_response["suggested_tools"] = []

            # Check if AI enhancement should be applied using selective strategy
            should_enhance = should_apply_ai_enhancement(
                func.__name__, args, kwargs, ctx
            )

            # If no context, no AI prompt, or selective strategy says no enhancement
            if (
                not ctx
                or not isinstance(base_response, dict)
                or "ai_prompt" not in base_response
                or not should_enhance
            ):
                # Remove ai_prompt from response if present and add status
                if isinstance(base_response, dict) and "ai_prompt" in base_response:
                    if not ctx:
                        ai_insights = {"status": "AI context not available"}
                    elif not should_enhance:
                        ai_insights = {
                            "status": "AI enhancement not needed for this request"
                        }
                    else:
                        ai_insights = {
                            "status": "AI enhancement temporarily unavailable"
                        }

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
                        ai_insights["status"] = "AI enhancement applied"
                    except json.JSONDecodeError:
                        ai_insights = {
                            "response": text_content.text,
                            "status": "AI enhancement applied (unstructured)",
                        }
                else:
                    ai_insights = {"error": "No AI response received"}

            except Exception as e:
                logger.warning(f"{operation_name} AI enhancement failed: {e}")
                ai_insights = {"error": "AI enhancement temporarily unavailable"}

            # Add AI insights to the response
            base_response["ai_insights"] = ai_insights
            return base_response

        return wrapper

    return decorator
