"""
AI Enhancement Module for IMAS MCP Server.

This module provides a decorator for AI-powered enhancement of MCP tool responses
using MCP context sampling when available. Uses clear enhancement strategies
for optimal performance and maintainability.
"""

import json
import logging
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, cast

from mcp.types import TextContent

logger = logging.getLogger(__name__)


class EnhancementStrategy(Enum):
    """Enhancement strategies for AI processing."""

    ALWAYS = "always"
    NEVER = "never"
    CONDITIONAL = "conditional"


class ToolCategory(Enum):
    """Tool categories for enhancement decisions."""

    SEARCH = "search"
    EXPLANATION = "explanation"
    ANALYSIS = "analysis"
    EXPORT = "export"
    OVERVIEW = "overview"
    IDENTIFIERS = "identifiers"


# Configuration mapping tools to their enhancement strategies
TOOL_ENHANCEMENT_CONFIG = {
    "search_imas": {
        "strategy": EnhancementStrategy.CONDITIONAL,
        "category": ToolCategory.SEARCH,
    },
    "explain_concept": {
        "strategy": EnhancementStrategy.ALWAYS,
        "category": ToolCategory.EXPLANATION,
    },
    "get_overview": {
        "strategy": EnhancementStrategy.ALWAYS,
        "category": ToolCategory.OVERVIEW,
    },
    "analyze_ids_structure": {
        "strategy": EnhancementStrategy.CONDITIONAL,
        "category": ToolCategory.ANALYSIS,
    },
    "explore_relationships": {
        "strategy": EnhancementStrategy.CONDITIONAL,
        "category": ToolCategory.ANALYSIS,
    },
    "explore_identifiers": {
        "strategy": EnhancementStrategy.NEVER,
        "category": ToolCategory.IDENTIFIERS,
    },
    "export_ids": {
        "strategy": EnhancementStrategy.CONDITIONAL,
        "category": ToolCategory.EXPORT,
    },
    "export_physics_domain": {
        "strategy": EnhancementStrategy.CONDITIONAL,
        "category": ToolCategory.EXPORT,
    },
}


class EnhancementDecisionEngine:
    """
    Engine for making clear AI enhancement decisions.

    This class centralizes the logic for determining when to apply AI enhancement
    using a clear, switch-case-like structure for maintainability.
    """

    @staticmethod
    def should_enhance(tool_name: str, args: tuple, kwargs: dict, ctx: Any) -> bool:
        """
        Determine if AI enhancement should be applied.

        Args:
            tool_name: Name of the tool function
            args: Function arguments
            kwargs: Function keyword arguments
            ctx: MCP context for AI enhancement

        Returns:
            Boolean indicating whether to apply AI enhancement
        """
        # No enhancement without context
        if not ctx:
            return False

        # Get tool configuration
        config = TOOL_ENHANCEMENT_CONFIG.get(
            tool_name,
            {"strategy": EnhancementStrategy.ALWAYS, "category": ToolCategory.OVERVIEW},
        )

        strategy = config["strategy"]
        category = config["category"]

        # Switch-case style strategy decision
        match strategy:
            case EnhancementStrategy.NEVER:
                return False
            case EnhancementStrategy.ALWAYS:
                return True
            case EnhancementStrategy.CONDITIONAL:
                return EnhancementDecisionEngine._evaluate_conditional(
                    category, tool_name, args, kwargs
                )
            case _:
                return True  # Default to enhancement

    @staticmethod
    def _evaluate_conditional(
        category: ToolCategory, tool_name: str, args: tuple, kwargs: dict
    ) -> bool:
        """
        Evaluate conditional enhancement using clear category-based logic.

        Args:
            category: Tool category for enhancement logic
            tool_name: Name of the tool function
            args: Function arguments
            kwargs: Function keyword arguments

        Returns:
            Boolean indicating whether to apply enhancement
        """
        try:
            # Switch-case style conditional evaluation
            match category:
                case ToolCategory.SEARCH:
                    return EnhancementDecisionEngine._should_enhance_search(
                        args, kwargs
                    )
                case ToolCategory.ANALYSIS:
                    return EnhancementDecisionEngine._should_enhance_analysis(
                        tool_name, args, kwargs
                    )
                case ToolCategory.EXPORT:
                    return EnhancementDecisionEngine._should_enhance_export(
                        tool_name, args, kwargs
                    )
                case _:
                    return True  # Default to enhancement for unknown categories

        except Exception as e:
            logger.warning(
                f"Error evaluating conditional enhancement for {tool_name}: {e}"
            )
            return True  # Default to enhancement on error

    @staticmethod
    def _should_enhance_search(args: tuple, kwargs: dict) -> bool:
        """Determine if search should use AI enhancement."""
        # Check search mode
        search_mode = kwargs.get("search_mode", "auto")
        if search_mode in ["comprehensive", "semantic"]:
            return True

        # Fast mode explicitly disables AI
        if search_mode == "fast":
            return False

        # Check query complexity
        query = args[1] if len(args) > 1 else kwargs.get("query", "")

        # Multiple terms in list
        if isinstance(query, list) and len(query) > 2:
            return True

        # Boolean operators in string
        if isinstance(query, str):
            boolean_ops = ["AND", "OR", "NOT"]
            if any(op in query.upper() for op in boolean_ops):
                return True

            # Simple single-word queries don't need AI enhancement
            words = query.strip().split()
            if len(words) == 1:
                return False

            # Long queries (more than 3 words)
            if len(words) > 3:
                return True

        # High result count requests
        max_results = kwargs.get("max_results", 10)
        if max_results > 15:
            return True

        return False

    @staticmethod
    def _should_enhance_analysis(tool_name: str, args: tuple, kwargs: dict) -> bool:
        """Determine if analysis tools should use AI enhancement."""
        match tool_name:
            case "analyze_ids_structure":
                return EnhancementDecisionEngine._should_enhance_structure_analysis(
                    args, kwargs
                )
            case "explore_relationships":
                return EnhancementDecisionEngine._should_enhance_relationships(
                    args, kwargs
                )
            case _:
                return True

    @staticmethod
    def _should_enhance_structure_analysis(args: tuple, kwargs: dict) -> bool:
        """Determine if structure analysis should use AI enhancement."""
        ids_name = args[1] if len(args) > 1 else kwargs.get("ids_name", "")

        # Complex IDS patterns that benefit from AI analysis
        complex_patterns = [
            "core_profiles",
            "equilibrium",
            "transport",
            "edge_profiles",
            "mhd",
            "disruption",
            "pellets",
            "wall",
            "ec_launchers",
        ]

        return any(pattern in ids_name.lower() for pattern in complex_patterns)

    @staticmethod
    def _should_enhance_relationships(args: tuple, kwargs: dict) -> bool:
        """Determine if relationship exploration should use AI enhancement."""
        # Deep analysis (high max_depth)
        max_depth = kwargs.get("max_depth", 2)
        if max_depth >= 3:
            return True

        # Complex physics domain paths
        path = args[1] if len(args) > 1 else kwargs.get("path", "")
        complex_path_patterns = [
            "profiles",
            "transport",
            "equilibrium",
            "mhd",
            "disruption",
            "heating",
            "current_drive",
            "edge",
            "pedestal",
        ]
        if any(pattern in path.lower() for pattern in complex_path_patterns):
            return True

        # Specific relationship types that need AI interpretation
        relationship_type = kwargs.get("relationship_type", "all")
        if relationship_type in ["physics", "measurement_dependencies"]:
            return True

        return False

    @staticmethod
    def _should_enhance_export(tool_name: str, args: tuple, kwargs: dict) -> bool:
        """Determine if export tools should use AI enhancement."""
        match tool_name:
            case "export_ids":
                return EnhancementDecisionEngine._should_enhance_bulk_export(
                    args, kwargs
                )
            case "export_physics_domain":
                return EnhancementDecisionEngine._should_enhance_physics_domain(
                    args, kwargs
                )
            case _:
                return True

    @staticmethod
    def _should_enhance_bulk_export(args: tuple, kwargs: dict) -> bool:
        """Determine if bulk export should use AI enhancement."""
        # Raw format explicitly disables AI
        output_format = kwargs.get("output_format", "structured")
        if output_format == "raw":
            return False

        # Enhanced format always enables AI
        if output_format == "enhanced":
            return True

        # Multiple IDS (4+) enable AI
        ids_list = args[1] if len(args) > 1 else kwargs.get("ids_list", [])
        if len(ids_list) > 3:
            return True

        # Full analysis with relationships and physics context (3+ IDS)
        include_relationships = kwargs.get("include_relationships", True)
        include_physics_context = kwargs.get("include_physics_context", True)
        if include_relationships and include_physics_context and len(ids_list) > 2:
            return True

        return False

    @staticmethod
    def _should_enhance_physics_domain(args: tuple, kwargs: dict) -> bool:
        """Determine if physics domain export should use AI enhancement."""
        # Comprehensive analysis depth
        analysis_depth = kwargs.get("analysis_depth", "focused")
        if analysis_depth == "comprehensive":
            return True

        # Cross-domain analysis
        include_cross_domain = kwargs.get("include_cross_domain", False)
        if include_cross_domain:
            return True

        # Large exports
        max_paths = kwargs.get("max_paths", 10)
        if max_paths > 20:
            return True

        return False


# AI prompts for different tool categories
AI_PROMPTS = {
    ToolCategory.SEARCH: """You are an IMAS search expert with deep knowledge of data relationships. Analyze relevance-ranked search results and provide:
1. 5 related search terms for plasma physics research, considering cross-references and physics concepts
2. Brief physics insights about the found data paths and their measurement context
3. Suggestions for complementary searches based on measurement relationships and connected IDS

The search results are ordered by relevance considering exact matches, path position, 
documentation content, path specificity, physics concepts, and cross-reference connectivity. 
Focus on practical physics relationships and measurement considerations that would lead to 
valuable follow-up searches using the rich relationship network.""",
    ToolCategory.EXPLANATION: """You are an IMAS physics expert providing clear explanations of plasma physics concepts. Provide:
1. Physics significance and context from physics concepts database
2. Domain-specific explanations appropriate for the detail level requested  
3. Related IMAS paths and measurement considerations
4. Practical applications and measurement scenarios

Focus on accurate physics explanations that connect theoretical concepts to practical 
IMAS data structures and measurement workflows.""",
    ToolCategory.OVERVIEW: """You are an IMAS data expert providing comprehensive overviews. Focus on:
1. Clear explanations of IMAS structure and physics domains
2. Practical guidance for researchers and data analysts
3. Navigation strategies for finding relevant data
4. Physics context and measurement workflows
Provide actionable insights for effective IMAS usage.""",
    ToolCategory.ANALYSIS: """You are an IMAS structural analysis expert. Analyze IDS organization and provide:
1. Physics domain context and measurement scope
2. Data hierarchy and organization logic
3. Critical measurement paths and their physics significance
4. Practical usage guidance for researchers
5. Integration patterns with other IDS
Focus on actionable insights for data analysis workflows.""",
    ToolCategory.EXPORT: """You are an IMAS data export expert. Analyze export selections and provide:
1. Data usage recommendations for the exported datasets
2. Physics context explaining relationships between selected IDS or data paths
3. Suggested analysis workflows utilizing the exported data
4. Integration patterns and measurement dependencies
5. Quality considerations and data validation approaches

Focus on practical guidance for researchers working with IMAS data exports,
emphasizing physics-aware analysis workflows and cross-IDS relationships.""",
}


def ai_enhancer(temperature: float = 0.3, max_tokens: int = 800):
    """
    Decorator for selective AI enhancement of MCP tool responses.

    Automatically adds AI insights to tool responses when MCP context is available
    and when the enhancement decision engine determines AI would be valuable.
    Falls back gracefully when context is unavailable or enhancement is not needed.

    Args:
        temperature: Sampling temperature for AI response
        max_tokens: Maximum tokens for AI response

    Usage:
        @ai_enhancer(temperature=0.3)
        async def search_imas(self, query: str, ctx: Optional[Context] = None):
            # Tool implementation returns base response
            results = self.perform_search(query)

            # Add AI prompt for conditional enhancement
            base_response = {
                "results": results,
                "ai_prompt": f"Analyze search for: {query}\\nResults: {results}"
            }

            # Decorator will apply AI enhancement if conditions are met
            return base_response
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # Extract ctx from kwargs
            ctx = kwargs.get("ctx")

            # Call the original tool function
            base_response = await func(*args, **kwargs)

            # Use the decision engine to determine if AI enhancement should be applied
            should_enhance = EnhancementDecisionEngine.should_enhance(
                func.__name__, args, kwargs, ctx
            )

            if not should_enhance:
                # Remove ai_prompt from response if present and provide status feedback
                if isinstance(base_response, dict) and "ai_prompt" in base_response:
                    del base_response["ai_prompt"]
                    base_response["ai_insights"] = {
                        "status": "AI enhancement not applied - conditions not met"
                    }
                return base_response

            # Check if AI prompt is available for enhancement
            if not isinstance(base_response, dict) or "ai_prompt" not in base_response:
                return base_response

            # Extract AI prompt and remove from response
            ai_prompt = base_response.pop("ai_prompt")

            # Get the appropriate system prompt for this tool category
            config = TOOL_ENHANCEMENT_CONFIG.get(
                func.__name__, {"category": ToolCategory.OVERVIEW}
            )
            system_prompt = AI_PROMPTS.get(
                config["category"], AI_PROMPTS[ToolCategory.OVERVIEW]
            )

            try:
                logger.debug(f"Attempting AI enhancement for {func.__name__}")

                # Use the FastMCP recommended approach with system_prompt parameter
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
                logger.warning(f"AI enhancement failed for {func.__name__}: {e}")
                # Log more detailed error information for debugging
                logger.debug(
                    f"AI enhancement error details: {type(e).__name__}: {str(e)}"
                )
                ai_insights = {"error": "AI enhancement temporarily unavailable"}

            # Add AI insights to the response
            base_response["ai_insights"] = ai_insights
            return base_response

        return wrapper

    return decorator
