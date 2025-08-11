"""
Enhancement strategy for selective AI sampling.

Determines when and how to apply AI enhancement to tool results.
Works with the existing MCP sampling decorator to make sampling selective rather than always-on.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EnhancementMode(Enum):
    """AI enhancement modes."""

    NEVER = "never"
    ALWAYS = "always"
    CONDITIONAL = "conditional"
    SMART = "smart"


class ToolType(Enum):
    """Tool type classifications for enhancement decisions."""

    SEARCH = "search"
    ANALYSIS = "analysis"
    EXPORT = "export"
    EXPLANATION = "explanation"
    OVERVIEW = "overview"
    RELATIONSHIPS = "relationships"
    IDENTIFIERS = "identifiers"


@dataclass
class EnhancementConfig:
    """Configuration for AI enhancement strategy."""

    mode: EnhancementMode = EnhancementMode.SMART
    max_result_count_for_enhancement: int = 50
    min_result_count_for_enhancement: int = 1
    enhancement_probability: float = 0.8
    tool_specific_settings: dict[ToolType, dict[str, Any]] | None = None

    def __post_init__(self):
        """Initialize default tool-specific settings."""
        if self.tool_specific_settings is None:
            self.tool_specific_settings = {
                ToolType.SEARCH: {
                    "enhance_empty_results": True,
                    "enhance_large_result_sets": True,
                    "enhancement_threshold": 0.7,
                },
                ToolType.EXPLANATION: {
                    "always_enhance": True,
                    "detail_level_boost": True,
                },
                ToolType.ANALYSIS: {
                    "enhance_complex_queries": True,
                    "physics_context_boost": True,
                },
                ToolType.EXPORT: {
                    "enhance_if_multiple_formats": False,
                    "enhance_structured_exports": True,
                },
                ToolType.OVERVIEW: {
                    "enhance_broad_queries": True,
                    "context_expansion": True,
                },
                ToolType.RELATIONSHIPS: {
                    "enhance_multi_path_results": True,
                    "cross_reference_boost": True,
                },
                ToolType.IDENTIFIERS: {
                    "enhance_if_ambiguous": True,
                    "suggestion_boost": False,
                },
            }


class EnhancementDecisionEngine:
    """Decides when to apply AI enhancement to tool results."""

    def __init__(self, config: EnhancementConfig | None = None):
        """Initialize enhancement decision engine."""
        self.config = config or EnhancementConfig()
        self.logger = logging.getLogger(__name__)

    def should_enhance(
        self,
        tool_name: str,
        query: str | list[str],
        result: Any,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """
        Determine if a tool result should be enhanced with AI sampling.

        Args:
            tool_name: Name of the tool that generated the result
            query: Original query/input to the tool
            result: Tool result to potentially enhance
            context: Additional context for decision making

        Returns:
            True if the result should be enhanced
        """
        # Quick mode checks
        if self.config.mode == EnhancementMode.NEVER:
            return False
        elif self.config.mode == EnhancementMode.ALWAYS:
            return True

        # Get tool type
        tool_type = self._classify_tool(tool_name)
        tool_settings = (
            self.config.tool_specific_settings.get(tool_type, {})
            if self.config.tool_specific_settings
            else {}
        )

        # Check for errors or invalid results
        if self._has_errors(result):
            return False

        # Apply conditional logic
        if self.config.mode == EnhancementMode.CONDITIONAL:
            return self._conditional_enhancement_decision(
                tool_type, query, result, tool_settings, context
            )

        # Apply smart logic (default)
        return self._smart_enhancement_decision(
            tool_type, query, result, tool_settings, context
        )

    def _classify_tool(self, tool_name: str) -> ToolType:
        """Classify tool by name to determine enhancement strategy."""
        tool_name_lower = tool_name.lower()

        if "search" in tool_name_lower:
            return ToolType.SEARCH
        elif "explain" in tool_name_lower or "concept" in tool_name_lower:
            return ToolType.EXPLANATION
        elif "analysis" in tool_name_lower or "analyze" in tool_name_lower:
            return ToolType.ANALYSIS
        elif "export" in tool_name_lower:
            return ToolType.EXPORT
        elif "overview" in tool_name_lower:
            return ToolType.OVERVIEW
        elif "relationship" in tool_name_lower or "relate" in tool_name_lower:
            return ToolType.RELATIONSHIPS
        elif "identifier" in tool_name_lower or "ids" in tool_name_lower:
            return ToolType.IDENTIFIERS

        # Default to search-like behavior
        return ToolType.SEARCH

    def _has_errors(self, result: Any) -> bool:
        """Check if result contains errors."""
        if isinstance(result, dict):
            return "error" in result or result.get("status") == "error"
        return False

    def _conditional_enhancement_decision(
        self,
        tool_type: ToolType,
        query: str | list[str],
        result: Any,
        tool_settings: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> bool:
        """Make conditional enhancement decision based on simple rules."""
        # Tool-specific conditional logic
        if tool_type == ToolType.EXPLANATION:
            return tool_settings.get("always_enhance", True)

        if tool_type == ToolType.SEARCH:
            # Enhance empty results if configured
            if self._is_empty_result(result) and tool_settings.get(
                "enhance_empty_results", True
            ):
                return True

            # Enhance large result sets if configured
            result_count = self._get_result_count(result)
            if (
                result_count > self.config.max_result_count_for_enhancement
                and tool_settings.get("enhance_large_result_sets", True)
            ):
                return True

        # Default: enhance if result count is within reasonable range
        result_count = self._get_result_count(result)
        return (
            self.config.min_result_count_for_enhancement
            <= result_count
            <= self.config.max_result_count_for_enhancement
        )

    def _smart_enhancement_decision(
        self,
        tool_type: ToolType,
        query: str | list[str],
        result: Any,
        tool_settings: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> bool:
        """Make smart enhancement decision using multiple factors."""
        enhancement_score = 0.0

        # Base score by tool type
        tool_base_scores = {
            ToolType.EXPLANATION: 0.9,
            ToolType.ANALYSIS: 0.8,
            ToolType.SEARCH: 0.7,
            ToolType.OVERVIEW: 0.7,
            ToolType.RELATIONSHIPS: 0.6,
            ToolType.EXPORT: 0.3,
            ToolType.IDENTIFIERS: 0.4,
        }
        enhancement_score += tool_base_scores.get(tool_type, 0.5)

        # Query complexity factors
        query_str = str(query) if isinstance(query, list) else query
        if len(query_str.split()) > 3:
            enhancement_score += 0.1  # Complex queries benefit from enhancement

        # Physics-related terms boost
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
        if any(term in query_str.lower() for term in physics_terms):
            enhancement_score += 0.1

        # Result-based factors
        result_count = self._get_result_count(result)

        # Empty results often need explanation
        if result_count == 0:
            enhancement_score += 0.2

        # Very large result sets benefit from summarization
        elif result_count > 20:
            enhancement_score += 0.15

        # Medium result sets are good candidates
        elif 3 <= result_count <= 10:
            enhancement_score += 0.1

        # Context-based factors
        if context:
            # User explicitly requested detailed analysis
            if context.get("detail_level") in ["advanced", "detailed"]:
                enhancement_score += 0.2

            # Follow-up query (user is exploring)
            if context.get("is_followup", False):
                enhancement_score += 0.1

        # Apply tool-specific boosts
        if tool_type == ToolType.SEARCH and tool_settings.get("enhancement_threshold"):
            threshold = tool_settings["enhancement_threshold"]
            return enhancement_score >= threshold

        # Default threshold
        return enhancement_score >= self.config.enhancement_probability

    def _is_empty_result(self, result: Any) -> bool:
        """Check if result is empty."""
        if isinstance(result, dict):
            # Check various result count indicators
            count_fields = ["count", "results_count", "total", "length"]
            for field in count_fields:
                if field in result and result[field] == 0:
                    return True

            # Check for empty result lists
            result_fields = ["results", "hits", "data", "paths", "items"]
            for field in result_fields:
                if field in result and len(result[field]) == 0:
                    return True

        return False

    def _get_result_count(self, result: Any) -> int:
        """Extract result count from result."""
        if isinstance(result, dict):
            # Try common count fields
            for field in ["count", "results_count", "total"]:
                if field in result and isinstance(result[field], int):
                    return result[field]

            # Try counting result lists
            for field in ["results", "hits", "data", "paths", "items"]:
                if field in result and isinstance(result[field], list):
                    return len(result[field])

        elif isinstance(result, list):
            return len(result)

        return 0 if not result else 1


# Global configuration instance
TOOL_ENHANCEMENT_CONFIG = EnhancementConfig()

# Global decision engine instance
_enhancement_engine = None


def get_enhancement_engine() -> EnhancementDecisionEngine:
    """Get global enhancement decision engine instance."""
    global _enhancement_engine
    if _enhancement_engine is None:
        _enhancement_engine = EnhancementDecisionEngine(TOOL_ENHANCEMENT_CONFIG)
    return _enhancement_engine


def configure_enhancement(config: EnhancementConfig) -> None:
    """Configure global enhancement settings."""
    global TOOL_ENHANCEMENT_CONFIG, _enhancement_engine
    TOOL_ENHANCEMENT_CONFIG = config
    _enhancement_engine = EnhancementDecisionEngine(config)


def should_enhance_result(
    tool_name: str,
    query: str | list[str],
    result: Any,
    context: dict[str, Any] | None = None,
) -> bool:
    """
    Convenience function to check if a result should be enhanced.

    Args:
        tool_name: Name of the tool that generated the result
        query: Original query/input to the tool
        result: Tool result to potentially enhance
        context: Additional context for decision making

    Returns:
        True if the result should be enhanced
    """
    engine = get_enhancement_engine()
    return engine.should_enhance(tool_name, query, result, context)
