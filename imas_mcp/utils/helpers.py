"""
Common helper functions for IMAS MCP tools.

This module provides reusable utility functions that are used across multiple tools,
following the DRY principle and improving code maintainability.
"""

import logging
from typing import Any, Dict, List, Optional

from imas_mcp.core.physics_context import get_physics_context_provider

logger = logging.getLogger(__name__)


class ToolHelper:
    """Helper class providing common functionality for IMAS MCP tools."""

    def __init__(self):
        self.physics_context = get_physics_context_provider()

    def handle_error(
        self, operation: str, error: Exception, suggestions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Standard error handling for tool operations.

        Args:
            operation: Description of the operation that failed
            error: The exception that occurred
            suggestions: Optional list of suggestions for the user

        Returns:
            Dictionary with standardized error response
        """
        logger.error(f"{operation} failed: {error}")

        default_suggestions = [
            "Check input parameters",
            "Try alternative search terms",
            "Use get_overview() to explore available data",
        ]

        return {
            "error": str(error),
            "operation": operation,
            "suggestions": suggestions or default_suggestions,
        }

    def validate_query_input(self, query: Any, operation_name: str) -> str:
        """
        Validate and normalize query input.

        Args:
            query: Query input to validate
            operation_name: Name of the operation for error messages

        Returns:
            Normalized query string

        Raises:
            ValueError: If query is invalid
        """
        if not query:
            raise ValueError(f"Query is required for {operation_name}")

        if isinstance(query, list):
            if not query:
                raise ValueError(f"Query list cannot be empty for {operation_name}")
            return " ".join(str(q) for q in query)

        return str(query).strip()

    def enhance_with_physics_context(
        self, query: str, search_results_count: int = 0
    ) -> Dict[str, Any]:
        """
        Get physics context enhancement for a query.

        Args:
            query: Search query to enhance
            search_results_count: Number of search results found

        Returns:
            Dictionary with physics context data
        """
        domain_context = self.physics_context.get_domain_context(query)
        unit_context = self.physics_context.get_unit_context(query)
        suggestions = self.physics_context.enhance_search_suggestions(
            query, search_results_count
        )

        enhancement = {}
        if domain_context:
            enhancement["domain_context"] = domain_context.model_dump()
        if unit_context:
            enhancement["unit_context"] = unit_context.model_dump()
        if suggestions:
            enhancement["suggestions"] = suggestions

        return enhancement

    def build_ai_prompt(
        self,
        operation: str,
        query: str,
        context_data: Dict[str, Any],
        results_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build standardized AI enhancement prompt.

        Args:
            operation: Type of operation (search, explain, export, etc.)
            query: User query
            context_data: Physics context and other relevant data
            results_summary: Summary of results found

        Returns:
            Formatted AI prompt string
        """
        prompt_parts = [f"{operation.title()} Request: {query}", "", "Context:"]

        if context_data:
            for key, value in context_data.items():
                if value:
                    prompt_parts.append(f"- {key}: {value}")

        if results_summary:
            prompt_parts.extend(["", "Results Summary:"])
            for key, value in results_summary.items():
                prompt_parts.append(f"- {key}: {value}")

        prompt_parts.extend(
            [
                "",
                "Provide analysis focusing on:",
                "1. Physics context and significance",
                "2. IMAS data structure relationships",
                "3. Practical guidance for researchers",
                "4. Suggestions for further exploration",
            ]
        )

        return "\n".join(prompt_parts)

    def create_no_results_response(
        self,
        query: str,
        operation: str,
        physics_suggestions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create standardized response when no results are found.

        Args:
            query: Original query
            operation: Type of operation
            physics_suggestions: Physics context based suggestions

        Returns:
            Dictionary with no results response
        """
        base_suggestions = [
            "Try a broader search term",
            "Check spelling of physics terms",
            "Use alternative terminology",
            "Explore related concepts with get_overview()",
        ]

        all_suggestions = base_suggestions
        if physics_suggestions:
            all_suggestions.extend(physics_suggestions)

        return {
            "query": query,
            "operation": operation,
            "results_count": 0,
            "message": f"No information found for '{query}'",
            "suggestions": all_suggestions,
        }
