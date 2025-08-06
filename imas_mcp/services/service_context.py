"""
Clean service context management for tool operations.

This module provides a minimal, focused context manager that handles only
the common service patterns without interfering with tool-specific logic.
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from functools import cached_property

from imas_mcp.models.context_models import QueryContext
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OperationContext(BaseModel):
    """
    Minimal context for common service operations.

    Only contains what's truly shared across all operations.
    """

    operation_type: str
    query_context: QueryContext

    # Common service results
    search_results: Optional[List[Any]] = None
    ai_prompts: Dict[str, str] = {}

    @property
    def query(self) -> Optional[str]:
        """Clean query access."""
        if isinstance(self.query_context.query, list):
            return " ".join(self.query_context.query)
        return self.query_context.query


class ServiceOrchestrator:
    """
    Clean service orchestrator with minimal responsibilities.

    Focuses only on:
    - Search execution
    - AI prompt generation
    - Response building delegation
    """

    def __init__(self, tool_instance):
        """Initialize with tool reference."""
        self.tool = tool_instance
        self.logger = logger

    @asynccontextmanager
    async def operation_context(
        self,
        operation_type: str,
        query_context: QueryContext,
    ):
        """
        Minimal context manager for common operations.

        Args:
            operation_type: Operation type (search, explain, export, etc.)
            query_context: Query configuration

        Yields:
            OperationContext: Clean context with common services
        """
        ctx = OperationContext(
            operation_type=operation_type,
            query_context=query_context,
        )

        # Setup physics context if available
        await self._setup_physics_context(ctx)

        try:
            yield ctx
        finally:
            # Minimal cleanup
            logger.debug(f"Operation {operation_type} completed")

    # =====================================
    # SEARCH SERVICE
    # =====================================

    @cached_property
    def search_config_creator(self):
        """Cached search configuration creator."""

        def _create_config(query_context: QueryContext):
            if hasattr(self.tool, "search_config"):
                return self.tool.search_config.create_config(
                    search_mode=query_context.search_mode,
                    max_results=query_context.max_results,
                    ids_filter=query_context.ids_filter,
                )
            return None

        return _create_config

    async def search(self, ctx: OperationContext) -> List[Any]:
        """Execute search with clean interface."""
        if not ctx.query:
            return []

        search_config = self.search_config_creator(ctx.query_context)
        if not search_config or not hasattr(self.tool, "_search_service"):
            return []

        ctx.search_results = await self.tool._search_service.search(
            ctx.query, search_config
        )

        results = ctx.search_results or []
        logger.debug(f"Search completed: {len(results)} results")
        return results

    # =====================================
    # AI PROMPT SERVICE
    # =====================================

    def generate_ai_prompts(self, ctx: OperationContext) -> Dict[str, str]:
        """Generate AI prompts with clean delegation."""
        if hasattr(self.tool, "generate_ai_prompts"):
            # Delegate to tool's AI prompt generation
            return self.tool.generate_ai_prompts(
                ctx.query or "",
                ctx.search_results or [],
                {"operation_type": ctx.operation_type},
            )

        # Minimal fallback
        return {"analysis": f"Analyze '{ctx.query}' for {ctx.operation_type}"}

    # =====================================
    # RESPONSE SERVICE
    # =====================================

    def build_search_response(self, ctx: OperationContext):
        """Build search response with clean delegation."""
        if not hasattr(self.tool, "response"):
            return ctx.search_results

        return self.tool.response.build_search_response(
            results=ctx.search_results or [],
            query=ctx.query,
            search_mode=ctx.query_context.search_mode,
            ids_filter=ctx.query_context.ids_filter,
            max_results=ctx.query_context.max_results,
            ai_response={},
            ai_prompt=ctx.ai_prompts,
        )

    def add_metadata(self, response: Any) -> Any:
        """Add metadata with clean delegation."""
        if hasattr(self.tool, "response") and hasattr(
            self.tool.response, "add_standard_metadata"
        ):
            return self.tool.response.add_standard_metadata(
                response, self.tool.get_tool_name()
            )
        return response

    # =====================================
    # PHYSICS SERVICE
    # =====================================

    async def _setup_physics_context(self, ctx: OperationContext):
        """Minimal physics context setup."""
        if not ctx.query or not hasattr(self.tool, "physics"):
            return

        try:
            physics_result = await self.tool.physics.enhance_query(ctx.query)
            if physics_result:
                # Store in AI prompts for tool use
                ctx.ai_prompts["physics_context"] = (
                    f"Physics context available for {ctx.query}"
                )
                logger.debug(f"Physics context enhanced for: {ctx.query}")
        except Exception as e:
            logger.debug(f"Physics enhancement skipped: {e}")


# =====================================
# CONVENIENCE FUNCTIONS
# =====================================


async def search_with_context(tool, query_context: QueryContext) -> List[Any]:
    """Convenience function for search operations."""
    orchestrator = ServiceOrchestrator(tool)
    async with orchestrator.operation_context("search", query_context) as ctx:
        return await orchestrator.search(ctx)


def build_search_response_with_context(tool, ctx: OperationContext):
    """Convenience function for search response building."""
    orchestrator = ServiceOrchestrator(tool)
    return orchestrator.build_search_response(ctx)
