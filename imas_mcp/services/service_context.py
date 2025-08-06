"""
Service context management for tool operations.

This module provides a context manager pattern for handling pre/post-processing
services consistently across all IMAS MCP tools.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.context_models import QueryContext, AIContext
from imas_mcp.search.search_strategy import SearchConfig
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class ServiceContext:
    """
    Context object for service orchestration across tool operations.

    Provides a unified interface for managing configuration, physics context,
    AI prompts, metadata, and results across different operation types.
    """

    operation_type: str
    query_context: QueryContext = field(default_factory=QueryContext)
    ai_context: AIContext = field(default_factory=AIContext)

    # Service states
    search_config: Optional[SearchConfig] = field(default=None, init=False)
    physics_context: Optional[Any] = field(default=None, init=False)
    result: Optional[BaseModel] = field(default=None, init=False)
    tool_context: Dict[str, Any] = field(default_factory=dict, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    @property
    def query(self) -> Optional[Union[str, List[str]]]:
        """Access query from composed QueryContext."""
        return self.query_context.query

    @query.setter
    def query(self, value: Optional[Union[str, List[str]]]):
        """Set query in composed QueryContext."""
        self.query_context.query = value

    @property
    def ai_prompt(self) -> Dict[str, str]:
        """Access AI prompts from composed AIContext."""
        return self.ai_context.ai_prompt or {}

    @property
    def ai_response(self) -> Dict[str, Any]:
        """Access AI response from composed AIContext."""
        return self.ai_context.ai_response or {}

    def update_search_context(
        self,
        search_mode: SearchMode,
        max_results: int,
        ids_filter: Optional[List[str]] = None,
    ):
        """Update query context with search parameters."""
        self.query_context.search_mode = search_mode
        self.query_context.max_results = max_results
        self.query_context.ids_filter = ids_filter

    async def enhance_with_physics(self, physics_service):
        """Add physics enhancement to the context."""
        if self.query:
            self.physics_context = await physics_service.enhance_query(self.query)
            if self.physics_context:
                if not self.ai_context.ai_response:
                    self.ai_context.ai_response = {}
                self.ai_context.ai_response["physics_context"] = self.physics_context
                logger.debug(f"Physics context added for query: {self.query}")

    def add_ai_prompts(self, prompts: Dict[str, str]):
        """Add AI prompts to the context."""
        if not self.ai_context.ai_prompt:
            self.ai_context.ai_prompt = {}
        self.ai_context.ai_prompt.update(prompts)

    def add_metadata(self, metadata: Dict[str, Any]):
        """Add metadata to the context."""
        self.metadata.update(metadata)

    def build_search_response(self, response_service, **kwargs):
        """Build a search response using the response service."""
        return response_service.build_search_response(
            ai_response=self.ai_response, ai_prompt=self.ai_prompt, **kwargs
        )


class ServiceOrchestrator:
    """
    Service orchestration manager for tools.

    Provides context manager functionality for consistent pre/post-processing
    service application across all tool operations.
    """

    def __init__(self, tool_instance):
        """Initialize with reference to the tool instance."""
        self.tool = tool_instance
        self.logger = logger

    @asynccontextmanager
    async def service_context(
        self,
        operation_type: str,
        query: Optional[str] = None,
        search_mode: Union[str, SearchMode] = "auto",
        max_results: int = 10,
        ids_filter: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Context manager for service orchestration.

        Handles pre-processing setup, yields control to tool logic,
        then applies post-processing services.

        Args:
            operation_type: Type of operation (search, explain, export, etc.)
            query: Optional search query
            search_mode: Search mode for search operations
            max_results: Maximum results for search operations
            ids_filter: IDS filter for search operations
            **kwargs: Additional context parameters

        Yields:
            ServiceContext: Configured context object for tool use
        """
        # Create service context
        ctx = ServiceContext(operation_type=operation_type)
        ctx.query = query
        ctx.tool_context = kwargs

        # Pre-processing setup based on operation type
        await self._setup_pre_services(
            ctx, search_mode, max_results, ids_filter, **kwargs
        )

        try:
            yield ctx
        finally:
            # Post-processing (always happens)
            await self._apply_post_services(ctx, **kwargs)

    async def _setup_pre_services(
        self,
        service_ctx: ServiceContext,
        search_mode: Union[str, SearchMode],
        max_results: int,
        ids_filter: Optional[List[str]],
        **kwargs,
    ):
        """Setup pre-processing services based on operation type."""

        # Remove ctx from kwargs to avoid conflicts
        kwargs = {k: v for k, v in kwargs.items() if k != "ctx"}

        # Configure search for search-based operations
        if service_ctx.operation_type in [
            "search",
            "explain",
            "relationships",
            "analysis",
        ]:
            service_ctx.search_config = self.tool.search_config.create_config(
                search_mode=search_mode,
                max_results=max_results,
                ids_filter=ids_filter,
            )

            # Optimize configuration for query
            if service_ctx.query:
                service_ctx.search_config = self.tool.search_config.optimize_for_query(
                    service_ctx.query, service_ctx.search_config
                )

        # Physics enhancement for applicable operations
        if service_ctx.query and service_ctx.operation_type in [
            "search",
            "explain",
            "export",
            "relationships",
        ]:
            await service_ctx.enhance_with_physics(self.tool.physics)

        # Generate operation-specific AI prompts
        if service_ctx.query:
            base_prompts = self.tool.generate_ai_prompts(
                service_ctx.query, [], tool_context=service_ctx.tool_context
            )
            service_ctx.add_ai_prompts(base_prompts)

            # Add operation-specific prompts
            operation_prompts = self._generate_operation_prompts(service_ctx)
            service_ctx.add_ai_prompts(operation_prompts)

        # Add operation metadata
        service_ctx.add_metadata(
            {
                "operation_type": service_ctx.operation_type,
                "tool_name": self.tool.get_tool_name(),
                "timestamp": "operation_start",
                "configuration": {
                    "physics_enabled": True,
                    "search_mode": search_mode if service_ctx.search_config else None,
                    "max_results": max_results if service_ctx.search_config else None,
                },
            }
        )

    async def _apply_post_services(self, service_ctx: ServiceContext, **kwargs):
        """Apply post-processing services to the result."""
        if service_ctx.result is None:
            return

        try:
            # Apply tool services (sampling and recommendations)
            service_ctx.result = await self.tool.apply_services(
                result=service_ctx.result, query=service_ctx.query, **kwargs
            )

            # Add standard metadata
            service_ctx.result = self.tool.response.add_standard_metadata(
                service_ctx.result, self.tool.get_tool_name()
            )

            logger.debug(f"Post-processing completed for {service_ctx.operation_type}")

        except Exception as e:
            logger.error(
                f"Post-processing failed for {service_ctx.operation_type}: {e}"
            )
            # Don't re-raise - preserve original result

    def _generate_operation_prompts(self, ctx: ServiceContext) -> Dict[str, str]:
        """Generate operation-specific AI prompts."""
        prompts = {}

        if ctx.operation_type == "search":
            prompts["search_guidance"] = f"""Search operation for: "{ctx.query}"
Provide search result analysis and recommendations for further exploration."""

        elif ctx.operation_type == "explain":
            prompts["explanation_guidance"] = f"""Concept explanation for: "{ctx.query}"
Provide comprehensive physics context and IMAS integration details."""

        elif ctx.operation_type == "export":
            export_type = ctx.tool_context.get("export_type", "general")
            prompts["export_guidance"] = f"""Export operation: {export_type}
Provide data usage recommendations and analysis workflow guidance."""

        elif ctx.operation_type == "relationships":
            prompts[
                "relationship_guidance"
            ] = f"""Relationship analysis for: "{ctx.query}"
Provide insights into data connections and physics relationships."""

        elif ctx.operation_type == "analysis":
            prompts["analysis_guidance"] = f"""Structural analysis for: "{ctx.query}"
Provide IDS structure insights and data organization details."""

        return prompts


# Mixin class for tools to use the service orchestrator
class ServiceOrchestrationMixin:
    """
    Mixin class providing service orchestration capabilities to tools.

    Tools can inherit from this mixin to get the service context manager.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._orchestrator = ServiceOrchestrator(self)

    def service_context(self, operation_type: str, **kwargs):
        """Access to the service context manager."""
        return self._orchestrator.service_context(operation_type, **kwargs)
