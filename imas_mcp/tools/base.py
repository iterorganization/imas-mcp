"""
Base tool functionality for IMAS MCP tools.

This module contains common functionality shared across all tool implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel

from imas_mcp.models.error_models import ToolError
from imas_mcp.models.result_models import SearchResult

from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.services.search_service import SearchService
from imas_mcp.search.engines.semantic_engine import SemanticSearchEngine
from imas_mcp.search.engines.lexical_engine import LexicalSearchEngine
from imas_mcp.search.engines.hybrid_engine import HybridSearchEngine
from imas_mcp.models.constants import SearchMode
from imas_mcp.services import (
    PhysicsService,
    ResponseService,
    DocumentService,
    SearchConfigurationService,
    SamplingService,
    ToolRecommendationService,
)
from imas_mcp.services.sampling import SamplingStrategy
from imas_mcp.services.tool_recommendations import RecommendationStrategy
from imas_mcp.services.service_context import ServiceOrchestrator

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Base class for all IMAS MCP tools with service injection."""

    # Class variables for template method pattern
    sampling_strategy: SamplingStrategy = SamplingStrategy.NO_SAMPLING
    recommendation_strategy: RecommendationStrategy = (
        RecommendationStrategy.SEARCH_BASED
    )
    max_recommended_tools: int = 4
    enable_sampling: bool = False
    enable_recommendations: bool = True

    def __init__(self, document_store: Optional[DocumentStore] = None):
        self.logger = logger
        self.document_store = document_store or DocumentStore()

        # Initialize service orchestrator
        self._orchestrator = ServiceOrchestrator(self)

        # Initialize search service
        self._search_service = self._create_search_service()

        # Initialize services
        self.physics = PhysicsService()
        self.response = ResponseService()
        self.documents = DocumentService(self.document_store)
        self.search_config = SearchConfigurationService()
        self.sampling = SamplingService()
        self.recommendations = ToolRecommendationService()

    @abstractmethod
    def get_tool_name(self) -> str:
        """Return the name of this tool."""
        pass

    def service_context(self, operation_type: str, **kwargs):
        """Access to the service context manager."""
        return self._orchestrator.service_context(operation_type, **kwargs)

    async def apply_sampling(self, result: BaseModel, **kwargs) -> BaseModel:
        """
        Template method for applying sampling to tool results.
        Subclasses can customize by setting sampling_strategy class variable.
        """
        if self.enable_sampling:
            return await self.sampling.apply_sampling(
                result=result,  # type: ignore
                strategy=self.sampling_strategy,
                **kwargs,
            )

        return result

    def generate_tool_recommendations(
        self, result: BaseModel, query: Optional[str] = None, **kwargs
    ) -> list:
        """
        Template method for generating tool recommendations.
        Subclasses can customize by setting recommendation_strategy class variable.
        Note: This method is mainly for compatibility. Use apply_services() instead.
        """
        if not self.enable_recommendations:
            return []

        return self.recommendations.generate_recommendations(
            result=result,
            strategy=self.recommendation_strategy,
            max_tools=self.max_recommended_tools,
            query=query,
            **kwargs,
        )

    async def apply_services(self, result: BaseModel, **kwargs) -> BaseModel:
        """
        Template method for applying all post-processing services.
        Called after tool execution but before response formatting.
        """

        # Apply recommendations (only for SearchResult)
        if self.enable_recommendations:
            result = self.recommendations.apply_recommendations(
                result=result,
                strategy=self.recommendation_strategy,
                max_tools=self.max_recommended_tools,
                **kwargs,
            )

        # Apply sampling last
        if self.enable_sampling:
            result = await self.apply_sampling(result, **kwargs)

        return result

    async def execute_search(
        self,
        query: str,
        search_mode: Union[str, SearchMode] = "auto",
        max_results: int = 10,
        ids_filter: Optional[List[str]] = None,
    ) -> SearchResult:
        """
        Unified search execution that returns a complete SearchResult.

        Args:
            query: Search query
            search_mode: Search mode to use
            max_results: Maximum results to return
            ids_filter: Optional IDS filter

        Returns:
            SearchResult with all search data and context
        """
        # Create and optimize configuration
        config = self.search_config.create_config(
            search_mode=search_mode,
            max_results=max_results,
            ids_filter=ids_filter,
        )
        config = self.search_config.optimize_for_query(query, config)

        # Execute search
        search_results = await self._search_service.search(query, config)

        # Generate AI prompts separately
        ai_prompt = self.generate_ai_prompts(query, search_results)
        ai_response = {}  # Will be populated by services later

        # Add physics context to response (always enabled)
        physics_context = await self.physics.enhance_query(query)
        if physics_context:
            ai_response["physics_context"] = physics_context

        # Build complete response using service
        response = self.response.build_search_response(
            results=search_results,
            query=query,
            search_mode=config.search_mode,
            ids_filter=ids_filter,
            max_results=max_results,
            ai_response=ai_response,
            ai_prompt=ai_prompt,
        )

        return response

    def generate_ai_prompts(
        self,
        query: str,
        results: List[Any],
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Generate AI prompts based on search results and context.

        Args:
            query: Original search query
            results: Search results
            tool_context: Optional tool-specific context

        Returns:
            Dictionary of AI prompts for processing
        """
        prompts = {}

        if not results:
            prompts["guidance"] = self._build_no_results_guidance(query)
        else:
            prompts["analysis"] = self._build_analysis_prompt(query, results)

        # Add tool-specific prompts
        if tool_context:
            prompts.update(self._build_tool_specific_prompts(tool_context))

        return prompts

    def _build_no_results_guidance(self, query: str) -> str:
        """Generate guidance for empty results."""
        return f"""No results found for "{query}". Suggest alternatives and related concepts."""

    def _build_analysis_prompt(self, query: str, results: List[Any]) -> str:
        """Generate analysis prompt for search results."""
        return f"""Analyze {len(results)} search results for "{query}" and provide insights."""

    def _build_tool_specific_prompts(
        self, tool_context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Override in subclasses to add tool-specific AI prompts."""
        return {}

    def _create_search_service(self) -> SearchService:
        """Create search service with appropriate engines."""
        # Create engines for each mode
        engines = {}
        for mode in [SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID]:
            engine = self._create_engine(mode.value)
            engines[mode] = engine

        return SearchService(engines)

    def _create_engine(self, engine_type: str):
        """Create a search engine of the specified type."""
        engine_map = {
            "semantic": SemanticSearchEngine,
            "lexical": LexicalSearchEngine,
            "hybrid": HybridSearchEngine,
        }

        if engine_type not in engine_map:
            raise ValueError(f"Unknown engine type: {engine_type}")

        engine_class = engine_map[engine_type]
        return engine_class(self.document_store)

    def _create_error_response(self, error_message: str, query: str = "") -> ToolError:
        """Create a standardized error response."""
        return ToolError(
            error=error_message,
            suggestions=[],
            context={
                "query": query,
                "tool": self.get_tool_name(),
                "status": "error",
            },
        )
