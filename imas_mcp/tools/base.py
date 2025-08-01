"""
Base tool functionality for IMAS MCP tools.

This module contains common functionality shared across all tool implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel

from imas_mcp.models.response_models import ErrorResponse

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

        # Initialize search service
        self._search_service = self._create_search_service()

        # Initialize business logic services
        self.physics = PhysicsService()
        self.response = ResponseService()
        self.documents = DocumentService(self.document_store)
        self.search_config = SearchConfigurationService()

        # New services for Phase 2.5
        self.sampling = SamplingService()
        self.recommendations = ToolRecommendationService()

    @abstractmethod
    def get_tool_name(self) -> str:
        """Return the name of this tool."""
        pass

    async def apply_sampling(self, result: BaseModel, **kwargs) -> BaseModel:
        """
        Template method for applying sampling to tool results.
        Subclasses can customize by setting sampling_strategy class variable.
        """
        if not self.enable_sampling:
            return result

        return await self.sampling.apply_sampling(
            result=result, strategy=self.sampling_strategy, **kwargs
        )

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

        # Apply recommendations (only for SearchResponse)
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

    def _create_error_response(
        self, error_message: str, query: str = ""
    ) -> ErrorResponse:
        """Create a standardized error response."""
        return ErrorResponse(
            error=error_message,
            suggestions=[],
            context={
                "query": query,
                "tool": self.get_tool_name(),
                "status": "error",
            },
        )
