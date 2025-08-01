# Decorated Service Implementation Plan

## Implementation Progress Checklist

### Foundation & Architecture ✅ COMPLETED

- [x] **Model Structure Cleanup** → [Completed Tasks Section](#completed-tasks-)

  - [x] Moved SearchHit from `response_models.py` to `search_strategy.py`
  - [x] Eliminated circular import between SearchResult and SearchHit
  - [x] Added `to_hit()` method to SearchResult for clean API conversion
  - [x] Updated all tool imports to use new SearchHit location

- [x] **Inheritance Pattern Implementation** → [SearchBase Class](#completed-tasks-)

  - [x] Created SearchBase class with common fields (score, rank, search_mode, highlights)
  - [x] SearchHit inherits from SearchBase and adds API-specific fields
  - [x] SearchResult inherits from SearchBase and adds internal document reference
  - [x] Eliminated field duplication and improved type safety through inheritance

- [x] **API Consistency** → [Response Models](#completed-tasks-)

  - [x] Removed `count` property alias from SearchResponse
  - [x] Updated all code to use `hit_count` consistently throughout the codebase
  - [x] Maintained backward compatibility through proper inheritance hierarchy

- [x] **Import Organization** → [Python Import Standards](#python-import-standards)
  - [x] All imports follow Python standards with standard library, third-party, then local imports

### Phase 1: Service Architecture Foundation ✅ COMPLETED

- [x] **Create Service Base Classes** → [Section 1.1](#11-create-service-base-classes)

  - [x] `imas_mcp/services/__init__.py`
  - [x] `imas_mcp/services/base.py`

- [x] **Physics Integration Service** → [Section 1.2](#12-physics-integration-service)

  - [x] `imas_mcp/services/physics.py`
  - [x] `enhance_query()` method implementation
  - [x] `get_concept_context()` method implementation

- [x] **Document Store Service** → [Section 1.3](#13-document-store-service)

  - [x] `imas_mcp/services/document.py`
  - [x] `validate_ids()` method implementation
  - [x] `get_documents_safe()` method implementation
  - [x] `create_ids_not_found_error()` method implementation

- [x] **Search Configuration Service** → [Section 1.4](#14-search-configuration-service)

  - [x] `imas_mcp/services/search_configuration.py`
  - [x] `create_config()` method implementation
  - [x] `optimize_for_query()` method implementation

- [x] **Response Building Service** → [Section 1.5](#15-response-building-service)

  - [x] `imas_mcp/services/response.py`
  - [x] `build_search_response()` method with `to_hit()` integration

- [x] **Service Tests** → [Testing Framework](#testing-framework)

  - [x] `tests/services/` directory structure created
  - [x] Unit tests for all 5 service classes
  - [x] All 32 service tests passing (100% success rate)
  - [x] `add_standard_metadata()` method implementation with real timestamps
  - [x] Comprehensive metadata tests (content, format, timestamp freshness)
  - [x] 6 SearchTool service integration tests passing

- [x] **Unit Tests for Services** → [Section 1.6](#16-unit-tests-for-services)
  - [x] `tests/services/test_physics_service.py`
  - [x] `tests/services/test_document_service.py`
  - [x] `tests/services/test_search_configuration_service.py`
  - [x] `tests/services/test_response_service.py`
  - [x] `tests/services/test_base.py`
  - [x] `tests/tools/test_search_tool_services.py`

### Phase 2: Search Tool Implementation ✅ COMPLETED

- [x] **Update BaseTool with Service Injection** → [Section 2.1](#21-update-basetool-with-service-injection)

  - [x] Modify `imas_mcp/tools/base.py`
  - [x] Add service initialization in constructor
  - [x] Add dependency injection pattern

- [x] **Refactor SearchTool** → [Section 2.2](#22-refactor-searchtool-with-service-composition)

  - [x] Update `imas_mcp/tools/search_tool.py`
  - [x] Replace manual logic with service calls
  - [x] Maintain existing decorator patterns
  - [x] Update `search_imas()` method implementation

- [x] **SearchTool Service Integration Tests** → [Section 2.3](#23-unit-tests-for-searchtool)
  - [x] `tests/tools/test_search_tool_services.py`
  - [x] Test physics enhancement through service
  - [x] Test search configuration optimization
  - [x] Test response building with results

### Phase 2.5: LLM Client Sampling Service Architecture 📋 PENDING

- [ ] **Create Sampling Service** → [Section 2.5.1](#251-create-sampling-service)

  - [ ] `imas_mcp/services/sampling.py`
  - [ ] Replace `@sample` decorator with service-based approach
  - [ ] Implement `apply_sample()` method with MCP context handling
  - [ ] Add sampling strategy decision engine (renamed from enhancement strategy)

- [ ] **Create Tool Recommendation Service** → [Section 2.5.2](#252-create-tool-recommendation-service)

  - [ ] `imas_mcp/services/tool_recommendations.py`
  - [ ] Replace `@recommend_tools` decorator with service-based approach
  - [ ] Implement `generate_recommendations()` method with context analysis
  - [ ] Add recommendation strategy engine for different tool types

- [ ] **Update BaseTool with Sampling & Recommendation Infrastructure** → [Section 2.5.3](#253-update-basetool-with-sampling--recommendation-infrastructure)

  - [ ] Add sampling and recommendation service injection to `imas_mcp/tools/base.py`
  - [ ] Implement template methods for sampling customization
  - [ ] Add `build_sample_prompt()` base method
  - [ ] Add `should_sample()` decision logic
  - [ ] Add `build_tool_recommendations()` base method
  - [ ] Add `should_recommend_tools()` decision logic
  - [ ] Add `get_sampling_config()` and `get_recommendation_config()` methods
  - [ ] Add `process_sample_result()` and `process_recommendations()` methods

- [ ] **Refactor SearchTool for Sampling & Recommendation Services** → [Section 2.5.4](#254-refactor-searchtool-for-sampling--recommendation-services)

  - [ ] Add sampling and recommendation configuration as class variables
  - [ ] Implement tool-specific `build_sample_prompt()` override
  - [ ] Implement tool-specific `build_tool_recommendations()` override
  - [ ] Remove `@sample` and `@recommend_tools` decorators
  - [ ] Integrate service calls in `search_imas()` method
  - [ ] Update response building to include sample insights and tool suggestions

- [ ] **Sampling & Recommendation Service Integration Tests** → [Section 2.5.5](#255-sampling--recommendation-service-integration-tests)
  - [ ] `tests/services/test_sampling_service.py`
  - [ ] `tests/services/test_tool_recommendations_service.py`
  - [ ] `tests/tools/test_search_tool_sampling.py`
  - [ ] `tests/tools/test_search_tool_recommendations.py`
  - [ ] Test sampling strategy decisions
  - [ ] Test recommendation generation for different result types
  - [ ] Test tool-specific customization
  - [ ] Test MCP context integration

### Phase 3: Remaining Tools Rollout 📋 PENDING

- [ ] **ExplainTool with Services** → [Section 3.1](#31-explaintool-with-service-composition)

  - [ ] Update `imas_mcp/tools/explain_tool.py`
  - [ ] Integrate PhysicsService for concept context
  - [ ] Use ResponseService for standardized responses

- [ ] **Integration Tests** → [Section 3.2](#32-integration-tests-for-multi-tool-service-usage)

  - [ ] `tests/integration/test_service_integration.py`
  - [ ] Test physics service consistency across tools
  - [ ] Test response service metadata consistency

- [ ] **Additional Tools Migration** 📋 PENDING
  - [ ] OverviewTool service integration
  - [ ] AnalysisTool service integration
  - [ ] RelationshipsTool service integration
  - [ ] IdentifiersTool service integration
  - [ ] ExportTool service integration

### Phase 4: Advanced Service Features 📋 PENDING

- [ ] **Cross-IDS Analysis Service** → [Section 4.1](#41-cross-ids-analysis-service)

  - [ ] `imas_mcp/services/cross_ids_analysis.py`
  - [ ] `analyze_relationships()` method implementation
  - [ ] Multi-IDS relationship analysis

- [ ] **Enhanced Export Tool** → [Section 4.2](#42-enhanced-export-tool-with-cross-ids-service)

  - [ ] Update `imas_mcp/tools/export_tool.py`
  - [ ] Integrate CrossIdsAnalysisService
  - [ ] Enhanced bulk export functionality

- [ ] **Comprehensive Service Tests** → [Section 4.3](#43-comprehensive-service-tests)
  - [ ] `tests/services/test_cross_ids_analysis.py`
  - [ ] Advanced service functionality testing

### Phase 5: Documentation & Migration 📋 PENDING

- [ ] **Service Architecture Documentation** → [Section 5.1](#51-service-architecture-documentation)

  - [ ] `docs/SERVICE_ARCHITECTURE.md`
  - [ ] Complete architecture documentation
  - [ ] Usage patterns and examples

- [ ] **Migration Guide** → [Section 5.2](#52-migration-guide)
  - [ ] `docs/MIGRATION_GUIDE.md`
  - [ ] Breaking changes documentation
  - [ ] Testing changes guide

### Success Metrics & Quality Gates 📋 PENDING

- [ ] **Code Coverage** → [Success Metrics Section](#success-metrics)

  - [ ] Minimum 90% test coverage for services
  - [ ] All critical paths tested

- [ ] **Performance Validation** → [Quality Gates Section](#quality-gates)

  - [ ] Response times within 10% of baseline
  - [ ] Performance benchmarks meet/exceed baseline

- [ ] **Feature Parity** → [Phase Completion Criteria](#phase-completion-criteria)

  - [ ] All features from comparison report implemented
  - [ ] API compatibility maintained

- [ ] **Integration Testing** → [Risk Mitigation Section](#risk-mitigation)
  - [ ] 100+ tests passing
  - [ ] Cross-tool functionality validated

---

## Implementation Status

### Completed Tasks ✅

1. **Model Structure Cleanup**:

   - Moved SearchHit from `response_models.py` to `search_strategy.py`
   - Eliminated circular import between SearchResult and SearchHit
   - Added `to_hit()` method to SearchResult for clean API conversion
   - Updated all tool imports to use new SearchHit location

2. **Inheritance Pattern Implementation**:

   - Created SearchBase class with common fields (score, rank, search_mode, highlights)
   - SearchHit inherits from SearchBase and adds API-specific fields (path, documentation, units, etc.)
   - SearchResult inherits from SearchBase and adds internal document reference
   - Eliminated field duplication and improved type safety through inheritance

3. **API Consistency**:

   - Removed `count` property alias from SearchResponse
   - Updated all code to use `hit_count` consistently throughout the codebase
   - Maintained backward compatibility through proper inheritance hierarchy

4. **Import Organization**: All imports follow Python standards with standard library, third-party, then local imports

5. **Service Architecture Foundation (Phase 1)**:

   - Created 5 core services: BaseService, PhysicsService, DocumentService, SearchConfigurationService, ResponseService
   - All 32 service unit tests passing (100% success rate)
   - Real timestamp generation and dynamic versioning implemented
   - Comprehensive metadata testing with content validation

6. **Search Tool Implementation (Phase 2)**:

   - Updated BaseTool with service injection and dependency injection pattern
   - Refactored SearchTool to use service composition instead of manual logic
   - Maintained existing decorator patterns (cache, validation, performance, error handling)
   - Created 6 SearchTool service integration tests, all passing
   - Total test count: 38 tests passing (32 service + 6 tool integration)

### In Progress Tasks 🚧

1. **LLM Client Sampling Service Architecture (Phase 2.5)**: Ready to implement sampling service to replace `@sample` decorator

### Pending Tasks 📋

1. **LLM Client Sampling Service (Phase 2.5)**: SamplingService, BaseTool sampling infrastructure, SearchTool sampling integration
2. **Remaining Tools Migration (Phase 3)**: ExplainTool, OverviewTool, AnalysisTool, RelationshipsTool, IdentifiersTool, ExportTool
3. **Advanced Service Features (Phase 4)**: Cross-IDS Analysis Service, Enhanced Export Tool
4. **Documentation & Migration (Phase 5)**: Service Architecture Documentation, Migration Guide
5. **Performance & Quality (Phase 6)**: Code Coverage, Performance Validation, Feature Parity Testing

---

## Service Implementation Architecture

## Executive Summary

This plan implements a composition-based architecture using services for business logic and decorators for cross-cutting concerns. The approach maintains Pydantic input/output models while achieving better separation of concerns, testability, and maintainability. Implementation starts with `search_imas` tool and rolls out to all tools systematically.

## Architecture Principles

### Core Design Decisions

- **Composition over Inheritance**: Services injected into tools via composition
- **Decorators for Cross-Cutting Concerns**: Cache, validation, error handling
- **Services for Business Logic**: Physics integration, response building, document operations
- **Pydantic Throughout**: Input validation and output models maintained
- **No Backwards Compatibility**: Clean break from old patterns
- **Behavior-Focused Testing**: Tests check features, not implementation details

### Python Import Standards

All code examples follow Python import organization standards:

1. **Standard library imports** (logging, typing, etc.)
2. **Third-party library imports** (pydantic, fastmcp, etc.)
3. **Local application imports** (imas_mcp modules)
4. **Relative imports** (from .base import BaseTool)

Imports are grouped with blank lines between groups and sorted alphabetically within each group. No imports are placed inline within functions unless absolutely necessary for TYPE_CHECKING.

### Service Responsibility Matrix

| Service                      | Responsibility                            | Decorator Equivalent                        |
| ---------------------------- | ----------------------------------------- | ------------------------------------------- |
| `PhysicsService`             | Physics integration, context enhancement  | Replaces physics integration decorator      |
| `ResponseService`            | Pydantic model construction, metadata     | Replaces response standardization decorator |
| `DocumentService`            | Document store access, validation         | Replaces document store access decorator    |
| `SearchConfigurationService` | Search configuration, optimization        | Replaces search config decorator            |
| `SamplingService`            | LLM client sampling, MCP context handling | Replaces `@sample` decorator                |
| `ToolRecommendationService`  | Tool suggestions, workflow guidance       | Replaces `@recommend_tools` decorator       |

## Phase 1: Service Architecture Foundation (Week 1)

### 1.1 Create Service Base Classes

**File: `imas_mcp/services/__init__.py`**

```python
"""
IMAS MCP Services Package.

Service layer for business logic separation from cross-cutting concerns.
"""

from .base import BaseService
from .physics import PhysicsService
from .response import ResponseService
from .document import DocumentService
from .search_configuration import SearchConfigurationService
from .sampling import SamplingService
from .tool_recommendations import ToolRecommendationService

__all__ = [
    "BaseService",
    "PhysicsService",
    "ResponseService",
    "DocumentService",
    "SearchConfigurationService",
    "SamplingService",
    "ToolRecommendationService",
]
```

**File: `imas_mcp/services/base.py`**

```python
"""Base service class for dependency injection."""

from abc import ABC
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class BaseService(ABC):
    """Base class for all services with common functionality."""

    def __init__(self):
        self.logger = logger

    async def initialize(self) -> None:
        """Initialize service resources. Override in subclasses."""
        pass

    async def cleanup(self) -> None:
        """Cleanup service resources. Override in subclasses."""
        pass
```

### 1.2 Physics Integration Service

**File: `imas_mcp/services/physics.py`**

```python
"""Physics integration service for IMAS tools."""

from typing import Optional, Dict, Any

from imas_mcp.physics_integration import physics_search, explain_physics_concept
from imas_mcp.models.physics_models import PhysicsSearchResult
from .base import BaseService

class PhysicsService(BaseService):
    """Service for physics integration and enhancement."""

    async def enhance_query(self, query: str) -> Optional[PhysicsSearchResult]:
        """
        Enhance query with physics context.

        Args:
            query: Search query to enhance

        Returns:
            Physics search result or None if enhancement fails
        """
        try:
            result = physics_search(query)
            self.logger.debug(f"Physics enhancement successful for: {query}")
            return result
        except Exception as e:
            self.logger.warning(f"Physics enhancement failed for '{query}': {e}")
            return None

    async def get_concept_context(self, concept: str, detail_level: str = "intermediate") -> Optional[Dict[str, Any]]:
        """Get physics context for a concept."""
        try:
            result = explain_physics_concept(concept, detail_level)
            return {
                "domain": result.domain,
                "description": result.description,
                "phenomena": result.phenomena,
                "typical_units": result.typical_units,
                "complexity_level": result.complexity_level,
            }
        except Exception as e:
            self.logger.warning(f"Concept context failed for '{concept}': {e}")
            return None
```

### 1.3 Document Store Service

**File: `imas_mcp/services/document.py`**

```python
"""Document store service for IMAS tools."""

from typing import List, Optional, Any

from imas_mcp.search.document_store import DocumentStore
from imas_mcp.models.response_models import ErrorResponse
from .base import BaseService

class DocumentService(BaseService):
    """Service for document store operations."""

    def __init__(self, document_store: Optional[DocumentStore] = None):
        super().__init__()
        self.store = document_store or DocumentStore()

    async def validate_ids(self, ids_names: List[str]) -> tuple[List[str], List[str]]:
        """
        Check IDS names against available IDS.

        Returns:
            Tuple of (valid_ids, invalid_ids)
        """
        available_ids = self.store.get_available_ids()
        valid_ids = [ids for ids in ids_names if ids in available_ids]
        invalid_ids = [ids for ids in ids_names if ids not in available_ids]
        return valid_ids, invalid_ids

    async def get_documents_safe(self, ids_name: str) -> List[Any]:
        """Get documents for IDS with error handling."""
        try:
            return self.store.get_documents_by_ids(ids_name)
        except Exception as e:
            self.logger.error(f"Failed to get documents for {ids_name}: {e}")
            return []

    def create_ids_not_found_error(self, ids_name: str, tool_name: str) -> ErrorResponse:
        """Create standardized IDS not found error."""
        available_ids = self.store.get_available_ids()
        return ErrorResponse(
            error=f"IDS '{ids_name}' not found",
            suggestions=[f"Try: {ids}" for ids in available_ids[:5]],
            context={
                "available_ids": available_ids[:10],
                "ids_name": ids_name,
                "tool": tool_name,
            },
        )
```

### 1.4 Search Configuration Service

**File: `imas_mcp/services/search_configuration.py`**

```python
"""Search configuration service."""

from typing import Union, List, Optional
from imas_mcp.search.search_strategy import SearchConfig
from imas_mcp.models.constants import SearchMode
from .base import BaseService

class SearchConfigurationService(BaseService):
    """Service for search configuration and optimization."""

    def create_config(
        self,
        search_mode: Union[str, SearchMode] = "auto",
        max_results: int = 10,
        ids_filter: Optional[Union[str, List[str]]] = None,
        enable_physics: bool = False,
    ) -> SearchConfig:
        """Create optimized search configuration."""

        # Convert string to SearchMode enum if needed
        if isinstance(search_mode, str):
            search_mode = SearchMode(search_mode)

        # Optimize max_results based on mode
        if search_mode == SearchMode.SEMANTIC and max_results > 20:
            self.logger.warning(f"Large result set ({max_results}) may impact semantic search performance")

        return SearchConfig(
            search_mode=search_mode,
            max_results=max_results,
            ids_filter=ids_filter,
            enable_physics_enhancement=enable_physics,
            similarity_threshold=0.0,
        )

    def optimize_for_query(self, query: Union[str, List[str]], base_config: SearchConfig) -> SearchConfig:
        """Optimize configuration based on query characteristics."""

        query_str = query if isinstance(query, str) else " ".join(query)

        # Adjust search mode based on query complexity
        if len(query_str.split()) > 5:
            # Complex queries benefit from semantic search
            base_config.search_mode = SearchMode.SEMANTIC
        elif any(op in query_str.upper() for op in ["AND", "OR", "NOT"]):
            # Boolean queries work better with lexical search
            base_config.search_mode = SearchMode.LEXICAL

        return base_config
```

### 1.5 Response Building Service

**File: `imas_mcp/services/response.py`**

```python
"""Response building service for consistent Pydantic model construction."""

from typing import Dict, Any, List, Optional, Type, TypeVar
from pydantic import BaseModel
from imas_mcp.models.response_models import SearchResponse
from imas_mcp.search.search_strategy import SearchHit
from imas_mcp.search.search_strategy import SearchResult
from imas_mcp.models.constants import SearchMode
from .base import BaseService

T = TypeVar('T', bound=BaseModel)

class ResponseService(BaseService):
    """Service for building standardized responses."""

    def build_search_response(
        self,
        results: List[SearchResult],
        query: str,
        search_mode: SearchMode,
        ai_insights: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """Build SearchResponse from search results."""

        # Convert SearchResult objects to SearchHit for API response
        hits = [result.to_hit() for result in results]

        return SearchResponse(
            hits=hits,
            search_mode=search_mode,
            query=query,
            ai_insights=ai_insights or {},
        )

    def add_standard_metadata(self, response: BaseModel, tool_name: str) -> BaseModel:
        """Add standard metadata to any response."""
        if hasattr(response, 'metadata'):
            if not response.metadata:
                response.metadata = {}
            response.metadata.update({
                "tool": tool_name,
                "processing_timestamp": "2024-01-01T00:00:00Z",  # Would use real timestamp
                "version": "1.0",
            })
        return response
```

### 1.6 Unit Tests for Services

**File: `tests/services/test_physics_service.py`**

```python
"""Tests for PhysicsService."""

import pytest
from unittest.mock import patch, MagicMock
from imas_mcp.services.physics import PhysicsService

class TestPhysicsService:
    """Test physics service functionality."""

    @pytest.fixture
    def physics_service(self):
        return PhysicsService()

    @pytest.mark.asyncio
    async def test_enhance_query_success(self, physics_service):
        """Test successful query enhancement."""
        with patch('imas_mcp.services.physics.physics_search') as mock_search:
            mock_result = MagicMock()
            mock_search.return_value = mock_result

            result = await physics_service.enhance_query("plasma temperature")

            assert result == mock_result
            mock_search.assert_called_once_with("plasma temperature")

    @pytest.mark.asyncio
    async def test_enhance_query_failure(self, physics_service):
        """Test query enhancement with failure."""
        with patch('imas_mcp.services.physics.physics_search') as mock_search:
            mock_search.side_effect = Exception("Physics search failed")

            result = await physics_service.enhance_query("invalid query")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_concept_context_success(self, physics_service):
        """Test successful concept context retrieval."""
        with patch('imas_mcp.services.physics.explain_physics_concept') as mock_explain:
            mock_result = MagicMock()
            mock_result.domain = "core_plasma"
            mock_result.description = "Test description"
            mock_result.phenomena = ["phenomenon1"]
            mock_result.typical_units = ["eV"]
            mock_result.complexity_level = "intermediate"
            mock_explain.return_value = mock_result

            result = await physics_service.get_concept_context("temperature")

            assert result["domain"] == "core_plasma"
            assert result["description"] == "Test description"
            assert "phenomenon1" in result["phenomena"]
            assert "eV" in result["typical_units"]
```

## Phase 2: Implement Search Tool with Services (Week 2)

### 2.1 Update BaseTool with Service Injection

**File: `imas_mcp/tools/base.py`**

```python
"""Base tool functionality with service injection."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

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
)

if TYPE_CHECKING:
    from imas_mcp.models.response_models import ErrorResponse

logger = logging.getLogger(__name__)

class BaseTool(ABC):
    """Base class for all IMAS MCP tools with service injection."""

    def __init__(self, document_store: Optional[DocumentStore] = None):
        self.logger = logger
        self.document_store = document_store or DocumentStore()

        # Initialize search service (existing pattern)
        self._search_service = self._create_search_service()

        # Initialize business logic services
        self.physics = PhysicsService()
        self.response = ResponseService()
        self.documents = DocumentService(self.document_store)
        self.search_config = SearchConfigurationService()

    @abstractmethod
    def get_tool_name(self) -> str:
        """Return the name of this tool."""
        pass

    def _create_search_service(self) -> SearchService:
        """Create search service with appropriate engines."""
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

    def _create_error_response(self, error_message: str, query: str = "") -> "ErrorResponse":
        """Create a standardized error response."""
        from imas_mcp.models.response_models import ErrorResponse

        return ErrorResponse(
            error=error_message,
            suggestions=[],
            context={
                "query": query,
                "tool": self.get_tool_name(),
                "status": "error",
            },
        )
```

### 2.2 Refactor SearchTool with Service Composition

**File: `imas_mcp/tools/search_tool.py`**

```python
"""Search tool implementation using service composition."""

import logging
from typing import Any, List, Optional, Union

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.response_models import SearchResponse
from imas_mcp.models.request_models import SearchInput

# Import only essential decorators
from imas_mcp.search.decorators import (
    cache_results,
    validate_input,
    measure_performance,
    handle_errors,
)

from .base import BaseTool

logger = logging.getLogger(__name__)

def mcp_tool(description: str):
    """Decorator to mark methods as MCP tools with descriptions."""
    def decorator(func):
        func._mcp_tool = True
        func._mcp_description = description
        return func
    return decorator

class SearchTool(BaseTool):
    """Tool for searching IMAS data paths using service composition."""

    def get_tool_name(self) -> str:
        return "search_imas"

    @cache_results(ttl=300, key_strategy="semantic")
    @validate_input(schema=SearchInput)
    @measure_performance(include_metrics=True, slow_threshold=1.0)
    @handle_errors(fallback="search_suggestions")
    @mcp_tool("Search for IMAS data paths with relevance-ordered results")
    async def search_imas(
        self,
        query: Union[str, List[str]],
        ids_filter: Optional[Union[str, List[str]]] = None,
        max_results: int = 10,
        search_mode: Union[str, SearchMode] = "auto",
        ctx: Optional[Any] = None,
    ) -> SearchResponse:
        """
        Search for IMAS data paths with relevance-ordered results.

        Uses service composition for business logic:
        - SearchConfigurationService: Creates and optimizes search configuration
        - PhysicsService: Enhances queries with physics context
        - ResponseService: Builds standardized Pydantic responses

        Args:
            query: Search term(s), physics concept, symbol, or pattern
            ids_filter: Optional specific IDS to search within
            max_results: Maximum number of results to return (1-100)
            search_mode: Search mode - "auto", "semantic", "lexical", or "hybrid"
            ctx: MCP context for enhancement

        Returns:
            SearchResponse with hits, metadata, and optional AI insights
        """

        # Create search configuration using service
        config = self.search_config.create_config(
            search_mode=search_mode,
            max_results=max_results,
            ids_filter=ids_filter,
            enable_physics=True,
        )

        # Optimize configuration based on query characteristics
        config = self.search_config.optimize_for_query(query, config)

        # Execute search through existing search service
        logger.info(f"Executing search: query='{query}' mode={config.search_mode} max_results={max_results}")
        search_results = await self._search_service.search(query, config)

        # Enhance with physics context if available
        physics_context = None
        if ctx:
            physics_context = await self.physics.enhance_query(
                query if isinstance(query, str) else " ".join(query)
            )

        # Prepare AI insights for potential sampling
        ai_insights = {"physics_context": physics_context}
        if not search_results:
            ai_insights["guidance"] = self._build_no_results_guidance(query)
        else:
            ai_insights["analysis_prompt"] = self._build_analysis_prompt(query, search_results)

        # Build response using service
        response = self.response.build_search_response(
            results=search_results,
            query=query if isinstance(query, str) else " ".join(query),
            search_mode=config.search_mode,
            ai_insights=ai_insights,
        )

        # Add standard metadata
        response = self.response.add_standard_metadata(response, self.get_tool_name())

        logger.info(f"Search completed: {len(search_results)} results returned")
        return response

    def _build_no_results_guidance(self, query: Union[str, List[str]]) -> str:
        """Build guidance for queries with no results."""
        query_str = query if isinstance(query, str) else " ".join(query)
        return f"""No results found for IMAS search: "{query_str}"

Provide helpful guidance including:
1. Alternative search terms or concepts to try
2. Common IMAS data paths that might be related
3. Physics context that might help refine the search
4. Suggestions for broader or narrower search strategies"""

    def _build_analysis_prompt(self, query: Union[str, List[str]], results: List[Any]) -> str:
        """Build analysis prompt for AI enhancement."""
        query_str = query if isinstance(query, str) else " ".join(query)

        top_results = results[:3]
        results_text = "\n".join([
            f"- {result.document.metadata.path_name}: {result.document.documentation[:100]}..."
            for result in top_results
        ])

        return f"""Search Results Analysis for: "{query_str}"
Found {len(results)} relevant paths in IMAS data dictionary.

Top results:
{results_text}

Provide detailed analysis including:
1. Physics context and significance of these paths
2. Recommended follow-up searches or related concepts
3. Data usage patterns and common workflows
4. Validation considerations for these measurements
5. Relationships between the found paths"""
```

### 2.3 Unit Tests for SearchTool

**File: `tests/tools/test_search_tool_services.py`**

```python
"""Tests for SearchTool with service composition."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from imas_mcp.tools.search_tool import SearchTool
from imas_mcp.models.response_models import SearchResponse
from imas_mcp.models.constants import SearchMode

class TestSearchToolServices:
    """Test SearchTool service composition functionality."""

    @pytest.fixture
    def search_tool(self):
        with patch('imas_mcp.tools.base.DocumentStore'):
            return SearchTool()

    @pytest.mark.asyncio
    async def test_search_with_physics_enhancement(self, search_tool):
        """Test search with physics enhancement through service."""

        # Mock search service
        search_tool._search_service.search = AsyncMock(return_value=[])

        # Mock physics service
        mock_physics_result = MagicMock()
        search_tool.physics.enhance_query = AsyncMock(return_value=mock_physics_result)

        # Mock response service
        mock_response = SearchResponse(
            hits=[],
            search_mode=SearchMode.SEMANTIC,
            query="test query",
            ai_insights={},
        )
        search_tool.response.build_search_response = MagicMock(return_value=mock_response)
        search_tool.response.add_standard_metadata = MagicMock(return_value=mock_response)

        # Mock context for physics enhancement
        mock_ctx = MagicMock()

        result = await search_tool.search_imas(
            query="plasma temperature",
            ctx=mock_ctx
        )

        # Verify services were called
        search_tool.physics.enhance_query.assert_called_once_with("plasma temperature")
        search_tool.response.build_search_response.assert_called_once()
        search_tool.response.add_standard_metadata.assert_called_once()

        assert isinstance(result, SearchResponse)

    @pytest.mark.asyncio
    async def test_search_configuration_optimization(self, search_tool):
        """Test search configuration optimization based on query."""

        # Mock services
        search_tool._search_service.search = AsyncMock(return_value=[])
        search_tool.physics.enhance_query = AsyncMock(return_value=None)

        mock_response = SearchResponse(
            hits=[],
            search_mode=SearchMode.SEMANTIC,
            query="complex query",
            ai_insights={},
        )
        search_tool.response.build_search_response = MagicMock(return_value=mock_response)
        search_tool.response.add_standard_metadata = MagicMock(return_value=mock_response)

        # Test with complex query that should trigger semantic search
        await search_tool.search_imas(
            query="plasma temperature profile equilibrium magnetic field"
        )

        # Verify search configuration service was used
        # (Implementation detail: service should optimize to semantic mode for complex queries)
        search_tool._search_service.search.assert_called_once()
        call_args = search_tool._search_service.search.call_args
        config = call_args[0][1]  # Second argument is config

        # Complex query should use semantic search
        assert config.search_mode == SearchMode.SEMANTIC

    @pytest.mark.asyncio
    async def test_response_building_with_results(self, search_tool):
        """Test response building when search returns results."""

        # Mock search results
        mock_result = MagicMock()
        mock_result.document.metadata.path_name = "core_profiles/temperature"
        mock_result.document.documentation = "Plasma temperature measurement"
        mock_result.score = 0.95

        search_tool._search_service.search = AsyncMock(return_value=[mock_result])
        search_tool.physics.enhance_query = AsyncMock(return_value=None)

        # Mock response service to capture arguments
        mock_response = SearchResponse(
            hits=[],
            search_mode=SearchMode.SEMANTIC,
            query="temperature",
            ai_insights={},
        )
        search_tool.response.build_search_response = MagicMock(return_value=mock_response)
        search_tool.response.add_standard_metadata = MagicMock(return_value=mock_response)

        result = await search_tool.search_imas(query="temperature")

        # Verify response service received correct arguments
        build_call = search_tool.response.build_search_response.call_args
        assert build_call[1]['query'] == "temperature"
        assert len(build_call[1]['results']) == 1
        assert 'analysis_prompt' in build_call[1]['ai_insights']

    @pytest.mark.asyncio
    async def test_no_results_guidance(self, search_tool):
        """Test guidance generation when no results found."""

        search_tool._search_service.search = AsyncMock(return_value=[])
        search_tool.physics.enhance_query = AsyncMock(return_value=None)

        mock_response = SearchResponse(
            hits=[],
            search_mode=SearchMode.SEMANTIC,
            query="nonexistent",
            ai_insights={},
        )
        search_tool.response.build_search_response = MagicMock(return_value=mock_response)
        search_tool.response.add_standard_metadata = MagicMock(return_value=mock_response)

        await search_tool.search_imas(query="nonexistent")

        # Verify guidance was built for empty results
        build_call = search_tool.response.build_search_response.call_args
        assert 'guidance' in build_call[1]['ai_insights']
        guidance = build_call[1]['ai_insights']['guidance']
        assert "No results found" in guidance
        assert "Alternative search terms" in guidance
```

        # Boolean query should be optimized to lexical
        assert config.search_mode == SearchMode.LEXICAL

````

## Phase 2.5: LLM Client Sampling Service Architecture

### 2.5.1 Create Sampling Service

**File: `imas_mcp/services/sampling.py`**

```python
"""Sampling service for LLM client integration."""

from enum import Enum
from typing import Any, Dict, Optional, Union
from imas_mcp.models.constants import SearchMode
from .base import BaseService

class SamplingStrategy(Enum):
    """Sampling strategy options."""
    NEVER = "never"
    ALWAYS = "always"
    CONDITIONAL = "conditional"
    SMART = "smart"

class SamplingService(BaseService):
    """Service for LLM client sampling operations."""

    async def apply_sample(
        self,
        sample_prompt: str,
        ctx: Any,
        temperature: float = 0.3,
        max_tokens: int = 800
    ) -> Dict[str, Any]:
        """
        Apply LLM sampling to generate insights.

        Replaces the functionality of @sample decorator with service-based approach.
        """
        try:
            if not ctx or not hasattr(ctx, "session"):
                return {"status": "unavailable", "reason": "No AI context available"}

            if not hasattr(ctx.session, "create_message"):
                return {"status": "unavailable", "reason": "AI not available in session"}

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
            self.logger.warning(f"Sampling failed: {e}")
            return {"status": "error", "reason": str(e)}

    def should_sample(
        self,
        strategy: SamplingStrategy,
        tool_name: str,
        query: Union[str, list],
        result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Decide whether to apply sampling based on strategy and context.

        Renamed and refactored from enhancement_strategy.py for clarity.
        """
        if strategy == SamplingStrategy.NEVER:
            return False
        elif strategy == SamplingStrategy.ALWAYS:
            return True

        # Smart sampling logic (conditional + intelligent decisions)
        if strategy in [SamplingStrategy.CONDITIONAL, SamplingStrategy.SMART]:
            return self._smart_sampling_decision(tool_name, query, result, context)

        return False

    def _smart_sampling_decision(
        self,
        tool_name: str,
        query: Union[str, list],
        result: Any,
        context: Optional[Dict[str, Any]],
    ) -> bool:
        """Make intelligent sampling decisions based on multiple factors."""
        sampling_score = 0.0

        # Tool-specific base scores
        tool_scores = {
            "explain_concept": 0.9,
            "analyze_ids_structure": 0.8,
            "search_imas": 0.7,
            "get_overview": 0.7,
            "explore_relationships": 0.6,
            "export_ids": 0.3,
            "explore_identifiers": 0.4,
        }
        sampling_score += tool_scores.get(tool_name, 0.5)

        # Query complexity factors
        query_str = str(query) if isinstance(query, list) else query
        if len(query_str.split()) > 3:
            sampling_score += 0.1

        # Physics-related terms boost
        physics_terms = [
            "plasma", "magnetic", "temperature", "pressure", "equilibrium",
            "transport", "heating", "current", "profile", "disruption"
        ]
        if any(term in query_str.lower() for term in physics_terms):
            sampling_score += 0.1

        # Result-based factors
        result_count = self._get_result_count(result)
        if result_count == 0:
            sampling_score += 0.2  # Empty results need explanation
        elif result_count > 20:
            sampling_score += 0.15  # Large result sets benefit from summarization
        elif 3 <= result_count <= 10:
            sampling_score += 0.1  # Medium result sets are good candidates

        return sampling_score >= 0.8

    def _get_result_count(self, result: Any) -> int:
        """Extract result count from various result structures."""
        if hasattr(result, 'hits') and hasattr(result.hits, '__len__'):
            return len(result.hits)
        elif hasattr(result, 'hit_count'):
            return result.hit_count
        elif isinstance(result, dict):
            for field in ["hits", "results", "data", "paths", "items"]:
                if field in result and isinstance(result[field], list):
                    return len(result[field])
        elif isinstance(result, list):
            return len(result)
        return 0 if not result else 1
````

### 2.5.2 Create Tool Recommendation Service

**File: `imas_mcp/services/tool_recommendations.py`**

```python
"""Tool recommendation service for intelligent workflow guidance."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from .base import BaseService

class RecommendationStrategy(Enum):
    """Tool recommendation strategy options."""
    SEARCH_BASED = "search_based"
    CONCEPT_BASED = "concept_based"
    ANALYSIS_BASED = "analysis_based"
    EXPORT_BASED = "export_based"
    OVERVIEW_BASED = "overview_based"
    RELATIONSHIPS_BASED = "relationships_based"
    IDENTIFIERS_BASED = "identifiers_based"

class ToolRecommendationService(BaseService):
    """Service for generating intelligent tool recommendations."""

    def generate_recommendations(
        self,
        result: Any,
        strategy: RecommendationStrategy = RecommendationStrategy.SEARCH_BASED,
        max_tools: int = 4,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """
        Generate tool recommendations based on result analysis.

        Replaces the functionality of @recommend_tools decorator.
        """
        try:
            if self._has_errors(result):
                return self._generate_error_recommendations()

            if strategy == RecommendationStrategy.SEARCH_BASED:
                return self._generate_search_recommendations(result, query, max_tools)
            elif strategy == RecommendationStrategy.CONCEPT_BASED:
                return self._generate_concept_recommendations(result, max_tools)
            elif strategy == RecommendationStrategy.ANALYSIS_BASED:
                return self._generate_analysis_recommendations(result, max_tools)
            elif strategy == RecommendationStrategy.EXPORT_BASED:
                return self._generate_export_recommendations(result, max_tools)
            elif strategy == RecommendationStrategy.OVERVIEW_BASED:
                return self._generate_overview_recommendations(result, max_tools)
            elif strategy == RecommendationStrategy.RELATIONSHIPS_BASED:
                return self._generate_relationships_recommendations(result, max_tools)
            elif strategy == RecommendationStrategy.IDENTIFIERS_BASED:
                return self._generate_identifiers_recommendations(result, max_tools)

            return self._generate_generic_recommendations(max_tools)

        except Exception as e:
            self.logger.warning(f"Tool recommendation generation failed: {e}")
            return self._generate_fallback_recommendations()

    def _generate_search_recommendations(
        self, result: Any, query: Optional[str], max_tools: int
    ) -> List[Dict[str, str]]:
        """Generate recommendations for search results."""
        recommendations = []

        # Extract search context
        hits = getattr(result, 'hits', [])
        hit_count = len(hits) if hits else 0

        if hit_count > 0:
            # Extract IDS names and domains from results
            ids_names = set()
            domains = set()

            for hit in hits[:5]:  # Analyze top 5 hits
                path = getattr(hit, 'path', '')
                if path and '/' in path:
                    ids_names.add(path.split('/')[0])

                physics_domain = getattr(hit, 'physics_domain', None)
                if physics_domain:
                    domains.add(physics_domain)

            # Suggest structure analysis for found IDS
            for ids_name in list(ids_names)[:2]:
                recommendations.append({
                    "tool": "analyze_ids_structure",
                    "reason": f"Analyze detailed structure of {ids_name} IDS",
                    "description": f"Get comprehensive structural analysis of {ids_name}"
                })

            # Suggest concept explanation for domains
            for domain in list(domains)[:2]:
                recommendations.append({
                    "tool": "explain_concept",
                    "reason": f"Learn more about {domain} physics domain",
                    "description": f"Get detailed explanation of {domain} concepts"
                })

            # Suggest relationship exploration
            if hit_count >= 3:
                recommendations.append({
                    "tool": "explore_relationships",
                    "reason": f"Explore data relationships for the {hit_count} found paths",
                    "description": "Discover how these data paths connect to other IMAS structures"
                })

            # Suggest export for large result sets
            if hit_count >= 5:
                recommendations.append({
                    "tool": "export_ids",
                    "reason": f"Export data for the {len(ids_names)} IDS found",
                    "description": "Export structured data for use in analysis workflows"
                })

        else:
            # No results - suggest broader exploration
            recommendations.extend([
                {
                    "tool": "get_overview",
                    "reason": "No results found - get overview of available data",
                    "description": "Explore IMAS data structure and available concepts"
                },
                {
                    "tool": "explore_identifiers",
                    "reason": "Search for related terms and identifiers",
                    "description": "Discover alternative search terms and data identifiers"
                }
            ])

            if query:
                recommendations.append({
                    "tool": "explain_concept",
                    "reason": f'Learn about "{query}" concept in fusion physics',
                    "description": "Get conceptual understanding and context"
                })

        return recommendations[:max_tools]

    def _generate_concept_recommendations(
        self, result: Any, max_tools: int
    ) -> List[Dict[str, str]]:
        """Generate recommendations for concept explanation results."""
        concept = getattr(result, 'concept', 'physics concept')

        recommendations = [
            {
                "tool": "search_imas",
                "reason": f"Find data paths related to {concept}",
                "description": f"Search for IMAS data containing {concept} measurements"
            },
            {
                "tool": "explore_identifiers",
                "reason": f"Explore identifiers and terms related to {concept}",
                "description": "Discover related concepts and terminology"
            }
        ]

        # Add domain-specific suggestions
        concept_lower = str(concept).lower()

        if any(term in concept_lower for term in ["temperature", "density", "pressure"]):
            recommendations.append({
                "tool": "search_imas",
                "reason": "Explore core plasma profiles",
                "description": "Search for core_profiles data containing plasma parameters"
            })

        if any(term in concept_lower for term in ["magnetic", "field", "equilibrium"]):
            recommendations.append({
                "tool": "analyze_ids_structure",
                "reason": "Analyze equilibrium IDS structure",
                "description": "Examine magnetic equilibrium data organization"
            })

        return recommendations[:max_tools]

    def _generate_analysis_recommendations(
        self, result: Any, max_tools: int
    ) -> List[Dict[str, str]]:
        """Generate recommendations for analysis results."""
        return [
            {
                "tool": "export_ids",
                "reason": "Export analysis results for further processing",
                "description": "Save structured analysis data for workflows"
            },
            {
                "tool": "explore_relationships",
                "reason": "Explore relationships between analyzed components",
                "description": "Understand connections and dependencies"
            },
            {
                "tool": "search_imas",
                "reason": "Search for related data paths",
                "description": "Find additional relevant IMAS data"
            }
        ][:max_tools]

    def _generate_error_recommendations(self) -> List[Dict[str, str]]:
        """Generate recommendations for error cases."""
        return [
            {
                "tool": "get_overview",
                "reason": "Get overview of available data and functionality",
                "description": "Explore IMAS capabilities and data structure"
            }
        ]

    def _generate_fallback_recommendations(self) -> List[Dict[str, str]]:
        """Generate fallback recommendations when generation fails."""
        return [
            {
                "tool": "search_imas",
                "reason": "Search for specific data paths",
                "description": "Find relevant IMAS data for your research"
            },
            {
                "tool": "get_overview",
                "reason": "Get overview of IMAS structure",
                "description": "Understand available data and capabilities"
            }
        ]

    def _has_errors(self, result: Any) -> bool:
        """Check if result contains errors."""
        if hasattr(result, 'error') or (isinstance(result, dict) and 'error' in result):
            return True
        return False
```

### 2.5.3 Update BaseTool with Sampling & Recommendation Infrastructure

**File: `imas_mcp/tools/base.py`**

Add to imports:

```python
from imas_mcp.services.sampling import SamplingService, SamplingStrategy
from imas_mcp.services.tool_recommendations import ToolRecommendationService, RecommendationStrategy
```

Add to class BaseTool:

```python
    # Class variables for template method pattern
    sampling_strategy: SamplingStrategy = SamplingStrategy.NO_SAMPLING
    recommendation_strategy: RecommendationStrategy = RecommendationStrategy.SEARCH_BASED
    max_recommended_tools: int = 4
    enable_sampling: bool = False
    enable_recommendations: bool = True

    def __init__(self):
        """Initialize base tool with service dependencies."""
        # Existing services
        self.physics_service = PhysicsService()
        self.response_service = ResponseService()
        self.documents_service = DocumentService()
        self.search_config_service = SearchConfigurationService()

        # New services
        self.sampling_service = SamplingService()
        self.tool_recommendation_service = ToolRecommendationService()

    def apply_sampling(self, result: Any, **kwargs) -> Any:
        """
        Template method for applying sampling to tool results.
        Subclasses can customize by setting sampling_strategy class variable.
        """
        if not self.enable_sampling:
            return result

        return self.sampling_service.apply_sampling(
            result=result,
            strategy=self.sampling_strategy,
            **kwargs
        )

    def generate_tool_recommendations(
        self, result: Any, query: Optional[str] = None, **kwargs
    ) -> List[Dict[str, str]]:
        """
        Template method for generating tool recommendations.
        Subclasses can customize by setting recommendation_strategy class variable.
        """
        if not self.enable_recommendations:
            return []

        return self.tool_recommendation_service.generate_recommendations(
            result=result,
            strategy=self.recommendation_strategy,
            max_tools=self.max_recommended_tools,
            query=query,
            **kwargs
        )

    def apply_services(self, result: Any, **kwargs) -> Any:
        """
        Template method for applying all post-processing services.
        Called after tool execution but before response formatting.
        """
        # Apply sampling first
        if self.enable_sampling:
            result = self.apply_sampling(result, **kwargs)

        # Generate recommendations
        if self.enable_recommendations:
            recommendations = self.generate_tool_recommendations(result, **kwargs)
            # Attach recommendations to result metadata
            if hasattr(result, '__dict__'):
                result.tool_recommendations = recommendations
            elif isinstance(result, dict):
                result['tool_recommendations'] = recommendations

        return result
```

### 2.5.4 Update SearchTool with Sampling & Recommendations

**File: `imas_mcp/tools/search_tool.py`**

Add class variables to enable services:

```python
class SearchTool(BaseTool):
    """IMAS search tool with service composition and template methods."""

    # Enable both services for search tool
    enable_sampling: bool = True
    enable_recommendations: bool = True

    # Use search-appropriate strategies
    sampling_strategy = SamplingStrategy.SMART
    recommendation_strategy = RecommendationStrategy.SEARCH_BASED
    max_recommended_tools: int = 5
```

Update the search method to use services:

```python
    async def search_imas(
        self,
        query: Union[str, List[str]],
        search_mode: str = "auto",
        max_results: int = 10,
        ids_filter: Optional[Union[str, List[str]]] = None,
        enable_physics: bool = True,
    ) -> Dict[str, Any]:
        """Enhanced search with sampling and recommendations."""
        try:
            # Execute search with existing logic
            result = await self._execute_search(
                query, search_mode, max_results, ids_filter, enable_physics
            )

            # Apply post-processing services
            result = self.apply_services(
                result=result,
                query=query,
                search_mode=search_mode
            )

            # Build response with services
            return self.response_service.build_search_response(
                result=result,
                query=query,
                search_mode=search_mode,
                tool_name="search_imas"
            )

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return self._build_error_response(str(e))
```

### 2.5.5 Create Integration Tests for Services

**File: `tests/integration/test_sampling_service.py`**

```python
"""Integration tests for sampling service."""

import pytest
from unittest.mock import MagicMock, patch
from imas_mcp.services.sampling import SamplingService, SamplingStrategy
from imas_mcp.search.search_result import SearchResult

class TestSamplingServiceIntegration:
    """Test sampling service integration with MCP context."""

    @pytest.fixture
    def sampling_service(self):
        return SamplingService()

    @pytest.fixture
    def mock_result(self):
        """Mock search result for testing."""
        result = SearchResult(
            hits=[],
            query="test query",
            search_mode="semantic",
            total_hits=0
        )
        return result

    @patch('imas_mcp.services.sampling.mcp.get_mcp_context')
    def test_sampling_with_mcp_context(self, mock_get_context, sampling_service, mock_result):
        """Test sampling service uses MCP context correctly."""
        # Mock MCP context
        mock_context = MagicMock()
        mock_context.sample.return_value = "Sampled result"
        mock_get_context.return_value = mock_context

        # Apply sampling
        result = sampling_service.apply_sampling(
            result=mock_result,
            strategy=SamplingStrategy.SMART,
            temperature=0.5
        )

        # Verify MCP context was used
        assert mock_get_context.called
        assert mock_context.sample.called

    def test_no_sampling_strategy(self, sampling_service, mock_result):
        """Test NO_SAMPLING strategy returns original result."""
        result = sampling_service.apply_sampling(
            result=mock_result,
            strategy=SamplingStrategy.NO_SAMPLING
        )

        assert result == mock_result

    @patch('imas_mcp.services.sampling.mcp.get_mcp_context')
    def test_sampling_error_handling(self, mock_get_context, sampling_service, mock_result):
        """Test sampling gracefully handles errors."""
        # Mock MCP context to raise exception
        mock_get_context.side_effect = Exception("MCP context error")

        result = sampling_service.apply_sampling(
            result=mock_result,
            strategy=SamplingStrategy.SMART
        )

        # Should return original result on error
        assert result == mock_result
```

**File: `tests/integration/test_tool_recommendation_service.py`**

```python
"""Integration tests for tool recommendation service."""

import pytest
from imas_mcp.services.tool_recommendations import ToolRecommendationService, RecommendationStrategy
from imas_mcp.search.search_result import SearchResult
from imas_mcp.search.search_strategy import SearchHit

class TestToolRecommendationServiceIntegration:
    """Test tool recommendation service integration."""

    @pytest.fixture
    def recommendation_service(self):
        return ToolRecommendationService()

    @pytest.fixture
    def search_result_with_hits(self):
        """Search result with multiple hits for testing."""
        hits = [
            SearchHit(
                path="core_profiles/profiles_1d/temperature",
                score=0.95,
                physics_domain="core_transport"
            ),
            SearchHit(
                path="equilibrium/time_slice/boundary",
                score=0.87,
                physics_domain="equilibrium"
            )
        ]
        return SearchResult(
            hits=hits,
            query="temperature profile",
            search_mode="semantic",
            total_hits=2
        )

    def test_search_based_recommendations(self, recommendation_service, search_result_with_hits):
        """Test search-based recommendation generation."""
        recommendations = recommendation_service.generate_recommendations(
            result=search_result_with_hits,
            strategy=RecommendationStrategy.SEARCH_BASED,
            max_tools=4,
            query="temperature profile"
        )

        assert len(recommendations) > 0
        assert len(recommendations) <= 4

        # Check recommendation structure
        for rec in recommendations:
            assert "tool" in rec
            assert "reason" in rec
            assert "description" in rec

    def test_concept_based_recommendations(self, recommendation_service):
        """Test concept-based recommendations."""
        mock_result = MagicMock()
        mock_result.concept = "temperature"

        recommendations = recommendation_service.generate_recommendations(
            result=mock_result,
            strategy=RecommendationStrategy.CONCEPT_BASED,
            max_tools=3
        )

        assert len(recommendations) <= 3
        # Should include search_imas tool for finding related data
        tool_names = [rec["tool"] for rec in recommendations]
        assert "search_imas" in tool_names

    def test_error_result_recommendations(self, recommendation_service):
        """Test recommendations for error results."""
        error_result = {"error": "Search failed"}

        recommendations = recommendation_service.generate_recommendations(
            result=error_result,
            strategy=RecommendationStrategy.SEARCH_BASED
        )

        # Should provide helpful fallback recommendations
        assert len(recommendations) > 0
        tool_names = [rec["tool"] for rec in recommendations]
        assert "get_overview" in tool_names
```

**File: `tests/integration/test_service_composition.py`**

```python
"""Integration tests for service composition in tools."""

import pytest
from unittest.mock import patch, MagicMock
from imas_mcp.tools.search_tool import SearchTool

class TestServiceComposition:
    """Test service integration across tools."""

    @pytest.fixture
    def search_tool(self):
        return SearchTool()

    @patch('imas_mcp.services.sampling.mcp.get_mcp_context')
    def test_search_tool_with_services(self, mock_get_context, search_tool):
        """Test SearchTool uses both sampling and recommendation services."""
        # Mock MCP context
        mock_context = MagicMock()
        mock_context.sample.return_value = "Sampled response"
        mock_get_context.return_value = mock_context

        # Mock search execution
        with patch.object(search_tool, '_execute_search') as mock_search:
            mock_result = MagicMock()
            mock_result.hits = []
            mock_result.total_hits = 0
            mock_search.return_value = mock_result

            # Execute search
            result = search_tool.search_imas(
                query="test query",
                search_mode="semantic"
            )

            # Verify services were integrated
            assert mock_search.called
            # Result should contain service-generated content
            assert isinstance(result, dict)

    def test_service_dependency_injection(self, search_tool):
        """Test all services are properly injected."""
        # Verify all required services are present
        assert hasattr(search_tool, 'sampling_service')
        assert hasattr(search_tool, 'tool_recommendation_service')
        assert hasattr(search_tool, 'physics_service')
        assert hasattr(search_tool, 'response_service')
        assert hasattr(search_tool, 'documents_service')
        assert hasattr(search_tool, 'search_config_service')

    def test_template_method_customization(self, search_tool):
        """Test template method pattern allows tool-specific customization."""
        # Verify SearchTool has appropriate settings
        assert search_tool.enable_sampling == True
        assert search_tool.enable_recommendations == True
        assert search_tool.max_recommended_tools == 5
```

**File: `imas_mcp/tools/base.py` (Updated sections)**

```python
from imas_mcp.services import (
    PhysicsService,
    ResponseService,
    DocumentService,
    SearchConfigurationService,
    SamplingService,  # Add sampling service
)
from imas_mcp.services.sampling import SamplingStrategy

class BaseTool(ABC):
    """Base class for all IMAS MCP tools with service injection."""

    # Default sampling configuration (tools can override)
    SAMPLING_TEMPERATURE: float = 0.3
    SAMPLING_MAX_TOKENS: int = 800
    SAMPLING_STRATEGY: SamplingStrategy = SamplingStrategy.SMART

    def __init__(self, document_store: Optional[DocumentStore] = None):
        # ... existing initialization ...
        self.sampling = SamplingService()  # Add sampling service

    def get_sampling_config(self) -> Dict[str, Any]:
        """Get tool-specific sampling configuration."""
        return {
            "temperature": self.SAMPLING_TEMPERATURE,
            "max_tokens": self.SAMPLING_MAX_TOKENS,
            "strategy": self.SAMPLING_STRATEGY,
        }

    def build_sample_prompt(
        self,
        query: Union[str, List[str]],
        result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build sample prompt for LLM client sampling.

        Template method - tools should override to customize prompting.
        """
        query_str = query if isinstance(query, str) else " ".join(query)
        tool_name = self.get_tool_name()

        return f"""Tool: {tool_name}
Query: "{query_str}"

Please provide analysis and insights for this {tool_name} result.

Include:
1. Context and significance of the results
2. Recommended follow-up actions
3. Relevant IMAS data patterns
4. Physics context where applicable"""

    async def apply_sampling_if_needed(
        self,
        query: Union[str, List[str]],
        result: Any,
        ctx: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Apply sampling to result if conditions are met.

        Main entry point for sampling integration in tool methods.
        """
        # Check if sampling should be applied
        if not self.should_sample(query, result, context) or ctx is None:
            return result

        # Build sample prompt
        sample_prompt = self.build_sample_prompt(query, result, context)

        # Apply sampling
        config = self.get_sampling_config()
        sample_result = await self.sampling.apply_sample(
            sample_prompt=sample_prompt,
            ctx=ctx,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )

        # Process and integrate result
        return await self.process_sample_result(sample_result, result)
```

### 2.5.3 Refactor SearchTool for Sampling Service

**File: `imas_mcp/tools/search_tool.py` (Updated sections)**

```python
class SearchTool(BaseTool):
    """Tool for searching IMAS data paths using service composition."""

    # SearchTool-specific sampling configuration
    SAMPLING_TEMPERATURE = 0.3
    SAMPLING_MAX_TOKENS = 800
    SAMPLING_STRATEGY = SamplingStrategy.CONDITIONAL

    def build_sample_prompt(
        self,
        query: Union[str, List[str]],
        result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build search-specific sample prompt."""
        query_str = query if isinstance(query, str) else " ".join(query)

        # Get result count for prompt customization
        result_count = self.sampling._get_result_count(result)

        if result_count == 0:
            return f"""Search Query Analysis: "{query_str}"

No results were found for this query in the IMAS data dictionary.

Please provide:
1. Suggestions for alternative search terms or queries
2. Possible related IMAS concepts or data paths
3. Common physics contexts where this term might appear
4. Recommended follow-up searches"""

        # Extract top results for prompt (implementation details)
        # ... sample prompt building logic ...

    async def search_imas(self, ...):
        """Search method with sampling integration."""
        # ... existing search logic ...

        # Apply sampling if needed (replaces @sample decorator)
        response = await self.apply_sampling_if_needed(
            query=query,
            result=response,
            ctx=ctx,
            context={"search_mode": str(config.search_mode), "result_count": len(search_results)}
        )

        return response
```

### 2.5.4 Sampling Service Integration Tests

**File: `tests/services/test_sampling_service.py`**

```python
"""Tests for SamplingService."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from imas_mcp.services.sampling import SamplingService, SamplingStrategy

class TestSamplingService:
    """Test sampling service functionality."""

    @pytest.mark.asyncio
    async def test_apply_sample_success(self, sampling_service):
        """Test successful LLM sampling."""
        # Mock MCP context and test sampling

    def test_should_sample_strategies(self, sampling_service):
        """Test different sampling strategies."""
        # Test NEVER, ALWAYS, CONDITIONAL, SMART strategies

    def test_smart_sampling_decision_factors(self, sampling_service):
        """Test smart sampling decision factors."""
        # Test physics terms, complexity, result counts
```

**File: `tests/tools/test_search_tool_sampling.py`**

```python
"""Tests for SearchTool sampling integration."""

import pytest
from imas_mcp.tools.search_tool import SearchTool
from imas_mcp.services.sampling import SamplingStrategy

class TestSearchToolSampling:
    """Test SearchTool sampling service integration."""

    @pytest.mark.asyncio
    async def test_sampling_integration(self, search_tool):
        """Test complete sampling integration."""
        # Test end-to-end sampling with SearchTool

    def test_build_sample_prompt_customization(self, search_tool):
        """Test SearchTool-specific sample prompt building."""
        # Test prompt customization for search results
```

## Phase 3: Rollout to Remaining Tools (Week 3)

### 3.1 ExplainTool with Service Composition

**File: `imas_mcp/tools/explain_tool.py`** (Key sections)

```python
"""Explain tool implementation with service composition."""

from typing import Optional, Union
from fastmcp import Context

from imas_mcp.models.response_models import ConceptResult, ErrorResponse
from imas_mcp.models.request_models import ExplainInput
from imas_mcp.models.constants import DetailLevel, SearchMode
from imas_mcp.core.data_model import IdsNode, PhysicsContext
from imas_mcp.search.decorators import (
    cache_results,
    validate_input,
    measure_performance,
    handle_errors,
)
from .base import BaseTool

class ExplainTool(BaseTool):
    """Tool for explaining IMAS concepts using service composition."""

    def get_tool_name(self) -> str:
        return "explain_concept"

    @cache_results(ttl=600, key_strategy="semantic")
    @validate_input(schema=ExplainInput)
    @measure_performance(include_metrics=True, slow_threshold=2.0)
    @handle_errors(fallback="concept_suggestions")
    @mcp_tool("Explain IMAS concepts with physics context")
    async def explain_concept(
        self,
        concept: str,
        detail_level: DetailLevel = DetailLevel.INTERMEDIATE,
        ctx: Optional[Context] = None,
    ) -> Union[ConceptResult, ErrorResponse]:
        """Explain IMAS concepts using service composition."""

        # Create search configuration for concept exploration
        config = self.search_config.create_config(
            search_mode=SearchMode.SEMANTIC,
            max_results=15,
            enable_physics=True,
        )

        # Search for concept-related content
        search_results = await self._search_service.search(concept, config)

        # Get physics context using service
        physics_context = await self.physics.get_concept_context(concept, detail_level.value)

        if not search_results:
            return ConceptResult(
                concept=concept,
                detail_level=detail_level,
                explanation="No information found for concept",
                related_topics=self._build_fallback_suggestions(concept),
                concept_explanation=None,
                nodes=[],
                physics_domains=[],
                physics_context=physics_context,
            )

        # Build nodes and extract physics domains using service
        nodes, physics_domains = self._build_concept_nodes(search_results)

        # Build comprehensive explanation
        explanation = self._build_concept_explanation(concept, physics_domains, len(search_results))

        # Build response
        response = ConceptResult(
            concept=concept,
            explanation=explanation,
            detail_level=detail_level,
            related_topics=self._build_related_topics(search_results, physics_domains),
            concept_explanation=None,  # AI enhancement will populate this
            nodes=nodes,
            physics_domains=physics_domains,
            physics_context=physics_context,
        )

        # Add metadata
        return self.response.add_standard_metadata(response, self.get_tool_name())
```

### 3.2 Integration Tests for Multi-Tool Service Usage

**File: `tests/integration/test_service_integration.py`**

```python
"""Integration tests for service composition across tools."""

import pytest
from unittest.mock import patch, MagicMock
from imas_mcp.tools import SearchTool, ExplainTool
from imas_mcp.services import PhysicsService, ResponseService

class TestServiceIntegration:
    """Test service integration across multiple tools."""

    @pytest.fixture
    def tools(self):
        with patch('imas_mcp.tools.base.DocumentStore'):
            search_tool = SearchTool()
            explain_tool = ExplainTool()
            return search_tool, explain_tool

    @pytest.mark.asyncio
    async def test_physics_service_consistency(self, tools):
        """Test physics service provides consistent results across tools."""
        search_tool, explain_tool = tools

        # Mock physics service methods
        mock_physics_result = MagicMock()
        mock_physics_result.physics_matches = []

        with patch.object(PhysicsService, 'enhance_query', return_value=mock_physics_result) as mock_enhance:
            with patch.object(PhysicsService, 'get_concept_context', return_value={"domain": "test"}) as mock_context:

                # Mock other dependencies
                search_tool._search_service.search = AsyncMock(return_value=[])
                explain_tool._search_service.search = AsyncMock(return_value=[])
                search_tool.response.build_search_response = MagicMock()

                # Test search tool
                await search_tool.search_imas("plasma", ctx=MagicMock())

                # Test explain tool
                await explain_tool.explain_concept("plasma", ctx=MagicMock())

                # Verify physics service was used consistently
                mock_enhance.assert_called()
                mock_context.assert_called()

    @pytest.mark.asyncio
    async def test_response_service_metadata_consistency(self, tools):
        """Test response service adds consistent metadata."""
        search_tool, explain_tool = tools

        # Mock dependencies
        search_tool._search_service.search = AsyncMock(return_value=[])
        explain_tool._search_service.search = AsyncMock(return_value=[])

        mock_search_response = MagicMock()
        mock_concept_response = MagicMock()

        with patch.object(ResponseService, 'add_standard_metadata') as mock_metadata:
            mock_metadata.side_effect = [mock_search_response, mock_concept_response]

            # Execute both tools
            await search_tool.search_imas("test")
            await explain_tool.explain_concept("test")

            # Verify metadata was added to both responses
            assert mock_metadata.call_count == 2

            # Check tool names were passed correctly
            calls = mock_metadata.call_args_list
            assert calls[0][0][1] == "search_imas"
            assert calls[1][0][1] == "explain_concept"
```

## Phase 4: Advanced Service Features (Week 4)

### 4.1 Cross-IDS Analysis Service

**File: `imas_mcp/services/cross_ids_analysis.py`**

```python
"""Service for cross-IDS relationship analysis."""

from typing import List, Dict, Any
from imas_mcp.search.search_strategy import SearchConfig
from imas_mcp.models.constants import SearchMode
from .base import BaseService

class CrossIdsAnalysisService(BaseService):
    """Service for analyzing relationships between IDS."""

    def __init__(self, search_service, document_service):
        super().__init__()
        self.search_service = search_service
        self.document_service = document_service

    async def analyze_relationships(self, ids_list: List[str]) -> Dict[str, Any]:
        """
        Analyze relationships between multiple IDS.

        Args:
            ids_list: List of IDS names to analyze

        Returns:
            Dictionary with relationship analysis
        """
        if len(ids_list) < 2:
            return {"error": "At least 2 IDS required for relationship analysis"}

        relationship_analysis = {}

        for i, ids1 in enumerate(ids_list):
            for ids2 in ids_list[i + 1:]:
                relationship_key = f"{ids1}_{ids2}"
                try:
                    # Search for relationships between IDS pairs
                    config = SearchConfig(
                        search_mode=SearchMode.SEMANTIC,
                        max_results=8,
                    )

                    search_results = await self.search_service.search(
                        f"{ids1} {ids2} relationships physics",
                        config
                    )

                    if search_results:
                        # Analyze physics connections
                        physics_connections = [
                            r for r in search_results
                            if r.document.metadata.physics_domain
                        ]

                        relationship_analysis[relationship_key] = {
                            "shared_concepts": len(search_results),
                            "physics_connections": len(physics_connections),
                            "top_connections": [
                                {
                                    "path": r.document.metadata.path_name,
                                    "relevance_score": r.score,
                                    "physics_domain": r.document.metadata.physics_domain,
                                    "context": r.document.documentation[:100],
                                }
                                for r in search_results[:3]
                            ],
                        }
                    else:
                        relationship_analysis[relationship_key] = {
                            "shared_concepts": 0,
                            "physics_connections": 0,
                            "top_connections": [],
                        }

                except Exception as e:
                    self.logger.error(f"Relationship analysis failed for {ids1}-{ids2}: {e}")
                    relationship_analysis[relationship_key] = {"error": str(e)}

        return relationship_analysis
```

### 4.2 Enhanced Export Tool with Cross-IDS Service

**File: `imas_mcp/tools/export_tool.py`** (Key sections)

```python
"""Export tool with cross-IDS analysis service."""

from typing import List, Optional, Union
from fastmcp import Context

from imas_mcp.search.document_store import DocumentStore
from imas_mcp.models.request_models import ExportIdsInput
from imas_mcp.models.response_models import IDSExport, ErrorResponse
from imas_mcp.search.decorators import (
    cache_results,
    validate_input,
    measure_performance,
    handle_errors,
)
from imas_mcp.services.cross_ids_analysis import CrossIdsAnalysisService
from .base import BaseTool

class ExportTool(BaseTool):
    """Tool for exporting IDS and physics domain data."""

    def __init__(self, document_store: Optional[DocumentStore] = None):
        super().__init__(document_store)
        # Initialize cross-IDS analysis service
        self.cross_ids_analysis = CrossIdsAnalysisService(
            self._search_service,
            self.documents
        )

    @cache_results(ttl=600, key_strategy="content_based")
    @validate_input(schema=ExportIdsInput)
    @measure_performance(include_metrics=True, slow_threshold=5.0)
    @handle_errors(fallback="export_suggestions")
    @mcp_tool("Export bulk IMAS data for multiple IDS")
    async def export_ids(
        self,
        ids_list: List[str],
        include_relationships: bool = True,
        include_physics: bool = True,
        output_format: str = "structured",
        ctx: Optional[Context] = None,
    ) -> Union[IDSExport, ErrorResponse]:
        """Export bulk IMAS data using service composition."""

        if not ids_list:
            return self._create_error_response("No IDS specified for bulk export")

        # Check IDS using document service
        valid_ids, invalid_ids = await self.documents.validate_ids(ids_list)

        if not valid_ids:
            return IDSExport(
                ids_names=ids_list,
                include_physics=include_physics,
                include_relationships=include_relationships,
                output_format=output_format,
                data={
                    "error": "No valid IDS names provided",
                    "invalid_ids": invalid_ids,
                    "suggestions": ["Check IDS name spelling", "Use get_overview to see available IDS"],
                },
            )

        # Build export data for each IDS
        export_data = await self._build_export_data(valid_ids, invalid_ids, output_format)

        # Add cross-IDS relationship analysis using service
        if include_relationships and len(valid_ids) > 1:
            cross_relationships = await self.cross_ids_analysis.analyze_relationships(valid_ids)
            export_data["cross_relationships"] = cross_relationships

        # Build final response
        response = IDSExport(
            ids_names=ids_list,
            include_physics=include_physics,
            include_relationships=include_relationships,
            output_format=output_format,
            data=export_data,
            metadata={"export_timestamp": "2024-01-01T00:00:00Z"},
        )

        return self.response.add_standard_metadata(response, self.get_tool_name())
```

### 4.3 Comprehensive Service Tests

**File: `tests/services/test_cross_ids_analysis.py`**

```python
"""Tests for CrossIdsAnalysisService."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from imas_mcp.services.cross_ids_analysis import CrossIdsAnalysisService

class TestCrossIdsAnalysisService:
    """Test cross-IDS analysis service functionality."""

    @pytest.fixture
    def service(self):
        mock_search_service = AsyncMock()
        mock_document_service = MagicMock()
        return CrossIdsAnalysisService(mock_search_service, mock_document_service)

    @pytest.mark.asyncio
    async def test_analyze_relationships_success(self, service):
        """Test successful relationship analysis between IDS."""

        # Mock search results
        mock_result1 = MagicMock()
        mock_result1.document.metadata.path_name = "core_profiles/temperature"
        mock_result1.document.metadata.physics_domain = "core_plasma"
        mock_result1.document.documentation = "Temperature profile data"
        mock_result1.score = 0.9

        mock_result2 = MagicMock()
        mock_result2.document.metadata.path_name = "equilibrium/magnetic_field"
        mock_result2.document.metadata.physics_domain = "equilibrium"
        mock_result2.document.documentation = "Magnetic field equilibrium"
        mock_result2.score = 0.8

        service.search_service.search = AsyncMock(return_value=[mock_result1, mock_result2])

        result = await service.analyze_relationships(["core_profiles", "equilibrium"])

        # Verify structure
        assert "core_profiles_equilibrium" in result
        relationship = result["core_profiles_equilibrium"]

        assert relationship["shared_concepts"] == 2
        assert relationship["physics_connections"] == 2
        assert len(relationship["top_connections"]) == 2

        # Verify connection details
        connection = relationship["top_connections"][0]
        assert connection["path"] == "core_profiles/temperature"
        assert connection["physics_domain"] == "core_plasma"
        assert connection["relevance_score"] == 0.9

    @pytest.mark.asyncio
    async def test_analyze_relationships_insufficient_ids(self, service):
        """Test relationship analysis with insufficient IDS."""

        result = await service.analyze_relationships(["single_ids"])

        assert "error" in result
        assert "At least 2 IDS required" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_relationships_search_failure(self, service):
        """Test relationship analysis with search service failure."""

        service.search_service.search = AsyncMock(side_effect=Exception("Search failed"))

        result = await service.analyze_relationships(["ids1", "ids2"])

        assert "ids1_ids2" in result
        assert "error" in result["ids1_ids2"]
        assert "Search failed" in result["ids1_ids2"]["error"]
```

## Phase 5: Documentation and Migration (Week 5)

### 5.1 Service Architecture Documentation

**File: `docs/SERVICE_ARCHITECTURE.md`**

````markdown
# Service Architecture Documentation

## Overview

The IMAS MCP tools use a service-oriented architecture with composition pattern for clear separation of concerns:

- **Decorators**: Handle cross-cutting concerns (caching, validation, error handling)
- **Services**: Contain business logic (physics integration, response building, document operations)
- **Tools**: Orchestrate services and handle MCP protocol

## Service Responsibilities

### PhysicsService

- Physics context enhancement for queries and concepts
- Integration with physics_search functionality
- Error handling for physics operations

### ResponseService

- Standardized Pydantic model construction
- Metadata addition and response formatting
- Consistent API response structure

### DocumentService

- Document store access and validation
- IDS validation and error responses
- Safe document retrieval operations

### SearchConfigurationService

- Search configuration optimization
- Query-based parameter tuning
- Performance optimization logic

### CrossIdsAnalysisService

- Multi-IDS relationship analysis
- Physics domain connection discovery
- Complex analytical operations

## Usage Patterns

### Basic Service Injection

```python
class MyTool(BaseTool):
    def __init__(self, document_store=None):
        super().__init__(document_store)
        # Services available as self.physics, self.response, etc.
```
````

### Service Method Calls

```python
async def my_tool_method(self, query: str):
    # Use services for business logic
    physics_context = await self.physics.enhance_query(query)
    config = self.search_config.create_config(search_mode="semantic")
    response = self.response.build_search_response(results, query, mode)
    return self.response.add_standard_metadata(response, self.get_tool_name())
```

````

### 5.2 Migration Guide

**File: `docs/MIGRATION_GUIDE.md`**

```markdown
# Migration Guide: Service Composition Architecture

## Breaking Changes

### Tool Method Signatures
No changes to public API - all tool method signatures remain the same.

### Internal Implementation Changes
- Tools now use service composition instead of inline business logic
- Decorators reduced to essential cross-cutting concerns only
- Response building standardized through ResponseService

## Testing Changes

### Service Mocking
```python
# Old approach - mock internal methods
with patch.object(tool, '_some_internal_method'):

# New approach - mock services
with patch.object(tool.physics, 'enhance_query'):
    with patch.object(tool.response, 'build_search_response'):
````

### Behavior-Focused Testing

```python
# Test behaviors and outcomes, not implementation details
async def test_search_with_physics_enhancement():
    result = await tool.search_imas("plasma", ctx=mock_ctx)

    # Verify behavior: physics enhancement occurred
    assert result.ai_insights['physics_context'] is not None

    # Verify outcome: proper response structure
    assert isinstance(result, SearchResponse)
    assert result.metadata['tool'] == 'search_imas'
```

## Benefits

1. **Testability**: Services can be individually unit tested
2. **Maintainability**: Business logic separated from infrastructure
3. **Reusability**: Services shared across multiple tools
4. **Clarity**: Clear separation of concerns and responsibilities

```

## Success Metrics

### Phase Completion Criteria

**Phase 1**: All services implemented and unit tested
- [ ] 5 service classes created
- [ ] 15+ unit tests passing
- [ ] Services integrate with BaseTool

**Phase 2**: SearchTool refactored and tested
- [ ] SearchTool uses service composition
- [ ] All existing SearchTool tests pass
- [ ] New service integration tests pass

**Phase 3**: All tools refactored
- [ ] 8 tools use service composition
- [ ] Integration tests validate cross-tool consistency
- [ ] Performance benchmarks meet/exceed baseline

**Phase 4**: Advanced features implemented
- [ ] Cross-IDS analysis service functional
- [ ] Export tools have feature parity with old implementation
- [ ] All critical features from comparison report implemented

**Phase 5**: Documentation and deployment
- [ ] Architecture documentation complete
- [ ] Migration guide published
- [ ] All tests passing (100+ tests)
- [ ] Performance validation complete

### Quality Gates

1. **Code Coverage**: Minimum 90% test coverage for services
2. **Performance**: Response times within 10% of baseline
3. **Feature Parity**: All features from comparison report implemented
4. **API Compatibility**: No breaking changes to public tool APIs

## Risk Mitigation

- **Incremental Rollout**: Implement one tool at a time
- **Comprehensive Testing**: Behavior-focused tests for each phase
- **Performance Monitoring**: Continuous benchmarking during migration
- **Rollback Plan**: Keep service interfaces simple for easy fallback

This plan provides a clear path to implement service composition while maintaining Pydantic models, achieving better separation of concerns, and ensuring comprehensive test coverage at each phase.
```
