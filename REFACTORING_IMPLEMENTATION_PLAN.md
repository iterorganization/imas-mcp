# IMAS MCP Refactoring Implementation Plan

## Overview

This document outlines a comprehensive refactoring plan to transform the monolithic IMAS MCP codebase into a clean, maintainable, and extensible system using the enhanced decorator pattern and modular architecture.

## Goals

- **Reduce complexity**: Break down 1800+ line monolithic Tools class
- **Improve maintainability**: Separate concerns using decorator pattern
- **Enhance testability**: Create focused, independently testable components
- **Increase reusability**: Make decorators and services reusable across tools
- **Better organization**: Restructure code into logical modules

## Current Problems

1. **Monolithic Tools class** (1800+ lines) with mixed responsibilities
2. **search_imas method** has 11 dependent private methods (160+ lines)
3. **Abandoned decorator pattern** that was actually working well
4. **Mixed concerns**: search, caching, validation, AI enhancement all coupled
5. **Poor testability**: Hard to test individual components in isolation

## Proposed Architecture

### Enhanced Decorator Pattern

```python
@cache_results(ttl=300, key_strategy="semantic")
@validate_input(schema=SearchInputSchema)
@ai_enhance(temperature=0.3, max_tokens=800)
@suggest_tools(strategy="search_based")
@measure_performance
@handle_errors(fallback="search_suggestions")
@mcp_tool("Search for IMAS data paths")
async def search_imas(self, query: str, ...) -> Dict[str, Any]:
    # Clean 20-30 lines of core orchestration only
    config = SearchConfig(mode=search_mode, max_results=max_results)
    results = await self.search_engine.search(query, config)
    return SearchResponse(results=results).model_dump()
```

## New Directory Structure

```
imas_mcp/
├── tools/                              # SPLIT: From monolithic tools.py (1800+ lines)
│   ├── __init__.py                     # Main Tools class that delegates
│   ├── base.py                         # Base tool functionality
│   ├── search.py                       # search_imas logic
│   ├── explain.py                      # explain_concept logic
│   ├── overview.py                     # get_overview logic
│   ├── relationships.py               # explore_relationships logic
│   ├── identifiers.py                 # explore_identifiers logic
│   ├── export.py                       # export_* tools logic
│   └── analysis.py                     # analyze_ids_structure logic
├── providers.py                        # KEEP: MCPProvider base class (25 lines)
├── resources/                          # SPLIT: From resources.py (150 lines)
│   ├── __init__.py                     # Main Resources class
│   └── schema.py                       # Schema resource implementations
├── search/
│   ├── decorators/                     # NEW: Focused decorators
│   │   ├── __init__.py
│   │   ├── cache.py                   # @cache_results decorator
│   │   ├── validation.py              # @validate_input decorator
│   │   ├── ai_enhancement.py          # @ai_enhance decorator
│   │   ├── suggestions.py             # @suggest_tools decorator
│   │   ├── performance.py             # @measure_performance decorator
│   │   └── error_handling.py          # @handle_errors decorator
│   ├── engines/                       # NEW: Pure search logic
│   │   ├── __init__.py
│   │   ├── base_engine.py
│   │   ├── semantic_engine.py
│   │   ├── lexical_engine.py
│   │   └── hybrid_engine.py
│   ├── services/                      # NEW: Business logic services
│   │   ├── __init__.py
│   │   ├── search_service.py          # Core search orchestration
│   │   ├── suggestion_service.py      # Tool suggestion logic
│   │   └── enhancement_service.py     # AI enhancement logic
│   └── schemas/                       # NEW: Input validation schemas
│       ├── __init__.py
│       ├── search_schemas.py
│       ├── export_schemas.py
│       └── analysis_schemas.py
├── core/                              # EXISTING: Keep as is
├── models/                            # EXISTING: Keep as is
├── physics_integration/               # EXISTING: Keep as is
└── utils/                             # EXISTING: Keep as is
```

---

## Implementation Phases

## Phase 1: Foundation Setup ✅ **COMPLETE**

### 1.1 Create New Directory Structure ✅ **COMPLETE**

- [x] ✅ **IMPLEMENTED** - Create `imas_mcp/tools/` directory
- [x] ✅ **IMPLEMENTED** - Create `imas_mcp/resources/` directory
- [x] ✅ **IMPLEMENTED** - Create `imas_mcp/search/decorators/` directory
- [x] ✅ **IMPLEMENTED** - Create `imas_mcp/search/engines/` directory
- [x] ✅ **IMPLEMENTED** - Create `imas_mcp/search/services/` directory
- [x] ✅ **IMPLEMENTED** - Create `imas_mcp/search/schemas/` directory

### 1.2 Split Large Files ✅ **COMPLETE**

- [x] ✅ **IMPLEMENTED** - Split `imas_mcp/tools.py` → `imas_mcp/tools/` modules (1800+ lines → 8 focused modules)
  - Status: ✅ **IMPLEMENTED**
  - Description: Monolithic Tools class split into focused tool modules
  - Key components: Search, Explain, Overview, Analysis, Relationships, Identifiers, Export tools
  - Files: `imas_mcp/tools/{search,explain,overview,analysis,relationships,identifiers,export,base}.py`
- [x] ✅ **IMPLEMENTED** - Split `imas_mcp/resources.py` → `imas_mcp/resources/schema.py` (150 lines)
  - Status: ✅ **IMPLEMENTED**
  - Description: Resources class moved to dedicated package
  - Key components: Schema resource serving with MCP integration
  - File: `imas_mcp/resources/schema.py`
- [x] ✅ **IMPLEMENTED** - Keep `imas_mcp/providers.py` as-is (25 lines, simple base class)
- [x] ✅ **IMPLEMENTED** - Update imports to reflect new structure

### 1.3 Create Base Classes ✅ **COMPLETE**

- [x] ✅ **IMPLEMENTED** - `imas_mcp/tools/base.py`
  - Status: ✅ **IMPLEMENTED**
  - Description: BaseTool abstract base class with common functionality
  - Key features: Abstract get_tool_name() method, standardized error response creation, logging support
  - File: `imas_mcp/tools/base.py` (26 lines)
  - Tests: `tests/test_base_tool.py` (7 tests passing)
- [x] ✅ **IMPLEMENTED** - `imas_mcp/search/engines/base_engine.py`
  - Status: ✅ **IMPLEMENTED**
  - Description: SearchEngine abstract base class for pure search logic
  - Key features: Abstract search interface, query validation, normalization, MockSearchEngine for testing
  - File: `imas_mcp/search/engines/base_engine.py` (174 lines)
  - Tests: `tests/test_search_engines.py` (22 tests passing)
- [x] ✅ **IMPLEMENTED** - `imas_mcp/search/services/search_service.py`
  - Status: ✅ **IMPLEMENTED**
  - Description: SearchService for orchestrating different search engines
  - Key features: Engine selection, mode resolution, health checking, SearchRequest/Response models
  - File: `imas_mcp/search/services/search_service.py` (241 lines)
  - Tests: `tests/test_search_service.py` (12 tests passing)

**Deliverables: ✅ ALL COMPLETE**

- ✅ New directory structure in place (6 packages created)
- ✅ Existing code moved without maintaining backwards compatibility
- ✅ Base classes defined with clear interfaces
- ✅ Comprehensive test suite (41 tests passing)
- ✅ Clean separation of concerns and abstractions ready for Phase 2

---

## Phase 2: Extract Search Engine (Days 3-4)

### 2.1 Create Pure Search Engine

```python
# imas_mcp/search/engines/base_engine.py
class SearchEngine:
    async def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Pure search logic - no caching, no AI, no suggestions"""
        pass

# imas_mcp/search/engines/semantic_engine.py
class SemanticSearchEngine(SearchEngine):
    async def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        # Semantic search implementation
        pass
```

### 2.2 Extract Search Logic from Tools

- [ ] Extract search logic from `search_imas` method
- [ ] Move to `SemanticSearchEngine`, `LexicalSearchEngine`, `HybridSearchEngine`
- [ ] Create `SearchConfig` for engine configuration
- [ ] Create `SearchResult` models for engine output

### 2.3 Create Search Service

```python
# imas_mcp/search/services/search_service.py
class SearchService:
    def __init__(self, engines: Dict[SearchMode, SearchEngine]):
        self.engines = engines

    async def search(self, request: SearchRequest) -> SearchResponse:
        engine = self.engines[request.mode]
        results = await engine.search(request.query, request.config)
        return SearchResponse(results=results)
```

**Deliverables:**

- Separate search engines for each mode
- Search service orchestrating engine selection
- Clean separation of search logic from tools

---

## Phase 3: Implement Core Decorators (Days 5-7)

### 3.1 Cache Decorator

```python
# imas_mcp/search/decorators/cache.py
def cache_results(ttl: int = 300, key_strategy: str = "semantic"):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = build_cache_key(args, kwargs, strategy=key_strategy)
            if cached := cache.get(cache_key):
                return cached
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return result
        return wrapper
    return decorator
```

### 3.2 Validation Decorator

```python
# imas_mcp/search/decorators/validation.py
def validate_input(schema: Type[BaseModel]):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract 'self' and create request object
            instance = args[0]
            request_data = {k: v for k, v in kwargs.items() if k != 'ctx'}

            try:
                validated = schema(**request_data)
                # Replace kwargs with validated data
                kwargs.update(validated.model_dump())
                return await func(*args, **kwargs)
            except ValidationError as e:
                return instance._create_error_response(str(e), request_data.get('query', ''))
        return wrapper
    return decorator
```

### 3.3 AI Enhancement Decorator

```python
# imas_mcp/search/decorators/ai_enhancement.py
def ai_enhance(temperature: float = 0.3, max_tokens: int = 800):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            ctx = kwargs.get('ctx')
            result = await func(*args, **kwargs)

            if ctx and 'ai_prompt' in result:
                enhancement = await apply_ai_enhancement(
                    result['ai_prompt'], ctx, temperature, max_tokens
                )
                result['ai_insights'] = enhancement

            return result
        return wrapper
    return decorator
```

### 3.4 Suggestions Decorator

```python
# imas_mcp/search/decorators/suggestions.py
def suggest_tools(strategy: str = "search_based"):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            if 'error' not in result and 'hits' in result:
                suggestions = generate_tool_suggestions(result, strategy)
                result['suggestions'] = suggestions

            return result
        return wrapper
    return decorator
```

**Deliverables:**

- Four core decorators implemented and tested
- Clean separation of cross-cutting concerns
- Reusable decorators for all tools

---

## Phase 4: Refactor Search Tool (Days 8-9)

### 4.1 Create Search Input Schema

```python
# imas_mcp/search/schemas/search_schemas.py
class SearchInputSchema(BaseModel):
    query: Union[str, List[str]]
    ids_name: Optional[str] = None
    max_results: int = Field(default=10, ge=1, le=100)
    search_mode: str = Field(default="auto")

    @field_validator('search_mode')
    def validate_search_mode(cls, v):
        valid_modes = ["auto", "semantic", "lexical", "hybrid"]
        if v.lower() not in valid_modes:
            raise ValueError(f"Invalid search_mode. Must be one of: {valid_modes}")
        return v.lower()
```

### 4.2 Create Simplified Search Tool

```python
# imas_mcp/tools/search.py
class SearchTool:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service

    @cache_results(ttl=300, key_strategy="semantic")
    @validate_input(SearchInputSchema)
    @ai_enhance(temperature=0.3, max_tokens=800)
    @suggest_tools(strategy="search_based")
    @measure_performance
    @handle_errors(fallback="search_suggestions")
    @mcp_tool("Search for IMAS data paths with relevance-ordered results")
    async def search_imas(
        self,
        query: Union[str, List[str]],
        ids_name: Optional[str] = None,
        max_results: int = 10,
        search_mode: str = "auto",
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Search for IMAS data paths with relevance-ordered results.

        Clean, focused implementation using decorator pattern.
        """
        # Convert to enum for service
        mode_enum = SearchMode(search_mode.upper())

        # Create search request
        request = SearchRequest(
            query=query,
            ids_name=ids_name,
            max_results=max_results,
            mode=mode_enum
        )

        # Execute search through service
        response = await self.search_service.search(request)

        # Add AI prompt for enhancement
        if len(response.hits) > 0:
            response.ai_prompt = self._build_ai_prompt(query, response.hits)

        return response.model_dump()

    def _build_ai_prompt(self, query: str, hits: List[SearchHit]) -> str:
        """Build AI enhancement prompt based on search results."""
        return f"""Search Results Analysis for: {query}
Found {len(hits)} relevant paths in IMAS data dictionary.

Top results:
{chr(10).join([f"- {hit.path}: {hit.documentation[:100]}..." for hit in hits[:3]])}

Provide enhanced analysis including:
1. Physics context and significance of these paths
2. Recommended follow-up searches or related concepts
3. Data usage patterns and common workflows
4. Validation considerations for these measurements"""
```

### 4.3 Update Main Tools Class

```python
# imas_mcp/tools/__init__.py
class Tools(MCPProvider):
    def __init__(self, ids_set: Optional[set[str]] = None):
        super().__init__()

        # Initialize services
        self.search_service = SearchService(...)

        # Initialize tool modules
        self.search_tool = SearchTool(self.search_service)
        self.explain_tool = ExplainTool(...)
        # ... other tools

    # Delegate to tool modules
    async def search_imas(self, *args, **kwargs):
        return await self.search_tool.search_imas(*args, **kwargs)
```

**Deliverables:**

- Clean 30-line search_imas method
- All complexity moved to decorators and services
- Maintainable and testable code structure

---

## Phase 5: Extract Remaining Tools (Days 10-12)

### 5.1 Create Tool Modules

- [ ] Extract `explain_concept` → `ExplainTool`
- [ ] Extract `get_overview` → `OverviewTool`
- [ ] Extract `analyze_ids_structure` → `AnalysisTool`
- [ ] Extract `explore_relationships` → `RelationshipTool`
- [ ] Extract `explore_identifiers` → `IdentifierTool`
- [ ] Extract `export_ids`, `export_physics_domain` → `ExportTool`

### 5.2 Apply Decorators to All Tools

```python
# Example: imas_mcp/tools/explain.py
class ExplainTool:
    @cache_results(ttl=600)  # Longer cache for explanations
    @validate_input(ExplainInputSchema)
    @ai_enhance(temperature=0.2, max_tokens=1000)  # More creative for explanations
    @handle_errors(fallback="concept_suggestions")
    @mcp_tool("Explain IMAS concepts with physics context")
    async def explain_concept(self, concept: str, detail_level: str = "intermediate", ctx: Optional[Context] = None):
        # Clean focused logic
        pass
```

### 5.3 Create Tool-Specific Schemas

```python
# imas_mcp/search/schemas/explain_schemas.py
class ExplainInputSchema(BaseModel):
    concept: str = Field(min_length=1)
    detail_level: str = Field(default="intermediate")

    @field_validator('detail_level')
    def validate_detail_level(cls, v):
        valid_levels = ["basic", "intermediate", "advanced"]
        if v not in valid_levels:
            raise ValueError(f"Invalid detail_level. Must be one of: {valid_levels}")
        return v
```

**Deliverables:**

- All 8 tools extracted into separate modules
- Consistent decorator application across tools
- Tool-specific validation schemas

---

## Phase 6: Testing & Optimization (Days 13-15)

### 6.1 Create Comprehensive Tests

```python
# tests/decorators/test_cache.py
class TestCacheDecorator:
    async def test_cache_hit_returns_cached_result(self):
        @cache_results(ttl=60)
        async def dummy_func():
            return {"result": "cached"}

        result1 = await dummy_func()
        result2 = await dummy_func()

        assert result1 == result2
        # Verify cache was used (mock cache backend)

# tests/tools/test_search_tool.py
class TestSearchTool:
    async def test_search_imas_with_decorators(self):
        tool = SearchTool(mock_search_service)
        result = await tool.search_imas("temperature", search_mode="semantic")

        assert "hits" in result
        assert "ai_insights" in result
        assert "suggestions" in result
```

### 6.2 Performance Testing

- [ ] Benchmark decorator overhead
- [ ] Test caching effectiveness
- [ ] Measure search performance improvements
- [ ] Validate memory usage

### 6.3 Integration Testing

- [ ] Test full decorator chains
- [ ] Verify MCP integration still works
- [ ] Test error handling paths
- [ ] Validate AI enhancement integration

**Deliverables:**

- Comprehensive test suite for all components
- Performance benchmarks and optimizations
- Full integration testing

---

## Migration Strategy

### Backward Compatibility

1. **Phase 1-3**: Keep existing `tools.py` functional alongside new implementation
2. **Phase 4**: Switch `search_imas` to new implementation
3. **Phase 5**: Migrate remaining tools one by one
4. **Phase 6**: Remove old `tools.py` after full migration

### Risk Mitigation

- **Feature flags**: Enable/disable decorator chains
- **A/B testing**: Compare old vs new implementation
- **Gradual rollout**: Migrate tools one at a time
- **Rollback plan**: Keep old implementation available

---

## Expected Benefits

### Code Quality

- **Lines of code**: Reduce from 1800+ to ~200 per tool module
- **Complexity**: Each decorator handles single concern
- **Maintainability**: Easy to modify/extend individual components

### Developer Experience

- **Testing**: Test decorators and tools independently
- **Debugging**: Clear separation of concerns
- **Extension**: Add new decorators without touching existing code

### Performance

- **Caching**: Efficient caching with configurable strategies
- **Modularity**: Load only needed components
- **Optimization**: Easy to optimize individual decorators

### Flexibility

- **Composition**: Mix and match decorators for different tools
- **Configuration**: Easy to adjust decorator parameters
- **Reusability**: Apply same decorators to new tools

---

## Success Criteria

1. **Functional**: All existing functionality preserved
2. **Performance**: No regression in response times
3. **Maintainability**: Each component <100 lines, single responsibility
4. **Testability**: >90% test coverage for all components
5. **Extensibility**: Adding new tool takes <1 day
6. **Developer satisfaction**: Easier to understand and modify

---

## Timeline Summary

- **Phase 1-2**: Foundation (4 days)
- **Phase 3**: Core decorators (3 days)
- **Phase 4**: Search tool refactor (2 days)
- **Phase 5**: Remaining tools (3 days)
- **Phase 6**: Testing & optimization (3 days)

**Total**: 15 days for complete refactoring

---

## Next Steps

1. **Get approval** for this implementation plan
2. **Create Phase 1** directory structure
3. **Start with cache decorator** as proof of concept
4. **Migrate search_imas** as first full implementation
5. **Iterate based on learnings**

This plan transforms the monolithic codebase into a clean, maintainable, and extensible system while preserving all existing functionality and improving developer experience.
