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
4. **Mixed concerns**: search, caching, validation, sampling all coupled
5. **Poor testability**: Hard to test individual components in isolation

## Proposed Architecture

### Enhanced Decorator Pattern

```python
@cache_results(ttl=300, key_strategy="semantic")
@validate_input(schema=SearchInputSchema)
@sample(temperature=0.3, max_tokens=800)
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
│   │   ├── sampling.py                 # @sample decorator
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
│   │   └── sampling_service.py        # Sampling logic
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

## Phase 2: Extract Search Engine (Days 3-4) ✅ IMPLEMENTED

**Phase Status: ✅ COMPLETE** - All search engine extraction and modular architecture goals achieved

**Phase Summary:**

- Pure search engines created with clean separation of concerns
- Search service orchestration layer implemented
- Search logic extracted from monolithic tools
- Modular Tools class with proper delegation
- Comprehensive test coverage (20 tests) with all tests passing

### 2.1 Create Pure Search Engine ✅ IMPLEMENTED

**Status: ✅ COMPLETE** - Pure search engines have been fully implemented with proper separation of concerns.

**Implementation Summary:**

- **Base Engine**: `imas_mcp/search/engines/base_engine.py` (130 lines) - Abstract SearchEngine class
- **Semantic Engine**: `imas_mcp/search/engines/semantic_engine.py` (160 lines) - Embedding-based similarity search
- **Lexical Engine**: `imas_mcp/search/engines/lexical_engine.py` (186 lines) - Whoosh full-text search
- **Hybrid Engine**: `imas_mcp/search/engines/hybrid_engine.py` (191 lines) - Combined weighted scoring
- **Engine Factory**: `imas_mcp/search/engines/__init__.py` (17 lines) - Type-based engine creation

**Key Features Implemented:**

- Configuration-driven initialization via SearchConfig
- Abstract search interface with consistent SearchResult output
- Engine type identification and validation
- Advanced query parsing (boolean operators, phrases)
- Vector similarity calculations with configurable thresholds
- Weighted result combination in hybrid mode
- Factory pattern for engine instantiation

**Testing**: 22 comprehensive tests covering all engines, initialization, and integration scenarios.

**Models**: SearchConfig and SearchResult implemented in `imas_mcp/search/search_strategy.py` (467 lines).

### 2.2 Extract Search Logic from Tools ✅ IMPLEMENTED

**Status: ✅ COMPLETE** - Search logic successfully extracted and modular Tools class implemented

**Completed Items:**

- ✅ **SearchConfig created**: Comprehensive configuration model in `search_strategy.py`
- ✅ **SearchResult models created**: Full result model with metadata in `search_strategy.py`
- ✅ **Tools structure split**: Individual tool classes in `imas_mcp/tools/` directory
- ✅ **Search engines created**: All three engines (Semantic, Lexical, Hybrid) fully implemented
- ✅ **Search logic extracted**: `search_imas` method implemented in `imas_mcp/tools/search.py`
- ✅ **Search service integration**: SearchService properly orchestrates engines
- ✅ **Tools class delegation**: Main Tools class delegates to Search tool with backward compatibility

**Implementation Details:**

- **File**: `imas_mcp/tools/search.py` (171 lines) - Complete search_imas implementation
- **File**: `imas_mcp/tools/__init__.py` (73 lines) - Main Tools class with delegation
- **Validation**: Comprehensive input validation for search mode and parameters
- **Error Handling**: Standardized error responses with proper logging
- **Testing**: 20 comprehensive tests covering all functionality (all passing)

### 2.3 Create Search Service ✅ IMPLEMENTED

**Status: ✅ COMPLETE** - Full SearchService implementation with advanced orchestration features

**Implementation Details:**

- **File**: `imas_mcp/search/services/search_service.py` (274 lines)
- **Core Features**: Engine orchestration, mode resolution, result post-processing
- **Advanced Features**: Health checking, engine registration, structured request/response models
- **Error Handling**: Custom SearchServiceError with query context
- **Auto Mode**: Intelligent mode selection based on query analysis
- **Testing Support**: Mock engines for development and testing

**Key Classes Implemented:**

- `SearchService`: Main orchestration service with async search execution
- `SearchRequest`: Structured request model for clean interfaces
- `SearchResponse`: Structured response with metadata and conversion utilities
- `SearchServiceError`: Custom exception with query context

**Deliverables: ✅ COMPLETE**

- ✅ Separate search engines for each mode (SemanticSearchEngine, LexicalSearchEngine, HybridSearchEngine)
- ✅ Search service orchestrating engine selection with intelligent mode resolution
- ✅ Clean separation of search logic from tools with service pattern

---

## Phase 3: Implement Core Decorators (Days 5-7)

### ✅ **IMPLEMENTED** - Core Decorators System

**Status**: All six core decorators fully implemented and tested with comprehensive functionality.

**Key Files**:

- `imas_mcp/search/decorators/cache.py` - Result caching with TTL and key strategies
- `imas_mcp/search/decorators/validation.py` - Pydantic input validation with error handling
- `imas_mcp/search/decorators/sampling.py` - AI-powered result enrichment (renamed from ai_enhancement)
- `imas_mcp/search/decorators/suggestions.py` - Tool recommendations with configurable max_suggestions
- `imas_mcp/search/decorators/performance.py` - Execution metrics and performance scoring
- `imas_mcp/search/decorators/error_handling.py` - Robust error handling with fallback support
- `imas_mcp/search/schemas/search_schemas.py` - Input validation schemas for all tools
- `tests/test_core_decorators.py` - 28 comprehensive tests (all passing)

**Main Functionality Implemented**:

- **Cache System**: LRU cache with TTL, key strategies (args_only, semantic), cache statistics
- **Input Validation**: Pydantic schema validation, standardized error responses, context preservation
- **Sampling Enhancement**: AI-powered result enrichment with configurable temperature and token limits
- **Tool Suggestions**: Intelligent follow-up tool recommendations with max_suggestions parameter (default: 4)
- **Performance Monitoring**: Execution timing, memory tracking, performance scoring, slow operation logging
- **Error Handling**: Standardized error responses, fallback suggestions, timeout handling, recovery actions

**Deliverables Completed**:

- ✅ Six core decorators implemented and tested
- ✅ Clean separation of cross-cutting concerns
- ✅ Reusable decorators for all tools
- ✅ Comprehensive input validation schemas
- ✅ Full test coverage with 28 passing tests

---

## Phase 4: Search Tool with Decorator Composition ✅ **COMPLETE**

### ✅ **IMPLEMENTED** - Search Tool Decorator Composition

**Status**: Comprehensive search tool implementation with full decorator composition completed and tested.

**Key Accomplishments**:

- **Search Tool Implementation**: `imas_mcp/tools/search.py` - Complete implementation with decorator composition
- **Input Validation**: Integrated SearchInputSchema for comprehensive parameter validation
- **Decorator Stack**: Applied all six core decorators (cache, validation, sampling, tool recommendations, performance, error handling)
- **Parameter Updates**: Updated max_recommendations → max_tools for consistency
- **Documentation Cleanup**: Removed phase references and "enhanced" terminology for descriptive content-based naming
- **Comprehensive Testing**: `tests/test_search_decorator_composition.py` with 18 passing tests

**Implementation Details**:

- **File**: `imas_mcp/tools/search.py` (206 lines) - Search tool with comprehensive decorator composition
- **Decorator Stack Applied**:
  - `@cache_results(ttl=300, key_strategy="semantic")` - Performance optimization
  - `@validate_input(schema=SearchInputSchema)` - Input validation with Pydantic
  - `@sample(temperature=0.3, max_tokens=800)` - AI insights and analysis
  - `@recommend_tools(strategy="search_based", max_tools=4)` - Follow-up tool recommendations
  - `@measure_performance(include_metrics=True, slow_threshold=1.0)` - Performance monitoring
  - `@handle_errors(fallback="search_suggestions")` - Robust error handling
  - `@mcp_tool("Search for IMAS data paths...")` - MCP integration

**Core Features**:

- **Multi-mode Search**: Auto, semantic, lexical, and hybrid search modes
- **Configuration-driven**: Uses SearchConfig for engine orchestration
- **AI Sampling Integration**: Builds context-aware prompts for enhanced insights
- **Performance Optimized**: Caching and performance monitoring built-in
- **Error Resilient**: Comprehensive error handling with fallback suggestions
- **Input Validation**: Pydantic schema validation for all parameters

**Testing Coverage**:

- **Core Decorator Tests**: 28 tests covering all decorator functionality
- **Decorator Composition Tests**: 18 tests specifically for search tool integration
- **Integration Testing**: Mock-based testing with comprehensive scenarios
- **Configuration Testing**: Search modes, IDS filtering, parameter validation
- **Physics Context**: Proper units (eV for plasma temperature, Tesla for magnetic field)

**Naming Consistency Updates**:

- Removed "enhanced", "phase", and "refactor" references
- Updated parameter naming: max_recommendations → max_tools
- Content-based descriptive naming throughout
- Test file renamed: test_search_decorator_composition.py

**Deliverables Completed**:

- ✅ Search tool with comprehensive decorator composition
- ✅ All decorators properly integrated and functional
- ✅ Comprehensive test suite with 46 total passing tests (28 + 18)
- ✅ Clean, maintainable code structure with proper separation of concerns
- ✅ Physics-appropriate test data and validation

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
- [ ] Validate sampling integration

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
- **✅ Phase 3**: Core decorators (3 days) - **COMPLETED**
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
