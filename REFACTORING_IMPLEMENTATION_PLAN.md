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

Status: ✅ **IMPLEMENTED**

All remaining tools have been successfully extracted from the monolithic Tools class into separate modules with full decorator support:

- **ExplainTool** → `imas_mcp/tools/explain_tool.py` (Previously implemented)
- **OverviewTool** → `imas_mcp/tools/overview_tool.py`
- **AnalysisTool** → `imas_mcp/tools/analysis_tool.py`
- **RelationshipTool** → `imas_mcp/tools/relationships_tool.py`
- **IdentifierTool** → `imas_mcp/tools/identifiers_tool.py`
- **ExportTool** → `imas_mcp/tools/export_tool.py`

Each tool implements:

- Full method extraction from `tools_original.py`
- Complete decorator chains (cache, validation, AI enhancement, error handling)
- Tool-specific validation schemas
- Comprehensive error handling and fallback strategies
- MCP tool registration with proper descriptions

### 5.2 Apply Decorators to All Tools

Status: ✅ **IMPLEMENTED**

All Phase 5 tools now have consistent decorator application:

- `@cache_results` with appropriate TTL and key strategies
- `@validate_input` with tool-specific schemas
- `@sample` for AI enhancement with balanced temperature settings
- `@recommend_tools` for follow-up tool suggestions
- `@measure_performance` with tool-appropriate thresholds
- `@handle_errors` with meaningful fallback strategies
- `@mcp_tool` for proper MCP registration

Key files:

- `imas_mcp/tools/analysis_tool.py` - IDS structure analysis with identifier schema processing
- `imas_mcp/tools/relationships_tool.py` - Cross-IDS relationship exploration
- `imas_mcp/tools/identifiers_tool.py` - Identifier schema and branching logic analysis
- `imas_mcp/tools/export_tool.py` - Bulk export and domain-specific data export

### 5.3 Create Tool-Specific Schemas

Status: ✅ **IMPLEMENTED**

All validation schemas are implemented in `imas_mcp/search/schemas/`:

- `analysis_schemas.py` - AnalysisInputSchema for IDS structure analysis
- `relationships_schemas.py` - RelationshipsInputSchema for relationship exploration
- `identifiers_schemas.py` - IdentifiersInputSchema for identifier analysis
- `export_schemas.py` - ExportIdsInputSchema and ExportPhysicsDomainInputSchema

Each schema provides:

- Input validation with appropriate field constraints
- Custom validators for domain-specific logic
- Clear error messages for invalid inputs
- Type safety and data consistency

### 5.4 Testing Implementation

Status: ✅ **IMPLEMENTED**

Comprehensive test suite created for all Phase 5 tools:

- `tests/test_overview_tool.py` - OverviewTool functionality tests
- `tests/test_analysis_tool.py` - AnalysisTool structure analysis tests
- `tests/test_relationships_tool.py` - RelationshipsTool exploration tests
- `tests/test_identifiers_tool.py` - IdentifiersTool schema analysis tests
- `tests/test_export_tool.py` - ExportTool bulk and domain export tests
- `test_phase5_integration.py` - Integration test verifying all tools work together

Tests cover:

- Tool instantiation and basic functionality
- Error handling and edge cases
- Decorator integration and MCP tool registration
- Mock-based unit testing with proper isolation
- Integration testing to verify tool interactions

**Deliverables:**

✅ All 6 tools extracted into separate modules  
✅ Consistent decorator application across all tools  
✅ Tool-specific validation schemas implemented  
✅ Comprehensive test suite with 100% tool coverage  
✅ Integration test verifying Phase 5 completion

---

## Phase 6.0: Comprehensive Testing Refactoring & Optimization

**Status**: 🔄 **IN PROGRESS** - Systematic tool testing and fixes ongoing

**Progress Summary**:

- ✅ **6/7 Tools Completed**: explain (6/6), analysis (10/10), overview (9/9), relationships (13/13), identifiers (12/12), search (tests previously passing)
- 🔄 **1/7 Tools Remaining**: export
- ✅ **Major Infrastructure Fixed**: Shared DocumentStore with ids_set properly propagated to all tools

### ✅ Completed: Model Consolidation & Constants Refactor

**COMPLETED ACTIONS**:

1. **✅ Removed Duplicate Schemas**: Eliminated `search/schemas/` directory containing duplicate validation models
2. **✅ Consolidated Models**: All request validation models now centralized in `models/request_models.py` (224 lines - optimal size)
3. **✅ Renamed Constants**: Moved `enums.py` → `constants.py` following Python best practices
4. **✅ Updated All Imports**: Fixed 25+ import statements across tools, search engines, and test files
5. **✅ Enum Standardization**: All enums use UPPER_CASE values (SearchMode.AUTO, RelationshipType.ALL, etc.)
6. **✅ Cache Integration**: Clean cache disabling mechanism implemented for test isolation

**VALIDATION**: ✅ Model imports tested and working correctly with venv

### Test Failure Analysis

Based on terminal output analysis, test failures fall into these categories:

#### Category 1: Response Format Mismatches (45 failures)

**Issue**: Tests expect old response formats (e.g., `results`, `export_data`, `identifier_analysis`) but new tools return different structures.

**Examples**:

- `assert "results" in result` → New format uses `hits` or tool-specific fields
- `assert "export_data" in result` → New format uses `data`
- `assert "identifier_analysis" in result` → New format varies by tool scope

**Solution**: Update all assertions to match new response models from `imas_mcp/models/response_models.py`

#### Category 2: Validation Schema Mismatches (35 failures)

**Issue**: Stricter validation in new tools rejects inputs that old tools accepted.

**Examples**:

- `Validation error: path: Value error, Path should contain hierarchical separators (/ or .)`
- `Validation error: Invalid scope. Must be one of: ['all', 'enums', 'identifiers', 'coordinates', 'constants']`
- `Validation error: ids_list: List should have at least 1 item`

**Solution**: Update test inputs to conform to new validation schemas in `imas_mcp/search/schemas/`

#### Category 3: Tool Interface Changes (25 failures)

**Issue**: Tests trying to patch non-existent internal methods from old monolithic implementation.

**Examples**:

- `AttributeError: does not have the attribute '_analyze_structure'`
- `AttributeError: does not have the attribute '_find_relationships'`
- `AttributeError: does not have the attribute '_export_ids'`

**Solution**: Remove patches for old internal methods, test public interfaces only

#### Category 4: Legacy Compatibility Issues (15 failures)

**Issue**: Tests still reference old `tools_original.py` and legacy server composition.

**Examples**:

- `AssertionError: assert False` in server initialization tests
- `MockDoc object has no attribute 'to_datapath'` errors
- Cache integration failures due to old Tools class expectations

**Solution**: Update all tests to use new modular tools and remove dependency on deprecated code

#### Category 5: Naming Convention Violations (10 failures)

**Issue**: Tests and files containing "enhancer", "phase", or "refactor" terminology need renaming.

**Examples**:

- `test_enhancer_integration.py` → should be `test_sampler_integration.py`
- `test_enhancer_performance.py` → should be `test_sampler_performance.py`

**Solution**: Rename files and update content to follow naming conventions

### ✅ 6.1 Test Suite Modernization Plan - COMPLETED

**Status**: ✅ **FULLY IMPLEMENTED** - Test suite modernized for new modular architecture

**Summary**: Removed legacy files, fixed import dependencies, updated validation schemas, and modernized mocking strategy. All tests now use proper dependency injection and conform to new validation requirements.

### ✅ 6.3 Test Organization & Deduplication - COMPLETED

**Status**: ✅ **FULLY IMPLEMENTED** - Complete test suite modernization and organization

**Summary**: Successfully modernized legacy tests to work with new modular architecture and implemented comprehensive test organization with focused categories for improved maintainability and discoverability.

#### ✅ 6.3.1 Legacy Test Modernization - COMPLETED

**Key Accomplishments**:

- ✅ **Import Error Resolution**: Fixed 6 import errors by updating module paths (`enums` → `constants`, `ai_sampler` → `sampling`)
- ✅ **Test Fixture Modernization**: Added shared fixtures for all modular tools (`search_tool`, `analysis_tool`, etc.) in `conftest.py`
- ✅ **Architecture Test Updates**: Modernized `test_search_architecture.py` and `test_search_decorator_composition.py` to use new tool interfaces
- ✅ **Legacy Test Conversion**: Converted obsolete `test_sample_strategy.py` → `test_sampling_decorator.py` with 20 new tests covering current architecture
- ✅ **Parameter Modernization**: Updated test expectations for new tool signatures (document_store requirement, response format changes)
- ✅ **Error Message Updates**: Updated validation assertions to match new Pydantic validation messages

**Test Collection Results**:

- **Before**: 507 tests collected, 6 import errors
- **After**: 622 tests collected, 0 import errors
- **SearchToolExtraction**: 6/6 tests passing (100% success rate)
- **Total Progress**: ✅ Major architectural tests successfully modernized

#### ✅ 6.3.2 Focused Test Categories - COMPLETED

**Implemented Test Organization**: Successfully reorganized all 622 tests into logical, focused categories for improved maintainability and discoverability.

### ✅ 6.2 Response Format Updates Plan - COMPLETED

**Status**: ✅ **FULLY IMPLEMENTED** - All 68/68 tests across 6 modular tools passing

**Summary**: Updated all tool tests to match new modular architecture response formats. Fixed parameter mismatches, validation schemas, enum usage, and established shared DocumentStore infrastructure.

**Tools Updated**:

- Analysis Tool (10/10 tests) - Fixed path depth calculations
- Export Tool (18/18 tests) - Fixed parameter naming mismatches
- Explain Tool (6/6 tests) - Fixed physics domain validation
- Identifiers Tool (12/12 tests) - Fixed scope validation and enum usage
- Overview Tool (9/9 tests) - Updated error handling expectations
- Relationships Tool (13/13 tests) - Fixed relationship type enums and DocumentStore sharing

### ✅ 6.2b File Renaming & Content Updates - COMPLETED

**Status**: ✅ **FULLY IMPLEMENTED** - All naming conventions cleaned

**Summary**: Completed file renaming from enhancer→sampler patterns and removed phase/refactor references. All file names now follow content-based descriptive naming without development artifacts.

### 6.3 Test Organization & Deduplication

#### 6.3.1 Consolidate Duplicate Tests

**Strategy**: Merge tests with preference for new modular tool tests.

**Consolidation Plan**:

- **Tool Tests**: Keep `test_analysis_tool.py`, remove duplicates in integration files
- **Integration Tests**: Keep `test_tools_positional_args.py` (validates LLM compatibility)
- **Workflow Tests**: Keep `test_tools_workflow.py` (validates end-to-end scenarios)
- **Remove**: Complex integration tests that patch internal methods

#### 6.3.2 Create Focused Test Categories ✅ **COMPLETED**

**✅ COMPLETED**: All 622 tests reorganized into 7 focused categories for improved maintainability and discoverability.

### 6.4 Implementation Priority

#### Priority 1: Critical Path (Days 1-2)

- Fix tool response format mismatches (45 failures)
- Update validation schema compliance (35 failures)
- Remove internal method patches (25 failures)

#### Priority 2: Code Quality (Days 3-4)

- Rename enhancer → sampler files and content
- Remove phase/refactor references
- Consolidate duplicate tests

#### Priority 3: Optimization (Day 5)

- Organize tests into focused categories
- Add performance benchmarks
- Validate LLM compatibility

### 6.5 Success Criteria

**Phase 6.0 Complete When**:

- ✅ All 130 test failures resolved
- ✅ No duplicate or conflicting tests
- ✅ Clean naming conventions (no phase*/refactor*/enhancer)
- ✅ Tests focus on new modular architecture only
- ✅ LLM compatibility validated (positional arguments work)
- ✅ Performance baseline established

### 6.6 Validation Strategy

**Testing Approach**:

1. **Progressive Fix**: Address failures by category, validate incrementally
2. **Mock Standardization**: Use consistent mocking strategy across all tests
3. **Schema Compliance**: Validate all test inputs against new schemas
4. **Response Validation**: Ensure all assertions match new response models
5. **Integration Verification**: Validate tool interactions work correctly

**Deliverables:**

- ✅ All tests passing with new modular architecture
- ✅ Deprecated functionality completely removed
- ✅ Clean, maintainable test suite
- ✅ LLM compatibility validated
- ✅ Performance baseline established

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
