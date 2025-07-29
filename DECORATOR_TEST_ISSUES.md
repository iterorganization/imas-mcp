# Decorator Test Issues - Analysis and Action Plan

*Generated on July 28, 2025 | Updated July 29, 2025*

## Overview

The decorator tests in `tests/decorators/` are failing due to multiple API mismatches between test expectations and the current implementation. These issues indicate the tests were written for an earlier version of the API and need systematic updates.

**Recent Progress**: Completed comprehensive search architecture refactoring, eliminating SearchComposer and search_with_params in favor of direct SearchService usage with proper Pydantic models.

## Test Status Summary

- **Core Decorators (`test_core_decorators.py`)**: ✅ **28/28 PASSING** (Fixed)
- **Sampler Integration (`test_sampler_integration.py`)**: ❌ **2/20 PASSING** (18 failures)
- **Tool Suggestions (`test_tool_suggestions.py`)**: ❓ Not yet analyzed
- **Sampling Decorator (`test_sampling_decorator.py`)**: ❓ Not yet analyzed

## Critical Issues Requiring Immediate Action

### 1. **✅ RESOLVED: Return Format Mismatch (dict vs SearchResponse)**

**Issue**: Tests expected dict responses but tools now return SearchResponse objects
- **Resolution**: Updated tests to use SearchResponse attributes (`.hits`, `.count`, `.search_mode`, `.query`)
- **Changes**: Fixed 28 tests to use `hasattr(result, 'field')` instead of `"field" in result`
- **Status**: ✅ **FIXED** - All search service integration tests (14/14) now passing

### 2. **✅ RESOLVED: Search Engine Async/Await Issues**

**Issue**: Search engines returned lists but service tried to await them
- **Error**: `object list can't be used in 'await' expression`
- **Resolution**: Updated test mocks to use `AsyncMock` instead of `Mock` for engine.search methods
- **Status**: ✅ **FIXED** - All async/await errors resolved

### 3. **✅ RESOLVED: SearchResult Document Validation**

**Issue**: Tests used Mock() objects but SearchResult requires proper Document instances
- **Resolution**: Created proper Document instances with DocumentMetadata in tests
- **Status**: ✅ **FIXED** - Pydantic validation errors resolved

### 4. **� PARTIAL: SearchConfig Attribute Mismatches**

**Issue**: Tests expect different attribute names than implementation
- **Fixed**: `config.search_mode` (was expecting `config.mode`) ✅
- **Fixed**: Tool name `"search_imas"` (was expecting `"search_tool"`) ✅ 
- **Remaining**: 3 tests with parameter/mock issues still failing

**Status**: � **MOSTLY RESOLVED** - Major attribute mismatches fixed, minor issues remain

### 3. **� MEDIUM: Export Format Validation Mismatch**

**Issue**: Test expectations don't match actual export format options
- **Tests Expect**: `"raw"`, `"enhanced"`, `"structured"`
- **API Accepts**: `'structured', 'json', 'yaml', 'markdown'`
- **Impact**: All export format tests fail

**Error Messages**:
```
KeyError: 'output_format'
Validation error: output_format: Input should be 'structured', 'json', 'yaml' or 'markdown'
```

**Tests Failing**:
- `test_raw_export_format`
- `test_structured_export_format`
- `test_enhanced_export_format`
- `test_invalid_export_format`
- `test_export_format_performance_optimization`
- `test_format_based_processing`

**Action Required**: Either update export tool to support expected formats or update tests to use correct formats.

**Issue**: Test expectations don't match actual export format options
- **Tests Expect**: `"raw"`, `"enhanced"`, `"structured"`
- **API Accepts**: `'structured', 'json', 'yaml', 'markdown'`
- **Impact**: All export format tests fail

**Error Messages**:
```
KeyError: 'output_format'
Validation error: output_format: Input should be 'structured', 'json', 'yaml' or 'markdown'
```

**Tests Failing**:
- `test_raw_export_format`
- `test_structured_export_format`
- `test_enhanced_export_format`
- `test_invalid_export_format`
- `test_export_format_performance_optimization`
- `test_format_based_processing`

**Action Required**: Either update export tool to support expected formats or update tests to use correct formats.

### 4. **🟡 MEDIUM: AI Insights Structure Differences**

**Issue**: Tests expect specific AI insights status messages that don't match implementation
- **Expected**: Specific status strings like `"AI enhancement applied"`
- **Actual**: Empty `{}` or different status structure
- **Impact**: AI enhancement assertions fail

**Tests Failing**:
- `test_structure_analysis_conditional_ai`
- `test_physics_domain_conditional_ai`

**Action Required**: Align AI insights structure between decorators and test expectations.

## Fixed Issues ✅

### Search Architecture Refactoring (July 29, 2025)
- **Issue**: SearchComposer and search_with_params() created API complexity with manual dict conversions
- **Solution**: Replaced with direct SearchService usage and proper Pydantic SearchResponse models, eliminated to_dict() methods throughout search pipeline
- **Status**: ✅ **36 tests passing** - Phase 1 enhanced search_imas tool with multi-mode search, selective AI enhancement, and multi-format export tools fully implemented with comprehensive decorator composition.

### Method Access Pattern Changes
- **Issue**: Tests used `server.tools.method_name()` but methods are now on specific tool objects
- **Solution**: Updated to use `server.tools.tool_name.method_name()`
- **Status**: ✅ Fixed in previous session

**Examples of fixes applied**:
```python
# Before (failing)
await server.tools.export_ids(...)
await server.tools.analyze_ids_structure(...)

# After (working)  
await server.tools.export_tool.export_ids(...)
await server.tools.analysis_tool.analyze_ids_structure(...)
```

### Parameter Value Corrections
- **Issue**: Invalid parameter values like `scope="summary"`
- **Solution**: Updated to valid values like `scope="all"`
- **Status**: ✅ Fixed in current session

## Action Plan

### Phase 1: Critical Fixes (Immediate)
1. **Fix `ids_filter`/`ids_name` mismatch**
   - [ ] Decide on unified parameter name
   - [ ] Update either schema or method signature
   - [ ] Test search functionality

2. **Fix SearchConfig `ids_set` issue**
   - [ ] Add missing attribute to SearchConfig
   - [ ] Update related search engine code
   - [ ] Test semantic search functionality

### Phase 2: Medium Priority Fixes
3. **Standardize Export Formats**
   - [ ] Review export tool format requirements
   - [ ] Either add missing formats or update tests
   - [ ] Ensure consistent format handling

4. **Align AI Insights Structure**
   - [ ] Document expected AI insights format
   - [ ] Update decorators or tests for consistency
   - [ ] Test AI enhancement features

### Phase 3: Complete Testing
5. **Run Full Test Suite**
   - [ ] Execute all decorator tests
   - [ ] Verify no regressions in other test suites
   - [ ] Document any remaining issues

## Test Files Analysis Status

| File | Status | Issues Found | Priority |
|------|--------|--------------|----------|
| `test_core_decorators.py` | ✅ Fixed | Cache isolation | Complete |
| `test_sampler_integration.py` | ❌ 18/20 failing | API mismatches | High |
| `test_tool_suggestions.py` | ❓ Pending | Unknown | Medium |
| `test_sampling_decorator.py` | ❓ Pending | Unknown | Medium |

## Dependencies and Related Components

### Components Requiring Updates
- `imas_mcp/models/request_models.py` - Schema definitions
- `imas_mcp/tools/search_tool.py` - Method signatures  
- `imas_mcp/search/search_strategy.py` - SearchConfig class
- `imas_mcp/tools/export_tool.py` - Format validation
- `imas_mcp/search/decorators/` - AI insights structure

### Test Files Requiring Updates
- `tests/decorators/test_sampler_integration.py` - Multiple API mismatches
- Potentially `test_tool_suggestions.py` and `test_sampling_decorator.py`

## Notes

- The core decorator functionality is working correctly (28/28 tests passing)
- Recent search architecture refactoring may have resolved some parameter mismatch issues
- Issues are primarily API contract mismatches, not decorator logic problems
- Some tests may need to be rewritten if the underlying API has fundamentally changed
- Consider adding integration tests that verify the decorator behavior with the actual current API
- The search_with_params and SearchComposer removal may require updating decorator tests that relied on those components

## Success Criteria

- [ ] All decorator tests passing
- [ ] No regressions in existing functionality
- [ ] Consistent API contracts between schemas, methods, and tests
- [ ] Proper AI enhancement behavior verification
- [ ] Export functionality working with expected formats
