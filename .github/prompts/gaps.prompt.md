# IMAS MCP Tools Priority Gap Implementation Prompt

You are an expert Python developer working on the IMAS MCP (Model Context Protocol) tools codebase. Your task is to implement missing functionality based on a priority number from the implementation plan.

## Input Parameters

- **Priority Number**: {priority_number} (1-5, corresponding to the priorities in IMPLEMENTATION_PLAN.md)

## Context Documents

- Review IMPLEMENTATION_PLAN.md for the specific priority details
- Review FEATURE_COMPARISON_REPORT.md for background context
- Examine existing code in imas_mcp/tools/ directory for current implementation patterns

## Development Requirements

### Code Quality Standards

- **Testing Approach**: Favor testing features and functionality over implementation specifics
- **Test Coverage**: All new code must be covered by new tests that are validated as working
- **No Backwards Compatibility**: Do not maintain backwards compatibility when refactoring
- **Import Standards**: All imports at top of file, no relative imports
- **Data Exchange**: Use Pydantic models for all data exchange

### Python Coding Standards

Follow the refactoring instructions from `.github/instructions/refactoring.instructions.md`:

- Do not use specifiers like enhanced, simple, optimized, advanced, intelligent, smart, improved, refactor, phase\* in file names or doc strings
- Add optional parameters to existing functions rather than creating new variants
- Do not create advanced, enhanced, v2, intelligent, smart, or similar suffixed/prefixed versions
- Update existing functionality in place by adding new capabilities as optional parameters
- Prefer parameter-driven feature expansion over tool proliferation

### Implementation Tasks by Priority

#### Priority 1: Physics Integration Restoration

**Scope**: Restore `physics_search()` integration in explain_tool.py
**Files to Modify**: `imas_mcp/tools/explain_tool.py`
**Test Requirements**:

- Test physics integration with valid concepts
- Test graceful fallback when physics_search fails
- Verify physics_context is properly included in response model

#### Priority 2: Overview Tool Question Analysis

**Scope**: Add question-specific analysis functionality to overview_tool.py
**Files to Modify**: `imas_mcp/tools/overview_tool.py`
**Test Requirements**:

- Test overview with and without query parameter
- Test question analysis response structure
- Test search integration for question-specific results
- Verify OverviewResult model includes new fields

#### Priority 3: Cross-IDS Relationship Analysis

**Scope**: Restore cross-IDS relationship analysis in export_tool.py
**Files to Modify**: `imas_mcp/tools/export_tool.py`
**Test Requirements**:

- Test relationship analysis with multiple IDS
- Test relationship analysis with single IDS (should skip)
- Test error handling in relationship discovery
- Verify IDSExport model includes relationship data

#### Priority 4: Conditional AI Enhancement

**Scope**: Create conditional sampling decorator system
**Files to Create**: `imas_mcp/search/decorators/conditional_sampling.py`
**Files to Modify**: All tool files to use conditional_sample decorator
**Test Requirements**:

- Test enhancement decision logic for each tool category
- Test conditional evaluation based on parameters
- Test strategy patterns (ALWAYS, NEVER, CONDITIONAL)
- Test decorator integration with existing tools

#### Priority 5: Document Store Integration Fixes

**Scope**: Replace mock data with real document store integration in overview_tool.py
**Files to Modify**: `imas_mcp/tools/overview_tool.py`
**Test Requirements**:

- Test real data retrieval from document store
- Test error handling when document store fails
- Test statistics generation with real data
- Test identifier summary integration

## Implementation Instructions

1. **Analyze Current State**:

   - Examine the target files for Priority {priority_number}
   - Identify existing patterns and architecture
   - Review current test coverage

2. **Design Implementation**:

   - Follow the specific solution outlined in IMPLEMENTATION_PLAN.md
   - Ensure new code follows existing patterns in the codebase
   - Design Pydantic models for any new data structures

3. **Implement Changes**:

   - Make minimal, focused changes to achieve the priority goal
   - Add new functionality as optional parameters where possible
   - Update existing functions rather than creating new variants
   - Place all imports at the top of files (no relative imports)

4. **Create Tests**:

   - Write comprehensive tests for new functionality
   - Test both success and error cases
   - Validate test coverage for all new code paths
   - Ensure tests focus on functionality over implementation details

5. **Validate Integration**:
   - Run existing test suite to ensure no regressions
   - Verify new tests pass consistently
   - Check that Pydantic models serialize/deserialize correctly
   - Validate that the implementation matches the priority specification

## Success Criteria

- [ ] Priority {priority_number} functionality fully implemented per IMPLEMENTATION_PLAN.md
- [ ] All new code covered by working tests
- [ ] No regression in existing test suite
- [ ] Code follows refactoring instructions (no backwards compatibility, parameter-driven expansion)
- [ ] All imports at top of file, no relative imports
- [ ] Pydantic models used for all data exchange
- [ ] Implementation focuses on functionality testing over implementation specifics

## Deliverables

1. **Modified/Created Files**: Implementation of Priority {priority_number} changes
2. **Test Files**: Comprehensive test coverage for new functionality
3. **Test Validation**: Proof that all new tests pass
4. **Integration Verification**: Confirmation that existing tests still pass

Focus on clean, testable, maintainable code that enhances the existing architecture rather than replacing it. The goal is to restore missing functionality while preserving the improved modular design of the new tools system.
