# Coverage Improvement Plan

**Current Coverage:** 65.69% (updated Dec 11, 2025)  
**Previous Coverage:** 49.20%  
**Target Coverage:** >95%  
**Date Created:** December 9, 2025

---

## Progress Summary

| Phase | Status | Est. Lines | Est. Gain |
|-------|--------|------------|-----------|
| Phase 1 | ✅ Complete | - | +16.49% |
| Phase 2 | ✅ Complete | - | - |
| Phase 3 | ✅ Complete | - | - |
| Task A | Not started | ~440 | +5% |
| Task B | Not started | ~280 | +3% |
| Task C | Not started | ~610 | +6% |
| Task D | Not started | ~350 | +4% |
| Task E | Not started | ~400 | +4% |
| Task F | Not started | ~300 | +3% |

**Test Suite:** 777 passed, 4 skipped (69.40s)

---

## Completed Phases

### Phase 1: Quick Wins ✅ Complete

- [x] **core/extractors/** - 417 lines ✅
- [x] **search/document_store.py** - 639 lines at 49.14% ✅
- [x] **search/engines/** ✅
- [x] **dd_accessor.py** - 189 lines at 79.89% ✅

### Phase 2: Medium Effort ✅ Complete

- [x] **clusters/** modules ✅
- [x] **core/xml_utils.py** - 98.55% ✅
- [x] **search/decorators/** ✅

### Phase 3: Higher Effort ✅ Complete

- [x] **services/docs_proxy_service.py** - 46.69% ✅
- [x] **cli.py** - 91.67% ✅
- [x] **graph_analyzer.py** - 96.51% ✅
- [x] **core/physics_categorization.py** - 100% ✅

---

## Remaining Tasks (Background Agent Sessions)

Each task is sized for a single Opus 4.5 background session (~300-500 lines, focused scope).

### Task A: XML Parser & Clusters Core (~440 lines, +5%)

**Scope:** Low-coverage core parsing modules requiring XML fixtures.

| Module | Lines | Current | Target |
|--------|-------|---------|--------|
| `core/xml_parser.py` | 291 | 0% | 80% |
| `core/clusters.py` | 195 | 24.62% | 80% |

**Setup required:**
- Create sample XML fixtures in `tests/fixtures/`
- Mock ElementTree parsing where needed
- Handle session-scoped mock conflicts in conftest.py

**Test file:** `tests/core/test_xml_parser.py`

---

### Task B: List Tool & Resource Accessors (~280 lines, +3%)

**Scope:** Tool and accessor modules with straightforward testing.

| Module | Lines | Current | Target |
|--------|-------|---------|--------|
| `tools/list_tool.py` | 146 | 16.44% | 85% |
| `resource_path_accessor.py` | 68 | 60.29% | 90% |
| `resource_provider.py` | 71 | 61.97% | 90% |

**Test approach:**
- Mock DD accessor for list tool tests
- Test format options (yaml, list, json, dict)
- Test path resolution and resource loading

**Test files:** `tests/tools/test_list_tool.py`, `tests/test_resource_accessors.py`

---

### Task C: Embeddings Suite (~610 lines, +6%)

**Scope:** Embeddings and encoder modules with API mocking.

| Module | Lines | Current | Target |
|--------|-------|---------|--------|
| `embeddings/encoder.py` | 225 | 75.11% | 90% |
| `embeddings/openrouter_client.py` | 133 | 57.14% | 85% |
| `embeddings/cache.py` | 106 | 77.36% | 92% |
| `embeddings/embeddings.py` | 85 | 70.59% | 90% |
| `embeddings/config.py` | 63 | 76.19% | 90% |

**Test approach:**
- Mock HTTP requests for OpenRouter client
- Test cache hit/miss scenarios
- Test encoder fallback behavior
- Test configuration validation

**Test files:** `tests/embeddings/` (expand existing tests)

---

### Task D: Search Infrastructure (~350 lines, +4%)

**Scope:** Search modules needing semantic/hybrid coverage.

| Module | Lines | Current | Target |
|--------|-------|---------|--------|
| `search/semantic_search.py` | 157 | 34.39% | 85% |
| `search/search_strategy.py` | 189 | 62.96% | 85% |

**Test approach:**
- Mock embedding encoder
- Test query routing and strategy selection
- Test result ranking and filtering

**Test files:** `tests/search/test_semantic_search.py`, `tests/search/test_search_strategy.py`

---

### Task E: Server & Services (~400 lines, +4%)

**Scope:** Server lifecycle and service manager modules.

| Module | Lines | Current | Target |
|--------|-------|---------|--------|
| `services/docs_server_manager.py` | 441 | 55.10% | 80% |
| `server.py` | 139 | 72.66% | 88% |

**Test approach:**
- Mock subprocess/process management
- Test server initialization and shutdown
- Test health monitoring
- Expand existing `test_docs_server_lifecycle.py`

**Test files:** `tests/test_docs_server_lifecycle.py`, `tests/test_server_extended.py`

---

### Task F: Clusters Extractor & Remaining Gaps (~300 lines, +3%)

**Scope:** Low-coverage cluster extraction and miscellaneous gaps.

| Module | Lines | Current | Target |
|--------|-------|---------|--------|
| `clusters/extractor.py` | 279 | 11.83% | 75% |
| `mappings/__init__.py` | 110 | 61.82% | 85% |
| `settings.py` | 82 | 74.39% | 90% |

**Test approach:**
- Test feature extraction methods
- Test mapping operations
- Test environment variable handling

**Test files:** `tests/clusters/test_extractor.py`, `tests/mappings/test_mappings.py`

---

## Modules with Good Coverage (Maintain)

These modules have good coverage. Add tests when fixing bugs.

| Module | Coverage |
|--------|----------|
| models/constants.py | 100% |
| models/context_models.py | 100% |
| models/error_models.py | 100% |
| clusters/labeling.py | 100% |
| clusters/preprocessing.py | 100% |
| core/physics_categorization.py | 100% |
| core/data_model.py | 99.32% |
| core/xml_utils.py | 98.55% |
| clusters/label_cache.py | 97.92% |
| models/result_models.py | 96.95% |
| graph_analyzer.py | 96.51% |
| tools/path_tool.py | 92.99% |
| search/tool_suggestions.py | 92.31% |
| cli.py | 91.67% |
| tools/identifiers_tool.py | 91.60% |
| tools/base.py | 90.00% |
| search/cache.py | 88.89% |
| core/exclusions.py | 88.24% |
| tools/overview_tool.py | 86.79% |
| health.py | 85.48% |

---

## Background Task Instructions

Each task above can be run as a single background agent session with this prompt pattern:

```
Improve test coverage for [Task X] modules in the imas-mcp project.

Target modules:
- [list modules with current coverage %]

Requirements:
1. Run `uv sync --extra test` before testing
2. Create/update test files in the appropriate `tests/` subdirectory
3. Achieve target coverage (80-90%) for each module
4. Run `uv run pytest --cov=imas_mcp -q` to verify coverage
5. Follow project conventions in AGENTS.md
6. Commit with conventional commit format
```

---

## Testing Strategy Notes

### Fixtures Available
- Mock DD accessor with sample data
- Mock embedding encoder
- Sample XML fragments for parser tests
- Mock HTTP server for docs proxy tests

### Test Organization
- Mirror source structure in `tests/`
- Use `conftest.py` for shared fixtures
- Mark slow tests with `@pytest.mark.slow`
- Use `pytest-asyncio` for async tests

### Running Coverage
```bash
uv run pytest --cov=imas_mcp --cov-report=term-missing --cov-report=html -q
```

---

## Progress Log

### 2025-12-11: Restructured for Background Tasks
- Reorganized remaining work into 6 focused tasks (A-F)
- Each task targets ~300-500 lines, suitable for single Opus 4.5 session
- Total estimated gain: +25% coverage (reaching ~90%)

### 2025-12-11: Phase 1 Complete, Phase 2-4 Progress
- **Coverage increased from 49.20% → 65.69% (+16.49%)**
- **Test count increased from 447 → 777 tests (+330 tests)**
- Phase 1 (Quick Wins) completed:
  - `core/extractors/` - All modules now have 84-100% coverage
  - `search/engines/` - All engines now have 58-72% coverage
  - `dd_accessor.py` - Now at 79.89%
  - `search/document_store.py` - Now at 49.14%
- Phase 4 progress:
  - `search/cache.py` - Now at 88.89%
  - `search/tool_suggestions.py` - Now at 92.31%

### 2025-12-10: Phase 2 - Partial Completion
- Created comprehensive tests for `clusters/` modules
- Created comprehensive tests for `search/decorators/`
- Created tests for `core/xml_utils.py`
- **Total new tests added:** ~170 tests
- **All tests passing:** 447 passed, 4 skipped
