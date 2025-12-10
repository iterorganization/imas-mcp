# Coverage Improvement Plan

**Current Coverage:** 49.20% (updated Dec 9, 2025)  
**Previous Coverage:** 45.48%  
**Target Coverage:** >95%  
**Date Created:** December 9, 2025

---

## Progress Summary

| Phase | Status | Coverage Gain |
|-------|--------|---------------|
| Phase 1 | Not started | - |
| Phase 2 | Not started | - |
| Phase 3 | **In progress** | +3.72% |
| Phase 4 | Not started | - |

---

## Phase 1: Quick Wins (Est. +15-20% coverage)

High-impact, relatively straightforward unit tests.

- [ ] **core/extractors/** - 417 lines at 0%
  - `base.py` (80 lines)
  - `coordinate_extractor.py` (21 lines)
  - `identifier_extractor.py` (137 lines)
  - `metadata_extractor.py` (98 lines)
  - `physics_extractor.py` (13 lines)
  - `semantic_extractor.py` (45 lines)
  - `validation_extractor.py` (23 lines)

- [ ] **search/document_store.py** - 443 lines at 31%
  - Focus on document indexing and retrieval
  - Test search query building
  - Test result filtering

- [ ] **search/engines/** - ~200 lines combined
  - `lexical_engine.py` (109 lines at 16%)
  - `hybrid_engine.py` (55 lines at 26%)
  - `semantic_engine.py` (34 lines at 35%)

- [ ] **dd_accessor.py** - 134 lines at 29%
  - Test IDS accessors
  - Test path resolution
  - Test caching behavior

---

## Phase 2: Medium Effort (Est. +15-20% coverage)

Requires more setup but still well-defined test boundaries.

- [ ] **clusters/** modules - ~500 lines combined
  - `clustering.py` (155 lines at 15%)
  - `extractor.py` (246 lines at 12%)
  - `labeler.py` (72 lines at 22%)
  - `preprocessing.py` (53 lines at 17%)
  - `label_cache.py` (68 lines at 29%)

- [ ] **core/xml_parser.py** - 310 lines at 0%
  - Test XML element parsing
  - Test path extraction
  - Test attribute handling

- [ ] **core/xml_utils.py** - 69 lines at 0%
  - Test utility functions

- [ ] **tools/list_tool.py** - 122 lines at 16%
  - Test path listing functionality
  - Test format options (yaml, list, json, dict)

- [ ] **search/decorators/** - ~130 lines combined
  - `cache.py` (32 lines at 59%)
  - `error_handling.py` (46 lines at 59%)
  - `performance.py` (50 lines at 51%)
  - `tool_recommendations.py` (62 lines at 13%)

---

## Phase 3: Higher Effort (Est. +10-15% coverage)

Requires mocking external services or complex setup.

- [x] **services/docs_proxy_service.py** - 272 lines at 18% → **46.69%** ✅
  - Mock HTTP requests
  - Test document fetching
  - Test error handling
  - Tests added: `tests/services/test_docs_proxy_service.py`

- [ ] **services/docs_server_manager.py** - 198 lines at 55%
  - Test server lifecycle
  - Test health monitoring
  - Mock process management
  - Note: Existing tests in `tests/test_docs_server_lifecycle.py`

- [x] **cli.py** - 60 lines at 0% → **91.67%** ✅
  - Test CLI argument parsing
  - Test command execution
  - Use click testing utilities
  - Tests added: `tests/test_cli.py`

- [x] **graph_analyzer.py** - 86 lines at 0% → **96.51%** ✅
  - Test graph construction
  - Test relationship detection
  - Tests added: `tests/test_graph_analyzer.py`

- [x] **core/physics_categorization.py** - 57 lines at 0% → **100%** ✅
  - Test physics domain classification
  - Tests added: `tests/core/test_physics_categorization.py`

- [ ] **core/clusters.py** - 147 lines at 25%
  - Test cluster operations
  - Note: Session-scoped mock in conftest.py prevents direct testing

---

## Phase 4: Final Polish (Est. +5% coverage)

Edge cases, error paths, and remaining gaps.

- [ ] **tools/** remaining gaps
  - `overview_tool.py` (60 lines at 71%)
  - `identifiers_tool.py` (31 lines at 74%)
  - `clusters_tool.py` (13 lines at 81%)
  - `docs_tool.py` (10 lines at 82%)
  - `search_tool.py` (5 lines at 84%)

- [ ] **embeddings/** remaining gaps
  - `openrouter_client.py` (62 lines at 53%)
  - `cache.py` (33 lines at 68%)
  - `encoder.py` (54 lines at 76%)

- [ ] **settings.py** - 30 lines at 63%
  - Test environment variable handling
  - Test configuration loading

- [ ] **server.py** - 45 lines at 68%
  - Test server initialization
  - Test error handling

- [ ] **mappings/__init__.py** - 42 lines at 62%
  - Test mapping operations

- [ ] **search/semantic_search.py** - 111 lines at 29%
  - Test semantic search functionality

- [ ] **search/cache.py** - 41 lines at 24%
  - Test cache hit/miss scenarios

- [ ] **search/tool_suggestions.py** - 45 lines at 13%
  - Test suggestion logic

---

## Modules with Good Coverage (Maintain)

These modules already have good coverage. Only add tests if bugs are found.

| Module | Coverage |
|--------|----------|
| models/constants.py | 100% |
| models/context_models.py | 100% |
| models/error_models.py | 100% |
| models/result_models.py | 97% |
| core/data_model.py | 99% |
| tools/path_tool.py | 93% |
| tools/base.py | 90% |
| core/exclusions.py | 88% |
| health.py | 85% |
| services/response.py | 84% |

---

## Testing Strategy Notes

### Fixtures to Create
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

### Viewing HTML Report
Open `htmlcov/index.html` in a browser for detailed line-by-line coverage.
