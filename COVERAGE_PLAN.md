# Coverage Improvement Plan

**Current Coverage:** 45.48%  
**Target Coverage:** >95%  
**Date Created:** December 9, 2025

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

- [x] **clusters/** modules - ~500 lines combined ✅ (completed 2025-12-10)
  - [x] `clustering.py` - Tests in `tests/clusters/test_clustering.py`
  - [x] `labeler.py` - Tests in `tests/clusters/test_labeler.py`
  - [x] `preprocessing.py` - Tests in `tests/clusters/test_preprocessing.py`
  - [x] `label_cache.py` - Tests in `tests/clusters/test_label_cache.py`

- [ ] **core/xml_parser.py** - 310 lines at 0%
  - Test XML element parsing
  - Test path extraction
  - Test attribute handling

- [x] **core/xml_utils.py** - 69 lines at 0% ✅ (completed 2025-12-10)
  - [x] Tests in `tests/core/test_xml_utils.py`

- [ ] **tools/list_tool.py** - 122 lines at 16%
  - Test path listing functionality
  - Test format options (yaml, list, json, dict)

- [x] **search/decorators/** - ~130 lines combined ✅ (completed 2025-12-10)
  - [x] `cache.py` - Tests in `tests/search/decorators/test_cache.py`
  - [x] `error_handling.py` - Tests in `tests/search/decorators/test_error_handling.py`
  - [x] `performance.py` - Tests in `tests/search/decorators/test_performance.py`
  - [x] `tool_recommendations.py` - Tests in `tests/search/decorators/test_tool_recommendations.py`

---

## Phase 3: Higher Effort (Est. +10-15% coverage)

Requires mocking external services or complex setup.

- [ ] **services/docs_proxy_service.py** - 272 lines at 18%
  - Mock HTTP requests
  - Test document fetching
  - Test error handling

- [ ] **services/docs_server_manager.py** - 198 lines at 55%
  - Test server lifecycle
  - Test health monitoring
  - Mock process management

- [ ] **cli.py** - 60 lines at 0%
  - Test CLI argument parsing
  - Test command execution
  - Use click testing utilities

- [ ] **graph_analyzer.py** - 86 lines at 0%
  - Test graph construction
  - Test relationship detection

- [ ] **core/physics_categorization.py** - 57 lines at 0%
  - Test physics domain classification

- [ ] **core/clusters.py** - 147 lines at 25%
  - Test cluster operations

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

---

## Progress Log

### 2025-12-10: Phase 2 - Partial Completion
- Created comprehensive tests for `clusters/` modules:
  - `test_clustering.py` - 33 tests covering similarity computation, centroid calculation, clustering operations
  - `test_preprocessing.py` - 18 tests covering path filtering and unit family building
  - `test_labeler.py` - 7 tests covering cluster label generation
  - `test_label_cache.py` - 21 tests covering SQLite-based label caching
- Created comprehensive tests for `search/decorators/`:
  - `test_cache.py` - 20 tests covering cache entries, SimpleCache, cache keys, decorators
  - `test_error_handling.py` - 28 tests covering error responses, suggestions, decorators
  - `test_performance.py` - 17 tests covering metrics, scoring, decorators
  - `test_tool_recommendations.py` - 16 tests covering search analysis, suggestions
- Created tests for `core/xml_utils.py` - 10 tests covering documentation building and XML utilities
- **Total new tests added:** ~170 tests
- **All tests passing:** 447 passed, 4 skipped
