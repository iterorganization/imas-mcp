# Coverage Improvement Plan

**Current Coverage:** 47.77%  
**Target Coverage:** >95%  
**Date Created:** December 9, 2025  
**Last Updated:** December 9, 2025

## Progress Summary
| Phase | Status | Coverage Gain |
|-------|--------|---------------|
| Phase 1: Quick Wins | Pending | Est. +15-20% |
| Phase 2: Medium Effort | Pending | Est. +15-20% |
| Phase 3: Higher Effort | Pending | Est. +10-15% |
| Phase 4: Final Polish | ✅ COMPLETED | **+2.15%** |

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

## Completed: Final Polish Coverage

Edge cases, error paths, and remaining gaps.

**Status: COMPLETED - December 9, 2025**
**Coverage Gain: +2.15%** (45.62% → 47.77%)

- [x] **tools/** remaining gaps
  - `overview_tool.py` ~~71%~~ → **86.67%** (+14.77%)
  - `identifiers_tool.py` ~~74%~~ → **91.60%** (+17.65%)
  - `clusters_tool.py` (81%)
  - `docs_tool.py` (82%)
  - `search_tool.py` (84%)

- [x] **embeddings/** remaining gaps
  - `openrouter_client.py` ~~53%~~ → **57.14%** (+4.14%)
  - `cache.py` ~~68%~~ → **76.47%** (+8.47%)
  - `encoder.py` ~~76%~~ → **79.11%** (+3.11%)

- [x] **settings.py** ~~63%~~ → **74.39%** (+10.98%)
  - Test environment variable handling
  - Test configuration loading

- [x] **server.py** ~~68%~~ → **72.66%** (+5.03%)
  - Test server initialization
  - Test error handling

- [x] **mappings/__init__.py** - 62% (partial)
  - Test mapping operations

- [x] **search/semantic_search.py** - partial coverage
  - Test semantic search functionality

- [x] **search/cache.py** ~~24%~~ → **88.89%** (+64.89%)
  - Test cache hit/miss scenarios

- [x] **search/tool_suggestions.py** ~~13%~~ → **92.31%** (+78.85%)
  - Test suggestion logic

### Test Files Added
- `tests/search/test_tool_suggestions.py`
- `tests/search/test_cache.py`
- `tests/search/test_semantic_search.py`
- `tests/embeddings/test_cache.py`
- `tests/embeddings/test_openrouter_client.py`
- `tests/mappings/test_path_map.py`
- `tests/test_settings.py`
- `tests/test_server_extended.py`
- `tests/tools/test_overview_tool.py`
- Extended `tests/tools/test_identifiers_tool.py`

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
