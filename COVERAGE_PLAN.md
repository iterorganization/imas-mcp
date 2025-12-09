# Coverage Improvement Plan

**Current Coverage:** 55.09%  
**Target Coverage:** >95%  
**Date Created:** December 9, 2025
**Last Updated:** December 9, 2025

---

## Phase 1: Quick Wins (Est. +15-20% coverage) ✅ COMPLETED (+9.61%)

High-impact, relatively straightforward unit tests.

- [x] **core/extractors/** - 417 lines at 0% → **93.6% avg**
  - `base.py` (80 lines) → 97.50%
  - `coordinate_extractor.py` (21 lines) → 95.24%
  - `identifier_extractor.py` (137 lines) → 84.67%
  - `metadata_extractor.py` (98 lines) → 86.73%
  - `physics_extractor.py` (13 lines) → 100%
  - `semantic_extractor.py` (45 lines) → 95.56%
  - `validation_extractor.py` (23 lines) → 100%

- [x] **search/document_store.py** - 443 lines at 31% → **46.17%**
  - Focus on document indexing and retrieval
  - Test search query building
  - Test result filtering

- [x] **search/engines/** - ~200 lines combined → **63.7% avg**
  - `lexical_engine.py` (109 lines at 16%) → 71.54%
  - `hybrid_engine.py` (55 lines at 26%) → 58.11%
  - `semantic_engine.py` (34 lines at 35%) → 61.54%

- [x] **dd_accessor.py** - 134 lines at 29% → **79.89%**
  - Test IDS accessors
  - Test path resolution
  - Test caching behavior

---

## Phase 2: Medium Effort (Est. +15-20% coverage)

Requires more setup but still well-defined test boundaries.

- [ ] **clusters/** modules - ~500 lines combined
  - `clustering.py` (183 lines at 15%)
  - `extractor.py` (279 lines at 12%)
  - `labeler.py` (92 lines at 22%)
  - `preprocessing.py` (64 lines at 17%)
  - `label_cache.py` (96 lines at 29%)

- [ ] **core/xml_parser.py** - 310 lines at 0%
  - Test XML element parsing
  - Test path extraction
  - Test attribute handling

- [ ] **core/xml_utils.py** - 69 lines (improved to 78%)
  - Test utility functions

- [ ] **tools/list_tool.py** - 146 lines at 16%
  - Test path listing functionality
  - Test format options (yaml, list, json, dict)

- [ ] **search/decorators/** - ~130 lines combined
  - `cache.py` (79 lines at 59%)
  - `error_handling.py` (113 lines at 59%)
  - `performance.py` (101 lines at 51%)
  - `tool_recommendations.py` (71 lines at 13%)

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
| core/extractors/base.py | 97.5% |
| core/extractors/physics_extractor.py | 100% |
| core/extractors/validation_extractor.py | 100% |
| core/extractors/semantic_extractor.py | 95.6% |
| core/extractors/coordinate_extractor.py | 95.2% |
| tools/path_tool.py | 93% |
| tools/base.py | 90% |
| core/exclusions.py | 88% |
| health.py | 85% |
| services/response.py | 84% |
| dd_accessor.py | 80% |

---

## Testing Strategy Notes

### Fixtures Created (Phase 1)
- ExtractorContext with mock DD accessor and sample XML
- Mock DocumentStore with configurable behavior
- Mock search engines (semantic, lexical, hybrid)
- Sample identifier schema XML

### Fixtures to Create (Phase 2+)
- Mock embedding encoder
- Sample XML fragments for parser tests
- Mock HTTP server for docs proxy tests
- Cluster data fixtures

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
