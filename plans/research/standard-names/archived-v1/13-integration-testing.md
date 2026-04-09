# Feature 13: Integration Testing & CI

**Status:** Pending
**Priority:** Medium — validates all other plans work together
**Depends on:** All other plans (09-12, 14) — at least Phase 1 of each
**Parallel with:** None — this is the final validation layer
**Estimated complexity:** Medium

---

## Problem

The SN pipeline lacks end-to-end tests that exercise the full 8-phase flow.
Individual phases are tested in their respective plans (09-12, 14), but
integration between phases — data flow, state transitions, error propagation,
and the complete `sn build` CLI experience — is not covered.

The existing test suite (`tests/sn/`) tests grammar, composition, and the
MCP server, but nothing for the pipeline orchestration, graph persistence,
link validation, or quality scoring.

## Approach

Three test layers:

1. **Pipeline integration tests** — full 8-phase flow with mocked LLM
2. **CLI integration tests** — `sn build`, `sn link`, `sn score`, `sn status`
3. **CI workflow updates** — ensure SN tests run in CI

### Pipeline Architecture (8 phases)

```
EXTRACT → COMPOSE → REVIEW → VALIDATE → DOCUMENT → PERSIST_NODES → LINK → SCORE
```

Each phase produces output consumed by the next. Integration tests verify
the full chain without mocking intermediate state.

---

## Phase 1: Pipeline Integration Tests

### Tasks

1. **Create pipeline test fixtures**
   - File: `tests/sn/conftest.py`
   - `mock_dd_paths()` — fixture returning DD path metadata for EXTRACT
   - `mock_llm_compose()` — returns valid compose output (grammar fields)
   - `mock_llm_review()` — returns review verdicts (accept/revise/reject)
   - `mock_llm_document()` — returns rich documentation with cross-refs
   - `mock_llm_score()` — returns SNScoreFields
   - `mock_graph_client()` — in-memory graph stub for PERSIST/LINK/SCORE
   - Each fixture returns data matching the exact Pydantic models used
     by workers

2. **Test full pipeline flow**
   - File: `tests/sn/test_pipeline_integration.py`
   - Test: `test_full_pipeline_dd_source`
     - Input: 5 DD paths from equilibrium IDS
     - Expected: all 8 phases execute in order
     - Verify: state transitions (extracted→composed→reviewed→validated→
       documented→persisted→linked→scored)
     - Verify: data flows between phases (compose output → review input)
     - Verify: total_cost accumulates across LLM phases
     - Mock: all LLM calls, graph client

3. **Test phase skipping**
   - Test: `test_skip_document_phase`
     - `--skip-document` → pipeline skips DOCUMENT, PERSIST_NODES still runs
       (with empty docs)
   - Test: `test_skip_link_phase`
     - `--skip-link` → pipeline skips LINK, SCORE still runs (score_link_quality=0)
   - Test: `test_skip_score_phase`
     - `--skip-score` → pipeline skips SCORE, results still persisted
   - Test: `test_no_persist_mode`
     - `--no-persist` → skips PERSIST_NODES, LINK, and SCORE
   - Test: `test_dry_run_mode`
     - `--dry-run` → no LLM calls, no graph writes, shows plan only

4. **Test error propagation**
   - Test: `test_compose_failure_stops_pipeline`
     - LLM error in COMPOSE → pipeline reports error, no downstream phases run
   - Test: `test_review_rejection_reduces_count`
     - REVIEW rejects 2/5 names → VALIDATE receives 3 names
   - Test: `test_validate_failure_reduces_count`
     - VALIDATE fails 1/3 names → DOCUMENT receives 2 names
   - Test: `test_persist_failure_stops_link_score`
     - Graph error in PERSIST_NODES → LINK and SCORE don't run

5. **Test pipeline phase dependencies**
   - Test: `test_phase_ordering`
     - Verify WorkerSpec depends_on chain:
       compose→review→validate→document→persist→link→score
   - Test: `test_parallel_phases_are_independent`
     - Phases without dependencies can run concurrently (none currently, but
       architecture supports it)

### Acceptance Criteria
- Full pipeline test exercises all 8 phases end-to-end
- Phase skipping tested for all skip flags
- Error propagation verified for each phase boundary
- No graph or MCP dependency (all mocked)

---

## Phase 2: CLI Integration Tests

### Tasks

1. **Test `sn build` command**
   - File: `tests/sn/test_cli_integration.py`
   - Test: `test_sn_build_basic`
     - `sn build --source dd --ids equilibrium --dry-run`
     - Verify: exits 0, shows pipeline plan
   - Test: `test_sn_build_with_all_flags`
     - `sn build --source dd --ids equilibrium --compose-model X --skip-document --no-persist`
     - Verify: flags propagated to state correctly
   - Test: `test_sn_build_summary_output`
     - Verify: summary includes phase stats, costs, timing

2. **Test `sn link` command**
   - File: `tests/sn/test_cli_integration.py`
   - Test: `test_sn_link_basic`
     - `sn link` — re-links names with unresolved references
   - Test: `test_sn_link_all`
     - `sn link --all` — re-validates all existing edges
   - Test: `test_sn_link_convergence`
     - Run link twice → fewer unresolved on second run

3. **Test `sn score` command**
   - File: `tests/sn/test_cli_integration.py`
   - Test: `test_sn_score_basic`
     - `sn score` — scores unscored names
   - Test: `test_sn_score_specific`
     - `sn score --name electron_temperature` — scores one name

4. **Test `sn status` command**
   - File: `tests/sn/test_cli_integration.py`
   - Test: `test_sn_status_includes_link_health`
     - Verify link health section in output
   - Test: `test_sn_status_includes_score_summary`
     - Verify score statistics in output

5. **Test `sn benchmark` command**
   - File: `tests/sn/test_cli_integration.py`
   - Test: `test_sn_benchmark_uses_shared_scorer`
     - Verify benchmark imports from `sn/scorer.py`
   - Test: `test_sn_benchmark_comparison`
     - `sn benchmark --compare` shows model comparison table

### Acceptance Criteria
- All CLI commands exit cleanly
- Flag propagation verified
- Output format validated

---

## Phase 3: Graph Integration Tests

### Tasks

1. **Test StandardName graph round-trip**
   - File: `tests/sn/test_graph_integration.py`
   - Test: `test_persist_and_read_back`
     - Write StandardName → read back → verify all fields
   - Test: `test_cross_references_edge_creation`
     - Persist two names → create CROSS_REFERENCES edge → verify
   - Test: `test_depends_on_edge_creation`
     - Persist two names → create DEPENDS_ON edge → verify
   - Test: `test_degree_counter_accuracy`
     - Create multiple edges → verify ref_in_degree, ref_out_degree
   - Test: `test_unresolved_refs_property`
     - Persist name with unresolved → verify property stored
     - Later persist target → re-link → verify edge created and property cleared

2. **Test catalog mirroring**
   - File: `tests/sn/test_graph_integration.py`
   - Test: `test_import_catalog_creates_nodes`
   - Test: `test_import_catalog_idempotent`
   - Test: `test_generated_vs_mirrored_provenance`

3. **Test schema compliance**
   - File: `tests/sn/test_schema_compliance.py`
   - Verify StandardName nodes comply with LinkML schema
   - Verify all relationship types are declared
   - Verify score dimensions match schema declaration
   - Verify StandardNameStatus enum values match code usage

### Acceptance Criteria
- Graph round-trip preserves all data
- Edge creation is idempotent
- Degree counters accurate after edge modifications
- Schema compliance passes

---

## Phase 4: CI Configuration

### Tasks

1. **Update test workflow**
   - File: `.github/workflows/test.yml` (or equivalent)
   - Ensure `tests/sn/` runs in CI
   - SN tests should not require:
     - Live graph connection (use mocks/fixtures)
     - MCP server running
     - API keys (mock LLM calls)
   - SN tests may optionally test against graph (mark with `@pytest.mark.graph`)

2. **Add SN benchmark CI job (optional)**
   - Only on manual trigger or tag push
   - Runs benchmark with reference set
   - Reports scores in CI summary

### Acceptance Criteria
- SN tests run in CI without external dependencies
- Graph tests can be skipped with marker
- CI reports SN test results clearly

---

## Phase 5: Test Coverage Targets

### Coverage Goals

| Module | Target | What to Test |
|--------|--------|-------------|
| `sn/pipeline.py` | 90% | Phase ordering, skip logic, error propagation |
| `sn/workers.py` | 85% | All 8 workers with mocked deps |
| `sn/scorer.py` | 95% | Composite function, calibration, dimensions |
| `sn/linker.py` | 95% | Resolution engine, classification, cycles |
| `sn/graph_ops.py` | 80% | CRUD, relationships, degree counters |
| `sn/state.py` | 90% | State transitions, stats accumulation |
| `sn/models.py` | 95% | Pydantic validation, serialization |
| `cli/sn.py` | 75% | CLI flag parsing, output format |

### Acceptance Criteria
- No module below 75% coverage
- Critical modules (scorer, linker, models) at 95%+
- Pipeline integration tests cover all 8 phases

---

## Files Modified / Created

| File | Change |
|------|--------|
| `tests/sn/conftest.py` | Pipeline fixtures, mock LLM/graph |
| `tests/sn/test_pipeline_integration.py` | NEW: 8-phase pipeline tests |
| `tests/sn/test_cli_integration.py` | NEW: CLI command tests |
| `tests/sn/test_graph_integration.py` | NEW: graph round-trip tests |
| `tests/sn/test_schema_compliance.py` | NEW: schema compliance tests |
| `.github/workflows/test.yml` | Add SN test job |

## Documentation Updates

- No external doc changes — tests are self-documenting
