> **⚠ Reference only** — remaining gaps from this plan are tracked in [`../gaps-dd-graph-quality.md`](../gaps-dd-graph-quality.md). This file is retained as implementation context.

# v5.0.0 Release — Graph Quality & Container Validation

## Phase 1: RC12 Release & Container Validation (in progress)

- [ ] Release v5.0.0-rc12 from upstream (iterorganization)
- [ ] Monitor CI pipeline (graph-quality → smoke-test → build-and-push)
- [ ] Pull DD-only container from GHCR, validate locally
- [ ] Exercise all MCP tools on container

## Phase 2: Graph Quality Fixes (Issue #28) — 16 deselected CI tests

### Problem Statement

The `graph-quality` CI job deselects 16 tests because the IMAS-only graph pushed to GHCR has known data issues. These must ALL pass before v5.0.0 final release. Root causes span schema gaps, DD build pipeline bugs, stale graph data, and missing COCOS build steps.

### Investigation Findings

**Status management (discovery CLIs):** System is well-designed. All pipelines follow claim-process-release with proper error handling. Status never advances on error. Orphan recovery via timeout handles edge cases. No high-severity issues. The 4 low-severity issues (Neo4j failure during status update) are self-healing via claim timeouts.

**DD build pipeline (issue #28 — 7 originally reported tests):**

### Fix 1: Schema — Add EmbeddingChange class to imas_dd.yaml
- **Test:** `test_all_labels_in_schema` — `EmbeddingChange` label not declared
- **Root cause:** `build_dd.py:315` defines `EmbeddingChangeType` class and `record_embedding_changes()` creates `EmbeddingChange` nodes, but no corresponding class in `imas_dd.yaml`
- **Fix:** Add `EmbeddingChange` class to `imas_codex/schemas/imas_dd.yaml` with properties: `path_id`, `change_type` (range: ChangeType), `dd_version`, `detected_at`, `old_hash`, `new_hash`
- **Files:** `imas_codex/schemas/imas_dd.yaml`

### Fix 2: Schema — Add cluster_input_hash to DDVersion
- **Test:** `test_no_undeclared_properties` — `DDVersion.cluster_input_hash` undeclared
- **Root cause:** `build_dd.py:2932-3167` writes `cluster_input_hash` to DDVersion nodes for cluster build caching, but not declared in schema
- **Fix:** Add `cluster_input_hash` slot to DDVersion in `imas_dd.yaml`
- **Files:** `imas_codex/schemas/imas_dd.yaml`

### Fix 3: Schema — Add tags to IMASSemanticCluster
- **Test:** `test_no_undeclared_properties` — `IMASSemanticCluster.tags` undeclared
- **Root cause:** `build_dd.py:3301,3313` writes `tags` from `ClusterLabel.tags` (`labeler.py:44`), not in schema
- **Fix:** Add `tags` slot (multivalued: true) to IMASSemanticCluster in `imas_dd.yaml`
- **Files:** `imas_codex/schemas/imas_dd.yaml`

### Fix 4: Graph migration — Remove stale test properties
- **Test:** `test_no_undeclared_properties` — `IMASNode.test_case`, `IMASNode.test_prop`
- **Root cause:** Properties from development/testing experiments, not written by any current code
- **Fix:** Cypher migration to remove these properties before graph push
- **Action:** `MATCH (n:IMASNode) WHERE n.test_case IS NOT NULL REMOVE n.test_case, n.test_prop`

### Fix 5: Code fix — Orphaned Unit node cleanup
- **Test:** `test_unit_nodes_have_relationships` — 361 orphaned Unit nodes (no HAS_UNIT)
- **Root cause:** `_create_unit_nodes()` creates Units from all DD versions, but `HAS_UNIT` relationships only created for current version batch. Units renamed/removed in later versions become orphaned.
- **Fix:** Add orphan Unit cleanup at end of `augment_dd_with_metadata()` — delete Unit nodes with no incoming HAS_UNIT relationships
- **Files:** `imas_codex/graph/build_dd.py` or `imas_codex/cli/imas_dd.py`

### Fix 6: Code fix — Include COCOS in DD build
- **Tests:** `test_cocos_reference_nodes` (0 COCOS, expected 16), `test_dd_versions_linked_to_cocos` (0 linked)
- **Root cause:** COCOS creation code exists (`_create_cocos_nodes()` in `build_dd.py:2053-2118`) but graph dump on GHCR was built before this was implemented, or build was interrupted
- **Fix:** Rebuild DD graph with current code ensures COCOS nodes are created. Verify `_create_cocos_nodes()` is called in the build pipeline.
- **Action:** Rebuild graph → re-push (no code changes if COCOS code is correct)

### Fix 7: Reconcile PhysicsDomainDD ↔ PhysicsDomain
- **Test:** `test_enum_values_valid` — invalid `physics_domain` values on IMASNode
- **Root cause:** Two competing enums:
  - `PhysicsDomainDD` (imas_dd.yaml:166) — 11 values: `equilibrium`, `core_profiles`, `edge_profiles`, `transport`, `mhd`, `waves`, `particles`, `diagnostics`, `engineering`, `control`, `general`
  - `PhysicsDomain` (physics_domains.yaml) — 22 values: detailed physics categories
  - IMASNode uses `PhysicsDomainDD` but test checks against `PhysicsDomain`
- **Fix options:**
  a) Unify: Make IMASNode use the canonical `PhysicsDomain` enum, build a mapping from DD values → canonical values
  b) Keep separate: Ensure test correctly checks `PhysicsDomainDD` for IMASNode
  c) Migrate: Map all `PhysicsDomainDD` values to nearest `PhysicsDomain` values and drop `PhysicsDomainDD`
- **Recommended:** Option (c) — single enum is cleanest. Build mapping table and migrate graph data.

### Fix 8: Test fix — Skip facility tests for DD-only graph
- **Test:** `test_facility_nodes_exist` — No Facility nodes in IMAS-only graph (by design)
- **Root cause:** Test unconditionally asserts Facility nodes exist, but DD-only graph has none
- **Fix:** Add `pytest.skip` when graph has no Facility nodes (detected via GraphMeta or label count)
- **Files:** `tests/graph/test_structural.py`

### Fix 9: Remaining 9 deselected tests (not in issue #28)
- `test_required_fields_present` — Likely passes after schema fixes
- `test_identifiers_non_null` — Likely passes after schema fixes
- `test_identifier_uniqueness` — Likely passes after schema fixes
- `test_constraints_created` — Requires `gc.initialize_schema()` to succeed
- `test_facility_id_edges_exist` — Skip for DD-only (no facility nodes)
- `test_no_zero_embeddings` — May need embedding rebuild
- `test_embeddings_are_normalized` — May need embedding rebuild
- `test_no_empty_string_descriptions` — Graph cleanup needed
- `test_wiki_pages_have_url` — Skip for DD-only (no wiki data)

### Execution Order (respecting dependencies)

```
1. Schema fixes (Fix 1, 2, 3)        → uv run build-models --force
2. Code fix (Fix 5 — orphan units)   → modify build_dd.py
3. Test fix (Fix 8 — DD-only skip)   → modify test files
4. PhysicsDomain reconciliation (Fix 7) → schema + migration
5. Rebuild DD graph on ITER           → uv run imas-codex dd build
6. Graph migrations (Fix 4)           → run Cypher before push
7. Re-push DD graph to GHCR           → uv run imas-codex graph push
8. Remove all --deselect lines from CI → update graph-quality.yml
9. Release RC with fixes              → verify all 16 tests pass
```

### Status Management — Investigation Summary

All discovery pipelines follow robust claim-process-release:
- **Paths:** `_revert_path_claims()` on SSH/LLM failure ✅
- **Code:** `release_path_file_scan_claim()` / `release_file_triage_claims()` ✅
- **Wiki:** `_release_claimed_pages()` with retry counters + recovery functions ✅
- **Signals:** Circuit breaker (3 consecutive failures → stop) + retry counters ✅
- **Infrastructure:** `is_infrastructure_error()` classification, dual-budget supervision ✅
- **Orphan recovery:** `reset_stale_claims()` reclaims after 300-600s timeout ✅

4 low-severity issues (all self-healing via timeouts):
1. Neo4j failure during `_mark_files_ingested()` — file stays scored, re-ingested on retry
2. Neo4j failure during `mark_pages_scored()` — page stays scanned, LLM cost wasted
3. Silent exception in `docs_score_worker()` — document stays claimed until orphan recovery
4. `_mark_file_failed()` failure — file stays scored until orphan recovery

**Verdict:** No code changes needed for status management. System is self-healing.
