# DD & Graph Quality — Outstanding Gaps

Consolidated from three partially-implemented plans now in `pending/`:
- `pending/dd-version-and-tool-filtering.md`
- `pending/graph-quality-v5-release.md`
- `pending/schema-compliance-remediation.md`

## Status Summary

| Source Plan | Implemented | Remaining |
|-------------|------------|-----------|
| dd-version-and-tool-filtering | Phase 1 (critical fix), regression tests, semver filtering | Bugs 2-5, Phase 4 enrichments |
| graph-quality-v5-release | All code/schema fixes (1-3, 5-8) | Graph rebuild, CI deselect removal, v5 release |
| schema-compliance-remediation | 5 of 6 issues fixed | Issue 6 (SignalEpoch error property) |

---

## Group 1: DD Version Tool Bugs

Source: `pending/dd-version-and-tool-filtering.md`

### Bug 2: Cluster scope syntax error
- `_search_by_path` (graph_search.py ~line 1508) places `AND c.scope = $scope` after MATCH without `WHERE`
- Produces a Cypher syntax error when scope parameter is provided
- **Severity: Medium** — affects cluster search with scope filter

### Bug 3: Overview path counts inflated
- `get_imas_overview` (graph_search.py ~line 1276) has no `node_category = 'data'` filter
- Path counts are inflated 2-3× because error/metadata nodes are counted
- **Severity: Medium** — misleading statistics

### Bug 4: Export includes error/metadata nodes
- `export_imas_ids` (~line 2112) and `export_imas_domain` have no `node_category` filter
- No `include_errors` parameter exists on export tools
- Exports include error and metadata nodes that should be filtered by default
- **Severity: Medium** — noisy exports

### Bug 5: Redundant query in list tool
- `list_imas_paths` still runs two queries for IDS-only mode; first result discarded
- **Severity: Low** — functionally correct, just wasteful

### Phase 4: Search enrichment gaps
- HAS_ERROR enrichment tests exist but are SKIPPED (`test_search_enrichment_query_includes_has_error`)
- Error fields not attached to search hits at query time
- Cluster-aware reranking is basic heuristic, not full deduplication

---

## Group 2: v5 Release Operational Work

Source: `pending/graph-quality-v5-release.md`

### Fix 4: Stale test properties in graph
- No migration to remove `test_case`/`test_prop` properties from production graph nodes
- Need to run cleanup Cypher after graph rebuild

### Fix 9: CI deselect removal
- All 16 `--deselect` lines still active in `.github/workflows/graph-quality.yml:225-240`
- Tests need validation against rebuilt graph before deselect lines can be removed
- Blocked on: graph rebuild + push to GHCR

### Release sequence
1. Rebuild graph with current code (all fixes applied)
2. Run stale property migration
3. Push to GHCR
4. Validate all 16 deselected tests pass
5. Remove `--deselect` lines from CI
6. Cut final v5.0.0 release tag

---

## Group 3: Schema Compliance — SignalEpoch Error Property

Source: `pending/schema-compliance-remediation.md` (Issue 6)

### Problem
- `discovery/mdsplus/graph_ops.py:247` writes `v.error = $error` on SignalEpoch nodes
- `error` property is **not declared** in the SignalEpoch schema (`facility.yaml`)
- This is an active schema violation on every failed SignalEpoch write

### Fix options (pick one)
1. **Add to schema**: Add `error: string` slot to SignalEpoch in `facility.yaml` — if we want to track error messages
2. **Remove from code**: Stop writing `v.error` in `graph_ops.py` — if error tracking isn't needed

### ⚠ Shortcut flag
This is a live schema violation that was missed during the remediation work. The other 5 issues were fixed but this one was skipped — likely because it requires a design decision about whether to keep error tracking on SignalEpoch nodes.

---

## Quick-Fix / Shortcut Flags

| Item | Description | Recommended Action |
|------|-------------|-------------------|
| SignalEpoch error property | Code writes undeclared property | Design decision needed, then fix schema or code |
| CI deselects | 16 tests bypassed in CI | Rebuild graph, validate, then remove |
