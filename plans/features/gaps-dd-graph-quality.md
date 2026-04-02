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

### ✅ Decision: Add `error` to schema (Option 1)

**Add `error: string` to SignalEpoch in `facility.yaml`.**

The codebase already implements a correct two-tier error handling pattern:

| Error type | Handler | Status change | Retryable? |
|-----------|---------|---------------|------------|
| **Transient** (SSH timeout, network) | `release_version_claim()` | None — stays `discovered` | Yes, automatic |
| **Permanent** (tree not found, corrupt data) | `mark_version_failed()` | → `failed` + `error` recorded | No — terminal |

The `error` field is the observability companion to the already-existing `failed` status in
`IngestionStatus`. Every other facility node type with a `failed` status has an `error` field
(CodeFile, WikiPage, FacilityPath). SignalEpoch must follow the same pattern.

**Why not "reset status for retry" on all errors:**
- Permanent errors (tree file not found on disk) will recur on every retry attempt
- This creates infinite retry loops — every discovery run wastes SSH connections hitting the same failures
- The transient retry path already exists via `release_version_claim()` + `claimed_at` timeout recovery
- Resetting permanently-failed nodes is an operational decision (migration query), not an automatic one

**Implementation:** One-line schema addition + `uv run build-models --force`.

---

## Group 4: MDSplus Worker Error Handling Inconsistency

Discovered during the SignalEpoch error property investigation.

### Bug 6: Extraction-returned errors released instead of marked failed

**File:** `discovery/mdsplus/workers.py` line 107
**Comparison:** `discovery/signals/parallel.py` line 2887

Both workers handle the same scenario — the remote extraction script returns an error in `ver_data["error"]`
(e.g., "tree file not found on disk") — but handle it differently:

| Worker | Handler called | Result |
|--------|---------------|--------|
| `signals/parallel.py` | `mark_version_failed(version_id, error_msg)` | Status → `failed`, error recorded ✅ |
| `mdsplus/workers.py` | `release_version_claim(version_id)` | Claim released, stays `discovered` ❌ |

The `mdsplus/workers.py` path creates a silent retry loop: every discovery run re-claims the same
broken version, hits the same error, releases it, and moves on — wasting SSH connections each time.
The error message is logged but never persisted.

**Severity: Medium** — silent performance drain on every rediscovery run for facilities with
broken tree versions.

### Fix

Change `mdsplus/workers.py` line 107 to call `mark_version_failed(version_id, ver_data["error"])`
instead of `release_version_claim(version_id)`. This matches the `signals/parallel.py` behavior
and is consistent with the two-tier error design: extraction-returned errors are permanent
(the tree file won't appear between runs), so they must be terminal.

### Tests

Add unit tests to `tests/discovery/mdsplus/`:
- Test that extraction-returned errors call `mark_version_failed()` (not `release_version_claim()`)
- Test that SSH/network exceptions still call `release_version_claim()` (transient → retry)
- Test that after `mark_version_failed()`, the node has `status='failed'` and `error` is populated
- Test that after `release_version_claim()`, the node stays `status='discovered'` with `claimed_at=null`

---

## Quick-Fix / Shortcut Flags

| Item | Description | Recommended Action |
|------|-------------|-------------------|
| ~~SignalEpoch error property~~ | ~~Code writes undeclared property~~ | **Resolved:** Add `error: string` to schema |
| CI deselects | 16 tests bypassed in CI | Rebuild graph, validate, then remove |

---

## Priority & Dependencies

**Priority: P1 — Release blocker**

| Depends On | Enables |
|-----------|---------|
| None | v5.0.0 release, container deployments |

### Execution order
1. Fix Bug 2 (cluster scope syntax) — immediate, small fix
2. Fix Bug 3 (overview counts) — add `node_category = 'data'` filter
3. Fix Bug 4 (export filtering) — add `node_category` filter + `include_errors` param
4. Fix Bug 5 (redundant query) — low priority optimization
5. Fix SignalEpoch Issue 6 — add `error: string` to schema (decision resolved)
6. Fix Bug 6 (mdsplus worker error handling) — call `mark_version_failed` + unit tests
7. Graph rebuild + GHCR push
8. Validate and remove 16 CI test deselects
9. Cut v5.0.0 release tag

## Overlap Notes

This gap document consolidates work from three pending plans:
- `pending/dd-version-and-tool-filtering.md` — Bugs 2-5 originate here
- `pending/graph-quality-v5-release.md` — CI deselects and release sequence originate here
- `pending/schema-compliance-remediation.md` — SignalEpoch Issue 6 originates here

**Conflict resolved:** Schema-compliance plan assumed removing SignalEpoch `error` property; graph-quality plan deferred to a design decision. **This document defers to the design decision approach** — the property should be evaluated before acting.

**Redundancy resolved:** Schema-compliance Issues 1-5 are already fixed in code. Only Issue 6 remains and is tracked here. The pending plans are reference material only.

## Documentation Updates

When this work is complete, update:
- [ ] `AGENTS.md` — if any new MCP tool parameters are added (e.g., `include_errors` on export)
- [ ] `plans/README.md` — mark this gap doc as complete, remove from active plans
- [ ] `.github/workflows/graph-quality.yml` — remove `--deselect` lines (part of implementation)
- [ ] Release notes for v5.0.0
