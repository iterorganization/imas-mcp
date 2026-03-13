# Check Pipeline & MCP Tool Gap Remediation

> **Status**: Planning  
> **Priority**: High — check failures hide data availability, MCP gaps block analytics  
> **Goal**: Fix systemic check failures, improve error classification, and close  
> gaps between MCP search_signals tool and direct Cypher query capabilities.

## Problem Summary

### Check Pipeline Issues

The signal check pipeline validates that enriched signals can return data via
their configured data access method. Three systemic bugs cause whole signal
cohorts to fail checks despite the underlying data being accessible:

| Bug | Affected Signals | Error Pattern |
|-----|-----------------|---------------|
| numpy missing in remote venv | 2,427 | `No module named 'numpy'` |
| JPF 100% check failure | 2,358 | `empty path` |
| PPF shot-specific failures | 3,230 | `ppfdata ier=210002` |

Combined, these affect 8,015 signals — 99.4% of all check failures. Only
50 signals (~0.6%) fail for genuine data access reasons.

### MCP Tool Gaps

The `search_signals` MCP tool uses vector similarity search via the
`facility_signal_desc_embedding` index. Compared to direct Cypher queries,
it has several gaps:

1. **No check outcome access**: Cannot query which signals pass/fail checks
2. **No batch analytics**: Cannot aggregate counts by status, domain, source
3. **No relationship traversal**: Cannot follow CHECKED_WITH, MEMBER_OF edges
4. **Duplicate signal IDs in results**: Vector search can return the same
   signal from multiple embedding matches

---

## Plan G: Check Pipeline Fixes

### G.1 Static Scanner Routing

**Root Cause**: Signals from magnetics_config, pf_coil_turns, greens_table,
jec2020_geometry, and sensor_calibration are NOT being routed to the
DeviceXMLScanner for checking. They fall through to a default MDSplus
check path that requires SSH and a remote Python environment with numpy.

**File**: `imas_codex/discovery/signals/parallel.py`

Find `_get_signal_scanner_type()` and add routing for static data sources:

```python
STATIC_DATA_SOURCES = {
    "device_xml", "magnetics_config", "pf_coil_turns",
    "greens_table", "jec2020_geometry", "sensor_calibration",
}

def _get_signal_scanner_type(signal: dict) -> str:
    dsn = signal.get("data_source_name", "")
    if dsn in STATIC_DATA_SOURCES:
        return "device_xml"  # Route to DeviceXMLScanner (local graph check)
    ...
```

**Impact**: 2,427 signals immediately route to local graph checks instead
of failing with numpy import errors.

### G.2 JPF Path Resolution Fix

**Root Cause**: JPF signals use `accessor` as the signal identifier (e.g.,
`"DA/C1D-IPLA"`), not `node_path` or `data_source_path`. The JPF scanner's
`check()` method builds batch entries with:
```python
batch.append({"id": s.id, "path": s.node_path or s.data_source_path or ""})
```

When both `node_path` and `data_source_path` are empty at check time (which
happens when the signal was enriched before these fields were properly set),
the path falls through to `""`, causing 100% failure.

**File**: `imas_codex/discovery/signals/scanners/jpf.py`

Fix the path resolution to include accessor:
```python
batch.append({
    "id": s.id,
    "path": s.accessor or s.data_source_path or s.node_path or ""
})
```

JPF signals have accessors like `"DA/C1D-IPLA"` which the remote
`check_jpf.py` script uses directly to call `dpf()`.

Additionally, verify the `claim_signals_for_check` query returns `s.accessor`:
```cypher
RETURN s.id AS id, s.accessor AS accessor, ...
```

The FacilitySignalModel constructed from the check claim includes `accessor`
but the JPF scanner uses `node_path` — this mismatch causes the empty path.

**Impact**: Unblocks 2,358 JPF signals for proper checking.

### G.3 Multi-Shot Check Fallback

**Root Cause**: PPF `ier=210002` means the data doesn't exist for the specific
shot used (99896). Many PPF DDAs only have data for certain shots.

**File**: `imas_codex/discovery/signals/scanners/ppf.py` (or parallel.py check worker)

Implement a fallback strategy:
1. Try reference_shot first
2. If `ier=210002`, try 2-3 additional recent shots from facility config
3. Only mark as `not_available_for_shot` if ALL shots fail with ier=210002
4. Store `checked_shots` list on the CHECKED_WITH relationship

**Facility config addition** (`jet.yaml`):
```yaml
data_systems:
  ppf:
    check_shots: [99896, 99000, 96000, 92000]
    # Spread across different campaigns for coverage
```

**Impact**: Many of the 3,230 PPF signals likely have data for other shots.
Multi-shot checking reveals whether the signal is truly dead or just
shot-dependent.

### G.4 Improved Error Classification

**File**: `imas_codex/discovery/signals/parallel.py`

Expand `_classify_check_error()`:

```python
def _classify_check_error(error: str) -> str:
    if not error:
        return "unknown"
    err_lower = error.lower()

    # PPF-specific errors
    if "ier=210002" in error:
        return "not_available_for_shot"
    if "ier=260000" in error:
        return "invalid_sequence"

    # Missing dependencies (infrastructure, not data)
    if "no module named" in err_lower:
        return "missing_dependency"

    # Empty path (infrastructure bug, not data)
    if "empty path" in err_lower:
        return "empty_path"

    # Existing classifications...
    if "not found" in err_lower or "TreeNNF" in error:
        return "node_not_found"
    ...
```

Add an `is_infrastructure_error()` predicate:
```python
INFRASTRUCTURE_ERRORS = {
    "missing_dependency", "empty_path", "timeout",
    "connection_error", "segfault", "script_crash",
}

def is_infrastructure_error(error_type: str) -> bool:
    return error_type in INFRASTRUCTURE_ERRORS
```

### G.5 Re-check Infrastructure Failures

After fixing G.1 and G.2, reset infrastructure-failed signals:

```cypher
// Reset signals that failed due to infrastructure bugs
MATCH (s:FacilitySignal {facility_id: 'jet'})-[c:CHECKED_WITH]->(da:DataAccess)
WHERE c.success = false
  AND c.error_type IN ['missing_dependency', 'empty_path']
SET s.status = 'enriched',
    s.claimed_at = null
DELETE c
```

Then re-run check: `uv run imas-codex discover signals jet --check-only`

---

## Plan H: MCP Tool Enhancements

### H.1 Add Check Outcome Queries to search_signals

**File**: MCP tool definition for `search_signals`

Add optional parameters:
- `check_status: str` — filter by check outcome ("passed", "failed", "unchecked")
- `error_type: str` — filter by error classification
- `include_check_details: bool` — include CHECKED_WITH relationship data

### H.2 Add Batch Analytics Tool

New MCP tool `signal_analytics` for aggregated queries:

```python
def signal_analytics(
    facility: str,
    group_by: list[str],  # e.g., ["status", "physics_domain", "data_source_name"]
    filters: dict | None = None,
) -> dict:
    """Aggregate signal counts by specified dimensions."""
```

This replaces the need for direct Cypher for common analytics queries.

### H.3 Deduplicate Search Results

The vector search can return the same FacilitySignal multiple times if
its embedding matches from different index entries. Add deduplication
by signal ID in the MCP tool result processing:

```python
seen_ids = set()
deduplicated = []
for result in raw_results:
    if result["id"] not in seen_ids:
        seen_ids.add(result["id"])
        deduplicated.append(result)
```

---

## Dependencies

| Plan | Depends On | Risk |
|------|-----------|------|
| G.1 (Static routing) | None | Low |
| G.2 (JPF path) | None | Low |
| G.3 (Multi-shot) | G.4 | Low |
| G.4 (Error classify) | None | Low |
| G.5 (Re-check) | G.1, G.2 | Low |
| H.1 (Check in MCP) | G.4 | Low |
| H.2 (Analytics tool) | None | Medium |
| H.3 (Dedup) | None | Low |
