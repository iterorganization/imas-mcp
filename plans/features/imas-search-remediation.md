# IMAS Search Quality Remediation Plan

> **Date**: 2026-03-09
> **Status**: Active
> **Depends on**: mcp-tool-quality-refactor.md (Phase 1 bugs now fixed)

## Executive Summary

Extensive testing of all MCP search tools against live graph data (944K nodes, 2.97M relationships) reveals that **`search_imas` is the highest-impact quality gap** remaining. The graph contains correct IMAS data — text search on `IMASPath.documentation` finds canonical paths like `equilibrium/time_slice/global_quantities/ip` for "plasma current" — but the scoring formula allows vector-similarity noise to overwhelm these text matches.

All other search tools (`search_signals`, `search_docs`, `search_code`) function correctly after the Phase 1 bug fixes from `mcp-tool-quality-refactor.md`. The remaining issues are concentrated in IMAS search quality and fetch completeness.

---

## Issues Identified

### Critical: IMAS Search Returns Wrong Results

**Symptom**: `search_imas("plasma current")` returns 7/10 results with documentation "Verbose description" (generic metadata paths like `pulse_schedule/lh/antenna/power_type/description`). Canonical paths like `magnetics/ip`, `equilibrium/time_slice/global_quantities/ip`, `summary/global_quantities/ip` are missing entirely despite existing in the graph WITH embeddings.

**Root Causes** (3 compounding issues):

1. **`_text_search_imas_paths()` is a dead stub** — returns `[]` always. Called from `_vector_search_imas_paths()` but never implemented. The intent was to merge text + vector results at the vector search stage, but the function body is just `return []`.

2. **Text scores are non-competitive** — `_text_search_imas_paths_by_query()` assigns fixed scores: 0.7 for leaf nodes, 0.6 for word matches, 0.4 for structures. Vector similarity returns 0.88+ for "Verbose description" paths. The merge formula `score * 0.7 + text_score * 0.3 + 0.1` can never push a 0.7 text result above a 0.88 vector result. Maximum hybrid score for text-only match: 0.7. Minimum vector score for garbage: 0.88.

3. **`_is_generic_metadata_path()` always returns `False`** — the filter for "Verbose description" paths was stubbed out. These paths have near-identical short documentation strings that produce misleadingly high vector similarity.

**Evidence**:
```
Text search for "plasma current" → 15+ correct paths (ip, j_parallel, j_phi)
Vector search for "plasma current" → 7/10 "Verbose description" garbage at 0.88+
Combined output → garbage dominates because 0.88 > 0.7
```

**Comparison with old IMAS MCP server**: The old server (`GraphSearchTool.search_imas_paths()`) uses pure vector search and returns **equally poor results** — same "Verbose description" pollution. Neither server implements hybrid search correctly. The codex server has the infrastructure (fulltext indexes created, `_text_search_imas_paths_by_query` works) but the scoring formula prevents text results from surfacing.

### Medium: Fetch Tool Gaps

1. **No IMAS path fetch** — `_fetch()` resolves WikiPage, Document, CodeFile, Image only. No way to fetch detailed IMAS path information (documentation, coordinates, units, version history, facility cross-references) by ID.

2. **Unparsed documents return "No resource found"** — Binary documents (PowerPoint, Excel without parsing) have 0 WikiChunks. `_fetch_wiki_document()` requires `HAS_CHUNK` relationship to return anything. Should return document metadata (title, URL, file type, size) even when content isn't parsed.

### Low: Embedding Coverage Gaps

| Node Type | Total | Embedded | Coverage | Notes |
|-----------|-------|----------|----------|-------|
| FacilitySignal | 66,510 | 66,510 | 100% | Fixed this session |
| WikiChunk | 115,697 | 115,697 | 100% | Complete |
| CodeChunk | 36,370 | 36,360 | 99.9% | Near complete |
| DataNode | 89,340 | 34,082 | 38.1% | 55K have no description |
| IMASPath | 61,366 | 22,160 | 36.1% | Many structural paths |
| CodeExample | 10,065 | 0 | 0% | No embedding index |

DataNode and IMASPath coverage is limited by nodes having no or very short descriptions. CodeExample has no embedding support in the schema.

### Low: Orphaned CodeChunks

4,419 CodeChunks (3,626 TCV, 793 JET) have no `source_file_id`, no `FROM_SOURCE` relationship, and no `HAS_CHUNK` relationship. Only connected via `AT_FACILITY`. These are legacy ingestion artifacts from MATLAB files and cannot be linked to source files.

---

## Remediation Steps

### Step 1: Fix IMAS text search scoring (Critical)

**File**: `imas_codex/agentic/search_tools.py`

Replace the fixed-score text search with score normalization that makes text results competitive with vector results.

**Changes**:
- In `_text_search_imas_paths_by_query()`: Replace fixed scores (0.7/0.6/0.4) with a relevance-based scoring function that normalizes to the same 0.8-1.0 range as vector scores. Score based on: exact documentation match (0.95), name match (0.9), word-in-path match (0.85), partial doc match (0.8).
- In `_search_imas()`: Change the merge formula. When a path appears in BOTH text and vector results, use `max(vector_score, text_score) + 0.05` bonus. When text-only, use the text score directly (now competitive).

### Step 2: Remove `_text_search_imas_paths()` stub

**File**: `imas_codex/agentic/search_tools.py`

Delete the dead `_text_search_imas_paths()` function that returns `[]`. Remove its call from `_vector_search_imas_paths()`. The actual text search is done by `_text_search_imas_paths_by_query()` called from `_search_imas()`.

### Step 3: Implement `_is_generic_metadata_path()` filter

**File**: `imas_codex/agentic/search_tools.py`

Replace the stub with actual filtering logic:
```python
def _is_generic_metadata_path(gc: GraphClient, path_id: str) -> bool:
    # Filter paths ending in /description, /name, /identifier/...
    # whose documentation is "Verbose description" or similar generic text
    parts = path_id.split("/")
    if parts[-1] in ("description", "name") and len(parts) > 3:
        return True
    return False
```

This catches the ~2000 "Verbose description" paths that pollute results without requiring a graph lookup.

### Step 4: Add IMAS path fetch to `_fetch()`

**File**: `imas_codex/agentic/search_tools.py`

Add `_fetch_imas_path()` resolver:
- Match by `IMASPath.id` (exact or contains)
- Return: full documentation, units, data type, coordinates, lifecycle, structure reference, version history, cluster membership, facility cross-references
- Register in the resolver chain in `_fetch()` after `_fetch_image`

### Step 5: Handle unparsed document metadata in fetch

**File**: `imas_codex/agentic/search_tools.py`

Modify `_fetch_wiki_document()`:
- First try normal chunk fetch (existing behavior)
- If no chunks found, fall back to metadata-only query:
  ```cypher
  MATCH (a:Document) WHERE a.id = $resource OR ...
  RETURN a.title, a.url, a.filename, a.file_type, a.file_size, a.description
  ```
- Format as metadata summary with note that content parsing is not available

### Step 6: Clean up orphaned CodeChunks (Optional)

4,419 orphaned CodeChunks with no source linkage. Options:
1. **Delete** — removes noise from search results (preferred)
2. **Tag** — mark as `orphaned` status to exclude from search

---

## Validation Plan

After implementing Steps 1-5, verify with these queries:

```python
# IMAS search: canonical paths must appear in top 5
result = _search_imas("plasma current", k=10)
assert "magnetics/ip" in result or "global_quantities/ip" in result
assert "Verbose description" not in result  # or at most 1 instance

# IMAS search: electron temperature
result = _search_imas("electron temperature profile", k=10)
assert "core_profiles" in result
assert "electrons/temperature" in result

# Fetch IMAS path
result = _fetch("equilibrium/time_slice/global_quantities/ip")
assert "Plasma current" in result
assert "unit" in result.lower()

# Fetch unparsed document
result = _fetch("jet:Eq_recon.ppt")
assert "No resource found" not in result  # should return metadata
```

## Priority

1. Steps 1-3 (IMAS search quality) — highest impact, fixes the primary user complaint
2. Step 4 (IMAS fetch) — enables IMAS path drill-down
3. Step 5 (unparsed doc fetch) — quality of life
4. Step 6 (orphan cleanup) — optional housekeeping
