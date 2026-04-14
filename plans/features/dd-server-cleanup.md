# DD Server: Surgical Cleanup

> Priority: P2 — Three small fixes. The DD server tools work well already.

## Problem

The DD MCP server is ~95% functional. Rubber-duck review with independent
code investigation identified exactly three real remaining gaps. Everything
else in the original 30KB plan is either already implemented or superseded.

## Verified Current State (rubber-duck confirmed)

### Already Done (do NOT reimplement)
- `IMASNodeChange` population in `build_dd.py` — covers units, docs, data_type, node_type, lifecycle, coordinates, timebase, maxoccur, ndim, identifier enum, added/removed/renamed
- `CodeMigrationGuide` / `CodeUpdateAction` Pydantic models exist
- Search-pattern generation exists
- Migration guide structured builder exists
- Markdown renderer exists
- `list_dd_paths` header shows path count
- All 16+ MCP tools functional with rich output

### Explicitly Removed
- **`explain_concept` tool** — frontier LLMs already explain fusion concepts. This tool would be a presentation wrapper over existing data tools, adding zero value.

## Fix 1: `list_dd_paths` Truncation Reporting

**Problem**: `GraphListTool.list_dd_paths()` sets `path_count=len(path_ids)` AFTER
the LIMIT clause. When results are truncated, the header says "10 paths — showing
first 10" when the real total may be much larger. The `total_paths` field on
`ListPathsResultItem` is never populated.

**Fix**:
1. Add a COUNT query before the LIMIT query in `list_dd_paths()`
2. Populate `ListPathsResultItem.total_paths` with the true count
3. Formatter already handles truncation display — just feed it the real number

**Files**: `imas_codex/tools/graph_search.py` (list method), tests

**Size**: ~15 lines of code

## Fix 2: Migration Guide API Cleanup

**Problem**: The migration guide tool has stale parameters and the structured
output is not exposed through the MCP interface.

**Fix**:
1. Verify `include_recipes` parameter actually controls output (rubber-duck found it may be dead in the new code path)
2. Either remove stale parameters or wire them correctly
3. Optionally expose `include_structured` for programmatic consumers

**Files**: `imas_codex/tools/migration_guide.py`, `imas_codex/llm/server.py`

**Size**: ~20 lines of cleanup

## Fix 3: Wire `PathFuzzyMatcher` into `check_dd_paths`

**Problem**: `PathFuzzyMatcher` exists in `imas_codex/search/fuzzy_matcher.py`
with `rapidfuzz`-based typo correction but is never called from `check_dd_paths`.
Currently only `RENAMED_TO` edge traversal is used for suggestions.

**Fix**:
1. In `check_dd_paths()`, when a path is not found AND no `RENAMED_TO` edge exists, call `PathFuzzyMatcher.suggest_paths()`
2. Populate `CheckPathsResultItem.suggestion` with the fuzzy match
3. Add IDS existence validation for the `ids` parameter

**Files**: `imas_codex/tools/graph_search.py`, tests

**Size**: ~30 lines + tests

## All Three Fixes Are Independent

```
Fix 1 (truncation)  → independent
Fix 2 (migration)   → independent
Fix 3 (fuzzy)       → independent
```

Can be implemented by 1-3 parallel agents.

## Documentation Updates

| Target | Update |
|--------|--------|
| `AGENTS.md` | Remove any `explain_concept` references if present |
| `plans/README.md` | Update status |
