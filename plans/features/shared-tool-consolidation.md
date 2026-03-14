# Plan: Shared Tool Consolidation

**Goal:** Eliminate the last standalone Cypher in the mapping pipeline by
enhancing `GraphListTool.list_imas_paths()` to return full metadata, then
clean up stale prompt references and add the optional cross-facility
mapping precedent enrichment.

**Priority:** Medium — all items are polish. The pipeline is fully operational.

---

## Phase 1: Enhance `GraphListTool` to Return Metadata

### Problem

`fetch_imas_subtree()` in `ids/tools.py` (line 47) maintains its own Cypher
because `GraphListTool.list_imas_paths()` returns only path IDs — no
`data_type`, `units`, `documentation`, `name`, or `node_type`.

The mapping pipeline needs this metadata in three places (`mapping.py`):
1. `gather_context()` line 395 — full IDS subtree for section assignment
2. `map_signals()` line 537 — leaf fields under a section
3. `discover_assembly()` line 708 — section structure for assembly

All three feed into `_format_subtree()` which consumes: `id`, `data_type`,
`units`, `documentation`.

### Implementation

**File:** `imas_codex/tools/graph_search.py` → `GraphListTool.list_imas_paths()`

Add a `response_profile` parameter (matching the pattern on `search_imas_paths`):

```python
async def list_imas_paths(
    self,
    paths: str,
    format: str = "yaml",
    leaf_only: bool = False,
    include_ids_prefix: bool = True,
    max_paths: int | None = None,
    dd_version: int | None = None,
    response_profile: str = "minimal",   # NEW
    ctx: Context | None = None,
) -> ListPathsResult:
```

When `response_profile="standard"`, the Cypher query joins `HAS_UNIT` and
returns `name`, `data_type`, `node_type`, `documentation`, `units` per path —
the same fields `fetch_imas_subtree` currently returns.

Change the two Cypher queries (IDS-match and prefix-match) from:

```cypher
RETURN p.id AS id
```

to (when `response_profile != "minimal"`):

```cypher
OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
RETURN p.id AS id, p.name AS name, p.data_type AS data_type,
       p.node_type AS node_type, p.documentation AS documentation,
       u.symbol AS units
```

**File:** `imas_codex/models/result_models.py` → `ListPathsResultItem`

Add an optional `path_details` field:

```python
class ListPathsResultItem(BaseModel):
    query: str
    path_count: int
    truncated_to: int | None = None
    paths: str | dict[str, Any] | list[str] | None = None
    path_details: list[dict[str, Any]] | None = None  # NEW
    error: str | None = None
```

When `response_profile="standard"`, `path_details` is populated with the
full metadata dicts and `paths` is still populated for backward compatibility.

**File:** `ListPathsResult`

Add `as_dicts()` method (matching the pattern on `FetchPathsResult` and
`SearchPathsResult`):

```python
def as_dicts(self) -> list[dict[str, Any]]:
    """Return flattened path details for pipeline consumption."""
    all_details: list[dict[str, Any]] = []
    for item in self.results:
        if item.path_details:
            all_details.extend(item.path_details)
    return all_details
```

### Consolidation

**File:** `imas_codex/ids/tools.py`

Replace `fetch_imas_subtree()` with delegation to `GraphListTool`:

```python
def fetch_imas_subtree(
    ids_name: str,
    path: str | None = None,
    *,
    gc: GraphClient | None = None,
    leaf_only: bool = False,
    max_paths: int | None = None,
    dd_version: int | None = None,
) -> list[dict[str, Any]]:
    """Return IDS tree structure with full metadata.

    Delegates to GraphListTool.list_imas_paths(response_profile="standard").
    """
    if gc is None:
        gc = GraphClient()

    from imas_codex.tools.graph_search import GraphListTool

    query = f"{ids_name}/{path}" if path else ids_name
    tool = GraphListTool(gc)
    result = _run_async(tool.list_imas_paths(
        paths=query,
        leaf_only=leaf_only,
        max_paths=max_paths,
        dd_version=dd_version,
        response_profile="standard",
    ))
    return result.as_dicts()
```

This eliminates the last standalone Cypher in `ids/tools.py`, making the
entire mapping pipeline consume shared tools for all DD context queries.

### MCP Tool Description Update

Update the `@mcp_tool` docstring to mention the new parameter:

```
"response_profile: 'minimal' (default, path IDs only) or 'standard' (includes data_type, units, documentation). "
```

### Formatter Update

**File:** `imas_codex/llm/search_formatters.py` → `format_list_report()`

When `path_details` is present on result items, format with metadata:

```python
if item.path_details:
    for d in item.path_details:
        parts.append(f"  {d['id']} ({d.get('data_type', '')}) [{d.get('units', '')}]")
elif isinstance(item.paths, list):
    for p in item.paths:
        parts.append(f"  {p}")
```

### Tests

**File:** `tests/tools/test_list_tool.py`

Add test for `response_profile="standard"`:

```python
async def test_list_imas_paths_standard_profile(self, list_tool):
    result = await list_tool.list_imas_paths(
        "equilibrium", response_profile="standard"
    )
    assert result.results[0].path_details is not None
    detail = result.results[0].path_details[0]
    assert "id" in detail
    assert "data_type" in detail
    assert "documentation" in detail
```

Add test for `as_dicts()` on `ListPathsResult`.

---

## Phase 2: Clean Enrichment Prompt Unit References

### Problem

`imas_codex/llm/prompts/signals/enrichment.md` still references `unit` in
prose and example outputs, but `SignalEnrichmentResult` no longer has a
`unit` field. The Pydantic schema injected via `{{ signal_enrichment_schema_fields }}`
and `{{ signal_enrichment_schema_example }}` correctly omits `unit`, but the
hardcoded examples and instructions contradict this.

This wastes prompt tokens (~200 tokens) and can confuse the LLM into
outputting a `unit` field that gets silently discarded.

### Changes

**File:** `imas_codex/llm/prompts/signals/enrichment.md`

1. **Line 110** — Remove `- Wiki units are authoritative — use them for `unit``
   (wiki_units are still useful context for writing descriptions, but the
   instruction to populate a `unit` field is wrong)

2. **Line 135** — Remove `- **Look for units** — code that converts units or
   applies scaling factors reveals physical units` (units discovery is handled
   by the code-based unit propagation pipeline, not LLM enrichment)

3. **Line 150** — Remove `- Extract units when mentioned in wiki documentation`

4. **Lines 425, 439, 456, 471** — Remove `"unit": ""` and `"unit": "eV"` from
   all four example outputs. The Pydantic schema no longer includes `unit`.

### What NOT to Change

- Keep `wiki_units` references in prose where they're used as **context for
  writing descriptions** (e.g., knowing the unit is eV helps describe "electron
  temperature profile"). Just don't instruct the LLM to output a `unit` field.
- Keep the `wiki_units: eV` in the example **input** (line 453) — this is
  context the LLM receives, not output it produces.
- Keep line 119 reference to "units" in the garbled data detection list — this
  is about detecting malformed input, not about outputting units.

---

## Phase 3: Cross-Facility Mapping Precedent (Optional)

### Problem

When mapping signals for facility X to an IDS, the LLM has no visibility
into how other facilities have already mapped similar signals. If TCV has
mapped its plasma current to `equilibrium/time_slice/global_quantities/ip`,
that's strong precedent for JET's mapping.

### Implementation

**File:** `imas_codex/ids/tools.py`

Add `fetch_cross_facility_mappings()` — this stays in `ids/tools.py` because
`IMASMapping` traversal is mapping-pipeline-specific with no MCP consumer value:

```python
def fetch_cross_facility_mappings(
    ids_name: str,
    exclude_facility: str,
    *,
    gc: GraphClient | None = None,
) -> list[dict[str, Any]]:
    """Return active mappings from other facilities to this IDS."""
    if gc is None:
        gc = GraphClient()

    return gc.query(
        """
        MATCH (m:IMASMapping)-[:POPULATES]->(ip:IMASNode)
        WHERE m.ids_name = $ids_name
          AND m.facility_id <> $exclude
          AND m.status IN ['active', 'validated']
        RETURN m.facility_id AS facility,
               ip.id AS target_path,
               m.status AS status
        ORDER BY m.facility_id, ip.id
        """,
        ids_name=ids_name,
        exclude=exclude_facility,
    )
```

**File:** `imas_codex/ids/mapping.py`

Add to `gather_context()`:

```python
cross_mappings = fetch_cross_facility_mappings(ids_name, facility, gc=gc)
```

Add formatter `_format_cross_facility_mappings()`.

**File:** `imas_codex/llm/prompts/mapping/section_assignment.md`

Add optional section:

```markdown
{% if cross_facility_mappings %}
### Cross-Facility Precedent

Other facilities have mapped signals to these IDS sections:

{{ cross_facility_mappings }}
{% endif %}
```

### Risk

Low. This enrichment is additive — if no other facilities have mappings for
the target IDS, the prompt section is omitted via Jinja conditional.

---

## Implementation Order

```
Phase 1 (core)     Enhance GraphListTool, consolidate fetch_imas_subtree
Phase 2 (cleanup)  Clean enrichment prompt unit references
Phase 3 (optional) Cross-facility mapping precedent
```

Phase 1 and Phase 2 are independent. Phase 3 depends on nothing.

## Files Modified

| File | Phase | Changes |
|------|-------|---------|
| `imas_codex/tools/graph_search.py` | 1 | Add `response_profile` to `list_imas_paths`, enrich Cypher |
| `imas_codex/models/result_models.py` | 1 | Add `path_details` to `ListPathsResultItem`, `as_dicts()` to `ListPathsResult` |
| `imas_codex/ids/tools.py` | 1 | Rewrite `fetch_imas_subtree` to delegate to shared tool |
| `imas_codex/llm/search_formatters.py` | 1 | Handle `path_details` in `format_list_report` |
| `tests/tools/test_list_tool.py` | 1 | Test standard profile and `as_dicts()` |
| `imas_codex/llm/prompts/signals/enrichment.md` | 2 | Remove stale unit field references |
| `imas_codex/ids/tools.py` | 3 | Add `fetch_cross_facility_mappings()` |
| `imas_codex/ids/mapping.py` | 3 | Wire cross-facility context into `gather_context()` |
| `imas_codex/llm/prompts/mapping/section_assignment.md` | 3 | Add `cross_facility_mappings` template section |
