# DD Change Detection & Migration Propagation

## Context

A critical gap analysis (research session 2026-04-03) revealed that the DD 3.42.0 → 4.1.0
`pf_active/circuit/connections` structural change (76→38 columns, 2-sides-per-element binary
format → signed-single-column format) is **invisible** to agents using the migration guide
MCP tool. The imas-python library already implements correct converters
(`_circuit_connections_3to4`/`_circuit_connections_4to3`), proving the DD maintainers
classify this as a breaking semantic change — our pipeline simply fails to surface it.

The root causes are four independent gaps that combine to create a complete blind spot:

1. **`maxoccur` not compared** — extracted and stored but never diffed between versions
2. **Breaking level bug** — all 80 `sign_convention` doc changes have `breaking_level=informational`
   instead of `breaking` due to a data-flow bug in `compute_version_changes()`
3. **Migration guide omits documentation changes** — only queries 5 change types + COCOS
4. **No semantic model change type** — changes where path/type are preserved but data
   format/interpretation changes have no dedicated representation

**Constraint:** Another agent is currently rebuilding the IMAS-DD graph. All graph-write
operations (backfills, schema migrations) must wait until that build completes. Code changes
and tests can proceed in parallel.

---

## Phase 1 — Fix Breaking Level Classification Bug

**Goal:** Ensure `sign_convention` documentation changes are classified as `breaking` and
`coordinate_convention` as `advisory`. This is the root cause of the `connections` change
being invisible.

### 1a. Fix semantic_type flow in `compute_version_changes()`

**Problem:** `compute_version_changes()` computes `breaking_level` for documentation
changes *before* semantic classification. The `change_entry` dict passed to
`_classify_breaking_level()` lacks a `semantic_type` key, so the function defaults
to `"none"` → returns `"informational"`. Later in `_batch_create_path_changes()`,
the pre-computed (wrong) value wins over the fallback that would use the correct
semantic_type.

Data flow today:
```
compute_version_changes():
  change_entry = {"field": "documentation", "old_value": ..., "new_value": ...}
  # NO semantic_type key!
  _classify_breaking_level("documentation", change_entry)
  # → reads change.get("semantic_type", "none") → "none" → "informational"

_batch_create_path_changes():
  semantic_type, keywords = classify_doc_change(old, new)  # CORRECT semantic_type
  change_data["semantic_type"] = semantic_type              # stored correctly
  change_data["breaking_level"] = change.get("breaking_level")  # "informational" from above!
  # fallback _classify_breaking_level() never reached
```

**Implementation:**

In `compute_version_changes()`, run `classify_doc_change()` for documentation fields
and include the result in `change_entry` BEFORE calling `_classify_breaking_level()`:

```python
# build_dd.py, inside the else branch at ~line 1417
else:
    if str(old_val) != str(new_val):
        change_entry = {
            "field": field,
            "old_value": str(old_val) if old_val else "",
            "new_value": str(new_val) if new_val else "",
        }
        # Classify doc changes semantically BEFORE computing breaking level
        if field == "documentation":
            semantic_type, keywords = classify_doc_change(
                change_entry["old_value"], change_entry["new_value"]
            )
            change_entry["semantic_type"] = semantic_type
            change_entry["keywords_detected"] = keywords
        change_entry["breaking_level"] = _classify_breaking_level(
            field, change_entry
        )
        changes.append(change_entry)
```

This ensures `_classify_breaking_level()` sees the correct `semantic_type` and returns
`"breaking"` for `sign_convention` changes.

**Files:**
- `imas_codex/graph/build_dd.py:1417-1427` — add semantic classification before breaking level

**Test cases:**
```python
# tests/graph/test_dd_build.py — add to TestComputeVersionChanges

def test_sign_convention_doc_change_is_breaking(self):
    """Documentation changes introducing sign convention keywords must be breaking."""
    from imas_codex.graph.build_dd import compute_version_changes

    old = {"ids/connections": {"documentation": "Matrix elements are 1 or 0."}}
    new = {"ids/connections": {"documentation": "Matrix elements are 1 if positive side, -1 if negative side, or 0."}}
    result = compute_version_changes(old, new)
    assert "ids/connections" in result["changed"]
    doc_changes = [c for c in result["changed"]["ids/connections"] if c["field"] == "documentation"]
    assert len(doc_changes) == 1
    assert doc_changes[0]["breaking_level"] == "breaking"
    assert doc_changes[0]["semantic_type"] == "sign_convention"

def test_coordinate_convention_doc_change_is_advisory(self):
    """Documentation changes introducing coordinate convention keywords must be advisory."""
    from imas_codex.graph.build_dd import compute_version_changes

    old = {"ids/phi": {"documentation": "Toroidal angle"}}
    new = {"ids/phi": {"documentation": "Toroidal angle (right-handed coordinate system)"}}
    result = compute_version_changes(old, new)
    doc_changes = [c for c in result["changed"]["ids/phi"] if c["field"] == "documentation"]
    assert doc_changes[0]["breaking_level"] == "advisory"
    assert doc_changes[0]["semantic_type"] == "coordinate_convention"

def test_plain_doc_change_is_informational(self):
    """Documentation changes without convention keywords remain informational."""
    from imas_codex.graph.build_dd import compute_version_changes

    old = {"ids/a": {"documentation": "Temperature"}}
    new = {"ids/a": {"documentation": "Temperature of the electrons"}}
    result = compute_version_changes(old, new)
    doc_changes = [c for c in result["changed"]["ids/a"] if c["field"] == "documentation"]
    assert doc_changes[0]["breaking_level"] == "informational"
```

### 1b. Remove duplicate semantic classification in `_batch_create_path_changes()`

**Problem:** After the fix above, `compute_version_changes()` produces `change_entry` dicts
that already contain `semantic_type` and `keywords_detected` for documentation changes.
But `_batch_create_path_changes()` at line 3241-3248 also calls `classify_doc_change()`
redundantly. This should use the pre-computed values (like it does for `breaking_level`).

**Implementation:**

```python
# build_dd.py ~line 3240-3248, replace:
if change["field"] == "documentation":
    semantic_type, keywords = classify_doc_change(...)
    change_data["semantic_type"] = semantic_type
    ...

# with:
if change["field"] == "documentation":
    # Use pre-computed semantic classification from compute_version_changes()
    change_data["semantic_type"] = change.get("semantic_type")
    kw = change.get("keywords_detected")
    if kw and not isinstance(kw, str):
        change_data["keywords_detected"] = json.dumps(kw)
    elif kw:
        change_data["keywords_detected"] = kw
    # Fallback for legacy callers that don't pre-compute
    if change_data["semantic_type"] is None:
        semantic_type, keywords = classify_doc_change(
            change.get("old_value", ""), change.get("new_value", "")
        )
        change_data["semantic_type"] = semantic_type
        if keywords:
            change_data["keywords_detected"] = json.dumps(keywords)
```

**Files:**
- `imas_codex/graph/build_dd.py:3240-3248` — use pre-computed semantic classification

---

## Phase 2 — Add `maxoccur` to Version Comparison

**Goal:** Detect array dimension changes between DD versions so that structural changes
like the `connections` column halving are tracked as `IMASNodeChange` nodes.

Independent of Phase 1. Can be implemented in parallel.

### 2a. Add `maxoccur` to comparison field list

**Implementation:**

```python
# build_dd.py:1387-1396, add "maxoccur" to the field tuple:
for field in (
    "units",
    "documentation",
    "data_type",
    "node_type",
    "cocos_label_transformation",
    "lifecycle_status",
    "coordinates",
    "timebasepath",
    "maxoccur",
):
```

The `maxoccur` field needs special handling similar to `units` — the field name in the
comparison loop doesn't match the `ChangeType` enum value (`maxoccur_changed`). Add a
field-name-to-change-type mapping:

```python
# In _batch_create_path_changes(), around line 3232:
FIELD_TO_CHANGE_TYPE = {
    "maxoccur": "maxoccur_changed",
}
# ...
change_data["change_type"] = FIELD_TO_CHANGE_TYPE.get(change["field"], change["field"])
```

**Files:**
- `imas_codex/graph/build_dd.py:1387-1396` — add `maxoccur` to field list
- `imas_codex/graph/build_dd.py:3232` — add field→change_type mapping

**Test cases:**
```python
# tests/graph/test_dd_build.py — add to TestComputeVersionChanges

def test_maxoccur_change_detected(self):
    """Changes in maxoccur should be detected."""
    from imas_codex.graph.build_dd import compute_version_changes

    old = {"ids/circuit": {"maxoccur": 76, "documentation": "Circuit matrix"}}
    new = {"ids/circuit": {"maxoccur": 38, "documentation": "Circuit matrix"}}
    result = compute_version_changes(old, new)
    assert "ids/circuit" in result["changed"]
    maxoccur_changes = [c for c in result["changed"]["ids/circuit"] if c["field"] == "maxoccur"]
    assert len(maxoccur_changes) == 1
    assert maxoccur_changes[0]["old_value"] == "76"
    assert maxoccur_changes[0]["new_value"] == "38"
    assert maxoccur_changes[0]["breaking_level"] == "advisory"

def test_maxoccur_none_to_value_not_change(self):
    """maxoccur going from None to a value on a new path is not a change."""
    from imas_codex.graph.build_dd import compute_version_changes

    old = {"ids/a": {"maxoccur": None}}
    new = {"ids/a": {"maxoccur": 10}}
    result = compute_version_changes(old, new)
    # None→10 should be detected (it's a real metadata change)
    assert "ids/a" in result["changed"]
```

### 2b. Handle `maxoccur` None→value and value→None edge cases

**Problem:** `maxoccur` can be `None` (unbounded) or an integer. The generic `str(old_val) != str(new_val)` comparison works but `str(None)` = `"None"` which is correct. However, we should suppress noise: if both are `None`, or both are `0`, skip.

**Implementation:**

Add a `maxoccur`-specific comparison branch similar to the `units` branch:

```python
if field == "maxoccur":
    old_mo = old_info.get("maxoccur")
    new_mo = new_info.get("maxoccur")
    if old_mo == new_mo:
        continue
    # Both None means no change
    if old_mo is None and new_mo is None:
        continue
    change_entry = {
        "field": "maxoccur",
        "old_value": str(old_mo) if old_mo is not None else "unbounded",
        "new_value": str(new_mo) if new_mo is not None else "unbounded",
    }
    change_entry["breaking_level"] = _classify_breaking_level("maxoccur_changed", change_entry)
    changes.append(change_entry)
    continue
```

**Files:**
- `imas_codex/graph/build_dd.py` — add maxoccur comparison branch in `compute_version_changes()`

---

## Phase 3 — Surface Semantic Documentation Changes in Migration Guide

**Goal:** Make documentation changes with `sign_convention` and `coordinate_convention`
semantic types visible in migration guide output. This is what makes the `connections`
change visible to agents.

Depends on Phase 1 (correct breaking_level). Can start code in parallel but needs
Phase 1 for correct data.

### 3a. Add `_get_semantic_doc_changes()` query function

**Implementation:**

```python
# migration_guide.py — new function after _get_additions()

def _get_semantic_doc_changes(
    gc: GraphClient, version_range: list[str], ids_filter: str | None = None
) -> list[dict]:
    """Get documentation changes with physics significance (sign/coordinate conventions)."""
    ids_clause = ""
    params: dict = {"versions": version_range}
    if ids_filter:
        ids_clause = "AND p.ids = $ids_filter"
        params["ids_filter"] = ids_filter
    return gc.query(
        f"""
        MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
        WHERE v.id IN $versions
          AND c.change_type = 'documentation'
          AND c.semantic_type IN ['sign_convention', 'coordinate_convention']
        MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
        WHERE true {ids_clause}
        RETURN p.ids AS ids, p.id AS path,
               c.semantic_type AS semantic_type,
               c.old_value AS old_doc, c.new_value AS new_doc,
               coalesce(c.breaking_level, 'informational') AS level,
               c.keywords_detected AS keywords,
               v.id AS version
        ORDER BY c.breaking_level DESC, c.semantic_type, p.ids, p.id
        """,
        **params,
    )
```

**Files:**
- `imas_codex/tools/migration_guide.py` — add new query function

### 3b. Integrate semantic doc changes into `build_migration_guide()`

**Implementation:**

In `build_migration_guide()` after the type changes section (~line 697), add:

```python
# --- Semantic documentation changes (sign/coordinate conventions) ---
semantic_doc_changes = _get_semantic_doc_changes(gc, version_range, ids_filter)
for sdc in semantic_doc_changes:
    path = sdc.get("path", "")
    ids_name = sdc.get("ids", path.split("/")[0] if "/" in path else "")
    ids_affected.add(ids_name)
    patterns = generate_search_patterns(path, "definition_change")
    search_patterns.setdefault(ids_name, []).extend(patterns)
    level = sdc.get("level", "informational")
    semantic = sdc.get("semantic_type", "")

    desc = f"Convention change ({semantic}): {path}"
    old_excerpt = (sdc.get("old_doc") or "")[:200]
    new_excerpt = (sdc.get("new_doc") or "")[:200]

    action = CodeUpdateAction(
        path=path,
        ids=ids_name,
        change_type="definition_change",
        severity="required" if level == "breaking" else "optional",
        search_patterns=patterns,
        path_fragments=path.split("/")[1:],
        description=desc,
        before=old_excerpt,
        after=new_excerpt,
    )
    if level == "breaking":
        required_actions.append(action)
    else:
        optional_actions.append(action)
```

**Files:**
- `imas_codex/tools/migration_guide.py:461-717` — add semantic doc changes section

### 3c. Add `definition_change` to `CodeUpdateAction` and rendering

**Implementation:**

1. Add `definition_change` to the `change_type` field description in `CodeUpdateAction`[^1]
2. Add a "Convention Changes" section to `format_migration_guide()`:

```python
# In format_migration_guide(), after the type changes section:
convention_actions = [a for a in guide.required_actions + guide.optional_actions
                      if a.change_type == "definition_change"]
if convention_actions:
    lines.append("## Convention Changes")
    lines.append("")
    lines.append("These changes affect data interpretation without changing path names or types.")
    lines.append("Code that reads these fields may produce **silently incorrect results**")
    lines.append("if not updated.")
    lines.append("")
    for action in convention_actions:
        severity_badge = "**BREAKING**" if action.severity == "required" else "advisory"
        lines.append(f"### `{action.path}` ({severity_badge})")
        lines.append("")
        lines.append(f"  {action.description}")
        if action.before:
            lines.append(f"  - **Before:** {action.before}")
        if action.after:
            lines.append(f"  - **After:** {action.after}")
        if action.search_patterns:
            patterns_str = ", ".join(f"`{p}`" for p in action.search_patterns[:3])
            lines.append(f"  - **Search for:** {patterns_str}")
        lines.append("")
```

**Files:**
- `imas_codex/models/migration_models.py` — update `change_type` description
- `imas_codex/tools/migration_guide.py` — add Convention Changes rendering section

### 3d. Add tests for migration guide convention changes

**Test cases:**
```python
# tests/tools/test_migration_guide.py (or appropriate test file)

def test_migration_guide_includes_sign_convention_changes(self, graph_client):
    """Migration guide should surface documentation changes with sign_convention semantic."""
    from imas_codex.tools.migration_guide import build_migration_guide

    guide = build_migration_guide(graph_client, "3.42.0", "4.1.0")
    definition_actions = [a for a in guide.required_actions
                         if a.change_type == "definition_change"]
    # The pf_active/circuit/connections sign_convention change should be present
    connections_actions = [a for a in definition_actions
                          if "connections" in a.path]
    assert len(connections_actions) >= 1, (
        "pf_active/circuit/connections sign_convention change must appear "
        "as a required action in migration guide"
    )

def test_migration_guide_format_has_convention_section(self, graph_client):
    """Formatted migration guide should have a Convention Changes section."""
    from imas_codex.tools.migration_guide import generate_migration_guide

    output = generate_migration_guide(graph_client, "3.42.0", "4.1.0")
    assert "Convention Changes" in output
```

---

## Phase 4 — Graph Data Backfill

**Goal:** Fix the 80 existing `sign_convention` documentation changes that have
`breaking_level=informational` in the graph, and any `coordinate_convention` changes
that should be `advisory`.

**IMPORTANT:** Must wait until the current DD graph rebuild completes. If the rebuild
uses the fixed code from Phase 1, this phase is unnecessary — the rebuild will
produce correct breaking levels. Check after rebuild completes.

### 4a. Verify post-rebuild state

After the current DD rebuild completes, check if the breaking levels are correct:

```cypher
MATCH (c:IMASNodeChange)
WHERE c.change_type = 'documentation' AND c.semantic_type = 'sign_convention'
RETURN c.breaking_level AS level, count(c) AS cnt
```

If all `sign_convention` changes have `breaking_level=breaking`, Phase 4b is unnecessary.

### 4b. Backfill breaking_level (only if needed)

If the rebuild used old code, run these two targeted updates:

```cypher
-- Fix sign_convention → breaking
MATCH (c:IMASNodeChange)
WHERE c.change_type = 'documentation'
  AND c.semantic_type = 'sign_convention'
  AND c.breaking_level <> 'breaking'
SET c.breaking_level = 'breaking'
RETURN count(c) AS updated

-- Fix coordinate_convention → advisory  
MATCH (c:IMASNodeChange)
WHERE c.change_type = 'documentation'
  AND c.semantic_type = 'coordinate_convention'
  AND c.breaking_level <> 'advisory'
SET c.breaking_level = 'advisory'
RETURN count(c) AS updated
```

**Files:** No code changes — graph shell or REPL only.

---

## Phase 5 — Add `convention_change` Schema Type (Optional Enhancement)

**Goal:** Introduce a first-class `ChangeType` for semantic model changes where path
name and data type are preserved but data format/interpretation changes fundamentally.
This gives the `connections` change (and similar future cases) a proper identity beyond
"documentation change with sign_convention keywords."

Independent of Phases 1-4 but benefits from them. Lower priority.

### 5a. Extend `ChangeType` enum in schema

**Implementation:**

```yaml
# imas_codex/schemas/imas_dd.yaml — add to ChangeType enum:
convention_change:
  description: >-
    Data format or interpretation convention changed without type/path change.
    The field retains its name and data type but values have different meaning
    across versions. Requires non-trivial data transformation during conversion.
    Examples: matrix column encoding (binary sides → signed single column),
    index convention changes, normalization changes.
```

Add a corresponding `SemanticChangeType`:

```yaml
# Add to SemanticChangeType enum:
data_format_change:
  description: >-
    The numeric encoding or dimensionality convention changed. The field name
    and data type are preserved but values have fundamentally different meaning
    across versions. Requires data transformation (not just sign flip).
```

**Files:**
- `imas_codex/schemas/imas_dd.yaml` — extend enums
- Run `uv run build-models --force` to regenerate

### 5b. Add convention_change detection heuristic

**Problem:** Convention changes are difficult to detect automatically because the
data_type and path don't change. The only signals are:

1. Documentation text changes describing different numeric encoding
2. Keywords: "dimension", "column", "format", "encoding", "index", "polarity"
3. Significant documentation length changes on array fields

**Implementation:**

Extend `classify_doc_change()` with additional keywords and a new category:

```python
# build_dd.py — extend keyword lists:
DATA_FORMAT_KEYWORDS = [
    "dimension",
    "second dimension",
    "first dimension",
    "column",
    "matrix elements",
    "encoding",
    "polarity",
    "format",
    "sides",
    "index convention",
]
```

And in `classify_doc_change()`, add a check after sign_convention and coordinate_convention:

```python
# After coordinate_convention check:
format_keywords = []
for kw in DATA_FORMAT_KEYWORDS:
    if kw in new_lower and kw not in old_lower:
        format_keywords.append(kw)
if format_keywords:
    return "data_format_change", format_keywords
```

This is heuristic and may need tuning. A conservative approach: only classify as
`data_format_change` when BOTH sign_convention keywords AND format keywords are detected
together (the `connections` case has both "negative" and "dimension" keywords).

**Files:**
- `imas_codex/graph/build_dd.py:80-135` — extend keyword lists and classification
- `imas_codex/graph/build_dd.py:1295-1327` — add breaking level rule for `convention_change`

### 5c. Update migration guide for convention_change type

Add `convention_change` handling to `_get_semantic_doc_changes()` and the migration
guide rendering. This extends the work in Phase 3 to also include the new change type.

**Files:**
- `imas_codex/tools/migration_guide.py` — query and render `convention_change` changes

---

## Phase 6 — Expand Comparison Coverage

**Goal:** Add remaining uncompared fields to version diffing for completeness.
Lowest priority — these are less critical than the sign convention and maxoccur gaps.

Independent of all other phases.

### 6a. Add `ndim` to comparison fields

**Problem:** If a field changes dimensionality (e.g., scalar → 1D array) the `data_type`
change detects `FLT_0D → FLT_1D`, so `ndim` is partially redundant. However, explicit
`ndim` tracking provides a cleaner signal for array shape changes.

**Implementation:** Add `"ndim"` to the comparison loop with a field-to-change-type mapping
of `"ndim" → "structure_changed"`.

**Files:**
- `imas_codex/graph/build_dd.py:1387-1396`

### 6b. Add `identifier_enum_name` to comparison fields

**Problem:** When a path's identifier schema changes (e.g., switching from one enum to
another), the semantic meaning of valid values changes.

**Implementation:** Add `"identifier_enum_name"` to comparison loop, mapping to a new
change type or `"structure_changed"`.

**Files:**
- `imas_codex/graph/build_dd.py:1387-1396`

---

## Parallel Execution Map

```
Phase 1 (breaking level bug)  ─────────┐
  1a: Fix semantic_type flow            │
  1b: Remove duplicate classification   │
                                        ├──▶ Phase 4 (backfill, if needed)
Phase 2 (maxoccur comparison)  ─────────┤
  2a: Add maxoccur to field list        │
  2b: Handle None edge cases            │
                                        │
Phase 3 (migration guide)  ────────────▶┤ (needs Phase 1 for correct data)
  3a: Add query function                │
  3b: Integrate into builder            │
  3c: Add rendering section             │
  3d: Add tests                         │
                                        │
Phase 5 (convention_change type)  ──────┘ (optional, enhances Phase 3)
  5a: Extend schema
  5b: Detection heuristic
  5c: Migration guide update

Phase 6 (expanded coverage)  ───────────── (fully independent)
  6a: ndim
  6b: identifier_enum_name
```

**Safe parallelism:**
- Phase 1 + Phase 2 + Phase 6: fully independent, can run simultaneously
- Phase 3: code can be written in parallel with Phase 1, but integration tests need Phase 1 merged first
- Phase 4: must wait for DD rebuild to complete, then check if needed
- Phase 5: depends on Phase 3 for migration guide structure, otherwise independent

---

## Documentation Updates

When all phases are complete, update:

| Target | What to update |
|--------|----------------|
| `AGENTS.md` | Add `convention_change` to ChangeType documentation if Phase 5 is done |
| `plans/README.md` | Add this plan, track completion |
| `docs/architecture/imas_dd.md` | Update change detection section with new coverage |
| `agents/schema-reference.md` | Auto-regenerated by `uv run build-models` |
| `tests/graph/test_dd_build.py` | New test class `TestSemanticDocChangeClassification` |
| `tests/tools/` | New migration guide convention change tests |
