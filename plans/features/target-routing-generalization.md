# Target Routing Generalization: Beyond Struct-Arrays

**Status**: Plan
**Created**: 2026-03-15
**Scope**: `imas_codex/ids/models.py`, `imas_codex/ids/mapping.py`, `imas_codex/ids/assembler.py`, `imas_codex/llm/prompts/mapping/`

## Motivation

The mapping pipeline currently assumes all signal-to-IMAS assignments land on
**struct-array sections** (`pf_active/coil`, `magnetics/flux_loop`, etc.). This
assumption is baked into model docstrings, field descriptions, prompt language,
assembly patterns, and the persisted graph structure.

This works for IDS types dominated by repeating elements (e.g., `pf_active` where
every signal source maps to a coil in the `coil[:]` array). But many IDS types
contain **non-array targets** that the pipeline cannot currently handle:

### Concrete examples of what fails today

| IDS | Target path | Structure | Why it breaks |
|-----|-------------|-----------|---------------|
| `magnetics` | `magnetics/ip/0/data` | Scalar time-dependent | `ip` is an `AoS1` with typically one element — the source is a single Ip measurement, not a repeating array of Ip coils |
| `equilibrium` | `equilibrium/time_slice/0/global_quantities/ip` | Scalar within time-slice | No struct-array section to assign; the signal maps to a leaf field inside a time_slice entry |
| `equilibrium` | `equilibrium/time_slice/0/profiles_1d/psi` | 1D profile within time-slice | Similar — the entry point is the profile, not a repeating element |
| `core_profiles` | `core_profiles/profiles_1d/0/electrons/density` | Profile data | Signals map to specific physics quantities within a single profile, not to an array of "electron" objects |
| `summary` | `summary/global_quantities/ip/value` | Pure scalar | No array structure at all; direct value assignment |

### Root cause

**Step 1 (assign_sections)** forces every signal source into a `SectionAssignment`
with `imas_section_path` described as "IMAS struct-array path". When a signal
belongs to a non-array path, the LLM must either:

1. Force-fit it to the nearest struct-array ancestor (losing precision)
2. Emit it as `unassigned` (losing the mapping entirely)
3. Hallucinate a struct-array path that doesn't exist

None of these are correct. The step conflates two distinct concerns:
- **Routing**: Which subtree of the IDS does this signal belong to?
- **Assembly structure**: How should the data be written (array element, scalar, profile)?

## Current State

### Models (`imas_codex/ids/models.py`)

```python
class SectionAssignment(BaseModel):
    """Map a signal source to an IMAS structural array section."""
    imas_section_path: str = Field(
        description="IMAS struct-array path (e.g. pf_active/coil)"
    )

class SignalMappingBatch(BaseModel):
    section_path: str = Field(description="IMAS struct-array section")

class AssemblyPattern(StrEnum):
    ARRAY_PER_NODE = "array_per_node"
    CONCATENATE = "concatenate"
    CONCATENATE_TRANSPOSE = "concatenate_transpose"
    MATRIX_ASSEMBLY = "matrix_assembly"
    NESTED_ARRAY = "nested_array"
    # No scalar/direct pattern exists

class AssemblyConfig(BaseModel):
    """Assembly configuration for one IMAS struct-array section."""
```

All docstrings, field descriptions, and enum values assume struct-array targets.

### Prompts (`imas_codex/llm/prompts/mapping/`)

- `section_assignment_system.md`: "assign facility signal sources to the correct
  IMAS structural array sections"
- `section_assignment.md`: "Each section represents a repeating element (e.g.,
  each coil, each channel)"
- `assembly_system.md`: All patterns describe array composition
- `signal_mapping_system.md`: "signal-level mappings from facility signal sources
  to specific IMAS fields within a structural section" — actually closest to
  correct since it talks about "fields within a section"

### Assembler (`imas_codex/ids/assembler.py`)

```python
def _build_graph_section(self, ids, section, epoch_id, mapping, gc):
    """Build one struct-array section from graph data."""
    struct_array = getattr(ids, section_name)
    struct_array.resize(len(flat_nodes))
    for i, node_data in enumerate(flat_nodes):
        entry = struct_array[i]
```

Hard-assumes every section is a resizable array. No code path for scalar writes.

### Pipeline (`imas_codex/ids/mapping.py`)

`map_signals()` iterates over `sections.assignments` and calls `fetch_imas_fields()`
scoped to each `assignment.imas_section_path`. For scalar targets, the subtree fetch
would still work (it returns fields under any path), but the prompt framing tells
the LLM to map within a "section" context.

`discover_assembly()` generates an `AssemblyConfig` per section. For scalar targets,
none of the existing `AssemblyPattern` values apply — a simple `set_nested()` call
is needed, not array population.

### Graph persistence (`persist_mapping_result`)

Creates `POPULATES` relationships from `IMASMapping` to `IMASNode` via
`section.imas_section_path`. This is structurally fine for any path depth
(struct-array or scalar), but the assembly config on the relationship
(structure, init_arrays, etc.) only makes sense for arrays.

## Implementation Plan

### Phase 1: Generalize the routing vocabulary

**Goal**: Replace "section assignment" language with "target routing" throughout.
The step's purpose is routing signals to IDS subtrees — not specifically to
struct-arrays.

**Changes**:

1. **Rename `SectionAssignment` → `TargetAssignment`**
   - Rename `imas_section_path` → `imas_target_path`
   - Update description: "IMAS subtree path — struct-array (e.g. pf_active/coil),
     time-slice container (e.g. equilibrium/time_slice), or scalar path
     (e.g. summary/global_quantities/ip)"
   - Add `target_type: TargetType` field (see Phase 2) — but default to
     `"auto"` initially so Phase 1 is backward-compatible

2. **Rename `SectionAssignmentBatch` → `TargetAssignmentBatch`**

3. **Rename `UnassignedSource`** — keep as-is, the name is already general

4. **Update `assign_sections()` → `assign_targets()`** in `mapping.py`
   - Function name, log messages, step_name

5. **Update prompts**:
   - `section_assignment_system.md`: Replace "structural array sections" with
     "IDS target paths" — include examples of struct-array, time-slice, and
     scalar targets
   - `section_assignment.md`: Update "Each section represents a repeating
     element" to explain the full range of target types

6. **Update `SignalMappingBatch.section_path`** → `target_path` with generalized
   description

7. **Update all references in `mapping.py`**: `assignment.imas_section_path` →
   `assignment.imas_target_path`, log messages, step names

8. **Update `ValidatedMappingResult.sections`** type annotation: `list[TargetAssignment]`

9. **Update `persist_mapping_result`**: Use `assignment.imas_target_path` instead
   of `section.imas_section_path`

### Phase 2: Add target type classification

**Goal**: Let the routing LLM classify what kind of IDS target it's assigning to,
so downstream steps (mapping, assembly) can adapt their behavior.

**Changes**:

1. **Add `TargetType` enum**:
   ```python
   class TargetType(StrEnum):
       STRUCT_ARRAY = "struct_array"      # e.g., pf_active/coil[:]
       TIME_SLICE = "time_slice"          # e.g., equilibrium/time_slice[:]
       SCALAR = "scalar"                  # e.g., summary/global_quantities/ip
       PROFILE = "profile"               # e.g., core_profiles/profiles_1d[0]/electrons
   ```

2. **Add `target_type` field to `TargetAssignment`** (default `STRUCT_ARRAY` for
   backward compat)

3. **Update system prompt** to instruct the LLM to classify target type based
   on IDS structure context (struct-array nodes have `node_type: "structure"`
   with array semantics in the DD)

4. **Provide the IDS tree context to include node types** — `fetch_imas_subtree()`
   already returns `node_type` per row. The prompt template under
   "### IDS Structure" should render `node_type` alongside each path so the
   LLM can distinguish struct-arrays (AoS) from scalars.

### Phase 3: Extend assembly patterns

**Goal**: Handle non-array assembly in `discover_assembly()` and the assembler.

**Changes**:

1. **Add `DIRECT` to `AssemblyPattern`**:
   ```python
   class AssemblyPattern(StrEnum):
       DIRECT = "direct"                          # Scalar write, no array manipulation
       ARRAY_PER_NODE = "array_per_node"           # Existing
       CONCATENATE = "concatenate"                 # Existing
       CONCATENATE_TRANSPOSE = "concatenate_transpose"
       MATRIX_ASSEMBLY = "matrix_assembly"
       NESTED_ARRAY = "nested_array"
   ```

2. **Update `assembly_system.md` prompt** to include the `direct` pattern:
   ```
   ### `direct`
   Signal source maps directly to a scalar or fixed-position field. No array
   resizing needed — value is written via set_nested(). Use this when the
   target is a leaf field or a specific element within a pre-existing structure
   (e.g., time_slice[0]/global_quantities/ip).
   ```

3. **Update `AssemblyConfig`** docstring: remove "struct-array" assumption

4. **Update `discover_assembly()`** — when `target_type` is `SCALAR` or `PROFILE`,
   the assembly step can potentially be skipped (auto-emit a `DIRECT` config)
   rather than calling the LLM. This saves an LLM call for trivial cases.

5. **Update `IDSAssembler._build_graph_section()`** — add `elif structure == "direct":`
   handler:
   ```python
   elif structure == "direct":
       # Scalar/leaf write: apply mappings directly to IDS root
       for node_data in flat_nodes:
           self._apply_mappings(ids, node_data, section_mappings, section_name, section_config)
   ```

### Phase 4: Pipeline optimization

**Goal**: Reduce unnecessary LLM calls for simple target types.

**Changes**:

1. **Skip `discover_assembly()` for `DIRECT` targets** — when `target_type` is
   `SCALAR`, auto-generate a `DIRECT` `AssemblyConfig` without an LLM call.
   Only call the LLM for struct-array and complex patterns.

2. **Adapt `map_signals()` context** — for scalar targets, the prompt should
   emphasize that the signal maps to a specific leaf path, not to "fields
   within a section". The context framing changes but the mapping model
   (`SignalMappingEntry`) stays the same.

3. **Test with IDS types that exercise all target types**:
   - `pf_active` → `STRUCT_ARRAY` (existing)
   - `magnetics` → mix of `STRUCT_ARRAY` (flux_loop, bpol_probe) and `SCALAR` (ip)
   - `equilibrium` → `TIME_SLICE` with nested profiles
   - `summary` → predominantly `SCALAR`

## Migration

No data migration required. The graph schema for `IMASMapping` and `POPULATES`
relationships is unaffected — the `imas_section_path` stored as `root_path`
on the POPULATES relationship works for any path depth.

Existing `IMASMapping` nodes with `POPULATES` relationships to struct-array
paths will continue to work since the assembler already dispatches on the
`structure` field of the POPULATES relationship.

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| LLM misclassifies target type | Medium | Phase 2 provides node_type context from DD; validation can reject impossible classifications |
| Rename breaks downstream consumers | Low | No external consumers of these internal models; all access is within `imas_codex/ids/` |
| Assembly code generation fails for DIRECT pattern | Low | DIRECT is simpler than array patterns — less can go wrong |
| Prompt changes degrade struct-array mapping quality | Medium | Run regression on pf_active/magnetics before+after; keep struct-array examples in prompt |

## Dependencies

- None — this is a self-contained refactoring of the mapping pipeline
- Complementary to `staged-mapping-pipeline.md` (data→error→metadata staging)
- The reasoning model tier (`get_model("reasoning")`) applies to `map_signals()`
  regardless of target type

## Estimated Scope

| Phase | Files touched | Complexity |
|-------|---------------|------------|
| Phase 1 (vocabulary) | models.py, mapping.py, 2 prompts | Mechanical rename, low risk |
| Phase 2 (target type) | models.py, mapping.py, 1 prompt, tools.py | Medium — requires prompt engineering for type classification |
| Phase 3 (assembly) | models.py, assembler.py, 1 prompt | Medium — new code path in assembler |
| Phase 4 (optimization) | mapping.py | Low — conditional skip logic |
