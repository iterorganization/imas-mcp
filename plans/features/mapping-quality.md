# Mapping Quality: Naming, Status, Validation, and Coverage

Status: **Planning**
Priority: High
Supersedes: None (complements `imas-mapping-combined.md`)

## Summary

The IMAS mapping pipeline produces output, but the output language is vague, the
status lifecycle is dishonest, validation is LLM self-review (not real checks),
field naming leaks internal implementation details, and coverage reporting is absent.
This plan addresses all of these issues as a cohesive set of changes.

## Issues Identified

### 1. Status Lifecycle: `validated` is Dishonest (Currently)

`persist_mapping_result()` unconditionally sets `status = 'validated'`, but no
actual validation occurs — the LLM reviews its own output in Step 3. Meanwhile
the assembler's `load_mapping()` filters on `status = 'active'`, creating a dead
gap: the pipeline produces mappings in `validated` status that the assembler
refuses to load because it wants `active`.

Current enum (`IMASMappingStatus`):
```
draft → validated → active → deprecated
```

Problems:
- `validated` is set without real validation (signal loading, transform execution,
  shape checking)
- Nothing transitions from `validated` to `active`
- `draft` is never used
- The implied human-review step between `validated` and `active` will never happen

### 2. Field Naming: Vague, Redundant, Implementation-Leaked

The `map show` output uses names that conflate internal graph concepts with domain
semantics:

| Current Name | Problem |
|---|---|
| `field_mappings` (container) | "field" is vague; "mapping" is overloaded with IMASMapping node |
| `signal_group_id` | Leaks the graph implementation; this is the source identifier |
| `group_key` | Redundant — always `signal_group_id` minus facility prefix |
| `imas_path` | Ambiguous — could be any IMAS path; doesn't parallel `source_id` |
| `transform_code` | "code" implies executable code; `"value"` is not code |
| `units_in` / `units_out` | Direction unclear without context |

### 3. Validation Is LLM Self-Review

Step 3 of the pipeline asks the LLM to review its own Step 2 output. This is
prompt engineering, not validation. Real validation means:
- Can the source signal be loaded from the graph?
- Does the transform expression execute without error?
- Do input/output shapes match what the assembler expects?
- Are there duplicate target paths (two sources → same IMAS field)?
- Do structural sections have data nodes available for assembly?

### 4. No Coverage Reporting

The pipeline produces 9 bindings for `pf_active` but the DD has ~55 leaf paths
under `pf_active/coil` alone. There's no indication of how much of the target IDS
is mapped, what's missing, or what can't be mapped.

### 5. Duplicate Target Paths Not Detected

In the JET pf_active output, both `pfcircuits/NNN/coil_connect` and
`pfcircuits/NNN/supply_connect` map to the same target `pf_active/circuit/description`.
This is a data conflict that should be flagged as an escalation.

### 6. All Transform Expressions Are Identity (`"value"`)

Every binding in the JET pf_active output has `transform_code: "value"`. For static
geometry (r, z, dr, dz in meters) this is correct. But `turnsperelement` mapping to
`coil/element/turns_with_sign` arguably needs sign handling, and the pipeline doesn't
flag this gap.

The transform execution engine exists (`ids/transforms.py` —
`execute_transform()` with math, numpy, `convert_units`, `cocos_sign`) but the LLM
isn't generating non-trivial transforms because Step 2 doesn't emphasize it.

### 7. Pipeline Is Not Exhaustive

Step 0 loads all `SignalGroup` nodes for the facility without IDS filtering (correct),
but Step 1 relies on the LLM to assign groups to sections (heuristic). There's no
systematic check of "which signals COULD map to this IDS" — only "which groups did
the LLM choose to assign." Unassigned groups vanish silently.

---

## Plan

### Phase 1: Rename Field Labels

Rename the output fields in `search_existing_mappings()` and all consuming code.
These are display/API names, not schema property names — the graph schema stores
these as relationship properties on `MAPS_TO_IMAS` and doesn't need changes.

| Current | New | Rationale |
|---|---|---|
| `field_mappings` (container) | `bindings` | A binding ties a source to a target. Clear, not overloaded. |
| `signal_group_id` | `source_id` | What provides the data. Parallels target_id. |
| `group_key` | *(removed)* | Redundant — derivable from source_id minus facility prefix |
| `imas_path` | `target_id` | What receives the data. Parallels source_id. |
| `transform_code` | `transform_expression` | It's a Python expression, not a code block |
| `units_in` | `source_units` | Clear direction |
| `units_out` | `target_units` | Clear direction |
| `source_property` | `source_property` | Keep — already clear |

#### Files to Change

| File | Changes |
|---|---|
| `imas_codex/ids/tools.py` | `search_existing_mappings()` RETURN aliases |
| `imas_codex/ids/models.py` | `FieldMappingEntry`, `ValidatedFieldMapping` field names |
| `imas_codex/ids/graph_ops.py` | `FieldMapping` dataclass, `load_field_mappings()` RETURN aliases |
| `imas_codex/ids/assembler.py` | References to `target_imas_path`, `transform_code`, `units_in/out` |
| `imas_codex/ids/transforms.py` | `execute_transform()` parameter name |
| `imas_codex/ids/mapping.py` | Prompt formatting, step references |
| `imas_codex/cli/map.py` | Display output |
| `imas_codex/schemas/facility.yaml` | `SignalGroup.maps_to_imas` slot description |
| Tests | All mapping-related test files |

#### Property Names on MAPS_TO_IMAS Relationship

The graph relationship properties also need renaming for consistency:

| Current Property | New Property |
|---|---|
| `transform_code` | `transform_expression` |
| `units_in` | `source_units` |
| `units_out` | `target_units` |
| `source_property` | `source_property` (keep) |
| `cocos_label` | `cocos_label` (keep) |
| `confidence` | `confidence` (keep) |

This requires a graph migration for any existing data. Add migration Cypher to
the rename script.

### Phase 2: Fix Status Lifecycle

Replace the current `IMASMappingStatus` enum with a lifecycle that matches reality:

```
generated → validated → active → deprecated
```

| Status | Meaning | Set By |
|---|---|---|
| `generated` | LLM pipeline produced output, no programmatic checks run | `persist_mapping_result()` after Step 3 |
| `validated` | Programmatic checks passed: signals exist, transforms execute, no shape errors, no duplicate targets | `validate_mapping()` (new function, Phase 3) |
| `active` | Promoted for use by the assembler | `imas-codex imas map activate` CLI command |
| `deprecated` | Superseded by a newer mapping | `imas-codex imas map clear` or new mapping generation |

#### Why Keep `active` Separate from `validated`?

Validation checks that bindings are **technically correct** (signal exists, transform
runs, types match). Promotion to `active` is a deliberate decision that this mapping
should be used for real IDS assembly — it gates deployment, not correctness.

`active` could be set automatically after validation succeeds (zero-friction default),
with a `--no-activate` flag for when you want to inspect first. This means no human
review is required, but you still have a gate.

#### Schema Change

```yaml
IMASMappingStatus:
  description: Status lifecycle for IMAS mapping orchestration
  permissible_values:
    generated:
      description: LLM pipeline produced field bindings, not yet verified
    validated:
      description: Programmatic checks passed — signals exist, transforms execute, shapes match
    active:
      description: Validated and promoted for use by the assembler
    deprecated:
      description: Superseded by a newer version
```

Remove `draft` — it's never used. If manual mapping authoring is needed in future,
use `generated` with a different `provider` string.

#### Code Changes

| File | Change |
|---|---|
| `imas_codex/schemas/facility.yaml` | Replace `IMASMappingStatus` enum values |
| `imas_codex/ids/models.py` | `persist_mapping_result()`: set `status = 'generated'` |
| `imas_codex/ids/graph_ops.py` | `load_mapping()`: already filters `status = 'active'` ✓ |
| `imas_codex/cli/map.py` | Add `activate` subcommand |
| `imas_codex/ids/mapping.py` | Pipeline sets `generated`, then runs validation, then sets `validated` or optionally `active` |

### Phase 3: Real Validation Step

Add a programmatic validation function that earns the `validated` status. This
replaces the LLM self-review (Step 3) with real checks.

#### `validate_mapping(facility, ids_name) → ValidationReport`

```python
@dataclass
class BindingCheck:
    source_id: str
    target_id: str
    source_exists: bool         # SignalGroup node exists in graph
    target_exists: bool         # IMASNode exists in graph
    transform_executes: bool    # execute_transform(sample_value, expr) succeeds
    units_compatible: bool      # pint can convert source_units → target_units
    shape_compatible: bool      # source dimensionality matches target ndim
    error: str | None = None    # Details if any check failed

@dataclass
class ValidationReport:
    mapping_id: str
    all_passed: bool
    binding_checks: list[BindingCheck]
    duplicate_targets: list[str]           # target_ids mapped by multiple sources
    coverage: CoverageReport
    escalations: list[EscalationFlag]      # From LLM + validation-generated
```

#### Checks Performed

1. **Source exists**: `MATCH (sg:SignalGroup {id: $source_id})` returns a node.
2. **Target exists**: Use existing `check_imas_paths()`.
3. **Transform executes**: Call `execute_transform(1.0, expr)` with a sentinel
   value. Catches syntax errors, missing functions, runtime exceptions.
4. **Units compatible**: Call existing `analyze_units(source_units, target_units)`.
   If incompatible AND no transform_expression handles conversion, fail.
5. **Shape compatible**: Compare source signal dimensionality (from SignalGroup
   member metadata) against target `ndim` (from IMASNode). Scalar→scalar OK,
   array→array OK, scalar→array escalation.
6. **Duplicate target detection**: Group bindings by `target_id`. Any target_id
   with >1 source_id is a conflict — escalate.
7. **Coverage report**: See Phase 4.

#### Integration into Pipeline

The pipeline becomes 4 steps:

```
Step 0: Gather context (programmatic)
Step 1: Assign groups to IMAS sections (LLM)
Step 2: Generate field bindings per section (LLM)
Step 3: Validate bindings (programmatic — replaces LLM self-review)
```

Step 3 is pure Python, no LLM call. Cost drops. Reliability goes up. The pipeline
sets `status = 'generated'` after Step 2 persists, then runs validation. If all
checks pass, status advances to `validated`. If `--activate` is set (default),
`validated` immediately advances to `active`.

#### Assembly-Time Validation

The assembler (`IDSAssembler._apply_mappings`) already catches `AttributeError`
and `TypeError` silently. These should be promoted to validation-time checks
instead of assembly-time surprises. The `validate_mapping()` function does a
dry-run of path resolution against the DD schema without requiring actual data.

### Phase 4: Coverage Reporting

After validation, report what percentage of the target IDS is covered:

```python
@dataclass
class CoverageReport:
    ids_name: str
    total_leaf_fields: int          # All data-bearing fields under the IDS
    mapped_fields: int              # Fields with at least one binding
    unmapped_required: list[str]    # Required fields with no binding
    unmapped_optional: list[str]    # Optional fields with no binding
    percentage: float               # mapped / total * 100
```

#### How It Works

1. Query all `IMASNode` leaf fields under the IDS:
   ```cypher
   MATCH (p:IMASNode)
   WHERE p.ids = $ids_name
   AND NOT p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']
   RETURN p.id AS id, p.data_type AS data_type, p.node_type AS node_type
   ```

2. Filter out deprecated paths (via `DEPRECATED_IN` relationship).

3. Compare against the set of `target_id` values in the mapping's bindings.

4. Report coverage percentage and list unmapped fields.

#### CLI Output

```
imas-codex imas map run jet pf_active

Mapping: jet:pf_active
Bindings: 9
Status: active

Coverage (pf_active/coil): 8/55 leaf fields (14.5%)
  Mapped: geometry/rectangle/r, geometry/rectangle/z, geometry/rectangle/width,
          geometry/rectangle/height, element/turns_with_sign, name
  Unmapped: current/data, current/time, resistance, ...

Coverage (pf_active/circuit): 1/12 leaf fields (8.3%)
  Mapped: description
  Unmapped: connection, type, ...

Escalations (1):
  DUPLICATE_TARGET: pf_active/circuit/description ← 2 sources
    pfcircuits/NNN/coil_connect, pfcircuits/NNN/supply_connect

Cost: $0.0042 (1,234 tokens)
```

### Phase 5: Exhaustive Signal Scanning

Currently, Step 0 loads all `SignalGroup` nodes and Step 1 asks the LLM to assign
them. Unassigned groups vanish. Fix this by making the graph the state ledger:

#### After Step 1: Persist Assignment State

For every signal group considered, record whether it was assigned or not:

```cypher
// Assigned groups: link to mapping
MATCH (m:IMASMapping {id: $mapping_id})
MATCH (sg:SignalGroup {id: $sg_id})
MERGE (m)-[:USES_SIGNAL_GROUP]->(sg)

// Unassigned groups: persist as explicit gaps
// Store in the LLM response as unassigned_groups
```

The `SectionAssignmentBatch.unassigned_groups` field already exists in the model
but is never surfaced in the CLI output or validation report. Surface it.

#### Coverage = Graph Query

After mapping runs, coverage is a pure graph query:

```cypher
// What fraction of enriched signal groups for this facility have IMAS bindings?
MATCH (sg:SignalGroup {facility_id: $facility, status: 'enriched'})
OPTIONAL MATCH (sg)-[:MAPS_TO_IMAS]->(ip:IMASNode)
RETURN sg.id, ip IS NOT NULL AS is_mapped
```

This doesn't require re-running the pipeline — it's the graph as state ledger.

### Phase 6: LLM Prompt Improvements for Transforms

The current prompts don't generate meaningful transform expressions because:
1. The field_mapping prompt doesn't emphasize that `transform_expression` should
   be a real Python expression
2. Unit conversion examples aren't shown
3. COCOS sign-flip paths are injected but not connected to the transform

#### Prompt Changes

In `imas_codex/llm/prompts/mapping/field_mapping.md`:

1. Add explicit examples of transform expressions:
   ```
   - Identity: "value" (source and target units match, no sign flip)
   - Unit conversion: "value * 1e-3" (eV → keV)
   - COCOS sign flip: "-value" (when COCOS conventions differ)
   - Angle conversion: "math.radians(value)" (degrees → radians)
   - Function call: "convert_units(value, 'mm', 'm')" (arbitrary unit conversion)
   ```

2. Connect COCOS paths to transform requirements:
   ```
   These IMAS paths require COCOS sign handling:
   {{ cocos_paths }}
   For each binding targeting one of these paths, the transform_expression
   MUST include sign handling (e.g., "-value" or "cocos_sign('ip_like', cocos_in=2, cocos_out=11)").
   ```

3. Require non-trivial transforms when units differ:
   ```
   If source_units ≠ target_units, the transform_expression MUST perform
   the conversion. Use convert_units(value, source_units, target_units)
   or an equivalent expression.
   Do NOT set transform_expression to "value" when units differ.
   ```

### Phase 7: Remove `group_key` from Display and Queries

`group_key` is always `signal_group_id` minus the facility prefix. It exists on
the `SignalGroup` node as a property (useful for intra-facility uniqueness), but
returning it alongside `signal_group_id` in the `map show` output is redundant.

#### Changes

1. Remove `group_key` from `search_existing_mappings()` RETURN clause
2. Remove `group_key` from `map show` JSON output
3. Keep `group_key` on the `SignalGroup` schema/node — it's useful for group
   creation and pattern detection, just not for mapping display

---

## Implementation Sequence

All phases are independently testable but should be implemented in order because
Phase 2 (status) and Phase 3 (validation) change the pipeline flow that Phase 1
(naming) and Phase 4–7 depend on.

```
Phase 1: Rename field labels (pure renaming, no logic changes)
Phase 2: Fix status lifecycle (schema + persist function)
Phase 3: Real validation step (new function, pipeline restructure)
Phase 4: Coverage reporting (new function, CLI formatting)
Phase 5: Exhaustive scanning (unassigned group tracking)
Phase 6: LLM prompt improvements (template edits)
Phase 7: Remove group_key redundancy (cleanup)
```

### Migration

Existing `jet:pf_active` mapping data in the graph needs property renames on
`MAPS_TO_IMAS` relationships:

```cypher
// Rename relationship properties
MATCH ()-[r:MAPS_TO_IMAS]->()
WHERE r.transform_code IS NOT NULL
SET r.transform_expression = r.transform_code
REMOVE r.transform_code

MATCH ()-[r:MAPS_TO_IMAS]->()
WHERE r.units_in IS NOT NULL
SET r.source_units = r.units_in, r.target_units = r.units_out
REMOVE r.units_in, r.units_out

// Update mapping status
MATCH (m:IMASMapping)
WHERE m.status = 'validated'
SET m.status = 'generated'
```

---

## Files

### Create

| File | Phase | Purpose |
|------|-------|---------|
| `imas_codex/ids/validation.py` | 3 | `validate_mapping()`, `ValidationReport`, `BindingCheck`, `CoverageReport` |

### Modify

| File | Phase | Changes |
|------|-------|---------|
| `imas_codex/schemas/facility.yaml` | 1, 2 | `IMASMappingStatus` enum values, `MAPS_TO_IMAS` property descriptions |
| `imas_codex/ids/models.py` | 1, 2 | Field renames on `FieldMappingEntry`, `ValidatedFieldMapping`; `persist_mapping_result()` status |
| `imas_codex/ids/tools.py` | 1, 7 | `search_existing_mappings()` RETURN aliases; remove `group_key` |
| `imas_codex/ids/graph_ops.py` | 1 | `FieldMapping` dataclass field renames, `load_field_mappings()` aliases |
| `imas_codex/ids/assembler.py` | 1 | References to renamed fields |
| `imas_codex/ids/transforms.py` | 1 | Parameter rename `transform_code` → `transform_expression` |
| `imas_codex/ids/mapping.py` | 2, 3, 5 | Pipeline flow: generate → validate → activate; remove LLM Step 3; surface unassigned groups |
| `imas_codex/cli/map.py` | 1, 2, 4 | Output field names; `activate` command; coverage display |
| `imas_codex/llm/prompts/mapping/field_mapping.md` | 6 | Transform expression examples, COCOS connection, unit conversion requirements |
| `imas_codex/llm/prompts/mapping/validation.md` | 3 | Remove or repurpose (no longer an LLM step) |
| Tests | 1–7 | Update assertions to use new field names and status values |

### Remove

| File/Code | Phase | Reason |
|-----------|-------|--------|
| LLM Step 3 (`_step3_validate`) in `mapping.py` | 3 | Replaced by programmatic `validate_mapping()` |
| `validation.md` prompt template | 3 | No longer used (or repurpose for error correction prompt) |

---

## Open Questions

1. **Auto-activate**: Default behavior — if validation passes, immediately set
   `active`? Or require explicit `imas-codex imas map activate jet pf_active`?
   Recommendation: auto-activate by default, `--no-activate` flag to defer.

2. **Transform expression safety**: `execute_transform()` uses `eval()` with
   restricted builtins. The current trust model (authored by agents/humans, not
   external input) is documented. Should validation also check that the expression
   only uses allowed names? Low priority — the restricted builtins already block
   dangerous operations.

3. **Coverage thresholds**: Should there be a minimum coverage percentage to reach
   `validated`? Or is low coverage an escalation, not a blocker? Recommendation:
   low coverage is an escalation, not a validation failure. Some IDS sections
   genuinely have limited available source data.

4. **Re-validation**: When signal discovery adds new groups, should existing
   mappings be re-validated? Recommendation: yes — `imas-codex imas map validate`
   re-runs validation and reports new coverage opportunities.
