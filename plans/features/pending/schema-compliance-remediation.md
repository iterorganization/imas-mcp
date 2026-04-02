# Schema Compliance Remediation

## Problem

15 test failures in `tests/graph/test_schema_compliance.py` caused by graph data drift — nodes with undeclared properties, null required fields, missing composite constraints, and incomplete property renames. These are NOT code regressions in the test logic; they are stale/malformed data from prior schema evolution and one-off migrations.

## Scope

5 failing tests across 4 node types and 1 constraint issue:

| Test | Failures | Root Cause |
|------|----------|------------|
| `test_no_undeclared_properties` | Image, MappingEvidence, SignalEpoch, SignalSource | Properties on graph nodes not declared in LinkML schema |
| `test_composite_constraints_correct` | SignalSource | Missing `(id, facility_id)` composite constraint |
| `test_required_fields_present` | Image (45,727), MappingEvidence (204×4), SignalSource (2×2) | Null values on required fields |
| `test_identifiers_non_null` | MappingEvidence (204) | Null `id` field |
| `test_identifier_uniqueness` | MappingEvidence (204) | All 204 have `id=None` → duplicates |

## Diagnosis

### Issue 1: Image — incomplete `source_url → url` rename

**Symptom**: 45,727 Image nodes have `source_url` set but `url` is null.

**Root cause**: Commit `76cf21b0` renamed the property in code and schema but the graph migration either didn't execute or was rolled back. All creation code now correctly uses `url`.

**Code status**: ✅ Fixed at source — all creation paths (`persist_images()`, `_extract_and_persist_images()`, `_ingest_image_document()`, `persist_document_figures()`) use `url`.

**Fix**: Graph migration — copy `source_url` → `url`, remove `source_url`. No rediscovery needed (45K images are expensive to re-download/process).

### Issue 2: MappingEvidence — malformed escalation nodes

**Symptom**: 204 nodes with null `id`, null required fields (`evidence_type`, `text`, `supports_mapping`), and undeclared properties (`imas_path`, `reason`, `severity`, `source_id`, `type`).

**Root cause**: `persist_mapping_result()` in `imas_codex/ids/models.py:520-538` creates MappingEvidence nodes via MERGE on `{source_id, imas_path, type}` — none of which are the schema `id` field. The function writes `source_id`, `imas_path`, `type`, `severity`, `reason` instead of the declared `id`, `evidence_type`, `text`, `supports_mapping`.

**Code status**: ❌ NOT fixed — `ids/models.py:520-538` still writes wrong properties and doesn't generate an `id`.

**Fix**: Two-part:
1. **Code fix**: Rewrite `persist_mapping_result()` escalation persistence to generate proper `id` (e.g., `f"{source_id}:{target_id}:escalation"`), map to schema properties (`evidence_type='escalation'`, `text=reason`, `supports_mapping=false`, etc.)
2. **Data fix**: Delete all 204 malformed nodes — they'll be recreated correctly on next mapping run

### Issue 3: SignalSource — undeclared mapping pipeline properties

**Symptom**: 299 nodes with undeclared `mapping_status`, `mapping_target_ids`, `mapping_target_path`, `mapping_target_type`. Also `mapping_claimed_at` and `mapping_claim_token` used for worker coordination.

**Root cause**: `imas_codex/ids/workers.py` (lines 140-266) implements a mapping pipeline that writes these properties directly to SignalSource nodes, but they were never added to the LinkML schema.

**Code status**: ❌ NOT fixed — schema is missing 6 property declarations that the code actively writes.

**Fix**: Add all 6 properties to the `SignalSource` class in `facility.yaml`:
- `mapping_status` (range: MappingStatus enum with values: assigned, mapped, validated)
- `mapping_target_ids` (string — target IDS name)
- `mapping_target_path` (string — full IMAS path)
- `mapping_target_type` (string — IMAS node type)
- `mapping_claimed_at` (datetime — worker coordination)
- `mapping_claim_token` (string — atomic claim UUID)

### Issue 4: SignalSource — 2 orphaned nodes with null required fields

**Symptom**: `jet:pf_coils:group1` and `jet:pf_coils:group2` have null `group_key`, `status`, and `facility_id`.

**Root cause**: Incomplete node creation from a failed/interrupted grouping operation. The creation code (`discovery/base/grouping.py:54-62`) always sets these fields, so these are orphans.

**Code status**: ✅ Fixed at source — `create_signal_source()` always populates required fields.

**Fix**: Delete the 2 orphaned nodes.

### Issue 5: SignalSource — missing composite constraint

**Symptom**: SignalSource has only `(id)` uniqueness constraint, but schema declares `facility_id` as required, triggering `needs_composite_constraint()`.

**Root cause**: Constraint was never created as composite. The `id` already embeds `facility_id` by convention (`"{facility}:{group_key}"`), but the test enforces schema-derived composite constraints.

**Fix**: Drop the `(id)` constraint and recreate as `(id, facility_id)`.

### Issue 6: SignalEpoch — undeclared `error` property

**Symptom**: 369 nodes have `error` property (values: "TREE-E-FOPENR: tree file not found (bulk remediation)" × 336, "No status set (bulk remediation)" × 33).

**Root cause**: One-off bulk remediation script set `error` on SignalEpoch nodes, but the property was never added to the schema. These are annotation artifacts, not functional data.

**Code status**: ✅ No code writes this property — it was a manual migration.

**Fix**: Remove the `error` property from all 369 SignalEpoch nodes.

## Implementation Plan

Three independent work streams, parallelizable:

### Agent 1: Schema declarations (Issues 3, 6)

Add missing properties to LinkML schema and remove stale graph properties.

1. Add 6 `mapping_*` properties to `SignalSource` in `facility.yaml` (with a new `MappingPipelineStatus` enum: `assigned`, `mapped`, `validated`)
2. Remove `error` property from 369 SignalEpoch nodes (inline Cypher via `graph shell`)
3. Run `uv run build-models --force`
4. Commit schema changes

### Agent 2: Code fix + data cleanup (Issues 2, 4)

Fix the MappingEvidence creation bug and clean up orphaned nodes.

1. Rewrite `persist_mapping_result()` escalation block in `ids/models.py:520-538`:
   - Generate deterministic `id`: `f"{source_id}:{target_id}:escalation"`
   - Map properties: `evidence_type='escalation'`, `text=reason`, `supports_mapping=False`, `confidence=0.0` (escalations are uncertain by definition)
   - Preserve `source_id` as `source_node_id` (declared in schema)
   - Preserve `imas_path` as a new optional `target_path` slot (add to schema) or use `url` field
2. Delete all 204 malformed MappingEvidence nodes (inline Cypher)
3. Delete 2 orphaned SignalSource nodes (`jet:pf_coils:group1`, `jet:pf_coils:group2`)
4. Commit code + data fixes

### Agent 3: Graph migrations (Issues 1, 5)

Run graph data migrations and fix constraints.

1. Rename Image `source_url` → `url` on all 45,727 nodes (batched LIMIT 5000 loop)
2. Remove old `source_url` property after rename
3. Drop SignalSource `(id)` constraint, recreate as `(id, facility_id)` composite
4. Verify with `SHOW CONSTRAINTS`

### Post-merge validation

After all three agents merge:

```bash
uv run pytest tests/graph/test_schema_compliance.py -v
```

All 9 tests should pass with 0 failures.

## Notes

- Image nodes should NOT be rediscovered (45K images are expensive). A property rename is the correct fix.
- MappingEvidence nodes SHOULD be deleted and will be recreated correctly on the next `imas-codex map run`.
- The 2 orphaned SignalSource nodes are safe to delete — they have no relationships or meaningful data.
- The SignalEpoch `error` property is annotation-only from a one-off remediation and carries no functional value.
- Agent 1 and Agent 3 have no code overlap. Agent 2 touches `ids/models.py` and `facility.yaml` (overlaps with Agent 1 on schema). Sequence: Agent 1 first (schema), then Agent 2 (code fix + schema addition), Agent 3 in parallel with either.
