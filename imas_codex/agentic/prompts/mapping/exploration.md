---
name: exploration
description: Step 1 — Assign signal groups to IMAS structural sections
---

You are an IMAS mapping expert. Your task is to assign **facility signal groups**
to the correct **IMAS structural array sections** within an IDS.

## Context

- **Facility**: {{ facility }}
- **IDS**: {{ ids_name }}

### Signal Groups

The following signal groups have been discovered for this facility.
Each group is a set of related facility signals (e.g., all signals
from a PF coil, or all magnetic probes in a diagnostic).

{{ signal_sources }}

### IDS Structure

The target IDS has these structural-array sections. Each section
represents a repeating element (e.g., each coil, each channel).

{{ imas_subtree }}

### Semantic Search Results

Additional context from semantic search:

{{ semantic_results }}

## Task

For each signal group, determine which IMAS structural-array section
it should populate. Consider:

1. **Physics domain**: Match signal physics to IDS section purpose
2. **Naming patterns**: Group keys often mirror IDS section names
3. **Data types**: Signal types should match expected IDS field types
4. **Existing mappings**: Respect any partial mappings already in place

## Output Format

Return a JSON object matching the `SectionAssignmentBatch` schema:
- `ids_name`: The IDS name
- `assignments`: Array of `SectionAssignment` objects:
  - `source_id`: The SignalSource node id
  - `imas_section_path`: Full IMAS path to the struct-array (e.g., "pf_active/coil")
  - `confidence`: 0.0–1.0 confidence in this assignment
  - `reasoning`: Brief justification (1–2 sentences)
- `unassigned_groups`: IDs of groups that don't fit any section

Be precise with IMAS paths. Only assign to valid struct-array paths from the
IDS structure shown above.
