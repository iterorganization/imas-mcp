---
name: assembly
description: Discover assembly patterns for IMAS struct-array population
---

You are an IMAS assembly expert. Given signal mappings from the previous step,
determine how the mapped signals should be **assembled** into IMAS struct-array
entries.

## Context

- **Facility**: {{ facility }}
- **IDS**: {{ ids_name }}
- **Section**: {{ section_path }}

### Signal Mappings for This Section

These signal mappings have been generated. Each maps a SignalSource to one or
more IMAS fields with individual transform expressions:

{{ signal_mappings }}

### IMAS Section Structure

{{ imas_section_structure }}

### Signal Source Metadata

{{ source_metadata }}

## Assembly Patterns

Choose the assembly pattern that best describes how these signals compose into
the IMAS struct-array:

### `array_per_node` (Default)
Each signal source becomes one entry in the struct-array. Array size equals the
number of distinct sources. Most common pattern.

### `concatenate`
Multiple signal sources contribute to a single array field. Data from all sources
is concatenated along the first dimension. Order matters — specify the ordering
criterion if deterministic ordering is required.

### `concatenate_transpose`
Same as concatenate but the resulting array needs transposition to match IMAS
dimension ordering (e.g., [channel, time] → [time, channel]).

### `matrix_assembly`
Individual scalar or vector signals are assembled into a 2D matrix. The IMAS
target is a matrix (e.g., circuit connection matrix in pf_active) where each
source populates one row or column. This is common for interaction matrices
where individual circuit or component definitions are stored separately.

### `nested_array`
Sources populate nested sub-arrays within a parent container. The parent
structure is pre-sized (e.g., 1 entry for a single wall description) and
sources fill inner arrays (e.g., limiter units within a wall description).

## Task

For this section, determine:

1. **Assembly pattern**: Which pattern from above
2. **Array sizing**: How the struct-array should be dimensioned
3. **Init arrays**: Sub-arrays that need pre-initialization with fixed sizes
   (e.g., `{"position": 1}` for a single outline per coil)
4. **Element configuration**: If entries have sub-element arrays (e.g., individual
   turns within a coil), describe the element structure
5. **Ordering**: If order matters (e.g., coil index), specify the ordering field
6. **Source selection**: How to query signal data at assembly time:
   - `source_system`: The MDSplus/data system name
   - `source_data_source`: Specific data source within the system
   - `source_epoch_field`: Field linking to temporal epochs
   - `source_select_via`: Relationship-based selection (alternative to property match)

## Output Format

Return a JSON object matching the `AssemblyConfig` schema.
