---
name: assembly
description: Per-call context for assembly pattern discovery (dynamic, changes per call)
---

Discover the assembly pattern for the following section.

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

### Identifier Schemas

These fields have enumerated valid values. Use these exact values when
populating identifier/type fields in assembly code:

{{ identifier_schemas }}

### Coordinate Specifications

Coordinate axes and dimensionality for array fields in this section.
Use this to determine array sizing and dimension ordering:

{{ coordinate_context }}
