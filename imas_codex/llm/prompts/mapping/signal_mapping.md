---
name: signal_mapping
description: Per-source context for signal-level mapping (dynamic, changes per call)
---

Generate signal-level mappings for the following source and context.

## Context

- **Facility**: {{ facility }}
- **IDS**: {{ ids_name }}
- **Section**: {{ section_path }}

### COCOS Sign-Flip Paths

These IMAS paths require sign flips for COCOS convention conversion:

{{ cocos_paths }}

### Source COCOS Context

{{ source_cocos }}

### Signal Source

This signal source has been assigned to the section above:

{{ signal_source_detail }}

### Code References

Code showing how this signal is read and what transforms/units it uses:

{{ code_references }}

### IMAS Section Fields

The target IMAS section has these fields. Each field has a path, data type,
units, and documentation:

{{ imas_fields }}

### Coordinate Specifications

Coordinate axes and dimensionality for array fields in this section.
Use this to match source signal dimensionality to the correct target field
and determine if dimension reordering is needed:

{{ coordinate_context }}

### Identifier Schemas

These target fields have enumerated valid values. Use these exact values
when populating identifier/type fields:

{{ identifier_schemas }}

### Unit Analysis

Unit compatibility analysis between signal and IMAS units:

{{ unit_analysis }}

### Semantic Candidates

Top IMAS paths matched by semantic similarity to this signal source.
Use these to identify likely target fields, but verify against the
section fields above:

{{ semantic_candidates }}

### Existing Mappings

Any existing mappings for this facility/IDS:

{{ existing_mappings }}

### Version History

Notable changes to target fields across DD versions. Check whether your
target DD version is before or after these changes:

{{ version_context }}

### IMAS Cluster Candidates

Some semantic candidates below are cluster members — IMAS paths that store
the same physical parameter in different IDSs. When a source maps to one
member of a cluster, evaluate whether it should also map to other members.

Cluster members from different IDSs (e.g., `core_profiles/.../ip` and
`equilibrium/.../ip`) are valid one-to-many mappings if the source signal
genuinely represents that quantity. Set appropriate confidence — the primary
IDS target (within the current `{{ ids_name }}`) should have higher confidence
than cross-IDS targets.

{{ cluster_candidates }}

{% if wiki_context %}
### Domain Documentation

Wiki documentation relevant to this physics domain (filtered by
IMAS relevance score):

{{ wiki_context }}
{% endif %}

{% if code_data_access %}
### Data Access Code Patterns

Code examples showing how similar signals are accessed at this facility
(filtered by data_access score):

{{ code_data_access }}
{% endif %}

{% if semantic_match_matrix %}
### Semantic Match Matrix

Cross-index cosine similarity matches for this source across IMAS fields,
wiki documentation, and code. Higher scores indicate stronger semantic
alignment. Use IMAS matches as primary mapping candidates and wiki/code
matches as supporting evidence for mapping decisions.

{{ semantic_match_matrix }}
{% endif %}
