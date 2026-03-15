---
name: section_assignment
description: Per-call context for section assignment (dynamic, changes per call)
---

Assign the following signal sources to IMAS sections.

## Context

- **Facility**: {{ facility }}
- **IDS**: {{ ids_name }}

{% if ids_description %}
### IDS Description

{{ ids_description }}
{% endif %}

### Signal Sources

The following signal sources have been discovered for this facility.
Each source is a set of related facility signals (e.g., all signals
from a PF coil, or all magnetic probes in a diagnostic). The physics
domain indicates the source's primary measurement category.

{{ signal_sources }}

### IDS Structure

The target IDS has these structural-array sections. Each section
represents a repeating element (e.g., each coil, each channel).

{{ imas_subtree }}

### Semantic Search Results

Additional context from semantic search:

{{ semantic_results }}

### Section Clusters

These semantic clusters group related IDS sections by physics concept.
Use these to understand which paths form physics-coherent groups:

{{ section_clusters }}

{% if cross_facility_mappings %}
### Cross-Facility Precedent

Other facilities have already mapped signals to these IDS sections.
Use this as strong evidence for where similar signals should be assigned:

{{ cross_facility_mappings }}
{% endif %}
