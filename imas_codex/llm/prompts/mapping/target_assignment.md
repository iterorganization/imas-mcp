---
name: target_assignment
description: Per-call context for target assignment (dynamic, changes per call)
---

Assign the following signal sources to IDS target paths.

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

The target IDS has these structural paths. Each path includes its node type
(structure, data, etc.) to help classify the target type. Paths with
array-of-structures semantics are struct-array targets; leaf data paths
are scalar targets; time-indexed containers are time-slice targets.

{{ imas_subtree }}

### Semantic Search Results

Additional context from semantic search:

{{ semantic_results }}

### Section Clusters

These semantic clusters group related IDS paths by physics concept.
Use these to understand which paths form physics-coherent groups:

{{ section_clusters }}

{% if cross_facility_mappings %}
### Cross-Facility Precedent

Other facilities have already mapped signals to these IDS paths.
Use this as strong evidence for where similar signals should be assigned:

{{ cross_facility_mappings }}
{% endif %}
