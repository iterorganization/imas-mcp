---
name: sn/compose_dd
description: Dynamic user prompt for SN composition — per-batch DD paths
used_by: imas_codex.sn.workers.compose_worker
task: composition
dynamic: true
schema_needs: []
---

Generate standard names for the following IMAS Data Dictionary paths.

## Context

**IDS:** {{ ids_name }}
{% if cluster_context %}
{{ cluster_context }}
{% endif %}

{% if existing_names %}
## Existing Standard Names (reuse when applicable)

These names already exist. **Reuse** them when the DD path measures the same
quantity — do not create a duplicate with different wording.

{% for name in existing_names %}
- {{ name }}
{% endfor %}
{% endif %}

## DD Paths to Name

{% for item in items %}
### {{ item.path }}
- **Description:** {{ item.description }}
- **Units:** {{ item.units or 'unspecified' }}
- **Data type:** {{ item.data_type or 'unspecified' }}
{% if item.cluster_label %}- **Cluster:** {{ item.cluster_label }}{% endif %}

{% endfor %}
