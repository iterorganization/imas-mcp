---
name: sn/compose_dd
description: Dynamic user prompt for SN composition — per-batch DD paths with enriched context
used_by: imas_codex.sn.workers.compose_worker
task: composition
dynamic: true
schema_needs: []
---

Generate standard names for the following IMAS Data Dictionary paths.

## Batch Context

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
{% if item.documentation and item.documentation != item.description %}- **DD Documentation:** {{ item.documentation }}{% endif %}
- **Unit:** {{ item.unit or 'dimensionless' }} *(authoritative — copy exactly, do NOT substitute or convert)*
- **Data type:** {{ item.data_type or 'unspecified' }}
{% if item.physics_domain %}- **Physics domain:** {{ item.physics_domain }}{% endif %}
{% if item.ndim is not none %}- **Dimensions:** {{ item.ndim }}D{% endif %}
{% if item.keywords %}- **Keywords:** {{ item.keywords | join(', ') if item.keywords is iterable and item.keywords is not string else item.keywords }}{% endif %}
{% if item.cluster_label %}- **Cluster:** {{ item.cluster_label }}{% endif %}
{% if item.cluster_description %}- **Cluster description:** {{ item.cluster_description }}{% endif %}
{% if item.parent_path %}- **Parent structure:** {{ item.parent_path }} ({{ item.parent_type or 'STRUCTURE' }}){% endif %}
{% if item.parent_description %}- **Parent description:** {{ item.parent_description }}{% endif %}
{% if item.coord_path %}- **Coordinate:** {{ item.coord_path }}{% if item.coord_unit %} ({{ item.coord_unit }}){% endif %}{% endif %}
{% if item.cluster_siblings %}- **Cross-IDS siblings:**
{% for sib in item.cluster_siblings[:5] %}  - {{ sib.path }} ({{ sib.unit or '?' }})
{% endfor %}{% endif %}

{% endfor %}
