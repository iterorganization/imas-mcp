---
name: sn/compose_dd
description: Dynamic user prompt for SN composition — per-batch DD paths with enriched context
used_by: imas_codex.sn.workers.compose_worker
task: composition
dynamic: true
schema_needs: []
---

Generate standard names for the following IMAS Data Dictionary paths.

## Unit Policy

The `unit` field for each path is pre-populated from the IMAS Data Dictionary
(`HAS_UNIT` relationship). It is **authoritative and final**:
- Do NOT include unit in your output — it will be injected from the DD at persistence time
- Use the provided unit to inform your naming (e.g., "eV" tells you this is an energy/temperature quantity)
- "dimensionless" means the quantity is genuinely unitless (e.g., safety factor, elongation)

## Common Anti-Patterns (AVOID these)

| ❌ Wrong | ✅ Correct | Why |
|----------|-----------|-----|
| `electron_temp` | `electron_temperature` | No abbreviations in `physical_base` |
| `langmuir_probe_electron_temperature` | `electron_temperature` | Method independence — measurement device is metadata, not name |
| `filtered_electron_density` | `electron_density` | Processing is metadata — `filtered_` is not a name segment |
| `core_electron_temperature` | `electron_temperature_core` | Position goes after physical_base per grammar |
| `Te` | `electron_temperature` | No symbol abbreviations |
| `electron_temperature_in_eV` | `electron_temperature` | Unit is never part of the name |
| `safety_factor_q` | `safety_factor` | No symbol suffixes |
| `plasma_current_IP` | `plasma_current` | No symbol suffixes |

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
- **Unit:** {{ item.unit or 'dimensionless' }} *(authoritative from DD — use for naming context only, do NOT output)*
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
{% endfor %}
- **Concept identity:** These {{ item.cluster_siblings|length + 1 }} cross-IDS paths represent the SAME physics concept. Generate ONE name that covers all of them.
{% endif %}

{% endfor %}

## Vocabulary Gaps

If a path requires a token that does **not** exist in a closed grammar segment
(e.g., a new `subject` species, a new `position`), do NOT invent an invalid name.
Instead, add the path to the `vocab_gaps` list in your response with:
- `source_id`: the DD path
- `segment`: which grammar segment is missing a token
- `needed_token`: the token value you would need
- `reason`: why this token is needed
