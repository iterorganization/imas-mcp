---
name: sn/compose_dd_name_only
description: Lean user prompt for SN composition in name-only batching mode (Workstream 2a)
used_by: imas_codex.standard_names.workers.compose_worker
task: composition
dynamic: true
schema_needs: []
---

Generate standard names for the following IMAS Data Dictionary paths.

This batch was assembled in **name-only mode**: paths share the same
`physics_domain` and authoritative unit but may span many different
semantic clusters and IDSs. **Your first task is to identify the
natural sub-groups within this batch**, then emit one name per path.

## Unit Policy

The `unit` field for each path is pre-populated from the IMAS Data
Dictionary and is **authoritative and final**:

- Do NOT include unit in your output — it is injected at persistence time
- Use the unit to disambiguate physics (e.g., `eV` vs `K` for temperature)
- `dimensionless` means the quantity is genuinely unitless

## Name-Only Mode — Reduced Context

To keep the batch wide and cache-friendly, this prompt intentionally
omits the deep per-item context (cluster siblings, cross-IDS equivalents,
COCOS guidance, version history, reviewer feedback). A subsequent
review / enrichment pass will attach those details. Your job here is
to produce **clean, grammar-compliant names** that correctly
identify the physical quantity — documentation depth can be terse.

### What this means for your output

- **Names** must still be fully grammar-compliant — the grammar
  check runs on every candidate, and failures trigger a retry.
- **Descriptions** should be concise (1–2 sentences) and faithful
  to the DD description. Do not invent detail you don't have.
- **Tags**: one or two tags per name is sufficient; prefer the
  physics domain and the quantity family (e.g., `transport`,
  `particle-flux`).
- **Documentation**: a single paragraph naming the physical
  observable, its units, and its typical usage context is enough.
  The enrichment pass will expand this later.

## Identify Natural Sub-Groups First

Before naming, scan the full path list and group items by the
**physical quantity** they represent (not by IDS). Typical
sub-groupings within a single `(physics_domain, unit)` batch:

- species-based: electron vs ion vs neutral
- orientation: parallel vs perpendicular vs radial vs toroidal
- process: flux vs source vs sink vs diffusivity
- state: volumetric vs surface vs line-integrated

Then emit **one** name per path, reusing the same base name when the
sub-group identity is the same (e.g., `electron_particle_flux_parallel`
and `ion_particle_flux_parallel` share a structure). Reuse existing
standard names from the "Existing Standard Names" list whenever the DD
path measures the same quantity — do not invent a synonym.

## Common Anti-Patterns (AVOID)

| ❌ Wrong | ✅ Correct | Why |
|----------|-----------|-----|
| `electron_temp` | `electron_temperature` | No abbreviations |
| `core_electron_temperature` | `electron_temperature_core` | Position after physical_base |
| `Te` | `electron_temperature` | No symbol abbreviations |
| `electron_temperature_in_eV` | `electron_temperature` | Unit is never part of the name |
| `current_from_passive_loop` | `passive_loop_current` | No `_from_` causation |
| `reconstructed_faraday_rotation_angle` | `faraday_rotation_angle` | Processing method is metadata |
| `geometric_minor_radius` | `minor_radius` | DD section prefix leaking in |

## Batch Consistency Check

Before finalizing, verify across your entire output:

1. **No synonymous names** — same concept = same name
2. **Consistent orientation suffixes** — all `_parallel`, not a mix of `_par`/`_parallel`
3. **No DD leakage** — no name starts with an IDS or DD section prefix

## Batch Context

{{ cluster_context }}

{% if existing_names %}
## Existing Standard Names (reuse when applicable)

These names already exist. **Reuse** them when the DD path measures the
same quantity — do not create a duplicate with different wording.

{% for name in existing_names %}
- {{ name }}
{% endfor %}
{% endif %}

{% if nearby_existing_names %}
## Nearby Existing Standard Names

These names are semantically close to items in this batch. Reuse if
they match; otherwise use them to calibrate style and specificity.

{% for name in nearby_existing_names %}
- **{{ name.id }}**: {{ name.description | default('', true) }} ({{ name.kind | default('scalar', true) }}, {{ name.unit | default('dimensionless', true) }})
{% endfor %}
{% endif %}

## DD Paths to Name

{% for item in items %}
### {{ item.path }}
{% if item.rate_hint %}
> ⚠️ **RATE QUANTITY:** DD documentation indicates a rate / time-derivative.
> Name MUST start with `tendency_of_`, `change_in_`, or `rate_of_change_of_`.
> Description must be consistent with the rate-marker prefix.
> Orientation tokens wrap the rate phrase:
>   ✅ `parallel_component_of_change_in_fast_electron_pressure`
>   ❌ `change_in_parallel_component_of_fast_electron_pressure`
{% endif %}
- **Description:** {{ item.description }}
- **Unit:** {{ item.unit or 'dimensionless' }} *(authoritative — do NOT output)*
{% if item.data_type %}- **Data type:** {{ item.data_type }}{% endif %}
{% if item.physics_domain %}- **Physics domain:** {{ item.physics_domain }}{% endif %}
{% if item.ndim is not none %}- **Dimensions:** {{ item.ndim }}D{% endif %}
{% if item.lifecycle_status %}- **Lifecycle:** {{ item.lifecycle_status }} ⚠️{% endif %}
{% if item.keywords %}- **Keywords:** {{ item.keywords | join(', ') if item.keywords is iterable and item.keywords is not string else item.keywords }}{% endif %}
{% if item.cocos_label %}- **COCOS transformation type:** `{{ item.cocos_label }}` — include a brief sign-convention sentence in documentation.{% endif %}
{% if item.parent_path %}- **Parent:** {{ item.parent_path }}{% endif %}
{% if item.previous_name %}- **⟳ Previous generation:** `{{ item.previous_name.name }}`{% if item.previous_name.review_status %} ({{ item.previous_name.review_status }}){% endif %}{% endif %}

{% endfor %}

## Vocabulary Gaps

If a path requires a token that does **not** exist in a closed grammar
segment (e.g., a new `subject` species or `position`), do NOT invent
an invalid name. Instead, add the path to the `vocab_gaps` list with:

- `source_id`: the DD path
- `segment`: which grammar segment is missing a token
- `needed_token`: the token value you would need
- `reason`: why this token is needed
