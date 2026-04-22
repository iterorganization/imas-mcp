---
name: sn/compose_dd
description: Dynamic user prompt for SN composition — per-batch DD paths with enriched context
used_by: imas_codex.standard_names.workers.compose_worker
task: composition
dynamic: true
schema_needs: []
---

Generate standard names for the following IMAS Data Dictionary paths.

{% if retry_reason %}
## ⚠️ Retry Context

{{ retry_reason }}
{% endif %}

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
| `current_from_passive_loop` | `passive_loop_current` | `_from_` implies causation — use device prefix for signals |
| `poloidal_flux` | `poloidal_magnetic_flux` | Use controlled vocabulary term; no synonymous short forms |
| `reconstructed_faraday_rotation_angle` | `faraday_rotation_angle` | Processing method is metadata, not part of the name |
| `geometric_minor_radius` | `minor_radius` | DD section prefix leaking into standard name |
| `flux_surface_averaged_elongation` | `elongation` | Elongation is a geometric property of a contour, not a flux-surface average |
| `energy_flux_at_wall_surface` | `energy_flux_at_wall` | Position token is `wall`, not `wall_surface` — the `_surface` suffix is redundant |
| `energy_due_to_recombination_at_ion_state` | `energy_due_to_recombination` | Process tokens are bare vocabulary entries — never append `_at_X` / `_in_X` / `_on_X` qualifiers |
| `energy_due_to_impurity_radiation_in_halo_region` | `halo_region_radiated_energy_due_to_impurity_radiation` | Region qualifiers go in the subject prefix, not after `due_to_<process>` |
| `vertical_coordinate_of_outline_point` | `vertical_coordinate_of_wall_outline_point` | Always qualify `outline_point` with its parent entity (`wall_`, `plasma_boundary_`) |

## Batch Consistency Check

Before finalizing your output, verify:
1. **No synonymous names** — if you used `magnetic_flux` in one entry, don't use just `flux` in another
2. **Consistent suffixes** — all boundary quantities use `_of_plasma_boundary`, not a mix of patterns
3. **No DD leakage** — none of your names start with an IDS or DD section name

## IDS Context

{% if ids_contexts %}
{% for ids_ctx in ids_contexts %}
### {{ ids_ctx.ids_name }}
{{ ids_ctx.ids_description }}
{% if ids_ctx.ids_documentation %}*{{ ids_ctx.ids_documentation }}*{% endif %}
{% if ids_ctx.top_sections %}
**Top-level sections:**
{% for sec in ids_ctx.top_sections %}
- `{{ sec.name }}` ({{ sec.data_type }}): {{ sec.description or 'no description' }}
{% endfor %}
{% endif %}
{% endfor %}
{% else %}
**Batch:** {{ ids_name }}
{% endif %}
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

{% if nearby_existing_names %}
## Nearby Existing Standard Names

These names already exist in the catalog. Reuse them if they match your source, or avoid creating duplicates:
{% for name in nearby_existing_names %}
- **{{ name.id }}**: {{ name.description | default('', true) }} ({{ name.kind | default('scalar', true) }}, {{ name.unit | default('dimensionless', true) }})
{% endfor %}
{% endif %}

{% if reference_exemplars %}
## REFERENCE EXEMPLARS — match this level of specificity

These validated standard names are semantically similar to items in this batch.
Use them as quality benchmarks for naming style, documentation depth, and field usage:

{% for ex in reference_exemplars %}
### `{{ ex.name }}`
- **Description:** {{ ex.description }}
- **Unit:** {{ ex.unit }}
{% if ex.tags %}- **Tags:** {{ ex.tags | join(', ') if ex.tags is iterable and ex.tags is not string else ex.tags }}{% endif %}
{% if ex.documentation %}- **Documentation (excerpt):** {{ ex.documentation[:500] }}{% endif %}

{% endfor %}
{% endif %}

## DD Paths to Name

{% for item in items %}
### {{ item.path }}
{% if item.rate_hint %}
> ⚠️ **HARD CONSTRAINT — RATE QUANTITY:** The DD documentation for this path
> indicates a rate / time-derivative quantity (phrases like "instantaneous
> change", "signed change", "rate of change", "time derivative",
> "per unit time"). Your name MUST begin with `tendency_of_` (preferred),
> `change_in_`, or `rate_of_change_of_`. NEVER use `instant_change_*` or
> `instantaneous_change_*` as a prefix. The description MUST be consistent
> with the rate-marker prefix (e.g. if the name is `tendency_of_X`, the
> description should read "Instantaneous signed change in X" or
> "Time derivative of X"). Do NOT produce a base-quantity name (e.g.
> `electron_density`) and then describe it as a rate — that is a critical
> drift error that quarantines the entry.
>
> **CRITICAL — rate + component ordering:** If the quantity is a rate of a
> vector component, the orientation token (`parallel`, `perpendicular`,
> `poloidal`, `toroidal`, `radial`, `diamagnetic`) MUST be placed OUTSIDE
> the rate marker, wrapping the rate phrase:
>   ✅ `parallel_component_of_change_in_fast_electron_pressure`
>   ✅ `poloidal_component_of_tendency_of_electron_velocity`
>   ❌ `change_in_parallel_component_of_fast_electron_pressure` (grammar rejects)
>   ❌ `change_in_poloidal_component_of_electron_velocity` (grammar rejects)
> The grammar parses `{orientation}_component_of_X` as a unit — the rate
> marker must modify the base quantity X, not intrude between orientation
> and `component_of`.
{% endif %}
- **Description:** {{ item.description }}
{% if item.documentation and item.documentation != item.description %}- **DD Documentation:** {{ item.documentation }}{% endif %}
- **Unit:** {{ item.unit or 'dimensionless' }} *(authoritative from DD — use for naming context only, do NOT output)*
- **Data type:** {{ item.data_type or 'unspecified' }}
{% if item.physics_domain %}- **Physics domain:** {{ item.physics_domain }}{% endif %}
{% if item.ndim is not none %}- **Dimensions:** {{ item.ndim }}D{% endif %}
{% if item.lifecycle_status %}- **Lifecycle:** {{ item.lifecycle_status }} ⚠️{% endif %}
{% if item.keywords %}- **Keywords:** {{ item.keywords | join(', ') if item.keywords is iterable and item.keywords is not string else item.keywords }}{% endif %}
{% if item.coordinate_paths %}- **Coordinates:** {{ item.coordinate_paths | join(', ') }}{% endif %}
{% if item.timebase %}- **Timebase:** {{ item.timebase }}{% endif %}
{% if item.cocos_label %}- **COCOS transformation type:** `{{ item.cocos_label }}`{% if item.cocos_expression %} — expression: `{{ item.cocos_expression }}`{% endif %}
{% if cocos_version is defined and cocos_version %}- **COCOS convention:** {{ cocos_version }}{% if dd_version %} (DD {{ dd_version }}){% endif %}{% endif %}
{% if item.cocos_guidance %}- **Sign convention guidance:** {{ item.cocos_guidance }}{% endif %}
  ⚠️ This quantity is COCOS-dependent. You MUST include a sign convention paragraph
  in the documentation section of the form:
  `Sign convention: Positive when <concrete physical condition consistent with COCOS {{ cocos_version | default('') }}>.`
  Write a CONCRETE plain-English condition (e.g. "the current flows counter-clockwise
  viewed from above"). If you cannot supply a concrete condition, omit that paragraph
  entirely and write `This quantity has no sign ambiguity.` instead.{% endif %}
{% if item.identifier_schema %}- **Identifier schema:** {{ item.identifier_schema }}{% if item.identifier_schema_doc %} — {{ item.identifier_schema_doc }}{% endif %}{% endif %}
{% if item.coord_path %}- **Coordinate:** {{ item.coord_path }}{% if item.coord_unit %} ({{ item.coord_unit }}){% endif %}{% endif %}
{% if item.parent_path %}- **Parent structure:** {{ item.parent_path }} ({{ item.parent_type or 'STRUCTURE' }}){% endif %}
{% if item.parent_description %}- **Parent description:** {{ item.parent_description }}{% endif %}
{% if item.clusters %}
- **Semantic clusters:**
{% for cl in item.clusters %}  - **{{ cl.label }}** ({{ cl.scope }}): {{ cl.description }}
    Members: {{ cl.members | join(', ') }}
{% endfor %}{% endif %}
{% if item.cross_ids_paths %}
- **Cross-IDS equivalents:** These paths in other IDSs represent the same quantity:
{% for xp in item.cross_ids_paths %}  - `{{ xp }}`
{% endfor %}  → Generate ONE name that covers all cross-IDS instances.
{% endif %}
{% if item.hybrid_neighbours %}
- **Hybrid-search neighbours** (physics-concept + structural cousins):
{% for n in item.hybrid_neighbours %}  - `{{ n.tag }}` [{{ n.unit }}, {{ n.physics_domain }}]: {{ n.doc_short }}{% if n.cocos_label %} (COCOS {{ n.cocos_label }}){% endif %}
{% endfor %}  → Reuse a `name:` entry above when your source measures the same quantity.
{% endif %}
{% if item.related_neighbours %}
- **Graph-relationship neighbours** (explicit cross-IDS peers):
{% for r in item.related_neighbours %}  - `{{ r.path }}` ({{ r.ids }}) — {{ r.relationship_type }}{% if r.via %} via {{ r.via }}{% endif %}
{% endfor %}  → These paths share structural relationships (cluster, coordinate, unit, identifier, COCOS) with this path.
{% endif %}
{% if item.error_fields %}
- **DD error companions:**
{% for ef in item.error_fields %}  - `{{ ef }}`
{% endfor %}  → This path has uncertainty companions. Consider producing an `*_uncertainty` variant, or skip if this path IS the error field.
{% endif %}
{% if item.version_history %}
- **Version history:**
{% for vh in item.version_history %}  - {{ vh.version }}: {{ vh.change_type }}{% if vh.description %} — {{ vh.description }}{% endif %}
{% endfor %}{% endif %}
{% if item.sibling_fields %}
- **Sibling fields** (same parent structure — use for documentation cross-references):
{% for sib in item.sibling_fields %}  - `{{ sib.path }}`: {{ sib.description or 'no description' }} ({{ sib.data_type or '?' }})
{% endfor %}{% endif %}
{% if item.previous_name %}
- **⟳ Previous generation:** `{{ item.previous_name.name }}` ({{ item.previous_name.review_status or 'drafted' }}{% if item.previous_name.reviewer_score %}, score={{ item.previous_name.reviewer_score | round(2) }}{% endif %}{% if item.previous_name.review_tier %}, {{ item.previous_name.review_tier }}{% endif %})
{% if item.previous_name.description %}- **Prior description:** {{ item.previous_name.description }}{% endif %}
{% if item.previous_name.documentation %}- **Prior documentation:** {{ item.previous_name.documentation }}{% endif %}
{% if item.previous_name.tags %}- **Prior tags:** {{ item.previous_name.tags | join(', ') if item.previous_name.tags is iterable and item.previous_name.tags is not string else item.previous_name.tags }}{% endif %}
{% if item.previous_name.links %}- **Prior links:** {{ item.previous_name.links | join(', ') if item.previous_name.links is iterable and item.previous_name.links is not string else item.previous_name.links }}{% endif %}
{% if item.previous_name.validation_issues %}- **⚠️ Validation issues from prior run:** {{ item.previous_name.validation_issues | join('; ') if item.previous_name.validation_issues is iterable and item.previous_name.validation_issues is not string else item.previous_name.validation_issues }}{% endif %}
{% if item.previous_name.linked_dd_paths %}- **Other DD paths sharing this name:** These paths were also mapped to `{{ item.previous_name.name }}` — your generated name should be appropriate for all of them:
{% for ldp in item.previous_name.linked_dd_paths %}  - `{{ ldp }}`
{% endfor %}{% endif %}
{% if item.previous_name.review_status == 'accepted' %}- **⚠️ This name was human-accepted** — only replace with a clearly better alternative.{% endif %}
{% endif %}
{% if item.review_feedback %}
- **📝 Prior reviewer feedback — you MUST address the issues below in your new name:**
  - **Previous name:** `{{ item.review_feedback.previous_name }}`{% if item.review_feedback.reviewer_score is not none %} (score={{ item.review_feedback.reviewer_score | round(2) }}{% if item.review_feedback.review_tier %}, tier={{ item.review_feedback.review_tier }}{% endif %}){% endif %}
{% if item.review_feedback.previous_description %}  - **Prior description:** {{ item.review_feedback.previous_description }}
{% endif %}{% if item.review_feedback.reviewer_scores %}  - **Rubric scores (out of 20 each):**
{% for dim, dim_score in item.review_feedback.reviewer_scores.items() %}{% if dim not in ('score', 'tier') and dim_score is number %}    - `{{ dim }}`: {{ dim_score }}
{% endif %}{% endfor %}{% endif %}{% if item.review_feedback.reviewer_comments %}  - **Reviewer critique:**
    {{ item.review_feedback.reviewer_comments | replace('\n', '\n    ') }}
{% endif %}  - **Instruction:** Produce a name that directly fixes every concrete issue raised above. Do NOT re-emit the previous name unchanged. If the reviewer flagged excessive length, redundant qualifiers, or convention violations, your new name must be shorter / cleaner / more idiomatic. If the reviewer was satisfied with a dimension (score ≥ 15), preserve that aspect.
{% endif %}
{% if item.cluster_siblings %}- **Cross-IDS siblings:**
{% for sib in item.cluster_siblings[:5] %}  - {{ sib.path }} ({{ sib.unit or '?' }})
{% endfor %}
- **Concept identity:** These {{ item.cluster_siblings|length + 1 }} cross-IDS paths represent the SAME physics concept. Generate ONE name that covers all of them.
{% endif %}

{% endfor %}

## Grammar Fields — MANDATORY

For **every** candidate you emit, populate the `grammar_fields` map with the
grammar-segment decomposition of `standard_name`. This is not optional — it
is how downstream tooling validates the round-trip
`parse(name) → compose() == name`.

Use only these keys (omit keys whose segment is absent from the name):

```
subject, process, physical_base, geometric_base,
component, basis, position, reducer, reference, statistic
```

**Examples:**

- `electron_temperature` →
  `{"subject": "electron", "physical_base": "temperature"}`
- `electron_temperature_core` →
  `{"subject": "electron", "physical_base": "temperature", "position": "core"}`
- `radial_component_of_magnetic_field` →
  `{"component": "radial", "physical_base": "magnetic_field"}`
- `minor_radius_of_plasma_boundary` →
  `{"physical_base": "minor_radius", "position": "plasma_boundary"}`
- `distance_between_plasma_boundary_and_closest_wall_point` →
  `{"physical_base": "distance_between_plasma_boundary_and_closest_wall_point"}`
  (open-vocabulary compound; whole name is the physical_base.)

If you cannot decompose the name, the name is wrong — revise it rather than
emit an empty `grammar_fields`.

## Vocabulary Gaps

If a path requires a token that does **not** exist in a closed grammar segment
(e.g., a new `subject` species, a new `position`), do NOT invent an invalid name.
Instead, add the path to the `vocab_gaps` list in your response with:
- `source_id`: the DD path
- `segment`: which grammar segment is missing a token
- `needed_token`: the token value you would need
- `reason`: why this token is needed
