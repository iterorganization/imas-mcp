## Standard Name Grammar

A valid standard name is composed from optional segments in a specific order:

**Canonical pattern:** {{ canonical_pattern }}

### Segment Order

{{ segment_order }}

### Exclusive Pairs

These segment pairs are mutually exclusive — never use both in the same name:
{% for pair in exclusive_pairs %}
- **{{ pair[0] }}** and **{{ pair[1] }}**
{% endfor %}

### Segment Vocabulary

{% for section in vocabulary_sections %}
#### {{ section.segment }}{% if section.is_open %} (open vocabulary){% endif %}

{{ section.description }}
{% if section.template %}
Template: `{{ section.template }}`
{% endif %}
{% if section.tokens %}
Valid tokens: {{ section.tokens | join(', ') }}
{% endif %}
{% if section.is_open %}
Use any physics quantity in snake_case (e.g., temperature, density, magnetic_field, pressure).
{% endif %}

{% endfor %}

### Naming Rules

#### Preposition Usage

- Use `_of_` for **static properties** of a device component or geometric object:
  `area_of_flux_loop`, `resistance_of_passive_loop`, `elongation_of_plasma_boundary`.
- Use **device-prefix pattern** for **dynamic signals/outputs** measured from or flowing through a component:
  `passive_loop_current`, `flux_loop_voltage`, `poloidal_field_coil_current`.
- **NEVER** use `_from_` — it implies causation (e.g., ❌ `current_from_passive_loop`).

#### Processing Verbs

Standard names describe the **physics quantity**, not how it was obtained.
NEVER include processing or reconstruction method prefixes:

| ❌ Forbidden prefix | ✅ Correct approach |
|---------------------|---------------------|
| `reconstructed_` | Use base physics name; tag with `reconstructed` |
| `measured_` | Use base physics name; tag with `measured` |
| `calculated_` | Use base physics name; tag with `derived` |
| `fitted_` | Use base physics name; processing is metadata |
| `averaged_` (temporal) | Use base physics name; note averaging in docs |

**Exception:** `flux_surface_averaged_` is a valid `transformation` segment because it
describes a **mathematical transformation** that changes the quantity's dimensionality
or coordinate dependence, not a processing step.

#### DD Path Independence

Standard names are **IDS-agnostic** — they describe the physics, not the data dictionary location.
Never include DD organizational prefixes in the name:

| ❌ DD leakage | ✅ Physics name | Why |
|---------------|-----------------|-----|
| `geometric_minor_radius` | `minor_radius` | `geometric_` is a DD section prefix |
| `radial_profile_of_psi` | `poloidal_magnetic_flux` | `radial_profile_of_` describes DD array layout |
| `equilibrium_plasma_current` | `plasma_current` | `equilibrium_` is an IDS name |

Similarly, never reference DD path structure in the description or documentation fields.
