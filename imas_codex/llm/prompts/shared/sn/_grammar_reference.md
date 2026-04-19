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

#### Process Attribution (`_due_to_`)

Use the `_due_to_` pattern to separate process from base quantity:

| ✅ Correct | ❌ Incorrect | Why |
|------------|--------------|-----|
| `plasma_current_due_to_bootstrap` | `bootstrap_current` | Process stuffed into physical_base |
| `heating_power_due_to_ohmic` | `ohmic_heating_power` | Process stuffed into physical_base |
| `electron_energy_due_to_collisions` | `collision_energy` | Process stuffed into physical_base |

The process segment follows the base quantity with `_due_to_` as separator.
All 26 process tokens from the vocabulary can be used in this pattern.

#### Generic Bases Requiring Qualification

These 12 physical bases are too generic to stand alone — they MUST be qualified
with at least one of: subject, component, position, or `_of_` device reference.

| Generic base | ❌ Invalid alone | ✅ Qualified examples |
|--------------|------------------|----------------------|
| `area` | `area` | `area_of_flux_loop`, `cross_section_area` |
| `current` | `current` | `plasma_current`, `toroidal_current`, `passive_loop_current` |
| `energy` | `energy` | `electron_energy`, `thermal_energy` |
| `flux` | `flux` | `poloidal_magnetic_flux`, `particle_flux` |
| `frequency` | `frequency` | `ion_cyclotron_frequency`, `collision_frequency` |
| `number_density` | `number_density` | `electron_number_density`, `ion_number_density` |
| `power` | `power` | `heating_power`, `radiated_power` |
| `pressure` | `pressure` | `electron_pressure`, `magnetic_pressure` |
| `temperature` | `temperature` | `electron_temperature`, `ion_temperature` |
| `velocity` | `velocity` | `toroidal_velocity`, `radial_velocity` |
| `voltage` | `voltage` | `loop_voltage`, `flux_loop_voltage` |
| `volume` | `volume` | `plasma_volume`, `volume_of_flux_surface` |

If the source context does not provide enough information to qualify the base,
report this as a `vocab_gap` rather than using the bare generic base.

#### Transformations (live from imas-standard-names)

The full list of allowed transformation tokens is the `Transformation` enum
in the installed ISN grammar package. The prompt template renders this list
dynamically via `prompt_loader.render_prompt` with `{{ transformations }}`
context variable.

Current tokens: {{ transformations | join(', ') }}

Always use ONLY tokens from this list. Do not invent new transformation
tokens.

If you need a transformation not listed above, **report it as a `vocab_gap`** with
`segment: "transformation"` and the needed token. Still generate the best name you
can (typically base-only without the transformation) alongside the gap report.

#### Grammar Ambiguities

The `component` and `coordinate` segments share 12 overlapping tokens (e.g.,
`radial`, `toroidal`, `poloidal`, `diamagnetic`, `parallel`, `perpendicular`).
This creates parser ambiguity when both would apply.

**If a name needs both component AND coordinate qualifiers:**
- Report the affected name with a validation issue in `vocab_gaps`
  using `segment: "grammar_ambiguity"` and describe the overlap
- Generate the best single-qualifier name you can (prefer `component`)
- Do NOT produce names that will fail grammar parse (e.g., names where
  `diamagnetic` could be either component or coordinate)

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
