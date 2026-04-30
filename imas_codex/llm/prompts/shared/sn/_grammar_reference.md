## Standard Name Grammar (vNext — rc21)

### Core Axioms

1. **One concept, one name** — every physical concept maps to exactly one canonical string.
2. **Liberal parser, strict generator** — the parser accepts legacy and colloquial forms with diagnostics; the canonical rendering is unique.
3. **Postfix locus** — spatial qualifiers (`_of_`, `_at_`, `_over_`) and mechanism (`_due_to_`) always follow the base quantity.
4. **Prefix projection** — axis projections (`radial_component_of_`) precede the base.
5. **Explicit operator scope** — prefix operators carry `_of_` as a scope marker (`gradient_of_X`); postfix operators concatenate directly (`X_magnitude`).
6. **Closed vocabularies** — operators, subjects, components, coordinates, processes, objects, geometry/positions, regions, and geometric_bases are CLOSED. Only `physical_base` is open. If no token in a closed segment fits, emit `vocab_gap`; never invent and **never absorb a closed token into `physical_base`**.

### 5-Group Internal Representation

Every standard name decomposes into five groups:

| Group | Role | Example tokens |
|-------|------|----------------|
| **operators** | Math ops, applied outer→inner | prefix: `time_derivative`, `gradient`, `normalized`, `per_toroidal_mode`, `per_poloidal_mode`, `cumulative_inside_flux_surface`; postfix: `magnitude`, `real_part`, `fourier_coefficient` |
| **projection** | Axis decomposition of vector/tensor | `radial_component_of_`, `toroidal_component_of_`, `parallel_component_of_` |
| **qualifiers** | Species or population prefix | `electron`, `ion`, `deuterium`, `fast_ion`, `thermal_electron` |
| **base** | Physical quantity (the ONE open slot — every other slot is closed) | `temperature`, `pressure`, `density`, `magnetic_field`, `safety_factor` |
| **locus + mechanism** | Where (postfix) + process (postfix) | `_of_plasma_boundary`, `_at_magnetic_axis`, `_over_core_region`, `_due_to_bootstrap` |

### Canonical Rendering

```
[operators] [projection_component_of_] [qualifiers] base [_of/_at/_over locus] [_due_to process]
```

**Operator templates:**
- **Unary prefix**: `op_of_` + inner → `time_derivative_of_electron_temperature`
- **Unary postfix**: inner + `_op` → `electron_temperature_magnitude`, `perturbed_magnetic_field_real_part`
- **Binary**: `op_of_` + A + `_and_`/`_to_` + B → `ratio_of_electron_to_ion_temperature`

### `_of_` Disambiguation

After vNext, `_of_` appears in exactly three structural roles:

| Role | Template | Disambiguator |
|------|----------|---------------|
| Prefix operator scope | `op_of_` + inner | Longest-match against operator registry |
| Binary operator | `op_of_` A `_and_`/`_to_` B | `_and_`/`_to_` keyword present |
| Locus (entity/geometry) | base + `_of_` + locus | Always the **last** `_of_` in the name |

### Key Rules

- **`physical_base` is the SINGLE open slot** — every other segment is closed.
  When no registered token fits a closed segment, report as `vocab_gap`. Never
  invent a new token, use a free-form string, **or absorb the missing token
  into `physical_base`**.
- **Generic bases require qualification** — tokens like `temperature`, `current`, `pressure`, `density` must have at least one qualifier (species, component, or locus).
- **Process attribution** uses `_due_to_` + a Process vocabulary noun: `plasma_current_due_to_bootstrap`, never `bootstrap_current`.
- **DD path independence** — names describe physics, not DD location. Never include IDS names or DD section prefixes.
- **No processing verbs** — `reconstructed_`, `measured_`, `fitted_` are provenance, not physics. Drop them.
- **Preposition discipline** — use `_of_` for properties of named entities, `_at_` for field values at points, `_over_` for region integrals. Never use `_from_`.
- **Spectral decomposition** — `per_toroidal_mode` and `per_poloidal_mode` are registered **unary prefix** operators. Canonical form: `per_toroidal_mode_of_X`. They indicate the quantity is resolved per Fourier toroidal/poloidal mode component. Do not penalise these as unknown operators.

### Rejected rc20 Forms

| ❌ rc20 form (now invalid) | ✅ vNext canonical | Reason |
|----------------------------|--------------------|----|
| `real_part_of_X` | `X_real_part` | Postfix operator, not prefix |
| `amplitude_of_X` | `X_amplitude` | Postfix operator, not prefix |
| `imaginary_part_of_X` | `X_imaginary_part` | Postfix operator, not prefix |
| Closed-vocab token absorbed into base | `<segment>_token_<rest>` decomposition | Place every closed token in its segment |
| `volume_averaged_X` (bare concat) | `volume_averaged_of_X` | Operator scope requires `_of_` |
| `electron_thermal_pressure` | `thermal_electron_pressure` | Population precedes species |
| `ion_rotation_frequency_toroidal` | `toroidal_component_of_ion_rotation_frequency` | No trailing component |
| `diamagnetic_component_of_X` | `X_due_to_diamagnetic_drift` | Diamagnetic is a drift, not an axis |

{% if closed_vocab_full %}
### Closed-Vocabulary Token Registry — EVERY closed segment, EVERY token

The following lists are the complete, authoritative closed vocabulary for
each segment.  **Any token that appears below is a CLOSED-segment token and
must be placed in its declared segment slot — never in `physical_base`.**

If you find yourself writing a `physical_base` value that contains one of
these tokens as an underscore-separated substring, STOP and re-decompose:
the token belongs in its closed segment.  Examples of the failure mode this
prevents:

- `toroidal_torque` → `toroidal` is in `component`; the correct decomposition
  is `component=toroidal, physical_base=torque`, rendered as
  `toroidal_component_of_torque`.
- `volume_averaged_electron_temperature` → `volume_averaged` is a
  `transformation`; `electron` is a `subject`. Render as
  `volume_averaged_of_electron_temperature`.
- `parallel_viscosity_current_density` → `parallel` is a `component`. Render
  as `parallel_component_of_viscosity_current_density`.

{% for vs in closed_vocab_full %}
#### `{{ vs.segment }}`{% if vs.aliases %} (alias{{ "es" if vs.aliases|length > 1 else "" }}: {{ vs.aliases | join(', ') }}){% endif %} — {{ vs.tokens | length }} tokens

```
{{ vs.tokens | join(', ') }}
```

{% endfor %}

### Decomposition Checklist — apply BEFORE you commit a name

For every candidate name, run these checks IN ORDER. If any fires, restructure
before emitting:

1. **Tokenise the candidate on `_`.**  Walk the resulting tokens left-to-right.
2. **For each token (and 2-token / 3-token compound), look it up in the
   closed-vocab registry above.**  If the token appears in any closed segment,
   it MUST occupy that segment slot in `grammar_fields`, not be absorbed into
   `physical_base`.
3. **Whitelist genuine atomic compounds** that happen to share a prefix with a
   closed token but are NOT decomposable: `poloidal_flux`, `minor_radius`,
   `cross_sectional_area`, `safety_factor`, `polarization_angle`,
   `ellipticity_angle`, `loop_voltage`, `internal_inductance`. These are
   single, lexicalised physics terms.
4. **If a closed-vocab token is present but no atomic compound rule exempts
   it**, restructure:
   - `<component>_<base>` → `<component>_component_of_<base>` (or place
     `<component>` in the `component` slot of `grammar_fields`).
   - `<subject>_<base>` → keep `<subject>` in the `subject` slot; never let
     it leak into `physical_base`.
   - `<transformation>_<base>` → `<transformation>_of_<base>` with the
     transformation in its own slot.
   - `<base>_<process>` → `<base>_due_to_<process>` with the process in
     its own slot.
   - `<base>_<region>` → `<base>_over_<region>` with the region in its own slot.
5. **Re-render the name** from the corrected `grammar_fields` and confirm the
   `physical_base` slot contains ONLY tokens that do not appear in any closed
   vocabulary above.
6. **If a needed token is missing from a closed registry**, emit a
   `vocab_gap` against that segment — DO NOT invent a token and DO NOT
   absorb it into `physical_base`.

This checklist directly addresses the dominant failure mode surfaced by
expert reviewers: closed-vocab tokens (toroidal, parallel, thermal,
e_cross_b_drift, normalized, fast_ion, …) crammed into `physical_base`
instead of placed in their correct grammar slot.
{% endif %}
