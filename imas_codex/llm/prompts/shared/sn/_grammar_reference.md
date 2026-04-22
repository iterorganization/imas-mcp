## Standard Name Grammar (vNext — rc21)

### Core Axioms

1. **One concept, one name** — every physical concept maps to exactly one canonical string.
2. **Liberal parser, strict generator** — the parser accepts legacy and colloquial forms with diagnostics; the canonical rendering is unique.
3. **Postfix locus** — spatial qualifiers (`_of_`, `_at_`, `_over_`) and mechanism (`_due_to_`) always follow the base quantity.
4. **Prefix projection** — axis projections (`radial_component_of_`) precede the base.
5. **Explicit operator scope** — prefix operators carry `_of_` as a scope marker (`gradient_of_X`); postfix operators concatenate directly (`X_magnitude`).
6. **Closed vocabularies** — physical_base, operators, subjects, loci, processes, components, and geometry_carriers are all closed registries. If no token fits, emit a `vocab_gap`; never invent.

### 5-Group Internal Representation

Every standard name decomposes into five groups:

| Group | Role | Example tokens |
|-------|------|----------------|
| **operators** | Math ops, applied outer→inner | prefix: `time_derivative`, `gradient`, `normalized`; postfix: `magnitude`, `real_part`, `fourier_coefficient` |
| **projection** | Axis decomposition of vector/tensor | `radial_component_of_`, `toroidal_component_of_`, `parallel_component_of_` |
| **qualifiers** | Species or population prefix | `electron`, `ion`, `deuterium`, `fast_ion`, `thermal_electron` |
| **base** | Physical quantity (**closed** vocab, ~250 tokens) | `temperature`, `pressure`, `density`, `magnetic_field`, `safety_factor` |
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

- **`physical_base` is closed** — if no registered base fits, report as `vocab_gap`. Never invent a new token or use free-form strings.
- **Generic bases require qualification** — tokens like `temperature`, `current`, `pressure`, `density` must have at least one qualifier (species, component, or locus).
- **Process attribution** uses `_due_to_` + a Process vocabulary noun: `plasma_current_due_to_bootstrap`, never `bootstrap_current`.
- **DD path independence** — names describe physics, not DD location. Never include IDS names or DD section prefixes.
- **No processing verbs** — `reconstructed_`, `measured_`, `fitted_` are provenance, not physics. Drop them.
- **Preposition discipline** — use `_of_` for properties of named entities, `_at_` for field values at points, `_over_` for region integrals. Never use `_from_`.

### Rejected rc20 Forms

| ❌ rc20 form (now invalid) | ✅ vNext canonical | Reason |
|----------------------------|--------------------|----|
| `real_part_of_X` | `X_real_part` | Postfix operator, not prefix |
| `amplitude_of_X` | `X_amplitude` | Postfix operator, not prefix |
| `imaginary_part_of_X` | `X_imaginary_part` | Postfix operator, not prefix |
| Open `physical_base` fallback | Closed vocab + VocabGap | No free-form base tokens |
| `volume_averaged_X` (bare concat) | `volume_averaged_of_X` | Operator scope requires `_of_` |
| `electron_thermal_pressure` | `thermal_electron_pressure` | Population precedes species |
| `ion_rotation_frequency_toroidal` | `toroidal_component_of_ion_rotation_frequency` | No trailing component |
| `diamagnetic_component_of_X` | `X_due_to_diamagnetic_drift` | Diamagnetic is a drift, not an axis |
