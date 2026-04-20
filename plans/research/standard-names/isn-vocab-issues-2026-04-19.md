# ISN Vocabulary Gap Issues — 2026-04-19

Identified during quarantine triage of 15 StandardName nodes after enrichment rotation.
Each entry documents a vocabulary gap in `imas-standard-names` that prevents valid
physics concepts from passing grammar validation.

---

## 1. Process token: `diamagnetic_drift`

**Segment:** `process` (used with `due_to_` template)
**Needed token:** `diamagnetic_drift`
**Physics justification:**
The diamagnetic drift is a fundamental plasma drift arising from pressure gradients
in a magnetized plasma: $v_{dia} = \frac{B \times \nabla p}{qnB^2}$. Transport and
edge physics codes routinely attribute momentum fluxes and diffusivities to this
drift. The DD has explicit `diamagnetic` subfields under transport model velocity,
current_density, and energy flux structures. Three quarantined names need this token:

- `ion_momentum_flux_due_to_diamagnetic_drift` (edge_plasma_physics)
- `ion_momentum_diffusivity_due_to_diamagnetic_drift` (edge_plasma_physics)
- `radial_component_of_ion_momentum_flux_due_to_diamagnetic_drift` (transport)

**Example SN:** `ion_momentum_flux_due_to_diamagnetic_drift`
**Current status:** All three names fail with "Token 'diamagnetic_drift' used with
'due_to_' template is missing from Process vocabulary"
**Proposed ISN change:** Add `diamagnetic_drift` to `processes.yml` alongside existing
process tokens like `bootstrap`, `ohmic_dissipation`, `impurity_radiation`.

---

## 2. Position token: `wall_outline_point`

**Segment:** `position` (used with `of_` template for geometry)
**Needed token:** `wall_outline_point`
**Physics justification:**
The wall outline (first-wall contour in the poloidal plane) is a fundamental
geometry used in plasma-wall interaction studies. The DD exposes R/Z coordinates
of wall outline points under `wall/description_2d/limiter/unit/outline/r` and `/z`.
The ISN already has `plasma_boundary_outline_point` as a position token;
`wall_outline_point` follows the same pattern for the wall contour.

**Example SN:** `vertical_coordinate_of_wall_outline_point`
**Current status:** `vertical_coordinate_of_outline_point` was generated with
unqualified `outline_point`. Even with correct qualification, `wall_outline_point`
is not in the Position vocabulary.
**Proposed ISN change:** Add `wall_outline_point` to `positions.yml`.

---

## 3. Position token: `wall` (verify availability)

**Segment:** `position` (used with `at_` template)
**Needed token:** `wall`
**Physics justification:**
Five quarantined names used `at_wall_surface` when they should have used `at_wall`.
This assumes `wall` is already a valid Position token. If it is not, it should be
added — "at the wall" is the most common position qualifier in plasma-wall
interaction physics. All SOL/divertor energy balance quantities reference fluxes
arriving at or emitted from the wall.

**Example SNs:**
- `emitted_radiation_energy_flux_at_wall`
- `electron_emitted_kinetic_energy_flux_at_wall`
- `ion_emitted_kinetic_energy_flux_at_wall`
- `current_driven_emitted_energy_flux_at_wall`
- `ion_emitted_energy_flux_due_to_recombination_at_wall`

**Proposed ISN change:** Verify `wall` exists in `positions.yml`; add if missing.

---

## Summary

| # | Segment  | Token                  | SNs affected | Priority |
|---|----------|------------------------|:------------:|----------|
| 1 | process  | `diamagnetic_drift`    | 3            | High     |
| 2 | position | `wall_outline_point`   | 1            | Medium   |
| 3 | position | `wall`                 | 5            | High     |

---

## rc16 Release — 2026-04-20

All three issues above were resolved in rc15 (2026-04-19). The rc16 release
extends vocabulary coverage substantially, closing 56 additional VocabGap
entries identified by querying the codex Neo4j graph:

```cypher
MATCH (vg:VocabGap)
RETURN vg.segment AS segment, vg.needed_token AS token, count(*) AS hits
ORDER BY hits DESC
```

### Tokens added per category

| Category        | Count | Key additions |
|-----------------|:-----:|---------------|
| Positions       |  16   | `geometric_axis`, `control_surface`, `separatrix`, `pedestal`, `pedestal_top`, `charge_state`, `ion_charge_state`, `along_line_of_sight`, `launching_position` |
| Objects         |  15   | `wave_beam`, `beam_tracing_beam`, `beam_tracing_ray`, `coordinate_system`, `ferritic_insert`, `pellet`, `vacuum_vessel`, `diagnostic_aperture`, `plasma_facing_component` |
| Subjects        |  13   | `thermal_neutral`, `total_ion`, `molecular_ion`, `molecular_neutral`, `cold_neutral`, `hot_neutral`, `trapped`, `co_passing`, `counter_passing`, `fast_particle` |
| Processes       |   6   | `resistive_dissipation`, `dreicer`, `coulomb_collisions`, `neutral_beam_shinethrough`, `surface_emission`, `halo_current` |
| Transformations |   6   | `perturbed`, `spectral_density`, `second_derivative_of`, `per_poloidal_mode`, `derivative_with_respect_to_rho_tor`, `line_of_sight_projected` |

**ISN release**: `v0.7.0rc16` (tag SHA: `6448c4e`)
**Codex bump**: `pyproject.toml` updated from `v0.7.0rc15` → `v0.7.0rc16`
**VocabGap count**: 1066 total (636 physical_base + 430 structured). Structured-segment
gaps reduced: ~56 tokens directly match VocabGap entries; remaining gaps are mostly
LLM grammar composition errors (compound terms placed in wrong segments) that require
prompt engineering rather than vocabulary additions.
