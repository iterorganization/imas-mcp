# Plan Audit + SN Corpus Snapshot (feature-tree triage)

## Purpose

Full audit of every plan in `plans/features/` comparing intent to implemented code,
with status verdict, action taken, and phasing/benefit assessment for everything that
stays in `features/`. Also includes: 3 SN examples per physics domain with length &
clarity assessment, a unit pass-through verification, and the gap that motivates a
new unit-integrity-test plan.

## Graph snapshot (at audit time)

```
StandardName (valid):         335
StandardName (quarantined):    50
StandardName total:           385   (all produced by claude-sonnet-4.6)
description coverage:           0 / 335  ← no enrichment has run yet
documentation coverage:         0 / 335
source_paths links:           100 % (avg 2.4 paths/name)
HAS_UNIT + HAS_COCOS edges:   385 / 385
PRODUCED_NAME back-edges:     822         (sources linked)
HAS_STANDARD_NAME back-edges: 819
```

DD-side node-category distribution (proves Plan 30 landed):

```
representation   2209
geometry         1095
fit_artifact      269
```

---

## Plan status matrix

| Plan | Location before | Verdict | Action |
|---|---|---|---|
| 27 sn-vector-hierarchy | features/standard-names/ | **superseded** by 28 (per own header) | moved → `completed/superseded/` |
| 28 sn-greenfield-pipeline | features/standard-names/ | **superseded** by 29 (per 29 header) | moved → `completed/superseded/` |
| 29 architectural-pivot | features/standard-names/ | **partial landed** — enrich pipeline + split command exist, 0 names enriched | **KEEP** with gap notes (below) |
| 30 dd-semantic-categories | features/standard-names/ | **completed** — graph proves it | **DELETED** (project rule: code is the doc) |
| 31 quality-bootstrap-v2 | features/standard-names/ | **in-progress** — WS-F/WS-G landed, rotation active | **KEEP** as active |
| 20 consistency-and-prompt-enrichment | pending/ | mostly absorbed into 23/29; two items remain (two-pass consolidate, regen CLI) | **KEEP** in `pending/`, flag for closure in new plan |
| 23 quality-parity | pending/ | Phases 0/1A/1B mostly landed; Phase 4 (async link resolution) unlanded | **KEEP** in `pending/`, flag remaining phases |
| dd-rebuild | features/ | pending, high-leverage (DD-upstream) | **KEEP** |
| dd-search-quality-ab | features/ | pending (search tool polish) | **KEEP** |
| dd-server-cleanup | features/ | pending (small) | **KEEP** |
| docs-refresh | features/ | pending (doc drift) | **KEEP** |
| search-quality-improvements | features/ | pending (reranking) | **KEEP** |
| sn-bootstrap-loop | features/ | partial — Layer 1 audits landed; Layer 2 batched review, Layer 3 consolidate unlanded | **KEEP** with gap note |
| NEW: sn-unit-integrity-tests | features/ | — | **CREATED** (see below) |
| NEW: sn-enrichment-rotation | features/ | — | **CREATED** (documentation phase, per user stated priority) |
| NEW: sn-coverage-closure | features/ | — | **CREATED** (target "all relevant DD nodes") |

## Detailed assessment of plans that remain in `features/`

### Active standard-name plans

**29 architectural-pivot** — core carrier for generate/enrich split.
- LANDED: `sn enrich` CLI, `enrich_state.py`, `enrich_workers.py`, `enrich_pipeline.py`, review subtree.
- LANDED: `sn-generate` model label in settings (inferred from code).
- GAP: 0 / 335 valid names have been enriched. Enrich pipeline is running-capable but
  has not been exercised in production. This is the first follow-up once grammar rc14+
  settles.
- Benefit: produces the description + documentation that downstream MCP search depends
  on for meaningful SN-based retrieval.
- Phasing: validate enrich on one domain (edge_plasma, 47 names) → critique → bulk run.
  Tracked in new `sn-enrichment-rotation.md`.

**31 quality-bootstrap-v2** — rotation driver.
- LANDED (per commits): WS-A prompt patches, WS-B audit extensions, WS-C NC-20-31 audits,
  WS-D dep bumps to rc13+, WS-F graph remediation, WS-G corpus health gates + dashboard +
  bootstrap orchestration script.
- GAP: rotation has not covered ~15 physics domains (see `sn-coverage-closure.md`).
- Benefit: operational framework for continuing quality bootstrap.
- Phasing: continue rotation, next target `magnetic_field_systems` (269 paths) then
  `gyrokinetics` (264) per prior session.

### Root-level plans

**dd-rebuild** — MUST stay, highest leverage for SN quality ceiling.
- User explicitly encouraged `--force` rebuild if it unlocks material gains.
- NodeCategory geometry already landed via Plan 30. But `documentation` text enrichment
  by sonnet-4.6 (the plan's core) has NOT been re-run for all IDSs.
- Benefit: richer DD descriptions → better SN prompts → better names AND better enrich
  documentation. Two birds.
- Phasing: (1) benchmark enrichment model pass on one IDS; (2) full-corpus run on a
  compute node; (3) re-embed, re-cluster, re-score; (4) archive old DD graph; (5)
  graph-push.

**dd-server-cleanup** — small, 3 surgical fixes. Run as a maintenance commit.

**dd-search-quality-ab** — adjacent MCP-tool polish. Not on SN critical path; run when
user-facing retrieval is the focus.

**docs-refresh** — doc drift cleanup. Important to schedule before the next external
release because the rotation work has added many concepts (audits, node_category,
validation_status, sn enrich) that aren't in README/AGENTS.

**search-quality-improvements** — reranking. Blocks on enrich being run (no
descriptions = nothing meaningful to rerank). Defer until `sn-enrichment-rotation`
has produced substantive docs.

**sn-bootstrap-loop** — Layer 1 audit framework landed. Layer 2 (batched reviewer LLM)
and Layer 3 (cross-batch consolidate) are still valuable but have been partially
absorbed into `sn review`. Downgrade scope: keep only the consolidation + showcase
examples phases; fold audit work into Plan 31.

### Pending-folder items (partial)

**pending/20** — two remaining items to fold into a new closure plan:
- two-pass generate→consolidate (4B in 20's numbering) — partial in compose_worker.
- `sn regenerate` CLI — **NOT NEEDED**; replaced by `sn clear --domain X` + `sn generate --domain X`.
  → recommend CLOSING this plan (superseded). Will do in a follow-up commit once confirmed.

**pending/23** — Phase 4 (async link resolution) is a standalone enhancement.
- Recommend: keep in pending/ until enrich pipeline has been exercised; async links only
  become useful once there are enriched documents to link.

---

## 3 SN examples per physics domain — clarity assessment

| Domain | n | Example (length) | Length grade |
|---|---|---|---|
| transport (113) | `radial_component_of_magnetic_field` (34) | ✅ clear, <40 char |
| | `toroidal_beta` (13) | ✅ excellent |
| | `poloidal_magnetic_flux` (22) | ✅ excellent |
| equilibrium (63) | `distance_to_closest_wall_point_of_plasma_boundary` (49) | ⚠️ borderline, 49 char |
| | `vacuum_toroidal_field_flux_function` (35) | ✅ |
| | `poloidal_component_of_magnetic_field_of_poloidal_magnetic_magnetic_field_probe` (78) | ❌ duplicated "magnetic", too long |
| edge_plasma_physics (47) | `parallel_component_of_ion_momentum_diffusivity` (46) | ✅ acceptable |
| | `parallel_component_of_ion_velocity` (34) | ✅ |
| | `thermal_electron_pressure` (25) | ✅ |
| radiation_meas_diag (31) | `major_radius_of_measurement_position` (36) | ✅ |
| | `radiated_power_due_to_impurity_radiation` (40) | ⚠️ tautology ("radiated power ... radiation") |
| | `toroidal_angle_of_delta_position` (32) | ⚠️ `delta_position` is grammar-weak |
| magnetohydrodynamics (31) | `grid_object_geometry` (20) | ⚠️ too generic for a canonical SN |
| | `major_radius_of_closest_wall_point` (34) | ✅ |
| | `flux_surface_averaged_toroidal_current_density` (46) | ✅ |
| plasma_wall_interactions (24) | `toroidal_component_of_magnetic_field` (36) | ✅ |
| | `vertical_component_of_magnetic_field` (36) | ✅ |
| | `ion_atomic_mass` (15) | ✅ |
| plasma_control (12) | `normalized_toroidal_beta` (24) | ✅ |
| | `plasma_internal_inductance` (26) | ✅ |
| | `lower_elongation_of_plasma_boundary` (35) | ✅ |
| turbulence (10) | `neutral_temperature` (19) | ✅ |
| | `ion_temperature` (15) | ✅ |
| | `thermal_ion_number_density` (26) | ✅ |
| magnetic_field_diagnostics (4) | `stokes_s0_of_fiber_optic_current_sensor` (39) | ⚠️ hardware-bound (NC-30 risk) |

**Length stats (valid only, n=335):** min 12, median 39, p75 47, p90 60, max 90,
avg ≈ 40. Target should be median ≤ 40, p90 ≤ 55. Currently p90 is 60 — acceptable
but close to the line. **10 names exceed 70 characters** — all candidates for shortening:

```
 90  flux_surface_averaged_inverse_square_magnetic_field_strength_times_squared_radial_gradient
 88  equilibrium_reconstruction_constraint_weight_of_vertical_iron_core_segment_magnetization
 86  equilibrium_reconstruction_constraint_weight_of_radial_iron_core_segment_magnetization
 81  derivative_of_flux_surface_cross_sectional_area_with_respect_to_radial_coordinate
 78  poloidal_component_of_magnetic_field_of_poloidal_magnetic_magnetic_field_probe   ← duplicated "magnetic"
 72  equilibrium_reconstruction_constraint_weight_of_toroidal_current_density
 72  parallel_component_of_current_density_due_to_non_inductive_current_drive
 71  measurement_time_of_vertical_iron_core_segment_magnetization_constraint
 71  normalized_toroidal_flux_coordinate_of_neoclassical_tearing_mode_center
 70  equilibrium_reconstruction_constraint_weight_of_mse_polarization_angle
```

**Anti-patterns surfaced:**
1. Repeated word (`magnetic_magnetic`) — audit miss, needs a new `repeated_token_check`.
2. `equilibrium_reconstruction_constraint_weight_of_X` family (5 of 10 longest) — genuine
   physics concept but warrants a vocabulary compaction proposal in ISN
   (`reconstruction_weight` as a single base noun?).
3. `*_of_fiber_optic_current_sensor` — hardware-instrument-bound name; NC-30 audit exists
   but hasn't fired on stokes_s0/s1/s2 because they're not `_of_X_sensor` pattern it recognises.
   Needs rule extension.

---

## Unit pass-through verification (code review)

```
imas_codex/standard_names/workers.py → compose_worker
  reads extracted path info which includes `unit` from DD HAS_UNIT edge
  LLM prompt marks `unit` as READ-ONLY
  post-LLM: candidate_dict['unit'] = extracted['unit']   ← OVERWRITE, not LLM value

imas_codex/standard_names/graph_ops.py → write_standard_names
  CREATE (sn:StandardName) SET sn += $payload
  MERGE (u:Unit {id: $unit})
  MERGE (sn)-[:HAS_UNIT]->(u)
```

**Verdict:** code pass-through is in place. However the audit (below) surfaced 4
SN nodes whose `unit` property disagrees with **every** DD path they link to — these
pre-date the latest sanitizer or were set by a pre-fix compose_worker. They need a one-time
backfill.

### Unit-integrity audit (single-run summary)

```
  match            248   (sn.unit == single DD path unit)
  no_dd_unit        75   (some source_paths have no HAS_UNIT edge on DD side — DD gap)
  multi_contains     8   (sn.unit is one of many DD-path units — acceptable)
  mismatch           4   (sn.unit disagrees with all DD path units — BUG)
```

The 4 mismatches (all pre-existing, to be fixed by new plan):

```
ion_average_charge_of_ion_state        SN='1'   DD='e'     ← DD-side bug (e is elementary charge unit; should be '1')
ion_average_square_charge_of_ion_state SN='1'   DD='e'     ← same DD-side bug
electron_temperature_peaking_factor    SN='1'   DD='eV'    ← SN correct, DD eV wrong (peaking factor IS dimensionless)
plasma_current                         SN='1'   DD='A'     ← SN WRONG (should be 'A'); was poisoned by pulse_schedule source in session 10 before the narrowing fix
```

Also **173 SN source_paths** point to DD nodes with no `HAS_UNIT` edge. These are
dimensionless quantities in the DD that lack an explicit `1` unit — a DD-schema gap.

---

## Gap-closure: four new plans proposed

1. **`sn-unit-integrity-tests.md`** — graph quality tests + one-time backfill to fix
   the 4 mismatches + a DD-upstream issue list for the unit gaps. Schema-driven test
   in `tests/graph/test_sn_unit_integrity.py`.

2. **`sn-enrichment-rotation.md`** — formalise the rotation-bootstrap approach for
   the documentation (enrich) phase. Per-domain $2 cap, graph-aware POSTLINK context,
   senior-review feedback loop, mirrors the naming rotation.

3. **`sn-coverage-closure.md`** — finish coverage for the ~15 domains untouched by
   rotation. Explicit priority order: `magnetic_field_systems` → `gyrokinetics` →
   `stellarator_geometry` → `wave_physics` → rest.

4. **Name-quality hygiene** — not a full plan, but three audit additions belong in
   Plan 31 WS-B:
   - `repeated_token_check` (catches `magnetic_magnetic`)
   - `length_soft_cap_check` (quarantine > 80 char; warn > 60)
   - `instrument_stokes_bind_check` (extends NC-30 to cover `stokes_X_of_Y_sensor`)

## Next steps (ordered)

1. Commit this audit + three new plans.
2. Run `sn-unit-integrity-tests` plan — small, safe, catches real bugs.
3. Run first enrich rotation (edge_plasma_physics, 47 names) per
   `sn-enrichment-rotation.md` → review → iterate.
4. Continue name-coverage rotation on `magnetic_field_systems` per
   `sn-coverage-closure.md`.
