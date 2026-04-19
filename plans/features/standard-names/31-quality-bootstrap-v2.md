# Plan 31 — Quality Bootstrap v2 (rc14)

> **Status:** draft, ready for rotation.
> **Input:** `plans/research/standard-names/12-full-graph-review.md` (706-entry review).
> **Supersedes:** none. **Depends on:** plan 30 (DD semantic categories
> `fit_artifact`, `representation`).
> **Structured for parallel Opus agents** — every workstream is
> self-contained with its own inputs, files, acceptance criteria, and
> review-gate. Cross-workstream DAG at §12.

---

## 1. Executive summary

The rc13 full-graph review found 121 / 705 names (17.2%) quarantined, with
five blocker clusters (B1–B5 in review §0). Most of the defect mass is
structural, not compose-prompt quality: classifier gaps account for
**≥85% of quarantines** (representation coefficients, reference-waveform
sentinels, solver boundary-conditions, `/diamagnetic` axis misuse). Prompt
contradictions, grammar-vocab gaps, and audit bugs account for the rest.

Plan 31 dispatches **7 parallel workstreams** (WS-A … WS-G). WS-A–WS-D
can proceed in parallel once plan 30 merges. WS-E (DD filings) is
external. WS-F (graph remediation) is sequential on the code-side
workstreams. WS-G (verification gates + rotation strategy) wraps
everything.

**Target:** rc14 bootstrap produces ≤ 5% quarantine rate, description
coverage ≥ 90%, reviewer score ≥ 0.80.

---

## 2. Dependencies and non-goals

**Hard dependencies:**

1. **Plan 30 merged** — `NodeCategory` enum adds `fit_artifact` and
   `representation`. Without this, WS-A's classifier exclusions have
   nowhere to route the excluded paths.
2. **Plan 29 merged** — split `sn generate` (names-only) + standalone
   `sn enrich` with POSTLINK. Without this, descriptions stay at 11.9%
   coverage and any quality uplift is a one-shot rewrite.
3. **ISN rc14 release** (from `~/Code/imas-standard-names`) — pins the
   new grammar vocabulary tokens. WS-A and WS-B use this version.

**Non-goals:**

- Tensor rank≥2 grammar (R1-D4) — deferred.
- `basis_frame` segment (R1-F5) — deferred.
- Rebuilding FacilitySignal MEASURES relations — separate rotation.
- Migrating `review_status='drafted'` consumers — covered by plan 29.

---

## 3. Workstream A — Classifier / Source-path exclusions

**Owner repo:** imas-codex
**Files:** `imas_codex/standard_names/classifier.py`,
`imas_codex/standard_names/source_paths.py`,
`imas_codex/schemas/imas_dd.yaml` (plan 30 extension),
`imas_codex/core/node_classifier.py`,
`imas_codex/standard_names/audits.py`
**Dependencies:** plan 30 merged.
**Parallelism unit:** single agent (deep classifier change).

### A.1 — Extend plan 30 NodeCategory enum (optional: P0)

Add a third value (or extend `fit_artifact`) to cover
`transport_solver_numerics/boundary_conditions_*` and
`transport_solver_numerics/solver_1d/*/coefficient*`. Recommend folding
into `fit_artifact` with broadened description — these are solver
internals and share the "not-an-independent-physics-concept" property.

```yaml
fit_artifact:
  description: >-
    Artefact of a fitting, reconstruction, or numerical-solver process
    — not an independent physics concept. Includes fit diagnostics
    (chi², residual, covariance, fitting_weight), per-fit provenance
    (reconstructed, measured, weight, time_measurement, rho_tor_norm),
    and transport-solver boundary-condition / coefficient configuration
    nodes under transport_solver_numerics/boundary_conditions_*/ and
    transport_solver_numerics/solver_1d/*/coefficient*.
```

Regenerate via `uv run build-models --force`. Update `classify_node_pass1`
rules to include the new path patterns.

### A.2 — Pulse-schedule reference exclusion

In `source_paths.py` (or wherever the SN extractor filters DD paths) add:

```python
PULSE_SCHEDULE_REFERENCE = re.compile(
    r"^pulse_schedule/.+/reference(_waveform)?(/.+)?$"
)
```

Any DD path matching this pattern is:
- Skipped by the SN compose queue (not offered for naming).
- Marked `NodeCategory=representation` in the graph (reuse plan 30
  category since the semantic is the same: "stored representation of
  another quantity, not a physics concept in its own right").

### A.3 — Diamagnetic-axis veto

Vector-container paths with `/diamagnetic` leaf are DD mis-labelings
(review §2.3). Add exclusion for paths matching:

```python
DIAMAGNETIC_AXIS = re.compile(
    r"^(.*)/(velocity|e_field|a_field|j_tot|b_field)/diamagnetic(/.+)?$"
)
```

These get the same `representation` routing as pulse_schedule.
Emit a tracking note pointing to DD-01 filing (§8) so the exclusion can
be removed when DD fixes the paths.

### A.4 — Missed representation suffixes in audit regex

File: `standard_names/audits.py`,
`representation_artifact_check` (line ~ look-up name at invocation).

Extend the regex alternation to catch:
- `_ggd_coefficients$`
- `_coefficient_on_ggd$`
- `_interpolation_coefficient(s?)(_on_ggd)?$`
- `_finite_element_coefficients_(real|imaginary)_part$`
- `_on_ggd$` (**heuristic** — only fire when DD source path includes
  `/ggd/` or `/grids_ggd/`; avoid false-positives on legitimate
  `_on_<other>` suffixes).

### A.5 — Acceptance criteria for WS-A

- `uv run pytest tests/standard_names/test_classifier.py -v` passes.
- Re-running `uv run imas-codex sn extract` on the rc13 corpus produces:
  - 0 paths matching `pulse_schedule/.*/reference.*`.
  - 0 paths matching `.*/ggd/.*/coefficients$`.
  - 0 paths matching `transport_solver_numerics/boundary_conditions_.*/value$`.
  - 0 paths ending in `/diamagnetic` under vector containers.
- Audit log shows zero `representation_artifact_check` hits on the
  re-composed batch.

### A.6 — Commit plan

1. Schema edit + `uv run build-models --force`.
2. Classifier rules + tests.
3. Audit regex extension + tests.
4. Lint: `uv run ruff check --fix . && uv run ruff format .`.
5. Commit: `feat(sn): extend classifier to veto pulse_schedule/reference and diamagnetic axis paths`.

---

## 4. Workstream B — Prompt & exemplar cleanup

**Owner repo:** imas-codex
**Files:**
- `imas_codex/llm/prompts/sn/compose_system.md`
- `imas_codex/llm/prompts/shared/sn/_exemplars_name_only.md`
- `imas_codex/llm/prompts/shared/sn/_grammar_reference.md`
- `imas_codex/llm/prompts/sn/compose_dd.md`
- `imas_codex/llm/prompts/sn/compose_signals.md`
- `imas_codex/llm/prompt_loader.py`

**Dependencies:** none (parallel to WS-A).
**Parallelism unit:** single agent (one prompt file at a time to preserve
git history; 4 logical commits).

### B.1 — Exemplar rewrite (P3, P5, P8)

Resolve direct contradictions with compose_system.md Rules 17/18/21.

**P3 rewrite (old `_at_` unconditional):**
```markdown
## P3 — Position qualifiers: `of_` vs `at_` (semantic match)

Positional qualifiers distinguish intrinsic geometric properties from
field values evaluated at a locus.

- **`of_<position>`** — intrinsic geometric property of a locus.
  GOOD: `major_radius_of_magnetic_axis`, `area_of_plasma_boundary`,
       `vertical_coordinate_of_x_point`.
- **`at_<position>`** — value of a field quantity sampled at a point,
  line, or surface.
  GOOD: `electron_temperature_at_magnetic_axis`,
       `poloidal_magnetic_flux_at_plasma_boundary`,
       `safety_factor_at_minimum_safety_factor`.

FORBIDDEN: `at_` for a geometric coordinate (wrong preposition).
   BAD: `major_radius_at_magnetic_axis` → use `major_radius_of_magnetic_axis`.
```

**P8 rewrite (align with Rule 17):**
```markdown
## P8 — Poloidal-plane coordinates

- `vertical_coordinate_of_<position>` (preferred; Rule 17).
- `major_radius_of_<position>` (preferred; Rule 17).
- `toroidal_angle_of_<position>` (preferred; Rule 17).

DEPRECATED: `vertical_position_of_<position>` (old form;
    `vertical_coordinate_` is the canonical Z coordinate segment).
```

**P5 review:** inspect for similar drift; align with `positions.yml`.

### B.2 — Transformation enum: delete static list, inject live

**File:** `_grammar_reference.md` lines 115–125.

Delete the 4-token "canonical transformation" list. Replace with:

```markdown
## Transformations (live from imas-standard-names)

The full list of allowed transformation tokens is the
`Transformation` enum in the installed ISN grammar package. Prompt
template renders this list dynamically via `prompt_loader.render_prompt`
with `{{ transformations }}` context variable.

Always use ONLY tokens from this list. Do not invent new transformation
tokens.
```

**Prompt loader change:** `render_prompt` for `compose_system.md` must
inject `transformations` from
`imas_standard_names.grammar.Transformation` enum. Write a
`_context_for_compose()` helper if one does not exist.

### B.3 — Anti-pattern exemplars

Add to `_exemplars_name_only.md` a FORBIDDEN PATTERNS section:

```markdown
## Forbidden patterns (anti-exemplars)

1. `due_to_<adjective>` — always use the process noun.
   BAD: `due_to_halo`, `due_to_ohmic`, `due_to_fast_ion`, `due_to_non_inductive`.
   GOOD: `due_to_halo_currents`, `due_to_ohmic_dissipation`,
         `due_to_fast_ions`, `due_to_non_inductive_drive`.

2. `_over_<quantity>` as a division surrogate.
   BAD: `ion_velocity_over_magnetic_field_strength`.
   GOOD: `ion_velocity_per_magnetic_field_strength`.
   Note: `over_<region>` (e.g. `over_halo_region`) is the valid Region
   segment.

3. `_ggd_coefficients`, `_finite_element_interpolation_coefficients_on_ggd`,
   `_on_ggd`, `_coefficient_on_ggd` — basis-function storage, not
   physics. Classifier excludes these; LLM must never propose them.

4. `_reference_waveform`, `_reference` on pulse_schedule paths —
   controller setpoints. Classifier excludes; LLM must never propose.

5. `diamagnetic_component_of_<vector>` — the diamagnetic drift is a
   vector quantity (`v_dia = (B × ∇p) / (q n B²)`), not a spatial axis.
   If you need its projection, first name the drift:
   `ion_diamagnetic_drift_velocity`; then project:
   `radial_component_of_ion_diamagnetic_drift_velocity`.

6. Duplicate-subject splitting on compound species.
   BAD: `deuterium_tritium_*` interpreted as two subjects.
   GOOD: treat `deuterium_tritium`, `deuterium_deuterium`,
         `tritium_tritium` as single compound-subject tokens.
```

### B.4 — Rule alignment

**compose_system.md:**

- Revise lines 54–56 (`_over_`) to cross-reference the Region segment —
  see B.3 §2.
- Demote Rule 22 (`diamagnetic_component_*`) from "must not generate" to
  an informational note — classifier vetoes before compose sees the
  path, so the rule is redundant but the reminder is useful.
- Add a new NC rule:
  *"NC-27: Compound-subject tokens (`deuterium_tritium`,
  `deuterium_deuterium`, `tritium_tritium`) are single tokens in ISN
  `subjects.yml`. Never decompose them into two subjects."*
- Add a new NC rule:
  *"NC-28: The suffix `_reference_waveform` denotes a controller
  setpoint, not a physics quantity. The SN extractor excludes these DD
  paths — do not propose names matching this pattern."*

### B.5 — compose_dd.md / compose_signals.md parity audit

Not reviewed in depth during the rc13 review. Agent runs a diff against
`compose_system.md` and ports any rule/example updates.

### B.6 — Acceptance criteria for WS-B

- `uv run pytest tests/llm/test_prompt_rendering.py` passes (add a test
  that the rendered compose_system.md contains the live transformations
  enum).
- Spot-check: render the prompt and grep for
  `vertical_position_of_x_point` → **zero hits**.
- Grep rendered prompt for `at_magnetic_axis` → appears in a GOOD
  context (field-at-locus) AND in a documented semantic contrast, never
  in a FORBIDDEN-only context.
- Lint/format + `uv run pytest`.

### B.7 — Commit plan

One commit per file: `docs(prompts): exemplar P3/P8 semantic split`,
`docs(prompts): live transformation enum`, `docs(prompts): NC-27/28
compound subject + reference_waveform`, `docs(prompts): anti-pattern
exemplars`.

---

## 5. Workstream C — Audit bug-fixes and additions

**Owner repo:** imas-codex
**Files:** `imas_codex/standard_names/audits.py`,
`tests/standard_names/test_audits.py`
**Dependencies:** none (parallel to WS-A, WS-B).
**Parallelism unit:** single agent; 6 logical commits (one per audit).

### C.1 — `multi_subject_check`: longest-match first

Current implementation splits the name on `_` and checks each token
against `subjects.yml`. Change to greedy longest-match so
`deuterium_tritium` binds as a single subject before `deuterium` +
`tritium` are considered. Add test cases:

```python
@pytest.mark.parametrize("name,expected_passes", [
    ("deuterium_tritium_fusion_power_density", True),
    ("tritium_to_deuterium_density_ratio", True),  # ratio exception, see C.5
    ("electron_deuterium_density", False),  # genuine multi-subject
])
```

### C.2 — `representation_artifact_check` extensions

See WS-A §A.4. Add regex alternations; add test cases for each new
pattern.

### C.3 — `density_unit_consistency_check` timestamp exception

Fires on `toroidal_current_density_constraint_measurement_time` because
name contains `density` but unit is `s`. The name ends in
`_constraint_measurement_time`; the density token refers to the
underlying constraint, not the quantity being measured.

**Fix:** skip the check when the name ends in
`_constraint_measurement_time`, `_constraint_weight`,
`_constraint_reconstructed`, `_constraint_measured`,
`_constraint_time_measurement`, `_constraint_position`. In rc14 these
are all classifier-excluded by plan 30, so this fix is defensive.

### C.4 — `implicit_field_check` device whitelist

Fires false-positive on `vacuum_toroidal_field_function` and on
`*_field_coil*` (where "field coil" is a device name). Add a whitelist:

```python
FIELD_DEVICE_WHITELIST = {
    "vacuum_toroidal_field_function",
    "resistance_of_poloidal_field_coil",
}
# plus any name containing "_field_coil" as a substring
```

### C.5 — `causal_due_to_check` adjective-to-noun map

Add canonical replacements to the error message:

```python
ADJECTIVE_TO_PROCESS = {
    "halo": "halo_currents",
    "ohmic": "ohmic_dissipation",
    "fast_ion": "fast_ions",  # or "fast_ion_slowing_down" in some contexts
    "non_inductive": "non_inductive_drive",
    "thermal": "thermal_fusion",  # when next to deuterium_*_fusion
}
```

Raise `AuditIssue` with a `suggested_fix=<replacement>` field.

### C.6 — New audit: `pulse_schedule_reference_check`

Complements WS-A §A.2 (classifier). Any name whose DD source path
matches `pulse_schedule/.+/reference(_waveform)?` or whose name ends in
`_reference` / `_reference_waveform` is emitted with
`severity=critical`. Reason code:
`"controller reference target; not a physics SN candidate"`.

### C.7 — New audit: `ratio_binary_operator_check` (ratio exception)

Accepts the `ratio_of_<A>_to_<B>` canonical form (ISN-10; WS-D). Rejects
the ad-hoc `<A>_to_<B>_density_ratio` form with a suggested rewrite.

### C.8 — `unit_validity_check` strengthening

Reject any unit string containing whitespace (`"Elementary Charge
Unit"`) or `^dimension` placeholder (`"m^dimension"`). Emit
`dd_upstream` severity tag so the issue threads through to DD filings.

### C.9 — Acceptance criteria for WS-C

- All new/modified checks have pytest coverage.
- `uv run pytest tests/standard_names/test_audits.py` passes.
- Re-run audits on rc13 corpus: 0 `multi_subject_check` false-positives on
  `deuterium_tritium_*`; 0 `density_unit_consistency_check` false-positives
  on `_constraint_measurement_time`; 0 `implicit_field_check`
  false-positives on `vacuum_toroidal_field_function` /
  `*_field_coil*`.

### C.10 — Commit plan

Six commits: `fix(audits): compound-subject greedy match`,
`fix(audits): representation artifact suffix coverage`,
`fix(audits): density check skip constraint-metadata suffixes`,
`fix(audits): implicit-field device whitelist`,
`feat(audits): pulse_schedule_reference_check`,
`feat(audits): ratio_binary_operator and unit validity`.

---

## 6. Workstream D — ISN grammar vocabulary additions (iter-standard-names)

**Owner repo:** iter-standard-names
(`~/Code/imas-standard-names/imas_standard_names/grammar/vocabularies/*.yml`)
**Files:** `processes.yml`, `subjects.yml`, `positions.yml`,
`components.yml`, `transformations.yml`, `binary_operators.yml`,
plus parser tests under `tests/grammar/*`.
**Dependencies:** none (parallel to WS-A..C).
**Parallelism unit:** single agent; rc14 release bump.

### D.1 — `processes.yml` additions

```yaml
# === rc14 additions (from imas-codex plan 31, rc13 review) ===
- e_cross_b_drift               # E×B drift mechanism (plasma_control, edge)
- thermal_fusion                # pure-thermal fusion (alongside beam_beam_, beam_thermal_)
- thermalization_of_fast_particles  # distribution-source thermalisation
- halo_currents                 # noun form for due_to_halo_currents
- fast_ions                     # plural noun form for due_to_fast_ions
- non_inductive_current_drive   # explicit synonym alongside non_inductive_drive
```

Decision point for agent: either add these as bare tokens OR extend
`thermalization` with an argument-template accepting `_of_<subject>`.
The simpler approach (bare tokens) is preferred for rc14; the
template-based approach can be explored in a later release.

### D.2 — `subjects.yml` additions

```yaml
# === rc14 additions ===
- suprathermal_electrons        # population tier (ECRH, LH-driven)
- thermal_electron              # (if not already present) explicit thermal population
- thermal_ion                   # (if not already present) explicit thermal population
```

Verify whether `thermal_electron` / `thermal_ion` are already accepted via
the `thermal_` qualifier decomposition. If yes, only `suprathermal_electrons`
is new.

### D.3 — `positions.yml` additions

```yaml
# === rc14 additions ===
- ferritic_element_centroid     # (synonym for ferritic_insert_centroid; DD ferritic/element/*)
- neutron_detector              # position mirror (was Objects-only)
- measurement_position          # for distance_between_measurement_position_and_separatrix_*
```

### D.4 — `components.yml` additions / removals

```yaml
# === rc14 additions ===
- normalized_parallel           # next-rotation: normalized_parallel_refractive_index
- normalized_perpendicular      # symmetry complement

# === rc14 removals (mislabeled) ===
# - translation                 # REMOVED: not a spatial axis; use Process/Transformation
```

### D.5 — `transformations.yml` additions

```yaml
# === rc14 additions ===
- electron_equivalent           # gas-injection counts measured in ionization electrons
- ratio_of                      # canonical form: ratio_of_<A>_to_<B>
```

### D.6 — Component/coordinate-prefix split (R1-F6)

This is the structural ISN change that eliminates pydantic
`scalar.name` errors on `major_radius_of_*`,
`vertical_coordinate_of_*`, `toroidal_angle_of_*` names. Scope:

- Add a new grammar segment `coordinate_prefix` with vocabulary file
  `coordinate_prefixes.yml`:
  - `major_radius_of`, `minor_radius_of`, `vertical_coordinate_of`,
    `toroidal_angle_of`, `poloidal_angle_of`,
    `normalized_toroidal_flux_coordinate_of`,
    `normalized_poloidal_flux_coordinate_of`.
- Parser precedence: coordinate_prefix binds before Component segment.
- Tests: compose of
  `major_radius_of_outline_point`,
  `vertical_coordinate_of_control_surface_normal_vector`,
  `toroidal_angle_of_outline_point` must parse without error.

**Scope boundary:** this is a larger refactor. If the agent assesses it
as too risky for rc14, document WS-D §D.6 as rc15 and land only §D.1–D.5
in rc14. Codex's AUD-04 covers the interim.

### D.7 — Parser error-message improvements

**File:** `imas_standard_names/grammar/parse_error.py` (or equivalent).

When `diamagnetic_component_of_X` is encountered, emit:

```
`diamagnetic` is a drift qualifier, not a spatial axis.
Did you mean: radial_component_of_<subject>_diamagnetic_drift_velocity?
```

When `<A>_to_<B>_density_ratio` is encountered:

```
Ad-hoc ratio compound. Use canonical form:
ratio_of_<A>_density_to_<B>_density
```

### D.8 — Acceptance criteria for WS-D

- `pytest` in iter-standard-names green.
- `rc14` version tagged and released.
- `imas-codex` updates its `pyproject.toml` pin to the new rc14 version.
- Parser accepts all 6 quarantined coordinate-prefix names from review §2.16
  (gated by D.6; otherwise AUD-04 handles the interim).

### D.9 — Commit plan (iter-standard-names)

Squash into one rc14 commit:
`feat(grammar): rc14 vocabulary extensions (codex plan 31)`.
Bump version in `pyproject.toml`. Release note references this plan.

---

## 7. Workstream E — IMAS DD upstream filings

**Owner repo:** IMAS Data Dictionary (external — upstream)
**Files (codex side):** `plans/research/standard-names/dd-issues-rc14/`
(optional: a small tracking directory with one .md per filed issue).
**Dependencies:** none (fully parallel; asynchronous external work).
**Parallelism unit:** one agent drafts all filings; a human submits them.

### E.1 — Draft each filing

For each of DD-01 … DD-06 (review §2.13 / §4.3), produce a short
markdown brief with:
- DD path(s) or pattern.
- Concrete evidence (quarantined SN names from rc13).
- Proposed fix.
- Impact on downstream (codex SN extractor, pydantic audits).

### E.2 — Acceptance criteria

- All 6 briefs drafted in `plans/research/standard-names/dd-issues-rc14/`.
- Links recorded in plan 31 §12 once each upstream issue is opened.

### E.3 — Commit plan

One commit per brief; PR title
`docs(sn): DD-<NN> filing brief — <short title>`.

---

## 8. Workstream F — Graph remediation (post-WS-A/B/C/D)

**Owner repo:** imas-codex
**Files:** `imas_codex/graph/sn_cleanup.py` (new),
`imas_codex/cli/sn.py` (new `sn cleanup` subcommand),
`scripts/remediate_rc13_to_rc14.py` (one-off migration).
**Dependencies:** WS-A, WS-B, WS-C merged; ISN rc14 released (WS-D D.1–D.5).
**Parallelism unit:** single agent (graph ops are serial).

### F.1 — Purge step

Delete StandardName nodes with status `quarantined` that match any of:
- `representation_artifact` audit issue.
- `fit_artifact` audit issue (once plan 30 has classified them).
- DD source path matches the WS-A exclusion patterns.

Expected deletion: **60 ± 10 nodes** (conservative estimate based on
rc13 review §2.1 + §2.2 + §2.8).

### F.2 — Re-compose surviving quarantines

Remaining ~30 quarantined names (after §F.1) are candidates for
re-compose with the rc14 prompt. Re-run `sn generate` on just those DD
source paths with `--recompose-quarantined` flag (add the flag if not
present).

### F.3 — Consolidation — `wave_absorbed_power` family

Collapse the 23-name family (review §2.10 Family A) into:
- Parent: `wave_absorbed_power` (kind=vector).
- Axis child SNs linked via `PART_OF`:
  - `electron_wave_absorbed_power`, `ion_wave_absorbed_power` (species).
  - `thermal_electron_wave_absorbed_power`, `fast_electron_wave_absorbed_power`,
    `thermal_ion_wave_absorbed_power`, `fast_ion_wave_absorbed_power`
    (population tier).
  - `..._inside_flux_surface`, `..._at_beam_tracing_point` (surface/location).
  - `..._per_toroidal_mode` (spectral).

Migration: existing standalone SNs remain; add `PART_OF` and
`REFERENCES` relationships. No deletions at this stage.

### F.4 — Consolidation — metric tensor g_ij

Add parent `metric_tensor_component`; link 12 existing `g11`..`g33`
SNs via `PART_OF`. Document the variance/index axes in the parent's
description.

### F.5 — Consolidation — gas-injection segment order normalization

Canonical form: `<species>_accumulated_gas_injection_<count_type>`.
Rename 3 ordering variants (review §2.10 Family G) to this form; add
alias SNs (`accumulated_gas_injection_of_<species>`) via `REFERENCES`.

### F.6 — Near-duplicate relationship population

Add `NEAR_DUPLICATE_OF` edges for the 7 pairs enumerated in review
§2.12. Edge weight = semantic-cluster Jaccard × unit equality (0 or 1).
Enrichment rotation consumes these as cross-reference inputs.

### F.7 — Acceptance criteria for WS-F

- Quarantine count post-remediation: ≤ 35 (5% of 705).
- All `pulse_schedule/reference`-sourced SNs deleted.
- All `ggd/coefficients`-sourced SNs deleted.
- `wave_absorbed_power` family has a parent+23 `PART_OF` edges.
- No orphan SNs (every node has at least one `SOURCED_FROM` relationship).

### F.8 — Commit plan

One commit per step F.1–F.6. `scripts/remediate_rc13_to_rc14.py` is
idempotent and records its deletions to a `remediation.log.json` for
audit.

---

## 9. Workstream G — Verification gates & rotation strategy

**Owner repo:** imas-codex
**Files:** `tests/standard_names/test_corpus_health.py` (new),
`scripts/sn_corpus_stats.py` (new),
`plans/features/standard-names/31-quality-bootstrap-v2.md` (this file,
with final checkpoint results).
**Dependencies:** WS-F complete.
**Parallelism unit:** single agent.

### G.1 — Corpus-health test suite

New pytest module that runs against a fresh graph populated from the
rc14 bootstrap rotation. Assertions:

- Quarantine rate ≤ 5%.
- Description coverage ≥ 90%.
- Description length median ≥ 1200 chars.
- Reviewer score mean ≥ 0.80.
- No `pulse_schedule/.*/reference.*` paths in any `SOURCED_FROM` edge.
- No `*_ggd_coefficients` / `_finite_element_coefficients_*` name.
- No `diamagnetic_component_of_*` name.
- No pydantic coordinate_prefix errors.

### G.2 — Rotation script

`scripts/rc14_bootstrap.sh`:
```bash
uv run imas-codex sn extract --rebuild          # WS-A exclusions apply
uv run imas-codex sn generate --name-only       # plan 29 split
uv run imas-codex sn organize                   # canonicalize + vector hierarchy
uv run imas-codex sn enrich                     # plan 29 standalone enrich
uv run imas-codex sn embed                      # embeddings for names+descriptions
uv run imas-codex sn review --report            # generate review dump
uv run pytest tests/standard_names/test_corpus_health.py -v
```

### G.3 — Checkpoint dashboard

Add a `make sn-dashboard` target that prints:
- Total SNs / valid / quarantined (%).
- Per-domain quarantine rate (heat-map).
- Audit-issue histogram.
- Description coverage + mean length.
- Reviewer score distribution.

### G.4 — Acceptance criteria for WS-G

- Test suite passes.
- Dashboard script runs clean.
- Final rc14 review dump archived at
  `plans/research/standard-names/13-rc14-review-dump.md` (dated).
- Plan 31 checked off in `plans/INDEX.md`.

### G.5 — Rotation cadence

- rc14 targets complete by the next sprint boundary.
- Rotations every ~4 weeks; each rotation produces a review report
  numbered sequentially (12, 13, 14, ...).
- Exit criteria for "SN quality stable": 3 consecutive rotations with
  ≤ 2% quarantine, no new blocker class.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Plan 30 slips — WS-A cannot begin. | WS-B, WS-C, WS-D can proceed without plan 30. Gate only WS-A. |
| ISN rc14 release delayed — WS-D D.1–D.5 not available. | Codex pins to a local development branch of iter-standard-names via editable install; WS-A/B/C can still land. |
| Coordinate-prefix split (D.6) destabilizes parser. | Land it separately as rc15 behind a feature flag; use AUD-04 interim in rc14. |
| DD upstream filings rejected or slow. | Classifier vetoes (WS-A) are the defensive layer; DD fixes are nice-to-have. |
| Consolidation (F.3–F.5) breaks downstream consumers. | Maintain back-compat aliases for one release; emit deprecation warnings on old names. |
| Reviewer score regression after enrichment rotation. | Keep rc13 prompts snapshot; A/B-compare enriched-description reviewer scores. |

---

## 11. Success criteria (summary)

Copied from review §5 for convenience:

- Quarantine rate ≤ 5% (from 17.2%).
- Description coverage ≥ 90% (from 11.9%).
- Description length median ≥ 1200 chars.
- Reviewer score mean ≥ 0.80.
- Zero prompt/exemplar contradictions.
- Zero pydantic coordinate_prefix errors.
- ≥ 10 grammar vocabulary tokens added (ISN).
- ≥ 4 DD issues filed.

---

## 12. Cross-workstream DAG

```
plan 30 ────────────┐
                    ▼
                  WS-A (classifier) ─────────┐
                                             ▼
WS-B (prompts)    WS-C (audits)   WS-D (ISN) ─▶ WS-F (graph remediation)
    │                 │                │              │
    └─────────────────┴────────────────┤              │
                                       │              │
WS-E (DD filings; async, parallel)    │              ▼
                                       │          WS-G (verification + rotation)
                                       └────────────▶ rc14 review dump
```

Legend: `─▶` is hard dependency; boxed columns can run concurrently.

---

## 13. Agent dispatch plan (Opus pool)

| Agent # | Workstream | Agent type | Notes |
|---------|-----------|-----------|-------|
| 1 | WS-A (classifier) | architect | Deep classifier change; needs plan 30. |
| 2 | WS-B (prompts) | engineer | 4 files, clear edits; can start immediately. |
| 3 | WS-C (audits) | engineer | 6 commits, test-driven. |
| 4 | WS-D (ISN) | engineer | Vocabulary YAML edits + ISN rc bump. |
| 5 | WS-E (DD briefs) | general-purpose | Drafting only; human submits. |
| 6 | WS-F (graph) | architect | After 1–4 land. |
| 7 | WS-G (verification) | engineer | After 6 lands. |

Agents 2, 3, 4, 5 can start **in parallel immediately**. Agents 1 and 6
gate on predecessors. Agent 7 is the final gate.

---

*— End of 31-quality-bootstrap-v2.md*
