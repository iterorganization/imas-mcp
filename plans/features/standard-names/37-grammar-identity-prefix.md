# Plan 37 — Grammar Identity-Prefix Completion (ISN rc21)

> **Status**: DRAFT — awaiting RD review then fleet dispatch
> **Supersedes**: Plan 36 Phase 0 (component postfix inversion, never landed in ISN)
> **Extends**: Plan 36 Phase 0 with Option E geometry/object prefix-lift

---

## Problem

Plan 36 shipped its codex-side implementation (Waves 1–3 on main) but its **Phase 0
ISN grammar inversion never reached the ISN repository** — `imas-standard-names` is
still pinned at `v0.7.0rc20` where:

```python
SEGMENT_TEMPLATES = {
    "component": "{token}_component_of",   # PREFIX with preposition
    "object":    "of_{token}",             # SUFFIX with preposition
    "geometry":  "of_{token}",             # SUFFIX with preposition
    "position":  "at_{token}",             # SUFFIX with preposition
    "region":    "over_{token}",           # SUFFIX with preposition
    "process":   "due_to_{token}",         # SUFFIX with preposition
}
```

This produces split-brained output. From 35 valid equilibrium names generated after
plan 36 code landed:

| Prefix form (grammar-compliant) | Postfix form (ad-hoc, also accepted) |
|---|---|
| `radial_component_of_magnetic_field_perturbation_at_control_surface` | `perturbed_magnetic_field_radial_component` |
| `toroidal_component_of_magnetic_field_perturbation` | `perturbed_magnetic_field_normal_component_at_control_surface` |
| `minor_radius_of_plasma_boundary` | — |
| `elongation_of_plasma_boundary` | — |

The LLM is emitting **both** forms for the same concept because our prompt examples
and the underlying grammar disagree.

### Two issues, one grammar cut

1. **Issue A (Plan 36 Phase 0 carry-over)**: `component` and `reduction` modifiers
   are PREFIX with `_of_` preposition. They should be POSTFIX without preposition
   (`X_radial_component`, `X_magnitude`, `X_real_part`).
2. **Issue B (NEW — Option E)**: `object` and `geometry` are SUFFIX with `_of_`
   preposition. But they describe **identity** (whose quantity? on what shape?),
   not extrinsic context. They should be PREFIX without preposition, matching the
   existing `subject` and `device` segments which already work this way.

Fix them together in ISN rc21. No data migration — current StandardName graph is
cleared between regen rotations anyway.

---

## Decision

Ship ISN rc21 with **paired** grammar changes:

- **Component/reduction postfix** (strip prep, move right) — completes plan 36 Phase 0
- **Object/geometry prefix-lift** (strip prep, move left into identity zone) — Option E

### Rationale

The grammar becomes a clean three-zone structure:

```
[identity prefix zone: no preps]  [base noun]  [modifier postfix zone: no preps]  [context suffix zone: with preps]
```

| Zone | Purpose | Preposition? | Segments |
|---|---|---|---|
| Identity prefix | Who/what owns the quantity | No | `subject`, `device`/`object`, `geometry` |
| Base | The quantity itself | — | `geometric_base` or `physical_base` |
| Modifier postfix | Mathematical operation on the base | No | `coordinate` or `component`, `reduction` |
| Context suffix | Where/over/why the measurement is made | Yes (`at_`, `over_`, `due_to_`) | `position`, `region`, `process` |

### Semantic test — why `geometry` is identity, `position` is context

- **Geometry** = the shape the quantity belongs to (`plasma_boundary_minor_radius` —
  the boundary's minor radius; the boundary *is* the referenced entity).
- **Position** = a point/location where the measurement is evaluated (`magnetic_field_at_plasma_core` —
  magnetic field evaluated at a core point; the core is not what the field "belongs to").
- **Region** = a volume over which the quantity is distributed/integrated
  (`ion_density_over_scrape_off_layer` — density in a volume; "over" preserves meaning).
- **Process** = a physical cause (`current_drive_due_to_electron_cyclotron_heating` —
  "due to" preserves causal meaning).

Identity belongs to the subject-prefix zone; context belongs to the preposition-suffix zone.

---

## Canonical example — every non-exclusive segment in one name

A maximal name using all non-exclusive segments:

**Under current rc20 grammar:**

```
root_mean_square_of_radial_component_of_deuterium_magnetic_field_of_langmuir_probe_array_of_plasma_boundary_over_scrape_off_layer_due_to_ion_cyclotron_heating
└────reduction────┘ └───component───┘  └subject┘ └physical_base┘ └──────object──────┘  └──geometry───┘ └──────region──────┘ └────────process─────────┘
```

**Under proposed rc21 (Option E + Phase 0):**

```
deuterium_langmuir_probe_array_plasma_boundary_magnetic_field_radial_component_root_mean_square_over_scrape_off_layer_due_to_ion_cyclotron_heating
└─subject┘ └──────object──────┘ └──geometry───┘ └physical_base┘ └──component──┘ └──reduction───┘ └──────region──────┘ └────────process─────────┘
```

### Segment legend (rc21)

| # | Segment | Form | Template | Example token | Exclusive with |
|---|---|---|---|---|---|
| 1 | `subject` | Identity prefix | `{token}` | `deuterium` | — |
| 2 | `device` | Identity prefix | `{token}` | `tf_coil` | `object` |
| 2' | `object` | Identity prefix | `{token}` | `langmuir_probe_array` | `device` |
| 3 | `geometry` | Identity prefix | `{token}` | `plasma_boundary` | `position` |
| 4 | `geometric_base` | Base noun | `{token}` | `minor_radius` | `physical_base` |
| 4' | `physical_base` | Base noun | `{token}` | `magnetic_field` | `geometric_base` |
| 5 | `coordinate` | Modifier postfix | `_{token}_coordinate` | `radial` | `component` |
| 5' | `component` | Modifier postfix | `_{token}_component` | `radial` | `coordinate` |
| 6 | `reduction` | Modifier postfix | `_magnitude`, `_real_part`, `_amplitude`, `_phase`, `_modulus`, `_root_mean_square`, `_time_average`, `_volume_integral`, `_flux_surface_average`, `_radial_derivative`, `_time_derivative` | `_root_mean_square` | — |
| 7 | `position` | Context suffix (prep) | `_at_{token}` | `_at_outer_midplane` | `geometry` |
| 8 | `region` | Context suffix (prep) | `_over_{token}` | `_over_scrape_off_layer` | — |
| 9 | `process` | Context suffix (prep) | `_due_to_{token}` | `_due_to_ion_cyclotron_heating` | — |

### Worked shorter examples

| Name (rc21) | Segment decomposition |
|---|---|
| `plasma_boundary_minor_radius` | geometry + geometric_base |
| `plasma_boundary_elongation` | geometry + geometric_base |
| `plasma_boundary_poloidal_magnetic_flux` | geometry + physical_base (prefixed by `poloidal_`) |
| `active_limiter_point_major_radius` | object + geometric_base |
| `plasma_velocity_radial_component` | physical_base + component |
| `plasma_velocity_radial_component_at_plasma_core` | physical_base + component + position |
| `deuterium_ion_density_over_scrape_off_layer` | subject + physical_base + region |
| `electron_temperature_time_average` | physical_base + reduction |
| `electron_temperature_time_average_due_to_ohmic_heating` | physical_base + reduction + process |

---

## Phases

### Phase 0 — ISN rc21 grammar cut (BLOCKING)

**0a.** `imas_standard_names/grammar/constants.py`

- Reorder `SEGMENT_ORDER` — move `object` and `geometry` from positions 6–7 (suffix zone)
  to positions 3–4 (prefix zone, between `device` and `geometric_base`):

  ```python
  SEGMENT_ORDER = (
      "subject", "device", "object", "geometry",
      "geometric_base", "physical_base",
      "coordinate", "component", "reduction",
      "position", "region", "process",
  )
  ```

- Rewrite `SEGMENT_TEMPLATES`:

  ```python
  SEGMENT_TEMPLATES = {
      # Identity prefix — plain token, no preposition
      "object":   None,
      "geometry": None,
      # Modifier postfix — suffix token, no preposition
      "coordinate": "_{token}_coordinate",
      "component":  "_{token}_component",
      # Context suffix — suffix with preposition (unchanged semantics)
      "position": "_at_{token}",
      "region":   "_over_{token}",
      "process":  "_due_to_{token}",
  }
  # subject, device already None (no change)
  ```

- Update `BASE_SEGMENT_INDICES`, `PREFIX_SEGMENTS`, `SUFFIX_SEGMENTS`,
  `SUFFIX_SEGMENTS_REVERSED` to reflect new order.

**0b.** `imas_standard_names/grammar/parser.py` (or equivalent — locate and patch
round-trip)

- Prefix-zone parse: left-to-right, match against closed vocab for
  `subject`/`device`/`object`/`geometry`, strip token, continue.
- Base parse: match remaining head against `geometric_base` ∪ `physical_base` vocab.
- Modifier-zone parse: right-to-left, strip closed suffix tokens
  (`_{token}_component`, `_magnitude`, `_real_part`, etc.) before base.
- Context-zone parse: right-to-left, strip `_at_`, `_over_`, `_due_to_` markers.

**0c.** `imas_standard_names/reductions.py`

- Rewrite `REDUCTION_PATTERNS` from PREFIX to SUFFIX form. Canonical set:
  `_magnitude`, `_real_part`, `_imaginary_part`, `_amplitude`, `_phase`, `_modulus`,
  `_time_average`, `_volume_integral`, `_flux_surface_average`, `_root_mean_square`,
  `_radial_derivative`, `_time_derivative`.
- Suffix-strip order: longest-first to disambiguate (`_flux_surface_average` before
  any substring match).

**0d.** `imas_standard_names/models.py`

- `StandardNameVectorEntry.magnitude` property: return `f"{self.name}_magnitude"`.
- `StandardNameVectorEntry` component properties: return `f"{self.name}_{axis}_component"`.
- `StandardNameComplexEntry`: return `f"{self.name}_real_part"`,
  `f"{self.name}_imaginary_part"`, `f"{self.name}_amplitude"`, `f"{self.name}_phase"`,
  `f"{self.name}_modulus"`.

**0e.** `imas_standard_names/grammar/specification.yml`

- Update examples to postfix-modifier + identity-prefix form.
- Regenerate any derived documentation.

**0f.** Tests

- Update all grammar round-trip fixtures to new canonical form.
- Add new fixtures:
  - `plasma_boundary_minor_radius` (geometry + geometric_base)
  - `active_limiter_point_major_radius` (object + geometric_base)
  - `plasma_velocity_radial_component_at_plasma_core` (physical_base + component + position)
  - `deuterium_electron_temperature_volume_integral_over_scrape_off_layer` (subject + physical_base + reduction + region)
- Negative fixtures that MUST fail (old prefix forms):
  - `radial_component_of_plasma_velocity` (prefix component)
  - `minor_radius_of_plasma_boundary` (geometry in suffix zone)
  - `magnitude_of_plasma_velocity` (prefix reduction)

**0g.** Tag `v0.7.0rc21` on upstream `iterorganization/IMAS-Standard-Names`.

### Phase 1 — Codex pin + models rebuild

**1a.** Bump pin in `pyproject.toml` (two lines: main and dev extras) to
`@v0.7.0rc21`.

**1b.** `uv sync` locally; verify import works and generated models reflect new
grammar.

**1c.** `uv run build-models --force` to rebuild any LinkML-derived artifacts that
reference grammar context.

### Phase 2 — Codex grammar-dependent code

**2a.** `imas_codex/standard_names/audits.py`

- Invert `amplitude_of_prefix_check` (and any other `_of_` prefix audits) to accept
  postfix and reject prefix.
- Add `geometry_position_prefix_check` — verify `object`/`geometry` appear **before**
  the base, not in the suffix zone.
- Ensure `derived_part_parent_presence_check` still works under the new ordering
  (vector/complex parts are now postfix; parent existence check logic unchanged,
  just matches different suffixes).

**2b.** `imas_codex/llm/prompts/shared/sn/_grammar_reference.md`

- Rewrite the canonical segment table using the three-zone structure above.
- Include the single-name maximal example from this plan (both annotated forms).
- Add a "DO NOT USE" section showing the old prefix-with-`_of_` forms as anti-patterns.
- Include the worked shorter examples table.

**2c.** `imas_codex/llm/prompts/sn/compose_system.md`

- Update NC-19 (was "COMPONENT PRECEDES BASE") → invert to "COMPONENT FOLLOWS BASE"
  with postfix enforcement.
- Delete NC-20 (complex-suffix-specific rule) — unified under generic postfix.
- Add NC-21: "IDENTITY BEFORE BASE" — `subject`/`device`/`object`/`geometry` tokens
  always precede the base; `_of_` is not a valid grammar particle between them and
  the base.

**2d.** `imas_codex/llm/prompts/shared/sn/_enrich_style_guide.md`

- Update examples to match new grammar.

**2e.** `imas_codex/standard_names/kind_derivation.py`

- Inspect any `to_isn_kind` logic that looks at suffix patterns — update rules for
  new postfix reductions and new prefix geometry/object.

**2f.** `imas_codex/standard_names/source_paths.py` and any utilities that build
canonical names from DD fragments — update to emit prefix-form where they currently
emit suffix-with-`_of_`.

### Phase 3 — Regeneration validation loop

**3a.** Clear all StandardName + StandardNameSource nodes.

**3b.** Regenerate a canary batch (equilibrium, `--target names --limit 20 -c 0.50
--single-pass`). Verify:

- All valid names follow three-zone structure.
- No `_of_` appears in any generated name (except as part of a multi-token base
  noun — rare).
- All `plasma_boundary_*` names cluster alphabetically.
- Grammar round-trip passes for every valid name.

**3c.** If canary passes, regenerate at full budget across multiple domains. Verify
`sn review` tier distribution is stable (no catastrophic regressions vs. the 92%
valid rate achieved in session 14).

**3d.** If any quarantines are grammar-round-trip failures on valid-looking postfix
forms, file ISN rc22 vocab/grammar fix and loop.

### Phase 4 — Documentation

**4a.** `AGENTS.md` — update the grammar reference section (if present) to match rc21.

**4b.** `docs/architecture/standard-names.md` — update ordering diagram; call out
the three-zone structure as the canonical model.

**4c.** `docs/architecture/standard-names-decisions.md` — add ADR entry for
rc21 grammar cut (Issue A + Issue B rationale).

**4d.** Delete this plan once rc21 is pinned, regeneration validates, and all docs
reflect the new grammar.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Parser disambiguation failure (prefix-zone tokens collide with base vocab) | Closed vocab for each identity segment; longest-match-first order; fuzz test with all pairwise combinations of identity tokens and base tokens |
| Plan 36 post-compose audits fail on new form | Phase 2a explicitly inverts them; test suite must cover both the positive and negative cases before Phase 3 |
| Downstream tooling (`sn export`, catalog YAML consumers) breaks | Names are regenerated, so catalog rewrites from scratch. Any consumers hard-coding prefix forms will be caught at Phase 3 validation |
| ISN rc21 rejected upstream | This is a greenfield change before any names are published at rc20+plan-36-examples; cost of keeping rc20 is perpetual grammar inconsistency |
| Modifier postfix longest-match ambiguity (e.g. `_time_derivative` vs `_derivative`) | Maintain explicit ordered list in `reductions.py`, longest first; unit tests for each |

---

## Acceptance criteria

- [ ] ISN `v0.7.0rc21` tag exists on upstream with paired grammar change
- [ ] Codex pinned at rc21; `uv sync` clean; `build-models` green
- [ ] `tests/standard_names/` full suite green including new grammar fixtures
- [ ] Canary equilibrium regen: ≥90% valid rate, zero `_of_` in valid names (except
      multi-token base nouns), all `plasma_boundary_*` cluster together
- [ ] `sn review` on canary completes with per-dim scores populated
- [ ] `docs/architecture/standard-names*.md` reflects rc21 three-zone grammar

---

## Notes

- No data migration: the current graph's 35 valid names from session-14 will be
  wiped during Phase 3a. They were produced under rc20 + plan-36-codex mixed
  grammar, so they're not worth preserving.
- The Phase 0 ISN work is a focused single-PR-scope change in `imas-standard-names`.
  Dispatch to a separate agent with full write access to that repo.
- Plan 36 Phase 0 text can be quoted into rc21 PR description as prior art; the
  reasoning is identical, only scope expands to include Option E.
