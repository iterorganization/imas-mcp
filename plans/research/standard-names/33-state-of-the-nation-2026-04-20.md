# SN Pipeline — State of the Nation (2026-04-20)

> Consolidated review triggered by user request after multi-day fleet rotations.
> Purpose: answer the seven concrete questions, signpost live research,
> and set the direction for the next work cycle.

## 1. Plan inventory — what's current, what's done

### Active feature plans (`plans/features/standard-names/`)

| Plan | Status | Comment |
|------|--------|---------|
| `29-architectural-pivot.md` | **Living roadmap** | Generate/enrich split shipped; F1–F4 ISN grammar fixes shipped in rc14/rc15; embedding backfill done. Some corpus-cleanup items still open (A.1 wave-absorbed family, D.1/D.2 DD-leaked names). |
| `31-quality-bootstrap-v2.md` | **Partially shipped** | WS-A (classifier) shipped as plan 30. WS-B/C/D (prompt contradictions, grammar, audit) shipped in rc14/rc15. WS-F (remediation) and WS-G (verification) run every rotation. Quarantine rate now 13% (185/1406) vs 17.2% baseline — target was ≤5%. |
| `32-extraction-prompt-overhaul.md` | **Phase 2 done, Phase 4 deferred** | A/B/C bake-off documented in `prompt-ab-results.md` — status quo (variant A) retained, variant C flagged for re-test with production cache block. `--until-complete` endpoint intentionally not implemented: running `sn generate` with no `-c` already runs to completion (user-confirmed direction). |

### Pending (reference-only, not work items)

| Plan | Recommendation |
|------|---------------|
| `pending/20-consistency-and-prompt-enrichment.md` | **File as superseded.** Its DD-enrichment, unit-safety, and naming-scope workstreams are all implemented (unit flows from `HAS_UNIT`, not LLM; DD context injected; node_category gates scope). |
| `pending/23-quality-parity.md` | **Keep as reference.** Phase 0 benchmark-harness concept still valid and partially in place (`scripts/prompt_ab_run.py`, `tests/standard_names/eval_sets/prompt_ab_v1.json`). Feeds §6 below. |

### Completed / superseded
`completed/` (14 items) is clean. `completed/superseded/` holds plans 27 + 28 + early 9/10/11.

### Live research worth reviewing
Signposted in priority order:

1. **`prompt-ab-results.md`** — the A/B test you asked for. Summary in §3.
2. **`11-quality-gap-analysis.md`** — ISN-catalog vs our output quality deltas; feeds benchmark design.
3. **`12-full-graph-review.md`** — the 706-name full audit that drove rc14.
4. **`isn-vocab-issues-2026-04-19.md`** — live list of ISN tokens we still need.
5. **`node-category-audit.md`** — classifier-filter audit (plan 32 WS-1).
6. **`gyrokinetics-extraction-debug.md`** — root-cause for the low-scoring gyro cluster.
7. **`dd-unit-bugs.md`** — documented DD-upstream unit issues (status quo, no filings per user).

## 2. Current graph state (2026-04-20, 16:30)

- **1,406 StandardName nodes** (up from 676 at session 11 baseline, +108 % in two days).
- 976 `valid` · 185 `quarantined` · 93 `needs_revision` · 88 unset · 63 `pending`.
- 478 scored ≥ 0.8 ("good"); 320 scored < 0.5 ("poor"); 905 scored total.
- 1,076 have both `description` and `documentation`.
- Sole compose model: `anthropic/claude-sonnet-4.6`.

### Per-domain quality (valid names only)

| Domain | n | avg score | Comment |
|--------|---:|----------:|---------|
| general | 68 | 0.98 | Reference-quality anchor |
| magnetic_field_systems | 13 | 0.97 | Narrow, well-bounded |
| particle_measurement_diag | 31 | 0.96 | ✓ |
| plasma_control | 25 | 0.95 | ✓ |
| transport | 224 | 0.91 | Biggest healthy cluster |
| equilibrium | 59 | 0.91 | ✓ |
| magnetohydrodynamics | 31 | 0.85 | ✓ |
| radiation_measurement_diag | 34 | 0.75 | Borderline |
| edge_plasma_physics | 61 | 0.66 | Rotation in-flight |
| computational_workflow | 17 | 0.65 | Meta-domain — expected |
| plant_systems | 13 | 0.60 | Low priority |
| auxiliary_heating | 72 | 0.57 | **Needs regen pass** |
| gyrokinetics | 51 | 0.37 | **Needs regen pass** (see debug doc) |
| plasma_wall_interactions | 50 | 0.34 | **Needs regen pass** |
| turbulence | 80 | 0.33 | **Needs regen pass** |
| electromagnetic_wave_diag | 48 | 0.24 | **Highest priority** |
| structural_components | 9 | 0.16 | Likely misdomained |
| fast_particles | 42 | 0.14 | **Highest priority** |
| waves | 7 | 0.08 | Near-worst; tiny sample |

## 3. A/B test results — you asked for these

Empirical run (`scripts/prompt_ab_run.py`, 20 paths × 3 variants,
opus-4.6, 2026-04-20). Full write-up in `prompt-ab-results.md`.

| Variant | Mean score | Pass@1 | Compose cost | Verdict |
|---------|-----------:|-------:|-------------:|---------|
| A (rich static, current default) | **0.805** | 0.75 | $0.042 | **Retained** |
| B (lean static, no siblings) | 0.755 | 0.65 | $0.036 | Rejected — loses 5 pp |
| C (lean + optional tool calls) | **0.815** | 0.75 | $0.037 | Flagged for re-test |

**Key finding:** quality is a statistical tie; all three variants fail
the same four "coordinate of named device" cases — a reviewer-side
controlled-vocabulary problem, not a prompt-quality problem.

**Open TODO:** re-run the harness with the **full production cache block**
(`compose_system.md`, 60 KB, cached). Expected production-A cost
≥ $0.08 per 20 paths — if confirmed, variant C passes the 50 % cost
gate and becomes the promoted default.

## 4. ISN grammar & vocab — is another round needed?

**Shipped in ISN rc14/rc15:**
- `Transformation` enum desync (F1)
- Fourier / toroidal-mode decomposition (F3)
- `over_` region-scoped integrals (F4)
- D5 senior-review expansion
- rc15 plasma-wall-interactions gap closure

**Outstanding gaps still firing (from graph `VocabGap` nodes):**
- Process: `diamagnetic_drift`, `diamagnetic_drift_velocity`
- Object: `neoclassical_tearing_mode`, `ion_state`, `ion_charge_state`,
  `secondary_separatrix`, `halo_current_area_end_point`,
  `strike_point`, `control_surface`, `geometric_axis`
- Object: RF launcher parts — `ec_launcher_mirror`,
  `lower_hybrid_antenna`, `electron_cyclotron_launcher_launching_position`,
  `beam_tracing_point`
- Qualifier: `per_toroidal_mode_number`, `per_toroidal_mode`,
  `inside_flux_surface`, `volume_averaged`, `vibrational_level`

**Recommendation:** Package an **rc16** bundle covering the object +
qualifier tokens before any further PWI/RF rotations. The
`diamagnetic_drift` process token has a live issue queue entry
already.

## 5. Grammar decomposition — are we persisting it?

**No — and this is a real gap.** `imas_standard_names.grammar.parse_standard_name`
returns a rich `StandardName` object with twelve segment fields
(`component`, `coordinate`, `subject`, `device`, `geometric_base`,
`physical_base`, `object`, `geometry`, `position`, `process`,
`transformation`, `binary_operator`, `secondary_base`). **None** of
these are written to the graph. The schema currently persists only
`id`, `unit`, `kind`, `physics_domain`, `tags`, `description`,
`documentation`, `source_paths`, etc.

**Consequence:** we cannot query "show every SN whose `physical_base`
is `temperature`" or "find radial-component names missing the
`radial_component_of_` prefix on equivalent vector-component
siblings". The R2 corpus-audit work used string-matching heuristics
precisely because the decomposition wasn't available.

**Proposed fix (schema change):**
```yaml
StandardName:
  attributes:
    grammar_component: {description: 'Component prefix (radial, toroidal, ...)', range: string}
    grammar_coordinate: {description: 'Coordinate qualifier', range: string}
    grammar_subject: {description: 'Particle subject (electron, ion, ...)', range: string}
    grammar_physical_base: {description: 'Core physical quantity', range: string}
    grammar_process: {description: 'Process / due_to_ token', range: string}
    grammar_transformation: {description: 'Transformation prefix', range: string}
    grammar_object: {description: 'Named physical object (separatrix, ...)', range: string}
    grammar_geometry: {description: 'Geometric qualifier (flux_surface_averaged, ...)', range: string}
    grammar_position: {description: 'Position qualifier (at_magnetic_axis, ...)', range: string}
    grammar_segments_hash: {description: 'Stable hash of the full segment tuple — for dedup', range: string}
```

Write-path: extend `persist_worker` to call `parse_standard_name()`
and `SET sn += grammar_fields` on the node. Existing names can be
back-filled by a one-shot loop. This unlocks:

1. **Base-duplication detection** — `MATCH (a)-(b) WHERE a.grammar_physical_base = b.grammar_physical_base AND ...`
2. **Auto-anti-pattern checks** — any node with a multi-token
   `physical_base` that should have been decomposed (e.g.
   `line_averaged_carbon_number_density` — the `line_averaged`
   qualifier leaked into the base).
3. **Benchmark coherence** — a benchmark set can assert per-segment
   equivalence without string matching.

## 6. Benchmark set — are we ready?

**Yes, partially.** `tests/standard_names/eval_sets/prompt_ab_v1.json`
is a 20-path stratified set (5 domains × 4 path kinds); the runner
(`scripts/prompt_ab_run.py`) works; reviewer scoring is calibrated
(`benchmark_calibration.yaml`, `sn_review_criteria.yaml`).

**What's missing before this is a production benchmark:**

1. **Holdout against the ISN catalog.** Pair ≥30 DD paths with
   known-good ISN catalog names (the list in plan 23 §0A is a
   solid seed: `psi → poloidal_magnetic_flux_profile`, `q →
   safety_factor`, etc.). Scoring layer: **exact-match bonus +
   reviewer score + grammar-segment overlap**.
2. **Two-reviewer consensus.** Single-reviewer scoring is
   self-reinforcing (see §7). Gate "published" examples on
   agreement between two different model families.
3. **Anti-pattern negatives.** ≥20 deliberately-bad names
   (`ip` → `ip`, `temperature` → `temp`, unit-mismatched,
   grammar-violating) to confirm the reviewer actually penalises
   them.

**Recommendation:** lift the benchmark work out of plan 23 and into a
dedicated plan `34-sn-benchmark-set.md` (new). Target deliverable:
a 50-path curated benchmark with pass criteria, runnable as
`imas-codex sn benchmark --set canonical`.

## 7. Reviewer model diversity — yes, we should rotate

Current state: reviewer is always the same model as compose
(`anthropic/claude-sonnet-4.6` in production; opus-4.6 in the A/B
harness). **This is a measurement anti-pattern.** A model scoring its
own output under the same rubric will overweight its own stylistic
preferences and underweight cross-family disagreements.

**Proposed reviewer pool (cost vs breadth):**

| Role | Model | Why |
|------|-------|-----|
| Primary reviewer | `anthropic/claude-opus-4.6` | Already calibrated; benchmark baseline |
| Cross-family reviewer | `google/gemini-3.1-pro-preview` | Independent tokenization + training data |
| Spot-check reviewer | `openai/gpt-5.4` | Third family; used on tier-boundary names only |

Apply as `sn review --reviewer-pool primary,cross-family` — score
recorded per reviewer; consolidated score = min(primary, cross).
Names flagged when reviewers disagree by > 0.2 get escalated for
manual inspection. This gives us a robust measurement instrument
and cleanly generalises to benchmarking future compose models.

**Implement as:** extend the `[sn.benchmark]` pyproject table with
`reviewer_pool` + a disagreement threshold; wire into
`standard_names/review/pipeline.py`.

## 8. Direction — robust checks vs prompt-led architecture

The honest answer: **both, in that order.**

The A/B test shows prompt density is not the bottleneck. Quality
gains come from:

1. **Structural filters** (classifier, unit safety, vocab gates) —
   already shipped, still the highest-leverage knobs.
2. **Grammar persistence** (§5) — cheap, unblocks whole categories
   of automated checks.
3. **Cross-family review** (§7) — the missing measurement layer.
4. **Prompt refinements** (variant C, exemplar curation) — marginal
   at this stage; prioritise after steps 1–3.

## 9. Concrete next steps

Ordered by leverage-per-effort:

1. **File plan 34 — Benchmark set v1** (this document's §6). Deliverable: 50-path curated set + runnable CLI.
2. **Ship grammar persistence** (this document's §5). Schema change + back-fill loop. Unblocks base-duplication and auto-anti-pattern queries.
3. **Ship cross-family review** (this document's §7). Config-level change; reuses existing review infra.
4. **Re-run A/B with production cache block** (plan 32 Phase 2 TODO). Confirms whether variant C is promoted.
5. **Package ISN rc16** with the object + qualifier tokens in §4. Blocker for PWI/RF domains reaching > 0.7 avg score.
6. **Continue regen rotations** on the four < 0.5 domains
   (fast_particles, waves, turbulence, plasma_wall_interactions) —
   in the existing `sn generate --below-score 0.6 --include-review-feedback` loop, unchanged.

## 10. Housekeeping

- `plans/README.md` is stale — references plan 28 as current (now superseded by 29). Updated alongside this doc.
- `pending/20-consistency-and-prompt-enrichment.md` is fully implemented. Moved to `completed/superseded/`.
- `pending/23-quality-parity.md` retained — its Phase 0 benchmark-harness design feeds plan 34.
