# 10: Standard Names Implementation Review

**Date:** 2026-04-08
**Scope:** Post-implementation review of Features 05–08
**Status:** Analysis complete — actionable findings

---

## 1. Catalog Assessment (309 Existing Names)

The imas-sn MCP server hosts **309 persisted standard names** — a substantial, high-quality catalog that serves as both the gold standard for benchmarking and the reference set for deduplication.

### Quality Evaluation

| Dimension | Assessment | Evidence |
|-----------|-----------|----------|
| Grammar validity | **Excellent** | 0 grammar errors, 0 schema errors across 309 entries |
| Documentation depth | **Outstanding** | Rich entries: LaTeX equations, governing physics, measurement methods, typical values across ITER/JET/DIII-D, sign conventions |
| Cross-referencing | **Good but fragile** | 810 forward-reference warnings — names linking to concepts not yet in catalog (e.g., `tokamak_scenario`, `gyrokinetic`) |
| Coverage breadth | **Strong core, gaps at edges** | 133 bare quantities, 32 subject-qualified, 40 device-qualified, 95 object-qualified, but only 3 component-qualified, 6 position-qualified, 0 process-qualified |
| Unit consistency | **Perfect** | 0 unit errors; proper SI notation (eV, m^-3, A, T, W.m^-2) |
| Provenance | **Empty** | All entries have empty `derived_from` — no DD path linking exists yet |

### Name Distribution by Grammar Pattern

```
Bare physical_base only:           133 (43%)  — temperature, safety_factor, time
Subject + physical_base:            32 (10%)  — electron_temperature, ion_density
Device + physical_base:             40 (13%)  — bolometer_radiated_power, flux_loop_...
Object (of_) qualified:             95 (31%)  — area_of_flux_loop, position_of_...
Component (vector) qualified:        3  (1%)  — toroidal_component_of_magnetic_field_...
Position (at_) qualified:            6  (2%)  — ..._at_magnetic_axis
Process (due_to_) qualified:         0  (0%)  — MISSING ENTIRELY
Binary operator compound:            0  (0%)  — MISSING ENTIRELY
Transformation (square_of, etc):     0  (0%)  — MISSING ENTIRELY
```

### Key Finding: Coverage Gaps

The catalog has **zero names** using:
- `due_to_<process>` pattern (e.g., `power_due_to_ohmic`, `current_due_to_bootstrap`)
- `binary_operator` compounds (e.g., `ratio_of_electron_pressure_to_magnetic_pressure`)
- `transformation` operators (e.g., `square_of_safety_factor`, `logarithm_of_collisionality`)
- `coordinate` prefix (geometric vector decomposition)

These grammar features exist in the grammar API (26 processes, 3 binary operators, 4 transformations) but have no catalog exemplars. This is critical for benchmarking — models cannot learn these patterns from examples.

### Name Quality Tiers (for benchmark reference)

**Outstanding** (rich documentation, correct grammar, cross-linked):
- `electron_temperature` — 500+ word doc, Spitzer formula, 6 cross-references, LaTeX
- `plasma_current` — integral equation, Rogowski coil methods, sign convention
- `safety_factor` — q-profile equation, stability boundaries, measurement methods
- `bootstrap_current` — neoclassical theory, 16 cross-references
- `position_of_magnetic_axis` — Shafranov shift, coordinate system, geometric

**Good** (correct grammar, adequate documentation):
- `toroidal_component_of_magnetic_field_at_magnetic_axis` — correct multi-field grammar
- `centroid_of_plasma_boundary` — geometric_base with geometry qualifier
- `bolometer_radiated_power` — device-qualified, calibration context
- `collisionality` — dimensionless quantity with physics context

**Adequate** (correct grammar, thin documentation):
- `area_of_poloidal_magnetic_field_probe` — 482 chars, no cross-references
- `tokamak_scenario` — metadata kind, no governing equation
- `time` — minimal documentation for fundamental coordinate

**Questionable** (grammar concerns or naming issues):
- `flux_surface_averaged_squared_toroidal_flux_coordinate_gradient_magnitude_divided_by_squared_magnetic_field_strength` — 113 characters as a single `physical_base`. Grammar API treats entire compound as open-vocabulary base rather than decomposing with transformations/operators. Valid but tests grammar extensibility limits.
- `h_mode`, `l_mode` — physics concepts rather than measurable quantities. `kind: metadata` is appropriate but no units.
- `banana_orbits`, `drift_waves` — conceptual entries, not quantities. Useful for cross-referencing but not measurable.

---

## 2. Implementation vs. Plan Compliance

### Feature 05: SN Build Pipeline ✅

| Deliverable | Status | Notes |
|------------|--------|-------|
| `sn/` module structure | ✅ Done | pipeline.py, workers.py, state.py, graph_ops.py, progress.py, models.py |
| `cli/sn.py` top-level group | ✅ Done | build, status, benchmark, publish commands |
| `llm/prompts/sn/` templates | ✅ Done | compose_dd.md, compose_signals.md |
| `sn/progress.py` display | ✅ Done | Uses StageDisplaySpec pattern |
| `sn/sources/` plugin system | ✅ Done | dd.py, signals.py, base.py |
| Graph schema updates | ⚠️ Partial | StandardName node exists but schema not in LinkML |
| End-to-end DD source | ✅ Done | 500 paths extracted for equilibrium IDS |

**Critical gap:** The compose worker uses a **heuristic keyword matcher** (`_extract_physical_base()`) that matches against 13 known bases. The LLM compose prompt exists (`compose_dd.md`) but is NOT wired into the compose worker. In non-dry-run mode, composition quality will be very poor — only bare `physical_base` names like "temperature" or "density" can be produced, with no subject/component/position qualification.

### Feature 06: Cross-Model Review ✅

| Deliverable | Status | Notes |
|------------|--------|-------|
| Review worker | ✅ Done | `review_worker()` in workers.py, batch processing |
| Review prompt template | ✅ Done | `sn/review.md` — comprehensive with grammar rules |
| Pydantic review models | ✅ Done | SNReviewVerdict, SNReviewItem, SNReviewBatch |
| Confidence tier classification | ✅ Done | In publish.py `confidence_tier()` |
| CLI options | ✅ Done | `--review-model`, `--skip-review` |
| Tests | ✅ Done | 19 tests covering accept/reject/revise paths |

**Strength:** Clean buffer separation (validated → reviewed → validate reads from reviewed), graceful degradation on LLM failure, intra-run dedup tracking.

### Feature 07: Benchmarking ⚠️ Needs Enhancement

| Deliverable | Status | Notes |
|------------|--------|-------|
| CLI command | ✅ Done | `sn benchmark --models X,Y` |
| Multi-model support | ✅ Done | Runs each model sequentially |
| Quality scoring vs reference | ✅ Done | precision/recall against REFERENCE_NAMES |
| Rich comparison table | ✅ Done | Names, Valid%, Fields%, Ref Match, Cost, Speed |
| JSON report export | ✅ Done | BenchmarkReport.to_json() / from_json() |
| Benchmark dataset | ⚠️ Weak | 30 hand-crafted names — misses 90% of grammar patterns |

**Opportunity:** The benchmark CLI is functional but the reference dataset is the wrong source of truth. The 309-entry catalog already exists and covers far more patterns. The benchmark should use the catalog as its gold standard, not 30 hand-picked entries.

### Feature 08: Publish ✅

| Deliverable | Status | Notes |
|------------|--------|-------|
| YAML generation | ✅ Done | `generate_catalog_files()` |
| PR creation | ✅ Done | `create_catalog_pr()` via `gh` CLI |
| Batching by IDS/domain | ✅ Done | `make_publish_batches()` |
| Confidence tier separation | ✅ Done | `confidence_tier()` — high/medium/low |
| Catalog dedup | ✅ Done | `check_catalog_duplicates()` |
| PR description template | ✅ Done | Summary table in PR body |
| Dry-run mode | ✅ Done | `--dry-run` flag |

---

## 3. Opportunity Gaps

### Gap 1: Compose Worker Has No LLM — CRITICAL

The compose worker in `workers.py` (lines 100-230) uses `_compose_single()` → `_extract_physical_base()` which is a 13-keyword heuristic. The Jinja2 prompt template `compose_dd.md` exists and is well-designed but is **not called**. This means:

- Running `sn build --source dd --ids equilibrium` (without `--dry-run`) will produce only bare names like "temperature", "density", "current" — missing subject, component, position, and all other qualifiers
- The review and validate phases work correctly but operate on garbage input
- The benchmark module correctly wires the LLM via `_run_model()` — so the compose prompt works, it's just not integrated into the pipeline worker

**Priority:** HIGH — this blocks any real standard name generation

### Gap 2: Benchmark Reference Set Should Draw from Catalog

The 30 entries in `benchmark_reference.py` are a hand-crafted subset covering:
- 6 simple physical bases (safety_factor, elongation, etc.)
- 6 subject-qualified (electron_temperature, ion_density, etc.)
- 5 component-qualified (j_tor, b_field_pol, etc.)
- 2 position-qualified
- 8 compound bases (plasma_current, loop_voltage, etc.)
- 3 geometric bases (minor_radius, major_radius, aspect_ratio)

Missing from reference:
- 0 device-qualified names (40 exist in catalog)
- 0 object-qualified names (95 exist in catalog)
- 0 process-qualified (due_to_) patterns
- 0 binary operator patterns
- 0 transformation patterns
- 0 coordinate (geometric vector) patterns

**Opportunity:** Use the 309-entry catalog as the authoritative reference. The benchmark should:
1. Load catalog entries via the imas-sn grammar API or MCP tools
2. Parse each catalog entry to extract grammar fields
3. Use these as the ground truth for precision/recall
4. Curate a **stratified** subset covering all grammar patterns at quality tiers

### Gap 3: Benchmark Needs Curated Labeled Examples for Prompt Injection

The user's key insight: the benchmark should not just measure pass/fail rates — it should **inject labeled examples** (poor → outstanding) into prompts and use an Opus 4.6 reviewer to evaluate how different models perform on nuanced quality dimensions.

**Proposed labeled example tiers:**

| Tier | Criteria | Example | Why This Tier |
|------|----------|---------|---------------|
| **Outstanding** | Correct grammar, rich documentation, cross-linked, precise units, appropriate kind | `electron_temperature` | Full physics context, Spitzer formula, measurement methods, typical values |
| **Good** | Correct grammar, adequate documentation, proper units | `toroidal_component_of_magnetic_field_at_magnetic_axis` | Multi-field grammar correct, but documentation could be richer |
| **Adequate** | Correct grammar, minimal documentation | `area_of_poloidal_magnetic_field_probe` | Grammar right but bare documentation, no cross-references |
| **Poor** | Grammar valid but naming questionable | `flux_surface_averaged_squared_toroidal_flux_coordinate_gradient_magnitude_divided_by_squared_magnetic_field_strength` | Should decompose with transformations rather than cramming into physical_base |
| **Conceptual** | Valid but not a measurable quantity | `banana_orbits`, `tokamak_operation`, `h_mode` | Physics concepts, not quantities — should use `kind: metadata` consistently |

**Benchmark workflow with Opus 4.6 reviewer:**
1. Extract DD paths (same as current)
2. Run each candidate model to generate names + documentation
3. Inject labeled examples into an Opus 4.6 review prompt:
   - "Here are examples at different quality levels: [Outstanding: ..., Good: ..., Poor: ...]"
   - "For each generated name, rate it on these dimensions and assign a tier"
4. Opus 4.6 returns per-name quality assessments with reasoning
5. Aggregate into per-model quality distributions

This transforms the benchmark from a binary "grammar valid/invalid" check into a **nuanced quality evaluation** that can distinguish between models producing technically valid but semantically poor names vs. models producing outstanding names.

### Gap 4: Model Name Format Issues

The benchmark CLI examples show `--models claude-sonnet-4,gpt-4o` but pyproject.toml uses `anthropic/claude-sonnet-4-6` format. The benchmark passes model strings directly to `acall_llm_structured()` without the `openrouter/` prefix required for cache_control preservation.

### Gap 5: No Signals Source Testing

The signals source (`sn/sources/signals.py`) exists but has no tests and has never been exercised end-to-end. The `compose_signals.md` prompt template exists but isn't wired into the worker either (same issue as compose_dd).

### Gap 6: StandardName Not in LinkML Schema

The `StandardName` node type is used in `graph_ops.py` but is not declared in `imas_codex/schemas/facility.yaml`. This means:
- No generated Pydantic model for StandardName
- No schema compliance tests
- No vector index management
- Properties are ad-hoc (written in Cypher SET statements)

### Gap 7: No Graph Write in Pipeline

The pipeline runs EXTRACT → COMPOSE → REVIEW → VALIDATE but never writes results to the graph. The `write_standard_names()` function exists in `graph_ops.py` but is never called from any worker. Generated names exist only in memory during the pipeline run and are lost when it completes.

---

## 4. Benchmark CLI Enhancement Strategy

The benchmark CLI (`sn benchmark`) is a **strength** if enhanced correctly. Rather than removing it, enhance it to become a comprehensive quality evaluation tool.

### Current Architecture (keep)
- `BenchmarkConfig` — model list, source/filter config, temperature
- `ModelResult` — per-model metrics (grammar valid, cost, speed, reference overlap)
- `BenchmarkReport` — aggregated results with JSON serialization
- `render_comparison_table()` — Rich table output
- Grammar validation (`validate_candidate()`) — round-trip parse/compose
- Reference comparison (`compare_to_reference()`) — precision/recall

### Enhancements Needed

#### A. Dynamic Reference from Catalog
Replace `REFERENCE_NAMES` dict with a function that loads from the imas-sn catalog:

```python
def load_catalog_reference(ids_filter=None, max_entries=100):
    """Load reference names from the imas-sn catalog.
    
    Returns dict mapping canonical_name → {name, grammar_fields, kind, unit, quality_tier}
    """
```

#### B. Stratified Quality Labels
Curate a quality label file (`benchmark_labels.yaml`) mapping names to quality tiers:

```yaml
outstanding:
  - electron_temperature
  - plasma_current
  - safety_factor
  - position_of_magnetic_axis
  - centroid_of_plasma_boundary
good:
  - toroidal_component_of_magnetic_field_at_magnetic_axis
  - bolometer_radiated_power
  - collisionality
adequate:
  - area_of_poloidal_magnetic_field_probe
  - tokamak_scenario
  - time
poor:
  - flux_surface_averaged_squared_toroidal_flux_coordinate_gradient_magnitude_divided_by_squared_magnetic_field_strength
conceptual:
  - banana_orbits
  - h_mode
  - tokamak_operation
```

#### C. Opus 4.6 Quality Reviewer
Add a `--reviewer-model` option that uses a frontier model to evaluate generated outputs:

```
imas-codex sn benchmark \
  --models anthropic/claude-sonnet-4-6,google/gemini-2.5-flash \
  --reviewer-model anthropic/claude-opus-4-6 \
  --ids equilibrium \
  --max-candidates 50
```

The reviewer model receives:
1. Grammar rules (same as compose prompt)
2. Labeled examples at each quality tier
3. The generated name + fields + documentation
4. Rubric: grammar correctness, semantic accuracy, naming conventions, documentation quality, unit consistency

Returns per-name quality tier assignment with reasoning.

#### D. Additional Metrics
- **Quality distribution**: % outstanding / good / adequate / poor per model
- **Grammar pattern coverage**: Does the model use subject, component, position, process, etc.?
- **Documentation richness**: Average doc length, equation count, cross-reference count
- **Naming consistency**: Same input → same output across temperature settings

---

## 5. Curated Benchmark Examples

### Outstanding Tier (5 examples)

These names demonstrate mastery of grammar, physics, documentation, and conventions:

1. **`electron_temperature`** — Subject-qualified scalar. Rich LaTeX documentation (Spitzer formula, thermal velocity equation). Typical values across 3 devices. 6+ cross-references. Unit: eV.

2. **`plasma_current`** — Bare compound base. Surface integral equation with proper notation. Sign convention documented. Measurement methods (Rogowski, equilibrium reconstruction). Unit: A.

3. **`safety_factor`** — Fundamental dimensionless quantity. Ratio definition with field line integrals. Stability boundary context (q>1, q>2). Unit: 1.

4. **`position_of_magnetic_axis`** — Geometric base (`position`) + geometry (`magnetic_axis`). Shafranov shift explained. Kind: vector. Unit: m.

5. **`bootstrap_current`** — Process-related physics. 16 cross-references (most in catalog). Neoclassical theory context. Unit: A.m^-2.

### Good Tier (5 examples)

Correct grammar, adequate physics, could improve documentation:

6. **`toroidal_component_of_magnetic_field_at_magnetic_axis`** — Component + physical_base + position. Multi-field grammar correctly composed. Documentation adequate but shorter.

7. **`centroid_of_plasma_boundary`** — Geometric base (`centroid`) + geometry (`plasma_boundary`). Bounding box formula. Kind: vector.

8. **`bolometer_radiated_power`** — Device-qualified. Calibration and measurement context. Unit: W.m^-2.

9. **`collisionality`** — Dimensionless parameter. Proper physics context linking to Coulomb collision theory.

10. **`outline_of_plasma_boundary`** — Geometric contour with parameterized equation. Kind: vector.

### Adequate Tier (5 examples)

Grammar correct, thin documentation, missing context:

11. **`area_of_poloidal_magnetic_field_probe`** — Object-qualified. Faraday's law reference but minimal. No typical values.

12. **`time`** — Fundamental coordinate. Very brief documentation for such an important quantity.

13. **`tokamak_scenario`** — Metadata kind. No governing equation (appropriate). Brief listing of scenario types.

14. **`neutron_activation_analysis`** — Measurement technique as a standard name. Good documentation but unusual — describes a technique rather than a quantity.

15. **`skin_current`** — Bare physical base. Minimal documentation for a nuanced concept.

### Poor/Questionable Tier (5 examples)

Grammar valid but naming approach debatable:

16. **`flux_surface_averaged_squared_toroidal_flux_coordinate_gradient_magnitude_divided_by_squared_magnetic_field_strength`** — 113 characters crammed into `physical_base`. Should use transformation operators (`square_of`) and binary operator (`ratio_of`) to decompose.

17. **`banana_orbits`** — Conceptual physics phenomenon, not a measurable quantity. `kind: metadata` but no clear use case for standard naming.

18. **`h_mode`** — Confinement regime concept. Documented well but not measurable. Should perhaps be in a separate conceptual namespace.

19. **`magnetic_field_probe_vertical_field`** — Ambiguous: is this `vertical_component_of_magnetic_field_of_probe` or `magnetic_field_measured_by_vertical_probe`? Grammar doesn't disambiguate.

20. **`hot_neutral_temperature_of_isotope`** — Valid grammar but unusual: `subject` should be `neutral` with `physical_base: temperature`. Instead uses compound physical_base.

---

## 6. Summary of Findings

### What Works Well
- 4-phase pipeline architecture (EXTRACT → COMPOSE → REVIEW → VALIDATE)
- Review phase with cross-model LLM and graceful degradation
- Publish module with YAML generation, batching, and dedup
- 87 tests across 3 test modules
- Rich progress display integration
- Existing catalog (309 entries) is high quality

### What Needs Work (Priority Order)
1. **Wire LLM into compose worker** — currently heuristic-only (CRITICAL)
2. **Upgrade benchmark reference** — use catalog, add quality tiers, add reviewer model
3. **Fix model name format** — add openrouter/ prefix handling
4. **Add StandardName to LinkML schema** — enable schema compliance
5. **Wire graph write into pipeline** — currently results are lost
6. **Test signals source** — untested code path
7. **Fill grammar pattern gaps** — catalog needs process, transformation, binary operator examples

### Benchmark Enhancement Path
The benchmark CLI is a **strength** if enhanced with:
- Catalog-sourced reference set (309 → dynamic)
- Stratified quality labels (outstanding → poor)
- Frontier model reviewer (Opus 4.6)
- Grammar pattern coverage metrics
- Documentation quality scoring

This transforms it from "grammar pass/fail counter" to "comprehensive model quality evaluator" — providing robust, actionable metrics for model selection.
