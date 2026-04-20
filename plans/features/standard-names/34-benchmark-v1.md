# Plan 34 — Standard Name Benchmark Set v1

> **Origin:** `plans/research/standard-names/33-state-of-the-nation-2026-04-20.md` §6.
> **Supersedes seed:** `plans/research/standard-names/pending/23-quality-parity.md` Phase 0.
> **Status:** Scaffolding landed 2026-04-21.  Expansion to full 50 paths + consensus
> reviewer gate pending (`cycle-cross-family-review` + subsequent fills).

## 1. Problem

Today we measure Standard Name (SN) quality in three ad-hoc ways:

1. **`scripts/prompt_ab_run.py` + `prompt_ab_v1.json`** — 20 stratified DD
   paths across 5 domains × 4 path-kinds, no golden names, score only
   from a single-model reviewer.  Great for prompt A/B, useless as a
   **quality regression gate**: a degraded prompt that produced
   self-consistent garbage would still pass the reviewer.
2. **`imas_codex/standard_names/benchmark.py` / `benchmark_reference.py`** —
   per-model LLM comparison harness.  Runs end-to-end pipeline, but
   uses extracted DD paths with no curated ground truth beyond grammar
   round-trip.
3. **`benchmark_calibration.yaml`** — reviewer scoring anchors (outstanding
   / good / adequate / poor).  Scores *names in isolation*; does not
   test the compose→review pipeline against a fixed DD-path → SN
   ground truth.

The A/B results (`prompt-ab-results.md`) and the 706-name audit
(`12-full-graph-review.md`) show a single-reviewer measurement
instrument drifts: reviewers over-reward stylistic conformance and
under-penalise base-duplication / wrong-transformation errors that
only a second model family or a human catches.

**What is missing:** a *fixed*, *stratified*, *dual-reviewed*
benchmark that locks in expected (DD path → SN) pairs plus
deliberately-bad negatives, runnable before every prompt change or
pipeline refactor.

## 2. Goals

1. **Curated 50-path positive set** across the 10 physics domains that
   represent the mix of healthy and stressed clusters identified in
   plan 33 §2.
2. **≥20 anti-pattern negatives** — deliberately-bad candidate names
   covering each failure mode we've seen in the graph audit.
3. **Two-reviewer consensus** on every published example — primary
   (`anthropic/claude-opus-4.6`) + cross-family (Gemini / GPT pool).
   Depends on `cycle-cross-family-review`.
4. **Hard gating thresholds** the pipeline must pass on a clean run
   before any prompt/schema change is merged:
   - positives: **pass@1 ≥ 0.80** (exact SN match or grammar-equivalent)
   - positives: **mean reviewer score ≥ 0.75**
   - negatives: **rejection rate ≥ 0.90** (low score *and* validation
     flag)
5. **Runnable harness** that:
   - Loads a JSON eval set (positives + negatives)
   - Invokes the production compose prompt (not a rewritten variant)
   - Scores with the production reviewer(s)
   - Emits markdown + JSON summary with per-domain breakdown

## 3. Scope

### 3.1 Physics-domain coverage (positives)

Target 5 positives per domain × 10 domains = 50.

| Domain (PhysicsDomain enum) | Picked because … |
|------|------|
| `equilibrium` | floor — highest-quality cluster, 0.91 avg |
| `core_plasma_physics` | floor — ISN-anchored kinetic profiles |
| `magnetic_field_diagnostics` | floor — 0.97 avg, narrow vocab |
| `transport` | floor — biggest healthy cluster (0.91 avg) |
| `waves` | stress — tiny, 0.08 avg, ISN `rc16` blocked |
| `fast_particles` | stress — 0.14 avg, lowest-priority regen domain |
| `turbulence` | stress — 0.33 avg, qualifier-heavy |
| `plasma_wall_interactions` | stress — 0.34 avg, object vocab gap |
| `gyrokinetics` | stress — 0.37 avg, normalized-quantity semantics |
| `edge_plasma_physics` | stress — 0.66 avg, divertor-flux edge cases |

Each positive entry records `{dd_path, physics_domain, expected_name,
expected_unit, rationale, source}` where `source` is
`isn_catalog` (ground truth from ISN published examples),
`graph_published` (name currently in graph with reviewer_score ≥ 0.9),
or `synthetic` (hand-curated — lower trust until second-reviewer
consensus).

### 3.2 Anti-pattern negatives (≥ 20)

Each negative records `{candidate_name, dd_path?, physics_domain,
rejection_reason, anti_pattern_category}`.

Categories (must all be represented):

1. `base_duplication` — `poloidal_magnetic_magnetic_field_probe_voltage`
2. `missing_qualifier` — `elongation` (should be
   `elongation_of_plasma_boundary` since DD path is boundary-scoped)
3. `wrong_transformation` — `line_averaged_carbon_number_density`
   (qualifier leaked into physical_base)
4. `llm_supplied_unit` — `electron_temperature_ev`
5. `component_as_name` — `toroidal` for a toroidal-component path
6. `wrong_physics_domain` — `safety_factor` assigned to
   `machine_operations` instead of `equilibrium`
7. `symbol_leakage` — `t_e`, `Te`, `B_t`, `ip`
8. `ids_leakage` — `pulse_schedule_plasma_current`
9. `processing_adjective` — `reconstructed_plasma_boundary`,
   `filtered_electron_temperature`
10. `component_prefix_order` — `electron_diffusivity_poloidal`
    (grammar violation — component must precede physical_base)

### 3.3 Two-reviewer consensus

Publishable entries (status `published`) require agreement between two
reviewer families with |score_primary − score_cross| ≤ 0.2.  Entries
flagged `tentative` until a second reviewer endorses.  This depends on
`cycle-cross-family-review` landing `sn.benchmark.reviewer-pool` into
the runner.

### 3.4 Gating

Run as part of pre-merge CI once the set is full.  Gating logic:

```text
pass_at_1     = exact_match_count / positive_count         ≥ 0.80
mean_score    = mean(reviewer_score over positives)        ≥ 0.75
reject_rate   = flagged_negatives / total_negatives        ≥ 0.90
```

Failure on any of the three ⇒ red build.  Per-domain breakdown surfaced
to help diagnose which cluster regressed.

## 4. Deliverables

### 4.1 Data — `tests/standard_names/eval_sets/benchmark_v1.json`

Schema:

```json
{
  "_meta": { "version": "v1", "target_size": {"positives": 50, "negatives": 20} },
  "positives": [
    {
      "dd_path": "equilibrium/time_slice/profiles_1d/psi",
      "physics_domain": "equilibrium",
      "expected_name": "poloidal_magnetic_flux",
      "expected_unit": "Wb",
      "rationale": "ISN catalog reference; DD path maps 1:1.",
      "source": "isn_catalog"
    }
  ],
  "negatives": [
    {
      "candidate_name": "poloidal_magnetic_magnetic_field_probe_voltage",
      "dd_path": "equilibrium/time_slice/constraints/b_field_pol_probe/measured",
      "physics_domain": "equilibrium",
      "rejection_reason": "tautological duplication of 'magnetic'",
      "anti_pattern_category": "base_duplication"
    }
  ]
}
```

**Initial population (scaffolding):**
- 20 high-confidence positives drawn from the ISN catalog (`imas_standard_names`
  package `resources/standard_name_examples/*.yml`) paired with the
  `source_paths` currently recorded on the matching `StandardName` graph node.
- 10 anti-pattern negatives drawn from graph `StandardName` nodes
  with `confidence < 0.3` or `validation_status = 'quarantined'`
  (e.g. `poloidal_magnetic_magnetic_field_probe_voltage`,
  `electron_diffusivity_poloidal`).
- Remaining 30 positives stubbed with `{dd_path: "TODO", expected_name: "TODO"}`
  under the right `physics_domain` — future cycles fill them.

### 4.2 Runner — `scripts/run_benchmark_v1.py`

Design mirrors `scripts/prompt_ab_run.py`:

1. Load `benchmark_v1.json`.
2. Filter out `TODO` stubs (skip with note).
3. For positives: enrich DD context from graph (unit, description,
   cluster siblings — same helper used by `prompt_ab_run.py`).
   Invoke compose prompt in batch.  For each (expected, generated):
   compute `exact_match`, `grammar_equivalent` (via
   `parse_standard_name` round-trip), and reviewer score.
4. For negatives: hand the candidate + DD context directly to the
   reviewer prompt and confirm `score < 0.5` OR a validation flag
   (`base_duplication`, `wrong_transformation`, etc.) fires.
5. Emit `plans/research/data/benchmark-v1.positives.jsonl`,
   `benchmark-v1.negatives.jsonl`, `benchmark-v1.summary.json`, and
   `benchmark-v1-summary.md`.

CLI surface:

```bash
uv run python scripts/run_benchmark_v1.py \
  --eval-set tests/standard_names/eval_sets/benchmark_v1.json \
  --output plans/research/data \
  --cost-cap 5.00 \
  --limit 3          # optional: cap positive+negative count for dry-run
  --mock             # optional: skip LLM calls, exercise schema only
```

### 4.3 Documentation

- This plan (`34-benchmark-v1.md`).
- Dry-run log: `plans/research/standard-names/benchmark-v1-dryrun.md`.

### 4.4 CLI integration — deferred

Target: `uv run imas-codex sn benchmark --set v1` wrapping the script.
**Not in v1 scope** — AGENTS.md policy is no CLI changes unless
naturally required; script-level access is enough while we iterate.
Promote to CLI once the set is full and gating is wired into CI.

## 5. Dependencies

| Dep | Blocks | Status |
|-----|--------|--------|
| `cycle-cross-family-review` | §3.3 two-reviewer consensus; §4.2 reviewer pool | pending |
| ISN rc16 vocab (plan 33 §4) | ~10 positives in waves/PWI/RF domains | pending |
| Grammar persistence (plan 33 §5) | `grammar_equivalent` scoring in §4.2 step 3 | pending |

The v1 scaffolding lands *before* these are complete.  Positives requiring
new vocab are left as `TODO` stubs.

## 6. Non-goals

- **No new CLI subcommand in v1.**  Script only.
- **No auto-repair loop.**  Benchmark is read-only; regen flows live
  under plan 31's bootstrap workstreams.
- **No model ranking.**  `benchmark.py` already compares models; this
  set is a *quality floor*, not a leaderboard.
- **No synthetic DD paths.**  Every path references a real DD entry.
- **No coverage gate**.  The set is intentionally stratified, not
  representative; per-domain pass rates matter, not overall.

## 7. Acceptance

Plan 34 is closed when:

1. `benchmark_v1.json` has 50 positives + 20 negatives, all with `source ∈
   {isn_catalog, graph_published, synthetic}` and no `TODO` stubs.
2. Two-reviewer consensus recorded for every published positive
   (|Δscore| ≤ 0.2).
3. Runner passes gating (≥ 0.80 pass@1, ≥ 0.75 mean score, ≥ 0.90 reject rate)
   on the production compose prompt + reviewer pool.
4. `make bench` (or equivalent `uv run` recipe) wired — optional.

v1 scaffolding (this commit) is complete when:

1. Plan filed.
2. `benchmark_v1.json` exists with ≥ 20 positives + ≥ 10 negatives filled.
3. Runner script loads, validates schema, and dry-runs end-to-end in
   `--mock` mode.
4. `plans/README.md` references plan 34.
