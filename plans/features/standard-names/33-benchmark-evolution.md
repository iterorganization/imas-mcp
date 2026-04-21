# Plan 33 — SN Benchmark Evolution Strategy

> **Status:** design / research — no code yet.
> **Follows:** `plans/research/standard-names/38-benchmark-cycle-1.md` (cycles 1+2).
> **Context:** `imas_codex/standard_names/benchmark.py`, `calibration.py`,
> `benchmark_calibration.yaml`, `llm/config/sn_review_criteria.yaml`,
> `standard_names/review/pipeline.py` (multi-reviewer production pipeline).

## 0. Why now

Cycle 2 results (post R1–R3 fixes) cluster in a tight 0.82 – 0.94 band for
all seven compose models. The benchmark still has structural issues that
blunt its ability to discriminate, to catch regressions, and to reflect
real name quality:

- **Single-reviewer bias** — only `claude-opus-4.6` scores. Phase 4 of
  cycle 1 showed mean-|Δ| 0.028 – 0.104 vs a Gemini second reviewer,
  with 20 % tier-disagreement on sonnet output. A single judge has
  systematic slant.
- **Hard rubric** — each of 4 dimensions is scored 0-20 with integer
  granularity; a single "dock 4 points" rule can move a name out of
  `outstanding` over a minor stylistic quibble.
- **Anchor sparsity / bias** — 32 calibration anchors, 13 of which are
  core_profiles/equilibrium "outstanding" exemplars. 21 IDSs and most
  physics domains are unrepresented.
- **Isolated workflow** — `sn benchmark` is a standalone one-shot
  command. No CI regression gate, no coupling to grammar sync, no
  persistence into the graph, not reflected in `sn status`.

This plan addresses these four axes and stages the work across three
phases.

---

## 1. Soft-criterion scoring deep dive

### 1.1 What the reviewer actually scores today

Source: `imas_codex/llm/prompts/sn/review_name_only.md` + `sn_review_criteria.yaml`.

Four integer dimensions (0-20 each), summed to 0-80, normalised to
0-1:

| Dim          | What it rewards                                                       | Hard docks                                                                 |
|--------------|-----------------------------------------------------------------------|----------------------------------------------------------------------------|
| grammar      | parses and round-trips; decomposition audit penalises absorbed tokens | −4 per absorbed closed-vocab token in `physical_base` (capped −8)          |
| semantic     | accurate physical quantity and decomposition choice                    | 0 on wrong physics qualifier (I2.7)                                        |
| convention   | snake_case, canonical ordering, no `_from_`, no DD leakage             | score_override: `convention=0` for duplicate names or DD prefix leakage   |
| completeness | required segments present, unit/kind consistent, tags sensible        | none                                                                       |

Tier boundaries: `outstanding ≥ 0.85`, `good ≥ 0.60`, `adequate ≥ 0.40`,
`poor` below.

### 1.2 What the calibration anchors do

32 entries in `benchmark_calibration.yaml`, each with `name`, `tier`,
`expected_score` (0-120 scale; divided by 120 to normalise for the
prompt). `calibration.get_calibration_for_prompt()` renders them into
the reviewer's system prompt as `{name, tier, score, reason}` examples.
Anchors are cached via the Anthropic prompt-cache block (≥90 % hit
rates observed), so adding more anchors is mostly a one-time expense.

**Help:** ground-truth in-context examples materially improved
inter-reviewer agreement in cycle 2 (tier disagreement dropped from
baseline ~30 % to 5-20 %).

**Bias risk:** 13/32 anchors are `outstanding` ISN reference entries,
all from core_profiles / equilibrium / magnetics. Reviewers score
equilibrium-domain candidates generously because there are explicit
near-neighbours in the anchor set; unseen domains (edge_profiles,
waves, ntms, divertors) get penalised harder for merely being
unfamiliar.

### 1.3 Is the user's "softer criterion" intuition warranted?

**Partially, yes — but the right fix is not a uniform rubric softening.**
The 0-20 integer scale is already coarse; uniformly lowering the bar
would collapse discrimination between merely-correct and excellent
names.

The real issues are **semantic equivalence blindness** and
**binary-feel grammar docking**:

- Two names with identical grammar-fields decomposition but different
  surface forms (`plasma_boundary_gap_angle` vs
  `angle_of_plasma_boundary_gap`) currently get different scores
  depending on which noun the reviewer thinks "should" be the
  `physical_base`. Phase 4 cycle 1 observed this exact disagreement
  between opus and gemini reviewers.
- The decomposition audit (`−4 per absorbed closed-vocab token`, capped
  at `−8`) is applied as a binary trigger: one borderline token knocks
  4 grammar points off even when the name is indisputably correct
  (e.g. `poloidal_flux` — `poloidal` is a closed-vocab coordinate but
  `poloidal_flux` is an accepted lexicalised atom).

### 1.4 Proposed refinements

**R1. Semantic-equivalence credit (Phase C).**
If two candidate names have identical `grammar_fields` decomposition
(after normalising to the declared grammar), they should receive
identical scores regardless of token order. Implemented as a
post-hoc **consensus lift**: after scoring a batch, group candidates
by grammar_fields hash; within each group, lift every member's score
to the in-group max (only when that max corresponds to an
`outstanding`/`good` tier — we do not want to lift a "poor" group
upward). Expected effect: eliminates the Phase-4 `plasma_boundary_gap_angle`
class of disagreement.

**R2. Multi-valid acceptance (Phase C).**
Looser variant of R1 intended for the LLM reviewer: add a prompt
stanza stating that when a candidate's grammar_fields decomposition
is valid and the surface form also round-trips, the reviewer MUST
check whether a previously-reviewed candidate in this batch has an
equivalent decomposition and, if so, score them within ±1 point of
each other on grammar and convention. Prevents reviewer drift
across a batch.

**R3. Rubric decomposition of `grammar` (Phase C).**
Replace the single `grammar (0-20)` dimension with 4 sub-dimensions
of 0-5 each, summed back to 0-20:

| Sub-dimension      | What it measures                                             |
|--------------------|--------------------------------------------------------------|
| parseable          | `parse_standard_name(name)` succeeds (0 or 5)                |
| round_trips        | `compose(parse(name)) == name` (0 or 5)                      |
| no_duplicated_tokens | No closed-vocab token repeated across segments (0-5 gradient) |
| decomposition_depth  | Segments populated beyond physical_base (0-5 gradient)     |

Benefits: preserves the summed 0-20 scale so reports keep working; but
when a failure is localised (e.g. only `no_duplicated_tokens` docks)
that is visible in downstream analytics. The existing decomposition
audit becomes a straightforward −1-per-hit on `no_duplicated_tokens`
instead of `−4 per hit capped at −8`. That's the "soft criterion"
the user wants, without flattening the scale.

---

## 2. Multi-family benchmark reviewer

### 2.1 Current state

`BenchmarkConfig.reviewer_model` is a single string. `score_with_reviewer`
loops batches through that one model. The production review pipeline
(`standard_names/review/pipeline.py` lines 528-578, 623-627) already
supports N ≥ 1 reviewers — the **first** one is canonical (mirrored onto
`StandardName` node fields), subsequent ones produce `Review` nodes linked
via `REVIEWS`. Aggregates (`review_count`, `review_mean_score`,
`review_disagreement`) are written by `graph_ops.update_review_aggregates`
with `disagreement_threshold` from `[sn.review].disagreement-threshold`.

### 2.2 Proposal

1. **Config source.** Reuse `[sn.review].models` as the source of the
   reviewer panel for benchmark. Rationale: the benchmark and the
   production review path should judge by the same rubric and the same
   panel — otherwise we are measuring a different target than we
   deploy. `[sn.benchmark].reviewer-model` becomes a compatibility
   fallback (single-reviewer mode if `[sn.review].models` is empty).

2. **Execution.** Extend `score_with_reviewer` to accept
   `reviewer_models: list[str]` and score every candidate with every
   reviewer. Canonical reviewer is `models[0]`. Cost estimate: 2-3× the
   current pass (3 reviewers in the default panel would take cycle 2
   from ~$5.66 → ~$12-15 per run, still inside budget headroom for
   regression gating at lower frequency).

3. **Aggregation.** For each candidate, compute:
   - `score_canonical` — score from `models[0]` (preserves back-compat).
   - `score_median` — robust aggregate across reviewers (used for
     regression tracking and leaderboard ranking).
   - `score_spread` — `max − min` across reviewers.
   - `contested: bool` — true iff `score_spread ≥ 0.15` OR tiers
     differ across reviewers. Flagged rows go into a dedicated
     "needs human inspection" table in the output.

4. **Report format.**
   - Per-model leaderboard shows median + spread column.
   - New section lists contested candidates (name, per-reviewer
     scores, per-reviewer tiers, suggested resolution if consensus-lift
     R1 would change the outcome).
   - JSON report: `ModelResult.quality_scores` keeps per-reviewer
     entries; `avg_quality_score` is computed from the median.

---

## 3. Calibration anchor expansion

### 3.1 Today

32 anchors (13 outstanding, 7 good, 5 adequate, 7 poor). Provenance
(from `benchmark_calibration.yaml` header + `reason` fields):

- Outstanding: ISN 42 reference entries + cycle-2 equilibrium boundary
  exemplars.
- Good: WEST standard-names catalog + shorter ISN entries.
- Adequate: entries that work but have quality issues.
- Poor: synthetic anti-patterns + observed benchmark failures.

Distribution by physics_domain (sampled from `reason` strings):
equilibrium ×7, core_profiles ×6, magnetics ×4, transport ×3,
wall ×2, misc ×10. Nothing from edge_profiles, waves, mhd,
ntms, divertors, sawteeth, pellets, lh, ic, radiation, gyrokinetics,
or 18 other IDSs.

### 3.2 Target

3-5 anchors per physics_domain across the ~32 active domains,
balanced across tiers (~2 outstanding/good, ~1 adequate, ~1-2 poor
per domain). Rough cardinality: **100-150 anchors total** (≈ 4× today).

### 3.3 Sourcing strategy

Mine existing `StandardName` nodes in the graph:

```cypher
// Outstanding candidates
MATCH (sn:StandardName)
WHERE sn.review_mean_score >= 0.90
  AND sn.review_count >= 2
  AND sn.review_disagreement = false
RETURN sn.id, sn.physics_domain, sn.review_mean_score, ...
ORDER BY sn.physics_domain, sn.review_mean_score DESC

// Poor candidates
MATCH (sn:StandardName)
WHERE sn.review_mean_score <= 0.50 AND sn.review_count >= 1
RETURN ...
```

Stratify: pick top-2 by score per domain for `outstanding`, bottom-2
for `poor`, median-2 for `good`/`adequate`. Dedupe against current
anchors by name.

### 3.4 Automation

Add `sn benchmark --update-calibration` mode:

1. Runs benchmark normally.
2. After scoring, for every candidate where all reviewers agree
   (`spread < 0.05`) AND score is extreme (`≥ 0.95` or `≤ 0.45`),
   propose as a new anchor.
3. Write proposals to `benchmark_calibration.proposals.yaml` for human
   review. **Do not auto-commit** — new anchors bias every future
   benchmark and must be vetted.

### 3.5 Cost implication

Current system prompt rendered with 32 anchors: ~40 KB. At 150 anchors
≈ 180 KB. Anthropic prompt-cache block handles this (read tokens are
cheap at 0.1× list; creation is paid once per TTL). Cycle 2 cache read
rate was 90-100 %, so the steady-state cost rise is ≈ 10-20 % per
run — absorbed.

---

## 4. Workflow integration

### 4.1 Benchmark results in `sn status`

Persist benchmark reports into the graph:

- Create a `Benchmark` node per run (timestamp, IDS filter, reviewer
  panel hash).
- Link `(:Benchmark)-[:EVALUATED]->(m:Model)` with
  properties {avg_score, cost, elapsed_s, fields_pop_pct, contested_count}.
- `sn status` queries the latest `Benchmark` node and renders a
  leaderboard panel showing per-model last-seen scores and δ vs
  the previous run.

### 4.2 Auto-trigger on ISN grammar sync

The SN grammar is versioned as `isn-grammar-rcN` tags. When
`scripts/sync_isn_grammar.py` (or the equivalent step) promotes
`rcN → rcN+1`, it invalidates every cached decomposition. Proposal:
that sync command emits a follow-up:

```
uv run imas-codex sn benchmark --ids equilibrium --max-candidates 20 \
    --baseline-compare .bench-work/last-baseline.json
```

If any model's best score drops by more than 0.05 vs the last
baseline, the sync is flagged. Does **not** block (grammar sync has
its own merits), but surfaces regressions immediately.

### 4.3 CI regression gate

Add `tests/benchmark/test_regression.py`:

- Runs a **pinned** 10-item equilibrium subset with `[sn.review].models[0]`
  only (single-reviewer, minimal cost — ~$0.30 per CI run).
- Asserts best-model score ≥ `baseline − 0.05`.
- Baseline is a JSON committed into the repo
  (`tests/benchmark/baseline.json`); updated by a manual
  `uv run imas-codex sn benchmark --refresh-baseline` when a rubric
  change is intentional.
- Marked `@pytest.mark.benchmark` and `@pytest.mark.skipif(not
  os.environ.get("RUN_SN_BENCHMARK"))` so default CI skips; a
  nightly job sets the env var.

---

## 5. Implementation staging

### Phase A — Multi-family benchmark reviewer (moderate, Opus 4.6)

**Goal:** benchmark uses the `[sn.review].models` panel; per-reviewer
and median scores surface in the report.

Files:
- `imas_codex/standard_names/benchmark.py` — extend `BenchmarkConfig`
  with `reviewer_models: list[str]`; update `score_with_reviewer` to
  loop reviewers; add per-reviewer and median score fields to
  `ModelResult.quality_scores`.
- `imas_codex/cli/sn.py` — `sn benchmark --reviewer-models a,b,c`
  flag; default pulls from `[sn.review].models`.
- `imas_codex/settings.py` — no new accessor needed; reuse
  `get_sn_review_primary_model()` / `get_sn_review_secondary_models()`.
- `tests/standard_names/test_benchmark.py` — unit test for
  aggregation (median, spread, contested flag).

Acceptance criteria:
- `sn benchmark` on equilibrium/20 items with 2 reviewers produces
  per-reviewer columns and a median column; `spread ≥ 0.15` rows
  listed in a "Contested" section.
- Single-reviewer mode (panel of 1) is byte-identical to today's
  output.
- Round-trip: JSON report serialises/deserialises with new fields.

Budget: ~$15-20 for 2 equilibrium dry-runs (3 reviewers × cycle-2 cost).

### Phase B — Calibration anchor expansion automation (moderate, Opus 4.6)

**Goal:** 100-150 domain-balanced anchors + automation to refresh.

Files:
- `scripts/calibration/propose_anchors.py` — mines the graph, emits
  `benchmark_calibration.proposals.yaml` (stratified by
  physics_domain × tier).
- `imas_codex/standard_names/benchmark.py` — `--update-calibration`
  mode that writes proposals after a benchmark run.
- `imas_codex/standard_names/benchmark_calibration.yaml` — manual
  one-time expansion (reviewed PR) from the first automated
  proposal file.
- `imas_codex/standard_names/calibration.py` — add
  `get_calibration_summary()` returning per-domain counts
  (for observability in `sn status`).

Acceptance criteria:
- `uv run python scripts/calibration/propose_anchors.py --target 120`
  emits YAML with ≥ 3 entries per represented domain and no
  duplicates vs current anchors.
- Post-expansion cycle on equilibrium: prompt cache creation tokens
  ≤ 2× cycle-2 baseline; read-rate ≥ 90 %.
- Reviewer score variance across repeated runs on the same dataset
  (stability test) does not increase by > 0.02 std dev vs pre-expansion.

Budget: ~$5 for stability test runs.

### Phase C — Rubric sub-dimension split + semantic equivalence (complex, Opus 4.7)

**Goal:** grammar split into 4 sub-dimensions, consensus-lift post-hoc
rule, reviewer-side multi-valid acceptance stanza.

Files:
- `imas_codex/llm/config/sn_review_criteria.yaml` — add
  `grammar.subdimensions` block with 4 × 0-5 entries.
- `imas_codex/llm/prompts/sn/review_name_only.md` + `review.md` —
  new rubric section; R2 multi-valid stanza; updated examples.
- `imas_codex/standard_names/models.py` — add
  `GrammarSubScores` (parseable, round_trips, no_duplicated_tokens,
  decomposition_depth) to `StandardNameQualityReviewNameOnly.scores`.
  Keep top-level `grammar` for back-compat — computed as the sum.
- `imas_codex/standard_names/benchmark.py` — R1 consensus-lift
  post-processing step over batches.
- `imas_codex/graph/linkml/standard_names.yaml` — optional
  `Review.grammar_sub_scores` attribute (JSON-blob).
- `tests/standard_names/test_review_rubric.py` — unit tests for
  sub-score summing and consensus lift.

Acceptance criteria:
- Golden-dataset test: Phase-4-cycle-1 `plasma_boundary_gap_angle`
  vs `angle_of_plasma_boundary_gap` now score within ±0.02 under
  consensus lift.
- Decomposition audit now docks `no_duplicated_tokens` only (not
  total grammar); `parseable` and `round_trips` remain hard-binary.
- Tier distribution shift: re-scoring cycle-2 data expected to move
  2-5 % of `adequate` names into `good` (softer criterion as
  requested by the user).
- Back-compat: existing JSON reports (without sub-scores) still
  load via `BenchmarkReport.from_json`.

Budget: ~$10-15 for re-scoring runs on existing cycle-2 data.

---

## 6. Open questions

1. **Panel size vs cost.** Is a 3-reviewer panel (opus + gpt-5.4 +
   gemini-3.1-pro) the right default, or should we default to 2
   (opus + gemini) and expose 3 as opt-in via
   `[sn.review].extended-panel`? Affects CI cost floor.

2. **Consensus lift scope.** Should R1 lift across _all_ candidates in
   a run, or only within a single extraction batch? Cross-batch
   lifting is stronger but risks masking prompt-instability effects.

3. **Anchor refresh cadence.** Auto-refresh after every benchmark,
   every ISN grammar sync, or quarterly? First is noisy; third may
   drift too slowly. Proposal: only at grammar-sync boundaries.

4. **Poor-tier anchors from the graph.** Mining `review_mean_score ≤ 0.50`
   produces anchors that were _judged_ poor by the current reviewer
   panel — risks a circular calibration. Should we instead curate
   poor anchors manually from ISN PR-review discussion threads?

5. **Baseline JSON in repo.** A committed `tests/benchmark/baseline.json`
   becomes a schema-evolution point (every rubric change requires a
   baseline refresh PR). Acceptable, or should the baseline live as
   an artefact uploaded by the nightly job and queried at test time?

6. **Grammar sub-dimension names exposed to reviewer.** The LLM
   reviewer needs to emit 4 integers instead of one for grammar.
   Pydantic structured output handles this, but we should measure
   whether asking for 4 sub-scores degrades non-grammar dimensions
   (attention dilution) before committing Phase C.
