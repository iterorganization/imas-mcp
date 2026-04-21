# Benchmark cycle 1 — evaluate → fix → rerun

**Status:** in progress — 2026-04-21
**Scope:** Run SN benchmark against `equilibrium` IDS (20 paths, 7 compose models,
claude-opus-4.6 as reviewer-judge); identify rubric/prompt gaps; apply targeted
fixes; rerun.

All raw artefacts live in `.bench-work/` (git-ignored): `cycle1-baseline.json`,
`cycle1-baseline.log`, later `cycle*-after.json`.

## Phase 1 — baseline

Command (with newly added `--force` flag, since every equilibrium path is
already processed and default extraction filters out processed paths):

```
uv run imas-codex sn benchmark --ids equilibrium --max-candidates 20 --force \
    --output .bench-work/cycle1-baseline.json
```

Reviewer: `anthropic/claude-opus-4.6` (from `[sn.benchmark].reviewer-model`).
Compose models: full default list of 7 from `[sn.benchmark].compose-models`.

### Summary table

Total run cost: **$6.15**. Duration: ~50 min wall-clock (dominated by opus-4.6
reviewer passes — 7 × 2 batches of 10 items each).

| Model                                | n  | grammar-valid | cost     | avg quality (0–1) | distribution                 |
|--------------------------------------|----|---------------|----------|-------------------|------------------------------|
| anthropic/claude-sonnet-4.6          | 20 | 100%          | $1.394   | 0.439             | 0 out / 0 good / 17 adeq / 3 poor |
| anthropic/claude-haiku-4.5           | 28 | 100%          | $0.838   | 0.359             | 0 out / 0 good / 5 adeq / 23 poor |
| openai/gpt-5.4                       | 19 | 100%          | $1.468   | 0.469             | 0 out / 0 good / 16 adeq / 3 poor |
| openai/gpt-5.4-mini                  | 20 | 100%          | $1.494   | 0.418             | 0 out / 2 good / 12 adeq / 6 poor |
| google/gemini-3.1-pro-preview        | 20 | 100%          | $0.879   | 0.474             | 0 out / 2 good / 14 adeq / 4 poor |
| google/gemini-3-flash-preview        | 20 | 100%          | $0.053   | 0.445             | 0 out / 2 good / 14 adeq / 4 poor |
| google/gemini-3.1-flash-lite-preview | 20 | 100%          | $0.027   | 0.422             | 0 out / 0 good / 12 adeq / 8 poor |

(haiku emitted 28 candidates — it split some DD paths into 2 names each.)

### Per-dimension average scores (0–20 per dimension, **full 6-dim rubric**)

| Model                                | gram | sem  | doc  | conv | comp | cplc |
|--------------------------------------|-----:|-----:|-----:|-----:|-----:|-----:|
| anthropic/claude-sonnet-4.6          | 13.1 | 14.8 | **0.0** | 13.1 | **2.0** | 9.7  |
| anthropic/claude-haiku-4.5           | 10.3 | 13.5 | **0.0** |  9.6 | **2.6** | 7.1  |
| openai/gpt-5.4                       | 13.5 | 15.3 | **0.0** | 13.4 | **3.8** | 10.3 |
| openai/gpt-5.4-mini                  | 10.7 | 13.7 | **0.0** | 11.3 | **4.0** | 10.5 |
| google/gemini-3.1-pro-preview        | 14.5 | 15.7 | **0.0** | 13.0 | **3.4** | 10.3 |
| google/gemini-3-flash-preview        | 12.1 | 14.8 | **0.0** | 12.5 | **3.0** | 11.1 |
| google/gemini-3.1-flash-lite-preview | 11.0 | 14.9 | **0.0** | 12.0 | **3.1** |  9.7 |

### Cache hit rates

| Model                                | read %   |
|--------------------------------------|---------:|
| anthropic/claude-sonnet-4.6          | 100%     |
| anthropic/claude-haiku-4.5           |  94%     |
| openai/gpt-5.4                       | 100%     |
| openai/gpt-5.4-mini                  | 100%     |
| google/gemini-3.1-pro-preview        |  90%     |
| google/gemini-3-flash-preview        |  95%     |
| google/gemini-3.1-flash-lite-preview |  95%     |

Cache infrastructure is healthy; prompt-cache block is stable across runs.

### Example high-/low-scoring names

Across **every** model the top two scorers are the two canonical equilibrium
boundary properties:

- `minor_radius_of_plasma_boundary`   ← `equilibrium/time_slice/boundary/minor_radius`
- `elongation_of_plasma_boundary`     ← `equilibrium/time_slice/boundary/elongation`

Both round-trip through the grammar with `physical_base=minor_radius|elongation`
+ `geometry=plasma_boundary`. All seven models converge here.

Bottom performers are universally the two *gap / distance* items:

- `equilibrium/time_slice/boundary/closest_wall_point/distance`  → every model
  produced a `distance_between_X_and_Y` form and was scored 0.23–0.34.
- `equilibrium/time_slice/boundary/gap/value` → `distance_between_gap_reference_point_and_plasma_boundary`
  style names, same low scores.

**Crucial finding:** these `distance_between_*` names DO round-trip through
the grammar (verified via `parse_standard_name` / `compose_standard_name`) —
they land as a single open-vocabulary `physical_base`. But the reviewer calls
them "unparseable" and drops grammar/convention scores. This is a reviewer
miscalibration, not a compose-step bug.

### Identified weaknesses per model

- **Haiku-4.5**: bottom of pack (0.36 avg). Over-generates (emits 28 names for
  20 items → it splits paths unnecessarily). grammar_fields almost always
  empty. Lowest convention score (9.6/20).
- **Sonnet-4.6**: strong semantic (14.8) but documentation dragged average
  down; `grammar_fields` populated 0% of the time.
- **GPT-5.4**: strongest semantic (15.3) and cheapest Anthropic-class quality
  per name ($0.077) among tier-1 models; prefers `_from_X_to_Y` distances
  which violates I1.1 (`_from_` forbidden).
- **GPT-5.4-mini**: only model that hit "good" tier twice; average name-length
  creeps above 7 tokens.
- **Gemini-3.1-Pro**: highest grammar score (14.5) and joint-highest average
  (0.474). Slightly expensive relative to flash.
- **Gemini-3-flash**: remarkable price/quality ($0.05 → 0.44 avg). Viable
  budget option.
- **Gemini-3.1-flash-lite**: very cheap ($0.03) but struggles with the gap-
  distance family (8/20 in poor tier).

## Phase 2 — identified root causes and proposed fixes

### R1. Benchmark reviewer rubric/compose output mismatch (BLOCKING)

`StandardNameCandidate` schema is explicitly **name-only**
(`imas_codex/standard_names/models.py:10-20` docstring: *"Documentation
(description, tags, links, etc.) is added by `sn enrich`."*). The compose-
stage output contains no `description`, `documentation`, `unit`, or `tags`.

`benchmark.score_with_reviewer` currently hard-codes `prompt="sn/review"` +
`StandardNameQualityReviewBatch` — the **full 6-dimension** rubric that
explicitly scores documentation quality. Result: `documentation_score = 0.00`
across every model on every candidate, and `completeness_score ≤ 4/20`
because tags/unit fields are also absent by construction.

This isn't a model deficiency, it's a rubric that is asking the wrong
questions. The 4-dim `sn/review_name_only` rubric
(`StandardNameQualityReviewNameOnlyBatch`, normalized over 80) already
exists for exactly this case — see
`imas_codex/standard_names/review/pipeline.py:1044-1055` which switches
prompts based on a `target` argument.

**Fix:** teach `score_with_reviewer` to honour a `review_target` setting
(default `"names"` — matching the current compose output fidelity) and
switch to `sn/review_name_only` + `StandardNameQualityReviewNameOnlyBatch`
accordingly. Map the 4 dimensions onto the existing `ModelResult` fields;
leave `documentation_score`/`compliance_score` unset (None) for name-only
runs instead of hard-writing 0.

Expected effect: averages should jump from ~0.44 → ~0.70 simply by
removing the two inapplicable dimensions; the **relative** ranking of
models is preserved. This matches the session-history expectation that
sonnet-4.6 runs around 76.5 avg.

### R2. Reviewer penalises open-vocabulary compound `physical_base` tokens

The review prompt flags `distance_between_X_and_Y` as "unparseable grammar"
and drops grammar + convention points even though:

1. `parse_standard_name("distance_between_plasma_boundary_and_closest_wall_point")`
   **succeeds**, producing `physical_base=distance_between_plasma_boundary_and_closest_wall_point`.
2. `compose_system.md` NC-14 explicitly prescribes this pattern for
   distances between two named features.

Every single model was penalised here, so this affects ~10% of the benchmark
mass. The fix belongs in **the review prompt**, not in compose (where NC-14
is already correct).

**Fix:** add a clarifying paragraph to `sn/review_name_only.md` (and to
`sn/review.md` for consistency) stating that (a) `physical_base` is an
**open vocabulary** that can absorb compound tokens, and (b) a name that
round-trips through `parse_standard_name / compose_standard_name` is by
definition grammatically valid. Reviewers must not reject a parse-valid
compound `physical_base` as "unparseable".

### R3. `grammar_fields` population is near-zero

Fields-consistent rate is 0% for sonnet/haiku/gpt-mini and only 10–17% for
the best models. This is because compose-stage models return bare names
without populating the companion `grammar_fields` map. Downstream tooling
(validation audits, review auto-verdicts) relies on this map.

**Fix:** strengthen `compose_dd.md` with a mandatory reminder that every
candidate MUST populate `grammar_fields` with the decomposition it used
when generating the name, plus a micro-example. The schema already
defaults this to `{}`, so the fix is purely a prompt-level nudge.

### R4. IDS-wide name `_of_plasma_boundary` consistency (already handled)

All models correctly produced `minor_radius_of_plasma_boundary` and
`elongation_of_plasma_boundary` — the existing NC-2 rule is working. No
change needed.

### R5. Haiku over-splits paths

Haiku emitted 28 candidates for 20 paths (some DD paths were named twice,
e.g. splitting `.../gap/value` into two families). Not a prompt bug
per se — likely a batching side-effect. Ignore for this cycle; revisit
if it persists in cycle 2.

## Phase 3 — applied changes and re-benchmark

### Commits

- `fix(sn): use name-only rubric in benchmark when compose output is name-only`
  (R1) — `BenchmarkConfig.review_target` defaults to `"names"`, so benchmark
  now scores against the 4-dim `sn/review_name_only` rubric (0–80, normalised
  /80). `--review-target full` opts back into the 6-dim rubric for post-enrich
  runs.
- `fix(sn): teach reviewer that open-vocabulary physical_base compounds are
  grammar-valid` (R2) — added a "Open-Vocabulary `physical_base`" section at
  the top of `sn/review_name_only.md` and `sn/review.md` anchoring grammar
  validity to the `parse → compose` round-trip.
- `fix(sn): require compose models to populate grammar_fields with worked
  examples` (R3) — added "Grammar Fields — MANDATORY" section to
  `sn/compose_dd.md` with 5 worked decomposition examples.

### Re-benchmark results

Command:

```
uv run imas-codex sn benchmark --ids equilibrium --max-candidates 20 --force \
    --output .bench-work/cycle1-after.json
```

Total run cost: **$5.66**. Per-dim scores now on 0–20 scale (grammar, semantic,
convention, completeness), normalised total /80.

| Model                                | cost   | avg quality | grammar | semantic | convention | complete | fields pop% | errors |
|--------------------------------------|--------|-------------|---------|----------|------------|----------|-------------|--------|
| anthropic/claude-sonnet-4.6          | $0.43  | 0.912       | 18.8    | 18.5     | 18.1       | 17.7     | 0%          | 0      |
| anthropic/claude-haiku-4.5           | $1.13  | 0.846       | 18.9    | 16.0     | 17.2       | 15.6     | 0%          | 3      |
| openai/gpt-5.4                       | $1.48  | **0.939**   | **19.6**| **19.2** | **18.6**   | 17.6     | 30%         | 0      |
| openai/gpt-5.4-mini                  | $1.50  | 0.884       | 18.8    | 17.6     | 17.6       | 16.8     | 30%         | 0      |
| google/gemini-3.1-pro-preview        | $1.03  | 0.938       | 19.4    | 19.1     | 18.4       | **18.2** | 29%         | 0      |
| google/gemini-3-flash-preview        | $0.06  | 0.882       | 16.8    | 18.6     | 17.6       | 17.6     | 26%         | 0      |
| google/gemini-3.1-flash-lite-preview | $0.03  | 0.818       | 14.6    | 18.1     | 15.8       | 16.9     | 23%         | 0      |

### Delta (cycle 2 − cycle 1, avg quality)

| Model                                | cycle 1 | cycle 2 | Δ       |
|--------------------------------------|---------|---------|---------|
| anthropic/claude-sonnet-4.6          | 0.439   | 0.912   | +0.474  |
| anthropic/claude-haiku-4.5           | 0.359   | 0.846   | +0.486  |
| openai/gpt-5.4                       | 0.469   | 0.939   | +0.470  |
| openai/gpt-5.4-mini                  | 0.418   | 0.884   | +0.466  |
| google/gemini-3.1-pro-preview        | 0.474   | 0.938   | +0.464  |
| google/gemini-3-flash-preview        | 0.445   | 0.882   | +0.437  |
| google/gemini-3.1-flash-lite-preview | 0.422   | 0.818   | +0.396  |

### Observations

1. **R1 (rubric switch) accounts for most of the Δ.** Dropping two
   inapplicable dimensions (documentation, compliance) roughly doubles every
   normalised score, and legitimate score headroom now lives in semantic and
   completeness (where the worst model is still ≥ 14.6/20 on grammar).

2. **R2 (open-vocabulary anchor) removes the "distance_between_X_and_Y"
   false-positive penalty.** Grammar scores now cluster at 18–19.6/20 for
   frontier models, vs the cycle-1 pattern where the reviewer marked perfectly
   parse-valid compounds as "unparseable".

3. **R3 (grammar_fields) partial success.** OpenAI and Gemini models now
   populate `grammar_fields` in 23–30% of candidates (up from 10–23%).
   **Anthropic models remain at 0%** — the `compose_dd.md` reminder does not
   overcome whatever default suppresses grammar_fields for Claude. Needs a
   deeper fix: either a schema-level `min_items` / required-field constraint
   on `grammar_fields`, or reinforcement inside `compose_system.md` itself.

4. **Frontier:** `openai/gpt-5.4` (75.2/80) and `google/gemini-3.1-pro-preview`
   (75.1/80) are statistically tied at the top. `claude-sonnet-4.6` close
   behind (73.0/80). `gemini-3.1-flash-lite-preview` (65.4/80) is the weakest
   but still "good"-tier.

5. **Haiku JSON truncation.** Haiku hit three `Invalid JSON` errors in this
   run (trailing comma at col 64225; EOF while parsing at cols 120599 and
   159461). Symptom is the same as before: Haiku over-emits tokens and hits
   the response budget. Orthogonal to the prompt fixes. Should be tracked
   separately — likely needs per-model `max_tokens` in the compose worker
   dispatch, or a smaller batch size for Haiku specifically.

6. **Gemini-flash grammar=16.8, flash-lite=14.6.** These two still produce
   some parse-invalid names; inspect individual items in
   `.bench-work/cycle1-after.json` during calibration (Phase 5) to decide
   whether to add them as anti-patterns or upstream with a targeted NC-rule.

### Budget

Cycle 1 baseline: $6.15. Cycle 2: $5.66. Remaining: ~$13.19.

## Phase 4 — multi-reviewer consistency

Re-reviewed cycle-2 frontier-model candidates (sonnet-4.6, gpt-5.4,
gemini-3.1-pro) with a **secondary reviewer** (`google/gemini-3.1-pro-preview`)
to measure reviewer variance against the canonical `claude-opus-4.6` judge.
Script: `.bench-work/rereview.py` — re-uses `score_with_reviewer()` without
re-composing, so cost is dominated by the cheap Gemini reviewer (<$1 total).

Artefacts: `.bench-work/cycle1-multirev.json`, `.bench-work/cycle1-multirev.log`.

| Compose model                  | n  | mean |Δ| | max |Δ| | tier disagree | mean opus | mean gemini |
|--------------------------------|----|----------|---------|---------------|-----------|-------------|
| anthropic/claude-sonnet-4.6    | 20 | 0.104    | 0.338   | 4 (20%)       | 0.912     | 0.916       |
| openai/gpt-5.4                 | 20 | 0.039    | 0.200   | 1 (5%)        | 0.939     | 0.976       |
| google/gemini-3.1-pro-preview  | 19 | 0.028    | 0.100   | 1 (5%)        | 0.938     | 0.955       |

**Agreement is high.** Mean absolute per-item score delta is 0.028–0.104;
mean-of-means matches within 0.037. Tier disagreement is ≤ 5 % for the
top two compose models and rises to 20 % only for sonnet-4.6.

### High-disagreement items (calibration candidates)

All 4 sonnet disagreements were on `vertical_coordinate_of_<X>` names
(geometric_axis, gap_reference_point, plasma_boundary_outline_point).
Gemini consistently downgraded these from `outstanding` → `adequate`
(Δ ≈ −0.3) because the NAME parses but the `grammar_fields` decomposition
sonnet emitted was inconsistent (e.g. missing the `physical_base` /
`position` split). Matches finding (3) in Phase 3 — Anthropic's
grammar_fields problem also inflates opus's grammar score relative to
gemini, which is stricter about the field map.

The other two disagreements (`plasma_boundary_gap_angle` vs
`angle_of_plasma_boundary_gap`) are **both grammatically valid**; opus
marked the `_of_plasma_boundary_gap` form as `good`, gemini as
`outstanding`. These are borderline-ordering cases (which noun is the
`physical_base`?) — a good target for a future compose_system NC-rule.

## Phase 5 — calibration updates

Added 5 entries to `imas_codex/standard_names/benchmark_calibration.yaml`:

**New exemplars (outstanding, all scored ≥ 0.95 by both reviewers on the
cycle-2 equilibrium benchmark):**

- `minor_radius_of_plasma_boundary` — textbook `physical_base_of_position`.
- `elongation_of_plasma_boundary` — same form, dimensionless shape.
- `distance_between_plasma_boundary_and_closest_wall_point` — locks in the
  NC-14 canonical form AND the open-vocabulary `physical_base` principle.

**New anti-patterns (poor):**

- `elongation_of_plasma_boundary` collapsed from `elongation_upper /
  elongation_lower` — semantic loss / namespace collision (observed on
  haiku + gpt-5.4-mini, scored 0.65 / 0.71 on the baseline).
- `vertical_coordinate_of_closest_wall_point` with malformed
  `grammar_fields` (invalid `geometric_base`, invented `geometry` key —
  observed on flash-lite across ≥ 6 items, scored 0.57–0.64).

Previous calibration was dominated by core_profiles / magnetics entries;
these new anchors give the benchmark reviewer concrete equilibrium-domain
reference points in both extremes of the tier distribution.

**Total entries now:** 32 (13 outstanding, 7 good, 5 adequate, 7 poor).

### Why no cycle-3 stability run?

The task plan optionally included a third small benchmark after calibration
update to confirm stability. Deferred because:

1. Cycle-2 scores are already tightly clustered (all 7 models in the 0.82 –
   0.94 band, no outliers or regressions).
2. Calibration entries live in the reviewer's system prompt (cached), so
   their effect shows up in every future benchmark at no extra cost — we
   will observe it on the next cycle organically.
3. Budget accounting: $6.15 (cycle 1) + $5.66 (cycle 2) + ~$1 (multi-rev)
   = ~$12.8 of $25. Retaining headroom for a post-R4/R5 cycle on a
   different IDS (e.g. `core_profiles` or `magnetics`) is more valuable
   than running the same equilibrium benchmark a third time.

## Remaining issues (for a future cycle)

1. **Anthropic `grammar_fields` at 0%.** The compose_dd.md reminder was
   insufficient. Options: (a) enforce `grammar_fields` as required in
   the Pydantic schema and let structured-output validation force it;
   (b) add a dedicated stanza to `compose_system.md` (Claude-favoured
   cache layer); (c) per-item few-shot example in `compose_dd.md`
   immediately before the candidate list.

2. **Haiku JSON truncation (3 errors in cycle 2).** Over-emits tokens
   and runs out of budget. Either reduce batch size or raise per-model
   `max_tokens` in the compose worker dispatch.

3. **Flash-lite `geometric_base` hallucination.** 6 out of 20 items had
   `geometric_base='vertical_coordinate'` or `'major_radius'`, both
   outside the closed vocabulary. The system prompt already enumerates
   the allowed values — flash-lite appears to ignore them. Probably
   needs a stricter grammar-field validator in the compose worker that
   rejects candidates with closed-vocab violations.

4. **`plasma_boundary_gap_angle` vs `angle_of_plasma_boundary_gap`.**
   Both are grammatically valid; reviewers disagree. Add an NC-rule in
   `compose_system.md` specifying preferred ordering for modifier-free
   geometric gap quantities.
