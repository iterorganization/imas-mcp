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

(To be filled in after applying fixes and re-running.)

## Phase 4 — multi-reviewer consistency

(Pending.)

## Phase 5 — calibration updates

(Pending — will add exemplars/anti-patterns once scores stabilise.)
