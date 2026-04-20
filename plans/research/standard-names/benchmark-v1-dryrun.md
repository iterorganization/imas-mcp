# Benchmark v1 — scaffolding dry-run (2026-04-21)

> Plan 34 scaffolding validation.  This is a **mock** dry-run: no LLM
> calls were made; the runner's canned mock returns `expected_name`
> verbatim for every positive and score 0.15/fail for every negative,
> so the reported gate pass rates here **measure the harness plumbing,
> not model quality**.  A real (cost-capped) run comes after
> `cycle-cross-family-review` lands the reviewer pool.

## Command

```bash
uv run python scripts/run_benchmark_v1.py --mock --output plans/research/data
```

## Observations

- Eval set loaded cleanly: **20 filled positives + 10 negatives**;
  31 `TODO` stubs correctly skipped.
- Graph enrichment path exercised — cluster-sibling lookup + unit/
  description fetch ran against live Neo4j without error.
- Compose + review batches produced valid `ComposeBatch` / `ReviewBatch`
  pydantic objects.
- Outputs written:
  - `plans/research/data/benchmark-v1.positives.jsonl` (20 rows)
  - `plans/research/data/benchmark-v1.negatives.jsonl` (10 rows)
  - `plans/research/data/benchmark-v1.summary.json`
  - `plans/research/data/benchmark-v1-summary.md`
- Gating logic fires correctly: in mock mode all three gates report
  pass (expected — mock fabricates passing answers).  Per-domain
  breakdown populates with the 5 seeded domains
  (`core_plasma_physics`, `edge_plasma_physics`, `equilibrium`,
  `magnetic_field_diagnostics`, `transport`).  The five stress
  domains (`waves`, `fast_particles`, `turbulence`,
  `plasma_wall_interactions`, `gyrokinetics`) show n=0 because only
  `TODO` stubs are filed there — future-cycle work.

## Mock-run summary

| Gate | Actual | Threshold | Pass |
|------|-------:|----------:|:----:|
| pass_at_1 | 1.0 | 0.8 | ✅ |
| mean_reviewer_score | 0.9 | 0.75 | ✅ |
| negatives_rejection_rate | 1.0 | 0.9 | ✅ |

## Per-domain (positives)

| Domain | n | exact_rate | mean_score |
|--------|--:|-----------:|-----------:|
| core_plasma_physics | 4 | 1.0 | 0.9 |
| edge_plasma_physics | 2 | 1.0 | 0.9 |
| equilibrium | 7 | 1.0 | 0.9 |
| magnetic_field_diagnostics | 5 | 1.0 | 0.9 |
| transport | 2 | 1.0 | 0.9 |

## What the mock did NOT prove

- **No model-quality signal.**  Real LLM calls are required to
  measure true pass@1 and reviewer score; the mock just validates
  wiring.
- **No cross-family consensus.**  Single-reviewer still — the
  reviewer-pool wiring belongs to `cycle-cross-family-review`.
- **No cost calibration.**  Compose + review are batched into two
  LLM calls total (mirroring `prompt_ab_run.py`); once live, a
  run over 20 positives + 10 negatives should cost ~$0.15–$0.30
  on `opus-4.6` at current pricing.

## Next step

Land `cycle-cross-family-review` (reviewer pool) and then
(a) fill the 30 `TODO` stubs with one batch per stress domain and
(b) execute the first live run with `--cost-cap 1.00`.  Commit the
live-run summary under `plans/research/data/benchmark-v1-live-<date>.md`
at that point.
