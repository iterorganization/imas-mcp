# 37 — `sn` target unification: consolidating generate, enrich, review

**Status:** implemented 2026-04-20
**Related commits:**
- `4d2f96a6` + `c9acaaaf` — Phase 1: `sn generate --target {names,docs,full}` routing
- `8c1e8185` — Phase 2: `sn review --target {names,docs,full}` with three-rubric dispatch
- `7f32556b` — tests for the unified `--target` behaviour

## Problem

Before this change, standard-names generation had three overlapping entry points:

- `sn generate --name-only` — a boolean flag that made compose skip documentation.
- `sn enrich` — a separate subcommand that filled docs on already-named entries.
- `sn review --name-only` — a boolean flag that swapped the review rubric
  from 6 dimensions (/120) to 4 name dimensions (/80).

The surface area scaled poorly:

- Every new compose/review mode forced a new boolean and a new command.
- The existence of a standalone `sn enrich` command made the lifecycle
  (generate names → enrich docs → review) split across two top-level verbs
  with no shared option vocabulary.
- A future "review just the docs without re-scoring the name" rubric had no
  place to live — the name-only boolean was overloaded.

## Decision

Collapse all three axes behind a single `--target` choice with values
`names`, `docs`, `full`.

- `sn generate --target names` — compose name+grammar only (equivalent to the
  old `--name-only`).
- `sn generate --target docs` — run the five-phase enrichment pipeline on
  already-named entries (equivalent to `sn enrich`).
- `sn generate --target full` (default) — full compose+enrich pass.
- `sn review --target {names,docs,full}` — pick the matching rubric.

`sn enrich` stays as a thin back-compat alias routing into
`_run_sn_docs_generation`. The old `--name-only` flag is a back-compat alias
for `--target names` on both generate and review. When both are present,
`--target` wins.

## Three-rubric review

Review gains a new third rubric `docs` alongside `names` and `full`:

| target | prompt                     | response model                              | dims | max |
|--------|----------------------------|---------------------------------------------|------|-----|
| names  | `sn/review_name_only`      | `StandardNameQualityReviewNameOnlyBatch`    | 4    | 80  |
| docs   | `sn/review_docs` (new)     | `StandardNameQualityReviewDocsBatch` (new)  | 4    | 80  |
| full   | `sn/review`                | `StandardNameQualityReviewBatch`            | 6    | 120 |

The `docs` rubric scores `description_quality`, `documentation_quality`,
`completeness`, `physics_accuracy` — deliberately NOT including grammar,
semantic accuracy of the name, or naming convention. Those were already
reviewed in a prior `--target names` pass and should not be re-litigated
when reviewing only new prose.

## Fidelity rank and downgrade guard

Reviews are ranked by fidelity: `name_only` < `docs` < `full`.

A lower-fidelity review run will NOT overwrite a higher-fidelity
`review_mode` already stamped on a `StandardName` node unless `--force`
is passed. This prevents a cheap sweep — e.g. a nightly
`sn review --target names` — from clobbering a careful `--target full`
review that took real LLM budget to produce.

The enum `StandardNameReviewMode` on the `StandardName` schema class
grows from two values (`full`, `name_only`) to three by adding `docs`.

## Docs batch size

Docs generation tends to produce much longer LLM outputs per item than
name generation. The hard name-batch-size of 25 is too large once
documentation prose is involved; the enrich pipeline previously used
`[tool.imas-codex.sn-enrich].batch-size = 12`.

We surface this as a first-class pyproject setting under the unified
command:

```toml
[tool.imas-codex.sn-generate]
docs-batch-size = 12   # used by --target docs
```

Resolution order for `--target docs`:

1. Explicit `--docs-batch-size` CLI flag, if given.
2. `[tool.imas-codex.sn-generate].docs-batch-size` (default 12).
3. `[tool.imas-codex.sn-enrich].batch-size` (legacy fallback).
4. Hardcoded default 8.

## Migration

No migration is required. All existing reviewer data with
`review_mode ∈ {full, name_only}` remains valid; the new `docs` value
only appears on records produced by `sn review --target docs`. The old
`--name-only` flag continues to work as a back-compat alias.
