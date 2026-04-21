# 46 — Standard-name linking & review-cost integrity fixes

**Status:** Implemented
**Date:** 2026-04-21
**Scope:** graph integrity (source→SN links), observability (Review cost/tokens)

## Problem

Two unrelated integrity gaps in the Standard Names subsystem surfaced together
during a `sn status` review:

1. **Missing `PRODUCED_NAME` edges.** A growing population of
   `StandardNameSource` nodes carried `status='composed'` or
   `status='attached'` but had no edge to any `StandardName`. Consumers that
   walk `(sn:StandardName)<-[:PRODUCED_NAME]-(src:StandardNameSource)` silently
   lost provenance for these nodes, yet the sources looked "done" from a
   pipeline-state perspective. A parallel population of 58 `StandardName`
   nodes existed without any producing source — unreachable via the same
   traversal.

2. **NULL cost / tokens on every `Review` node.** All 1019 `Review` nodes in
   the graph had `cost_usd IS NULL`, `tokens_in IS NULL`, `tokens_out IS NULL`,
   so no budget post-mortem was possible despite the spend being ~fully
   instrumented in the LLM client. `sn generate` rotations report per-rotation
   cost correctly, but per-review accounting had been dark.

## Root cause

### Bug 1 — SET-before-MATCH in source linker

The Cypher used by both `workers.py` (`_update_sources_after_compose`,
`_update_sources_after_attach`) and `graph_ops.py` (`mark_sources_composed`,
`mark_sources_attached`) had the shape:

```cypher
MATCH (sns:StandardNameSource {id: $source_id})
SET sns.status = $status, sns.composed_name = $name  // write committed here
WITH sns
MATCH (sn:StandardName {id: $sn_id})
MERGE (sns)-[:PRODUCED_NAME]->(sn)
```

If the `StandardName` MATCH returned zero rows (target deleted mid-pipeline,
missing due to an earlier failure, id mismatch), the row filters out of the
`WITH` stream — but the `SET` has already committed in the previous clause.
The source was left stamped `composed`/`attached` with no edge. The batch-level
summary reported success because the writer compared `records returned after
the second MATCH` against `rows actually written`, and Neo4j doesn't surface
"partial" for a sequence of Cypher clauses separated by `WITH`.

### Bug 2 — Missing kwargs in review persistence

`graph_ops.write_reviews` accepts and writes `cost_usd`, `tokens_in`,
`tokens_out` in its Cypher (already correct), but the call chain that builds
Review records was:

```
review_review_worker
 └─> _review_single_batch        # returned {_items, _cost, _tokens, ...}
      └─> acall_llm_structured   # returned (result, cost, tokens)
 └─> _build_review_record(...)   # defaulted cost_usd/tokens_in/tokens_out=None
```

`_review_single_batch` discarded the prompt/completion split, and the
amortisation step in `review_review_worker` never computed a per-item cost to
pass down. `_build_review_record` therefore wrote `None` for every Review node.

## Fix

### Cypher restructure (all four writer paths)

Move the `StandardName` MATCH *before* the `SET`, so an absent target filters
the whole row out before any mutation:

```cypher
MATCH (sns:StandardNameSource {id: $source_id})
MATCH (sn:StandardName {id: $sn_id})         // <- before SET
SET sns.status = $status,
    sns.composed_name = $name,
    sns.produced_sn_id = $sn_id              // <- new scalar mirror for audit
MERGE (sns)-[:PRODUCED_NAME]->(sn)
```

Callers now compare `linked_count` (rows returned) against the input batch
size and emit a warning when they diverge. `produced_sn_id` is a new scalar
property on `StandardNameSource` — the adjacent scalar makes orphan-edge
detection one property scan instead of a MATCH, and makes the invariant
"`status in {composed,attached}` ⇒ `produced_sn_id IS NOT NULL`" trivially
checkable.

Files: `imas_codex/standard_names/workers.py`,
`imas_codex/standard_names/graph_ops.py`,
`imas_codex/schemas/standard_name.yaml`.

### Cost/token plumbing

- `LLMResult.__slots__` now carries `input_tokens` and `output_tokens` (prompt
  and completion tokens respectively, drawn from `response.usage.*`). Total
  `tokens` is unchanged so existing tuple-unpacking `result, cost, tokens =
  await acall_llm_structured(...)` remains compatible.
- `_review_single_batch` returns `_input_tokens` / `_output_tokens` in its
  result dict alongside `_cost` / `_tokens`, and falls back via `getattr` so
  test mocks that return a plain tuple still work.
- `review_review_worker` amortises batch cost/input/output tokens across the
  reviewed items (both the canonical reviewer and the secondary-model loop),
  and passes them to `_build_review_record`.

Files: `imas_codex/discovery/base/llm.py`,
`imas_codex/standard_names/review/pipeline.py`.

## Backfill

Orphan composed/attached sources (249 total) were partitioned by whether a
`StandardName` with a matching `source_paths` entry could be found:

- **51 orphan sources → 77 edges recovered** via a source_paths heuristic:
  `(s:StandardNameSource)` where `s.id IN sn.source_paths` (both in `dd:<path>`
  form). 30 sources had a single SN match, 21 had multiple (all distinct
  semantic siblings, all edges written).
- **198 orphan sources rolled back** to `status='extracted'` because no SN had
  them in `source_paths` — the SN was never produced (likely deleted by an
  abandoned compose cycle). These re-enter the normal compose queue.
- **58 orphan StandardNames remain** — pre-existing legacy from the enrichment
  pipeline, acknowledged in doc 35 (`metadata-qa-report`). Out of scope here.

Review cost/tokens are **not** backfillable: the LLM call history is not
retained per-review, only the resulting `reviewer_scores` JSON. From
2026-04-21 forward, new reviews will populate correctly.

## Post-fix graph state

```
StandardNameSource    extracted=4563  composed=1296  attached=378  ready=80  skipped=19
Orphan composed/attached sources: 0  (was 249)
Orphan StandardName (no source edge): 58  (unchanged, legacy)
Review nodes:         1019     with cost recorded: 0 (0.0%, will grow on next review cycle)
```

## Observability

`sn status` dashboard extended with three new tables, visible via
`uv run imas-codex sn status`:

- **Linking Integrity** — orphan SN count, orphan composed/attached source
  count. Both should stay at 0 (SN count excepted while the 58 legacy names
  remain).
- **Review Cost** — total Review nodes, % with `cost_usd` populated, cumulative
  USD, cumulative input/output tokens.
- **Review Cost by Reviewer Model** — per-model breakdown ordered by cost
  desc. Makes model-selection cost comparisons a single command.

## Tests

8 new tests total, all passing:

- `tests/standard_names/test_source_linking.py` (5 tests):
  - Cypher emits the `MATCH sn`-before-`SET` ordering in all four writer paths.
  - `produced_sn_id` is written alongside the MERGE.
  - Partial linking triggers a WARNING log.
- `tests/standard_names/test_review_cost.py` (3 tests):
  - `_build_review_record` forwards `cost_usd` / `tokens_in` / `tokens_out`.
  - `graph_ops.write_reviews` Cypher binds them to the Review node.
  - Amortisation in `review_review_worker` distributes batch totals.

Ran `uv run pytest tests/standard_names/test_review_pipeline.py
tests/standard_names/test_review_rubrics.py
tests/standard_names/test_source_linking.py
tests/standard_names/test_review_cost.py` → 35/35 passing.

Pre-existing unrelated failures not caused by this change:
`test_benchmark.py::TestCalibrationDataset::test_calibration_loads`
(dataset size drift), `test_graph_ops.py::TestCocosScalarDefaulting::test_vector_not_defaulted`
(cocos default scalar/vector heuristic).

## Commits

- `75a7e1a7` fix(sn): gate source linking on StandardName match to prevent orphan edges
- `bd246dba` test(sn): guardrail Cypher invariants for source→SN linking
- `097fbf24` fix(sn): populate Review cost_usd and token counts
- `66247232` feat(sn): surface linking integrity and review cost in sn status
- `e573f3a0` fix(sn): tolerate tuple LLM returns in review batch token accounting

## Follow-ups (not in this change)

- 58 legacy orphan StandardNames: defer to a targeted cleanup session. Need to
  decide whether to (a) delete them, (b) re-attach to a source via
  `source_paths` reverse search if any, or (c) keep them as curated nodes with
  a `provenance='manual'` marker. Doc 35 has the earlier analysis.
- Consider a `sn integrity` CLI command that asserts the two invariants
  (0 orphan sources, ≤N orphan SNs) and exits non-zero for CI use.
