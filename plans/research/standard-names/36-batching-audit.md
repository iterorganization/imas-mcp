# 36 — Batching Audit and Token-Aware Improvements

**Status**: implemented
**Date**: 2025-07-24

## Current Behaviour (pre-change)

### Generate pipeline (`sn generate`)

Batching happens in `enrichment.py` via two grouping strategies:

1. **`group_by_concept_and_unit`** — groups enriched paths by
   `(grouping_cluster_id × unit)`.  Unclustered paths sub-group by
   `(IDS × parent_path × unit)`.  `max_batch_size=25` splits
   oversized groups.

2. **`group_for_name_only`** — groups by `(physics_domain × unit)`.
   `batch_size=50` (configurable via `name-only-batch-size` in
   pyproject).

**Problem**: cluster × unit fragments heavily. Most clusters contain
1–3 paths, yielding singleton batches that pay the full system-prompt
cost (~3 k tokens) for a single candidate. Logs show sustained
"Batch X: 1 composed" patterns.

### Enrich pipeline (`sn enrich`)

Batching in `enrich_workers.py::_build_batches` — simple sequential
chunking of claimed items into fixed-size slices (`batch_size=8` default
in state, `_ENRICH_BATCH_SIZE=10` module constant). No concept grouping,
no token awareness.

**Problem**: no pyproject knob for enrich batch-size default. Enrich packs
~800–1200 tokens per name (docs, exemplars, DD context) so batches should
be smaller (8–15) to stay within budget.

### Token budget

Neither pipeline estimates token usage before dispatch. The 200k
context window is never measured — oversized batches would silently
truncate or fail.

## Changes Implemented

### 1. Batching module (`imas_codex/standard_names/batching.py`)

New module with three responsibilities:

- **Grouping strategies**: `group_by_cluster_and_unit` (current, for full
  compose), `group_by_domain_and_unit` (name-only), `group_for_enrich`
  (sequential chunking with claim-token).
- **Token estimation**: `estimate_batch_tokens()` using `len(text) / 4`
  heuristic (tiktoken not a dependency).
- **Pre-flight check**: `pre_flight_token_check()` splits any batch
  exceeding `max_tokens` (default 150k, configurable) into sub-batches.

### 2. Configuration (`pyproject.toml`)

```toml
[tool.imas-codex.sn-generate]
batch-size = 25
name-only-batch-size = 50
max-tokens = 150000

[tool.imas-codex.sn-enrich]
batch-size = 12
max-tokens = 150000
```

### 3. Integration

- `enrichment.py` functions now delegate to `batching.py` for the
  chunking + splitting logic while retaining their context-building.
- `enrich_workers.py` reads `batch-size` from pyproject if state
  doesn't override.
- Pre-flight token check runs automatically after batch construction
  in both generate and enrich extract workers.

### 4. Before/After batch-size distribution

**Full compose (cluster × unit)**: unchanged — inherently fine-grained
because cluster scoping is semantically correct for rich-context mode.

**Name-only (domain × unit)**: already dense (mean ~13 at batch_size=50).
No change to grouping, but pre-flight check now guards against outliers.

**Enrich**: default moves from 10 → 12 (pyproject knob), with pre-flight
token check splitting any oversized batches.
