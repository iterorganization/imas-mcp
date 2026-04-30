# 42 — Polling-Based Compose and Review Workers

**Status:** implementing
**Author:** architect agent
**Created:** 2025-07-23

## Problem

The SN compose and review pipelines use `asyncio.gather(*tasks)` to fan out
all batches at once, with an `asyncio.Semaphore` limiting concurrency. This
has three failure modes:

1. **Silent batch drops** — budget reservation math can reject batches before
   they run. Fixed partially in `20c96f71` (1.5× multiplier, adaptive budget)
   but the root cause remains: all reservations are attempted upfront.

2. **No work stealing** — if one worker stalls on a slow LLM response, others
   cannot pick up remaining batches. The semaphore serialises but doesn't
   redistribute.

3. **No crash recovery** — if the process dies mid-gather, completed batches
   are persisted (graph-state-machine) but there's no mechanism to resume
   from the last incomplete batch without re-running extract.

## Target Architecture

Replace the gather fan-out with **N independent polling tasks** per worker
type. Each task loops:

```
while not should_stop:
    items = claim_work(limit=batch_size)
    if not items:
        if globally_drained(): break
        await asyncio.sleep(jitter)
        continue
    lease = budget_manager.reserve(estimated_cost)
    if lease is None:
        await asyncio.sleep(jitter)
        continue
    process(items, lease)
    release_claim(items)
```

This is the same pattern used by `discovery/wiki/graph_ops.py` (claim_token +
ORDER BY rand() + retry_on_deadlock).

## Scope

**This task:** compose + review workers only.
**Deferred:** enrich and link workers remain on current model (follow-up todo).

## Work-Queue Semantics

### Compose Worker

**Node label:** `StandardNameSource`
**Eligible predicate:**
```cypher
MATCH (sns:StandardNameSource)
WHERE sns.status = 'extracted'
  AND (sns.claimed_at IS NULL
       OR sns.claimed_at < datetime() - duration($cutoff))
ORDER BY rand()
LIMIT $limit
```

**Claim:** SET `claimed_at = datetime()`, `claim_token = $token`
**Verify:** Second query `WHERE claim_token = $token` (two-step verify)
**On success:** existing `_update_sources_after_compose()` sets status='composed'
**On failure:** SET `claimed_at = null`, `claim_token = null`
**Stale claim timeout:** 300s (5 minutes, matching discovery default)

### Review Worker

**Node label:** `StandardName`
**Eligible predicate:**
```cypher
MATCH (sn:StandardName)
WHERE sn.validation_status = 'valid'
  AND sn.reviewed_name_at IS NULL
  AND (sn.claimed_at IS NULL
       OR sn.claimed_at < datetime() - duration($cutoff))
ORDER BY rand()
LIMIT $limit
```

For docs review (`target='docs'`), replace `reviewed_name_at` with
`reviewed_docs_at` and add `sn.reviewed_name_at IS NOT NULL` (docs gate).

**Claim/verify/stale:** Same token pattern as compose.
**On success:** `write_name_review_results()` stamps `reviewed_name_at`
**On failure:** Clear `claimed_at`, `claim_token`

### Batch Size

Compose and review operate on **batches** (multiple sources/names per LLM
call), not individual items. The claim function fetches `batch_size` items
which become one LLM call.

- **Compose batch_size:** Inherited from `get_generate_batch_config()` (default ~15 items)
- **Review batch_size:** `state.batch_size` (default 15 names)

## Concurrency Knobs

| Parameter | Default | Upper bound | Source |
|-----------|---------|-------------|--------|
| `compose_workers` | 2 | 4 | `get_compose_concurrency()` |
| `review_workers` | `state.concurrency` (2) | 3 | TurnConfig.concurrency |
| `compose_batch_size` | 15 | 30 | generate batch config |
| `review_batch_size` | 15 | 25 | state.batch_size |

Upper bounds are set by the OpenRouter rate ceiling observed in commit
`26b1d278` (~60 req/min for reasoning models).

## Budget Integration

Each worker independently reserves budget before processing:

```python
lease = budget_manager.reserve(estimated_cost)
if lease is None:
    # Budget exhausted — don't drop, just wait and retry
    await asyncio.sleep(random.uniform(1.0, 3.0))
    continue  # Next iteration will re-check
```

**Key invariant:** Budget-reserve failure causes `sleep + retry`, never
silent drop. If the budget is truly exhausted (pool ≤ 0 and no active
leases that will release), the termination check will catch it.

**BudgetManager thread safety:** The `threading.Lock` inside BudgetManager
protects all mutations. In an async context with `asyncio.to_thread()` for
graph operations, the critical sections (pure arithmetic) complete in
microseconds, so no deadlock risk. Multiple async workers calling `reserve()`
concurrently is safe.

## Termination Semantics

The turn-runner spawns N polling workers and awaits completion. A worker
terminates when ALL of:

1. **No claim returned** — the claim query returned zero items
2. **Globally drained** — a separate query confirms zero eligible items
   remain for this worker type (including those claimed by other workers)
3. **No active leases** — `budget_manager._reserved` is empty (no in-flight
   work from any worker)

Implementation via a shared `IdleTracker`:

```python
class IdleTracker:
    """Track how long all workers have been idle."""
    def __init__(self, timeout: float = 15.0):
        self._last_work = time.monotonic()
        self._lock = threading.Lock()
        self._timeout = timeout

    def mark_work(self):
        with self._lock:
            self._last_work = time.monotonic()

    def is_idle(self) -> bool:
        with self._lock:
            return time.monotonic() - self._last_work > self._timeout
```

A worker breaks out of its loop when:
```python
if not claimed_items:
    eligible = await asyncio.to_thread(count_eligible, worker_type)
    if eligible == 0 and idle_tracker.is_idle():
        break
    await asyncio.sleep(random.uniform(0.5, 2.0))
    continue
```

### Edge Cases

- **Budget exhausted but work remains:** Workers keep polling (sleep loop)
  but no lease is granted. They break on idle_timeout once all active leases
  complete. This is correct — exhausted budget is a stop condition for compose
  (the engine's `stop_fn`), not for review.

- **Worker mid-request when "idle" measured:** The `claimed_at IS NOT NULL`
  fence in the eligible-count query correctly excludes items being processed.
  The `active_leases > 0` check in the termination gate prevents premature
  exit.

- **All workers claim the same items:** Impossible — `ORDER BY rand()` +
  `claim_token` two-step verify means at most one worker wins each item.

## Interaction with Existing Phases

### Extract

Extract runs first as today. It produces `StandardNameSource` nodes with
`status='extracted'`. The compose workers poll for these nodes instead of
consuming `state.extracted` in-memory batches.

**Key change:** Extract still populates `state.extracted` for enrichment
and context (IDS prefetch, COCOS, domain vocab). But the compose worker
re-groups the claimed sources into LLM batches on-the-fly rather than
using pre-built batch boundaries.

**Alternative (simpler, chosen):** Keep extract's batching. Claim sources
individually from the graph, but batch them locally for the LLM call. This
preserves the per-batch enrichment (IDS context, domain vocab, COCOS) while
adding work-stealing.

### Validate / Consolidate / Persist

These three downstream workers are **unchanged**. They operate on
`state.composed`, `state.validated`, and `state.consolidated` in-memory
lists. Each compose-polling worker appends to `state.composed` as it
completes batches.

Thread safety for `state.composed.extend()`: Since asyncio is single-
threaded (GIL + event loop), concurrent `extend()` calls from different
worker tasks are safe — they interleave at the await boundary, not mid-call.

### Consolidate and Persist

Run after all compose workers have terminated (same as today's
`depends_on=["compose_phase"]` in the engine DAG).

### Review vs Compose Interaction

In `sn run` turns, review runs in a separate phase AFTER generate.
In standalone `sn review`, compose is not involved. No interaction.

### `--min-score` Regen Path

Regen operates on `StandardName` nodes (not `StandardNameSource`), selecting
via `reviewer_score < min_score`. This is a different queue entirely — it
creates new `StandardNameSource` nodes via `fetch_low_score_sources()` +
`extract_specific_paths()`, which then flow through the normal compose
pipeline. No conflict with polling — the regen-created sources get
`status='extracted'` and are claimed like any other.

## Failure Modes

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Worker crash (exception) | `asyncio.gather` captures it | Other workers continue; crashed sources have stale `claimed_at` → reclaimed after 5min |
| LLM timeout (>60s) | `acall_llm_structured` raises | Worker catches, releases claim, moves to next batch |
| Neo4j deadlock | `TransientError` | `@retry_on_deadlock()` with exponential backoff (5 attempts) |
| Orphaned claims | `claimed_at` > 5 minutes old | Next poll iteration reclaims them |
| Budget exhaustion mid-batch | `charge_soft()` records overspend | Overspend logged; no batch dropped |

## File Changes

| File | Change |
|------|--------|
| `standard_names/graph_ops.py` | Add `claim_compose_sources()`, `release_compose_claims()`, `count_eligible_compose()` |
| `standard_names/workers.py` | Refactor `compose_worker` from gather to polling loop |
| `standard_names/review/pipeline.py` | Refactor `review_review_worker` from gather to polling loop |
| `standard_names/review/graph_ops.py` | Add `claim_review_names()`, `release_review_claims()`, `count_eligible_review()` |
| `tests/standard_names/test_polling_claims.py` | Unit tests for claim functions |

## Rubber-Duck Review Response

### Q1: Termination — budget-exhausted vs no-eligible-work

**Issue raised:** When budget is exhausted but eligible work remains, workers
will idle-poll forever (budget.reserve returns None, so they sleep+retry).

**Resolution:** The compose worker already has a `state.should_stop()` check
that includes `budget_exhausted`. When budget is exhausted:
1. `state.should_stop()` returns True (for compose — downstream phases use
   a different stop function)
2. Workers break on `should_stop()` check at the top of the loop
3. This is correct because the engine's `stop_fn` for compose phases checks
   budget, while downstream `_downstream_should_stop` only checks
   `stop_requested`.

For review workers (which have their own BudgetManager), the same
`state.should_stop()` check applies — review state checks its own budget.

### Q2: BudgetManager memory model under concurrent reserve/release

**Issue raised:** Multiple async tasks calling `reserve()` concurrently.

**Resolution:** BudgetManager uses `threading.Lock` which is safe for asyncio
because:
- `reserve()` does only arithmetic (no I/O) under the lock
- Lock acquisition is effectively instant (microseconds)
- Even if two tasks call `reserve()` "simultaneously", they sequence through
  the lock and the invariant `pool + reserved + spent = total` is maintained
- `charge_soft()` uses the same lock pattern

No change needed.

### Q3: `--min-score` regen path interaction

**Issue raised:** Regen sources are StandardName nodes, not StandardNameSource.

**Resolution:** Regen creates new StandardNameSource nodes (via
`fetch_low_score_sources` → `extract_specific_paths` → `merge_standard_name_sources`)
with `status='extracted'`. These enter the compose polling queue identically
to first-run sources. No special handling needed.

### Q4: Review finishes before compose drains

**Issue raised:** In `sn run` turns, review runs as a separate phase after
generate completes. But what about standalone `sn review`?

**Resolution:** In standalone `sn review`, there is no compose phase. The
review workers poll `StandardName` nodes that already have
`validation_status='valid'`. No interaction. In `sn run` turns, the phase
ordering (generate → review) is enforced by the turn orchestrator — review
doesn't start until generate completes. No change needed.
