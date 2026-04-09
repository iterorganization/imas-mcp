# 10: Pipeline Fixes

**Status:** Pending
**Priority:** High — prevents wasted LLM budget
**Depends on:** 09 (LLM compose working)
**Effort:** 2-4 hours

## Problem

Two bugs in the current pipeline:

### Bug 1: Extract dedup is a no-op

`extract_worker` (workers.py:68-75) fetches existing standard names from the
graph but never uses them to filter candidates:

```python
existing = get_existing_standard_names()  # Fetched...
wlog.info("Found %d raw candidates, %d existing names", len(raw), len(existing))
return raw  # ...but never filtered!
```

Every `sn build` run will re-compose names that already exist in the graph,
wasting LLM budget.

### Bug 2: Confusing field naming

`compose_worker` stores its output in `state.validated` (line 158), which
is then read by the VALIDATE worker. But `validated` implies post-validation
data. This will confuse anyone reading the code.

The state fields should follow the pipeline: `candidates → composed → reviewed → validated`.

## Tasks

### 1. Fix extract dedup

In `extract_worker`, filter candidates against existing names:

```python
existing = get_existing_standard_names()
existing_sources = {n.get("derived_from_dd") or n.get("derived_from_signal")
                    for n in existing if n.get("derived_from_dd") or n.get("derived_from_signal")}
filtered = [c for c in raw if c.get("path", c.get("signal_id")) not in existing_sources]
```

### 2. Rename state fields

In `state.py`, rename for clarity:

| Old | New | Written by | Read by |
|-----|-----|-----------|---------|
| `validated` (used as compose output) | `composed` | compose_worker | review_worker |
| `reviewed` | `reviewed` | review_worker | validate_worker |
| (new) | `validated` | validate_worker | (terminal / publish) |

Update all references in `workers.py`.

### 3. Review caps existing names smartly

`_build_review_context()` (workers.py:439) caps existing names at 200:
```python
"existing_names": sorted(existing_names)[:200]
```

Instead, filter to names from the same IDS/domain as the current batch,
then cap at 200. This gives better dedup context.

## Acceptance Criteria

- Running `sn build` twice on the same IDS skips already-composed names on the second run
- State field names match the pipeline phase that writes them
- Review context includes domain-relevant existing names (not random 200)

## Testing

- Run `sn build --source dd --ids equilibrium` twice
- Second run should have fewer candidates (already-existing names filtered out)
- All tests pass with renamed state fields
