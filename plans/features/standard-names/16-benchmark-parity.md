# 16: Benchmark / Build Prompt Parity & Caching

**Status:** Ready to implement
**Depends on:** None (standalone fix)
**Blocks:** 18 (calibration — benchmark results meaningless until parity fixed)
**Agent:** engineer

## Problem

The `sn benchmark` command constructs LLM prompts differently from `sn build`,
making benchmark metrics (cost, speed, quality) unreliable for production model
selection.

### Specific gaps

| Aspect | Build (`workers.py`) | Benchmark (`benchmark.py`) |
|--------|---------------------|---------------------------|
| System prompt | `sn/compose_system` via `build_compose_context()` | None |
| User prompt | `sn/compose_dd` with `cluster_context` | `sn/compose_dd` without `cluster_context` |
| Message structure | `[system, user]` — enables prompt caching | `[user]` — no caching possible |
| Context source | `build_compose_context()` (grammar rules, vocabulary, examples, tokamak ranges) | `build_grammar_context()` (bare enum lists only) |
| Prompt caching | `inject_cache_control()` on system message | Not applicable (no system message) |

### Impact

- Cost metrics inflated (no cache hits)
- Speed metrics pessimistic (larger uncached prompts)
- Quality metrics unreliable (model sees different context)
- Benchmark cannot validate whether caching actually works

## Phase 1: Shared prompt construction

**Files:** `imas_codex/sn/benchmark.py`

Replace custom prompt construction in `_run_model()` with the same path used by
`compose_worker()` in `workers.py`:

```python
# Current (broken):
prompt_context = {"items": items, "ids_name": group_key, "existing_names": ..., **grammar_ctx}
prompt_text = render_prompt("sn/compose_dd", prompt_context)
messages = [{"role": "user", "content": prompt_text}]

# Fixed:
from imas_codex.sn.context import build_compose_context
context = build_compose_context()
system_prompt = render_prompt("sn/compose_system", context)
user_context = {
    "items": items,
    "ids_name": group_key,
    "existing_names": sorted(existing)[:200],
    "cluster_context": batch.get("context", ""),
    **context,
}
user_prompt = render_prompt("sn/compose_dd", user_context)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]
```

- Remove `build_grammar_context()` from benchmark.py (dead code after this change)
- Keep the function in benchmark.py only if `score_with_reviewer()` needs it
- Preserve `batch.context` (cluster_context) through extraction → benchmark

### Extraction fix

`_extract_candidates()` currently drops `batch.context` when converting
`ExtractionBatch` to plain dicts (line 494-500). Fix:

```python
result.append({
    "group_key": batch.group_key,
    "items": batch.items,
    "existing_names": list(batch.existing_names),
    "context": batch.context,  # ADD THIS
})
```

## Phase 2: Cache hit reporting

**Files:** `imas_codex/sn/benchmark.py`, `imas_codex/discovery/base/llm.py`

Add cache hit/miss tracking to benchmark output:

1. Extend `acall_llm_structured()` return or use response metadata to detect
   prompt cache hits (OpenRouter returns `usage.cache_creation_input_tokens`
   and `usage.cache_read_input_tokens`)
2. Add to `ModelResult`:
   ```python
   cache_read_tokens: int = 0
   cache_creation_tokens: int = 0
   cache_hit_rate: float = 0.0
   ```
3. Display in Rich table: "Cache %" column

**Note:** This requires checking what `litellm` exposes in the response object.
The `usage` dict from OpenRouter includes cache fields. `acall_llm_structured()`
currently returns `(result, cost, tokens)` — we may need to return the full
usage dict or add a cache-specific return value.

## Phase 3: Render system prompt once, verify caching

The system prompt should be rendered **once** before the model loop, since it's
identical across all batches and models. This matches the build pipeline pattern
(line 152 in workers.py).

```python
context = build_compose_context()
system_prompt = render_prompt("sn/compose_system", context)

for model in config.models:
    result = await _run_model(
        model=model,
        extraction_batches=extraction_batches,
        config=config,
        reference=REFERENCE_NAMES,
        system_prompt=system_prompt,  # pass pre-rendered
        context=context,             # for user prompt rendering
    )
```

## Acceptance criteria

1. `sn benchmark` uses identical prompt construction as `sn build`
2. System/user message split enables prompt caching
3. `cluster_context` is preserved through extraction
4. Cache hit rate is reported in benchmark output
5. Running the same benchmark twice shows cache hits on second run
6. `build_grammar_context()` removed from benchmark.py (or moved to shared location)

## Test plan

- Unit test: `_run_model()` constructs `[system, user]` messages
- Unit test: extraction preserves `context` field
- Integration: run benchmark with `--verbose`, confirm cache token counts in logs
