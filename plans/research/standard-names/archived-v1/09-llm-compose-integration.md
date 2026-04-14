# Feature 09: LLM Compose Integration

**Status:** Pending
**Priority:** CRITICAL — blocks all downstream phases
**Depends on:** Features 05 (pipeline exists), 01 (grammar API exports)
**Parallel with:** 11a (schema alignment), 12 (benchmark enhancement)
**Estimated complexity:** Medium-high

---

## Problem

The compose worker (`workers.py:96–248`) uses a 13-keyword heuristic
(`_extract_physical_base()`) instead of LLM calls. The compose prompt
templates (`compose_dd.md`, `compose_signals.md`) exist and are well-designed
but are **not wired** into the pipeline. The benchmark module (`_run_model()`)
correctly uses `acall_llm_structured()` with the same prompts — proving the
integration pattern works — but the build pipeline itself cannot produce
quality names.

## Approach

Replace the heuristic compose worker with batched async LLM calls following
the proven `review_worker` pattern. Refactor the extract→compose data flow
to use `ExtractionBatch` objects (already defined in `sn/sources/base.py`)
instead of flat dicts.

---

## Phase 1: Extract→Compose Data Contract

The extract worker currently stores flat `list[dict]` in `state.candidates`.
The compose worker needs grouped batches with IDS/domain context for coherent
LLM prompts.

### Tasks

1. **Add `extraction_batches` field to `SNBuildState`**
   - File: `imas_codex/standard_names/state.py`
   - Type: `list[ExtractionBatch]` (from `sn/sources/base.py`)
   - Extract worker populates this instead of (or in addition to) `state.candidates`

2. **Refactor `extract_worker` to use source modules**
   - File: `imas_codex/standard_names/workers.py` (lines 34–88)
   - DD source: call `extract_dd_candidates()` from `sn/sources/dd.py`
   - Signals source: call `extract_signal_candidates()` from `sn/sources/signals.py`
   - Store results as `state.extraction_batches`
   - Keep `state.candidates` as a flat view for progress tracking

3. **Add compose model config**
   - Add `compose_model` field to `SNBuildState`
   - CLI: `--compose-model` option in `sn build` (default: `get_model("language")`)
   - File: `imas_codex/cli/sn.py`

### Acceptance Criteria
- Extract worker produces `ExtractionBatch` objects with group_key, items, existing_names
- State exposes both batched and flat views of candidates
- Compose model is configurable from CLI

---

## Phase 2: Wire LLM into Compose Worker

Replace the heuristic with batched async LLM calls matching the benchmark's
`_run_model()` pattern.

### Tasks

1. **Rewrite `compose_worker()` for batched LLM**
   - File: `imas_codex/standard_names/workers.py`
   - Remove `_compose_single()` and `_extract_physical_base()` entirely
   - Add `_compose_batch()` async function following `_review_batch()` pattern:
     ```python
     async def _compose_batch(
         batch: ExtractionBatch,
         model: str,
         grammar_enums: dict,
         wlog: logging.LoggerAdapter,
     ) -> tuple[list[dict], int, float, int]:
         """Compose standard names for one extraction batch via LLM."""
         prompt_template = "sn/compose_dd" if batch.source == "dd" else "sn/compose_signals"
         context = {
             "items": batch.items,
             "ids_name": batch.group_key,  # or facility/domain for signals
             "existing_names": sorted(batch.existing_names),
             **grammar_enums,
         }
         # For signals source, add facility and domain context
         if batch.source == "signals":
             context["facility"] = batch.group_key.split("/")[0]  # or from state
             context["domain"] = batch.group_key
         
         prompt_text = render_prompt(prompt_template, context)
         messages = [{"role": "user", "content": prompt_text}]
         
         result, cost, tokens = await acall_llm_structured(
             model=model,
             messages=messages,
             response_model=SNComposeBatch,
         )
         # Convert SNCandidate objects to pipeline dicts
         composed = []
         for c in result.candidates:
             composed.append({
                 "id": c.standard_name,
                 "source_type": batch.source,
                 "source_id": c.source_id,
                 **c.fields,
                 "confidence": c.confidence,
                 "reason": c.reason,
                 "units": _find_units(batch.items, c.source_id),
             })
         return composed, len(result.skipped), cost, tokens
     ```

2. **Integrate batch loop into compose_worker()**
   - Iterate over `state.extraction_batches`
   - Call `_compose_batch()` for each
   - Track cost in `state.compose_stats.cost`
   - Track tokens for logging
   - Respect `state.should_stop()` between batches
   - Record batch progress every N batches

3. **Reuse `_get_grammar_enums()`**
   - Already exists at line 258 — use it in compose worker too
   - Consider moving to a shared location (e.g., `sn/grammar_context.py`)

### Acceptance Criteria
- `sn build --source dd --ids equilibrium` produces LLM-generated names with subject, component, position qualifiers
- Cost and token tracking works in compose phase
- `sn build --dry-run` still skips LLM calls
- `sn build --source signals --facility tcv` works with signals prompt

---

## Phase 3: Prompt Enhancement

The existing compose prompts generate name + fields + confidence + reason,
but NOT rich documentation. Compose should produce a short `reason` only.
Full documentation generation is deferred to Feature 10 (DOCUMENT phase).

However, the prompts need minor improvements for quality:

### Tasks

1. **Add few-shot examples to compose prompts**
   - File: `imas_codex/llm/prompts/sn/compose_dd.md`
   - Add 5-8 diverse examples covering all grammar patterns:
     - Subject-qualified: `electron_temperature`
     - Component-qualified: `toroidal_component_of_magnetic_field`
     - Position-qualified: `electron_temperature_at_magnetic_axis`
     - Process-qualified: `power_due_to_ohmic`
     - Device-qualified: `bolometer_radiated_power`
     - Geometric base: `position_of_magnetic_axis`
   - Similarly for `compose_signals.md`

2. **Add system/user message split for prompt caching**
   - Current: single user message with everything
   - Better: system message (grammar rules + enums + examples) + user message (batch items)
   - System message is static → cacheable via `inject_cache_control()`
   - Modify `_compose_batch()` to use two-message format

3. **Validate compose output against grammar**
   - After LLM returns `SNComposeBatch`, run `parse_standard_name()` on each candidate
   - Log warnings for grammar-invalid names but keep them for review phase
   - Add `grammar_valid` field to pipeline dict

### Acceptance Criteria
- Compose prompts include diverse few-shot examples
- System/user message split enables prompt caching
- Grammar validation runs inline during compose

---

## Phase 4: Tests

### Tasks

1. **Unit tests for compose worker with mocked LLM**
   - File: `tests/sn/test_compose_worker.py`
   - Mock `acall_llm_structured` to return pre-built `SNComposeBatch`
   - Test: DD source produces expected pipeline dicts
   - Test: Signals source uses correct prompt template
   - Test: Dry-run mode skips LLM
   - Test: Empty candidates handled gracefully
   - Test: Cost tracking accumulates correctly
   - Test: `should_stop()` interrupts batch loop

2. **Integration test with extraction batches**
   - Verify `extract_worker` → `compose_worker` data flow
   - Verify `ExtractionBatch` → compose → validate chain

3. **Test prompt rendering**
   - Verify templates render without errors for DD and signals contexts
   - Verify grammar enums are populated

### Acceptance Criteria
- All tests pass with mocked LLM
- Coverage ≥95% on new/modified code in workers.py

---

## Files Modified

| File | Change |
|------|--------|
| `imas_codex/standard_names/workers.py` | Rewrite compose_worker, delete heuristics |
| `imas_codex/standard_names/state.py` | Add extraction_batches, compose_model |
| `imas_codex/cli/sn.py` | Add --compose-model flag |
| `imas_codex/llm/prompts/sn/compose_dd.md` | Add few-shot examples |
| `imas_codex/llm/prompts/sn/compose_signals.md` | Add few-shot examples |
| `tests/sn/test_compose_worker.py` | New test file |
| `tests/sn/test_workers.py` | Update existing tests |

## Documentation Updates

- AGENTS.md: Document `--compose-model` CLI option
- Prompt templates: Self-documenting via frontmatter
