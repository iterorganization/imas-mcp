# 19: SN Benchmark Parity, Lifecycle Management & Model Selection

**Status:** Ready to implement
**Supersedes:** Plans 16, 17, 18
**Scope:** DD source only — signals source parity is future work
**Agent type:** Fleet (4 phases, parallel where possible)

## Problem Statement

Three blockers prevent production-quality standard name minting:

1. **Benchmark/mint prompt parity gap** — `sn benchmark` uses user-only messages
   with thin grammar context. `sn mint` uses system/user split with full context
   (grammar rules, vocabulary, examples, cluster context). Benchmark results
   don't reflect production behavior, prompt caching can't work, and model
   selection decisions are unreliable.

2. **No reset/clear for standard names** — All other discovery domains have
   `--reset-to` infrastructure. StandardName is cross-facility (no `facility_id`),
   so needs adapted scoping by `review_status`, `source`, and `ids_filter`.

3. **Weak reviewer and thin reference set** — 30 reference entries from 4 IDSs,
   ad-hoc inline reviewer prompt, no calibration examples, no structured scoring.

## Architecture Notes

### StandardName node ownership model

A `StandardName` node can be linked to multiple DD paths or facility signals via
`(IMASNode)-[:HAS_STANDARD_NAME]->(sn)` relationships. `clear` and `reset`
operations must:

1. Filter/remove **relationships** first
2. Only delete the `StandardName` **node** if it becomes orphaned (no remaining
   `HAS_STANDARD_NAME` edges pointing to it)

This prevents corrupting names that are shared across sources.

### Prompt caching architecture

Caching is provider-side (OpenRouter). Our infrastructure already supports it:
- `inject_cache_control()` adds `cache_control: {"type": "ephemeral"}` breakpoints
- `openrouter/` model prefix preserves these blocks through LiteLLM
- OpenRouter returns `usage.cache_creation_input_tokens` and
  `usage.cache_read_input_tokens`

The fix is to use the same prompt architecture as mint, then confirm caching works
by reading cache token counts from the LLM response metadata.

### Lifecycle states (from current graph reality)

Only these `review_status` values are persisted: `drafted`, `published`, `accepted`.
The plan does not assume `rejected`, `skipped`, or `imported` exist unless
explicitly added.

---

## Phase 1: Foundation (2 parallel agents)

### Phase 1A: Benchmark prompt parity — Sonnet 4.6 (engineer)

**Files:**
- `imas_codex/standard_names/benchmark.py` — main changes
- `tests/sn/test_benchmark.py` — update tests

**Changes:**

1. **Replace `_run_model()` prompt construction** with the mint pipeline's pattern:
   ```python
   from imas_codex.standard_names.context import build_compose_context
   context = build_compose_context()
   system_prompt = render_prompt("sn/compose_system", context)
   # ... per batch:
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

2. **Render system prompt once** before the model loop in `run_benchmark()`,
   pass it to `_run_model()` as a parameter (matches mint pattern in workers.py:152).

3. **Fix `_extract_candidates()`** to preserve `batch.context` (cluster_context):
   ```python
   result.append({
       "group_key": batch.group_key,
       "items": batch.items,
       "existing_names": list(batch.existing_names),
       "context": batch.context,  # ADD THIS
   })
   ```

4. **Remove `build_grammar_context()`** from benchmark.py. It is replaced by
   `build_compose_context()` from `context.py`. The `coordinates` key currently
   maps to `Component` enum — this bug goes away because `build_compose_context()`
   handles it correctly.

5. **Update `run_benchmark()`** to pass `context` and `system_prompt` through.

6. **Basic cache smoke test**: After prompt parity changes, run benchmark with
   2+ batches and a single model. Log `cache_creation_input_tokens` and
   `cache_read_input_tokens` from the response. Phase 1A is not complete until
   cache usage is confirmed in logs. If the response object from
   `acall_llm_structured()` does not currently expose cache fields, add logging
   of the raw `usage` dict.

7. **Update tests**: `test_build_grammar_context_keys` in `test_benchmark.py`
   imports `build_grammar_context` — update or replace with tests that verify
   the benchmark uses `build_compose_context()`. Add test that extraction
   preserves `context` field. Add test that `_run_model()` constructs
   `[system, user]` messages.

**Acceptance criteria:**
- `sn benchmark` uses identical prompt construction as `sn mint`
- System/user message split enables prompt caching
- `cluster_context` is preserved through extraction
- Cache usage confirmed in logs (cache_creation or cache_read tokens > 0)
- All existing tests pass, new tests cover parity

### Phase 1B: Lifecycle management — Sonnet 4.6 (engineer)

**Files:**
- `imas_codex/standard_names/graph_ops.py` — add reset/clear functions
- `imas_codex/cli/sn.py` — add `sn reset` and `sn clear` commands
- `tests/sn/test_graph_ops.py` — add tests

**Changes:**

1. **Add `reset_standard_names()`** to graph_ops.py:
   ```python
   def reset_standard_names(
       *,
       from_status: str = "drafted",
       to_status: str | None = None,  # None = clear fields only
       source_filter: str | None = None,
       ids_filter: str | None = None,
       dry_run: bool = False,
   ) -> int:
   ```
   Fields to clear on reset: `embedding`, `embedded_at`, `model`, `generated_at`,
   `confidence`. Fields to preserve: `id`, `source`, `source_path`, `created_at`.
   Relationships to remove: `HAS_STANDARD_NAME`, `HAS_UNIT`.

2. **Add `clear_standard_names()`** to graph_ops.py:
   ```python
   def clear_standard_names(
       *,
       status_filter: list[str] | None = None,
       source_filter: str | None = None,
       ids_filter: str | None = None,
       include_accepted: bool = False,
       dry_run: bool = False,
   ) -> int:
   ```
   **Safety model (relationship-first):**
   - When `ids_filter` or `source_filter` is set, delete matching
     `HAS_STANDARD_NAME` relationships first
   - Then delete `StandardName` nodes that have zero remaining
     `HAS_STANDARD_NAME` edges (orphaned)
   - Default: only delete nodes with `review_status IN ['drafted']`
   - Accepted names require explicit `--include-accepted` flag (not just
     `--confirm` — the flag name must be unambiguous)
   - Always log count before deletion
   - `dry_run` returns count without deleting

3. **Add `sn reset` CLI command:**
   ```bash
   imas-codex sn reset --status drafted
   imas-codex sn reset --status published --to drafted
   imas-codex sn reset --status drafted --source dd --ids equilibrium
   imas-codex sn reset --dry-run
   ```

4. **Add `sn clear` CLI command:**
   ```bash
   imas-codex sn clear --status drafted
   imas-codex sn clear --status drafted --source dd --ids equilibrium
   imas-codex sn clear --all --include-accepted  # dangerous
   imas-codex sn clear --dry-run
   ```

5. **Wire `--reset-to` into `sn mint`:**
   ```python
   @click.option("--reset-to", type=click.Choice(["extracted", "drafted"]))
   ```
   - `extracted` = clear matching SN nodes, re-run full pipeline
   - `drafted` = reset existing drafted names, re-compose

6. **Tests:** Unit tests for `reset_standard_names()` and
   `clear_standard_names()`. Test that clear refuses accepted without
   `include_accepted`. Test relationship-first deletion logic. Test `--reset-to`
   triggers reset before pipeline.

**Acceptance criteria:**
- `sn reset --status drafted` resets nodes and clears embeddings
- `sn clear --status drafted` deletes only drafted names
- `sn clear --all` requires `--include-accepted` for accepted names
- Scoped clear (by IDS/source) removes relationships first, orphans only
- `sn status` shows correct counts after reset/clear
- `sn mint --reset-to drafted` resets then rebuilds

---

## Phase 2: Calibration & Reviewer (2 parallel agents)

**Depends on:** Phase 1A (benchmark parity must be fixed)

### Phase 2A: Expand gold reference — Sonnet 4.6 (engineer)

**Files:**
- `imas_codex/standard_names/benchmark_reference.py`
- `tests/sn/test_benchmark.py`

**Changes:**

1. Expand `REFERENCE_NAMES` from 30 to ~50 entries:

   | IDS | Current | Target |
   |-----|---------|--------|
   | equilibrium | 18 | 20 |
   | core_profiles | 6 | 10 |
   | magnetics | 4 | 6 |
   | summary | 2 | 4 |
   | core_transport | 0 | 4 |
   | mhd_linear | 0 | 2 |
   | nbi | 0 | 2 |
   | edge_profiles | 0 | 2 |

2. Fix the questionable rogowski_coil reference entry (maps `major_radius`
   position to `rogowski_coil` object, which is not physically meaningful).

3. Source new entries from `imas-standard-names` package's
   `resources/standard_name_examples/` where available. Use DD search tools
   to find appropriate source paths for each IDS.

4. All entries must pass round-trip validation at import time (existing
   infrastructure enforces this).

5. Update test that checks reference count.

**Acceptance criteria:**
- 50+ reference entries across 8+ IDSs
- All entries pass grammar round-trip
- Rogowski_coil entry fixed or removed
- Tests updated and passing

### Phase 2B: Calibration dataset & reviewer enhancement — Opus 4.6 (architect)

**Files:**
- `imas_codex/standard_names/benchmark_calibration.yaml` (new)
- `imas_codex/llm/prompts/sn/review_benchmark.md` (new template)
- `imas_codex/standard_names/benchmark.py` — update `score_with_reviewer()`
- `imas_codex/standard_names/benchmark_labels.yaml` — retire (replaced by calibration)
- `tests/sn/test_benchmark.py`

**Changes:**

1. **Create calibration dataset** (`benchmark_calibration.yaml`):
   ~15 hand-crafted full entries spanning 4 quality tiers. Source outstanding/good
   entries from `imas-standard-names` examples. Hand-craft poor examples.

   ```yaml
   entries:
     - name: electron_temperature
       tier: outstanding
       expected_score: 95
       description: "Temperature of the electron population."
       documentation: >
         Electron temperature $T_e$ is a fundamental plasma parameter...
       unit: eV
       kind: scalar
       tags: [core_profiles, equilibrium]
       fields:
         physical_base: temperature
         subject: electron
       reason: >
         Canonical physics quantity. Rich documentation. Perfect grammar.
   ```

   | Tier | Count | Score Range |
   |------|-------|-------------|
   | outstanding | 4 | 85-100 |
   | good | 4 | 60-79 |
   | adequate | 4 | 40-59 |
   | poor | 3 | 0-39 |

2. **Create reviewer prompt template** (`sn/review_benchmark.md`):
   Proper Jinja2 template with system/user split. Includes grammar rules,
   calibration entries as anchors, and structured rubric.

3. **Add 5-dimensional scoring** to `QualityReview` model:
   ```python
   class QualityReview(BaseModel):
       name: str
       quality_tier: str
       score: int = Field(ge=0, le=100)
       grammar_score: int = Field(ge=0, le=20)
       semantic_score: int = Field(ge=0, le=20)
       documentation_score: int = Field(ge=0, le=20)
       convention_score: int = Field(ge=0, le=20)
       completeness_score: int = Field(ge=0, le=20)
       reasoning: str
   ```

4. **Update `score_with_reviewer()`**: Replace inline rubric string with
   template rendering. Use system/user message split (enables caching for
   multi-batch reviewer runs). Load calibration entries and pass as template
   context.

5. **Retire `benchmark_labels.yaml`** — replaced by calibration dataset.
   Update `load_quality_labels()` or replace with calibration loader.

**Acceptance criteria:**
- Calibration YAML loads and validates (15 entries, 4 tiers)
- Reviewer uses Jinja2 template with system/user split
- 5-dimensional scores sum to 0-100
- Calibration entries appear as scoring anchors in reviewer prompt
- Running benchmark with `--reviewer-model` produces dimensional scores
- Tests for calibration loading, template rendering, review model

---

## Phase 3: Cache Reporting & Model Selection Runbook — Opus 4.6 (architect)

**Depends on:** Phase 2 (needs calibrated benchmark for meaningful results)

**Files:**
- `imas_codex/standard_names/benchmark.py` — add cache reporting to ModelResult and table
- `plans/features/standard-names/model-selection-runbook.md` (new)

**Changes:**

1. **Add cache hit reporting** to `ModelResult`:
   ```python
   cache_read_tokens: int = 0
   cache_creation_tokens: int = 0
   ```
   Extract from response metadata in `_run_model()`. Check what `litellm`
   exposes — OpenRouter includes `usage.cache_creation_input_tokens` and
   `usage.cache_read_input_tokens`. If `acall_llm_structured()` doesn't
   currently return these, extend its return or add response metadata logging.

2. **Add "Cache %" column** to `render_comparison_table()` Rich output.
   Calculate as `cache_read / (cache_read + cache_creation) * 100`.

3. **Create model selection runbook** — a document (in plans/) with:
   - Exact CLI commands to run multi-model comparison
   - Cost cap per run ($2 maximum per benchmark execution)
   - Approved model list with openrouter/ prefixes
   - Decision criteria table:

     | Metric | Weight | Threshold |
     |--------|--------|-----------|
     | Grammar valid % | Critical | ≥95% |
     | Fields consistent % | High | ≥85% |
     | Reference recall | High | ≥60% |
     | Avg quality score | High | ≥65 |
     | Cost per name | Medium | <$0.01 |
     | Names/min | Medium | >30 |
     | Cache hit rate | Low | >50% |

   - Recommended model candidates for compose and review phases
   - Instructions for interpreting results

4. **Run one validation benchmark** (cost-capped at $1) with a single model
   to confirm the full stack works end-to-end: prompt parity + caching +
   calibrated reviewer + dimensional scoring.

**Acceptance criteria:**
- Cache hit/miss tokens reported in benchmark output
- Model selection runbook written with exact CLI invocations
- One successful end-to-end benchmark run with dimensional scoring

---

## Phase 4: Documentation — Sonnet 4.6 (engineer)

**Depends on:** Phases 1-3

**Files:**
- `AGENTS.md` — update SN CLI section
- `.github/skills/project-dev/SKILL.md` — add SN testing info
- `.github/skills/service-ops/SKILL.md` — add LLM proxy for SN context
- `.github/agents/engineer.agent.md` — mention SN pipeline
- `agents/README.md` — update if needed
- `plans/features/standard-names/00-implementation-order.md` — update status
- Delete superseded plans: 16, 17, 18

**Changes:**

1. **AGENTS.md updates:**
   - Update CLI command table: add `sn reset`, `sn clear`, `sn mint --reset-to`
   - Update StandardName lifecycle section with reset/clear semantics
   - Add cache verification notes to benchmark section
   - Document model selection workflow

2. **Skill updates** (informative, not prescriptive):
   - `project-dev/SKILL.md`: Add SN test commands
     (`uv run pytest tests/sn/ -v`), note that SN tests don't require
     Neo4j unless marked `@pytest.mark.graph`
   - `service-ops/SKILL.md`: Add note that `sn mint` and `sn benchmark`
     require the LLM proxy to be running. Document how to check proxy
     status and what ports are involved. Mention that prompt caching is
     provider-side via OpenRouter.

3. **Agent updates** (informative, not prescriptive):
   - `engineer.agent.md`: Add SN module to list of commonly-modified
     areas (imas_codex/standard_names/, tests/sn/, imas_codex/llm/prompts/sn/)
   - `agents/README.md`: Mention SN pipeline if not already covered

4. **Plan cleanup:**
   - Delete `16-benchmark-parity.md`, `17-sn-lifecycle-management.md`,
     `18-benchmark-calibration.md` (superseded by this plan)
   - Update `00-implementation-order.md` with Plan 19 status

**Acceptance criteria:**
- All documentation reflects current CLI commands and workflows
- Skills are informative (describe what exists, not what to do)
- Superseded plans deleted
- Implementation order updated

---

## Fleet Dispatch Summary

| Phase | Agent | Model | Depends On | Parallel With |
|-------|-------|-------|------------|---------------|
| 1A | engineer | Sonnet 4.6 | — | 1B |
| 1B | engineer | Sonnet 4.6 | — | 1A |
| 2A | engineer | Sonnet 4.6 | 1A | 2B |
| 2B | architect | Opus 4.6 | 1A | 2A |
| 3 | architect | Opus 4.6 | 2A, 2B | — |
| 4 | engineer | Sonnet 4.6 | 3 | — |

**Total: 6 agent dispatches across 4 phases.**

Phase 1 is fully parallel (2 agents). Phase 2 is fully parallel (2 agents).
Phases 3 and 4 are sequential.

## Test Plan Summary

| Test | Phase | File |
|------|-------|------|
| Benchmark uses system/user messages | 1A | test_benchmark.py |
| Extraction preserves context field | 1A | test_benchmark.py |
| Cache tokens appear in logs | 1A | test_benchmark.py or manual |
| `build_grammar_context()` removed cleanly | 1A | test_benchmark.py |
| `reset_standard_names()` clears fields | 1B | test_graph_ops.py |
| `clear_standard_names()` relationship-first | 1B | test_graph_ops.py |
| Clear refuses accepted without flag | 1B | test_graph_ops.py |
| Reference set round-trip (50+ entries) | 2A | test_benchmark.py |
| Calibration YAML loads | 2B | test_benchmark.py |
| Reviewer template renders | 2B | test_benchmark.py |
| QualityReview 5-dimensional model | 2B | test_benchmark.py |
| Cache reporting in output | 3 | test_benchmark.py |
| E2E benchmark with scoring | 3 | manual (cost-capped) |
