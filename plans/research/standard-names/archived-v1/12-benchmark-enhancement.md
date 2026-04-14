# Feature 12: Quality Scoring & Benchmark

**Status:** Pending
**Priority:** High — scoring grounds quality; benchmark drives model decisions
**Depends on:** Feature 11 Phase 2 (PERSIST_NODES), Feature 14 (LINK)
**Parallel with:** 14 (linker Phase 1) — scorer and linker can be developed simultaneously
**Estimated complexity:** High

---

## Problem

Two interrelated gaps:

1. **No quality scoring in the build pipeline.** Generated names are either
   accepted or rejected by the binary REVIEW phase, but there's no continuous
   quality metric. Discovery pipelines (code, wiki, path) all have rich
   scoring with calibrated dimensions — standard names need the same.

2. **Benchmark is entangled with review.** The current benchmark
   (`sn/benchmark.py`) contains a review worker that should be reusable
   independently. The benchmark should be a thin CLI that runs the build
   pipeline with different models and compares results — it should import
   the scoring module, not own it.

## Approach

**Two deliverables:**

- **Scorer module** (`sn/scorer.py`): Reusable quality scoring for standard
  names, usable by both the build pipeline (as a SCORE phase) and the
  benchmark (for comparison). Follows discovery scoring patterns.
- **Benchmark CLI** (`sn benchmark`): Thin orchestration that runs the build
  pipeline with different models, scores results, and compares. Imports
  from scorer — doesn't implement scoring itself.

### Score Architecture (follows discovery patterns)

The discovery pipeline's scoring system provides the template:

| Discovery Pattern | SN Equivalent |
|-------------------|---------------|
| `CodeScoreFields(BaseModel)` | `SNScoreFields(BaseModel)` |
| `max_composite()` | `gated_composite()` (semantic+units gate) |
| `sample_code_dimension_calibration()` | `sample_sn_calibration()` |
| `score_worker()` claim→score→persist | `score_worker()` claim→score→persist |
| `StageDisplaySpec` progress | `StageDisplaySpec` progress |

**Key difference from discovery scoring:** Discovery uses `max_composite()`
(composite = max of all dimensions). SN scoring uses a **gated composite**
where `score_semantic_accuracy` and `score_units_consistency` are gate
dimensions — if either is below threshold (0.3), the composite is forced
to 0.0 regardless of how good documentation or conventions are. A
well-documented wrong name is worthless.

### Score Dimensions

| Dimension | What It Measures | Gate? |
|-----------|-----------------|-------|
| `score_semantic_accuracy` | Does the name capture the physics concept? | Yes |
| `score_units_consistency` | Are the canonical units correct? | Yes |
| `score_catalog_convention` | Does naming match catalog patterns? | No |
| `score_documentation_grounding` | Documentation quality and accuracy | No |
| `score_link_quality` | Cross-reference completeness and accuracy | No |

**NOT included:** `score_grammar` — duplicates the VALIDATE phase which
already does deterministic grammar round-trip validation.

### Pipeline Position

```
EXTRACT → COMPOSE → REVIEW → VALIDATE → DOCUMENT → PERSIST_NODES → LINK → SCORE
```

SCORE is the final phase, running after LINK because `score_link_quality`
depends on link resolution results.

---

## Phase 1: Score Model & Composite Function

### Tasks

1. **Create `SNScoreFields` base model**
   - File: `imas_codex/standard_names/scorer.py`
   ```python
   from pydantic import BaseModel, Field

   class SNScoreFields(BaseModel):
       """Score dimensions for standard name quality assessment."""
       score_semantic_accuracy: float = Field(
           ge=0.0, le=1.0,
           description="How well the name captures the physics concept"
       )
       score_catalog_convention: float = Field(
           ge=0.0, le=1.0,
           description="Adherence to catalog naming conventions"
       )
       score_units_consistency: float = Field(
           ge=0.0, le=1.0,
           description="Correctness of canonical units"
       )
       score_documentation_grounding: float = Field(
           ge=0.0, le=1.0,
           description="Quality of generated documentation"
       )
       score_link_quality: float = Field(
           ge=0.0, le=1.0,
           description="Cross-reference completeness and accuracy"
       )

       def get_score_dict(self) -> dict[str, float]:
           """Return dict of dimension_name → score for graph persistence."""
           return self.model_dump()
   ```

2. **Implement gated composite function**
   - File: `imas_codex/standard_names/scorer.py`
   ```python
   GATE_DIMENSIONS = {"score_semantic_accuracy", "score_units_consistency"}
   GATE_THRESHOLD = 0.3
   DIMENSION_WEIGHTS = {
       "score_semantic_accuracy": 0.30,
       "score_catalog_convention": 0.15,
       "score_units_consistency": 0.25,
       "score_documentation_grounding": 0.20,
       "score_link_quality": 0.10,
   }

   def gated_composite(scores: SNScoreFields) -> float:
       """Compute gated composite score.

       If any gate dimension is below threshold, composite is 0.0.
       Otherwise, weighted mean of all dimensions.
       """
       score_dict = scores.get_score_dict()
       for gate_dim in GATE_DIMENSIONS:
           if score_dict.get(gate_dim, 0.0) < GATE_THRESHOLD:
               return 0.0
       return sum(
           score_dict[dim] * weight
           for dim, weight in DIMENSION_WEIGHTS.items()
           if dim in score_dict
       )
   ```

3. **Create scoring LLM prompt**
   - File: `imas_codex/llm/prompts/sn/score.md`
   - System message (static, cacheable):
     - Role: fusion physics quality assessor
     - Rubric for each dimension with examples at 5 levels (0.0, 0.25, 0.5, 0.75, 1.0)
     - Calibration examples from catalog at each quality tier:
       - Outstanding (0.9+): `electron_temperature` — full LaTeX, typical values, measurement methods
       - Good (0.7-0.9): `plasma_current` — solid physics context, equations
       - Adequate (0.5-0.7): `toroidal_magnetic_field` — correct but sparse docs
       - Poor (0.25-0.5): a name with wrong units or misleading semantics
       - Failing (<0.25): a name that doesn't match its physics concept
     - Gate rule explanation: semantic+units must be ≥0.3
   - User message (dynamic):
     - The standard name entry (all fields)
     - Its documentation
     - Its link resolution status (resolved/unresolved counts)
     - Related catalog entries for comparison
   - Response model: `SNScoreFields`

### Acceptance Criteria
- `SNScoreFields` validates all dimensions are 0.0-1.0
- `gated_composite()` returns 0.0 when gates fail
- `gated_composite()` returns weighted mean when gates pass
- Prompt includes calibration examples at 5 quality levels

---

## Phase 2: Dynamic Calibration

### Design

Follow the discovery scoring calibration pattern: periodically sample
already-scored StandardName nodes from the graph at 5 quality levels per
dimension, inject these as few-shot examples into the scoring prompt.
This keeps scoring consistent across runs and models.

### Tasks

1. **Implement calibration sampling**
   - File: `imas_codex/standard_names/scorer.py`
   ```python
   _calibration_cache: dict[str, Any] = {}
   _calibration_ttl: float = 300.0  # 5-minute TTL

   def sample_sn_calibration(gc: GraphClient) -> dict[str, list[dict]]:
       """Sample scored StandardName nodes at 5 levels per dimension.

       Returns: {dimension_name: [{name, score, snippet}, ...]}
       Follows discovery/code/scorer.py pattern.
       """
       now = time.time()
       if _calibration_cache.get("timestamp", 0) + _calibration_ttl > now:
           return _calibration_cache.get("data", {})

       calibration = {}
       for dim in SNScoreFields.model_fields:
           samples = gc.query(f"""
               MATCH (sn:StandardName)
               WHERE sn.{dim} IS NOT NULL
               WITH sn, sn.{dim} AS score
               ORDER BY score
               WITH collect({{name: sn.id, score: score,
                   desc: substring(sn.description, 0, 200)}}) AS all_items
               RETURN [
                   all_items[0],
                   all_items[toInteger(size(all_items)*0.25)],
                   all_items[toInteger(size(all_items)*0.5)],
                   all_items[toInteger(size(all_items)*0.75)],
                   all_items[size(all_items)-1]
               ] AS levels
           """)
           calibration[dim] = samples[0]["levels"] if samples else []

       _calibration_cache["data"] = calibration
       _calibration_cache["timestamp"] = now
       return calibration
   ```

2. **Integrate calibration into prompt**
   - Scoring prompt includes calibration examples when available
   - Falls back to static examples for first run (no scored nodes yet)
   - Calibration refreshed every 5 minutes (300s TTL)

### Acceptance Criteria
- Calibration samples 5 levels per dimension
- TTL cache prevents excessive graph queries
- Graceful fallback when no scored nodes exist

---

## Phase 3: SCORE Pipeline Worker

### Tasks

1. **Implement `score_worker()`**
   - File: `imas_codex/standard_names/workers.py`
   - Pattern: claim→score→persist→release (follows discovery workers)
   - Steps:
     1. Claim batch of `status='linked'` (or `status='persisted'`) nodes
     2. Rebuild calibration examples (periodically, via TTL cache)
     3. For each name: build scoring prompt with calibration
     4. Call LLM for structured output → `SNScoreFields`
     5. Compute `gated_composite()`
     6. Persist scores to graph (SET all score dimensions + composite + scored_at)
     7. Update status to `scored`
     8. Release claim
   - Batch size: 5-10 names per LLM call
   - Uses `get_model("language")` (scoring is classification, not generation)
   - Cost tracking in `state.score_stats`

2. **Add SCORE phase to pipeline**
   - File: `imas_codex/standard_names/pipeline.py`
   ```python
   WorkerSpec(
       "score",
       "score_phase",
       score_worker,
       depends_on=["link_phase"],
       enabled=not state.skip_score,
   ),
   ```

3. **Add state fields**
   - File: `imas_codex/standard_names/state.py`
   - `score_stats: WorkerStats` — phase tracking
   - `score_phase: PipelinePhase` — supervision
   - `skip_score: bool = False` — CLI control
   - `score_model: str | None = None` — model override

4. **Add progress display**
   - File: `imas_codex/standard_names/progress.py`
   - SCORE stage: shows scored count, average composite, cost
   - Shows gate failure count (how many names gated to 0.0)

5. **Add CLI flags**
   - File: `imas_codex/cli/sn.py`
   - `--skip-score` — bypass scoring phase
   - `--score-model` — override model for scoring
   - `sn build` summary includes score statistics:
     ```
     Scores: avg=0.72, gated=3/87, top=electron_temperature (0.95)
     ```

### Acceptance Criteria
- SCORE phase runs after LINK in `sn build`
- Scores persisted to graph with all 5 dimensions + composite
- Gate failures produce composite=0.0
- Cost tracking included in pipeline summary
- `--skip-score` bypasses the phase

---

## Phase 4: Benchmark Enhancement

### Design

The benchmark becomes a thin orchestration CLI that:
1. Runs `sn build` with different `--compose-model` values
2. Scores results using the shared `sn/scorer.py` module
3. Compares across models
4. Reports which model produces the best names

The benchmark does NOT implement its own scoring — it reuses the scorer.

### Tasks

1. **Refactor benchmark to use shared scorer**
   - File: `imas_codex/standard_names/benchmark.py`
   - Remove internal scoring logic
   - Import `SNScoreFields`, `gated_composite`, `sample_sn_calibration`
     from `sn/scorer.py`
   - `_run_model()` now:
     1. Runs pipeline (compose + review + validate)
     2. Calls scorer for each result → `SNScoreFields`
     3. Computes `gated_composite()` for each
     4. Returns model-level aggregate stats

2. **Add model comparison report**
   - File: `imas_codex/standard_names/benchmark.py`
   - Compare models on:
     - Average composite score
     - Gate failure rate
     - Per-dimension breakdown
     - Cost per name
   - Rich table output:
     ```
     Model Comparison (31 reference items):
     ┌────────────────────────┬────────┬────────┬────────┬────────┐
     │ Model                  │ Avg    │ Gated  │ Cost   │ Sem.   │
     ├────────────────────────┼────────┼────────┼────────┼────────┤
     │ openrouter/claude-4-so │  0.82  │  1/31  │ $0.45  │  0.89  │
     │ openrouter/gpt-4.1     │  0.76  │  3/31  │ $0.32  │  0.81  │
     │ openrouter/gemini-2.5  │  0.71  │  5/31  │ $0.28  │  0.75  │
     └────────────────────────┴────────┴────────┴────────┴────────┘
     ```

3. **Update reference set management**
   - Keep the existing 31-entry reference set
   - Add scored reference entries (name + expected score range)
   - Benchmark validates that scorer produces consistent scores for
     reference entries across models

4. **Update CLI**
   - File: `imas_codex/cli/sn.py`
   - `sn benchmark` remains unchanged interface-wise
   - Internally uses shared scorer instead of ad-hoc review
   - `sn benchmark --compare` shows side-by-side model comparison
   - `sn benchmark --score-only` scores existing results without re-running

### Acceptance Criteria
- Benchmark imports all scoring from `sn/scorer.py`
- No duplicate scoring logic in benchmark
- Model comparison report includes all 5 dimensions + composite
- Reference set includes score expectations
- `sn benchmark` works independently of `sn build` pipeline

---

## Phase 5: Standalone `sn score` Command

### Tasks

1. **Add `sn score` CLI command**
   - File: `imas_codex/cli/sn.py`
   - Score or re-score existing StandardName nodes in the graph
   - Modes:
     - `sn score` — score all unscored names
     - `sn score --all` — re-score everything
     - `sn score --name electron_temperature` — score specific name
   - Uses same scorer module as pipeline and benchmark
   - Useful for: re-scoring after model changes, scoring catalog mirrors

### Acceptance Criteria
- `sn score` scores unscored names
- Uses shared scorer module
- Results persisted to graph

---

## Phase 6: Tests

### Tasks

1. **Test score model and composite**
   - File: `tests/sn/test_scorer.py`
   - Test `SNScoreFields` validation
   - Test `gated_composite()`:
     - Gates pass → weighted mean
     - Semantic gate fails → 0.0
     - Units gate fails → 0.0
     - Both gates fail → 0.0
     - Edge cases: exactly at threshold (0.3)
   - Test `get_score_dict()` output format

2. **Test calibration**
   - File: `tests/sn/test_scorer.py`
   - Test `sample_sn_calibration()` with mocked graph
   - Test TTL cache behavior
   - Test fallback when no scored nodes exist

3. **Test score worker**
   - File: `tests/sn/test_score_worker.py`
   - Mock LLM returning `SNScoreFields`
   - Verify scores persisted to graph
   - Verify status transitions
   - Test batch processing

4. **Test benchmark model comparison**
   - File: `tests/sn/test_benchmark.py`
   - Test comparison report generation
   - Test reference set scoring consistency
   - Verify benchmark imports from shared scorer

### Acceptance Criteria
- All tests pass with mocked LLM and graph
- Gated composite tested for all gate combinations
- Calibration tested with TTL expiry
- Benchmark tested independently of build pipeline

---

## Files Modified / Created

| File | Change |
|------|--------|
| `imas_codex/standard_names/scorer.py` | NEW: SNScoreFields, gated_composite, calibration, score functions |
| `imas_codex/standard_names/workers.py` | Add score_worker() |
| `imas_codex/standard_names/pipeline.py` | Add SCORE WorkerSpec |
| `imas_codex/standard_names/state.py` | Add score_stats, score_phase, skip_score |
| `imas_codex/standard_names/progress.py` | Add score stage display |
| `imas_codex/standard_names/benchmark.py` | Refactor to use shared scorer |
| `imas_codex/cli/sn.py` | Add --skip-score, --score-model, sn score, sn benchmark --compare |
| `imas_codex/llm/prompts/sn/score.md` | NEW: scoring prompt template |
| `tests/sn/test_scorer.py` | NEW: scorer unit tests |
| `tests/sn/test_score_worker.py` | NEW: worker tests |
| `tests/sn/test_benchmark.py` | Updated: uses shared scorer |

## Documentation Updates

- AGENTS.md: Document SCORE phase, `sn score` command, scoring dimensions
- AGENTS.md: Document gated composite formula
- AGENTS.md: Document benchmark model comparison workflow
