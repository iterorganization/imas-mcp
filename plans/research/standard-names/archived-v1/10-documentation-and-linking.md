# Feature 10: Documentation Generation

**Status:** Pending
**Priority:** High — names without documentation are low-value
**Depends on:** Feature 09 (LLM compose must be working)
**Parallel with:** 12 (benchmark Phase 1)
**Estimated complexity:** Medium-high

---

## Problem

The SN pipeline generates names with grammar fields and a short `reason`,
but no rich documentation. The existing 309-entry catalog demonstrates the
quality bar: entries like `electron_temperature` have 500+ word descriptions,
LaTeX equations, governing physics, measurement methods, typical values, and
cross-references to other standard names.

Generated names without this documentation will be rejected during catalog
review.

## Approach

Add a DOCUMENT phase after VALIDATE that generates rich documentation via
LLM. This is a **separate concern from link validation** (Plan 14) and
**separate from quality scoring** (Plan 12). The revised pipeline:

```
EXTRACT → COMPOSE → REVIEW → VALIDATE → DOCUMENT → PERSIST_NODES → LINK → SCORE → PERSIST_GRAPH
```

DOCUMENT focuses solely on generating high-quality physics documentation
for each validated name. Cross-reference mentions in the documentation are
recorded as structured data but **not validated here** — that's the LINK
phase's job (Plan 14) which runs after nodes are persisted to the graph.

---

## Phase 1: DOCUMENT Phase — LLM Documentation Generation

### Design

A new `document_worker()` that takes validated names and generates rich
documentation for each. This is a separate LLM call from compose because:

1. Compose focuses on *naming accuracy* (grammar fields, name selection)
2. Documentation focuses on *explanation quality* (physics context, equations,
   measurement methods, typical values, cross-references)
3. Different prompt structure: compose needs batch context (many paths at once),
   document needs deep context per name (similar names, DD path details, units)
4. Token budget: combining both would exceed context windows for complex names

### Tasks

1. **Create documentation prompt template**
   - File: `imas_codex/llm/prompts/sn/document.md`
   - System message (static, cacheable):
     - Role definition: fusion physics documentation expert
     - Documentation quality rubric (from review doc Section 5):
       - Outstanding: LaTeX equations, governing physics, measurement methods,
         typical values across devices (ITER/JET/DIII-D), sign conventions,
         cross-references
       - Good: physics context, relevant equations, measurement approaches
       - Adequate: definition, units, basic physics context
     - 3-5 exemplar entries at Outstanding tier (full YAML)
     - Grammar rules summary (for context, not for naming)
   - User message (dynamic):
     - The standard name and its grammar fields
     - Source DD path description and metadata
     - Related names from catalog (for cross-referencing)
     - Units and data type

2. **Create Pydantic response models**
   - File: `imas_codex/sn/models.py`
   ```python
   class SNDocumentation(BaseModel):
       """Rich documentation for a single standard name."""
       source_id: str
       standard_name: str
       kind: str  # physical, geometric, metadata
       description: str  # Rich markdown with LaTeX
       governing_equations: list[str]  # LaTeX equation strings
       measurement_methods: list[str]  # How this quantity is measured
       typical_values: dict[str, str]  # device → "range (units)"
       cross_reference_mentions: list[str]  # Other SN IDs mentioned in docs
       dependency_mentions: list[str]  # SNs this quantity functionally depends on
       tags: list[str]  # Classification tags
       sign_convention: str | None  # Sign convention notes
   ```
   **Key distinction**: `cross_reference_mentions` and `dependency_mentions`
   are **unvalidated** text extracted from the LLM output. They become
   graph relationships only after the LINK phase (Plan 14) validates them.

   ```python
   class SNDocumentBatch(BaseModel):
       """LLM response for batch documentation generation."""
       entries: list[SNDocumentation]
   ```

3. **Implement `document_worker()`**
   - File: `imas_codex/sn/workers.py`
   - Pattern: follows `review_worker()` structure
   - Reads from `state.validated` (or `state.reviewed`)
   - Batches: 3-5 names per LLM call (smaller batches than compose
     because documentation is verbose)
   - Stores results in `state.documented: list[dict]`
   - Tracks cost in `state.document_stats`
   - Uses `get_model("reasoning")` by default (documentation requires
     deeper physics knowledge)
   - CLI: `--document-model` and `--skip-document` flags

4. **Add DOCUMENT phase to pipeline**
   - File: `imas_codex/sn/pipeline.py`
   - New WorkerSpec between VALIDATE and PERSIST_NODES:
     ```python
     WorkerSpec(
         "document",
         "document_phase",
         document_worker,
         depends_on=["validate_phase"],
         enabled=not state.skip_document,
     ),
     ```

5. **Add state fields**
   - File: `imas_codex/sn/state.py`
   - `documented: list[dict]` — documented names
   - `document_stats: WorkerStats` — phase tracking
   - `document_phase: PipelinePhase` — supervision
   - `skip_document: bool = False` — CLI control
   - `document_model: str | None = None` — model override
   - Update `total_cost` property to include document phase

### Acceptance Criteria
- `sn build --source dd --ids equilibrium` generates documentation for each name
- Documentation includes governing equations, measurement methods, typical values
- Cross-reference mentions are extracted but NOT validated in this phase
- `--skip-document` flag bypasses the phase
- Cost tracking includes document phase

---

## Phase 2: Progress Display Updates

### Tasks

1. **Add DOCUMENT stage to progress display**
   - File: `imas_codex/sn/progress.py`
   - New `StageDisplaySpec` entry for document phase
   - Shows cost (LLM), rate, current name being documented

2. **Update pipeline summary**
   - File: `imas_codex/cli/sn.py`
   - Summary table includes document stats
   - Total cost includes compose + review + document phases

### Acceptance Criteria
- Progress display shows DOCUMENT stage with cost tracking
- Resource section includes updated cost totals

---

## Phase 3: Tests

### Tasks

1. **Test document worker**
   - File: `tests/sn/test_document_worker.py`
   - Mock `acall_llm_structured` returning `SNDocumentBatch`
   - Verify documentation fields are populated
   - Verify `cross_reference_mentions` and `dependency_mentions` extracted
   - Verify cost tracking
   - Test skip-document mode
   - Test empty input handling

2. **Test prompt template rendering**
   - Verify `sn/document.md` renders without errors
   - Verify exemplar entries are included
   - Verify system/user message split for cache efficiency

### Acceptance Criteria
- All tests pass with mocked LLM
- No graph or MCP dependency in tests

---

## Files Modified / Created

| File | Change |
|------|--------|
| `imas_codex/sn/workers.py` | Add `document_worker()` |
| `imas_codex/sn/state.py` | Add documented, document_stats, document_phase |
| `imas_codex/sn/pipeline.py` | Add DOCUMENT WorkerSpec |
| `imas_codex/sn/models.py` | Add SNDocumentation, SNDocumentBatch |
| `imas_codex/sn/progress.py` | Add document stage display |
| `imas_codex/cli/sn.py` | Add --skip-document, --document-model |
| `imas_codex/llm/prompts/sn/document.md` | New prompt template |
| `tests/sn/test_document_worker.py` | New test file |

## Documentation Updates

- AGENTS.md: Document DOCUMENT phase and CLI flags
- Prompt templates: Self-documenting via frontmatter
