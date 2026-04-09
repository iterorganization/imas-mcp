# 09: Standard Name Generation Pipeline

**Status:** Ready to implement
**Replaces:** Previous 09, 10, 11 (which are archived by this plan)
**Effort:** 4 phases, parallelizable

## Executive Summary

Replace the heuristic `_compose_single()` with LLM-backed composition using
rich prompt context from `imas-standard-names` backing functions.  The pipeline
runs as: **EXTRACT → COMPOSE → VALIDATE → PERSIST** via `sn build`.

Key design decisions (from rubber-duck critique):

1. **Existing names are a reuse list, not a banned list** — many DD paths can
   map to the same StandardName.  Dedup by `source_id`, not `standard_name`.
2. **VALIDATE stays separate** — it is the quality gate.  Reports `validate_valid`,
   `validate_invalid`, `fields_consistent` as distinct metrics.
3. **Checkpoint/resume via source-level skip** — extract skips sources already
   linked via `(:IMASNode)-[:HAS_STANDARD_NAME]->(sn)` unless `--force`.
4. **Explicit provenance** — persist model name, generation timestamp,
   `review_status=skipped` (v1), never default confidence to 1.0.
5. **No UnitOfWork** — graph MERGE is idempotent.  Resumability comes from
   source-level skip, not undo stacks.
6. **Relationship direction: entity → concept** — `(:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)`
   not `(:StandardName)-[:DERIVED_FROM]->(src)`.  Consistent with `MEASURES`.
7. **State fields are past tense** — `extracted`, `composed`, `reviewed`, `validated`.

## Architecture

```
sn build --source dd --ids equilibrium --limit 200

  EXTRACT (sync)           COMPOSE (async LLM)        VALIDATE (sync)          PERSIST (sync)
  ┌─────────────┐         ┌──────────────────┐       ┌───────────────┐       ┌──────────────────┐
  │ Query graph  │         │ Render prompts   │       │ Grammar parse │       │ MERGE StandardName│
  │ Skip named   │ ──────▶ │ acall_structured │ ────▶ │ Fields check  │ ────▶ │ HAS_STANDARD_NAME│
  │ Cluster batch│         │ Semaphore(5)     │       │ Normalize     │       │ Provenance props │
  └─────────────┘         └──────────────────┘       └───────────────┘       └──────────────────┘
       state.extracted       state.composed            state.validated          graph writes
```

### State Field Naming (all past tense)

| Field | Written by | Read by |
|-------|-----------|---------|
| `extracted` | extract | compose |
| `composed` | compose | review or validate |
| `reviewed` | review | validate |
| `validated` | validate | persist |

### Relationship Direction: IMASNode → StandardName

The StandardName is the canonical physics concept hub.  Entities point TO it:

```
(:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
(:FacilitySignal)-[:MEASURES]->(sn:StandardName)
```

**NOT** `(:StandardName)-[:DERIVED_FROM]->(src)`.  Reasons:

1. **StandardName is the anchor, not the dependent.** "electron_temperature" is
   a canonical concept that multiple DD paths and signals reference.  DERIVED_FROM
   puts the name as the dependent entity, which is semantically backwards.
2. **Consistent with FacilitySignal pattern.** Signals already use
   `MEASURES -> StandardName`.  DD paths should follow the same entity-to-concept
   direction: `HAS_STANDARD_NAME -> StandardName`.
3. **Natural traversal.** The most common query is "what standard name for this
   DD path?" — `(node)-[:HAS_STANDARD_NAME]->(sn)` is direct.
4. **Provenance is properties, not relationships.** How the name was generated
   (model, source_type, generated_at, confidence) are properties on StandardName,
   not a separate relationship.

This means `write_standard_names()` in `graph_ops.py` changes from:
```cypher
MERGE (sn)-[:DERIVED_FROM]->(src)  -- OLD: wrong direction
```
to:
```cypher
MERGE (src)-[:HAS_STANDARD_NAME]->(sn)  -- NEW: entity points to concept
```

### Prompt Design

**System prompt** (`sn/compose_system.md`) — static, prompt-cached (~15KB):
- Grammar rules, segment descriptions, critical distinctions
- Vocabulary tokens with descriptions (not bare enums)
- Field guidance and requirements
- Curated examples grouped by pattern (~42 examples)
- Tokamak parameter ranges for grounding documentation

**User prompt** (`sn/compose_dd.md`) — dynamic per-batch (~3KB):
- IDS name, cluster context
- DD paths with descriptions, units, data types
- Existing standard names to reuse (not avoid)

The system prompt content comes from `imas_standard_names` backing functions:
- `grammar.constants.SEGMENT_RULES` → segment descriptions + critical distinctions
- `grammar.constants.SEGMENT_TOKEN_MAP` → vocabulary with descriptions
- `grammar.constants.SEGMENT_TEMPLATES` → template application rules
- `grammar.field_schemas.FIELD_GUIDANCE` → per-field requirements
- `tools.grammar._build_canonical_pattern()` → composition pattern
- `resources/tokamak_parameters/` → grounding data
- `resources/standard_name_examples/` → curated examples

These are assembled into Jinja include files at build time (cached in-process).

### Batching Strategy

Hierarchical grouping for semantic coherence:

1. **Primary:** Semantic cluster (`IMASSemanticCluster` via `IN_CLUSTER`)
2. **Secondary:** IDS name (for unclustered paths or large clusters)
3. **Tertiary:** Path prefix depth 3 (for groups > 25)
4. **Cap:** 25 paths per batch

```
equilibrium paths (1583)
  ├── Cluster: "Radial Profile Coordinates" (12 paths) → 1 batch
  ├── Cluster: "Plasma Boundary Geometry" (39 paths) → 2 batches
  ├── Unclustered (1383 paths)
  │   ├── time_slice/profiles_1d (52) → 3 batches
  │   ├── time_slice/boundary (39) → 2 batches
  │   └── ...
```

### Existing Names = Reuse List

The graph model supports many-to-one: multiple DD paths can HAS_STANDARD_NAME
the same StandardName.  The LLM prompt should say:

> These standard names already exist.  **Reuse** them when the DD path measures
> the same quantity.  Only create a new name when no existing name fits.

Deduplication is by `source_id` (DD path), not `standard_name`.

### Repo Architecture (future direction)

With graph-backed generation, the repos have clear responsibilities:

| Repo | Role | Contains |
|------|------|----------|
| **imas-standard-names** | Grammar library + web docs | Grammar rules, enums, compose/parse, validation, Quarto rendering, tokamak params, examples |
| **imas-standard-names-catalog** | Export target | YAML files exported from graph via `sn publish` |
| **imas-codex** | Generation platform | LLM pipeline, graph storage, MCP tools, semantic search |

**imas-standard-names refactoring (Phase 1 prerequisite):**

The MCP server backing functions contain rich prompt context builders that are
currently private (`_build_canonical_pattern()`, `_get_segment_descriptions()`).
These should be made **public API** in the grammar library so imas-codex can
import them cleanly.  Specifically:

1. Make `tools.grammar._build_*()` functions public (drop underscore prefix)
2. Move them from `tools/` to `grammar/context.py` (they're grammar knowledge,
   not MCP tool internals)
3. Remove MCP server code from imas-standard-names (future — after pipeline proves out)

The generation capability (UnitOfWork, catalog CRUD, MCP tools) is migrating to
imas-codex.  The grammar library retains: rules definition, validation, resources,
web rendering.

---

## Phase 1: Prompt Templates

**Agent:** engineer (3 files, well-specified)
**Depends on:** nothing

Create the Jinja include files that assemble rich grammar context from
`imas_standard_names` backing functions.

### Files to Create

**`imas_codex/sn/context.py`** — Grammar context builder (replaces `build_grammar_context()`):

```python
"""Rich grammar context for SN compose prompts.

Imports segment rules, vocabulary, field guidance, and examples from
imas_standard_names backing functions.  Assembles them into template
variables for Jinja2 rendering.

Caches assembled context in-process (same pattern as wiki calibration).
"""

def build_compose_context() -> dict[str, Any]:
    """Build rich context dict for sn/compose_system.md template.

    Returns keys: grammar_rules, vocabulary, field_guidance, examples,
    tokamak_ranges, segment_descriptions, critical_distinctions,
    template_rules, plus the original enum lists for the user prompt.
    """
    ...
```

Internals:
- Import `SEGMENT_RULES`, `SEGMENT_ORDER`, `SEGMENT_TEMPLATES`,
  `SEGMENT_TOKEN_MAP` from `imas_standard_names.grammar.constants`
- Import `FIELD_GUIDANCE`, `TYPE_SPECIFIC_REQUIREMENTS` from
  `imas_standard_names.grammar.field_schemas`
- Import `_build_canonical_pattern()`, `_get_segment_descriptions()`,
  `_build_segment_usage_guidance()` from `imas_standard_names.tools.grammar`
  (make public wrappers if needed)
- Load tokamak parameters from `imas_standard_names/resources/tokamak_parameters/`
- Load curated examples from `imas_standard_names/resources/standard_name_examples/`
- Cache result in module-level dict with TTL (same pattern as `_wiki_calibration_cache`)

**`imas_codex/llm/prompts/sn/compose_system.md`** — System prompt template:

```markdown
---
name: sn/compose_system
description: Static system prompt for SN composition (prompt-cached)
task: composition
dynamic: false
---

You are a physics nomenclature expert generating standard names ...

## Grammar Rules

{{ grammar_rules }}

## Vocabulary

{{ vocabulary }}

## Segment Descriptions and Critical Distinctions

{{ segment_descriptions }}
{{ critical_distinctions }}

## Field Guidance

{{ field_guidance }}

## Composition Pattern

{{ canonical_pattern }}

## Examples

{{ examples }}

## Output Format
...
```

**`imas_codex/llm/prompts/sn/compose_dd.md`** — Rewrite user prompt (dynamic):

Strip the grammar rules (now in system prompt).  Keep only:
- IDS name, cluster context
- DD paths with descriptions, units, data types
- Existing standard names (as reuse list, not banned list)
- Output format specification

### Acceptance Criteria

- `build_compose_context()` returns a dict with all required keys
- System prompt renders to ~15KB of structured grammar context
- User prompt renders to ~3KB of per-batch data
- Grammar context includes segment descriptions (not bare enum lists)
- Tokamak parameters are loaded and available for documentation grounding
- At least 40 curated examples are included

### Tests

- Unit test: `build_compose_context()` returns expected keys
- Unit test: System prompt renders without errors
- Unit test: User prompt renders with sample batch data

---

## Phase 2: Extract + Batching Improvements

**Agent:** engineer (2-3 files, well-specified)
**Depends on:** nothing (parallel with Phase 1)

### 2a: Source-level skip (resumability)

In `graph_ops.py`, add a function to get already-named source IDs:

```python
def get_named_source_ids() -> set[str]:
    """Return source_ids already linked via HAS_STANDARD_NAME."""
    with GraphClient() as gc:
        results = gc.query("""
            MATCH (src)-[:HAS_STANDARD_NAME]->(sn:StandardName)
            RETURN DISTINCT src.id AS source_id
        """)
        return {r["source_id"] for r in results}
```

In `extract_worker`, skip already-named sources:

```python
named = get_named_source_ids()
raw = [c for c in raw if c.get("path", c.get("signal_id")) not in named]
wlog.info("Skipped %d already-named sources", len(named & ...))
```

Add `--force` flag to `sn build` CLI to bypass skip logic.

### 2b: Cluster-based batching

In `sources/dd.py`, replace flat IDS grouping with hierarchical batching:

```python
def _build_semantic_batches(results: list[dict], cap: int = 25) -> list[ExtractionBatch]:
    """Group paths by cluster → IDS → prefix for LLM-coherent batches."""
    # 1. Separate clustered vs unclustered
    # 2. Group clustered paths by cluster_id, sub-group by IDS
    # 3. Group unclustered paths by IDS, sub-group by prefix depth 3
    # 4. Split any group > cap into sub-batches
```

### 2c: State field rename

In `state.py`, rename all state fields to past tense:

| Old | New | Written by | Read by |
|-----|-----|-----------|---------|
| `candidates` | `extracted` | extract | compose |
| `validated` (used as compose output) | `composed` | compose | review or validate |
| `reviewed` | `reviewed` | review | validate |
| (new) | `validated` | validate | persist |

Update all references in `workers.py`, `pipeline.py`, `progress.py`.

### Acceptance Criteria

- `sn build` second run on same IDS skips already-derived names
- `sn build --force` re-processes all paths
- Batches are ≤25 paths, grouped by cluster/IDS/prefix
- State fields follow pipeline phase naming

### Tests

- Unit test: `get_derived_source_ids()` returns expected IDs
- Unit test: `_build_semantic_batches()` respects cap, groups by cluster
- Unit test: State field wiring (compose writes composed, validate writes validated)

---

## Phase 3: LLM Compose Worker

**Agent:** architect (5+ files, async LLM, prompt rendering)
**Depends on:** Phase 1 (prompt templates), Phase 2 (batching + state)

Replace `_compose_single()` heuristic with batch LLM calls.

### Core Changes

**`workers.py` — `compose_worker`:**

```python
async def compose_worker(state: SNBuildState, **_kwargs) -> None:
    """LLM-generate standard names from extracted batches.

    Uses acall_llm_structured() with system/user prompt split
    for prompt caching.  Runs batches concurrently with semaphore.
    """
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model
    from imas_codex.sn.context import build_compose_context

    model = get_model("language")
    context = build_compose_context()

    # Render system prompt once (cached via prompt caching)
    system_prompt = render_prompt("sn/compose_system", context)

    # Process batches concurrently
    sem = asyncio.Semaphore(5)
    tasks = []
    for batch in state.extracted:  # ExtractionBatch objects
        tasks.append(_compose_batch(batch, model, system_prompt, context, sem, state))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successful results
    composed = []
    for r in results:
        if isinstance(r, list):
            composed.extend(r)
        elif isinstance(r, Exception):
            state.compose_stats.errors += 1

    state.composed = composed
```

**`workers.py` — `_compose_batch`:**

```python
async def _compose_batch(
    batch: ExtractionBatch,
    model: str,
    system_prompt: str,
    context: dict,
    sem: asyncio.Semaphore,
    state: SNBuildState,
) -> list[dict]:
    """Compose standard names for a single batch via LLM."""
    async with sem:
        user_context = {
            "items": batch.items,
            "ids_name": batch.group_key,
            "existing_names": sorted(batch.existing_names)[:200],
            "cluster_context": batch.context,
        }
        user_prompt = render_prompt("sn/compose_dd", {**context, **user_context})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result, cost, tokens = await acall_llm_structured(
            model=model,
            messages=messages,
            response_model=SNComposeBatch,
        )

        state.compose_stats.cost += cost
        state.compose_stats.total_tokens += tokens
        state.compose_stats.record_batch(len(batch.items))

        return [c.model_dump() for c in result.candidates]
```

### Key Design Notes

- **System/user split**: System prompt is ~15KB static content.  With
  OpenRouter prompt caching, this is cached after the first call, reducing
  latency and cost for subsequent batches.
- **Semaphore(5)**: Limits concurrent LLM calls.  Matches wiki worker pattern.
- **ExtractionBatch objects**: compose_worker receives `state.candidates` as
  a list of ExtractionBatch (from Phase 2 batching).  Each batch has items,
  group_key, context, existing_names.
- **Cost tracking**: Per-batch cost accumulated on `state.compose_stats`.

### Validate Worker Update

Add fields-consistency check (from benchmark's `validate_candidate()`):

```python
# In validate_worker, after grammar round-trip:
from imas_standard_names.grammar import StandardName, compose_standard_name

fields = entry.get("fields", {})
if fields:
    sn = StandardName(**_parse_fields(fields))
    from_fields = compose_standard_name(sn)
    if from_fields != normalized:
        entry["fields_consistent"] = False
    else:
        entry["fields_consistent"] = True
```

Report distinct metrics:
- `validate_valid` — name passes grammar round-trip
- `validate_invalid` — name fails grammar parse
- `validate_fields_consistent` — fields compose to same name
- `validate_fields_inconsistent` — fields compose to different name

### Persist Worker (new)

Add a PERSIST phase after VALIDATE:

```python
async def persist_worker(state: SNBuildState, **_kwargs) -> None:
    """Write validated standard names to graph with provenance."""
    if state.dry_run:
        state.persist_phase.mark_done()
        return

    from imas_codex.settings import get_model
    from imas_codex.sn.graph_ops import write_standard_names

    # Enrich with provenance before writing
    model = get_model("language")
    for entry in state.validated:
        entry.setdefault("model", model)
        entry.setdefault("review_status", "skipped")  # v1: no review
        entry.setdefault("generated_at", datetime.now(UTC).isoformat())
        # confidence comes from LLM output — never default to 1.0

    written = write_standard_names(state.validated)
    state.stats["persist_written"] = written
    state.persist_phase.mark_done()
```

`write_standard_names()` must be updated to create `HAS_STANDARD_NAME`
relationships (entity → concept) instead of `DERIVED_FROM` (concept → entity):

```cypher
-- NEW: entity points to concept
UNWIND $batch AS b
MERGE (sn:StandardName {id: b.id})
SET sn += b.properties
WITH sn, b
MATCH (src:IMASNode {id: b.source_id})
MERGE (src)-[:HAS_STANDARD_NAME]->(sn)
```

### Pipeline Wiring

Update `pipeline.py` to wire 4 phases:

```python
workers = [
    WorkerSpec("extract", "extract_phase", extract_worker),
    WorkerSpec("compose", "compose_phase", compose_worker, depends_on=["extract_phase"]),
    WorkerSpec("validate", "validate_phase", validate_worker, depends_on=["compose_phase"]),
    WorkerSpec("persist", "persist_phase", persist_worker, depends_on=["validate_phase"]),
]
```

Review phase is removed from default pipeline.  Can be re-added later as an
optional phase between compose and validate.

### Schema Update

Add provenance fields and `HAS_STANDARD_NAME` relationship to `facility.yaml`:

```yaml
StandardName:
  attributes:
    # ... existing fields ...
    model:
      description: LLM model that generated this name
    review_status:
      description: Review lifecycle status
      range: StandardNameReviewStatus  # new enum: skipped, pending, reviewed
    generated_at:
      description: When this name was generated
      range: datetime

# In IMASNode (imas_dd.yaml) or a new relationship class:
# IMASNode needs a slot to express the HAS_STANDARD_NAME relationship.
# This follows the dual property + relationship model.
```

**Relationship declaration:** Add `has_standard_name` slot to IMASNode in
`imas_dd.yaml` (or equivalent):

```yaml
has_standard_name:
  description: Standard name for this DD path
  range: StandardName
  annotations:
    relationship_type: HAS_STANDARD_NAME
```

This creates both a property (`n.has_standard_name`) and a relationship
`(:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)` via `create_nodes()`.

> **Note:** Since StandardName is in `facility.yaml` and IMASNode is in
> `imas_dd.yaml`, the cross-schema reference needs verification during
> Phase 3 implementation.  The MERGE in `write_standard_names()` can create
> the relationship directly without the slot if cross-schema refs are complex.

### Acceptance Criteria

- `sn build --source dd --ids equilibrium --limit 50` produces StandardName
  nodes in the graph via LLM composition
- System prompt is rendered once, user prompt per batch
- Validate reports `validate_valid`, `validate_invalid`, `fields_consistent`
- Persist writes provenance (model, review_status, generated_at, confidence)
- Cost tracking works across batches
- `--dry-run` skips LLM calls and graph writes

### Tests

- Integration test: compose_worker with mocked LLM produces valid candidates
- Unit test: validate_worker reports fields_consistent metric
- Unit test: persist_worker enriches entries with provenance
- Unit test: pipeline wires 4 phases with correct dependencies

---

## Phase 4: End-to-End Verification

**Agent:** architect (testing + prompt tuning)
**Depends on:** Phases 1-3

### Steps

1. Run: `sn build --source dd --ids equilibrium --limit 50 --dry-run`
   - Verify: extraction count, batch sizes, no LLM calls
2. Run: `sn build --source dd --ids equilibrium --limit 50`
   - Verify: StandardName nodes created in graph
   - Verify: DERIVED_FROM relationships exist
   - Verify: provenance fields populated
   - Check: grammar validity rate (target > 80%)
   - Check: fields consistency rate (target > 70%)
3. Run again (same command):
   - Verify: source-level skip works (fewer new candidates)
4. Run: `sn build --source dd --ids equilibrium --limit 50 --force`
   - Verify: all paths re-processed
5. Inspect prompt quality:
   - Review system prompt size and content
   - Check prompt cache hit rate in logs
   - Compare composed names against benchmark reference set

### Acceptance Criteria

- End-to-end pipeline produces valid StandardName nodes
- Grammar validity rate > 80% on equilibrium paths
- Source-level skip prevents duplicate LLM spend on re-runs
- Cost tracking accurate within 5% of actual API cost

---

## Documentation Updates

After all phases complete:

| Target | Update |
|--------|--------|
| `AGENTS.md` | Document `sn build` pipeline architecture |
| `plans/features/standard-names/00-implementation-order.md` | Update to reflect new plan structure |
| `plans/README.md` | Update plan status |

## Archived Plans

Plans 09 (previous), 10, 11 are superseded by this consolidated plan.
The content from those plans is incorporated here:
- Plan 10 bug fixes → Phase 2 (source skip, state rename, batch improvement)
- Plan 11 publish validation → deferred until Phase 4 evaluation reveals needs
- Plan 09 schema providers → Phase 1 (context builder)
