# Standard Name Pipeline: Graph-Primary Architecture with StandardNameSource Nodes

## Problem Statement

The `sn generate` pipeline has two categories of problems:

### A. Extraction Coverage Gaps (6 bugs)

Empirical analysis of the live graph reveals:

- **11,441** dynamic IMASNode paths with descriptions are eligible for standard names
- **342** StandardName nodes exist (3% coverage)
- **1,042** unclustered quantity-type paths lack standard names
- Each run processes only **~251 unique paths** (2.2% of pool) due to LIMIT-on-rows
- Repeated runs **stall on the same front slice** (deterministic ORDER BY + post-LIMIT filter)

Root causes: LIMIT applied to expanded rows not paths, missing `cluster_scope` field,
post-LIMIT unnamed filter, inverted grouping policy.

### B. Architectural Weaknesses

The pipeline uses a hybrid in-memory/graph architecture:
- **EXTRACT and COMPOSE are in-memory** — if the pipeline crashes, all extraction work is lost
- **No pre-composition graph tracking** — no way to query "which paths are queued for SN generation?"
- **Workers can't independently discover work** — compose reads `state.extracted` (Python list)
- **Not generic across sources** — DD and signal extraction have separate code paths with no shared tracking
- **Status lost on DD clear** — if SN tracking were on IMASNode, `clear_dd_graph()` (DETACH DELETE) destroys it

## Design: StandardNameSource Intermediary Node

**Reviewed and approved through 4 rounds of rubber-duck critique.**

### Core Concept

Introduce `StandardNameSource` — a graph-primary work-tracking node that registers source
quantities needing standard names. It sits between source nodes (IMASNode,
FacilitySignal) and StandardName nodes, providing:

1. **Graph-primary orchestration** — workers claim StandardNameSource nodes from graph, not memory
2. **Explicit progress tracking** — `count(StandardNameSource WHERE status='extracted')` = remaining work
3. **Crash resilience** — graph state survives pipeline restarts
4. **Source independence** — survives DD clear+rebuild (not a DD node type)
5. **Generic across sources** — same node for DD paths and facility signals
6. **Minimal design** — stores only orchestration metadata; source descriptions and units
   fetched at compose time via relationship joins (no data duplication, no staleness)

### Schema

```yaml
StandardNameSource:
  description: >-
    Work-tracking node for standard name composition pipeline.
    Created by EXTRACT, claimed by COMPOSE. Minimal design — source
    metadata fetched at compose time via FROM_DD_PATH/FROM_SIGNAL joins.
    
    Lifecycle: extracted → composed | attached | vocab_gap | failed | stale
    
    Generic across source types. Survives DD clear+rebuild because
    it is NOT a DD node type.
  class_uri: sn:StandardNameSource
  attributes:
    id:
      identifier: true
      description: "Composite key: {source_type}:{source_id}"
      required: true
    source_type:
      range: StandardNameSourceType
      description: >-
        Source kind — only `dd` and `signals` are valid for pipeline nodes.
        `manual` and `reference` are StandardName provenance values that
        never enter the composition pipeline. Enforce via test.
      required: true
    source_id:
      description: ID of the source node (IMASNode.id or FacilitySignal.id)
      required: true
    status:
      range: StandardNameSourceStatus
      required: true
    classification:
      description: "Path classification: quantity"
    batch_key:
      description: "Grouping key for composition (e.g., cluster_123:eV)"
    extracted_from_dd_version:
      description: DD version at extraction time
    # Retry tracking
    attempt_count:
      description: Number of composition attempts (incremented on failure)
      range: integer
    last_error:
      description: Error message from most recent failed attempt
    failed_at:
      range: datetime
    # Worker coordination
    claimed_at:
      range: datetime
    claim_token:
      description: UUID for atomic claiming
    extracted_at:
      range: datetime
    composed_at:
      range: datetime

StandardNameSourceStatus:
  permissible_values:
    extracted:
      description: Registered for composition, awaiting LLM processing
    composed:
      description: StandardName successfully generated via LLM
    attached:
      description: Matched to existing StandardName without LLM (auto-attach)
    vocab_gap:
      description: Composition blocked by missing grammar vocabulary token
    failed:
      description: Composition failed after max attempts (terminal)
    stale:
      description: Source node no longer exists after DD rebuild/clear
```

### Relationships

```
(StandardNameSource)-[:FROM_DD_PATH]->(IMASNode)        — DD source link
(StandardNameSource)-[:FROM_SIGNAL]->(FacilitySignal)   — signal source link
(StandardNameSource)-[:PRODUCED_NAME]->(StandardName)   — composition result (lineage)
(IMASNode)-[:HAS_STANDARD_NAME]->(StandardName) — authoritative semantic mapping (unchanged)
```

`HAS_STANDARD_NAME` remains the authoritative link. `PRODUCED_NAME` tracks which
queue item produced which name (composition lineage / auditability).

### Indexes

- `StandardNameSource(status)` — claim queries
- `StandardNameSource(batch_key)` — batch grouping
- `StandardNameSource(source_id)` — reconciliation joins
- `StandardNameSource(claim_token)` — claim verification

### Lifecycle Diagram

```
                    ┌─────────────────┐
                    │    extracted     │ ← EXTRACT creates (MERGE)
                    └────────┬────────┘
                             │ COMPOSE claims batch
                    ┌────────┴────────┐
              ┌─────┤   compose try   ├─────┐─────────┐
              │     └─────────────────┘     │         │
              ▼                             ▼         ▼
     ┌────────────┐              ┌──────────────┐  ┌────────┐
     │  composed   │              │  vocab_gap   │  │attached│
     └────────────┘              └──────────────┘  └────────┘
                                        │
                            (attempt < max)
                                        │
                                        ▼
                               ┌────────────────┐
                               │   extracted     │ (retry)
                               └────────────────┘
                                        │
                           (attempt >= max)
                                        │
                                        ▼
                               ┌────────────────┐
                               │    failed       │ (terminal)
                               └────────────────┘

     Source deleted → ┌────────┐ → Source rebuilt → ┌───────────┐
                      │  stale │                    │ extracted  │
                      └────────┘                    └───────────┘
```

## New Pipeline Flow

### EXTRACT (graph writer, runs once)

1. Query IMASNode candidates (with all coverage gap fixes applied)
2. Classify each path via `classifier.py`
3. Only quantity-classified paths get StandardNameSource nodes (metadata/skip counted in stats only)
4. Enrich with cluster selection → compute `batch_key`
5. MERGE StandardNameSource nodes to graph (batch UNWIND) with FROM_DD_PATH relationships
6. ON CREATE: status=extracted; ON MATCH: only requeue if stale or --force
7. `extract_phase.mark_done()` when all batches written
8. Progress: "Extracted X quantity sources (Y metadata, Z skip)"

**Re-extract semantics (ON CREATE vs ON MATCH):**

```cypher
UNWIND $batch AS item
MERGE (src:StandardNameSource {id: item.id})
ON CREATE SET
  src.source_type = item.source_type,
  src.source_id = item.source_id,
  src.status = 'extracted',
  src.classification = item.classification,
  src.batch_key = item.batch_key,
  src.extracted_from_dd_version = item.dd_version,
  src.extracted_at = datetime()
ON MATCH SET
  src.status = CASE
    WHEN $force THEN 'extracted'
    WHEN src.status = 'stale' THEN 'extracted'
    ELSE src.status
  END,
  src.batch_key = CASE
    WHEN $force OR src.status IN ['extracted', 'stale'] THEN item.batch_key
    ELSE src.batch_key
  END,
  src.extracted_from_dd_version = item.dd_version,
  src.claimed_at = CASE WHEN $force OR src.status = 'stale' THEN null ELSE src.claimed_at END,
  src.claim_token = CASE WHEN $force OR src.status = 'stale' THEN null ELSE src.claim_token END,
  src.attempt_count = CASE WHEN $force THEN 0 ELSE src.attempt_count END,
  src.last_error = CASE WHEN $force THEN null ELSE src.last_error END,
  src.failed_at = CASE WHEN $force THEN null ELSE src.failed_at END
WITH src, item
MATCH (p:IMASNode {id: item.source_id})
MERGE (src)-[:FROM_DD_PATH]->(p)
```

### COMPOSE (graph-primary claim loop)

1. `has_work_fn`: `MATCH (src:StandardNameSource {status: 'extracted'}) WHERE src.claimed_at IS NULL RETURN count(src) > 0`
2. Atomic full-batch claim (all members of a batch_key in single transaction)
3. Verify claim + fetch source metadata via OPTIONAL MATCH join to IMASNode
4. Handle missing sources → mark stale (token-verified)
5. LLM compose → persist StandardName → create PRODUCED_NAME + HAS_STANDARD_NAME
6. Mark StandardNameSource composed/attached/vocab_gap (token-verified)
7. On failure: increment attempt_count, return to extracted or terminal failed
8. Progress: "Composed X/Y sources (Z attached, W vocab_gap, V failed)"

**Atomic full-batch claim (single transaction):**

```cypher
MATCH (src:StandardNameSource {status: 'extracted'})
WHERE src.claimed_at IS NULL 
   OR src.claimed_at < datetime() - duration({seconds: $timeout})
WITH src.batch_key AS bk, collect(src) AS all_members
WHERE ALL(m IN all_members WHERE
  m.claimed_at IS NULL OR m.claimed_at < datetime() - duration({seconds: $timeout})
)
ORDER BY rand()
LIMIT 1
UNWIND all_members AS src
SET src.claimed_at = datetime(), src.claim_token = $token
RETURN all_members[0].batch_key AS batch_key, size(all_members) AS batch_size
```

**Verify + fetch source metadata:**

```cypher
MATCH (src:StandardNameSource {claim_token: $token})
OPTIONAL MATCH (p:IMASNode {id: src.source_id})
OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (p)-[:IN_IDS]->(ids:IDS)
OPTIONAL MATCH (p)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
OPTIONAL MATCH (p)-[:HAS_PARENT]->(parent:IMASNode)
RETURN src.id AS standard_name_source_id, src.source_id AS path,
       p IS NOT NULL AS source_exists,
       p.description, p.documentation, u.id AS unit,
       ids.id AS ids_name, c.id AS cluster_id, c.label AS cluster_label,
       c.scope AS cluster_scope, parent.id AS parent_path,
       p.cocos_label_transformation AS cocos_label,
       p.physics_domain, p.data_type
```

**Token-verified failure with durable retry:**

```cypher
MATCH (src:StandardNameSource {claim_token: $token})
SET src.attempt_count = coalesce(src.attempt_count, 0) + 1,
    src.last_error = $error_msg,
    src.failed_at = datetime(),
    src.status = CASE
      WHEN coalesce(src.attempt_count, 0) + 1 >= $max_attempts THEN 'failed'
      ELSE 'extracted'
    END,
    src.claimed_at = null,
    src.claim_token = null
```

**All state transitions are token-verified:**

```python
def mark_sources_composed(token, results) -> int:
    # MATCH by claim_token, not just id

def mark_sources_stale(token, source_ids) -> int:
    # MATCH by claim_token

def release_batch_claim(token) -> int:
    # MATCH by claim_token
```

Every helper returns `count(src)` for logging/assertion.

### VALIDATE, CONSOLIDATE, PERSIST (unchanged)

Operate on StandardName nodes directly. No changes needed.

### Reconciliation after DD clear+rebuild

```cypher
MATCH (src:StandardNameSource {source_type: 'dd'})
WHERE NOT (src)-[:FROM_DD_PATH]->(:IMASNode)
WITH src
OPTIONAL MATCH (p:IMASNode {id: src.source_id})
WITH src, p
FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
  MERGE (src)-[:FROM_DD_PATH]->(p)
)
SET src.status = CASE
  WHEN p IS NOT NULL AND src.status = 'stale' THEN 'extracted'
  WHEN p IS NULL THEN 'stale'
  ELSE src.status
END,
src.claimed_at = CASE
  WHEN p IS NOT NULL AND src.status = 'stale' THEN null
  ELSE src.claimed_at
END,
src.claim_token = CASE
  WHEN p IS NOT NULL AND src.status = 'stale' THEN null
  ELSE src.claim_token
END
```

## Coverage Gap Fixes (integrated into new architecture)

### Gap 1: LIMIT on rows → LIMIT on paths (CRITICAL)

Move LIMIT before OPTIONAL MATCH fan-out in extraction query:

```cypher
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE {filters}
WITH DISTINCT n, ids
ORDER BY ids.id, n.id
LIMIT $limit
// THEN join enrichment
OPTIONAL MATCH (n)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
...
```

### Gap 2: Stall prevention (CRITICAL)

With StandardNameSource, the stall problem is eliminated by design:
- EXTRACT creates StandardNameSource with status=extracted (only for unnamed paths)
- COMPOSE claims from StandardNameSource, not from the raw query
- Re-runs: ON MATCH preserves composed/attached status, only processes new/stale

Additionally, add pre-LIMIT unnamed exclusion for the EXTRACT query:

```cypher
AND NOT EXISTS { MATCH (n)-[:HAS_STANDARD_NAME]->(:StandardName) }
```

And pre-LIMIT StandardNameSource exclusion (skip paths already registered):

```cypher
AND NOT EXISTS { MATCH (:StandardNameSource {source_id: n.id, source_type: 'dd'})
                 WHERE NOT (_.status IN ['stale', 'failed']) }
```

### Gap 3: Missing `cluster_scope` (MODERATE-HIGH)

Add `c.scope AS cluster_scope` to extraction query RETURN clause.

### Gap 4: Inverted grouping policy (MODERATE)

Implement `select_grouping_cluster()` with reversed scope priority:
- **Primary cluster** (per-item context): IDS > domain > global
- **Grouping cluster** (batch_key): global > domain > IDS

### Gap 5: Unclustered path handling (LOW-MODERATE)

Use `unclustered/{ids_name}/{parent_path}/{unit}` for batch_key.
Rootless paths use IDS name as fallback.

### Gap 6: Observability (LOW)

Report unique path count, not row count.

### Gap 7: Source type inconsistency (NEW — from RD review)

Normalize `signal` vs `signals` to match the `StandardNameSourceType` enum (`signals`).

## Implementation Plan

### Phase 0: Naming Standardization (prerequisite)

Standardize all `SN` abbreviations to `StandardName` / `standard_name` across
schema, code, tests, and graph. This must land first — all subsequent phases use
the new naming.

**Scope exclusion — public `sn` namespace retained:**
- CLI command group: `sn generate`, `sn review`, `sn status`, etc. (`imas_codex/cli/sn.py` file name)
- Config namespace: `[tool.imas-codex.sn.benchmark]` in `pyproject.toml`
- Config accessors: `get_sn_benchmark_compose_models()`, `get_sn_benchmark_reviewer_model()`
- File names: `imas_codex/llm/sn_tools.py`, `tests/standard_names/test_sn_tools.py`

These are public-facing surfaces where `sn` is a deliberate CLI abbreviation for "standard name".
The rename applies to **internal class/model/state/file identifiers** only.

#### Phase 0a: Rename Pydantic models, state classes, and progress display

Mechanical rename of 18 Python classes and all their import sites (~309 references).

**Renames (models.py):**

| Old | New |
|-----|-----|
| `SNCandidate` | `StandardNameCandidate` |
| `SNVocabGap` | `StandardNameVocabGap` |
| `SNAttachment` | `StandardNameAttachment` |
| `SNComposeBatch` | `StandardNameComposeBatch` |
| `SNProvenance` | `StandardNameProvenance` |
| `SNPublishEntry` | `StandardNamePublishEntry` |
| `SNPublishBatch` | `StandardNamePublishBatch` |
| `SNReviewVerdict` | `StandardNameReviewVerdict` |
| `SNReviewItem` | `StandardNameReviewItem` |
| `SNReviewBatch` | `StandardNameReviewBatch` |
| `SNQualityScore` | `StandardNameQualityScore` |
| `SNQualityReview` | `StandardNameQualityReview` |
| `SNQualityReviewBatch` | `StandardNameQualityReviewBatch` |
| `SNEnrichItem` | `StandardNameEnrichItem` |
| `SNEnrichBatch` | `StandardNameEnrichBatch` |

**Renames (state + progress):**

| Old | New |
|-----|-----|
| `SNBuildState` (state.py) | `StandardNameBuildState` |
| `SNReviewState` (review/state.py) | `StandardNameReviewState` |
| `SNProgressDisplay` (progress.py) | `StandardNameProgressDisplay` |

**Files impacted** (~20 source + test files):
- `imas_codex/standard_names/models.py` — class definitions
- `imas_codex/standard_names/state.py` — `SNBuildState`
- `imas_codex/standard_names/review/state.py` — `SNReviewState`
- `imas_codex/standard_names/progress.py` — `SNProgressDisplay`
- `imas_codex/standard_names/workers.py` — imports + type annotations
- `imas_codex/standard_names/pipeline.py` — imports
- `imas_codex/standard_names/publish.py` — imports + type annotations
- `imas_codex/standard_names/benchmark.py` — imports + type annotations
- `imas_codex/standard_names/review/pipeline.py` — imports + type annotations
- `imas_codex/standard_names/review/consolidation.py` — imports
- `imas_codex/cli/sn.py` — imports
- `tests/standard_names/test_scoring.py` — imports
- `tests/standard_names/test_benchmark.py` — imports
- `tests/standard_names/test_graph_state_machine.py` — imports
- `tests/standard_names/test_catalog_import.py` — imports
- `tests/standard_names/test_integration.py` — imports

**Documentation files also requiring SN→StandardName updates:**
- `AGENTS.md` — `SNComposeBatch`, `SNAttachment` in module descriptions
- `docs/architecture/standard-names.md` — `SNComposeBatch`, `SNCandidate`, `SNBuildState` in examples
- `plans/features/sn-bootstrap-loop.md` — `SNQualityReviewBatch` references (active plan)
- `docs/architecture/boundary.md` — `SNQualityScore` references
- `plans/features/standard-names/pending/20-consistency-and-prompt-enrichment.md` — `SNCandidate`, `SNComposeBatch` references

**Method:** `sed -i 's/\bSNCandidate\b/StandardNameCandidate/g'` etc. across all files,
then `uv run ruff check --fix . && uv run ruff format .` to clean up.

**Acceptance:** `uv run pytest tests/standard_names/` passes with zero `SN[A-Z]` references
remaining in source files (excluding completed plans and SNAPSHOT log strings).

#### Phase 0b: Schema renames

Two schema changes that affect generated models and graph relationship types.

**1. Rename `StandardNameSource` enum → `StandardNameSourceType`**

The existing `StandardNameSource` enum (values: dd, signals, manual, reference) describes
source *types*. Renaming to `StandardNameSourceType` frees the `StandardNameSource` name
for the new node class (Phase 1).

- `imas_codex/schemas/standard_name.yaml`: rename enum + update `source_type` range on `StandardName`
- `uv run build-models --force`
- Update all code referencing the generated enum class name

**Graph impact:** None — the enum stores string values (dd, signals, etc.), not the Python class name.

**2. Rename `HAS_SN_VOCAB_GAP` relationship → `HAS_STANDARD_NAME_VOCAB_GAP`**

- `imas_codex/schemas/standard_name.yaml`: update VocabGap description + examples
- `imas_codex/schemas/imas_dd.yaml`: update relationship_type annotation
- `imas_codex/schemas/facility.yaml`: update relationship_type annotation
- `uv run build-models --force`
- Update code in `imas_codex/standard_names/graph_ops.py` (VocabGap persistence)
- Update code in `imas_codex/cli/sn.py` (status query)
- Update tests in `tests/standard_names/test_vocab_gaps.py`

**Acceptance:** `uv run build-models --force` succeeds. `grep -r 'HAS_SN_VOCAB_GAP' imas_codex/ tests/`
returns zero matches. `uv run pytest tests/standard_names/` passes.

#### Phase 0c: Graph relationship migration

Migrate existing `HAS_SN_VOCAB_GAP` relationships in the live graph to
`HAS_STANDARD_NAME_VOCAB_GAP`. Run as inline Cypher per project guidelines
(no migration scripts).

```cypher
MATCH (src)-[old:HAS_SN_VOCAB_GAP]->(vg:VocabGap)
WITH src, old, vg, properties(old) AS props
MERGE (src)-[new:HAS_STANDARD_NAME_VOCAB_GAP]->(vg)
SET new = props
DELETE old
RETURN count(new) AS migrated
```

**Acceptance:** `MATCH ()-[r:HAS_SN_VOCAB_GAP]->() RETURN count(r)` returns 0.
Schema compliance tests pass.

#### Phase 0d: Source-agnostic path references

Make StandardName data source references generic — supporting DD paths, facility signals,
and future source types in a unified model. **Reviewed through 4 rounds of RD critique.**

**Coordinated ISN change:** `ids_paths` → `dd_paths` rename in imas-standard-names repo.
See `~/Code/imas-standard-names/plans/features/07-dd-paths-rename.md`.

##### Design rationale

Standard names are many-to-one: `electron_temperature` may originate from DD and later
gain signal sources. No source is "primary" — the name evolves to represent the physics
concept across all data sources. The schema must reflect this.

**Naming convention:**
- `dd_paths` — used everywhere that refers specifically to DD paths (replaces `ids_paths`)
- `source_paths` — generic graph property holding ALL source IDs (union of DD + signals + future)
- `source_types` — unordered set of contributing source types

**`source_paths` entry format (colon discriminator):**

| Source Type | Format | Example | Discriminator |
|-------------|--------|---------|---------------|
| DD | bare path | `equilibrium/time_slice/profiles_1d/psi` | No colon |
| Signal | `facility:path` | `tcv:ip/measured` | Contains colon |
| Future | `namespace:id` | TBD | Contains colon |

DD paths never contain colons. All non-DD sources use `namespace:` prefix.
Enforced centrally via `encode_source_path()` / `parse_source_path()`.

##### Schema changes on StandardName

**Remove:**
- `source_path` (singular) — replaced by `source_paths` (multivalued union)
- `source_type` (singular enum) — replaced by `source_types` (multivalued set)

**Add / rename:**
```yaml
source_types:
  description: >-
    Unordered set of source types that contribute to this standard name.
    E.g., ["dd"] for DD-only, ["dd", "signals"] for multi-source.
    Denormalized cache — authoritative source is StandardNameSource
    relationships. Rebuild: COLLECT(DISTINCT sns.source_type).
  multivalued: true
  range: StandardNameSourceType

source_paths:
  description: >-
    All source entity IDs mapped to this standard name (union list).
    DD paths are bare (no colon): "equilibrium/time_slice/profiles_1d/psi".
    Signal IDs are facility-prefixed: "tcv:ip/measured".
    Denormalized cache — authoritative source is StandardNameSource
    relationships via PRODUCED_NAME.
  multivalued: true
  range: string
```

**Rename:** `imas_paths` → `source_paths` (same slot, new name + expanded semantics)

##### New utility module: `imas_codex/standard_names/source_paths.py`

Central encode/parse/split/merge utilities. All callers go through this —
no hand-rolled string construction.

```python
def encode_source_path(source_type: str, source_id: str) -> str:
    """Encode a source ID for storage in source_paths.
    DD paths stored bare; all others namespaced with colon."""

def parse_source_path(path: str) -> tuple[str, str]:
    """Parse source_paths entry → (source_type, source_id)."""

def split_source_paths(paths: list[str]) -> dict[str, list[str]]:
    """Split source_paths into typed subsets: {'dd': [...], 'signals': [...]}."""

def merge_source_paths(existing: list[str], new: list[str]) -> list[str]:
    """Deduplicated union of source paths. Deterministic sort for tests/export."""

def merge_source_types(existing: list[str], new: list[str]) -> list[str]:
    """Deduplicated union of source types. Sorted for deterministic output."""
```

##### Code renames (standard_names/ scope only)

**`imas_paths` → `source_paths`** (~88 references in 19 files):
- Schema: `standard_name.yaml` property rename + description update
- Graph property: all Cypher `sn.imas_paths` → `sn.source_paths`
- Dict keys: `"imas_paths"` → `"source_paths"` in graph_ops, workers, consolidation, etc.
- Tests: all `imas_paths` references in `tests/standard_names/`

**`ids_paths` → `dd_paths`** (~62 references in 11 files):
- Pydantic model: `StandardNameCandidate.ids_paths` → `.dd_paths`
- Workers: `c.ids_paths` → `c.dd_paths`, mapping to `"source_paths"` in graph dict
- Publish: output `dd_paths` in catalog YAML (was `ids_paths`)
- Catalog import: read `dd_paths` from YAML → merge into `source_paths`
- Consolidation: `ids_paths` variable names → `dd_paths`
- Seed: `ids_paths` → `dd_paths`
- Tests: all `ids_paths` references in `tests/standard_names/`

**`source_path` → remove** (~8 references in 5 files):
- Schema: remove slot from StandardName
- graph_ops.py: remove from write/read queries
- publish.py: derive catalog provenance from `dd_paths[0]` (sorted, deterministic)
- benchmark.py, benchmark_reference.py: remove references

**`source_type` → `source_types`** (~15 references in 8 files):
- Schema: remove singular slot, add multivalued `source_types`
- graph_ops.py: update write queries (append-if-missing semantics)
- workers.py: write `source_types: ["dd"]` or `["signals"]`
- catalog_import.py: write `source_types` from catalog context
- CLI `--source` filter: `'dd' IN sn.source_types` (set membership)

**NOT changed** (different schema, different meaning):
- `imas_paths` on WikiChunk/WikiPage (facility.yaml) — wiki extraction
- `imas_paths_found`, `imas_paths_mentioned`, `ocr_imas_paths` — wiki fields
- `extract_imas_paths()` — wiki entity extraction function

**Prompt changes:**
- `sn/review.md`: `item.imas_paths` → `item.dd_paths` (review context is DD-specific)

##### Graph migration

Atomic migration — rename property + remove legacy fields on all StandardName nodes:

```cypher
-- Step 1: Migrate imas_paths → source_paths (idempotent union)
MATCH (sn:StandardName)
WHERE sn.imas_paths IS NOT NULL
WITH sn, sn.imas_paths AS old_paths,
     coalesce(sn.source_paths, []) AS existing
WITH sn, [p IN old_paths WHERE NOT p IN existing] + existing AS merged
SET sn.source_paths = merged
REMOVE sn.imas_paths
RETURN count(sn) AS paths_migrated

-- Step 2: Migrate source_type → source_types (singular to set)
MATCH (sn:StandardName)
WHERE sn.source_type IS NOT NULL
WITH sn, sn.source_type AS old_type,
     coalesce(sn.source_types, []) AS existing
WITH sn, CASE WHEN old_type IN existing THEN existing
              ELSE existing + old_type END AS merged
SET sn.source_types = merged
REMOVE sn.source_type
RETURN count(sn) AS types_migrated

-- Step 3: Remove source_path (singular)
MATCH (sn:StandardName)
WHERE sn.source_path IS NOT NULL
REMOVE sn.source_path
RETURN count(sn) AS removed
```

##### Acceptance criteria

- `rg '\bimas_paths\b' imas_codex/standard_names/ imas_codex/schemas/standard_name.yaml tests/standard_names/` → 0 matches
- `rg '\bids_paths\b' imas_codex/standard_names/ tests/standard_names/` → 0 matches
- `rg '\bsource_path\b' imas_codex/schemas/standard_name.yaml` → 0 matches (singular removed)
- `rg '\bsource_type\b' imas_codex/schemas/standard_name.yaml` → 0 matches (singular removed)
- `MATCH (sn:StandardName) WHERE sn.imas_paths IS NOT NULL RETURN count(sn)` → 0
- `MATCH (sn:StandardName) WHERE sn.source_path IS NOT NULL RETURN count(sn)` → 0
- `MATCH (sn:StandardName) WHERE sn.source_type IS NOT NULL RETURN count(sn)` → 0
- `uv run build-models --force` succeeds
- `uv run pytest tests/standard_names/` passes
- `source_paths.py` has 100% test coverage

### Phase 1: Schema — StandardNameSource node type

**Prerequisite:** Phase 0b has renamed the `StandardNameSource` enum to `StandardNameSourceType`,
freeing the `StandardNameSource` name for the new node class.

**Files:**
- `imas_codex/schemas/standard_name.yaml` — add StandardNameSource class, StandardNameSourceStatus enum,
  FROM_DD_PATH/FROM_SIGNAL/PRODUCED_NAME relationship annotations
- Run `uv run build-models --force` to regenerate models

**Acceptance:** `StandardNameSource` and `StandardNameSourceStatus` importable from generated models.
No name collision with `StandardNameSourceType` enum.

### Phase 2: Graph operations — StandardNameSource CRUD

**Files:**
- `imas_codex/standard_names/graph_ops.py` — new functions:
  - `merge_standard_name_sources(sources, force)` — batch MERGE with ON CREATE/ON MATCH.
    Rejects `source_type` values `manual`/`reference` with `ValueError` before writing
    (only `dd` and `signals` are valid pipeline sources).
  - `claim_standard_name_source_batch(timeout, token)` — atomic full-batch claim
  - `fetch_claimed_source_metadata(token)` — verify + join source data
  - `mark_sources_composed(token, results)` — token-verified status update
  - `mark_sources_attached(token, results)` — auto-attach status
  - `mark_sources_vocab_gap(token, results)` — vocab gap status
  - `mark_sources_failed(token, error, max_attempts)` — durable retry + terminal
  - `mark_sources_stale(token, source_ids)` — missing source detection
  - `release_standard_name_source_claims(token)` — batch release
  - `reconcile_standard_name_sources(source_type)` — post-rebuild reconciliation
  - `get_standard_name_source_stats()` — status counts for progress/CLI

**All functions return affected row count. All state transitions are token-verified.**

### Phase 3: Fix extraction query + enrichment

**Files:**
- `imas_codex/standard_names/sources/dd.py` — restructure `_ENRICHED_QUERY`:
  1. LIMIT on distinct paths (Gap 1)
  2. Pre-LIMIT unnamed + StandardNameSource exclusion (Gap 2)
  3. Add `c.scope AS cluster_scope` (Gap 3)
  4. Fix observability (Gap 6)
  5. Add `force` parameter

- `imas_codex/standard_names/enrichment.py`:
  1. Implement reversed `select_grouping_cluster()` (Gap 4)
  2. Enhance unclustered grouping (Gap 5)

### Phase 4: Rewrite extract worker

**Files:**
- `imas_codex/standard_names/workers.py` — `extract_worker()`:
  1. Query + classify + enrich (existing logic, with fixes)
  2. Compute batch_key per item
  3. Call `merge_standard_name_sources()` to write StandardNameSource nodes to graph
  4. Report stats: quantity/metadata/skip counts
  5. `mark_done()` — extract is a single-pass writer, not a loop

- `imas_codex/standard_names/state.py` — remove `extracted` in-memory list
  (compose reads from graph, not state)

### Phase 5: Rewrite compose worker

**Files:**
- `imas_codex/standard_names/workers.py` — `compose_worker()`:
  1. Convert from in-memory batch reader to graph-primary claim loop
  2. `has_work_fn` queries StandardNameSource status
  3. Claim loop: `claim_standard_name_source_batch()` → `fetch_claimed_source_metadata()` →
     handle missing sources → LLM compose → persist StandardName → mark outcomes
  4. Attachment handling: detect existing matches → mark `attached`
  5. Vocab gap handling: detect grammar gaps → mark `vocab_gap`
  6. Error handling: `mark_sources_failed()` with durable retry
  7. Rich progress via WorkerStats

- `imas_codex/standard_names/pipeline.py` — update compose WorkerSpec:
  - Add `has_work_fn` for PipelinePhase
  - Compose now uses claim loop pattern (similar to validate/persist)

### Phase 6: Source type normalization

**Files:**
- `imas_codex/schemas/standard_name.yaml` — verify enum value is `signals`
- `imas_codex/standard_names/graph_ops.py` — normalize all `signal` → `signals`
- `imas_codex/standard_names/workers.py` — normalize source_type references

### Phase 7: CLI integration

**Files:**
- `imas_codex/cli/sn.py`:
  - `sn status` — show StandardNameSource statistics alongside StandardName stats
  - `sn generate` — display StandardNameSource progress in Rich panels
  - `sn reconcile` — new subcommand for post-rebuild reconciliation
  - `sn clear` — handle StandardNameSource cleanup alongside StandardName

### Phase 8: Tests

**Files:**
- `tests/standard_names/test_standard_name_source_schema.py` — schema compliance
- `tests/standard_names/test_standard_name_source_graph_ops.py` — CRUD, claims, reconciliation
- `tests/standard_names/test_extraction_coverage.py` — gap fixes (LIMIT, scope, grouping)
- Update existing tests for new worker signatures

**Test cases:**
1. StandardNameSource MERGE creates on first run, preserves composed on re-run
2. StandardNameSource MERGE requeues stale, resets all on --force
3. Atomic batch claim — full batch or nothing
4. Stale-claim timeout reclaim
5. Missing source detection → stale marking
6. Token-verified state transitions (no clobber from slow workers)
7. Durable retry: attempt_count increments, transitions to failed at max
8. Reconciliation: re-links after rebuild, marks stale when missing
9. LIMIT applies to paths not rows
10. cluster_scope flows through enrichment
11. Grouping cluster uses global > domain > IDS
12. Unclustered paths use IDS + parent grouping
13. No StandardNameSource created for metadata/skip paths
14. StandardNameSource rejects source_type `manual`/`reference` (only `dd`/`signals` valid for pipeline nodes)

### Phase 9: Documentation

| Target | Update |
|--------|--------|
| `AGENTS.md` | Update SN pipeline section: StandardNameSource node, graph-primary compose, lifecycle, reconciliation. Replace all `SN*` references with `StandardName*` |
| `README.md` | Mention `sn reconcile` command |
| `plans/README.md` | Update plan status |
| Schema reference | Auto-generated via `build-models` |

## Dependencies

```
Phase 0a-0c (naming) ──┐
Phase 0d (source-agnostic) ──┤→ Phase 1 (schema) → Phase 2 (graph ops) → Phase 3 (extraction fixes)
                              │                                          → Phase 4 (extract worker)
                              │                                          → Phase 5 (compose worker)
ISN dd_paths rename ──────────┘  (coordinated, not blocking)
Phase 6 (normalization) — independent (can run in parallel with Phase 0)
Phase 7 (CLI) depends on Phases 4, 5
Phase 8 (tests) depends on Phases 1-7
Phase 9 (docs) depends on all phases
```

**Fleet dispatch:** Phases 0a-0d are mechanical renames suitable for parallel agent execution
(different file sets). Phase 0d has a coordinated ISN plan but is not blocked by it — codex
can land first, ISN follows. Phases 3-5 are the core architectural work requiring sequential
implementation by a single agent.

## Scope

**v1 (this plan): DD source only, with source-agnostic schema.**

The schema (`source_paths`, `source_types`, `StandardNameSource`) is designed for multi-source
from day one. v1 implementation populates DD sources only. Signal support follows the same
pattern with no schema changes:
- `(StandardNameSource {source_type: 'signals'})-[:FROM_SIGNAL]->(FacilitySignal)`
- Signal `source_paths` entries: `"tcv:ip/measured"` (facility-prefixed)
- Signal `source_types` append: `["dd", "signals"]` on existing names
- Signal batch_key: `{physics_domain}:{diagnostic}:{unit}`
- Signal reconciliation: same stale detection via `source_id` join

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 0a-0c (naming) | Low-Medium | Mechanical rename, high file count — use sed + ruff format, verify with grep + pytest |
| 0d (source-agnostic) | Medium | Schema + code + graph migration atomic. Central utility prevents drift. ISN coordination adds scheduling risk |
| 1 (schema) | Low | Additive change, no existing data affected |
| 2 (graph ops) | Low | New functions, existing ops unchanged |
| 3 (extraction) | Medium | Query restructure — test against live graph |
| 4 (extract worker) | Medium | Behavioral change — writes to graph instead of memory |
| 5 (compose worker) | High | Major rewrite — claim loop replaces in-memory reader |
| 6 (normalization) | Low | String substitution with grep verification |
| 7 (CLI) | Low | Additive commands |
| 8 (tests) | Low | New tests |
| 9 (docs) | Low | Text updates |

## Design Decisions (from RD review)

1. **Minimal StandardNameSource** — no duplicated source metadata. Join at compose time. Eliminates staleness management.
2. **5-state lifecycle** — extracted, composed, attached, vocab_gap, failed, stale. No `skipped` — only quantity paths get StandardNameSource.
3. **Full-batch atomic claim** — single Cypher transaction ensures batch integrity.
4. **Token-verified transitions** — all state changes match `claim_token`, preventing clobber from slow/crashed workers.
5. **Durable retry** — `attempt_count` + `last_error` on StandardNameSource. Returns to `extracted` until `max_attempts` (3), then terminal `failed`.
6. **PRODUCED_NAME relationship** — separate from HAS_STANDARD_NAME. Tracks composition lineage.
7. **Reconciliation** — `sn reconcile` command re-links after DD rebuild, revives stale→extracted, marks missing sources.
8. **Batch semantics** — "all currently extracted members with matching batch_key". Partial batches after failure are valid.
9. **--force resets** — clears attempt_count, last_error, failed_at alongside status→extracted.
10. **Naming convention** — all SN abbreviations expanded to `StandardName` (PascalCase classes) / `standard_name` (snake_case functions/files). `StandardNameSource` enum renamed to `StandardNameSourceType` to avoid collision with the new `StandardNameSource` node class. `HAS_SN_VOCAB_GAP` relationship renamed to `HAS_STANDARD_NAME_VOCAB_GAP`. Public `sn` CLI namespace and config accessors retained.
11. **Source type constraint** — `StandardNameSource` nodes only allow `source_type` values `dd` and `signals` (pipeline sources). `manual` and `reference` are `StandardName` provenance values that never enter the composition pipeline. Enforced by test, not by a separate enum — avoids enum proliferation while preserving the invariant.
12. **Source-agnostic references (from RD rounds 1-4):**
    - `source_paths` is a denormalized union cache on StandardName. Authoritative provenance lives in StandardNameSource typed relationships (`FROM_DD_PATH`, `FROM_SIGNAL`). Rebuildable from graph.
    - `source_types` is an unordered set (no primary source concept). Standard names are many-to-one — any source can contribute without displacing others.
    - Colon discriminator enforced centrally via `encode_source_path()` / `parse_source_path()`. DD paths are bare (no colon), all other sources are namespace-prefixed.
    - `source_path` (singular) and `source_type` (singular) removed entirely — no backward compat, no deprecation period (greenfield project).
    - ISN catalog exports DD subset only (`dd_paths`). Signal provenance stays in graph.
    - Neo4j stores `source_types` as list; code treats as set. Export/tests normalize via deterministic sort.
