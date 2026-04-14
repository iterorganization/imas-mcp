# 11: Rich Standard Name Composition

**Status:** Ready to implement
**Depends on:** 09 (pipeline working — DONE)
**Enables:** 12 (catalog import), 13 (publish pipeline)
**Agent:** architect (cross-module: schema + prompts + models + workers + graph_ops + tests)

## Problem

The compose worker generates bare StandardName nodes with only `id`, `physical_base`,
and `confidence`. The catalog schema requires ~12 rich fields: description, documentation
(LaTeX equations, inline cross-references, typical tokamak values, measurement methods),
unit, kind (scalar/vector/metadata), tags, links, ids_paths, status, validity_domain,
constraints, and grammar decomposition.

The current `SNCandidate` Pydantic model captures only `source_id`, `standard_name`,
`fields`, `confidence`, and `reason`. The graph schema (`StandardName` in standard_name.yaml)
has `description`, `unit`, `source`, `source_path`, `confidence` — but no
documentation, kind, tags, links, ids_paths, or grammar fields.

Additional bugs found via audit:
- `write_standard_names()` uses unconditional SET (not coalesce) — re-runs erase data
- Schema doc says `(FacilitySignal)-[:MEASURES]->(StandardName)` but code uses
  `HAS_STANDARD_NAME` for both entity types. `sources/signals.py` queries MEASURES
  which is never written, so signal dedup is broken (silent re-processing).
- `review_status` enum has `candidate` (not past tense). User requires all past tense.
- Embedding generation not wired — persist worker never calls embed functions.
- Zero tests for `sn/graph_ops.py` — all DB operations untested.

## Approach

Split into two deployment units for parallelism:
- **Wave 1 (Phases 1+4):** Schema extension + persist fix + tests. Unlocks Plans 12/13.
- **Wave 2 (Phases 2+3+5):** LLM compose upgrade. Can run parallel with Plan 12.

## Phase 1: Extend graph schema + fix consistency

**Files:** `imas_codex/schemas/standard_name.yaml`, `imas_codex/standard_names/sources/signals.py`

### 1a. Add rich fields to StandardName

```yaml
documentation:
  description: >-
    Rich documentation with LaTeX equations, governing physics,
    measurement methods, typical values, sign conventions.
    Uses [name](#name) inline links to other standard names.
kind:
  description: Entry kind — scalar, vector, or metadata
  range: StandardNameKind
tags:
  description: Classification tags from controlled vocabulary
  multivalued: true
  range: string
links:
  description: Internal cross-references to related standard names (name only)
  multivalued: true
  range: string
ids_paths:
  description: IMAS DD paths mapped to this standard name
  multivalued: true
  range: string
validity_domain:
  description: Physical region where this quantity is defined
constraints:
  description: Physical/mathematical constraints (e.g., T_e > 0)
  multivalued: true
  range: string
subject:
  description: Particle species (electron, ion, deuterium, etc.)
component:
  description: Vector component (radial, toroidal, vertical, etc.)
coordinate:
  description: Coordinate qualifier
position:
  description: Spatial location qualifier (magnetic_axis, midplane, etc.)
process:
  description: Physical process qualifier (ohmic, bootstrap, etc.)
```

### 1b. Add StandardNameKind enum

```yaml
StandardNameKind:
  permissible_values:
    scalar: { description: Scalar quantity }
    vector: { description: Vector quantity (R,Z or multi-component) }
    metadata: { description: Non-measurable concept or classification }
```

### 1c. Fix StandardNameReviewStatus enum (all past tense)

Replace current `candidate/accepted/rejected/skipped` with:

```yaml
StandardNameReviewStatus:
  permissible_values:
    drafted: { description: LLM-generated, awaiting review }
    published: { description: Exported to catalog PR for review }
    accepted: { description: Imported from merged catalog entry }
    rejected: { description: Reviewed and rejected }
    skipped: { description: Skipped during review (e.g., low confidence) }
```

### 1d. Fix schema doc — HAS_STANDARD_NAME for all entity types

In `StandardName` class description, change:
```
- (FacilitySignal)-[:MEASURES]->(StandardName)
```
to:
```
- (FacilitySignal)-[:HAS_STANDARD_NAME]->(StandardName)
```

### 1e. Fix signals.py MEASURES query

In `sn/sources/signals.py`, change the MEASURES query to HAS_STANDARD_NAME
so signal dedup works correctly (reads must match writes).

Run `uv run build-models --force` after schema changes.

**Acceptance:**
- `uv run pytest tests/graph/test_schema_compliance.py` passes
- `grep -r MEASURES imas_codex/standard_names/` returns zero matches

## Phase 2: Extend LLM response model

**Files:** `imas_codex/standard_names/models.py`

Extend `SNCandidate` with rich fields matching the catalog schema:

```python
class SNCandidate(BaseModel):
    """Full standard name entry from LLM composition."""

    source_id: str = Field(description="Source entity ID (DD path or signal ID)")
    standard_name: str = Field(description="Composed standard name in snake_case")
    description: str = Field(description="One sentence, <120 chars")
    documentation: str = Field(description="Rich docs with LaTeX, links, typical values")
    unit: str | None = Field(default=None, description="SI unit string (eV, m, A, etc.)")
    kind: Literal["scalar", "vector", "metadata"] = Field(description="Entry kind")
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    links: list[str] = Field(default_factory=list, description="Related standard names")
    ids_paths: list[str] = Field(default_factory=list, description="Mapped IMAS DD paths")
    fields: dict[str, str] = Field(description="Grammar fields used")
    confidence: float = Field(ge=0, le=1, description="Naming confidence")
    reason: str = Field(description="Brief justification")
    validity_domain: str | None = Field(default=None, description="Physical region")
    constraints: list[str] = Field(default_factory=list, description="Physical constraints")
```

Update `compose_worker` in `workers.py` to pass all new fields through to
`state.composed` dicts (currently only passes id, source_type, source_id,
fields, confidence, reason — lines 193-202).

**Acceptance:** Pydantic model validates against sample LLM output.

## Phase 3: Update prompts for rich generation

**Files:**
- `imas_codex/llm/prompts/sn/compose_system.md` — add output format section
- `imas_codex/llm/prompts/sn/compose_dd.md` — update expected output

Add to system prompt:

1. **Output format specification** — JSON schema for rich entries with all fields
2. **Documentation template** — opening paragraph, governing equations, physical
   significance, measurement methods, typical values (cite specific machines from
   tokamak_parameters), sign conventions, inline `[name](#name)` cross-references
3. **Tags guidance** — controlled vocabulary from catalog (primary: physics domain,
   secondary: characteristics like spatial-profile, time-dependent, derived)
4. **Kind classification rules** — scalar (single value per point), vector (R,Z or
   components), metadata (concepts, techniques, not measurable quantities)
5. **Links guidance** — only reference other standard names, 4-8 per entry,
   validate existence against existing names list

The compose_dd.md user prompt already passes existing names — extend to include
existing names with their descriptions for cross-referencing context.

**Acceptance:** Dry-run `sn build --ids equilibrium --limit 5 --dry-run` shows
rich entries in progress output.

## Phase 4: Fix persist, graph_ops, and wire embedding

**Files:**
- `imas_codex/standard_names/workers.py` (persist_worker)
- `imas_codex/standard_names/graph_ops.py` (write_standard_names)

### 4a. Fix unconditional SET overwrite bug

Current code does `SET sn.description = b.description` which nulls out existing
data on re-runs. **This is a data corruption risk** — if Plan 12 imports 309
catalog entries as `accepted`, any subsequent `sn build` on overlapping names
will downgrade them to `drafted` and erase catalog-authoritative fields.

Fix with coalesce for ALL fields:

```cypher
SET sn.description = coalesce(b.description, sn.description),
    sn.documentation = coalesce(b.documentation, sn.documentation),
    sn.kind = coalesce(b.kind, sn.kind),
    sn.tags = coalesce(b.tags, sn.tags),
    ...
```

### 4b. Write all rich fields

Extend `write_standard_names()` to persist: description, documentation, kind,
tags, links, ids_paths, validity_domain, constraints, all grammar fields,
model, generated_at, review_status='drafted'.

### 4c. Link Unit (HAS_UNIT)

The schema defines `unit` with `range: Unit`. By schema convention
this creates a `HAS_UNIT` relationship. After MERGE StandardName:

```cypher
WITH sn, b
WHERE b.unit IS NOT NULL
MERGE (u:Unit {id: b.unit})
MERGE (sn)-[:HAS_UNIT]->(u)
```

### 4d. Wire embedding generation

The persist worker must call `embed_descriptions_batch()` from
`imas_codex/embeddings/description.py` after writing StandardName nodes.
Without this, Plan 14's MCP vector search returns zero results.

```python
# After write_standard_names()
from imas_codex.embeddings.description import embed_descriptions_batch
embed_descriptions_batch("StandardName", [n["id"] for n in validated])
```

**Acceptance:**
- `sn build --ids equilibrium --limit 10` generates rich entries
- Graph query shows StandardName nodes with documentation, kind, tags, unit
- Re-run doesn't null out existing fields
- `HAS_UNIT` relationships exist
- StandardName nodes have non-null `embedding` property

## Phase 5: Update validate worker

**Files:** `imas_codex/standard_names/workers.py` (validate_worker)

Add validation checks:
- `description` is present and <120 chars
- `documentation` is present and >200 chars
- `unit` is valid (matches known Unit nodes or standard SI patterns)
- `kind` is valid enum value
- `tags` are from controlled vocabulary
- `links` reference existing standard names (warn, don't fail)
- `ids_paths` contain valid DD paths (check via graph query)

Report new metrics: `doc_present`, `doc_length_ok`, `unit_valid`, `kind_valid`.

**Acceptance:** Validate worker reports new metrics in progress display.

## Test Plan

### New: `tests/sn/test_graph_ops.py` (CRITICAL — currently 0 tests)

- `test_write_standard_names_creates_nodes` — write then read back all fields
- `test_write_standard_names_coalesce` — write once, write again with None fields,
  verify first values preserved (prevents data corruption on re-runs)
- `test_write_standard_names_dd_relationship` — verify HAS_STANDARD_NAME created for DD
- `test_write_standard_names_signal_relationship` — verify HAS_STANDARD_NAME for signals
- `test_write_standard_names_unit_relationship` — verify HAS_UNIT created
- `test_get_validated_standard_names_filters` — confidence and ids_filter work
- `test_get_existing_standard_names_dedup` — returns correct set

### New: `tests/sn/conftest.py`

Shared fixtures: sample candidates with all fields, mock GraphClient, state factory.

### Existing test updates

- `tests/sn/test_publish.py` — update for new SNCandidate fields
- `tests/sn/test_review.py` — update for new review_status values

## Documentation Updates

- `AGENTS.md` — document new schema fields, review_status lifecycle
- `plans/README.md` — update plan status
