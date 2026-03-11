# Signal Enrichment Pipeline v2

> **Status**: Planning  
> **Priority**: Critical — blocks mapping quality and IDS assembly  
> **Scope**: Discover → Enrich → Map pipeline for FacilitySignal and SignalGroup  
> **Principle**: LLM cost is not a constraint. Quality of descriptions and mappings is the sole objective. Deterministic steps must use function calls, not LLM inference.

## Problem Statement

The current signal enrichment pipeline produces descriptions that are too
generic for high-quality IMAS field mapping. Five critical gaps degrade
enrichment context, and the downstream mapping pipeline receives signal group
descriptions that lack the specificity needed for correct transform generation
and unit handling.

The pipeline must be redesigned as a chained LLM process:
**discover → enrich → map**, where each stage builds on the previous stage's
output and deterministic steps (context fetches, field validation, unit checks,
COCOS lookups) are executed as programmatic function calls rather than LLM
inference.

---

## Architecture Overview

```
┌─────────┐    ┌─────────┐    ┌──────────────┐    ┌───────────┐    ┌─────────┐
│  Seed   │───▶│  Scan   │───▶│   Discover   │───▶│  Enrich   │───▶│   Map   │
│ (trees) │    │ (nodes) │    │  (signals)   │    │  (LLM)    │    │  (LLM)  │
└─────────┘    └─────────┘    │              │    │           │    │         │
                              │ ✦ Normalize  │    │ ✦ Context │    │ ✦ Sect. │
                              │   filters    │    │   inject  │    │   assign│
                              │ ✦ Signal     │    │ ✦ Struct. │    │ ✦ Field │
                              │   groups     │    │   output  │    │   map   │
                              │ ✦ Set props  │    │ ✦ Post-   │    │ ✦ Valid │
                              │   (dual)     │    │   prop    │    │   (det.)│
                              └──────────────┘    └───────────┘    └─────────┘
                                    │                   │                │
                                    └───────────────────┴────────────────┘
                                      All share normalized schema + properties
```

---

## Phase 1: Normalize Discovery Schema and Properties

**Goal**: Every FacilitySignal has consistent, queryable metadata regardless
of discovery source. The dual property+relationship model is maintained for
all class-ranged slots. Normalization happens at discover time.

### 1.1 Set `data_source_node` Property Consistently (Critical Bug)

The schema already defines `data_source_node` correctly as a dual
property+relationship slot (`range: SignalNode`, `relationship_type:
HAS_DATA_SOURCE_NODE`). The bug is that **most scanners only create the
relationship edge without setting the property**:

| Scanner | Edge (`HAS_DATA_SOURCE_NODE`) | Property (`data_source_node`) |
|---------|------|------|
| JET `device_xml` | ✅ | ✅ |
| TCV `tree_traversal` | ✅ (28,690) | ❌ null |
| TCV `tdi_scan` | ❌ | ❌ |
| TCV `wiki_extraction` | ❌ | ❌ |

This violates the dual property+relationship design. Every scanner that
creates a `HAS_DATA_SOURCE_NODE` edge must also set the `data_source_node`
property on the `FacilitySignal`.

**Code fix**: Update the signal promotion/creation code in each scanner to
set `data_source_node = signal_node_id` when creating the relationship.

**Graph migration**: Backfill the property for existing signals:
```cypher
MATCH (s:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
WHERE s.data_source_node IS NULL
SET s.data_source_node = sn.id
```

**Downstream fix**: The `enrich_worker` tree context filter at line 2950
uses `s.get("data_source_node")` which is correct once the property is
populated. No need to switch to edge-existence checks — the property filter
is the designed fast path. With the migration and scanner fix, it works.

### 1.2 Set `data_source_path` Consistently

`data_source_path` is null for all TCV `tree_traversal` signals. Each
scanner should set it at discovery time:

| Scanner | Source for `data_source_path` |
|---------|------|
| `device_xml` | MDSplus path (already set) |
| `tree_traversal` | `SignalNode.path` (derive from edge target) |
| `tdi_scan` | TDI function file path |
| `wiki_extraction` | Wiki page URL |

**Graph migration**: Backfill for existing signals:
```cypher
MATCH (s:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
WHERE s.data_source_path IS NULL
SET s.data_source_path = sn.path
```

### 1.3 Define Canonical vs Scanner-Specific Fields

Fields on `FacilitySignal` fall into two categories:

**Canonical fields** (set by all scanners, used in enrichment/mapping):
- `accessor` — native access expression (required)
- `data_source_name` — tree/function/page name
- `data_source_path` — full path for direct access
- `data_source_node` — SignalNode provenance (property + edge)
- `discovery_source` — which scanner found it
- `unit` — physical units (from metadata, not LLM)
- `name` — human-readable name
- `description` — physics description (LLM-enriched)

**Scanner-specific fields** (set only by specific scanners, NOT injected into
shared enrichment context):
- `tdi_function` — TDI wrapper function (tdi_scan only)
- `tdi_quantity` — TDI quantity argument (tdi_scan only)

These scanner-specific fields are legitimate FacilitySignal properties in the
schema, but the enrichment prompt should not inject them as though they are
universal. They are context for the TDI scanner's signals only and should
appear in the per-signal section of the prompt only when present, labeled by
their scanner-specific origin.

### 1.4 Normalize `enrichment_source` Values

`mark_signals_enriched()` does not set `enrichment_source`, leaving it null
for directly-enriched signals. Only `propagate_signal_group_enrichment` sets
`signal_group_propagation`.

**Fix**: Set `enrichment_source = 'direct'` in `mark_signals_enriched()` and
`enrichment_source = 'direct_underspecified'` in `mark_signals_underspecified()`.

### 1.5 Add `--rescan` Flag to Signal Discovery CLI

The `discover signals` CLI currently has `--scan-only` and `--enrich-only`
but **no `--rescan` flag** (unlike `discover wiki --rescan` and
`discover code --rescan`).

A `--rescan` flag for signals should re-discover signals from data sources
(re-scan trees, re-run TDI scan) but NOT re-enrich. Re-enrichment is a
separate concern controlled by resetting signal status to `discovered`.

**Action**: Add `--rescan` flag to `imas_codex/cli/discover/signals.py` with
behavior analogous to `discover wiki --rescan`:
- Reset previously discovered signals to be re-scanned from their data source
- Does NOT re-enrich — enrichment is triggered by `--enrich` on signals in
  `discovered` status
- Document in CLI help text

---

## Phase 2: Fix Context Injection Gaps

**Goal**: Every enrichment context source reaches the LLM prompt with
maximum available information.

### 2.1 Re-run TDI Scanner for `source_code`

**Problem**: All 189 TDI functions in the graph have empty `source_code`
fields. The graph predates the scanner fix (commit `603604fd`) that corrected
TDI source code serialization. 338 signals reference these functions.

**Action**: After adding `--rescan` (Phase 1.5), re-scan TDI functions:
```bash
uv run imas-codex discover signals tcv --rescan
```
This is a deterministic step — no LLM calls. The scanner fix is already in
the codebase; only the graph data is stale.

### 2.2 Inject Deterministic Code References

**Problem**: 2,208 signals are reachable via the graph path:
```
CodeChunk →[CONTAINS_REF]→ DataReference →[RESOLVES_TO_NODE]→ SignalNode
   ←[HAS_DATA_SOURCE_NODE]← FacilitySignal
```
These code references contain real usage examples (how the signal is read,
units used, transformations applied) but are never injected into the
enrichment prompt. Currently only `_fetch_code_context()` (vector search)
is used — it matches by group-level semantic similarity, missing specific
signal-level code references.

**Action**: Add `fetch_signal_code_refs()` — a deterministic graph traversal
that fetches code chunks directly linked to each signal via the path above:
```cypher
MATCH (s:FacilitySignal {id: $signal_id})
      -[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
      <-[:RESOLVES_TO_NODE]-(dr:DataReference)
      <-[:CONTAINS_REF]-(cc:CodeChunk)
RETURN cc.text AS code, cc.language AS language,
       cc.source_file AS file LIMIT 3
```
Inject these as `## Direct Code References` in the per-signal prompt section,
before the existing group-level `## Source Code Context`.

### 2.3 Fix Wiki Path Matching for TCV

**Problem**: `_find_wiki_context()` looks up `data_source_path` from the
signal dict, but TCV signals have `data_source_path = null`. Wiki chunks
with `mdsplus_paths_mentioned` (1,829 chunks, 6,955 path references) are
never matched.

**Fix**: After Phase 1.2 backfills `data_source_path`, this resolves itself.
Additionally, add a fallback in `_find_wiki_context()`:
```python
path = signal.get("data_source_path") or signal.get("accessor")
```

### 2.4 Enrich SignalGroup Nodes with Structured Output

**Problem**: `SignalGroup` nodes have `description`, `keywords`, and
`physics_domain` fields in the schema but these are currently only populated
by propagation from the representative signal. The group-level description
is identical to the representative's — it doesn't describe the *group pattern*
(e.g., "Array of poloidal magnetic field measurements from N probes").

**Action**: After enriching the representative, make an LLM call to generate
group-level metadata using a **Pydantic structured output model** — the same
pattern used by `TriageBatch`, `ScoreBatch`, `WikiScoreBatch`, and
`SignalEnrichmentBatch` elsewhere in the codebase:

```python
class SignalGroupEnrichmentResult(BaseModel):
    """LLM enrichment for a SignalGroup node."""
    description: str = Field(
        description="What this group of signals represents as a collection. "
        "Describe the shared measurement type and how members differ "
        "(e.g., by probe index, channel number, coil position)."
    )
    physics_domain: PhysicsDomain
    diagnostic: str = Field(default="")
    keywords: list[str] = Field(default_factory=list, max_length=5)
    member_variation: str = Field(
        description="How individual members differ within the group "
        "(e.g., 'spatial position', 'channel index', 'coil number')"
    )
```

The prompt provides the representative's enriched metadata, list of member
accessors, and available tree context. The LLM assigns fields against the
templated schema — the same `call_llm_structured()` / `response_format`
pattern used in all other discovery pipelines.

This group-level description feeds directly into the mapping pipeline's
`signal_group_detail` context.

---

## Phase 3: Post-Propagation Individualization

**Goal**: After group enrichment is propagated, generate individualized
descriptions for member signals that capture their specific identity within
the group.

### 3.1 Design

Currently, `propagate_signal_group_enrichment()` copies the representative's
description verbatim to all members. For a group of 181 `MAGB` probes, all
get the identical description "Measured magnetic field from probe MAGB_NNN".

**New step**: After propagation, make a second LLM call per group that
takes:

**Input (deterministic, via function calls)**:
- The group description (from Phase 2.4)
- The representative's full enrichment
- List of all member signal IDs, accessors, and names
- Any member-specific tree context (parent path, position info)

**LLM task**: Generate an individualized description per member using
structured output:

```python
class IndividualizedSignal(BaseModel):
    signal_id: str
    description: str = Field(
        description="Individualized physics description for this specific "
        "member of the group."
    )

class IndividualizedBatch(BaseModel):
    results: list[IndividualizedSignal]
```

Examples of individualized vs group descriptions:
- **Group**: "Poloidal magnetic field probe array — 181 probes measuring B_pol"
- **Member MAGB_001**: "Poloidal magnetic field from probe 1 at lower inboard midplane"
- **Member MAGB_002**: "Poloidal magnetic field from probe 2 at upper inboard midplane"

### 3.2 When to Run

This runs as a new pipeline stage **after** `propagate_signal_group_enrichment`
completes for a group. The flow becomes:

1. Enrich representative → `mark_signals_enriched([representative])`
2. Propagate to members → `propagate_signal_group_enrichment(rep_id, enrichment)`
3. Enrich group node → `enrich_signal_group(group_id)` *(new, Phase 2.4)*
4. Individualize members → `individualize_group_members(group_id)` *(new)*

Steps 3 and 4 can be batched — process all recently-propagated groups in one
pass.

### 3.3 Unwind Descriptions

- **SignalGroup.description**: Describes the group pattern (from Phase 2.4)
- **FacilitySignal.description**: Each member gets its own individualized
  description (from Phase 3.1) that includes the group context plus
  member-specific details
- **enrichment_source**: Set to `'individualized'` for signals that received
  post-propagation individualization

---

## Phase 4: Structured Enrichment Output

**Goal**: The enrichment model captures structured metadata in validated
fields, not free-form descriptions. Descriptions describe physics; structured
fields capture units, coordinates, and conventions.

### 4.1 Separation of Concerns

The enrichment description should be a **good physics description** — what the
signal measures, the diagnostic technique, and the physical context. It should
NOT embed units, coordinate systems, or COCOS indices as prose. These are
orthogonal structured data captured in their own validated fields.

**Anti-pattern** (conflating concerns in description):
> "Plasma current measured in Amperes using a Rogowski coil, positive
> direction is co-current with toroidal field, in COCOS 2 convention,
> returned as a 1D time-dependent array"

**Correct pattern** (description is physics, metadata is structured):
- `description`: "Plasma current measured by Rogowski coil"
- `unit`: "A" (validated by pint)
- `sign_convention`: "positive = co-current"
- `cocos`: 2
- `physics_domain`: PhysicsDomain.magnetics

### 4.2 Extend `SignalEnrichmentResult` Pydantic Model

The current model has `units_extracted` (plain string) and `sign_convention`
(plain string). These should be strengthened:

```python
class SignalEnrichmentResult(BaseModel):
    # ... existing fields ...

    # Unit field — extracted from metadata, NOT guessed
    units_extracted: str = Field(
        default="",
        description="Units ONLY if present in input metadata. "
        "Do NOT infer or guess — copy from input or leave empty.",
    )

    # Sign convention — from code or wiki
    sign_convention: str = Field(
        default="",
        description="Sign convention if discoverable from context. "
        "E.g., 'positive = co-current'. Leave empty if not stated.",
    )

    # COCOS index if identifiable
    cocos_index: int | None = Field(
        default=None,
        description="COCOS convention index (1-11) if explicitly stated "
        "in source code or documentation. Do NOT guess.",
    )
```

The `unit` field on the graph schema (`FacilitySignal.unit`) already exists.
The enrichment pipeline should write `units_extracted` to this field only
when it has high confidence (from MDSplus metadata or explicit code/wiki
reference). Downstream, the mapping pipeline validates unit compatibility
via pint (`analyze_units()` in `tools.py`).

### 4.3 Update Enrichment Prompt

Rather than adding "description requirements for IMAS mapping" that would
conflate concerns, update the prompt to:

1. **Emphasize description quality**: The description should capture what the
   signal measures and the physics context. It should be specific enough to
   distinguish this signal from similar ones.

2. **Separate structured extraction**: The prompt already has sections for
   units and sign conventions. Reinforce that these go in their dedicated
   fields, not the description.

3. **Remove any instruction to embed units/coords in descriptions**: If
   present, these instructions create conflated output.

### 4.4 Drop Context Quality Thresholds from the Prompt

The current context quality assessment (`high`/`medium`/`low` criteria table)
in the enrichment prompt is prescriptive in a way that constrains the LLM
unhelpfully. Specific threshold rules like "TDI source code + wiki OR 3+
context sources → high" are fragile and don't generalize across scanner types.

**Replace with**: A simpler instruction:
- `high`: Confident classification with multiple corroborating sources
- `medium`: Reasonable classification with some ambiguity
- `low`: Insufficient context to classify meaningfully — accessor and path
  name are the only inputs

The LLM should assess quality holistically, not count context sources
against a threshold table. The downstream routing (`low` → underspecified)
remains unchanged.

---

## Phase 5: Mapping Pipeline Context Injection

**Goal**: The mapping pipeline receives maximum context from enrichment to
produce accurate field mappings with correct transforms.

### 5.1 Current Mapping Pipeline Assessment

The mapping pipeline (`imas_codex/ids/mapping.py`) has a well-designed
4-step architecture:

| Step | Type | Input | Output |
|------|------|-------|--------|
| 0 | Deterministic | Graph queries | Signal groups, IMAS subtree, semantic matches, existing mappings, COCOS paths |
| 1 | LLM | All groups + IMAS tree | Section assignments |
| 2 | LLM (per section) | Group detail + IMAS fields + units + COCOS | Field mapping entries |
| 3 | Deterministic | Proposed mappings | Validated results (source/target existence, transform execution, unit compat) |

**Assessment**: The mapping prompt is lean and well-structured. The
`signal_group_detail` is injected as raw JSON from `query_signal_groups()`.

### 5.2 Context Injection: Enrichment vs Mapping

**Decision**: Inject context at **both** stages, but with different purposes
and different types:

| Context Type | Enrichment Stage | Mapping Stage |
|-------------|-----------------|---------------|
| Signal description | **Generated** here | **Consumed** (via `signal_group_detail`) |
| Units | **Extracted** to `unit` field | **Consumed** for transform generation |
| Sign convention | **Extracted** to field | **Consumed** for COCOS handling |
| COCOS index | **Extracted** to field | **Consumed** for sign flip logic |
| Tree context (parent, siblings) | Injected for physics understanding | Not needed — captured in description and structured fields |
| Wiki context | Injected for extraction quality | Not needed — captured in description and structured fields |
| Code references | Injected for unit/sign extraction | Inject for **transform examples** |
| COCOS paths | Not relevant | Injected (deterministic) |
| IMAS field structure | Not relevant | Injected (deterministic) |
| Unit compatibility | Not relevant | Injected (deterministic, pint-backed) |

**Key insight**: The enrichment stage captures physics meaning in the
description AND populates structured fields (units, signs, COCOS). The
mapping stage combines these structured fields with deterministic IMAS
field metadata. Adding raw wiki/tree context to the mapping prompt would
bloat it — the structured fields already encode what was learned.

**Exception — Code references for transforms**: For groups where the
representative has direct code references (via `RESOLVES_TO_NODE`), inject
a `code_context` field into `signal_group_detail` showing how the signal is
read/transformed in existing code. This helps the LLM generate accurate
`transform_expression` values.

### 5.3 Extend `query_signal_groups()` for Richer Context

Currently the query returns minimal fields. Extend to include:

```cypher
OPTIONAL MATCH (rep:FacilitySignal {id: sg.representative_id})
RETURN ...
       rep.description AS representative_description,
       rep.sign_convention AS sign_convention,
       rep.unit AS unit,
       rep.cocos AS cocos,
       rep.diagnostic AS diagnostic,
       rep.discovery_source AS discovery_source,
       rep.accessor AS representative_accessor
```

This gives the mapping LLM the enriched description AND the structured fields
that were extracted during enrichment.

### 5.4 Add Code References to Signal Group Detail

For groups with code references reachable via `RESOLVES_TO_NODE`:
```cypher
OPTIONAL MATCH (rep:FacilitySignal {id: sg.representative_id})
    -[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
    <-[:RESOLVES_TO_NODE]-(dr:DataReference)
    <-[:CONTAINS_REF]-(cc:CodeChunk)
WITH sg, ..., collect(cc.text)[..2] AS code_refs
```

### 5.5 Extend `sample_members` with Accessors

`sample_members` currently returns only IDs. Extend:
```cypher
collect(DISTINCT {id: m.id, accessor: m.accessor, name: m.name})[..5]
    AS sample_members
```

---

## Phase 6: Prompt Architecture

**Goal**: Establish prompt ordering and caching principles as documented
engineering practice.

### 6.1 Add Prompt Ordering Guidance to AGENTS.md

The codebase already follows the pattern of system prompt first (static),
user prompt second (dynamic) — this is optimal for LLM prompt caching
(Anthropic, Google, and OpenAI all cache prefix-matched prompts). The
`inject_cache_control()` function in `llm.py` adds cache breakpoints
to the last system message.

However, this is undocumented. Add to the **LLM Prompts** section of
AGENTS.md:

```markdown
### Prompt Structure and Caching

All prompts follow a **static-first ordering** to maximize prompt cache
hit rates:

1. **System prompt** (static/quasi-static): Schema definitions, enum
   values, classification rules, output format. These change rarely and
   should be injected first. `inject_cache_control()` places a cache
   breakpoint at the end of the system message.
2. **User prompt** (dynamic): Per-batch signal data, context chunks, and
   specific instructions. This varies per LLM call.

When building prompts, ensure that `{% include %}` blocks for schema
definitions and static rules appear **before** dynamic Jinja2 template
variables. This maximizes the cacheable prefix length.
```

### 6.2 Audit Existing Prompts

Verify that all current prompts follow static-first ordering:

| Prompt | Static Section | Dynamic Section | Order |
|--------|---------------|-----------------|-------|
| `enrichment.md` | Schema, physics domains, diagnostic categories, rules | Per-batch signals, context | ✅ Correct |
| `field_mapping.md` | Task instructions, transform rules, escalation rules | Signal group, IMAS fields, unit analysis | ✅ Correct |
| `exploration.md` | Task instructions | Groups, subtree, semantic results | ✅ Correct |

---

## Phase 7: Graph Data Migrations

**Goal**: Clean up legacy data from previous schema versions.

### 7.1 Backfill Properties

```cypher
-- 1. Set data_source_node property for dual model compliance
MATCH (s:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
WHERE s.data_source_node IS NULL
SET s.data_source_node = sn.id

-- 2. Set data_source_path from SignalNode
MATCH (s:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
WHERE s.data_source_path IS NULL
SET s.data_source_path = sn.path
```

### 7.2 Clean Up Legacy Properties

```cypher
-- Remove pre-SignalGroup pattern properties
MATCH (s:FacilitySignal)
WHERE s.pattern_representative_id IS NOT NULL
REMOVE s.pattern_representative_id, s.pattern_template

-- Normalize enrichment_source from old value
MATCH (s:FacilitySignal)
WHERE s.enrichment_source = 'pattern_propagation'
SET s.enrichment_source = 'signal_group_propagation'
```

### 7.3 Rename `_accessor_to_pattern()` in Code

Rename `_accessor_to_pattern()` → `_accessor_to_group_key()` in
`parallel.py` for consistency with the SignalGroup schema.

---

## Phase 8: Data Reset and Re-Discovery

**Goal**: Once the enrichment pipeline is stable and tested, clear existing
signal data and re-run discovery for maximum quality.

### 8.1 Pre-Conditions (Must Complete Before Reset)

- [ ] Phase 1: Property normalization + dual model compliance
- [ ] Phase 2: Context injection gaps filled (TDI source_code, code refs, wiki)
- [ ] Phase 3: Post-propagation individualization working
- [ ] Phase 4: Structured enrichment output (no conflated descriptions)
- [ ] Phase 5: Mapping pipeline context extended
- [ ] Phase 6: Prompt ordering documented
- [ ] Phase 7: Graph migrations run
- [ ] Integration tests pass: enrich → propagate → group enrich → individualize → map
- [ ] Dry run on ~100 signals verifying quality improvement

### 8.2 Assessment: Code Bug vs Outdated Data

The `data_source_node` property gap is **both code and data** — scanners
don't set the property (code bug), and existing signals lack it (data gap).
Fix the scanners first, then run migrations, then re-discover.

The TDI `source_code` gap is **outdated data** — the scanner fix exists
(commit `603604fd`) but the graph data predates it. A `--rescan` fixes this.

The wiki path matching gap is **both** — `data_source_path` is not set at
discover time for TCV (data/code), AND `_find_wiki_context()` doesn't fall
back to accessor (code).

**Recommendation**: Fix all code bugs (Phases 1-7), verify on a small
subset, then full data reset (Phase 8).

### 8.3 Reset Procedure

```bash
# 1. Clear FacilitySignal data
uv run imas-codex graph migrate clear-signals --facility tcv
uv run imas-codex graph migrate clear-signals --facility jet

# 2. Clear SignalGroup nodes
uv run imas-codex graph migrate clear-signal-groups --facility tcv
uv run imas-codex graph migrate clear-signal-groups --facility jet

# 3. Re-discover (with normalized properties)
uv run imas-codex discover signals tcv
uv run imas-codex discover signals jet

# 4. Re-enrich (includes group detection, enrichment, propagation,
#    group enrichment, individualization)
uv run imas-codex discover signals tcv --enrich
uv run imas-codex discover signals jet --enrich

# 5. Re-check
uv run imas-codex discover signals tcv --check
uv run imas-codex discover signals jet --check

# 6. Re-embed
uv run imas-codex embed index --labels FacilitySignal,SignalGroup
```

---

## Implementation Order

```
Phase 1 ──────────────────────────────────────────── Normalize Properties
  1.1  Set data_source_node in all scanners [scanner code + migration]
  1.2  Set data_source_path in all scanners [scanner code + migration]
  1.3  Define canonical vs scanner-specific  [documentation]
  1.4  Normalize enrichment_source           [mark functions]
  1.5  Add --rescan flag to discover signals [CLI]

Phase 2 ──────────────────────────────────────────── Context Gaps
  2.1  Re-scan TDI functions                 [--rescan command]
  2.2  Add fetch_signal_code_refs()          [new function + prompt injection]
  2.3  Fix wiki path matching fallback       [1 line + Phase 1.2]
  2.4  SignalGroup structured LLM enrichment [new model + function]

Phase 3 ──────────────────────────────────────────── Individualize
  3.1  IndividualizedBatch Pydantic model    [new model]
  3.2  individualize_group_members()         [new function]
  3.3  Wire into enrich_worker pipeline      [pipeline integration]

Phase 4 ──────────────────────────────────────────── Structured Output
  4.1  Enforce separation of concerns        [prompt update]
  4.2  Extend SignalEnrichmentResult model   [model update]
  4.3  Simplify context quality assessment   [prompt update]

Phase 5 ──────────────────────────────────────────── Mapping Context
  5.1  Extend query_signal_groups()          [query update]
  5.2  Add code references to group detail   [query + format]
  5.3  Extend sample_members with accessors  [query update]

Phase 6 ──────────────────────────────────────────── Prompt Architecture
  6.1  Document prompt ordering in AGENTS.md [docs]
  6.2  Audit existing prompts                [review]

Phase 7 ──────────────────────────────────────────── Migrations
  7.1  Backfill properties                   [Cypher migration]
  7.2  Clean up legacy properties            [Cypher migration]
  7.3  Rename _accessor_to_pattern()         [function rename]

Phase 8 ──────────────────────────────────────────── Reset + Re-Run
  8.1  Verify pre-conditions                 [checklist]
  8.2  Clear and re-discover all signals     [CLI commands]
  8.3  Validate quality improvement          [comparison metrics]
```

---

## Deterministic vs LLM Steps

A core principle of the chained pipeline is that **deterministic steps use
function calls**, not LLM inference:

| Operation | Type | Implementation |
|-----------|------|----------------|
| Fetch tree context (parent, siblings, epochs) | Deterministic | `fetch_tree_context()` — Cypher query |
| Fetch code references for signal | Deterministic | `fetch_signal_code_refs()` — Cypher query (new) |
| Fetch wiki context by path | Deterministic | `_find_wiki_context()` — dict lookup |
| Fetch wiki context by semantics | Deterministic | `_fetch_group_wiki_context()` — vector search |
| Fetch code context by semantics | Deterministic | `_fetch_code_context()` — vector search |
| Fetch TDI source code | Deterministic | `get_tdi_source()` — graph query |
| Detect signal groups | Deterministic | `detect_signal_groups()` — accessor pattern matching |
| Generate signal description | **LLM** | `enrich_worker()` — structured output |
| Assess context quality | **LLM** | Part of enrichment structured output |
| Generate group description | **LLM** | `enrich_signal_group()` — structured output (new) |
| Individualize member descriptions | **LLM** | `individualize_group_members()` — structured output (new) |
| Assign groups to IMAS sections | **LLM** | `_step1_assign_sections()` |
| Generate field mappings | **LLM** | `_step2_field_mappings()` |
| Fetch IMAS subtree structure | Deterministic | `fetch_imas_subtree()` — Cypher query |
| Fetch IMAS field details | Deterministic | `fetch_imas_fields()` — Cypher query |
| Semantic search IMAS paths | Deterministic | `search_imas_semantic()` — vector search |
| Get COCOS sign-flip paths | Deterministic | `get_sign_flip_paths()` — coded tables |
| Analyze unit compatibility | Deterministic | `analyze_units()` — pint library |
| Validate mapping paths exist | Deterministic | `check_imas_paths()` — Cypher query |
| Validate transform execution | Deterministic | `validate_mapping()` — Python eval |
| Check duplicate mappings | Deterministic | `validate_mapping()` — set comparison |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| TCV signals with `data_source_node` property | 0 / 28,690 | 28,690 / 28,690 |
| TCV signals with `data_source_path` property | 0 / 28,690 | 28,690 / 28,690 |
| Signals with TDI source code | 0 / 338 | 338 / 338 |
| Signals with code references | 0 / 2,208 (unused) | 2,208 / 2,208 |
| Signals with wiki context (path match) | ~0 TCV | ~6,955 path matches |
| Unique descriptions (vs templated) | ~60% | >95% |
| SignalGroup descriptions (group-specific) | 0 | all groups |
| Mapping field accuracy (manual review on 50 mappings) | baseline TBD | >90% |
| Sign convention coverage | TCV 3.9%, JET 18% | >50% |
| `enrichment_source` always set | ~50% null | 0% null |
