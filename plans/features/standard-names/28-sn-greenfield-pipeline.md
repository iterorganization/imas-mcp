# SN Greenfield Pipeline Redesign

## Problem Statement

The standard name pipeline generates names and documentation in a single LLM call per
batch, entangling two fundamentally different cognitive tasks: **naming** (grammar + physics
classification) and **documentation** (exposition + context synthesis). This monolithic
approach limits both naming quality and documentation depth — evidenced by `sn enrich`
existing as a post-hoc repair for weak compose-time documentation, and review scores
consistently underperforming on the documentation dimension.

All existing standard names **will be cleared and regenerated** as the pipeline evolves.
This plan designs the ideal pipeline from zero, incorporating vector hierarchy, documentation
inheritance, and multi-pass LLM architecture as natural pipeline stages rather than
bolted-on additions.

### Why Greenfield

| Aspect | Current (incremental) | Greenfield |
|--------|----------------------|------------|
| **Starting state** | ~800 drafted SNs with mixed quality | Empty graph — clean slate |
| **Vector hierarchy** | Bolted-on GROUP phase after consolidation | Natural pipeline stage between NAME and ENRICH |
| **Documentation** | Single-pass → post-hoc enrichment repair | Two-stage: name first, document with full context |
| **Dedup** | Relies on existing catalog as anchor | Within-run concept registry, global canonicalization |
| **Context retrieval** | All context gathered before single LLM call | Dynamic retrieval between stages — generated names unlock richer context |
| **Quality gating** | Validation as a separate phase | Inline gates with retry branching |

---

## Design Principles

1. **Separation of concerns**: naming (grammar + physics) and documentation (exposition +
   context) are different cognitive tasks → different LLM calls with different prompts,
   contexts, and potentially different models.

2. **Context retrieval between LLM calls**: the POSTLINK stage is the critical innovation —
   once we HAVE the name, we can do much richer context retrieval than was possible before
   naming. This mirrors the proven discovery paths pipeline (TRIAGE → ENRICH → SCORE).

3. **Vector hierarchy as first-class pipeline stage**: GROUP runs naturally between NAME
   and ENRICH as a deterministic pass on consolidated names. Not an afterthought.

4. **Document-once pattern**: vector parents get canonical documentation; components get
   only direction-specific details with back-references. ~80% documentation deduplication
   within vector families.

5. **Global canonicalization before expensive enrichment**: the within-run concept registry
   (CONSOLIDATE) ensures no duplicate names reach the expensive ENRICH stage. Critical for
   greenfield where there's no prior catalog to dedup against.

6. **True branching on confidence and characteristics**: the pipeline routes names through
   different prompt templates based on type (vector parent / component / magnitude / standalone
   scalar) and quality (retry low-confidence, quarantine invalid).

---

## Evidence

### Decomposition Already Works

The existing `--name-only` flag demonstrates that naming can be cleanly separated from
documentation. It produces names, nulls out doc fields, and relies on `sn enrich` for
documentation — exactly the NAME → ENRICH decomposition proposed here. The difference
is that this plan makes the decomposition structural, not optional.

### Multi-Pass Produces Better Results

The discovery paths pipeline uses a three-pass architecture that produces dramatically
better results than single-pass:

```
TRIAGE (LLM pass 1, cheap model, directory structure only)
  → ENRICH (SSH analysis, no LLM — gathers deep evidence)
  → SCORE (LLM pass 2, enriched context, authoritative scoring)
```

Key insight: the enrichment step between LLM calls retrieves context that wasn't available
during triage. The same principle applies here — once we have the standard name, we can
search for wiki documentation, facility signals, and similar names that weren't available
during the naming pass.

### Model Selection Matters

Enrichment benchmark (5-node pilot with real graph writes):

| Model | Time | Cost | Quality |
|-------|------|------|---------|
| gemini-3.1-flash-lite | 2.6s | $0.019 | Adequate — generic descriptions |
| claude-sonnet-4.6 | 14.4s | $0.021 | Excellent — precise physics context |
| claude-opus-4.6 | 25.7s | $0.021 | Excellent — includes formal definitions |

For naming: sonnet is sufficient (grammar + classification task).
For documentation: sonnet or opus produces dramatically better physics exposition.
Cost difference is negligible (~$5 across full regeneration).

### Documentation Redundancy Is Real

117 component SNs carry 237 KB of documentation, with ~80% overlap within vector families.
The document-once pattern reduces this to ~147 KB stored while improving consistency —
update the parent and all components inherit the correction.

---

## Pipeline Architecture

### Overview

```
EXTRACT → PRELINK → NAME → VALIDATE₁ → CONSOLIDATE → GROUP → POSTLINK → ENRICH → VALIDATE₂ → PERSIST
                                ↓                                              ↓
                           quarantine                                     retry / quarantine
```

Ten stages. Two LLM calls (NAME, ENRICH). Two validation gates (VALIDATE₁, VALIDATE₂).
One deterministic grouping pass (GROUP). Two context retrieval passes (PRELINK, POSTLINK).

### Stage 1: EXTRACT

**Purpose**: Query source paths, classify, build initial batches.
**LLM**: None.
**Changes from current**: Minimal — already works well.

- Query DD paths by `node_category` ∈ {`physical_quantity`, `geometry`} (from DD classification plan)
- Also extract from facility signals (`--source signals`)
- Classify via path classifier (quantity/metadata/skip)
- Select primary + grouping clusters
- Build batches grouped by (grouping_cluster × unit), max batch size configurable
- Write `StandardNameSource` nodes for crash resilience

**Key difference for greenfield**: no existing StandardName nodes to auto-attach to. The
`attachment` code path only activates after the first generation run populates the registry.

### Stage 2: PRELINK (structured context retrieval before naming)

**Purpose**: Gather authoritative structured context that helps the LLM choose the right
canonical term. This is the current `_enrich_batch_items()` + `_prefetch_ids_context()`
extracted as an explicit stage.

**LLM**: None — pure graph queries.
**What it retrieves per path**:

| Context | Source | Why before NAME |
|---------|--------|-----------------|
| Cross-IDS siblings | `IN_CLUSTER` traversal | Prevents synonymous names across IDSs |
| Unit + domain | DD `HAS_UNIT` relationship | Informs physics classification |
| COCOS info | `HAS_COCOS` relationship | Sign convention awareness |
| Coordinate specs | `HAS_COORDINATE` traversal | Dimensional context |
| Identifier schemas | `HAS_IDENTIFIER_SCHEMA` | Type classification |
| Sibling fields | Same parent in DD | Related quantities |
| IDS description + top sections | IDS metadata | IDS-level context |
| Within-run registry | Previously named batches | Cross-batch dedup (greenfield critical) |

**Within-run registry**: In greenfield, there are no "existing names" to dedup against.
PRELINK maintains a process-wide registry of names generated so far. Each batch's NAME
output is registered before the next batch's PRELINK runs. This provides the same dedup
signal that the prior catalog used to provide.

### Stage 3: NAME (LLM call #1 — naming only)

**Purpose**: Generate the standard name, kind, and grammar fields. Nothing else.
**Model**: `get_model("reasoning")` — sonnet-class (good at naming, fast, cost-effective).

**Prompt design** (stripped down from current compose):

| Included | Excluded |
|----------|----------|
| Grammar vocabulary + segment order | Documentation quality guidance |
| Anti-patterns table | LaTeX formatting rules |
| Template rules + naming guidance | Cross-reference instructions |
| PRELINK context (siblings, unit, COCOS) | Tags, links, validity_domain, constraints |
| Within-run registry (dedup) | Detailed documentation examples |
| Category-specific guidance (physical vs geometry) | — |

**Output model** (`StandardNameBatch`):
```python
class NameCandidate(BaseModel):
    source_id: str        # DD path or signal ID
    name: str             # The standard name
    kind: str             # scalar / vector / metadata
    fields: dict          # Grammar segments (physical_base, subject, etc.)
    confidence: float     # 0-1
    reason: str           # Brief justification

class StandardNameBatch(BaseModel):
    candidates: list[NameCandidate]
    attachments: list[...]  # Paths mapping to existing names in registry
    skipped: list[...]      # Non-quantity paths
    vocab_gaps: list[...]   # Missing grammar tokens
```

**Post-processing** (same as current):
- Unit injection from DD (authoritative)
- Physics domain injection from DD
- COCOS metadata injection
- Grammar round-trip normalization (parse → compose → verify)

**Batch sizing**: Larger batches possible (30-40 paths) since the task is simpler. The
prompt is ~50% smaller without documentation rules — more token budget for paths.

### Stage 4: VALIDATE₁ (early grammar gate)

**Purpose**: Catch malformed names before investing in consolidation or enrichment.
**LLM**: None.

Checks (subset of current validation):
1. Grammar round-trip: `compose(parse(name)) == name`
2. Pydantic model construction: `create_standard_name_entry(...)` succeeds
3. Unit consistency: DD unit is valid for the claimed kind
4. No vocabulary violations: all segments use known tokens

**Branching**:
- ✅ Valid → CONSOLIDATE
- ❌ Grammar failure → **quarantine** with error detail (retry with smaller batch or manual fix)
- ⚠️ Vocabulary gap → record `VocabGap` node, skip (same as current)

**Why early**: prevents wasted ENRICH spend on names that will fail final validation.
Currently, validation runs after COMPOSE which includes expensive documentation.

### Stage 5: CONSOLIDATE (global canonicalization)

**Purpose**: Ensure every concept has exactly one canonical name before expensive enrichment.
**LLM**: None (could optionally use LLM for synonym resolution in complex cases).

Operations:
1. **Cross-batch dedup**: detect identical names from different batches → merge metadata
2. **Synonym detection**: names with high embedding similarity but different strings → flag
3. **Unit consistency**: same name from different sources must have same unit
4. **Physics domain reconciliation**: majority vote across sources
5. **Source path aggregation**: collect all DD paths that map to each canonical name
6. **Update within-run registry**: register all canonical names for subsequent batches

**Key for greenfield**: this is where the "single source of truth" for the generation run
crystallizes. Every name that passes consolidation is a canonical entry that GROUP and
ENRICH will process.

### Stage 6: GROUP (deterministic vector hierarchy — zero LLM)

**Purpose**: detect vector families from component names, create vector parents and
magnitudes.
**LLM**: None — fully deterministic.

This stage incorporates the full design from `27-sn-vector-hierarchy.md`:

1. **Parse** all consolidated `*_component_of_*` scalar SNs via ISN grammar
2. **Reconstruct parent** by removing only the `component` segment — preserving ALL
   qualifiers (subject, process, position, etc.)
3. **Group** by reconstructed parent signature → candidate vector families
4. **Eligibility check**:
   - Exclude interpolation/numerical artifacts
   - Exclude tensor/anisotropy (parallel+perpendicular pressure/temperature)
   - Require minimum 2 components
5. **Create** vector parent SN (kind=vector, unit from components)
6. **Create** magnitude companion (`magnitude_of_{parent}`, kind=scalar)
7. **Validate** all new names via ISN grammar round-trip
8. **Wire** relationships: HAS_COMPONENT (vector → component), HAS_MAGNITUDE (vector → magnitude)

**Scope**: ~37 vector parents + ~37 magnitude scalars + ~117 linked existing components.

**Why after CONSOLIDATE**: vector detection from `_component_of_` is only reliable after
global normalization. Same family split across batches would never regroup otherwise.

### Stage 7: POSTLINK (rich context retrieval — triggered by names)

**Purpose**: gather expository context that the naming pass couldn't access because the
names didn't exist yet. This is the critical innovation.
**LLM**: None — pure graph queries + vector search.

| Context | Retrieval method | Why after NAME |
|---------|-----------------|----------------|
| Wiki documentation | Vector search over `wiki_chunk_embedding` using SN description | Needs the name/description to search |
| Code examples | Vector search over `code_chunk_embedding` | Needs the name to find relevant code |
| Facility signals | `HAS_STANDARD_NAME` traversal (from prior runs) or description match | Cross-references to real measurements |
| Similar SN documentation | Vector search over `standard_name_desc_embedding` | Only meaningful once names exist |
| Vector family context | `HAS_COMPONENT` / `HAS_MAGNITUDE` traversal | Only available after GROUP |
| Diagnostic context | Signal → Diagnostic traversal | Which diagnostics measure this? |

**Vector-aware context assembly**:
- **For vector parents**: list all known components + their coordinate bases
- **For components**: retrieve parent's documentation (if parent was enriched first) for
  document-once inheritance
- **For magnitudes**: retrieve parent's documentation for norm-specific context
- **For standalone scalars**: standard retrieval (wiki, code, signals)

**Implementation**: new module `imas_codex/standard_names/postlink.py` with per-type
context builders. Each returns a `PostlinkContext` dataclass that the ENRICH prompt
template consumes.

### Stage 8: ENRICH (LLM call #2 — documentation only)

**Purpose**: generate comprehensive physics documentation given a named SN + rich POSTLINK
context.
**Model**: `get_model("reasoning")` or dedicated enrichment model — sonnet or opus for
maximum documentation quality.

**Prompt design** (focused on exposition):

| Included | Excluded |
|----------|----------|
| Named SN + kind + unit | Grammar rules (already validated) |
| POSTLINK context (wiki, code, signals) | Anti-patterns table |
| Documentation quality guidance (LaTeX, equations, typical values) | Naming guidance |
| Category-specific doc templates | Vocabulary/segment descriptions |
| Cross-references to other SNs in this batch | — |
| COCOS sign convention details | — |

**Output model** (`StandardNameEnrichBatch`):
```python
class EnrichItem(BaseModel):
    standard_name: str
    description: str          # One-sentence, <120 chars
    documentation: str        # Rich LaTeX, equations, typical values
    tags: list[str]           # Classification tags
    links: list[str]          # Cross-references to other SNs
    validity_domain: str      # Plasma region (core, SOL, confined)
    constraints: str          # Physical constraints
```

**Branching by type** (different prompt templates):

| Type | Prompt variant | Context emphasis |
|------|---------------|-----------------|
| **Vector parent** | Canonical physics documentation | Governing equations, all components listed, coordinate decomposition |
| **Scalar component** | Differential documentation only | "Direction-specific details for {component} of {parent}. See {parent} for governing physics." Parent docs provided as context. |
| **Magnitude** | Norm-specific documentation | "The magnitude |**B**| of {parent}." Parent docs provided as context. |
| **Standalone scalar** | Full documentation | Standard: definition, equations, measurement, typical values, COCOS |
| **Geometry** | Hardware documentation | Physical location, engineering context, installation parameters |

**Batch sizing**: Smaller batches (10-15 items) since each item carries richer per-item
context from POSTLINK. Quality over throughput.

**Documentation inheritance execution order**:
1. Enrich vector parents FIRST (37 LLM calls — canonical docs)
2. Then enrich components with parent docs as context (117 calls — differential only)
3. Then enrich magnitudes with parent docs (37 calls — norm-specific)
4. Then enrich standalone scalars (remaining — standard docs)

### Stage 9: VALIDATE₂ (comprehensive final validation)

**Purpose**: full ISN 3-layer validation + cross-referencing + hierarchy consistency.
**LLM**: None.

Checks:
1. **ISN 3-layer**: Pydantic → semantic → description (existing)
2. **Documentation link resolution**: every `[name](#name)` link resolves to a real SN
3. **Vector hierarchy consistency**: all components' units match parent's unit
4. **Description quality**: minimum length, contains key physics terms
5. **Tag validation**: only valid secondary tags (from ISN vocabulary)

**Branching**:
- ✅ All checks pass → PERSIST
- ⚠️ Documentation quality low → **retry ENRICH** with richer context or opus model
- ❌ Critical failure → **quarantine** (grammar, unit, kind problems)

### Stage 10: PERSIST (write + embed)

**Purpose**: write validated SNs to graph, embed descriptions.
**Changes from current**: Minimal — already works well.

- Write with coalesce semantics (existing)
- Embed descriptions via embedding server (existing)
- Wire all relationships: HAS_STANDARD_NAME, HAS_UNIT, HAS_COMPONENT, HAS_MAGNITUDE
- Set provenance fields: model, generated_at, dd_version, source_paths

---

## Branching Model

The pipeline is not purely linear. Genuine branching occurs at three decision points:

### After VALIDATE₁ (quality gate)

```
NAME output → VALIDATE₁
                ├── ✅ valid grammar     → CONSOLIDATE
                ├── ❌ grammar failure   → quarantine (log, skip, optionally retry)
                └── ⚠️ vocabulary gap   → VocabGap node (log, skip)
```

### After CONSOLIDATE (dedup routing)

```
CONSOLIDATE output
    ├── 🆕 new canonical name     → GROUP / POSTLINK / ENRICH
    ├── 🔗 attachment (existing)  → link only, skip ENRICH
    └── ⚠️ conflict (same name,  → arbitrate (higher confidence wins)
         different metadata)         or merge
```

### After VALIDATE₂ (quality gate with retry)

```
ENRICH output → VALIDATE₂
                  ├── ✅ all checks pass  → PERSIST
                  ├── ⚠️ doc quality low  → retry ENRICH (richer context / opus model)
                  └── ❌ critical failure  → quarantine
```

### ENRICH prompt branching (by type)

```
POSTLINK context → ENRICH
                     ├── vector parent    → canonical doc prompt
                     ├── component        → differential doc prompt (parent docs as context)
                     ├── magnitude        → norm-specific prompt (parent docs as context)
                     ├── standalone scalar → full documentation prompt
                     └── geometry         → hardware documentation prompt
```

This is genuine branching — different code paths, different prompt templates, different
context assembly. Not just error handling.

---

## Cost Model

### Per-Generation Run (~800 standard names from DD source)

| Stage | LLM calls | Model | Estimated cost |
|-------|-----------|-------|---------------|
| EXTRACT | 0 | — | $0 |
| PRELINK | 0 | — | $0 |
| NAME | ~35 batches × 1 call | sonnet | ~$15 |
| VALIDATE₁ | 0 | — | $0 |
| CONSOLIDATE | 0 (optional LLM for synonym resolution) | — | $0-5 |
| GROUP | 0 | — | $0 |
| POSTLINK | 0 | — | $0 |
| ENRICH | ~80 batches × 1 call | sonnet/opus | ~$40 |
| VALIDATE₂ | 0 | — | $0 |
| PERSIST | 0 | — | $0 |
| **Total** | **~115 calls** | | **~$55-60** |

### Comparison with Current Pipeline

| Pipeline | LLM calls | Cost | Quality |
|----------|-----------|------|---------|
| Current (single compose) | ~35 | ~$30 | Naming: good, Docs: adequate |
| Greenfield (NAME + ENRICH) | ~115 | ~$55-60 | Naming: good, Docs: excellent |
| With retry on low-quality | ~125 | ~$65 | Both: excellent |

The ~$30 marginal cost buys dramatically better documentation, vector hierarchy, and
documentation inheritance. Well within the $200 budget.

### Including Vector Hierarchy

| Item | Count | Cost |
|------|-------|------|
| Vector parent enrichment | ~37 | ~$2 |
| Magnitude enrichment | ~37 | ~$2 |
| Component re-enrichment (differential docs) | ~117 | ~$5 |
| **Vector total** | ~191 | ~$9 |

Grand total with vectors: ~$65-70 per full generation run.

---

## Implementation Phases

### Phase 0: Prerequisites (from DD Classification Plan)

- DD `node_category` labels in place (`physical_quantity`, `geometry`)
- DD nodes re-enriched with sonnet (better descriptions → better SN context)
- Classifier bugs fixed (reversed traversal, overbroad coordinate)

### Phase 1: Pipeline Decomposition (NAME / ENRICH split)

**Goal**: Replace single COMPOSE with separate NAME and ENRICH stages.

1. Create naming-focused prompt (`sn/name_system.md`, `sn/name_dd.md`)
   - Strip documentation guidance from compose_system.md
   - Keep grammar, vocabulary, anti-patterns, naming rules
   - Smaller system prompt (~15K tokens vs ~30K)

2. Create documentation-focused prompt (`sn/enrich_system_v2.md`, `sn/enrich_dd.md`)
   - Keep documentation quality guidance, LaTeX rules
   - Add vector-aware sections (parent/component/magnitude templates)
   - Add POSTLINK context slots

3. Implement `name_worker()` — adapted from current compose_worker with doc fields stripped
4. Implement `enrich_worker_v2()` — adapted from current enrich_worker with richer context
5. Update pipeline DAG registration to wire NAME → ENRICH

**Tests**: existing `sn generate --name-only` tests provide baseline. Add tests for:
- NAME produces valid grammar but no documentation
- ENRICH produces documentation but doesn't change names

### Phase 2: PRELINK / POSTLINK Context Retrieval

**Goal**: Extract context gathering into explicit pipeline stages.

1. Extract `_enrich_batch_items()` and `_prefetch_ids_context()` into `prelink.py`
2. Create `postlink.py` with per-type context builders:
   - `build_vector_parent_context(name, components)`
   - `build_component_context(name, parent_docs)`
   - `build_magnitude_context(name, parent_docs)`
   - `build_scalar_context(name)` — wiki/code/signal retrieval
3. Within-run concept registry for greenfield dedup
4. Wire PRELINK before NAME, POSTLINK before ENRICH in pipeline DAG

**Tests**: verify POSTLINK retrieves relevant context for each SN type.

### Phase 3: VALIDATE₁ (early grammar gate)

**Goal**: Catch malformed names before expensive enrichment.

1. Extract grammar validation from current validate_worker into `validate_grammar()`
2. Wire as gate between NAME and CONSOLIDATE
3. Implement quarantine routing for failed names
4. Add retry logic (optional: re-submit failed names in smaller batches)

**Tests**: parametrized tests with known-invalid names → quarantine.

### Phase 4: GROUP (vector hierarchy)

**Goal**: Detect vector families, create parents + magnitudes.

1. Add `HAS_COMPONENT`, `HAS_MAGNITUDE` to SN schema
2. Implement component grouping (`group_vector_families()`)
3. Implement eligibility checks (tensor exclusion, min 2 components)
4. Create vector parent + magnitude SNs (deterministic naming)
5. Golden test fixture: all `_component_of_` names → exact parent set

**Tests**: see `27-sn-vector-hierarchy.md` for detailed test strategy.

### Phase 5: Documentation Inheritance in ENRICH

**Goal**: Document-once pattern for vector families.

1. Implement execution ordering: parents → components → magnitudes → standalone
2. Create branching prompt templates per SN type
3. Component prompts include parent docs as authoritative context
4. Magnitude prompts include parent docs for norm-specific context

**Tests**: verify component docs are shorter than parent docs; verify cross-references resolve.

### Phase 6: VALIDATE₂ + Quality Gating

**Goal**: Comprehensive final validation with retry.

1. Implement documentation link resolution
2. Implement vector hierarchy consistency check
3. Implement quality-based retry routing (low doc quality → retry ENRICH with opus)
4. Wire retry loop with configurable max attempts

**Tests**: verify retry improves quality scores; verify quarantine captures critical failures.

### Phase 7: MCP Tool Integration

**Goal**: Surface vector hierarchy and type-aware filtering in MCP tools.

1. `search_standard_names`: show HAS_COMPONENT children for vector results
2. `search_standard_names`: show vector parent for scalar component results
3. Add `kind` filter parameter
4. Add `node_category` filter (from DD plan) to `search_dd_paths`, `list_dd_paths`

---

## Relationship to Other Plans

| Plan | Interaction |
|------|-------------|
| **`dd-unified-classification.md`** | **Prerequisite.** Provides `node_category` labels for EXTRACT source filtering. DD enrichment quality (flash-lite → sonnet) directly improves SN naming context. Must execute first. |
| **`27-sn-vector-hierarchy.md`** | **Incorporated.** The vector hierarchy design (37 parents, grouping logic, eligibility checks, doc inheritance) is now Stage 6 (GROUP) + Stage 8 (ENRICH branching) of this pipeline. Research findings preserved in that plan; implementation details here. |
| **`isn-standard-name-kind.md`** | **Concluded.** ISN requires no changes. Vector support exists in ISN today. |
| **`26-sn-pipeline-quality-iteration.md`** | **Superseded.** Quality improvements are structural in this redesign, not iterative patches. |

---

## Open Questions

1. **CONSOLIDATE LLM**: should synonym resolution use an LLM call (e.g., "are these two
   names synonymous?") or can embedding similarity + heuristics suffice? Start with
   heuristic, add LLM if precision is insufficient.

2. **Within-run registry ordering**: batches are processed concurrently (semaphore=5).
   The registry must be thread-safe and updated atomically between batches. Current
   implementation uses async semaphore — may need a shared registry with locking.

3. **ENRICH retry budget**: how many retries before quarantine? Suggest max 2 retries
   with progressively richer context (retry 1: add wiki context, retry 2: switch to opus).

4. **Geometry SN documentation**: geometry nodes (coil positions, vessel outlines) need
   engineering-focused documentation, not plasma physics documentation. The ENRICH prompt
   must branch on `node_category` as well as SN type.

5. **Signal-sourced names**: when `--source signals` is used, PRELINK and POSTLINK need
   different retrieval strategies (facility-specific context vs DD-wide context). This is
   a Phase 2+ consideration.

---

## RD Review History

### Round 1 (Pre-creation critique — incorporated into design)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | Need early validation before GROUP/LINK | Blocking | Added VALIDATE₁ between NAME and CONSOLIDATE |
| 2 | Greenfield removes main dedup anchor | Blocking | Added within-run concept registry in PRELINK |
| 3 | Some LINK context needed before NAME | Blocking | Split into PRELINK (structured, before NAME) and POSTLINK (expository, after GROUP) |
| 4 | GROUP should run on globally normalized names | Blocking | GROUP runs after CONSOLIDATE, not on raw batch output |
| 5 | Pipeline is mostly linear, not truly branched | Moderate | Added genuine branching: VALIDATE₁ gates, CONSOLIDATE routing, ENRICH type-branching, VALIDATE₂ retry |
| 6 | Batch sizing should differ by stage | Minor | NAME: 30-40 items, ENRICH: 10-15 items |
| 7 | REVIEW as inline quality gate | Minor | Added VALIDATE₂ retry routing; full REVIEW remains as separate command for catalog-wide scoring |
| 8 | Consolidate should run after NAME AND after ENRICH | Minor | Primary CONSOLIDATE after NAME; light field-merge after ENRICH if needed |
