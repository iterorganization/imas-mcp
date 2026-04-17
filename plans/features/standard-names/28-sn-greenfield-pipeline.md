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

1. **Graph is the ledger**: every worker that alters data writes to the graph immediately.
   The graph is the single source of truth, the state machine, and the coordination
   mechanism between workers. No in-memory state shuttling between pipeline stages.
   Workers claim work from the graph and write results back to the graph.

2. **Workers are named for what they do**: `extract` finds source paths, `name` generates
   names, `organize` canonicalizes and builds hierarchy, `enrich` writes documentation,
   `embed` computes vector embeddings. No `persist` — every worker persists its own output.

3. **Separation of concerns**: naming (grammar + physics) and documentation (exposition +
   context) are different cognitive tasks → different LLM calls with different prompts,
   contexts, and potentially different models.

4. **Context retrieval between LLM calls**: POSTLINK (inside `enrich_worker`) retrieves
   rich expository context that wasn't available before naming. Wiki documentation, code
   examples, facility signals — all searchable only once the standard name exists in the
   graph. Mirrors the proven discovery paths pipeline (TRIAGE → ENRICH → SCORE).

5. **Vector hierarchy as first-class pipeline stage**: GROUP (inside `organize_worker`)
   runs on globally canonicalized names. Not an afterthought bolted onto composition.

6. **Document-once pattern**: vector parents get canonical documentation; components get
   only direction-specific details with back-references. ~80% documentation deduplication
   within vector families.

7. **Global canonicalization before expensive enrichment**: `organize_worker` ensures no
   duplicate or conflicting names reach the expensive `enrich_worker`. In greenfield, the
   graph itself provides the dedup anchor — previously named batches are already committed.

8. **True branching on characteristics**: the `enrich_worker` routes names through different
   prompt templates based on type (vector parent / component / magnitude / standalone scalar /
   geometry) and retries on quality failure.

9. **Proven claim pattern for worker coordination**: workers claim items via
   `claimed_at` + `claim_token` with `@retry_on_deadlock()` and `ORDER BY rand()` —
   the same battle-tested pattern used by all discovery pipelines. Status enums
   (`review_status`, `validation_status`, `StandardNameSource.status`) define durable
   state transitions. Provenance timestamps (`generated_at`, `validated_at`,
   `consolidated_at`, `enriched_at`, `embedded_at`) track when each stage completed
   and gate downstream workers via `WHERE sn.consolidated_at IS NOT NULL AND
   sn.enriched_at IS NULL AND sn.claimed_at IS NULL`.


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
extract → name → organize → enrich → embed
           ↓         ↓         ↓        ↓
      [graph write] [graph write] [graph write] [graph write]
```

Five async workers. Each writes to the graph. The graph is the coordination mechanism —
workers claim items via status enum + claimed_at/claim_token, following the proven discovery pipeline pattern.

```
StandardNameSource lifecycle:  extracted → composed | attached | vocab_gap | failed
StandardName progression:       generated_at → validated_at → consolidated_at → enriched_at → embedded_at
```

### Worker 1: `extract_worker`

**Purpose**: Query source paths, classify, build batches, write source nodes.
**LLM**: None.
**Changes from current**: Minimal — already works well and already writes to graph.

- Query DD paths by `node_category` ∈ {`quantity`, `geometry`} (from DD classification plan)
- Also extract from facility signals (`--source signals`)
- Classify via path classifier (quantity/metadata/skip)
- Select primary + grouping clusters
- Build batches grouped by (grouping_cluster × unit), max batch size configurable
- **Graph write**: `StandardNameSource` nodes with `status = 'extracted'` and `batch_key`

Each `StandardNameSource` node carries a `batch_key` (e.g., `cluster_id/unit` or
`unclustered/ids/parent/unit`) that preserves the semantic grouping. The graph is
now the source of truth for what needs naming — `name_worker` claims sources by
`batch_key`, not random selection.

**Two-level coordination** (critical for LLM context quality):

| Level | Mechanism | Purpose |
|-------|-----------|---------|
| **Batch selection** | `name_worker` iterates unclaimed `batch_key` values | Preserves cluster×unit grouping — semantically similar paths go to the same LLM call |
| **Within-batch anti-deadlock** | `ORDER BY rand()` inside `claim_standard_name_source_batch(batch_key)` | Prevents lock convoys when parallel workers compete for the same batch |

This is the existing proven pattern from `graph_ops.py`: `claim_standard_name_source_batch()`
filters by `batch_key` first, then applies `ORDER BY rand()` within that batch. The LLM
receives a coherent set of paths sharing the same physics cluster and unit — this enables
focused context injection (IDS description, cluster semantics, coordinate conventions) and
produces better names than random path selection would.

### Worker 2: `name_worker` (replaces compose_worker)

**Purpose**: generate the standard name, kind, and grammar fields. Write valid names
to the graph immediately. Invalid names never become StandardName nodes.
**Model**: `get_model("reasoning")` — sonnet-class (good at naming, fast, cost-effective).
**Pattern**: async claim loop — claims `StandardNameSource` batches **by `batch_key`** from
graph, preserving the semantic grouping established by `extract_worker`.

**Internal sub-steps** (not separate workers — they form one atomic unit of work per batch):

#### 2a. PRELINK (context gathering — graph reads only)

Gather authoritative structured context that helps the LLM choose the right canonical term.
This is the current `_enrich_batch_items()` + `_prefetch_ids_context()` extracted cleanly.

| Context | Source | Why before naming |
|---------|--------|-------------------|
| Cross-IDS siblings | `IN_CLUSTER` traversal | Prevents synonymous names across IDSs |
| Unit + domain | DD `HAS_UNIT` relationship | Informs physics classification |
| COCOS info | `HAS_COCOS` relationship | Sign convention awareness |
| Coordinate specs | `HAS_COORDINATE` traversal | Dimensional context |
| Identifier schemas | `HAS_IDENTIFIER_SCHEMA` | Type classification |
| Sibling fields | Same parent in DD | Related quantities |
| IDS description + top sections | IDS metadata | IDS-level context |
| Already-named SNs | Graph query on existing `StandardName` nodes | Cross-batch dedup |

**Within-run dedup via graph**: in greenfield, previously named batches are already
committed as `StandardName` nodes. PRELINK queries these — the graph itself is the
dedup registry. No in-memory concept registry needed. This is the key simplification
from the graph-as-ledger architecture.

#### 2b. NAME (LLM call #1 — naming only)

**Prompt design** (stripped down from current compose):

| Included | Excluded |
|----------|----------|
| Grammar vocabulary + segment order | Documentation quality guidance |
| Anti-patterns table | LaTeX formatting rules |
| Template rules + naming guidance | Cross-reference instructions |
| PRELINK context (siblings, unit, COCOS) | Tags, links, validity_domain, constraints |
| Existing SN names from graph (dedup) | Detailed documentation examples |
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
    attachments: list[...]  # Paths mapping to existing names in graph
    skipped: list[...]      # Non-quantity paths
    vocab_gaps: list[...]   # Missing grammar tokens
```

**Post-processing** (same as current):
- Unit injection from DD (authoritative)
- Physics domain injection from DD
- COCOS metadata injection
- Grammar round-trip normalization (parse → compose → verify)

**Batch sizing**: larger batches possible (30-40 paths) since the task is simpler. The
prompt is ~50% smaller without documentation rules — more token budget for paths.

#### 2c. VALIDATE₁ (early grammar gate — no graph write)

Pre-validation before committing names to graph. Invalid names **never become
StandardName nodes** — they are rejected before the graph write.

Checks (subset of current validation):
1. Grammar round-trip: `compose(parse(name)) == name`
2. Pydantic model construction: `create_standard_name_entry(...)` succeeds
3. Unit consistency: DD unit is valid for the claimed kind
4. No vocabulary violations: all segments use known tokens

**Routing**:
- ✅ Valid → write to graph (2d below)
- ❌ Grammar failure → rejection reason written to `StandardNameSource.last_error`,
  source status → `failed`. The failed source is visible in graph for observability.
- ⚠️ Vocabulary gap → `VocabGap` node created, source status → `vocab_gap`

#### 2d. Graph write (per batch)

- **Creates `StandardName` nodes with `review_status = 'drafted'`, `validation_status = 'valid'`, `generated_at`, `validated_at`
- **Creates** `HAS_STANDARD_NAME` relationships from source IMASNode/FacilitySignal
- **Creates** `HAS_UNIT` relationship
- **Updates** `StandardNameSource` status → `composed` | `attached` | `vocab_gap` | `failed`
- Sets `review_status = 'drafted'`, `validation_status = 'valid'` (passed VALIDATE₁)

This is the critical design point: **names are committed to the graph immediately after
validation, not buffered for a final persist**. Subsequent name_worker batches will see
these names via PRELINK graph queries, providing natural cross-batch dedup.

**Parallel race note**: concurrent name_worker instances can generate near-duplicate names
before either sees the other's write. This is acceptable — `organize_worker` is the
authoritative canonicalization pass. The graph write is an optimistic commit; organize
is the global consistency check.

### Worker 3: `organize_worker` (barriered reducer — not a claim loop)

**Purpose**: global canonicalization + vector hierarchy creation. Runs as a single pass
after `name_worker` completes (barriered via `depends_on: ["name_phase"]`).
**LLM**: None (could optionally use LLM for synonym resolution in complex cases).
**Pattern**: single-pass reducer, not a claim loop. Reads all named SNs, writes results.

This is explicitly a **barrier/reducer** — it processes the complete set of named SNs
from the current run, not incremental batches. It must wait for all naming to complete.

#### 3a. CONSOLIDATE (canonicalization)

Operations on all `StandardName` nodes where `validated_at IS NOT NULL AND consolidated_at IS NULL`:
1. **Cross-batch dedup**: detect identical names from different batches → merge source paths
2. **Synonym detection**: names with high embedding similarity but different strings → flag
3. **Unit consistency**: same name from different sources must have same unit
4. **Physics domain reconciliation**: majority vote across sources
5. **Source path aggregation**: collect all DD paths that map to each canonical name

#### 3b. GROUP (deterministic vector hierarchy — zero LLM)

Incorporates the full design from `27-sn-vector-hierarchy.md`:

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

#### 3c. Graph write (single pass)

- **Sets `consolidated_at = datetime()` on all approved canonical names
- **Creates** new vector parent + magnitude `StandardName` nodes (with `named_at` AND
  ``consolidated_at` set — they skip the name_worker since they're deterministically generated)
- **Creates** `HAS_COMPONENT`, `HAS_MAGNITUDE` relationships
- **Merges** duplicate names (loser gets `superseded_by` pointing to winner)
- Logs conflicts and coverage gaps to state for reporting

**Why after NAME completes**: vector detection from `_component_of_` is only reliable
after global naming. Same family split across batches would never regroup otherwise.
CONSOLIDATE must also see all names to detect synonyms/conflicts.

### Worker 4: `enrich_worker` (documentation via LLM)

**Purpose**: generate comprehensive physics documentation for named + organized SNs.
**Model**: `get_model("reasoning")` or dedicated enrichment model — sonnet or opus for
maximum documentation quality.
**Pattern**: async claim loop (claims `StandardName` nodes where `consolidated_at IS NOT NULL
AND enriched_at IS NULL AND claimed_at IS NULL`).

**Internal sub-steps** (per batch):

#### 4a. POSTLINK (rich context retrieval — triggered by names)

The critical innovation: once the name exists in the graph, we can retrieve expository
context that was unavailable during naming.

| Context | Retrieval method | Why after naming |
|---------|-----------------|------------------|
| Wiki documentation | Vector search over `wiki_chunk_embedding` using SN description | Needs the name/description to search |
| Code examples | Vector search over `code_chunk_embedding` | Needs the name to find relevant code |
| Facility signals | `HAS_STANDARD_NAME` traversal or description match | Cross-references to real measurements |
| Similar SN documentation | Vector search over `standard_name_desc_embedding` | Only meaningful once names exist |
| Vector family context | `HAS_COMPONENT` / `HAS_MAGNITUDE` traversal | Only available after GROUP |
| Diagnostic context | Signal → Diagnostic traversal | Which diagnostics measure this? |

**Vector-aware context assembly**:
- **For vector parents**: list all known components + their coordinate bases
- **For components**: retrieve parent's documentation (if parent enriched first) for
  document-once inheritance
- **For magnitudes**: retrieve parent's documentation for norm-specific context
- **For standalone scalars**: standard retrieval (wiki, code, signals)

**Implementation**: `imas_codex/standard_names/postlink.py` with per-type context builders.

#### 4b. ENRICH (LLM call #2 — documentation only)

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

**Batch sizing**: smaller batches (10-15 items) since each item carries richer per-item
context from POSTLINK. Quality over throughput.

**Documentation inheritance execution order**:
1. Enrich vector parents FIRST (37 LLM calls — canonical docs)
2. Then enrich components with parent docs as context (117 calls — differential only)
3. Then enrich magnitudes with parent docs (37 calls — norm-specific)
4. Then enrich standalone scalars (remaining — standard docs)

#### 4c. VALIDATE₂ (documentation quality gate)

Checks:
1. **ISN 3-layer**: Pydantic → semantic → description (existing)
2. **Documentation link resolution**: every `[name](#name)` link resolves to a real SN
3. **Vector hierarchy consistency**: all components' units match parent's unit
4. **Description quality**: minimum length, contains key physics terms
5. **Tag validation**: only valid secondary tags (from ISN vocabulary)

**Routing**:
- ✅ All checks pass → graph write
- ⚠️ Documentation quality low → **retry ENRICH** with richer context or opus model
  (max 2 retries before accepting as-is)
- ❌ Critical failure → quarantine (set `validation_status = 'quarantined'`)

#### 4d. Graph write (per batch)

- **Updates** `StandardName` nodes: description, documentation, tags, links, validity_domain,
  constraints, validation_issues, validation_layer_summary
- **Sets `enriched_at = datetime()`, clears `claimed_at`, `claim_token``
- **Sets** `model` (which model produced the documentation)
- **Updates** `validation_status` based on VALIDATE₂ results

### Worker 5: `embed_worker` (replaces persist_worker)

**Purpose**: compute vector embeddings for enriched StandardNames.
**LLM**: None — uses embedding server.
**Pattern**: async claim loop (claims `StandardName` nodes where `enriched_at IS NOT NULL
AND embedded_at IS NULL AND claimed_at IS NULL`).
**Changes from current**: rename only — the current persist_worker already does exactly this.

- Claims batch from graph via `claim_token` pattern
- Computes embeddings via embedding server
- **Graph write**: sets `embedding`, `embedded_at = datetime()` on StandardName nodes


### Graph State Machine

Workers coordinate using the **proven claim pattern** from the discovery pipeline
(`claims.py`): status enum for durable state, `claimed_at` + `claim_token` for
worker coordination, `@retry_on_deadlock()` + `ORDER BY rand()` for deadlock avoidance.

```
StandardNameSource (per DD path / signal):
  extracted ──→ composed     (name_worker: LLM generated a new SN)
       │         attached     (name_worker: auto-linked to existing SN)
       │         vocab_gap    (name_worker: grammar token missing)
       └──────→ failed        (name_worker: validation rejected pre-graph-write)
       └──────→ stale         (sn reconcile: source entity removed)

StandardName (per canonical name):
  drafted ──→ drafted        (worker progression via status-gated claim queries)
  review_status:  drafted → published → accepted (human review lifecycle)
  validation_status: pending → valid | quarantined (ISN grammar checks)
```


**Claim patterns** — two distinct approaches for the two node types:

**StandardNameSource claims** (name_worker — batch-key-grouped):
```python
@retry_on_deadlock()
def claim_standard_name_source_batch(batch_key, limit=50):
    token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query("""
            MATCH (sns:StandardNameSource)
            WHERE sns.batch_key = $batch_key
              AND (
                (sns.status = 'extracted' AND sns.claimed_at IS NULL)
                OR (sns.claimed_at IS NOT NULL
                    AND sns.claimed_at < datetime() - duration({minutes: $timeout}))
              )
            WITH sns ORDER BY rand() LIMIT $limit
            SET sns.claimed_at = datetime(), sns.claim_token = $token
        """, ...)
        return token, list(gc.query(
            "MATCH (sns:StandardNameSource {claim_token: $token}) RETURN ...", ...))
```

The `batch_key` filter **preserves semantic grouping** — all claimed sources share
the same cluster x unit context. `ORDER BY rand()` is safe here because it only
randomizes within a single batch, not across the entire population. This is the
existing proven pattern from `graph_ops.py::claim_standard_name_source_batch()`.

**StandardName claims** (enrich_worker, embed_worker — random selection):
```python
@retry_on_deadlock()
def claim_for_enrichment(limit=20):
    token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query("""
            MATCH (sn:StandardName)
            WHERE sn.review_status = 'drafted'
              AND sn.validation_status = 'valid'
              AND sn.consolidated_at IS NOT NULL
              AND sn.enriched_at IS NULL
              AND (sn.claimed_at IS NULL
                   OR sn.claimed_at < datetime() - duration($timeout))
            WITH sn ORDER BY rand() LIMIT $limit
            SET sn.claimed_at = datetime(), sn.claim_token = $token
        """, ...)
        return token, list(gc.query(
            "MATCH (sn:StandardName {claim_token: $token}) RETURN ...", ...))
```

For enrichment and embedding, batch grouping is less critical — each name is
enriched independently with its own POSTLINK context. `ORDER BY rand()` is the
correct choice here for maximum parallelism.

**Worker → claim query → mark fields**:

| Worker | Claims where | On success sets |
|--------|-------------|-----------------|
| `name_worker` | `StandardNameSource.batch_key = $key AND status = 'extracted' AND claimed_at IS NULL` | Source: `status = 'composed'`, `composed_at`. SN: `review_status = 'drafted'`, `generated_at`, `validated_at`, `validation_status = 'valid'` |
| `organize_worker` | `StandardName.validated_at IS NOT NULL AND consolidated_at IS NULL` (barrier — runs after name_phase completes) | `consolidated_at`, vector hierarchy relationships |
| `enrich_worker` | `StandardName.consolidated_at IS NOT NULL AND enriched_at IS NULL AND claimed_at IS NULL` | `enriched_at`, documentation, tags, links |
| `embed_worker` | `StandardName.enriched_at IS NOT NULL AND embedded_at IS NULL AND claimed_at IS NULL` | `embedded_at`, embedding vector |

**Key**: `claimed_at IS NULL` in every claim query. `claim_token` two-step verify prevents
double-claiming. `@retry_on_deadlock()` handles Neo4j transient errors. `ORDER BY rand()`
avoids lock convoys. Timestamps (`generated_at`, `validated_at`, etc.) are **provenance
markers** for observability — the actual gating is via status checks and NULL timestamp
tests, coordinated by claimed_at/claim_token.


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

### Phase 0: Prerequisites (from DD Classification Plan) — ✅ complete

- ✅ DD `node_category` labels in place (`quantity`, `geometry`, `coordinate`, `metadata`)
  — verifiable via `get_ids_summary` / `list_dd_paths --node-category`.
- ✅ DD nodes re-enriched with richer sonnet descriptions (see distribution probes in
  `plans/features/dd-search-quality-ab.md`).
- ✅ Classifier bugs fixed (reversed traversal, overbroad coordinate).
- ✅ Existing claim infrastructure (`claimed_at`, `claim_token`, `@retry_on_deadlock`)
  covers new workers.
- ✅ `cocos_label_transformation` and `cocos_transformation_expression` present on all
  COCOS-dependent DD leaves — used directly in PRELINK + name-worker prompt.
- ✅ `find_related_dd_paths` cross-IDS similarity bug (self-hits, missing cocos_kin) is
  **fixed** — `cocos_kin` section now returns COCOS-matched siblings on live graph.
- ✅ `search_dd_paths` / `list_dd_paths` accept `cocos_transformation_type` filter and
  surface the COCOS line in the formatter — usable from inside name_worker prompts.

### Phase 1: `name_worker` (replaces compose_worker)

**Goal**: Replace single COMPOSE with NAME-only LLM call + immediate graph write.

1. Add `named_at` timestamp to StandardName schema (LinkML)
2. Create naming-focused prompt (`sn/name_system.md`, `sn/name_dd.md`)
   - Strip documentation guidance from compose_system.md
   - Keep grammar, vocabulary, anti-patterns, naming rules
   - Smaller system prompt (~15K tokens vs ~30K)
3. Extract PRELINK context gathering from `_enrich_batch_items()` into clean module
   - Use the fixed `find_related_dd_paths` to retrieve cross-IDS siblings
   - Include `cocos_kin` as a first-class context slot (COCOS-aware grouping)
   - Filter candidate sources by `node_category=quantity` via `list_dd_paths`
4. Implement `name_worker()` with claim loop:
   - Claims StandardNameSource batches from graph
   - Runs PRELINK (graph reads)
   - Calls NAME LLM (naming only)
   - Runs VALIDATE₁ (grammar gate, in-memory)
   - Writes valid StandardName nodes to graph with `named_at`
   - Updates StandardNameSource status (composed/failed/vocab_gap)
5. Rename `persist_worker` → `embed_worker` (no behavior change, just naming)
6. Update pipeline DAG: extract → name → (organize) → enrich → embed

**Tests**: existing `sn generate --name-only` tests provide baseline. Add tests for:
- NAME produces valid grammar but no documentation
- Graph write creates StandardName with `named_at` set
- Failed names write rejection reason to StandardNameSource.last_error
- Cross-batch dedup via graph query in PRELINK

### Phase 2: `organize_worker` (consolidate + group)

**Goal**: Global canonicalization + vector hierarchy as barriered reducer.

1. Add `organized_at` timestamp to StandardName schema
2. Implement `organize_worker()` as single-pass reducer (depends_on name_phase)
3. CONSOLIDATE sub-phase: cross-batch dedup, synonym detection, conflict resolution
4. GROUP sub-phase: detect vector families, create parents + magnitudes
5. Add `HAS_COMPONENT`, `HAS_MAGNITUDE` to SN schema
6. Graph write: set `organized_at`, create vector/magnitude SNs, wire relationships
7. Implement `superseded_by` for merged duplicates

**Tests**: see `27-sn-vector-hierarchy.md` for detailed vector test strategy. Add:
- Consolidation merges identical names from different batches
- Vector parent creation from component names
- Eligibility checks (tensor exclusion, min 2 components)
- `organized_at` set on all approved names

### Phase 3: `enrich_worker` (POSTLINK + LLM documentation)

**Goal**: Generate documentation with rich dynamically-retrieved context.

1. Add `enriched_at` timestamp to StandardName schema
2. Create `postlink.py` with per-type context builders:
   - `build_vector_parent_context(name, components)`
   - `build_component_context(name, parent_docs)`
   - `build_magnitude_context(name, parent_docs)`
   - `build_scalar_context(name)` — wiki/code/signal retrieval
3. Create documentation-focused prompt (`sn/enrich_system_v2.md`, `sn/enrich_dd.md`)
   - Add vector-aware sections (parent/component/magnitude templates)
   - Add POSTLINK context slots
   - Branching prompt templates per SN type
4. Implement `enrich_worker()` with claim loop:
   - Claims organized SNs from graph
   - Runs POSTLINK (context retrieval)
   - Calls ENRICH LLM (documentation only, branching by type)
   - Runs VALIDATE₂ (quality gate)
   - Writes documentation to graph with `enriched_at`
   - Retry on low quality (max 2 retries)
5. Documentation inheritance execution order: parents → components → magnitudes → standalone

**Tests**: verify component docs shorter than parent docs; verify cross-references resolve;
verify retry improves quality scores; verify quarantine captures critical failures.

### Phase 4: MCP Tool Integration

**Goal**: Surface vector hierarchy and pipeline-aware filtering in MCP tools.

1. `search_standard_names`: show HAS_COMPONENT children for vector results
2. `search_standard_names`: show vector parent for scalar component results
3. Add `kind` filter parameter
4. ✅ `node_category` filter on `search_dd_paths`/`list_dd_paths` — **shipped**
5. ✅ `cocos_transformation_type` filter on `search_dd_paths`/`list_dd_paths` — **shipped**
   (see `plans/features/dd-search-quality-ab.md`). Use from `sn review` / `sn benchmark`
   to scope scoring runs to a single COCOS family.

### Phase 5: Follow-ons from DD search quality A/B (Dec 2024)

Newly unlocked by the DD tool upgrades — worth planning once Phase 1 lands:

1. **COCOS-aware PRELINK filtering**: when a source has
   `cocos_label_transformation`, scope the "related paths" slot to the same label via
   `find_related_dd_paths(..., relationship_types=["cocos_kin"])`. Expect stronger
   sign-convention consistency in generated names.
2. **Keyword vocab-gap detection**: the LLM-enriched `keywords` field on DD nodes is
   dense and physics-specific. Cross-check vocab gaps (`VocabGap` nodes) against DD
   keywords to propose high-precision additions to the ISN grammar vocabulary before
   re-running name_worker.
3. **Unit-companion anchoring for name consolidation**: CONSOLIDATE can use the
   `unit_companions` output of `find_related_dd_paths` as a weak-dedup signal
   (same physical quantity across IDSs → single SN).

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

2. **Parallel name_worker races**: concurrent name_worker instances can generate
   near-duplicate names before either sees the other's write. The organize_worker handles
   this post-hoc. Should we add a semaphore to serialize graph writes within name_worker,
   or accept the race and let organize clean up?

3. **ENRICH retry budget**: how many retries before accepting as-is? Suggest max 2 retries
   with progressively richer context (retry 1: add wiki context, retry 2: switch to opus).

4. **Geometry SN documentation**: geometry nodes (coil positions, vessel outlines) need
   engineering-focused documentation, not plasma physics documentation. The ENRICH prompt
   must branch on `node_category` as well as SN type.

5. **Signal-sourced names**: when `--source signals` is used, PRELINK and POSTLINK need
   different retrieval strategies (facility-specific context vs DD-wide context). This is
   a Phase 2+ consideration.

6. **Run scoping**: for production use (non-greenfield), a `run_id` on StandardNameSource
   and StandardName would prevent mixing runs. In greenfield mode (graph starts empty)
   this is unnecessary. Add as optional infrastructure when moving to incremental updates.

---

## RD Review History

### Round 1 (Pre-creation critique — incorporated into design)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | Need early validation before GROUP/LINK | Blocking | Added VALIDATE₁ between NAME and CONSOLIDATE |
| 2 | Greenfield removes main dedup anchor | Blocking | Added within-run concept registry in PRELINK |
| 3 | Some LINK context needed before NAME | Blocking | Split into PRELINK (structured, before NAME) and POSTLINK (expository, after GROUP) |
| 4 | GROUP should run on globally normalized names | Blocking | GROUP runs after CONSOLIDATE, not on raw batch output |

### Round 2 (Graph-as-ledger restructuring)

User feedback: the plan described a linear 10-stage pipeline with a final PERSIST. Every
stage that alters data should write to the graph. The graph is the ledger and state machine.

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | Need run scoping (`run_id`) for mixing runs/retries | Blocking (production) | Noted as Open Question #6. Greenfield starts empty — not blocking for initial implementation. |
| 2 | Parallel name_worker races can create near-duplicates | Medium | Accepted: organize_worker is authoritative canonicalization. Graph write is optimistic; organize is global consistency. |
| 3 | Ledger vs mutable state machine conflation | Medium | Resolved: accept state machine model. Rejection history visible via StandardNameSource.last_error + status. |
| 4 | organize_worker is a reducer, not a claim-loop worker | Medium | Modeled as barriered phase (`depends_on: ["name_phase"]`), single-pass. |
| 5 | Pipeline progress uses proven claim pattern | Medium | Use existing `claimed_at` + `claim_token` + status enum + provenance timestamps (not timestamps as primary coordination). |
| 6 | Within-run registry can be the graph itself | Minor | Adopted: PRELINK queries existing StandardName nodes. No in-memory registry needed. |
| 7 | Workers should be named for what they do | Minor | Adopted: persist → embed, compose → name, new organize worker. |
