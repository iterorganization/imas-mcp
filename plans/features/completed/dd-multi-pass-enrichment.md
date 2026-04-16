# DD Multi-Pass Enrichment

## Problem Statement

The current DD enrichment pipeline generates descriptions in a single LLM pass. The
enrichment context for each path includes ancestor documentation, sibling **names**,
child summaries, cluster labels, and unit/coordinate metadata — but NOT sibling
**descriptions**. This means the LLM can see that `temperature` has a sibling called
`density`, but not that `density` was described as "Local electron number density from
Thomson scattering / charge exchange diagnostics, expressed on the radial grid."

With the model upgrade to sonnet (from the DD Unified Classification plan), single-pass
enrichment will produce significantly better descriptions. Multi-pass adds a second
dimension: **contextual refinement** where each path's description benefits from
knowing what its siblings, cluster peers, and cross-IDS duplicates actually say.

### Evidence for Multi-Pass Value

The SN Greenfield Pipeline (`28-sn-greenfield-pipeline.md`) splits naming from
documentation precisely because documentation quality improves dramatically when the
pipeline has access to generated content from prior stages. The same principle applies
to DD enrichment: a description of `equilibrium/time_slice/profiles_1d/psi` improves
when the LLM knows how `core_profiles/profiles_1d/psi` was described — enabling
disambiguation, consistent terminology, and cross-referencing.

**Concrete failure modes** in single-pass descriptions:
1. **Ambiguous sibling pairs**: `r` / `z` / `phi` under `profiles_1d` get generic
   descriptions because the LLM doesn't know how the other spatial coordinates were
   described — each is enriched in isolation
2. **Cross-IDS inconsistency**: `psi` in `equilibrium` and `core_profiles` may get
   different descriptions (one mentioning poloidal flux, the other toroidal) because
   neither knows about the other
3. **Cluster members drift**: paths grouped in the same semantic cluster get
   independently generated descriptions that may use inconsistent terminology

### Cost/Benefit

| | Single-pass (current) | Multi-pass (this plan) |
|---|---|---|
| **LLM calls** | ~234 batches × 1 call | ~234 batches × 2 calls |
| **Cost** | ~$50 (sonnet) | ~$100 (sonnet × 2) |
| **Time** | ~3 hours | ~6 hours (sequential barrier) |
| **Description quality** | Good (sonnet is capable) | Better (contextual awareness) |
| **Downstream impact** | — | Better embeddings, better SN context, better MCP |

Total: ~$100 for the full rebuild. Well within the $200 budget. The 3-hour time
penalty is acceptable for a one-off rebuild.

---

## Design

### New Status: `refined`

Add a `refined` status to the `IMASNodeStatus` enum, extending the lifecycle:

```
built → enriched → refined → embedded
```

| Status | Meaning | Set By |
|--------|---------|--------|
| `built` | Path node created from DD XML | `build_worker` |
| `enriched` | Pass 1 description generated (single-path context) | `enrich_worker` |
| `refined` | Pass 2 description refined (sibling + peer context) | `refine_worker` |
| `embedded` | Vector embedding computed from refined description | `embed_worker` |

### New Worker: `refine_worker`

A sixth async worker in the DD pipeline that slots between enrich and embed:

```
EXTRACT → BUILD ──→ ENRICH ──→ REFINE ──→ CLUSTER
                └──→ EMBED ──────────────↗
```

The refine worker claims `enriched` nodes (Pass 1 complete) and writes `refined`
status. The embed worker is updated to claim `refined` nodes instead of `enriched`.

### Sibling-Ready Barrier (Query-Level, Not Global)

The key design challenge: the refine worker needs sibling descriptions from Pass 1.
If sibling A is enriched but sibling B is still `built`, refining A would miss B's
description.

**Solution**: The claim query for the refine worker includes a sibling-readiness
predicate — only claim nodes whose enrichable siblings are ALL past `built`:

```cypher
MATCH (n:IMASNode {status: 'enriched'})
WHERE n.node_category IN $enrichable_categories
  AND (n.claimed_at IS NULL
       OR n.claimed_at < datetime() - duration($cutoff))
  -- Sibling barrier: all enrichable siblings must be at least enriched
  AND NOT EXISTS {
    MATCH (n)-[:HAS_PARENT]->(parent)<-[:HAS_PARENT]-(sib:IMASNode)
    WHERE sib.node_category IN $enrichable_categories
      AND sib.status = 'built'
  }
WITH n ORDER BY rand() LIMIT $limit
SET n.claimed_at = datetime(), n.claim_token = $token
```

This creates a **natural wavefront**: leaf paths with few siblings become claimable
first (their siblings are enriched quickly). Deep subtrees with many siblings become
claimable later. This is per-parent, not global — no explicit barrier coordination
needed.

**Edge case**: Root-level paths and paths without enrichable siblings have no barrier
constraint and are immediately claimable once enriched.

### Refinement Context Gathering

For each claimed batch, the refine worker gathers richer context than Pass 1:

| Context Source | Pass 1 (enrich) | Pass 2 (refine) |
|----------------|-----------------|-----------------|
| Ancestor documentation | ✓ | ✓ |
| Sibling **names** | ✓ | ✓ |
| Sibling **descriptions** | ✗ | ✓ (from Pass 1) |
| Child summaries | ✓ | ✓ |
| Unit / coordinates | ✓ | ✓ |
| Cluster label | ✓ | ✓ |
| Cluster **peer descriptions** | ✗ | ✓ (same-cluster paths) |
| Cross-IDS duplicates | ✗ | ✓ (same name, different IDS) |
| Own Pass 1 description | ✗ | ✓ (refinement input) |

New context queries for `gather_refinement_context()`:

1. **Sibling descriptions**: Extend existing sibling query to return `sibling.description`
   (populated by Pass 1)
2. **Cluster peer descriptions**: `(n)-[:IN_CLUSTER]->(cl)<-[:IN_CLUSTER]-(peer)` —
   other paths in the same semantic cluster with their Pass 1 descriptions
3. **Cross-IDS duplicates**: Paths with the same leaf name in different IDSs:
   `MATCH (dup:IMASNode) WHERE dup.name = n.name AND dup.ids <> n.ids`

### Refinement Prompt

A separate prompt template (`imas/refinement.md`) with different objectives than the
enrichment prompt:

**System prompt** (static, cacheable):
- "You are refining an existing description of an IMAS data path"
- "Use sibling descriptions for consistent terminology within the family"
- "Use cluster peer descriptions to disambiguate from similar paths"
- "Use cross-IDS duplicates to ensure consistent physics descriptions"
- Output: refined description + optional disambiguation note

**User prompt** (dynamic, per-batch):
- Current path + Pass 1 description (the input to refine)
- Sibling descriptions (family context)
- Cluster peer descriptions (semantic neighborhood)
- Cross-IDS duplicates with their descriptions (disambiguation)
- Same unit/coordinate/ancestor context as Pass 1

The prompt instructs the LLM to:
1. **Preserve** correct content from Pass 1
2. **Disambiguate** from similar-but-different paths using sibling/peer context
3. **Cross-reference** related paths when useful ("See also: `core_profiles/.../psi`")
4. **Standardize** terminology within a family (if sibling says "poloidal", use "poloidal")

### Refinement Hash

A separate `refinement_hash` field on IMASNode, computed from:

```python
def compute_refinement_hash(
    pass1_description: str,
    sibling_descriptions: list[str],
    cluster_peers: list[str],
    model_name: str,
) -> str:
    """Hash includes all Pass 2 inputs for idempotency."""
    combined = f"{model_name}:{pass1_description}:" + \
               ":".join(sorted(sibling_descriptions)) + \
               ":".join(sorted(cluster_peers))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
```

This means:
- Re-running with same inputs → hash matches → skip (idempotent)
- Changing Pass 1 description → hash changes → re-refine
- New sibling added → hash changes → re-refine affected family
- Model change → hash changes → re-refine all

### Pipeline Wiring

The `run_dd_build_engine()` function adds the refine worker and updates dependencies:

```python
workers = [
    WorkerSpec("extract", "extract_phase", extract_worker),
    WorkerSpec("build", "build_phase", build_worker, depends_on=["extract_phase"]),
    WorkerSpec("enrich", "enrich_phase", enrich_worker, depends_on=["build_phase"]),
    WorkerSpec("refine", "refine_phase", refine_worker, depends_on=["enrich_phase"]),
    WorkerSpec("embed", "embed_phase", embed_worker, count=4, depends_on=["refine_phase"]),
    WorkerSpec("cluster", "cluster_phase", cluster_worker,
               depends_on=["enrich_phase", "embed_phase"]),
]
```

**Key changes**:
- `embed_worker` now depends on `refine_phase` (not `build_phase`)
- `embed_worker` claims `refined` nodes (not `enriched`)
- `refine_worker` depends on `enrich_phase`
- `cluster_worker` still depends on both enrich and embed (unchanged)

### DDBuildState Extensions

```python
@dataclass
class DDBuildState(DiscoveryStateBase):
    # ... existing fields ...

    # New: refinement tracking
    refine_stats: WorkerStats = field(default_factory=WorkerStats)
    refine_phase: PipelinePhase = field(init=False)

    @property
    def skip_refinement_hash(self) -> bool:
        """Bypass per-path refinement hash check."""
        return self.force or self.reset_to in ("extracted", "built", "enriched")
```

### Reset Extensions

Update `_RESET_CLEAR_FIELDS` to include the new status and fields:

```python
_RESET_CLEAR_FIELDS = {
    "built": [
        "description", "keywords", "enrichment_hash", "enrichment_model",
        "enrichment_source", "enriched_at", "physics_domain",
        "refinement_hash", "refined_at",  # NEW
        "embedding", "embedding_hash", "embedded_at",
    ],
    "enriched": [
        "refinement_hash", "refined_at",  # NEW
        "embedding", "embedding_hash", "embedded_at",
    ],
    "refined": [  # NEW
        "embedding", "embedding_hash", "embedded_at",
    ],
}

_RESET_SOURCE_STATUSES = {
    "built": ["enriched", "refined", "embedded"],
    "enriched": ["refined", "embedded"],
    "refined": ["embedded"],  # NEW
}
```

### `--reset-to enriched` Use Case

With the new status, `--reset-to enriched` becomes the fast path for re-refinement:
skip extract + build + Pass 1 enrichment, re-run only Pass 2 (refinement) + embed.
Useful when:
- Refinement prompt template changes
- New cluster assignments alter peer context
- Cross-IDS duplicate resolution improves

Cost: ~$50 (Pass 2 only) + embedding time.

---

## Implementation Steps

### Step 1: Schema Extension

**File**: `imas_codex/schemas/imas_dd.yaml`

Add `refined` to `IMASNodeStatus`:

```yaml
IMASNodeStatus:
  permissible_values:
    built:
      description: Path node created in graph from DD XML extraction
    enriched:
      description: LLM-generated description and keywords populated (Pass 1)
    refined:
      description: Description refined with sibling and peer context (Pass 2)
    embedded:
      description: Vector embedding generated from refined description
```

Add `refinement_hash` and `refined_at` to `IMASNode` attributes:

```yaml
refinement_hash:
  description: >-
    SHA256 hash of refinement context (Pass 1 description + sibling descriptions
    + cluster peer descriptions + model name). Used for idempotency — refinement
    is skipped when hash matches.
  range: string
refined_at:
  description: ISO 8601 timestamp of when this node was last refined
  range: datetime
```

### Step 2: Graph Operations

**File**: `imas_codex/graph/dd_graph_ops.py`

New functions:

- `claim_paths_for_refinement(limit, ids_filter)` — claim `enriched` nodes with
  sibling-readiness predicate (see Cypher above)
- `mark_paths_refined(updates)` — set `status='refined'`, clear `claimed_at`,
  write `refinement_hash` and `refined_at`
- `release_refinement_claims(path_ids)` — release on error
- `has_pending_refinement(ids_filter)` — check for remaining enriched nodes

Update existing:
- `claim_paths_for_embedding()` — change claim status from `enriched` to `refined`
- `has_pending_embedding()` — change status check from `enriched` to `refined`
- `_RESET_CLEAR_FIELDS` — add refinement fields, add `refined` target
- `_RESET_SOURCE_STATUSES` — add `refined` throughout

### Step 3: Context Gathering

**File**: `imas_codex/graph/dd_enrichment.py`

New function: `gather_refinement_context(client, paths, ids_info)` — extends
`gather_path_context()` with three additional queries:

1. **Sibling descriptions**: modify sibling query to also return `sibling.description`
2. **Cluster peer descriptions**: new query via `IN_CLUSTER` relationships
3. **Cross-IDS duplicates**: new query matching `name` across IDSs

New function: `build_refinement_messages(batch_contexts, ids_info)` — constructs
the system + user messages for the refinement prompt.

New function: `compute_refinement_hash(pass1_desc, sibling_descs, peer_descs, model)`.

### Step 4: Refinement Prompt

**File**: `imas_codex/llm/prompts/imas/refinement.md`

New prompt template (separate from `enrichment.md`) with objectives:
- Refine, don't rewrite — preserve correct Pass 1 content
- Use sibling descriptions for family-consistent terminology
- Use cluster peers for disambiguation
- Use cross-IDS duplicates for physics consistency
- Output: refined description + keywords (same Pydantic model as enrichment)

### Step 5: Refine Worker

**File**: `imas_codex/graph/dd_workers.py`

New `refine_worker(state, **_kwargs)` async function following the proven pattern:
1. Poll `claim_paths_for_refinement()`
2. Gather refinement context (sibling descriptions, cluster peers, cross-IDS)
3. Check refinement hash for idempotency
4. Call LLM with refinement prompt
5. Write results via `mark_paths_refined()`
6. Track cost on `state.refine_stats`

Update `DDBuildState`:
- Add `refine_stats`, `refine_phase`
- Add `skip_refinement_hash` property

Update `run_dd_build_engine()`:
- Add `refine_worker` to worker list
- Update `embed_worker` dependency from `build_phase` to `refine_phase`
- Wire `refine_phase.has_work_fn` — pending refinement OR enrich not done
- Wire `embed_phase.has_work_fn` — pending embedding OR refine not done

Update `embed_worker`:
- Claim `refined` nodes instead of `enriched`

### Step 6: CLI + Display

**File**: `imas_codex/cli/imas_dd.py`

- Add `--reset-to enriched` documentation for re-refinement use case
- Ensure `--reset-to built` clears refinement fields

**File**: `imas_codex/cli/dd_display.py` (or equivalent display code)

- Add refine phase to progress display (6 phases instead of 5)

### Step 7: Tests

- Test `claim_paths_for_refinement()` sibling barrier — assert nodes are NOT
  claimable when any enrichable sibling is still `built`
- Test natural wavefront — leaf paths with no siblings are immediately claimable
- Test hash idempotency — same inputs produce same hash, matching hash skips refinement
- Test reset cascades — `--reset-to built` clears refinement fields,
  `--reset-to enriched` clears only refinement + embedding fields
- Test embed worker now requires `refined` status
- Test pipeline wiring — refine depends on enrich, embed depends on refine

---

## Execution Order (relative to DD Unified Classification)

This plan integrates with the DD Unified Classification plan. The execution order is:

1. **DD Unified Classification Steps 0–12**: Schema, classifier, tests, code changes
2. **This plan Step 1**: Schema extension (add `refined` status + fields)
3. **DD Unified Classification Step 13**: In-place reclassification
4. **This plan Steps 2–6**: Graph ops, context gathering, prompt, worker, CLI
5. **DD Unified Classification Step 14** (modified): Full rebuild with both passes:
   `imas-codex imas dd build --reset-to built --model openrouter/anthropic/claude-sonnet-4.6`
   — This now runs: enrich (Pass 1, ~$50) → refine (Pass 2, ~$50) → embed → cluster
6. **DD Unified Classification Step 15**: MCP tool augmentation
7. **Tests from both plans**

Total rebuild cost: ~$100 (Pass 1 + Pass 2 with sonnet). Time: ~6 hours.

The schema extension (Step 1 here) can be combined with the DD Unified schema changes
(Step 3 there) in a single commit to `imas_dd.yaml`.

---

## Future Extensions

### Selective Refinement

Not all nodes benefit equally from Pass 2. A future optimization: skip refinement for
nodes where:
- No enrichable siblings exist (no family context to add)
- No cluster peers exist (no disambiguation needed)
- No cross-IDS duplicates (no physics consistency to check)

These nodes could advance directly from `enriched` → `refined` (status promotion
without LLM call). The refine worker would sort claimed nodes into "needs LLM" and
"auto-promote" categories.

### Feedback Loop

If SN generation or MCP search reveals that certain descriptions are inadequate, a
targeted re-refinement could be triggered:
1. Set specific nodes back to `enriched` (clear refinement fields)
2. Re-run `--reset-to enriched` for just those paths
3. Re-embed the refined paths

### Triple-Pass for Structure Nodes

Structure nodes (containers) could benefit from a third pass that summarizes their
children's refined descriptions. Currently, structure node descriptions are generated
from child names only. With refined child descriptions available, a summarization pass
could produce much richer container descriptions. This is future work — the value
depends on how structure descriptions are used downstream.

---

## Dependencies

| Dependency | Status | Notes |
|-----------|--------|-------|
| DD Unified Classification (schema + classifier) | Prerequisite | Provides `node_category` for enrichable filtering |
| Sonnet model threading (`--model` CLI flag) | Prerequisite | Must thread through `DDBuildState` |
| ISN | None | No ISN dependency |

## Open Questions

1. **Batch size for refinement**: Pass 1 uses batch size 50. Refinement context is
   richer (sibling descriptions, cluster peers), so prompts are longer. Should we
   reduce to 30? Or does sonnet's large context window handle it fine?

2. **Cluster peer cap**: How many cluster peer descriptions to include? Clusters can
   have 50+ members. Cap at 10–15 most similar (by embedding distance)?

3. **Cross-IDS duplicate cap**: Some names appear in 10+ IDSs (e.g., `time`, `psi`).
   Cap at 5 most relevant (same physics domain)?
