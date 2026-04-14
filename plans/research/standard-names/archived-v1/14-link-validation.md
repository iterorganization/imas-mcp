# Feature 14: Link Validation & Graph Mirroring

**Status:** Pending
**Priority:** High — links are what makes the catalog navigable
**Depends on:** Feature 11 Phase 2 (PERSIST_NODES must exist)
**Parallel with:** 12 (scorer — can be developed simultaneously)
**Estimated complexity:** Medium

---

## Problem

The SN pipeline generates documentation with cross-reference mentions
("see also electron_density") and dependency relationships ("derived from
plasma_current"), but these are never validated or written to the graph.
The existing catalog has 810 forward-reference warnings. Names without
validated links have limited navigability and discoverability.

Link validation is fundamentally different from documentation generation:
- **DOCUMENT** (Plan 10) is LLM-based, expensive, generates text
- **LINK** is pure Python, cheap, validates references and writes edges
- **LINK** must run after PERSIST_NODES (Plan 11) because it creates
  graph edges between existing nodes
- **LINK** needs multiple runs because forward references can only be
  resolved when both source and target names exist

## Approach

A separate LINK worker in the build pipeline, plus a standalone `sn link`
CLI command for iterative re-runs. The LINK phase:

1. Reads `cross_reference_mentions` and `dependency_mentions` from
   documented names (output of DOCUMENT phase, persisted in graph)
2. Resolves each mention against existing StandardName nodes in the graph
3. For resolved refs: creates `CROSS_REFERENCES` or `DEPENDS_ON` edges
4. For unresolved refs: stores as `unresolved_refs` node property
5. Updates `ref_in_degree` and `ref_out_degree` counters

### Pipeline Position

```
EXTRACT → COMPOSE → REVIEW → VALIDATE → DOCUMENT → PERSIST_NODES → LINK → SCORE
```

LINK runs after PERSIST_NODES (nodes must exist to create edges) and before
SCORE (score_link_quality depends on link resolution results).

### Relationship Types

Two separate relationship types with different semantics:

| Relationship | Semantics | Example |
|-------------|-----------|---------|
| `CROSS_REFERENCES` | Navigational / "see also" | electron_temperature → ion_temperature |
| `DEPENDS_ON` | Functional / physics dependency | safety_factor → plasma_current |

Both are `(StandardName)-[:REL]->(StandardName)` self-referential edges.
This follows the `WikiPage LINKS_TO WikiPage` pattern in `facility.yaml`.

### Forward Reference Strategy

Forward references (names that reference other names not yet in the graph)
cannot become graph edges because the target node doesn't exist. Strategy:

1. **During `sn build`**: store unresolved mentions as `unresolved_refs`
   node property (string list)
2. **During `sn link`** (standalone re-run): re-check `unresolved_refs`
   against current graph, convert resolved ones to edges
3. **Convergence**: after multiple `sn build` runs generating different
   name batches, `sn link` resolves accumulated forward refs

This matches the user's requirement that LINK "will likely require
multiple runs as we can only link documents when the source and targets
exist."

---

## Phase 1: Link Resolution Engine

### Design

Pure Python module that resolves cross-reference mentions to graph edges.
No LLM calls — this is deterministic string matching against the graph.

### Tasks

1. **Create link resolution module**
   - File: `imas_codex/standard_names/linker.py`
   - Core function: `resolve_references(mentions, known_names) -> LinkResolution`

   ```python
   class LinkResolution(BaseModel):
       """Result of resolving a single name's references."""
       source_id: str
       resolved_cross_refs: list[str]  # Matched to existing StandardName
       resolved_dependencies: list[str]  # Matched to existing StandardName
       unresolved: list[str]  # No match found
       stale: list[str]  # Previously unresolved, now resolvable

   class LinkBatchResult(BaseModel):
       """Result of processing a batch of names."""
       total_mentions: int
       resolved: int
       unresolved: int
       stale_resolved: int  # Previously unresolved, now resolved
       edges_created: int
       circular_refs: list[tuple[str, str]]  # (A→B, B→A) cycles
       resolutions: list[LinkResolution]
   ```

2. **Implement reference classification**
   - File: `imas_codex/standard_names/linker.py`
   - Classify each mention from DOCUMENT phase output:
     - `cross_reference_mentions` → candidate `CROSS_REFERENCES` edges
     - `dependency_mentions` → candidate `DEPENDS_ON` edges
   - Resolution logic:
     1. Load known names: `MATCH (sn:StandardName) RETURN sn.id`
     2. For each mention, check if it exists in known names
     3. Resolved: queue for edge creation
     4. Unresolved: add to `unresolved_refs` property
   - Circular reference detection:
     - Build directed graph of resolved dependencies
     - Detect cycles using DFS
     - Log warnings for cycles (don't block — circular deps happen in physics)

3. **Implement graph edge writing**
   - File: `imas_codex/standard_names/graph_ops.py`
   - Uses explicit Cypher (not `create_nodes()`) because:
     - Multivalued self-referential edges
     - Need `MERGE` for idempotent re-runs
     - Need relationship timestamps
   - Functions (defined in Plan 11, implemented here):
     - `write_cross_references(gc, source_id, target_ids)`
     - `write_depends_on(gc, source_id, target_ids)`
     - `update_degree_counters(gc, name_id)`
     - `clear_resolved_refs(gc, name_id, resolved_ids)`
     - `get_unresolved_names(gc, limit) -> list[dict]` — for re-run mode

   ```python
   def get_unresolved_names(gc: GraphClient, limit: int = 100) -> list[dict]:
       """Get names with unresolved references for re-linking."""
       return list(gc.query("""
           MATCH (sn:StandardName)
           WHERE size(sn.unresolved_refs) > 0
           RETURN sn.id AS id, sn.unresolved_refs AS unresolved
           ORDER BY size(sn.unresolved_refs) DESC
           LIMIT $limit
       """, limit=limit))

   def update_degree_counters(gc: GraphClient, name_ids: list[str]) -> int:
       """Recalculate ref_in_degree and ref_out_degree from actual edges."""
       return gc.query("""
           UNWIND $ids AS name_id
           MATCH (sn:StandardName {id: name_id})
           OPTIONAL MATCH (sn)-[out:CROSS_REFERENCES|DEPENDS_ON]->()
           OPTIONAL MATCH ()-[inc:CROSS_REFERENCES|DEPENDS_ON]->(sn)
           WITH sn, count(DISTINCT out) AS out_deg, count(DISTINCT inc) AS in_deg
           SET sn.ref_out_degree = out_deg, sn.ref_in_degree = in_deg
           RETURN count(sn) AS updated
       """, ids=name_ids)
   ```

### Acceptance Criteria
- `resolve_references()` correctly classifies mentions
- Resolved refs create graph edges
- Unresolved refs stored as node properties
- Circular references detected and logged
- All functions are idempotent for re-runs

---

## Phase 2: LINK Pipeline Worker

### Tasks

1. **Implement `link_worker()`**
   - File: `imas_codex/standard_names/workers.py`
   - Runs after PERSIST_NODES in the pipeline
   - Pattern: claim→process→persist→release (follows discovery workers)
   - Steps:
     1. Load all known StandardName IDs from graph
     2. For each persisted name in current batch:
        a. Read `cross_reference_mentions` and `dependency_mentions`
        b. Resolve against known names
        c. Write resolved edges
        d. Store unresolved as node property
        e. Update degree counters
     3. Report statistics
   - No LLM calls — pure graph operations
   - Fast: should process hundreds of names per second

2. **Add LINK phase to pipeline**
   - File: `imas_codex/standard_names/pipeline.py`
   ```python
   WorkerSpec(
       "link",
       "link_phase",
       link_worker,
       depends_on=["persist_nodes_phase"],
       enabled=not state.skip_link,
   ),
   ```

3. **Add state fields**
   - File: `imas_codex/standard_names/state.py`
   - `link_stats: WorkerStats` — phase tracking
   - `link_phase: PipelinePhase` — supervision
   - `skip_link: bool = False` — CLI control
   - `link_resolution: LinkBatchResult | None` — results

4. **Add progress display**
   - File: `imas_codex/standard_names/progress.py`
   - LINK stage: shows processed count, resolved/unresolved ratio
   - No cost display (pure Python, no LLM)

### Acceptance Criteria
- LINK phase runs after PERSIST_NODES in `sn build`
- Cross-references and dependencies become graph edges
- `--skip-link` flag bypasses the phase

---

## Phase 3: Standalone `sn link` CLI Command

### Design

A standalone CLI command for iterative re-linking. This is the mechanism
for resolving forward references that couldn't be resolved during `sn build`
because the target names didn't exist yet.

### Tasks

1. **Add `sn link` CLI command**
   - File: `imas_codex/cli/sn.py`
   - Modes:
     - `sn link` — re-link all names with unresolved references
     - `sn link --all` — re-link all names (revalidate existing edges)
     - `sn link --name electron_temperature` — re-link specific name
   - Steps:
     1. Query graph for names with `unresolved_refs` (or all names if --all)
     2. Load current known names
     3. For each name: resolve previously-unresolved mentions
     4. Create new edges for newly-resolved refs
     5. Update `unresolved_refs` to remove resolved entries
     6. Update degree counters
   - Report: "Resolved X/Y previously-unresolved references"

2. **Add `sn status` link health section**
   - File: `imas_codex/cli/sn.py`
   - Add to existing `sn status` output:
     ```
     Link Health:
       Total edges:  1,234 (890 CROSS_REFERENCES + 344 DEPENDS_ON)
       Unresolved:   45 names have unresolved references
       Avg in-degree: 3.2
       Avg out-degree: 2.8
       Circular deps: 3 detected
     ```

3. **Add link summary to `sn build` output**
   - File: `imas_codex/cli/sn.py`
   - After LINK phase completes, show summary:
     ```
     Links: 87 resolved, 12 unresolved (run `sn link` to retry)
     ```

### Acceptance Criteria
- `sn link` resolves previously-unresolved references
- Multiple `sn link` runs converge (fewer unresolved each time)
- `sn status` shows link health metrics
- `sn build` summary includes link statistics

---

## Phase 4: Tests

### Tasks

1. **Test link resolution engine**
   - File: `tests/sn/test_linker.py`
   - Test `resolve_references()` with:
     - All references resolved
     - Some unresolved
     - Circular references
     - Empty mentions
     - Mixed cross-references and dependencies
   - Deterministic fixtures, no graph dependency

2. **Test graph edge operations**
   - File: `tests/sn/test_link_graph_ops.py`
   - Test `write_cross_references()` idempotency
   - Test `write_depends_on()` idempotency
   - Test `update_degree_counters()` accuracy
   - Test `get_unresolved_names()` query
   - These require graph fixtures (follow existing test patterns)

3. **Test `sn link` CLI command**
   - File: `tests/sn/test_link_cli.py`
   - Test standalone re-linking mode
   - Test convergence across multiple runs
   - Mock graph operations

4. **Test LINK pipeline phase integration**
   - File: `tests/sn/test_pipeline_link.py`
   - Test LINK runs after PERSIST_NODES
   - Test --skip-link flag
   - Test pipeline with and without LINK enabled

### Acceptance Criteria
- All tests pass
- Link resolution has 100% coverage for classification logic
- Graph operations tested for idempotency
- CLI tested for all modes

---

## Files Modified / Created

| File | Change |
|------|--------|
| `imas_codex/standard_names/linker.py` | NEW: Link resolution engine |
| `imas_codex/standard_names/graph_ops.py` | Add explicit relationship-writing functions |
| `imas_codex/standard_names/workers.py` | Add link_worker() |
| `imas_codex/standard_names/pipeline.py` | Add LINK WorkerSpec |
| `imas_codex/standard_names/state.py` | Add link_stats, link_phase, skip_link |
| `imas_codex/standard_names/progress.py` | Add link stage display |
| `imas_codex/cli/sn.py` | Add `sn link` command, --skip-link flag |
| `tests/sn/test_linker.py` | NEW: resolution engine tests |
| `tests/sn/test_link_graph_ops.py` | NEW: graph operation tests |
| `tests/sn/test_link_cli.py` | NEW: CLI tests |
| `tests/sn/test_pipeline_link.py` | NEW: pipeline integration tests |

## Documentation Updates

- AGENTS.md: Document `sn link` command and iterative re-linking workflow
- AGENTS.md: Document LINK phase position in pipeline
- AGENTS.md: Document CROSS_REFERENCES vs DEPENDS_ON relationship semantics
