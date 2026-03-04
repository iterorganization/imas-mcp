# Signals Pipeline Completion

> Complete the MDSplus unification deferred items, document the signals CLI
> with example graph queries, and validate the full pipeline end-to-end
> against the live TCV facility with schema compliance checks.

## Status: Done

## Context

The MDSplus unification plan (`mdsplus-unification.md`) is implemented with
three deferred items remaining:

1. `mdsplus/enrichment.py` — still consumed by `discovery/static/workers.py`
   and `agentic/prompt_loader.py`
2. `remote/scripts/extract_tdi_functions.py` — still consumed by
   `discovery/signals/tdi.py`
3. `TreeNode.is_static` property — used in 30+ Cypher queries across
   `mdsplus/extraction.py` and `discovery/mdsplus/graph_ops.py`

Additionally, three pre-existing schema compliance failures exist:
- `SoftwareRepo`: undeclared property `vcs_type`
- `DataAccess.connection_template`: 1 node with null required field
- `FacilityPath.path_purpose`: invalid enum value `software_project`

---

## Phase 1: Delete `mdsplus/enrichment.py` ✅

**Goal:** Absorb Pydantic models and prompt builders into the signals
enrichment pipeline; delete the standalone module.

The two consumers are:
- `discovery/static/workers.py` → `enrich_worker` uses
  `StaticNodeBatch`, `_build_system_prompt`, `_build_user_prompt`
- `agentic/prompt_loader.py` → `_provide_static_enrichment_schema` imports
  `StaticNodeBatch`, `StaticNodeResult`

Steps:
1. Move `StaticNodeBatch` / `StaticNodeResult` models into
   `discovery/mdsplus/` (they describe tree node enrichment, not signals)
2. Move `_build_system_prompt` / `_build_user_prompt` alongside the models
3. Update `discovery/static/workers.py` import to new location
4. Update `prompt_loader.py` schema provider to new location
5. Delete `mdsplus/enrichment.py`
6. Run tests, commit

## Phase 2: Retire `extract_tdi_functions.py` ✅

**Goal:** Make `discovery/signals/tdi.py` use the newer
`extract_tdi_signals.py` script (which already exists and is more capable),
removing the last consumer of the old script.

Steps:
1. Audit the difference between `extract_tdi_functions.py` and
   `extract_tdi_signals.py` — the latter already does runtime probing
2. Update `discovery/signals/tdi.py` to call `extract_tdi_signals.py`
   instead, adapting the response parsing
3. Update `discovery/signals/parallel.py` references
4. Delete `remote/scripts/extract_tdi_functions.py`
5. Run tests, commit

## Phase 3: Remove `TreeNode.is_static` ✅

**Goal:** Replace all `is_static` filters with tree-name or version-based
queries that work identically for all tree types.

The `is_static` flag was a type discriminator when static and dynamic trees
had separate pipelines. Now that all trees flow through the same extraction
pipeline, a node's tree name and version linkage carry the same information.

Steps:
1. In `mdsplus/extraction.py`: replace `n.is_static = true` filters with
   tree-name membership checks (the versioned trees are known from config)
2. In `discovery/mdsplus/graph_ops.py`: replace all 17 `is_static` filters
   with equivalent `tree_name`-based or `introduced_version IS NOT NULL`
   conditions
3. Stop setting `is_static` on new nodes in extraction
4. Remove `is_static` from the LinkML schema (`facility.yaml`)
5. Rebuild models: `uv run build-models --force`
6. Write a one-time migration query to drop the `is_static` property from
   existing graph nodes
7. Run tests, commit

## Phase 4: Fix Schema Compliance Failures ✅

**Goal:** Fix the three pre-existing schema compliance test failures.

1. **`SoftwareRepo.vcs_type`**: Add `vcs_type` to the `SoftwareRepo` class
   in `imas_codex/schemas/facility.yaml`
2. **`DataAccess.connection_template` null**: Either make the field optional
   in the schema, or backfill the null node in the graph
3. **`FacilityPath.path_purpose` invalid enum `software_project`**: Add
   `software_project` to the `ResourcePurpose` enum in the schema, or
   reclassify the node to a valid enum value
4. Rebuild models, run `tests/graph/test_schema_compliance.py` until green,
   commit

## Phase 5: Signals CLI Documentation ✅

**Goal:** Create comprehensive documentation for the signals discovery
pipeline at `docs/architecture/signals.md`.

The document should cover:

### CLI Reference
- `imas-codex discover signals <facility>` — full option reference
- `imas-codex discover clear <facility> -d signals` — clearing data
- `imas-codex discover status <facility> -d signals` — checking progress
- Scanner selection (`--scanners mdsplus,tdi`)
- Scan-only vs enrich-only modes
- Signal limits, cost limits, time limits

### Pipeline Architecture
- Scanner plugin system (mdsplus, tdi, ppf, edas)
- Scan → Enrich → Check flow
- Tree extraction: versioned vs epoched vs shot-scoped
- TDI linkage: TreeNode ← RESOLVES_TO ← TDIFunction
- Enrichment context injection (wiki, code, tree hierarchy, TDI source)

### Graph Schema
- Node types: FacilitySignal, TreeNode, TreeModelVersion, TreeNodePattern,
  TDIFunction, DataAccess, Diagnostic, MDSplusTree
- Key relationships: SOURCE_NODE, DATA_ACCESS, AT_FACILITY,
  BELONGS_TO_DIAGNOSTIC, MAPS_TO_IMAS, RESOLVES_TO_TREE_NODE,
  INTRODUCED_IN, REMOVED_IN, HAS_UNIT, HAS_PATTERN

### Example Cypher Queries

Include working queries for each of these use cases:

1. **Semantic search** — find signals by physics meaning using vector
   embeddings (e.g., "electron density profile")
2. **Path access** — navigate from a tree path to the signal, its unit,
   its diagnostic, and its IMAS mapping
3. **Preferential accessor selection** — given a signal with both a raw
   tree path and a TDI function accessor, query both and show which is
   preferred
4. **Data access pattern resolution** — from a DataAccess node, find all
   signals and their access templates
5. **Epoch-aware queries** — find signals that exist at a given shot number
   via TreeModelVersion applicability ranges
6. **Cross-domain traversal** — from a signal, traverse to related wiki
   documentation, code chunks, and IMAS paths
7. **Diagnostic inventory** — list all diagnostics and their signal counts
8. **TDI function resolution** — from a TDI function, follow RESOLVES_TO
   edges to the underlying TreeNodes and their signals

## Phase 6: E2E Testing ✅

**Goal:** Validate the full signals pipeline against the live TCV facility
by processing real data through every stage and verifying graph output.

### Test Structure

Create `tests/integration/test_signals_e2e.py`. Each test function follows
the same pattern:

1. Clear: `imas-codex discover clear tcv -d signals --force`
2. Run the pipeline for the domain under test with `--signal-limit 50`
3. Query the graph to validate node counts, relationships, and properties
4. Assert schema compliance

### Test Cases

#### E2E-1: Static Tree (versioned)
```
imas-codex discover signals tcv -s mdsplus --scan-only -n 50
```
Validate:
- TreeModelVersion nodes created for configured versions
- TreeNode nodes created with correct tree_name, facility_id, path
- FacilitySignal nodes promoted from leaf TreeNodes
- SOURCE_NODE edges from FacilitySignal → TreeNode
- DataAccess node created with correct method_type
- Unit nodes created from SSH extraction

#### E2E-2: Dynamic Tree (results subtree)
```
imas-codex discover signals tcv -s mdsplus --scan-only -n 50
```
Validate:
- TreeNode nodes created for `results` subtree
- FacilitySignal nodes promoted with correct tree_name
- node_usages filtering applied (only NUMERIC/SIGNAL)

#### E2E-3: TDI Functions
```
imas-codex discover signals tcv -s tdi --scan-only -n 50
```
Validate:
- TDIFunction nodes created with source_code, quantities, build_paths
- FacilitySignal nodes created from TDI quantities
- DataAccess node for TDI access pattern

#### E2E-4: TDI Linkage
Run after E2E-1 and E2E-3 (don't clear between):
Validate:
- RESOLVES_TO_TREE_NODE edges exist between TDIFunction and TreeNode
- FacilitySignal.preferred_accessor populated for linked signals

#### E2E-5: Enrichment
```
imas-codex discover signals tcv --enrich-only -n 50 -c 2.0
```
Run after scan-only tests (don't clear). Validate:
- FacilitySignal nodes transition from `discovered` → `enriched`
- description, physics_domain populated by LLM
- tree context injected (verify via enrichment log)

#### E2E-6: Full Pipeline
Clear and run the full pipeline end-to-end:
```
imas-codex discover signals tcv -n 50 -c 2.0
```
Validate all of the above in a single run.

### Graph Validation Queries

After each E2E test, run validation queries:
- All FacilitySignal nodes have non-null `facility_id`, `status`, `accessor`
- All TreeNode nodes have non-null `path`, `tree_name`, `facility_id`
- All SOURCE_NODE edges connect FacilitySignal → TreeNode
- All DATA_ACCESS edges connect FacilitySignal → DataAccess
- All AT_FACILITY edges connect to the correct Facility node
- No orphaned TreeNode nodes (every leaf has a FacilitySignal or is STRUCTURE)
- Signal IDs follow the expected format: `{facility}:{tree}/{path}`

### Schema Compliance

Run `tests/graph/test_schema_compliance.py` after each E2E test to verify
the generated data is schema-compliant. All 9 compliance tests must pass.

## Phase 7: Fix All Errors ✅

**Goal:** Iterative bug-fixing phase. Run each E2E test, fix failures,
re-clear and re-run until all 50 items process to completion. Then run the
full graph unit test suite and fix any failures, including those outside
the immediate scope of these changes.

This phase has no pre-defined steps — it is driven by test output. The
agent should:
1. Run E2E tests sequentially (E2E-1 through E2E-6)
2. On failure: diagnose, fix, re-clear, re-run from the beginning of that
   test
3. After all E2E tests pass: run `uv run pytest tests/graph/` and fix any
   failures
4. After graph tests pass: run `uv run pytest tests/` full suite and fix
   any remaining failures
5. Final commit with all fixes
