# Code Discovery Pipeline: From Code Evidence to Signal Validation

## Objective

Use the `discover code` CLI to ingest analysis code from TCV, extract data access patterns (MDSplus paths, TDI function calls), link them to TreeNodes and FacilitySignals, and use the code evidence to validate and enrich the signal graph. The approach learns from real code how scientists access data rather than probing signals blindly.

## Pipeline Architecture

```
discovered → triaged → (enriched) → scored → ingested | skipped | failed | stale
```

| Phase | Worker | Input Status | Output Status | Method |
|-------|--------|-------------|---------------|--------|
| SCAN | scan_worker | FacilityPath (scored) | CodeFile (discovered) | SSH file enumeration |
| TRIAGE | triage_worker | discovered | triaged / skipped | Per-dimension LLM scoring (9 dims) |
| ENRICH | enrich_worker | triaged (composite ≥ 0.75) | triaged + is_enriched | rg patterns + preview text |
| SCORE | score_worker | triaged + is_enriched | scored / skipped | Full LLM scoring (9 dims + enrichment) |
| CODE | code_worker | scored (interest_score ≥ 0.75) | ingested / failed | tree-sitter chunk, embed |
| LINK | link_worker | ingested | ingested + evidence_linked | DataReference → TreeNode → Signal |

**Composite formula:** `max(dims) * (1 + mean(nonzero_dims)) / 2`

**Score dimensions:** modeling_code, analysis_code, operations_code, data_access, workflow, visualization, documentation, imas, convention

**Calibration:** Triage uses triage_* from general population (triaged+scored+ingested). Score uses score_* from graduate cohort (scored+ingested only). Both prompts include dimension-specific calibration examples.

**Preview text:** Extracted during enrichment (head of file, max 40 lines / 2KB). Used by score worker but NOT persisted to graph.

**Sibling context:** Triage groups files by parent directory and includes sibling filenames for context.

## Graph State (cleared — ready for fresh run)

All code content has been cleared via `discover clear --domain code tcv --force`. The graph now contains only the structural foundation (FacilityPaths, TreeNodes, FacilitySignals) without any code analysis artifacts.

| Node Type | Count | Notes |
|-----------|-------|-------|
| FacilityPath | ~12,237 | scored, scan markers reset (files_scanned=null) |
| CodeFile | 0 | cleared |
| CodeChunk | 0 | cleared |
| CodeExample | 0 | cleared |
| DataReference | 0 | cleared |
| FacilitySignal | ~64,950 | code_evidence_count/has_code_evidence reset to null |
| TreeNode | ~82,717 | MDSplus tree nodes (unchanged) |

### Referential Integrity Fix (commit 6e27907)

The previous graph had DataReference and DataAccess nodes missing `AT_FACILITY` edges, causing `test_facility_id_edges_exist` to fail. Root causes fixed:

- **DataReference** (ingestion/graph.py): `link_chunks_to_tree_nodes()` and `link_example_chunks_to_tree_nodes()` now create AT_FACILITY edges when merging DataReference nodes.
- **DataAccess** (discovery/signals/tdi.py): `create_tdi_data_access()` switched from `gc.create_node()` (which skips relationship creation) to explicit Cypher with AT_FACILITY edge.
- **clear_facility_code** (scanner.py): Now deletes orphaned DataReference nodes, resets FacilityPath scan markers, and clears FacilitySignal code evidence properties for a complete clean slate.

With the cleared state and source fixes, all newly created nodes will have proper AT_FACILITY edges.

## Known Code Access Patterns (from SSH inspection)

| Pattern | Language | Example | Prevalence |
|---------|----------|---------|------------|
| `conn.tdi()` | Python | `conn.tdi(r'\results::psi')` | High — tcvpy core access |
| `tcv.shot().tdi()` | Python | `tcv.shot(shot).tdi('tcv_ip()')` | High — high-level wrapper |
| `tree.getNode()` | Python | `self._MDSTree.getNode(self._root+'::psi')` | Medium — direct MDSplus |
| `tdi()` | MATLAB | `tdi('\results::thomson:te')` | Dominant — 71k+ MATLAB LOC |
| `mdsopen()` | MATLAB | `mdsopen(pulno)` | Common — MATLAB tree open |
| `mdsvalue()` | MATLAB | `mdsvalue(['\results::ece_lfs:channel_00' int2str(i)])` | Common — MATLAB read |
| `TdiExecute()` | Python | `TdiExecute("tcv_eq('PSI')")` | Low — raw MDSplus API |
| Parameterized | Python | `conn.tdi(r'\results::psi[*,*,$1]', time)` | Medium — sliced access |
| Channel loop | Python/MATLAB | `'\results::ece_lfs:channel_' + format(i, '03')` | Medium — diagnostic arrays |
| Shot-dependent | MATLAB | `if shot > 50237: base = r'\results::ece_lfs'` | Low — version branching |

**Unique references found**: 786 unique MDSplus paths, 18 unique TDI function calls across all inspected code.

---

## Phase 0: Fix the `discover code` CLI ✅ COMPLETE

All import errors fixed, pipeline validated end-to-end. Workers start, claim, SSH scan, triage, enrich, score, ingest, and link without crashes. 77 tests pass. Pipeline ran for 30+ min across multiple sessions.

Key bugs fixed:
- Import rename `_persist_discovered_files` → `_persist_code_files`
- Tree-sitter timeout guard for large auto-generated files
- `asyncio.to_thread` for `pipeline.run()` to prevent event loop blocking
- Embedding batch size reduction to prevent GPU OOM
- `mdsplus_paths` metadata key mismatch for graph linking

---

## Phase 1: Extend MDSplusExtractor with Multi-Language Patterns ✅ COMPLETE

MATLAB (`tdi()`, `mdsvalue()`, `mdsopen()`), Python wrappers (`conn.tdi()`, `tcv.shot().tdi()`), Fortran (`MDS_OPEN`, `MDS_GET`), and IDL (`mdsvalue`) patterns added to `imas_codex/ingestion/extractors/mdsplus.py`. All patterns pass unit tests with real TCV code snippets. Existing tests unaffected.

```python
# Fortran MDS_OPEN/MDS_GET: call MDS_OPEN('tcv_shot', shot)
r"MDS_(?:OPEN|GET|VALUE)\s*\(\s*['\"]\\\\?([^'\"]+)['\"]",

# IDL mdsvalue: mdsvalue, '\results::ip'
r"mdsvalue\s*,\s*['\"]\\\\?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*::[A-Za-z0-9_:\.]+)['\"]",
```

### Step 1.4: Tests

Write targeted unit tests with real code snippets from TCV:

```python
def test_extract_matlab_tdi():
    text = "data = tdi('\\results::thomson:te');"
    refs = extract_mdsplus_paths(text)
    assert any(r.path == "\\RESULTS::THOMSON:TE" for r in refs)

def test_extract_matlab_mdsvalue():
    text = "ip = mdsvalue('\\results::i_p');"
    refs = extract_mdsplus_paths(text)
    assert any(r.path == "\\RESULTS::I_P" for r in refs)

def test_extract_conn_tdi():
    text = "data = conn.tdi(r'\\results::psi')"
    refs = extract_mdsplus_paths(text)
    assert any(r.path == "\\RESULTS::PSI" for r in refs)
```

### Exit Criteria
- All new patterns pass unit tests with real TCV code snippets
- Existing tests still pass (no regression)
- Re-run ingestion on already-discovered files extracts significantly more references

---

## Phase 2: Expand Enrichment Pattern Registry ✅ COMPLETE

DATA_ACCESS_PATTERNS expanded with `mdsplus_tdi`, `mdsplus_matlab`, `tcvpy`, `eqtools`, `aug_shotfile`, `uda_client`, `edas_jt60sa`, `tcv_wrappers`, `jet_access`, `jt60sa_access`, and `shot_reference` patterns. All added to PATTERN_REGISTRY in `imas_codex/discovery/paths/enrichment.py`.

---

## Phase 3: Run Code Discovery from Clean State

### Goal
Run the full triage→enrich→score→code→link pipeline from scratch. All previous code data has been cleared via `discover clear --domain code tcv --force`, so the pipeline starts fresh without legacy artifacts from the old single-pass model.

### Step 3.0: Clear Code Content (DONE)

```bash
uv run imas-codex discover clear --domain code tcv --force
```

This deletes all CodeFile, CodeChunk, CodeExample, and DataReference nodes for TCV, resets FacilityPath scan markers (`files_scanned`, `last_file_scan_at`, `evidence_linked` → null), and clears FacilitySignal code evidence properties.

### Step 3.1: Full Pipeline Run (scan → triage → enrich → score → code → link)

Since all data is cleared, we need the scan worker to re-enumerate files from FacilityPaths:

```bash
uv run imas-codex discover code tcv --time 240 -c 20.0 --code-workers 4
```

**Pipeline flow:**
1. `scan_worker` reads scored FacilityPaths, SSH enumerates files → creates CodeFile (discovered)
2. `triage_worker` claims discovered files → per-dimension LLM scoring → status triaged/skipped
3. `enrich_worker` claims triaged files (composite ≥ 0.75) → rg patterns + preview text → is_enriched=true
4. `score_worker` claims enriched files → full 9-dimension LLM scoring → status scored/skipped
5. `code_worker` claims scored files (interest_score ≥ 0.75) → tree-sitter chunk + embed → status ingested/failed
6. `link_worker` claims ingested files → DataReference → TreeNode → FacilitySignal evidence propagation

### Step 3.2: Monitor Progress

```cypher
MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
RETURN cf.status, count(cf) AS n
ORDER BY n DESC
```

```cypher
// Verify dimension properties populated
MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
WHERE cf.triage_composite IS NOT NULL
RETURN count(cf) AS triaged,
       avg(cf.triage_composite) AS avg_composite,
       avg(cf.triage_data_access) AS avg_data_access,
       avg(cf.triage_imas) AS avg_imas
```

### Step 3.3: Verify Referential Integrity

After ingestion, confirm the AT_FACILITY fix is working:

```bash
uv run pytest tests/graph/test_referential_integrity.py::test_facility_id_edges_exist -v
```

### Exit Criteria
- All FacilityPaths re-scanned (files_scanned populated)
- All discovered CodeFiles triaged with 9-dimension properties
- High-value files enriched and scored
- Scored files ingested with CodeChunks + embeddings
- DataReference nodes have AT_FACILITY edges (integrity test passes)
- FacilitySignals annotated with code evidence from fresh ingestion

---

## Phase 4: Code-Evidence Signal Linking ✅ PARTIALLY COMPLETE

`link_worker` in `workers.py` and `link_code_evidence_to_signals()` in `graph_ops.py` are fully implemented and operational. The automated link worker runs as part of the standard pipeline and propagates evidence from DataReference → TreeNode → FacilitySignal.

Current state: 248 FacilitySignals have code evidence from 158 DataReferences.

### Remaining work

The ingestion pipeline's `link_chunks_to_tree_nodes()` already creates DataReference → TreeNode relationships. Verify these exist:

```cypher
MATCH (dr:DataReference)-[:RESOLVES_TO_TREE_NODE]->(tn:TreeNode)
WHERE tn.facility_id = 'tcv'
RETURN count(dr) AS refs_with_tree_node,
       count(DISTINCT tn.id) AS unique_tree_nodes
```

### Step 4.2: Propagate Code Evidence to FacilitySignals

TreeNodes link to FacilitySignals via node_path matching. Create a Cypher query that propagates code evidence:

```cypher
// Find signals that have code evidence via TreeNode matching
MATCH (dr:DataReference)-[:RESOLVES_TO_TREE_NODE]->(tn:TreeNode)
WHERE tn.facility_id = 'tcv'
WITH tn, count(dr) AS code_ref_count, collect(DISTINCT dr.source_file) AS source_files

MATCH (sig:FacilitySignal)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
WHERE sig.node_path = tn.node_path
  OR sig.node_path = tn.path
SET sig.code_evidence_count = code_ref_count,
    sig.code_evidence_files = source_files[..5],
    sig.has_code_evidence = true
RETURN count(sig) AS signals_with_code_evidence
```

### Step 4.3: Code-Evidence Based Signal Checking

Signals with code evidence are high-confidence — someone wrote code to access them. Prioritize these for checking:

```cypher
// Signals with code evidence but not yet checked
MATCH (sig:FacilitySignal)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
WHERE sig.has_code_evidence = true
  AND sig.check_status IS NULL
RETURN sig.id, sig.node_path, sig.code_evidence_count
ORDER BY sig.code_evidence_count DESC
```

### Step 4.4: Extract Shot Numbers from Code

Code often contains known-good shot numbers. Extract and store these:

```cypher
// Find code with shot references
MATCH (cc:CodeChunk)
WHERE cc.text =~ '.*shot\s*=\s*\d{4,6}.*'
RETURN cc.path, cc.text
LIMIT 20
```

Known shots from TCV code inspection: 48952, 55608, 55759, 56016, 58499, 60797. These can be used as `check_shots` for signal validation rather than probing with arbitrary recent shots.

### Exit Criteria
- DataReference → TreeNode links verified
- FacilitySignals annotated with code evidence counts
- Priority signal checking uses code-evidence ordering
- Known-good shot numbers extracted from code for validation

---

## Phase 5: Enrich DataAccess Nodes

### Goal
Create comprehensive DataAccess nodes that document all access methods, with templates and examples drawn from discovered code.

### Step 5.1: Populate DataAccess Templates from Code Evidence

Use code chunks to generate access templates:

```cypher
// Find code examples for each access pattern
MATCH (cc:CodeChunk)
WHERE cc.mdsplus_ref_count > 0
RETURN cc.language, cc.text, cc.path
ORDER BY cc.mdsplus_ref_count DESC
LIMIT 50
```

### Step 5.2: Create Language-Specific DataAccess Nodes

For each access method×language combination, create DataAccess nodes with:
- `imports_template`: Required imports
- `connection_template`: How to establish connection
- `data_template`: How to read data
- `time_template`: How to get time bases
- `cleanup_template`: How to close/cleanup

### Step 5.3: Link Signals to Access Methods

```cypher
// Connect signals to their DataAccess method based on code evidence
MATCH (sig:FacilitySignal)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
WHERE sig.has_code_evidence = true
WITH sig
MATCH (da:DataAccess)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
WHERE sig.data_access = da.method_type
MERGE (sig)-[:DATA_ACCESS]->(da)
```

### Exit Criteria
- DataAccess nodes cover Python (tcvpy, direct MDSplus) and MATLAB access
- Templates are populated with real code examples
- FacilitySignals linked to their access methods

---

## Phase 6: Fix Signal Check Routing

### Goal
Fix the static tree signal checking bug and use code evidence to improve check accuracy.

### Step 6.1: Fix Static Tree Routing

The check_worker routes all `tree_traversal` signals through `tcv_shot`, but static tree signals must be checked with `Tree('static', version)` not `Tree('tcv_shot', shot)`.

Fix `_resolve_check_tree()` to detect independent trees (static, vsystem, etc.) and route them correctly:
- Static: `Tree('static', 1)` or `Tree('static', version)` — version-based, not shot-based
- Other independent trees: Use tree name directly with appropriate versioning

### Step 6.2: Use Code-Evidence Shots for Checking

Instead of probing with arbitrary recent shots, use shot numbers found in code:

```cypher
// Get code-evidenced signals with their known shots
MATCH (sig:FacilitySignal {has_code_evidence: true})-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
WHERE sig.check_status IS NULL OR sig.check_status = 'fail'
WITH sig
MATCH (cc:CodeChunk)-[:CONTAINS_REF]->(dr:DataReference)-[:RESOLVES_TO_TREE_NODE]->(tn:TreeNode)
WHERE tn.node_path = sig.node_path
  AND cc.text =~ '.*shot\s*=\s*(\d{4,6}).*'
RETURN sig.id, sig.node_path, collect(DISTINCT cc.path) AS code_files
```

### Step 6.3: Recheck Failed Signals with Correct Routing

After fixing routing:
```bash
uv run imas-codex discover signals tcv --recheck-failed --time 30
```

### Exit Criteria
- Static tree signals checked correctly (or marked appropriately)
- Code-evidence signals validated with known-good shots
- Check success rate improves significantly from current 50%

---

## Phase 7: Agent-Driven Iterative Exploration

### Goal
Use agent runs to discover additional data access patterns and populate DataAccess templates for less common access methods.

### Step 7.1: Agent Code Search

Use `search_code` MCP tool to find additional access patterns:
```python
search_code("reading MDSplus data", facility="tcv")
search_code("shot data access pattern", facility="tcv")
search_code("tdi function call", facility="tcv")
```

### Step 7.2: Agent Signal Mapping

Use `map_signals_to_imas` to map code-evidenced signals to IMAS:
```python
map_signals_to_imas(facility='tcv', physics_domain='magnetics')
map_signals_to_imas(facility='tcv', physics_domain='equilibrium')
```

### Step 7.3: Continuous Discovery Loop

The code discovery pipeline is designed for iterative runs:
1. Discover new code → extract patterns → link to signals
2. Signals with code evidence → higher check priority
3. Code examples → DataAccess templates → better IMAS mapping
4. New FacilityPaths scored → feed back into code discovery

### Exit Criteria
- All major TCV access patterns documented with DataAccess nodes
- Signal→IMAS mapping enriched with code evidence
- DataAccess templates usable for generating access code

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| CodeFile nodes discovered | > 10,000 | ✅ ~13,172 |
| CodeChunks ingested | > 50,000 | ~12,865 (partial) |
| Unique MDSplus paths extracted | > 500 | 158 (partial) |
| Signals with code evidence | > 5,000 | 248 (partial) |
| Languages covered | Python, MATLAB, Fortran, IDL | ✅ Python, Fortran, MATLAB, C, IDL |
| Signal check success rate | > 70% (up from 50%) | TBD |
| DataAccess templates | Cover all major access methods | TBD |
| All files triaged (new pipeline) | 100% of discovered | 0% (re-triage needed) |

## Dependencies

- Embedding server running (for CodeChunk embeddings)
- Neo4j graph running (for all node operations)
- SSH access to TCV (for file scanning and `rg` enrichment)
- LLM API access (for file scoring — language model)

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| SSH timeouts on large directories | Time-limited runs, depth=1 scanning, exclude lists |
| LLM cost overruns | Cost limits per run ($5-$20), triage pass filters 70-80% |
| MATLAB tree-sitter parsing issues | Fallback to SentenceSplitter for problematic files |
| Pattern false positives | MDSplus paths validated against TreeNode graph |
| Static tree signals unfixable | Mark as computational (not stored data), skip check |
