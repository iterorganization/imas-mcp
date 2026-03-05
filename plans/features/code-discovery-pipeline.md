# Code Discovery Pipeline: From Code Evidence to Signal Validation

## Objective

Use the `discover code` CLI to ingest analysis code from fusion facilities, extract data access patterns (MDSplus paths, TDI function calls), link them to TreeNodes and FacilitySignals, and use the code evidence to validate and enrich the signal graph. The approach learns from real code how scientists access data rather than probing signals blindly.

## Architecture

The pipeline uses direct tree-sitter parsing for code chunking, LiteLLM for structured LLM calls, and Cypher UNWIND for graph writes. No framework abstractions — plain functions and async workers following the same supervised worker pattern as wiki and paths discovery.

### Dependencies

| Component | Package | Purpose |
|-----------|---------|---------|
| AST parsing | `tree-sitter` + `tree-sitter-language-pack` | Python, Fortran, MATLAB, Julia, C/C++ chunking |
| IDL/GDL parsing | `tree-sitter-gdl` | IDL `.pro` file chunking via AST boundaries |
| Embeddings | `Encoder` (`imas_codex.embeddings`) | Batch embedding of code chunks |
| LLM scoring | `call_llm_structured()` via LiteLLM | Triage and score workers |
| Graph writes | Direct Cypher `UNWIND` | Node creation, relationship merging |
| Graph reads | `db.index.vector.queryNodes()` | Semantic search over code chunks |
| Remote access | `run_python_script()` / `async_run_python_script()` | File fetching, `rg` enrichment, `fd` scanning |

### Chunking Architecture

```
detect_language(path) → language string
         │
         ▼
language in TEXT_SPLITTER_LANGUAGES?       ("tdi", "text", "markdown", ...)
    ├── YES → chunk_text()                 (sliding window, char-based)
    └── NO  → chunk_code()                 (tree-sitter AST boundaries)
                  │
                  ├── idl/gdl → tree_sitter_gdl parser
                  └── other   → tree_sitter_language_pack parser
                  │
                  └── Exception? → retry with chunk_text()
                                        │
                                        └── Exception? → mark file "failed"
```

**Supported tree-sitter languages:** Python, Fortran, MATLAB, Julia, C, C++, IDL/GDL

**Text-splitter only:** TDI (`.fun`), markdown, RST, HTML, documents

**IDL/GDL parser status:** `tree-sitter-gdl` correctly identifies `procedure_definition` and `function_definition` boundaries (the top-level nodes needed for chunking). 81% of IDL patterns parse without errors. Five error categories remain — struct member assignment, pointer dereference, system variable assignment, arrow method calls, and class method definitions. These don't affect chunking (top-level boundaries are still correct) but will need fixing for future AST-based data access extraction. See `plans/features/tree-sitter-gdl-fixes.md` for details.

### Pipeline Stages

```
discovered → triaged → (enriched) → scored → ingested | skipped | failed | stale
```

| Phase | Worker | Input Status | Output Status | Method |
|-------|--------|-------------|---------------|--------|
| SCAN | scan_worker | FacilityPath (scored) | CodeFile (discovered) | SSH `fd` + `rg` enrichment |
| TRIAGE | triage_worker | discovered | triaged / skipped | Per-dimension LLM scoring (9 dims) |
| ENRICH | enrich_worker | triaged (composite ≥ 0.75) | triaged + is_enriched | `rg` patterns + preview text |
| SCORE | score_worker | triaged + is_enriched | scored / skipped | Full LLM scoring (9 dims + enrichment) |
| CODE | code_worker | scored (interest_score ≥ 0.75) | ingested / failed | tree-sitter chunk + embed |
| LINK | link_worker | ingested | ingested + evidence_linked | DataReference → TreeNode → Signal |

**Workers:** All workers use `SupervisedWorkerGroup` with `PipelinePhase` tracking and orphan recovery via `claimed_at` timestamps.

**Composite formula:** `max(dims) * (1 + mean(nonzero_dims)) / 2`

**Score dimensions:** modeling_code, analysis_code, operations_code, data_access, workflow, visualization, documentation, imas, convention

**Dual-pass scoring:**
- Triage uses `triage_*` properties from all triaged+scored CodeFiles (general population)
- Score uses `score_*` properties from scored-only CodeFiles (graduate cohort that passed triage+enrichment)
- Both inject dynamic calibration examples (60s TTL cache) sampled per dimension at 5 score levels

**Preview text:** Extracted during enrichment (head of file, max 40 lines / 2KB). Used by score worker but NOT persisted to graph.

**Sibling context:** Triage groups files by parent directory and includes sibling filenames for context.

### Key Implementation Files

| File | Purpose |
|------|---------|
| `imas_codex/discovery/code/parallel.py` | Orchestrator: `run_parallel_code_discovery()` |
| `imas_codex/discovery/code/workers.py` | Async workers: scan, triage, enrich, score, code, link |
| `imas_codex/discovery/code/scanner.py` | SSH file enumeration + CodeFile node creation |
| `imas_codex/discovery/code/scorer.py` | Dual-pass LLM scoring with calibration |
| `imas_codex/discovery/code/enrichment.py` | File-level `rg` pattern matching |
| `imas_codex/discovery/code/graph_ops.py` | Claim coordination + graph queries |
| `imas_codex/discovery/code/state.py` | `FileDiscoveryState` shared state |
| `imas_codex/discovery/code/progress.py` | Rich TUI progress display |
| `imas_codex/ingestion/chunkers.py` | `chunk_code()` + `chunk_text()` |
| `imas_codex/ingestion/extractors/mdsplus.py` | MDSplus path extraction (all languages) |
| `imas_codex/ingestion/readers/remote.py` | Language detection, file categories |

### Known Code Access Patterns (from SSH inspection)

| Pattern | Language | Example | Prevalence |
|---------|----------|---------|------------|
| `conn.tdi()` | Python | `conn.tdi(r'\results::psi')` | High — tcvpy core access |
| `tcv.shot().tdi()` | Python | `tcv.shot(shot).tdi('tcv_ip()')` | High — high-level wrapper |
| `tree.getNode()` | Python | `self._MDSTree.getNode(self._root+'::psi')` | Medium — direct MDSplus |
| `tdi()` | MATLAB | `tdi('\results::thomson:te')` | Dominant — 71k+ MATLAB LOC |
| `mdsopen/mdsvalue` | MATLAB | `mdsvalue(['\results::ece_lfs:channel_00' int2str(i)])` | Common |
| `mdsopen/mdsvalue` | IDL | `mdsvalue('\results::i_p')` | Common — TCV IDL codebase |
| `mds$open/mds$value` | IDL | `mds$value('\results::thomson:te')` | Common — CRPP IDL |
| `MDS_OPEN/MDS_GET` | Fortran | `call MDS_OPEN('tcv_shot', shot)` | Low — Fortran codes |
| Channel loop | Python/MATLAB/IDL | `'\results::ece_lfs:channel_' + format(i, '03')` | Medium |

---

## Completed Work

### Phase 0: Pipeline Foundation ✅

All import errors fixed, pipeline validated end-to-end. Workers start, claim, SSH scan, triage, enrich, score, ingest, and link without crashes. Key fixes:
- Import rename `_persist_discovered_files` → `_persist_code_files`
- Tree-sitter timeout guard for large auto-generated files
- `asyncio.to_thread` for `pipeline.run()` to prevent event loop blocking
- Embedding batch size reduction to prevent GPU OOM
- `mdsplus_paths` metadata key mismatch for graph linking
- AT_FACILITY edges on DataReference and DataAccess nodes

### Phase 1: Multi-Language MDSplus Extraction ✅

MATLAB, Python wrapper, Fortran, and IDL patterns added to `imas_codex/ingestion/extractors/mdsplus.py`. All patterns pass unit tests with real TCV code snippets.

### Phase 2: Enrichment Pattern Registry ✅

DATA_ACCESS_PATTERNS expanded with `mdsplus_tdi`, `mdsplus_matlab`, `tcvpy`, `eqtools`, `aug_shotfile`, `uda_client`, `edas_jt60sa`, `tcv_wrappers`, `jet_access`, `jt60sa_access`, and `shot_reference` patterns in `imas_codex/discovery/paths/enrichment.py`.

### LlamaIndex Removal ✅

All LlamaIndex dependencies removed (commits `62f00ae`, `e3cf4aa`, `f224c2e`):
- `chunk_code()` uses direct tree-sitter via `tree-sitter-language-pack` and `tree-sitter-gdl`
- `chunk_text()` uses sliding window with configurable overlap
- Code search uses direct Cypher `db.index.vector.queryNodes()` instead of `VectorStoreIndex`
- Embeddings use `Encoder` from `imas_codex.embeddings`

### IDL/GDL Tree-Sitter Support ✅

`tree-sitter-gdl` integrated for IDL `.pro` file parsing. The parser correctly detects `procedure_definition` and `function_definition` AST boundaries for chunking. Five categories of parse error identified and documented in `plans/features/tree-sitter-gdl-fixes.md` — these don't affect chunking but will need fixing for AST-based extraction.

---

## Phase 3: Full Pipeline Run

### Goal
Run the complete triage→enrich→score→code→link pipeline from a clean state.

### Step 3.1: Clear and Run

```bash
uv run imas-codex discover clear --domain code tcv --force
uv run imas-codex discover code tcv --time 240 -c 20.0 --code-workers 4
```

### Step 3.2: Monitor

```cypher
MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
RETURN cf.status, count(cf) AS n ORDER BY n DESC
```

### Exit Criteria
- All FacilityPaths re-scanned
- CodeFiles triaged with 9-dimension properties
- High-value files enriched, scored, ingested with CodeChunks + embeddings
- DataReference → TreeNode links with AT_FACILITY edges
- FacilitySignals annotated with code evidence

---

## Phase 4: Code-Evidence Signal Linking

### Goal
Use code evidence to validate and prioritize FacilitySignals. The `link_worker` propagates DataReference → TreeNode → FacilitySignal evidence automatically during pipeline runs.

### Step 4.1: Verify Evidence Chain

```cypher
MATCH (dr:DataReference)-[:RESOLVES_TO_TREE_NODE]->(tn:TreeNode)
WHERE tn.facility_id = 'tcv'
RETURN count(dr) AS refs, count(DISTINCT tn.id) AS tree_nodes
```

### Step 4.2: Prioritize Signal Checking by Code Evidence

Signals referenced in real code are high-confidence targets:

```cypher
MATCH (sig:FacilitySignal)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
WHERE sig.has_code_evidence = true AND sig.check_status IS NULL
RETURN sig.id, sig.node_path, sig.code_evidence_count
ORDER BY sig.code_evidence_count DESC
```

### Step 4.3: Extract Known-Good Shot Numbers

Code contains shot numbers used by scientists. Use these for signal validation instead of probing with arbitrary shots:

```cypher
MATCH (cc:CodeChunk)
WHERE cc.text =~ '.*shot\s*=\s*\d{4,6}.*'
RETURN cc.path, cc.text LIMIT 20
```

Known shots from TCV code: 48952, 55608, 55759, 56016, 58499, 60797.

---

## Phase 5: DataAccess Node Enrichment

### Goal
Create DataAccess nodes with language-specific templates drawn from discovered code.

### Step 5.1: Generate Templates from Code Evidence

For each access method × language combination, populate:
- `imports_template`: Required imports
- `connection_template`: How to establish connection
- `data_template`: How to read data
- `time_template`: How to get time bases

### Step 5.2: Link Signals to Access Methods

```cypher
MATCH (sig:FacilitySignal {has_code_evidence: true})-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
MATCH (da:DataAccess)-[:AT_FACILITY]->(f)
WHERE sig.data_access = da.method_type
MERGE (sig)-[:DATA_ACCESS]->(da)
```

---

## Phase 6: Signal Check Routing Fix

### Goal
Fix static tree signal checking and use code evidence for check accuracy.

### Step 6.1: Fix Static Tree Routing

`_resolve_check_tree()` currently routes all `tree_traversal` signals through `tcv_shot`. Static tree signals need `Tree('static', version)` routing.

### Step 6.2: Recheck with Code-Evidence Shots

```bash
uv run imas-codex discover signals tcv --recheck-failed --time 30
```

---

## Phase 7: Continuous Discovery Loop

The pipeline is designed for iterative runs:
1. Discover new code → extract patterns → link to signals
2. Signals with code evidence → higher check priority
3. Code examples → DataAccess templates → better IMAS mapping
4. New FacilityPaths scored → feed back into code discovery

Agent-driven exploration via `search_code()`, `map_signals_to_imas()` MCP tools.

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| CodeFile nodes discovered | > 10,000 | ~13,172 (pre-clear) |
| CodeChunks ingested | > 50,000 | TBD (fresh run) |
| Unique MDSplus paths extracted | > 500 | TBD (fresh run) |
| Signals with code evidence | > 5,000 | TBD (fresh run) |
| Languages covered | Python, MATLAB, Fortran, IDL, C | ✅ |
| Signal check success rate | > 70% | TBD |

## Dependencies

- Embedding server running (for CodeChunk embeddings)
- Neo4j graph running (for all node operations)
- SSH access to facility (for file scanning and `rg` enrichment)
- LLM API access (for file scoring — language model)

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| SSH timeouts on large directories | Time-limited runs, depth=1 scanning, exclude lists |
| LLM cost overruns | Cost limits per run ($5-$20), triage filters 70-80% |
| tree-sitter parse errors (IDL) | Graceful fallback to `chunk_text()` for problematic files |
| Pattern false positives | MDSplus paths validated against TreeNode graph |
| Static tree signals unfixable | Mark as computational (not stored data), skip check |
