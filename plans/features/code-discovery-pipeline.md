# Code Discovery Pipeline: From Code Evidence to Signal Validation

## Objective

Use the `discover code` CLI to ingest analysis code from TCV, extract data access patterns (MDSplus paths, TDI function calls), link them to TreeNodes and FacilitySignals, and use the code evidence to validate and enrich the signal graph. The approach learns from real code how scientists access data rather than probing signals blindly.

## Current State

### Graph State (as of investigation)

| Node Type | Count | Notes |
|-----------|-------|-------|
| FacilityPath | 12,237 | score ≥ 0.7, none file-scanned |
| CodeFile | 0 | Pipeline never run |
| CodeChunk | 0 | No code ingested |
| DataReference | 0 | No references extracted |
| FacilitySignal | 64,950 | 14,746 check_success, 14,747 check_fail, 25,464 discovered |
| TreeNode | 82,717 | MDSplus tree nodes enumerated |
| DataAccess | 4 | tcv:mdsplus:tree_tdi, tcv:mdsplus:static, tcv:mdsplus:vsystem, tcv:tdi:functions |

### Known Code Access Patterns from SSH Inspection

Inspected 4 TCV codebases (tcvpy, eqtools, MATLAB analysis, liuqe) via SSH:

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

### Root Cause: `discover code` CLI Failure

The CLI crashes immediately on startup. All scan workers crash in a loop (10 restart attempts each) with:

```
ImportError: cannot import name '_persist_discovered_files' from 'imas_codex.discovery.code.scanner'
```

**Root cause**: `workers.py` imports `_persist_discovered_files` (lines 47, 112) but the function was renamed to `_persist_code_files` in `scanner.py` (line 285). The import name was not updated after the rename.

---

## Phase 0: Fix the `discover code` CLI

### Goal
Make the `discover code` CLI fully operational through iterative E2E testing, graph inspection, and bug fixes.

### Step 0.1: Fix the Import Error

Fix the broken import in `workers.py`:
- `_persist_discovered_files` → `_persist_code_files` (2 references: import line 47, usage line 112)

### Step 0.2: Smoke Test — Scan Only (2 min time limit)

```bash
uv run imas-codex discover code tcv --scan-only --time 2 --max-paths 10
```

**Success criteria**: Workers start, claim FacilityPaths, SSH to TCV, enumerate files, create CodeFile nodes in graph.

**Graph verification after**:
```cypher
MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
RETURN cf.status, cf.language, count(cf) AS n
ORDER BY n DESC
```

Expected: CodeFile nodes with status='discovered', languages including python, fortran, matlab.

### Step 0.3: Score Test — Small Batch (5 min, $2 cost limit)

```bash
uv run imas-codex discover code tcv --score-only --time 5 -c 2.0
```

**Success criteria**: score_worker claims CodeFiles, runs triage (keep/skip), scores kept files with multi-dimensional LLM scoring.

**Graph verification after**:
```cypher
MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
WHERE cf.interest_score IS NOT NULL
RETURN cf.interest_score, cf.file_category, cf.path
ORDER BY cf.interest_score DESC
LIMIT 20
```

Expected: Files scored 0.0-1.0 with categories like 'analysis', 'data_access', 'modeling'.

### Step 0.4: Ingestion Test — Full Pipeline (10 min, $3 cost limit)

```bash
uv run imas-codex discover code tcv --time 10 -c 3.0 --max-paths 20
```

**Success criteria**: All 4 workers run (scan → triage → score → ingest). CodeChunk nodes created with embeddings. MDSplusExtractor extracts path references.

**Graph verification after**:
```cypher
// Check CodeChunks
MATCH (cc:CodeChunk)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
RETURN count(cc) AS chunks, 
       count(CASE WHEN cc.embedding IS NOT NULL THEN 1 END) AS embedded

// Check extracted references
MATCH (cc:CodeChunk)
WHERE cc.mdsplus_ref_count > 0
RETURN cc.path, cc.mdsplus_ref_count
ORDER BY cc.mdsplus_ref_count DESC
LIMIT 10
```

### Step 0.5: Enrichment Verification

```bash
# Verify rg enrichment ran
```

```cypher
MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
WHERE cf.is_enriched = true
RETURN cf.pattern_categories, cf.total_pattern_matches, cf.path
ORDER BY cf.total_pattern_matches DESC
LIMIT 20
```

### Step 0.6: Fix and Iterate

After each step, inspect logs and graph state:
- Read `~/.local/share/imas-codex/logs/code_tcv.log` for errors
- Run graph queries to verify node creation and relationships
- Fix any issues found before proceeding to next step
- Repeat failed steps after fixes

### Exit Criteria
- All 4 workers complete without crashes
- CodeFile nodes created with scores and enrichment
- CodeChunk nodes created with embeddings
- MDSplus references extracted from Python code chunks
- Pipeline can run for 30+ min without errors

---

## Phase 1: Extend MDSplusExtractor with Multi-Language Patterns

### Goal
Extend `imas_codex/ingestion/extractors/mdsplus.py` to capture data access patterns from MATLAB (dominant at TCV: 71k+ LOC), Fortran, IDL, and more Python access patterns.

### Step 1.1: Add MATLAB MDSplus Patterns

MATLAB is the dominant language at TCV but currently has zero pattern coverage in MDSplusExtractor.

**New patterns to add to `MDSPLUS_PATH_PATTERNS`**:

```python
# MATLAB tdi() calls: tdi('\results::thomson:te')
r"tdi\s*\(\s*['\"]\\\\?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*::[A-Za-z0-9_:\.]+)['\"]",

# MATLAB mdsvalue() calls: mdsvalue('\results::ip')
r"mdsvalue\s*\(\s*['\"]\\\\?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*::[A-Za-z0-9_:\.]+)['\"]",

# MATLAB mdsvalue with concatenation: mdsvalue(['\results::ece_lfs:channel_00' int2str(i)])
r"mdsvalue\s*\(\s*\[\s*['\"]\\\\?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*::[A-Za-z0-9_:\.]+)",

# MATLAB mdsopen: mdsopen('tcv_shot', pulno) — captures tree name
r"mdsopen\s*\(\s*['\"]([A-Za-z_][A-Za-z0-9_]+)['\"]",
```

**New patterns for `TDI_FUNCTION_PATTERNS`**:

```python
# MATLAB-style TDI function calls: tdi('tcv_ip()'), tdi('tcv_eq("PSI")')
r"tdi\s*\(\s*['\"](\w+)\s*\(",

# mdsvalue with TDI function: mdsvalue('tcv_eq("PSI")')
r"mdsvalue\s*\(\s*['\"](\w+)\s*\(",
```

### Step 1.2: Add Python Wrapper Patterns

Current patterns miss common Python access patterns seen at TCV:

```python
# tcvpy connection.tdi: conn.tdi(r'\results::psi')
r"\.tdi\s*\(\s*r?['\"]\\\\?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*::[A-Za-z0-9_:\.]+)['\"]",

# tcvpy shot access: tcv.shot(shot).tdi('tcv_ip()')
r"\.tdi\s*\(\s*['\"](\w+)\s*\(",
```

### Step 1.3: Add Fortran/IDL Patterns

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

## Phase 2: Expand Enrichment Pattern Registry

### Goal
Add data access patterns to `PATTERN_REGISTRY` in `imas_codex/discovery/paths/enrichment.py` to maximize detection of code that accesses fusion data. These patterns are used for both directory-level and file-level `rg` enrichment.

### Step 2.1: Extend DATA_ACCESS_PATTERNS

Add patterns for access methods seen in real TCV code but not currently matched:

```python
DATA_ACCESS_PATTERNS = {
    # Existing patterns (keep all):
    "mdsplus": r"(mdsconnect|mdsopen|mdsvalue|MDSplus|TdiExecute|connection\.openTree|TreeNode)",
    "ppf": ...,
    
    # NEW patterns to add:
    "mdsplus_tdi": r"(\.tdi\(|conn\.tdi|connection\.tdi|tdi\s*\()",
    "mdsplus_matlab": r"(mdsopen|mdsclose|mdsvalue|mdsdisconnect|mdsconnect)",
    "tcvpy": r"(tcvpy|tcv\.shot|from\s+tcvpy|import\s+tcv)",
    "eqtools": r"(eqtools|EqdskReader|CModTCVMachine|TRIPPYMachine)",
    "aug_shotfile": r"(dd\.open|sf\.getSignal|sf\.getParameter|AUGshotfile)",
    "uda_client": r"(pyuda\.Client|UdaClient|uda\.get|labcom\.read|labcom_read)",
    "edas_jt60sa": r"(eGIS|eSLICE|eSURF|EDASDB|edas\.read)",
}
```

### Step 2.2: Add Facility-Specific Wrapper Patterns

```python
# TCV-specific access wrappers
"tcv_wrappers": r"(tcv_eq|tcv_get|tcv_psitbx|tcv_ip|tcv_bt|tcv_ne|tcv_te)",

# JET-specific access
"jet_access": r"(jet\.ppf|jpf\.read|jet\.data|sal\.get|sal\.list)",

# JT-60SA access
"jt60sa_access": r"(labcom|lhd_read|nifs_access|edasdb)",
```

### Step 2.3: Add Shot Number Detection Pattern

For identifying code that references specific shots (useful for finding known-good test shots):

```python
"shot_reference": r"(shot\s*=\s*\d{4,6}|pulse\s*=\s*\d{4,6}|pulno\s*=\s*\d{4,6}|shot_number|pulse_number)",
```

### Exit Criteria
- New patterns added to PATTERN_REGISTRY
- Pattern categories cover all known TCV access patterns
- rg enrichment detects data access code in MATLAB, Python, Fortran, IDL files

---

## Phase 3: Run Code Discovery to Completion

### Goal
Run the full code discovery pipeline across all scored FacilityPaths at TCV to build the CodeFile → CodeChunk → DataReference subgraph.

### Step 3.1: Production Scan Run

```bash
uv run imas-codex discover code tcv --scan-only --time 120
```

Scan all 12,237 scored FacilityPaths. Creates CodeFile nodes with language, enrichment data.

**Monitor**:
```cypher
MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
RETURN cf.language, cf.status, count(cf) AS n
ORDER BY n DESC
```

### Step 3.2: Score and Ingest Run

```bash
uv run imas-codex discover code tcv --time 240 -c 20.0 --code-workers 4
```

Score discovered files, ingest high-value ones. Multiple runs may be needed — the pipeline is interrupt-safe and resumes from where it stopped.

**Monitor**:
```cypher
// Progress
MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
RETURN cf.status, count(cf) AS n ORDER BY n DESC

// Top scored files
MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
WHERE cf.interest_score IS NOT NULL
RETURN cf.path, cf.interest_score, cf.file_category, cf.language
ORDER BY cf.interest_score DESC LIMIT 20

// Ingestion progress
MATCH (cc:CodeChunk)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
RETURN count(cc) AS chunks,
       count(CASE WHEN cc.embedding IS NOT NULL THEN 1 END) AS embedded,
       count(CASE WHEN cc.mdsplus_ref_count > 0 THEN 1 END) AS with_refs
```

### Step 3.3: Verify Reference Extraction

```cypher
// MDSplus paths extracted from code
MATCH (cc:CodeChunk)
WHERE cc.mdsplus_ref_count > 0
WITH cc.path AS file, sum(cc.mdsplus_ref_count) AS total_refs
RETURN file, total_refs
ORDER BY total_refs DESC
LIMIT 30

// Unique MDSplus paths found in code
MATCH (dr:DataReference)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
RETURN count(DISTINCT dr.path) AS unique_paths, count(dr) AS total_refs
```

### Exit Criteria
- All scored FacilityPaths scanned for files
- High-value CodeFiles scored and ingested
- CodeChunk nodes with embeddings
- DataReference nodes created from extracted MDSplus paths
- Language coverage includes Python, MATLAB, Fortran

---

## Phase 4: Code-Evidence Signal Linking

### Goal
Link DataReferences extracted from code to TreeNodes and propagate evidence to FacilitySignals. Creates a code→tree→signal evidence chain that validates which signals are actually used by scientists.

### Step 4.1: Verify DataReference → TreeNode Links

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

| Metric | Target |
|--------|--------|
| CodeFile nodes discovered | > 10,000 |
| CodeChunks ingested | > 50,000 |
| Unique MDSplus paths extracted | > 500 |
| Signals with code evidence | > 5,000 |
| Languages covered | Python, MATLAB, Fortran, IDL |
| Signal check success rate | > 70% (up from 50%) |
| DataAccess templates | Cover all major access methods |

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
