# Enrichment Pipeline Improvements

Status: **In Progress**
Priority: High — directly impacts discovery quality and paths→files handoff

## Summary

Improvements to the enrichment/rescore pipeline addressing data flow gaps,
concurrent execution, pattern matching expansion, and pipeline continuity
into file discovery.

## Changes Implemented

### 1. Pattern Evidence Now Reaches the Rescorer

**Problem:** The rescore prompt described `pattern_categories` as key evidence,
but `_rescore_with_llm()` never included pattern_categories, read_matches, or
write_matches in the user message. The LLM was told to use evidence it never
received — likely the root cause of the 14.5% parse error rate and high "no
change" rate.

**Fix:** Both `_rescore_with_llm()` and `_async_rescore_with_llm()` now include:
- `pattern_categories` (JSON-decoded from graph property)
- `read_matches` / `write_matches` counts
- `enrich_warnings` (sync version was missing this; async already had it)

The data flow is now: `enrich_directories.py` → `mark_enrichment_complete()` →
graph → `claim_paths_for_rescoring()` → `_rescore_with_llm()` user prompt →
rescorer LLM.

### 2. Standalone persist_enrichment() Fixed

**Problem:** `persist_enrichment()` in `enrichment.py` (used by
`run_enrichment_pipeline()`) only wrote 6 of 10 enrichment fields. Missing:
`pattern_categories`, `read_matches`, `write_matches`, `enrich_warnings`.

**Fix:** `to_graph_dict()` and `persist_enrichment()` now write all fields,
matching what `mark_enrichment_complete()` in `parallel.py` already does.
Dicts are JSON-serialized and warnings are comma-joined for Neo4j storage.

### 3. Rescore Prompt Improved

**Problem:** The prompt didn't distinguish between "rg unavailable" (empty
pattern_categories `{}`) and "rg found no matches" (categories present but
specific ones missing). This caused the LLM to incorrectly reduce scores
when it had no evidence either way.

**Fix:** Added explicit guidance:
- Empty `{}` = no evidence, keep original scores
- Categories present but specific one missing = evidence of absence, reduce
- Added `enrich_warnings` as recognized evidence field
- Added guidance on handling partial timeouts

### 4. Concurrent du + rg Execution

**Problem:** `enrich_directories.py` ran du → tokei → rg sequentially per
directory. du and rg are completely independent I/O operations.

**Fix:** du and rg now run concurrently via `ThreadPoolExecutor` (stdlib, no
external deps, Python 3.8+ compatible). tokei still runs after du because its
timeout scales by `total_bytes`. Per-directory speedup ~30-40% when both
commands are slow.

**Timeout independence:** Each command has its own timeout. A du timeout
produces a `du_timeout` warning without blocking rg results. A tokei timeout
produces `tokei_timeout(NB)` without losing du/rg data. This was partially
implemented before; now all three streams are fully independent.

## Assessment: Additional rg Pattern Opportunities

### Current State

8 pattern groups with 40+ patterns across data_access, imas, modeling,
analysis, operations, workflow, visualization, documentation dimensions.

### Recommended Additions

**COCOS (Coordinate Convention) Detection:**
```python
"cocos": r"(cocos|COCOS|coordinate_convention|cocos_define|transform_cocos)"
```
Dimension: `score_imas` — COCOS awareness strongly correlates with IMAS integration.

**Error/Uncertainty Quantification:**
```python
"uncertainty": r"(error_upper|error_lower|uncertainty|confidence_interval|standard_deviation)"
```
Dimension: `score_analysis_code` — codes that track uncertainties are more sophisticated analysis tools.

**Shot/Pulse Number Handling:**
```python
"shot_handling": r"(shot_number|pulse_number|shot_no|pulse_no|shot_list|pulse_list|discharge)"
```
Dimension: `score_data_access` — shot-aware code definitely accesses experimental data.

**Unit Handling:**
```python
"units": r"(pint\.|astropy\.units|units\.convert|unit_conversion|SI_units)"
```
Dimension: `score_analysis_code` — unit-aware code is higher quality analysis.

**Configuration/Parameter Files:**
```python
"config_io": r"(namelist|fortran_namelist|f90nml|configparser|jetto\.jset|astra\.in)"
```
Dimension: `score_modeling_code` — namelist I/O is a strong modeling code signal.

### Not Recommended

- **Generic Python patterns** (import numpy, import scipy) — too noisy, no signal
- **Logging/testing patterns** — orthogonal to physics relevance
- **Build system patterns** (cmake, make) — already captured by has_makefile flag

## Assessment: Pattern Storage — File Nodes vs Directory Nodes

### Current State

Pattern matches are stored at the **directory level** (FacilityPath nodes):
`pattern_categories`, `read_matches`, `write_matches`. This is appropriate
because rg searches entire directory trees — match counts are aggregated.

### File-Level Pattern Storage

**When it becomes valuable:** After file discovery creates SourceFile nodes
linked to FacilityPath via `CONTAINS` relationships, we could run targeted
rg per-file during or after file scanning to determine which specific files
contain which patterns.

**Schema readiness:** SourceFile already has a `patterns_matched` property
in the LinkML schema (string type for JSON storage).

**Implementation path:**
1. During `_scan_remote_path()`, optionally run per-file rg for high-scoring paths
2. Store as `patterns_matched` JSON on each SourceFile node
3. Use during ingestion prioritization — files with `mdsplus` or `imas` patterns
   get ingested first

**Cost/benefit:** Per-file rg is expensive (one rg invocation per file per
pattern category). Better approach: use the directory-level pattern_categories
to prioritize which files to ingest, then extract patterns during ingestion
when the file content is already being parsed.

**Recommendation:** Keep patterns at directory level for discovery. During
ingestion, the code parser already extracts imports, function calls, and
data access patterns — store those as structured metadata on SourceFile/CodeChunk
nodes rather than running rg again.

## Paths → Files Pipeline Design

### Current Flow

```
discover paths <facility>        discover files <facility>        ingest run <facility>
┌─────────────────────┐          ┌──────────────────────┐         ┌─────────────────┐
│ 1. Scan dirs (SSH)  │          │ 1. Query scored paths│         │ 1. Queue files  │
│ 2. Score dirs (LLM) │──────>   │ 2. List files (SSH)  │──────>  │ 2. Parse code   │
│ 3. Enrich (du/rg)   │  score≥N │ 3. Score files (LLM) │  score  │ 3. Extract meta │
│ 4. Rescore (LLM)    │          │ 4. Create SourceFile │  ≥ 0.5  │ 4. Embed chunks │
└─────────────────────┘          └──────────────────────┘         └─────────────────┘
     FacilityPath                     SourceFile                    CodeChunk
     (status: scored)                 (status: discovered)          TreeNode
```

### How Enrichment Feeds File Discovery

1. **Score filtering:** `discover files` queries paths with `score ≥ min_score`.
   Enrichment+rescore refines these scores, ensuring only genuinely valuable
   directories get file-scanned.

2. **Pattern-informed file scoring:** The file scorer prompt could receive
   parent directory enrichment data (pattern_categories) to inform file-level
   scoring decisions. Currently not implemented.

3. **Language breakdown → extension filtering:** tokei's language breakdown
   tells us what languages exist in a directory. The file scanner already
   filters by supported extensions, but could be smarter about priorities.

### Proposed Improvements

**Pass enrichment context to file scorer:**
When scoring files, include the parent FacilityPath's enrichment data in the
file scorer prompt. If the parent directory has `{"mdsplus": 15, "imas": 3}`,
the file scorer knows these files likely access MDSplus and IMAS.

**Priority scanning based on patterns:**
Scan directories with richer pattern evidence first. A directory with
`pattern_categories: {"mdsplus": 20, "equilibrium": 8}` should be scanned
before one with only `{"plotting": 2}`.

**Ingestion priority from enrichment:**
When queueing files for ingestion, use the parent directory's pattern_categories
to set ingestion priority. Files in IMAS-heavy directories get ingested first.

### Key Relationships

```
(FacilityPath)-[:CONTAINS]->(SourceFile)-[:HAS_CHUNK]->(CodeChunk)
(FacilityPath)-[:AT_FACILITY]->(Facility)
(SourceFile)-[:AT_FACILITY]->(Facility)
(CodeChunk)-[:REFERENCES_IDS]->(IMASPath)
```

The `CONTAINS` relationship is the bridge. File scanning creates it;
ingestion traverses it. Enrichment data on the FacilityPath node propagates
context downstream through this relationship.
