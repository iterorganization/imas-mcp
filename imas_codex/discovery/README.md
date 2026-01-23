# Discovery Pipeline

Graph-led filesystem discovery with parallel scan and score workers.

## Design Principles

1. **Single Command Interface**: One `discover` command handles all phases internally
2. **Graph as Truth**: All state in Neo4j, commands are idempotent
3. **Scan Once, Score Many**: SSH enumeration is focus-agnostic, LLM scoring is focus-aware
4. **Multi-Dimensional Scores**: Score all dimensions in one LLM call, store separately
5. **Automatic Seeding**: Facility roots are seeded on first discovery run

## Key Clarifications

### Focus: Natural Language, Not Enum

The `--focus` flag accepts **natural language**, NOT enum values. Examples:

```bash
# All these work - the LLM interprets semantically
uv run imas-codex discover iter --focus "equilibrium codes for control"
uv run imas-codex discover iter --focus "data processing pipelines"
uv run imas-codex discover iter --focus "IMAS integration"
```

**Current implementation**: Focus is passed to the LLM scorer prompt as context.
No preprocessing is done. The LLM interprets it semantically.

**Default (no focus)**: When no `--focus` is provided, we score **all dimensions**
(`code`, `data`, `imas`, `docs`) in a single LLM call. This is the most efficient
approach since we get all scores for the price of one.

### Seeding: LLM-Assisted Root Discovery

**Problem with config-based seeding**: Hardcoded `actionable_paths` in config files
are unreliable - they may not exist, may have moved, or may miss important directories.

**Solution**: Use SSH + fd to discover real root directories, then LLM to filter.

#### Proposed Seeding Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. FILESYSTEM DISCOVERY                      │
│                                                                 │
│  ssh facility "fd -t d --max-depth 2 / 2>/dev/null | head -200" │
│  Returns: /home, /work, /opt, /data, /usr/local, etc.          │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                    2. EXCLUSION FILTER                          │
│                                                                 │
│  Remove system paths: /proc, /sys, /dev, /run, /tmp, /boot     │
│  Remove known noise: /usr/share, /var/log, /etc                │
│  (Uses exclude.yaml patterns)                                  │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                    3. LLM SEED SELECTION                        │
│                                                                 │
│  Prompt: "Given this facility context and directory list,      │
│           identify top 10-20 directories most likely to        │
│           contain: physics codes, IMAS tools, scientific       │
│           data, or documentation. Return JSON array."          │
│                                                                 │
│  Input: Facility description + candidate directories           │
│  Output: Ranked list with confidence scores                    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                    4. GRAPH SEEDING                             │
│                                                                 │
│  Create FacilityPath nodes with status='discovered'            │
│  Set initial interest_score from LLM confidence                │
│  Begin discovery loop                                          │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation Pseudocode

```python
async def discover_seeds(facility: str, max_seeds: int = 20) -> list[str]:
    """Discover seed paths via SSH + LLM, not config files."""
    
    # 1. Quick filesystem scan from root
    output = run_script_via_stdin(
        "fd -t d --max-depth 2 / 2>/dev/null | head -500",
        ssh_host=get_ssh_host(facility),
        timeout=30
    )
    candidates = output.strip().split('\n')
    
    # 2. Apply exclusion patterns
    exclude_config = get_discovery_config().exclusions
    candidates = [
        p for p in candidates 
        if not any(re.match(pat, p) for pat in exclude_config.path_prefixes)
    ]
    
    # 3. LLM selection
    facility_info = get_facility(facility)
    prompt = f"""
    Facility: {facility_info.get('name', facility)}
    Description: {facility_info.get('description', 'Fusion research facility')}
    
    These directories exist on the filesystem:
    {chr(10).join(candidates[:200])}
    
    Select the 10-20 directories most likely to contain:
    - Physics simulation or analysis codes
    - IMAS/IMAS-related tools
    - Scientific data (shots, experiments)
    - Documentation or publications
    
    Return JSON: {{"seeds": [{{"path": "/work/imas", "confidence": 0.9, "reason": "..."}}]}}
    """
    
    response = await llm_call(prompt)
    seeds = response["seeds"][:max_seeds]
    
    return [s["path"] for s in seeds]
```

#### Fallback Behavior

If SSH fails or returns empty, fall back to safe defaults:
- `/home` - user directories
- `/work` - scratch/project space (common on HPC)
- `/data` - data directories

**Config-based seeds are DEPRECATED** - the LLM can make better decisions
from real filesystem state than we can from static config.

### Child Directory Scanning Criteria

**All child directories are scanned automatically** - there is no score-based filter
for whether to enumerate children. The discovery breadth-first expansion works as:

```
Parent scanned → ALL child dirs created as FacilityPath nodes
                 ↓
Child nodes created with status='discovered'
                 ↓  
Scanner claims next batch of 'discovered' paths
                 ↓
Process repeats
```

**Scoring affects PRIORITIZATION, not inclusion**:
- High-scoring paths: Processed first (priority queue)
- Low-scoring paths: Still processed, but later
- Skipped paths: Only excluded by pattern matching (exclude.yaml)

**The DEPTH is the primary constraint**, not the score:

```yaml
# In exclude.yaml
max_depth: 15  # Stop recursing beyond this depth
```

### State Rename: 'expanded' → 'enriched'

The `expanded` state was confusing because it implied recursive descent into
subdirectories. The actual behavior is **rich scanning** (rg + dust) without
changing scope. 

**New name: `enriched`**

| State | Meaning |
|-------|---------|
| `discovered` | Path found, not yet enumerated |
| `listing` | Worker actively enumerating (transient) |
| `listed` | Has file/dir counts, children created |
| `scoring` | Worker actively scoring (transient) |
| `scored` | LLM scored all dimensions |
| `enriched` | **Rich scan complete** (rg matches, size) |
| `skipped` | Score below threshold |
| `excluded` | Matched exclusion pattern |
| `stale` | Needs re-discovery |

**Transition to `enriched`**:
- Trigger: `interest_score >= 0.5` after scoring
- Action: Run `rg` patterns + `dust` size estimation
- Purpose: Gather evidence for high-value paths before ingestion

This happens IN PLACE - no new child paths are created. The path already
has children from the initial listing phase.

### Status Transitions and Scoring

**Path stays `listed` until scored**, then becomes `scored`. The state machine is:

```
discovered → listing → listed → scoring → scored
```

**With multi-dimensional scoring**, a path is scored when the LLM evaluates it.
All dimensions are computed in ONE LLM call. We don't wait for each dimension
separately - that would be 4x more expensive.

### Interest Score: Max, Not Mean

**Recommendation: Use MAX** of dimension scores, not weighted mean.

**Why**: A pure data directory should score 0.9 even with 0.0 for code.
Weighted mean would unfairly penalize specialized directories.

```python
# Proposed calculation
interest_score = max(
    score_code or 0,
    score_data or 0, 
    score_imas or 0,
    score_docs or 0
)
```

This ensures:
- Data-only directories score highly on `score_data`
- Code-only directories score highly on `score_code`
- Mixed directories don't get artificially boosted
- NULL scores are treated as 0 (not scored yet)

### Enriched State Explained

`enriched` means we run a **rich scan** for additional evidence:

1. **Trigger**: Path scores above threshold (`interest_score >= 0.5`)
2. **Action**: Run `rg` (pattern search) and `dust` (size) inside the directory
3. **Scope**: Only the current directory, not subdirs (they're already separate nodes)
4. **Result**: Enhanced data (`rg_matches`, `size_bytes`) stored in graph
5. **No new children**: Children were already created during the listing phase

**Enrichment is in-place**: The path transitions from `scored` → `enriched` without
creating new FacilityPath nodes. This is purely about gathering more evidence
for high-value directories before ingestion.

### 50 Paths/Second Throughput

**Yes, ~50 paths/second is achievable** with light scan mode. Current implementation:

From [parallel.py](parallel.py#L230-L240):
```python
# Use enable_rg=False and enable_size=False for speed
results = await loop.run_in_executor(
    None,
    lambda: scan_paths(fac, pts, enable_rg=False, enable_size=False),
)
```

The light scan uses `fd -t d -d 1` (fast) instead of `rg` (slow).
Batch size is 50 paths per SSH call, and SSH overhead is ~1.8s regardless of batch size.

**Measured rates**:
- Light scan (fd only): 50-100 paths/second
- Rich scan (rg + dust): 5-10 paths/second

### Python Persistence

**No Python restarts in current CLI**. The question arose from the README mentioning
"worker persistence" as a recommendation.

**Current behavior**: The `discover` command runs async workers in a single Python
process. Workers persist for the entire run. No restarts occur.

**The concern was about**: If we had a design where each batch spawned a new
Python process, the ~5s import overhead would dominate. We don't do this.

## CLI Interface

### Primary Command

```bash
uv run imas-codex discover <facility> [options]
```

The command orchestrates scanning and scoring internally. No subcommands needed.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--focus TEXT` | (all dims) | Natural language focus for scoring (e.g., "equilibrium codes") |
| `--cost-limit` | 10.0 | Maximum LLM cost in USD |
| `--limit N` | (none) | Maximum paths to process |
| `--threshold` | 0.3 | Minimum score to expand |
| `--scan-workers` | 2 | Parallel SSH scanner workers |
| `--score-workers` | 2 | Parallel LLM scorer workers |

### Examples

```bash
# Full discovery with default settings (scores all dimensions)
uv run imas-codex discover iter

# Focus scoring on specific topic
uv run imas-codex discover iter --focus "equilibrium codes for real-time control"

# Cost-limited run
uv run imas-codex discover epfl --cost-limit 5.0

# Quick test with limit
uv run imas-codex discover epfl --limit 100
```

### Admin Commands

Keep minimal admin commands in the `discovery` group:

```bash
uv run imas-codex discovery status <facility>   # Show progress
uv run imas-codex discovery clear <facility>    # Delete all paths
```

Remove these commands (no longer needed):
- `discovery seed` - Auto-seeds on first run
- `discovery inspect` - Use graph queries directly

## Multi-Dimensional Scoring

### Schema-Defined Dimensions

Scoring dimensions are defined in schema, not hardcoded. Each dimension:
- Has its own score property (`score_<dimension>`)
- Tracks when it was last scored (`<dimension>_scored_at`)
- Is independently idempotent

```yaml
# FacilityPath scoring dimensions (facility.yaml)
FacilityPath:
  attributes:
    # === Primary scores (0.0-1.0) ===
    score_code:
      description: Source code discovery value
      range: float
    score_data:
      description: Scientific data file value
      range: float
    score_imas:
      description: IMAS integration relevance
      range: float
    score_docs:
      description: Documentation and papers value
      range: float
    
    # === Computed aggregate ===
    interest_score:
      description: Weighted aggregate for prioritization
      range: float
    
    # === Dimension timestamps (for idempotency) ===
    code_scored_at:
      range: datetime
    data_scored_at:
      range: datetime
    imas_scored_at:
      range: datetime
    docs_scored_at:
      range: datetime
```

### Idempotent Rescore Logic

When `--focus dimension` is specified, only rescore if:
1. The dimension has never been scored (`score_<dim>` is NULL), OR
2. The `--force` flag is set

```python
# Pseudo-code for dimension-aware scoring
def should_score(path: FacilityPath, dimension: str, force: bool) -> bool:
    score_prop = f"score_{dimension}"
    if getattr(path, score_prop) is None:
        return True  # Never scored for this dimension
    if force:
        return True  # Explicit rescore requested
    return False  # Already scored, skip

# Graph query for work queue
MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
WHERE p.status = 'listed'
  AND p.score_code IS NULL  // Only paths missing this dimension
RETURN p.id, p.path
LIMIT $batch_size
```

### Orphan Recovery with Dimension Awareness

Transient state recovery considers existing scores:

```cypher
// Reset orphaned scoring states
MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
WHERE p.status = 'scoring'
  AND p.claimed_at < datetime() - duration('PT10M')
SET p.status = CASE
  // If ANY dimension already scored, return to 'scored' not 'listed'
  WHEN p.score_code IS NOT NULL OR p.score_data IS NOT NULL 
       OR p.score_imas IS NOT NULL THEN 'scored'
  ELSE 'listed'
END,
p.claimed_at = null
```

This ensures partially-scored paths don't lose their existing scores.

## Architecture

```
                    ┌─────────────┐
                    │  discover   │  Single CLI command
                    │    iter     │  Auto-seeds on first run
                    └──────┬──────┘
                           │
        ┌──────────────────▼──────────────────┐
        │         1. AUTO-SEEDING             │
        │  If graph empty: seed from config   │
        │  actionable_paths → discovered      │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │         2. SCANNER WORKERS          │
        │  Claims discovered paths atomically │
        │  Runs SSH batch script (fd/ls)      │
        │  Creates child FacilityPath nodes   │
        │  FOCUS-AGNOSTIC                     │
        └──────────────────┬──────────────────┘
                           │
                    discovered → listed
                           │
        ┌──────────────────▼──────────────────┐
        │         3. SCORER WORKERS           │
        │  Claims listed paths atomically     │
        │  Checks which dimensions need score │
        │  Batches for LLM scoring            │
        │  DIMENSION-AWARE                    │
        └──────────────────┬──────────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         All dims       Partial      Low value
          scored        scored        (skip)
              │            │            │
           scored       listed      skipped
              │       (more dims)        │
              ▼            │            ▼
            DONE           │          DONE
                           ▼
                    Continue scoring
```

## State Machine

### State Ordering (Arrow of Time)

States have an implicit order that defines valid transitions and fallback behavior:

```
  ORDER   STATE        FALLBACK     TRANSIENT?
  ─────   ─────        ────────     ──────────
    0     discovered   -            No
    1     listing      discovered   Yes
    2     listed       -            No
    3     scoring      listed       Yes
    4     scored       -            No
    4     skipped      -            No  (terminal alternative)
    4     excluded     -            No  (terminal alternative)
    5     enriched     -            No
    6     stale        -            No  (triggers re-entry at 0)
```

### Long-Lived States (work queues)

| State | Description | Next Action |
|-------|-------------|-------------|
| `discovered` | Path found, awaiting enumeration | Scanner claims |
| `listed` | Enumerated, awaiting LLM score | Scorer claims |
| `scored` | LLM scored, interest_score set | Done or ingest |
| `enriched` | High-value, rich scan completed | Ready for ingestion |
| `skipped` | Low value (score < threshold) | None |
| `excluded` | Matched exclusion pattern | None |
| `stale` | Path may have changed | Re-discover |

### Transient States (worker locks)

| State | Description | Fallback | Timeout |
|-------|-------------|----------|---------|
| `listing` | Scanner actively processing | `discovered` | 10 min |
| `scoring` | Scorer actively processing | `listed` | 10 min |

### Orphan Recovery

When starting a discovery command, first reset orphaned transient states:

```cypher
// Reset orphaned listing states (claimed > 10 min ago)
MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
WHERE p.status = 'listing' 
  AND p.claimed_at < datetime() - duration('PT10M')
SET p.status = 'discovered', p.claimed_at = null

// Reset orphaned scoring states
MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
WHERE p.status = 'scoring'
  AND p.claimed_at < datetime() - duration('PT10M')
SET p.status = 'listed', p.claimed_at = null
```

### State Transitions

```
discovered ──(claim)──> listing ──(complete)──> listed
                            │
                        (timeout)
                            │
                            ▼
                       discovered

listed ──(claim)──> scoring ──(score >= threshold)──> scored
                        │           │
                        │      (score < 0.2)
                        │           │
                        │           ▼
                        │       skipped
                        │
                   (score > 0.5)
                        │
                        ▼
                    enriched   (terminal, ready for ingestion)

scored ──(rescore)──> listed  (when new --focus requested)

stale ──(rediscover)──> discovered
```

## Two-Pass Scan/Score

### Phase 1: Light Scan (default)
- **Goal**: Fast breadth-first enumeration
- **Tools**: `fd -t d --max-depth 1` or `ls` (fallback)
- **Data**: file_count, dir_count, child_names (first 30)
- **No**: rg patterns, dust size
- **Performance**: ~50 paths/second

### Phase 2: Rich Scan (for high-scoring paths)
- **Trigger**: `score >= 0.5` or explicit `expand_to` set
- **Goal**: Deep evidence for informed scoring
- **Tools**: 
  - `rg` for IMAS/MDSplus/physics patterns
  - `dust` for size (with timeouts)
- **Data**: rg_matches, size_bytes, patterns_found
- **Safety**: Timeout per directory, skip if >10s

### Preventing Hangs

| Tool | Safeguard | Implementation |
|------|-----------|----------------|
| `rg` | `--max-depth 2 --max-count 100` | Limit search depth and matches |
| `dust` | Timeout wrapper (5s) | `timeout 5 dust -d 1` |
| `fd` | `--max-depth 1` | Single level only |
| All | Large dir skip | If dir_count > 10000, skip detailed scan |

## Dimension-Based Scoring

### The Challenge

Traditional single-score discovery has two problems:
1. **Lost context**: A path scored 0.3 for "equilibrium" might be 0.9 for "data processing"
2. **Expensive rescores**: Changing focus requires re-evaluating all paths

### Solution: Multi-Dimensional Scores

Each path has independent scores per dimension. The LLM evaluates all dimensions
in a single call but stores them separately:

| Dimension | Purpose | Example High-Value Paths |
|-----------|---------|-------------------------|
| `score_code` | Source code | /home/codes/liuqe/, /work/imas/python/ |
| `score_data` | Scientific data | /work/shots/, /data/thomson/ |
| `score_imas` | IMAS relevance | /work/imas/, /home/tools/imas2mds/ |
| `score_docs` | Documentation | /home/docs/, /work/manuals/ |

### Idempotent Dimension Scoring

The `--focus` flag filters which dimensions to score:

```bash
# Score all dimensions (default, most efficient)
uv run imas-codex discover iter

# Score only code dimension (for new paths)
uv run imas-codex discover iter --focus code

# Score code AND imas dimensions
uv run imas-codex discover iter --focus code --focus imas
```

**Idempotency**: A path is only re-scored for a dimension if:
1. Its `score_<dim>` property is NULL, OR
2. `--force` flag is set

This enables incremental discovery across multiple runs.

### Query by Dimension

```cypher
// Find high-value data pipeline directories
MATCH (p:FacilityPath)
WHERE p.path_purpose = 'data_pipeline' AND p.score_data > 0.7
RETURN p.path, p.score_data

// Find IMAS-relevant code
MATCH (p:FacilityPath)
WHERE p.score_code > 0.5 AND p.score_imas > 0.5
RETURN p.path

// Find all paths related to equilibrium physics
MATCH (p:FacilityPath)
WHERE 'equilibrium' IN p.physics_domains
RETURN p.path, p.path_purpose, p.score_code
```

## Configuration Files

### Pattern Files (Reviewed and Up-to-Date)

| File | Purpose | Status |
|------|---------|--------|
| `config/patterns/exclude.yaml` | Exclusion patterns | ✅ Comprehensive |
| `config/patterns/file_types.yaml` | Extension categorization | ✅ Current |
| `config/patterns/scoring/base.yaml` | Dimension weights, thresholds | ✅ Current |
| `config/patterns/scoring/data_systems.yaml` | IMAS, MDSplus, HDF5, UDA, PPF | ✅ Comprehensive |
| `config/patterns/scoring/physics.yaml` | Physics domain patterns | ✅ Current |

**Exclusion patterns** (`exclude.yaml`) cover:
- VCS: `.git`, `.svn`, `.hg`
- Python: `__pycache__`, `.venv`, `*.egg-info`, `.pytest_cache`
- Node: `node_modules`, `.npm`
- Build: `build`, `dist`, `cmake-build-*`, `target`
- System: `/proc`, `/sys`, `/dev`, `/tmp`
- Large dir threshold: 10,000 entries

**Data system patterns** (`data_systems.yaml`) cover:
- IMAS: `imas.imasdef`, `imas.DBEntry`, `put_slice`, `get_slice`, `imaspy`
- MDSplus: `MDSplus`, `Tree(`, `TdiCompile`, `Connection(`
- HDF5: `h5py`, `HDFStore`, `.h5`, `.hdf5`
- NetCDF: `netCDF4`, `xarray.*open_dataset`, `.nc`
- UDA: `pyuda`, `uda.Client`
- PPF: `ppfgo`, `ppfget`

**Physics patterns** (`physics.yaml`) cover:
- Equilibrium: EFIT, CHEASE, HELENA, LIUQE, psi_*, q_profile
- Transport: ASTRA, JETTO, TRANSP, TGLF, diffusivity
- MHD: JOREK, MARS, MISHKA, sawtooth, ELM, disruption
- Heating: NBI, ECRH, ICRH, power_deposition
- Diagnostics: thomson, ece, interferometer, magnetics
- COCOS: cocos, sign_convention, coordinate_system

### Config vs Schema

| Concern | Config (YAML) | Schema (LinkML) |
|---------|---------------|-----------------|
| **Patterns** | Regex for rg detection | N/A |
| **Weights** | Dimension scoring weights | N/A |
| **Enums** | N/A | PathPurpose, PathStatus |
| **Thresholds** | expand/skip thresholds | N/A |
| **Categories** | Pattern groups (imas, mdsplus) | N/A |
| **Validation** | Runtime | Build-time (Pydantic) |

**Principle**: Schema defines *what* we store, config defines *how* we compute.

## Performance Profile

### SSH Latency
- **Baseline**: ~1.8s per SSH connection (network + auth)
- **Batch efficiency**: Same 1.8s whether 1 or 100 paths in batch

### Python Overhead
- **Local import**: ~4-5s (imas_codex imports)
- **Mitigation**: Persistent worker processes, batch operations
- **NOT an issue for scanning**: Scanner uses `bash -s` via stdin, not `python3 -c`

### Where Python Is Used Over SSH

| Module | Uses Python | Why | Performance Impact |
|--------|-------------|-----|-------------------|
| `discovery/scanner.py` | No | Uses fd/rg/dust bash scripts | Fast (bash only) |
| `mdsplus/metadata.py` | Yes | Needs MDSplus.Tree imports | ~4s startup per call |
| `mdsplus/batch_discovery.py` | Yes | Needs MDSplus for tree introspection | ~4s startup per call |
| `code_examples/facility_reader.py` | Yes | Reads Python source files | ~4s startup per call |

**Key insight**: Discovery scanning is fast because it uses bash scripts. MDSplus introspection
is slow because it needs Python with MDSplus imports. These are different use cases.

### Scan Throughput

**Verified experimentally** (2025-01-09):

| Operation | Throughput | Notes |
|-----------|-----------|-------|
| Light scan (fd only) | 25-50 paths/s | 1.8s SSH + parallel fd in batch |
| Rich scan (fd + rg + dust) | 5-10 paths/s | rg dominates with pattern matching |
| MDSplus metadata | ~1 tree/4s | Python startup is bottleneck |

**Math**: 50 paths batched in single SSH call = 1.8s + ~0.02s/path = ~3s total = 17 paths/s.
With parallel workers (3-4), can sustain 50+ paths/s aggregate.

### Recommendations
1. **Large batches**: 50+ paths per SSH call
2. **Worker persistence**: Don't restart Python between batches
3. **Parallel workers**: 2-4 scanner workers, 1-2 scorer workers
4. **Skip .bashrc**: Non-interactive SSH (`ssh -T bash -s`) avoids bashrc

## Graph Schema Alignment

### Current Issues (Jan 2026)

1. **Status mismatch**: CLI uses `pending/scanning/scored`, schema has
   `discovered/listed/scanned/analyzed/explored`
2. **Duplicate enums**: `PathStatus` (common.yaml) vs `DiscoveryStatus` (facility.yaml)
3. **Extra properties**: `total_files` vs `file_count`, `score` vs `interest_score`
4. **Missing enum value**: LLM returns `empty_directory`, not in PathPurpose

### Proposed Schema Changes

#### 1. Unified PathStatus Enum (common.yaml)

Remove `DiscoveryStatus` from facility.yaml. Update `PathStatus` to:

```yaml
PathStatus:
  description: >-
    Lifecycle status for facility path discovery.
    Transient states (listing, scoring) have fallback states for orphan recovery.
  permissible_values:
    discovered:
      description: Path found, awaiting enumeration
    listing:
      description: Scanner worker active (fallback → discovered)
    listed:
      description: Enumerated (file_count, dir_count set), awaiting score
    scoring:
      description: Scorer worker active (fallback → listed or scored)
    scored:
      description: All requested dimensions scored
    enriched:
      description: High-value path with rich scan data (rg matches, size)
    skipped:
      description: Low value or dead-end
    excluded:
      description: Matched exclusion pattern (not enumerated)
    stale:
      description: Path may have changed, needs re-discovery
```

#### 2. PathPurpose Enum (facility.yaml)

Directory classification for future filtering and search. Design for searchability:
"Show me all data pipeline directories" or "Find physics code locations".

```yaml
PathPurpose:
  description: >-
    LLM-classified purpose of a directory. Enables semantic search across
    facility filesystems: "find all data pipeline directories", 
    "locate physics simulation code", etc.
  permissible_values:
    # === Code categories ===
    physics_code:
      description: Physics simulation, equilibrium, transport codes
    data_pipeline:
      description: Data processing infrastructure (ETL, workflows, pipelines)
    analysis_scripts:
      description: Data analysis scripts and notebooks
    control_systems:
      description: Real-time plasma control code
    
    # === Data categories ===
    raw_data:
      description: Raw experimental or simulation data
    processed_data:
      description: Processed/reduced scientific data (HDF5, NetCDF)
    calibration:
      description: Instrument calibration files
    
    # === Documentation ===
    documentation:
      description: Technical documentation, manuals
    publications:
      description: Papers, reports, theses
    
    # === Infrastructure ===
    configuration:
      description: Configuration files, settings
    build_artifacts:
      description: Compiled outputs, caches, build directories
    test_files:
      description: Test suites and testing infrastructure
    deployment:
      description: Deployment scripts, containers, CI/CD
    
    # === Structural ===
    user_home:
      description: Personal user directories
    project_root:
      description: Top-level project directory
    library:
      description: Shared libraries or third-party code
    archive:
      description: Backup or archived directories
    system:
      description: OS or infrastructure directories
    empty_directory:
      description: No files, may contain subdirectories
    unknown:
      description: Cannot determine purpose
```

#### 3. Use Existing PhysicsDomain

We already have a comprehensive `PhysicsDomain` enum in `imas_codex/core/physics_domain.py`
generated from LinkML. This includes 22 domains covering:

- `EQUILIBRIUM`, `TRANSPORT`, `MAGNETOHYDRODYNAMICS`, `TURBULENCE`
- `AUXILIARY_HEATING`, `CURRENT_DRIVE`
- `DIVERTOR_PHYSICS`, `EDGE_PLASMA_PHYSICS`, `PLASMA_WALL_INTERACTIONS`
- `PARTICLE_MEASUREMENT_DIAGNOSTICS`, `ELECTROMAGNETIC_WAVE_DIAGNOSTICS`, etc.
- `PLASMA_CONTROL`, `MACHINE_OPERATIONS`
- `DATA_MANAGEMENT`, `COMPUTATIONAL_WORKFLOW`

**Do not duplicate** - import from `imas_codex.core.physics_domain.PhysicsDomain`.

#### 4. FacilityPath Scoring Properties

```yaml
FacilityPath:
  attributes:
    # === Scoring dimensions (0.0-1.0, null = not scored) ===
    score_code:
      description: Source code discovery value (physics, analysis, control)
      range: float
    score_data:
      description: Scientific data file value (raw, processed, calibration)
      range: float
    score_imas:
      description: IMAS/IDS integration relevance
      range: float
    score_docs:
      description: Documentation and publication value
      range: float
    
    # === Aggregate for prioritization ===
    interest_score:
      description: Weighted aggregate (max of dimensions by default)
      range: float
    
    # === Dimension timestamps (idempotency) ===
    code_scored_at:
      description: When score_code was computed
      range: datetime
    data_scored_at:
      range: datetime
    imas_scored_at:
      range: datetime
    docs_scored_at:
      range: datetime
    
    # === Classification ===
    path_purpose:
      description: LLM-classified purpose
      range: PathPurpose
    physics_domains:
      description: Relevant physics domains
      multivalued: true
    keywords:
      description: Searchable keywords from LLM (max 5)
      multivalued: true
    
    # === Worker coordination ===
    claimed_at:
      description: When worker claimed this path (orphan detection)
      range: datetime
    
    # === Cost tracking ===
    score_cost:
      description: Total LLM cost in USD for all scoring
      range: float
```

## Clean Slate for Testing

Before testing the new approach, clear existing data:

```cypher
// Clear all EPFL and ITER FacilityPath nodes
MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility)
WHERE f.id IN ['epfl', 'iter']
DETACH DELETE p
```

Or via CLI:
```bash
uv run imas-codex discovery clear epfl --force
uv run imas-codex discovery clear iter --force
```

## Migration Plan

### Phase 1: Schema Update

1. **Update `common.yaml`**:
   - Replace `PathStatus` enum with unified version (add `listing`, `scoring` transients)

2. **Update `facility.yaml`**:
   - Remove `DiscoveryStatus` enum (duplicate)
   - Extend `PathPurpose` with new categories (`data_pipeline`, etc.)
   - Add scoring dimension properties to `FacilityPath`

3. **Regenerate models**:
   ```bash
   uv run python scripts/build_schemas.py
   ```

### Phase 2: CLI Refactor

1. **Simplify `discover` command**:
   - Auto-seed on first run (no separate `seed` command)
   - Add `--focus` for dimension filtering
   - Handle scan/score phases internally

2. **Remove commands**:
   - `discovery seed` - Auto-seeds
   - `discovery inspect` - Use graph queries

3. **Keep commands**:
   - `discovery status` - Progress overview
   - `discovery clear` - Reset for testing

### Phase 3: Scorer Update

1. **Dimension-aware scoring**:
   - Check which dimensions are NULL before scoring
   - Skip paths already scored for requested dimension
   - Update only requested dimension properties

2. **Orphan recovery update**:
   - Check for existing scores before resetting status
   - Paths with any score → `scored` not `listed`

## Next Steps

1. [ ] Update `PathStatus` enum in `common.yaml`
2. [ ] Extend `PathPurpose` enum in `facility.yaml`
3. [ ] Add scoring dimension properties to `FacilityPath`
4. [ ] Remove `DiscoveryStatus` enum
5. [ ] Clear EPFL/ITER paths from graph
6. [ ] Refactor CLI to auto-seed and use `--focus`
7. [ ] Update scorer for dimension-aware idempotency
8. [ ] Test end-to-end discovery flow
