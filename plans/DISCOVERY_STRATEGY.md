# Discovery Strategy

> **Goal**: Exhaustive, multi-dimensional file discovery across fusion facilities with flexible scoring and efficient ingestion.

## Executive Summary

This document outlines a three-phase strategy for code discovery:

1. **Map Phase**: Exhaustive file enumeration (no LLM, fd/rg tools)
2. **Score Phase**: Multi-dimensional relevance scoring (ReAct agent, LLM-driven)
3. **Ingest Phase**: Optimized pipeline with streaming and local embedding

Key innovations:
- **Multi-dimensional scoring** instead of binary skip/include
- **Persistent pattern filters** in YAML for reproducible searches
- **Tool requirements** as project-level, not facility-specific
- **Graph-driven workflow** using existing SourceFile/FacilityPath nodes

## Current State Assessment

### Graph Entities (EPFL)

| Entity | Count | Notes |
|--------|-------|-------|
| FacilityPath | 65 | Directories with exploration metadata |
| SourceFile | 1822 | Files queued/processed |
| CodeExample | 1017 | Processed files with metadata |
| CodeChunk | 0 | Not yet implemented |

### FacilityPath Status Distribution

| Status | Count | Meaning |
|--------|-------|---------|
| ingested | 28 | Fully processed |
| discovered | 13 | Found, awaiting exploration |
| excluded | 8 | Intentionally skipped |
| flagged | 8 | Needs review |
| scanned | 6 | Pattern search complete |
| analyzed | 2 | Deep analysis done |

### Remote Environment (EPFL)

| Resource | Details |
|----------|---------|
| Python | 3.9.25 with pip 21.3.1 |
| GPU | Not available (no nvidia-smi) |
| Fast tools | rg 14.1.1, fd 10.2.0, scc 3.4.0, tokei 12.1.2, dust 1.1.1 |
| Filesystem | ~1035 home directories, ~1960 IMAS/MDSplus files (depth 4) |

### Implications

- **Remote embedding not feasible**: No GPU, PyTorch unavailable
- **Remote chunking possible**: Python 3.9 supports tree-sitter
- **Fast tools available**: All enumeration can use fd/rg
- **Scale manageable**: ~2K files per facility is tractable

## Phase 1: Map Agent (No LLM)

### Goal
Build a complete file inventory with metadata, stored in graph as SourceFile nodes.

### Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                        MAP AGENT                                 │
│                                                                  │
│  1. Load pattern filters from YAML                              │
│  2. Run fd with file extension filters                          │
│  3. For each file: run rg to check pattern matches              │
│  4. Compute metadata: size, mtime, hash (optional)              │
│  5. Create SourceFile nodes with multi-dimensional scores       │
│  6. Maintain exploration frontier in graph                       │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern Filters (YAML)

```yaml
# config/patterns/discovery.yaml
version: "1.0"
description: "Multi-dimensional discovery patterns"

file_extensions:
  code:
    - py
    - f90
    - f
    - pro  # IDL
    - m    # MATLAB
  config:
    - yaml
    - json
    - xml
    - cfg

dimensions:
  mdsplus:
    description: "MDSplus data access patterns"
    patterns:
      - pattern: "MDSplus|mdsplus"
        weight: 1.0
      - pattern: "Tree\\(|TdiCompile|TdiExecute"
        weight: 0.9
      - pattern: "tcvtree|tcv_shot"
        weight: 0.8

  imas:
    description: "IMAS/IDS integration"
    patterns:
      - pattern: "imas|ids_properties"
        weight: 1.0
      - pattern: "equilibrium|core_profiles"
        weight: 0.8
      - pattern: "write_ids|read_ids|put_slice"
        weight: 0.9

  physics:
    description: "Physics/analysis code"
    patterns:
      - pattern: "equilibrium|psi_norm|q_profile"
        weight: 0.9
      - pattern: "transport|diffusion|flux"
        weight: 0.8
      - pattern: "COCOS|cocos"
        weight: 0.7

  geometry:
    description: "Geometry/topology code"
    patterns:
      - pattern: "boundary|separatrix|lcfs"
        weight: 0.9
      - pattern: "flux_surface|contour"
        weight: 0.8

exclude_paths:
  - ".git"
  - "__pycache__"
  - "node_modules"
  - ".venv"
  - "venv"
  - "build"
  - "dist"
```

### SourceFile Schema Extension

The existing SourceFile node already has:
- `id`, `path`, `facility_id`, `status`
- `interest_score` (single dimension)
- `patterns_matched` (list of strings)

**Proposed additions** (via graph properties, no schema change needed):

```python
# Additional properties for multi-dimensional scoring
{
    # Dimension scores (0.0 - 1.0)
    "score_mdsplus": 0.9,
    "score_imas": 0.7,
    "score_physics": 0.5,
    "score_geometry": 0.0,
    
    # Metadata
    "file_size": 12345,
    "file_mtime": "2024-01-15T10:30:00Z",
    "file_hash": "sha256:abc123...",  # Optional, for change detection
    
    # Discovery metadata
    "discovered_at": "2025-01-15T10:30:00Z",
    "discovered_by": "map_agent_v1",
    "pattern_version": "1.0",
}
```

### Implementation Sketch

```python
# imas_codex/discovery/map_agent.py

class MapAgent:
    """No-LLM file discovery agent using fast tools."""
    
    def __init__(self, facility: str, config: DiscoveryConfig):
        self.facility = facility
        self.config = config
        self.graph = GraphClient()
    
    async def discover_files(self, root_paths: list[str]) -> DiscoveryResult:
        """Exhaustive file discovery with pattern matching."""
        
        for root in root_paths:
            # 1. Enumerate all code files
            files = await self._enumerate_files(root)
            
            # 2. Score each file across dimensions
            for batch in chunked(files, 100):
                scored = await self._score_batch(batch)
                
                # 3. Create SourceFile nodes
                self._persist_files(scored)
        
        return DiscoveryResult(...)
    
    async def _enumerate_files(self, root: str) -> list[str]:
        """Use fd to enumerate files by extension."""
        extensions = self.config.file_extensions["code"]
        ext_args = " ".join(f"-e {ext}" for ext in extensions)
        
        cmd = f"fd -t f {ext_args} {root}"
        result = await ssh_async(cmd, self.facility)
        return result.strip().split("\n")
    
    async def _score_batch(self, files: list[str]) -> list[ScoredFile]:
        """Score files across all dimensions using rg."""
        results = []
        
        for dimension, config in self.config.dimensions.items():
            # Build combined pattern for this dimension
            patterns = "|".join(p["pattern"] for p in config["patterns"])
            
            # Run rg on all files
            cmd = f"rg -l '{patterns}' {' '.join(files)}"
            matching = await ssh_async(cmd, self.facility)
            
            # Record matches
            for file in matching.strip().split("\n"):
                results.append((file, dimension, config["patterns"]))
        
        return self._aggregate_scores(results)
```

### Frontier Management

Instead of tracking "explored" directories separately, use FacilityPath nodes:

```cypher
-- Get unexplored paths ordered by priority
MATCH (fp:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: "epfl"})
WHERE fp.status IN ["discovered", "scanned"]
RETURN fp.path, fp.interest_score
ORDER BY fp.interest_score DESC
LIMIT 10
```

## Phase 2: Score Agent (ReAct, LLM-Driven)

### Goal
Enrich high-value files with semantic understanding, relationships, and quality assessment.

### When to Use LLM

| Task | LLM Needed? | Reason |
|------|-------------|--------|
| File enumeration | No | fd is exhaustive |
| Pattern matching | No | rg is deterministic |
| Dimension scoring | No | Rule-based from YAML |
| **Code summarization** | **Yes** | Semantic understanding |
| **Relationship discovery** | **Yes** | Cross-file analysis |
| **Quality assessment** | **Yes** | Judgment required |

### Budget-Aware Scoring

With $20/session budget (~10M tokens at Claude 3.5):

```python
# Prioritization strategy
files = get_pending_files(facility, min_score=0.7, limit=1000)

# Estimate token budget
avg_file_size = 2000  # tokens
files_per_session = 10_000_000 / (avg_file_size + 500)  # ~4500 files

# Batch by priority
for batch in prioritized_batches(files, batch_size=50):
    enrichments = await score_agent.enrich_batch(batch)
    persist_enrichments(enrichments)
```

### ReAct Agent Design

```python
# imas_codex/agents/score_agent.py

class ScoreAgent:
    """ReAct agent for semantic file enrichment."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.llm = LiteLLM(model=model)
        self.tools = [
            ReadFileTool(),
            SearchCodeTool(),
            SearchIMASTool(),
            GetTreeStructureTool(),
        ]
    
    async def enrich_file(self, source_file: SourceFile) -> Enrichment:
        """Use ReAct loop to analyze a single file."""
        
        prompt = f"""
        Analyze this source file and provide:
        1. Brief description (1-2 sentences)
        2. Key functions/classes and their purposes
        3. Data access patterns (MDSplus trees, IDS paths)
        4. Related files or dependencies
        5. Code quality assessment (1-5)
        
        File: {source_file.path}
        Dimension scores: mdsplus={source_file.score_mdsplus}, imas={source_file.score_imas}
        
        Use tools to read the file and explore relationships.
        """
        
        return await self.react_loop(prompt)
```

## Phase 3: Ingestion Pipeline

### Current Pipeline

```
Scout → queue files → imas-codex ingest queue → imas-codex ingest run
                                                        │
                                                        ▼
                                                 LlamaIndex pipeline
                                                        │
                                                        ▼
                                                 Local embedding
```

### Optimized Pipeline

```
Map Agent ──► SourceFile nodes ──► Score Agent ──► Priority queue
                                                        │
                 ┌──────────────────────────────────────┘
                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    STREAMING INGESTION                       │
    │                                                              │
    │  1. Fetch file via SSH (paramiko/fabric)                    │
    │  2. Parse with tree-sitter (local)                          │
    │  3. Chunk semantically (local)                              │
    │  4. Embed chunks (local, HuggingFace)                       │
    │  5. Create CodeChunk nodes                                   │
    │  6. Link to TreeNode/IDS paths                              │
    │  7. Update SourceFile status                                 │
    └─────────────────────────────────────────────────────────────┘
```

### Streaming Architecture

```python
# imas_codex/code_examples/streaming.py

async def ingest_stream(facility: str, batch_size: int = 10):
    """Stream files from remote to local processing."""
    
    # Get prioritized queue
    async for batch in get_pending_batches(facility, batch_size):
        
        # Parallel fetch
        files = await asyncio.gather(*[
            fetch_file_content(f.path, facility)
            for f in batch
        ])
        
        # Process locally
        for source_file, content in zip(batch, files):
            try:
                # Parse and chunk
                chunks = parse_and_chunk(content, source_file.language)
                
                # Embed
                embeddings = embed_batch([c.text for c in chunks])
                
                # Create nodes
                create_code_chunks(source_file, chunks, embeddings)
                
                # Update status
                update_source_file_status(source_file, "ready")
                
            except Exception as e:
                update_source_file_status(source_file, "failed", error=str(e))
```

## Tool Requirements

### Project-Level Definition

Move tool requirements from facility YAML to project config:

```yaml
# config/tool_requirements.yaml

version: "1.0"

required_tools:
  - name: fd
    min_version: "8.0.0"
    install_url: "https://github.com/sharkdp/fd/releases"
    purpose: "Fast file enumeration"
    
  - name: rg
    min_version: "13.0.0"
    install_url: "https://github.com/BurntSushi/ripgrep/releases"
    purpose: "Fast pattern search"
    
  - name: scc
    min_version: "3.0.0"
    install_url: "https://github.com/boyter/scc/releases"
    purpose: "Code complexity metrics"
    optional: true

optional_tools:
  - name: tokei
    purpose: "Lines of code statistics"
    
  - name: dust
    purpose: "Disk usage visualization"
```

### Check Utility

Add to REPL utilities:

```python
def check_tools(facility: str = "epfl") -> dict:
    """Check tool availability and versions."""
    from imas_codex.discovery import get_tool_requirements
    
    requirements = get_tool_requirements()
    results = {}
    
    for tool in requirements["required_tools"]:
        version = ssh(f"{tool['name']} --version 2>/dev/null | head -1", facility)
        results[tool["name"]] = {
            "available": bool(version.strip()),
            "version": version.strip() if version else None,
            "required": tool["min_version"],
            "ok": version_gte(version, tool["min_version"]) if version else False,
        }
    
    return results
```

## Implementation Phases

### Phase 1: Map Agent (Week 1-2)

| Task | Priority | Effort |
|------|----------|--------|
| Create `config/patterns/discovery.yaml` | High | 2h |
| Add `check_tools()` utility | High | 1h |
| Implement `MapAgent` class | High | 4h |
| Add dimension score properties to SourceFile | Medium | 1h |
| Test on EPFL ~/home sample | High | 2h |

### Phase 2: Score Agent (Week 2-3)

| Task | Priority | Effort |
|------|----------|--------|
| Design enrichment schema | High | 2h |
| Implement `ScoreAgent` ReAct loop | High | 4h |
| Add batch prioritization | Medium | 2h |
| Budget tracking | Medium | 1h |
| Test on high-value files | High | 2h |

### Phase 3: Streaming Ingestion (Week 3-4)

| Task | Priority | Effort |
|------|----------|--------|
| Refactor pipeline for streaming | High | 4h |
| Add parallel file fetching | Medium | 2h |
| Optimize embedding batch size | Medium | 1h |
| Add resume capability | High | 2h |
| Performance benchmarking | Medium | 2h |

## Metrics

### Discovery Metrics

```cypher
-- Discovery coverage
MATCH (sf:SourceFile)-[:FACILITY_ID]->(f:Facility {id: $facility})
WITH count(*) AS total,
     sum(CASE WHEN sf.status = 'ready' THEN 1 ELSE 0 END) AS ready,
     sum(CASE WHEN sf.status = 'failed' THEN 1 ELSE 0 END) AS failed
RETURN total, ready, failed, 
       toFloat(ready) / total AS completion_rate
```

### Dimension Coverage

```cypher
-- Files by dimension
MATCH (sf:SourceFile)-[:FACILITY_ID]->(f:Facility {id: $facility})
WHERE sf.score_mdsplus IS NOT NULL
RETURN 
    count(CASE WHEN sf.score_mdsplus > 0.5 THEN 1 END) AS high_mdsplus,
    count(CASE WHEN sf.score_imas > 0.5 THEN 1 END) AS high_imas,
    count(CASE WHEN sf.score_physics > 0.5 THEN 1 END) AS high_physics
```

## Open Questions — RESOLVED

### 1. File Hashing for Change Detection ✅

**Decision**: Use **metadata fingerprinting** (size + mtime) for all files, with optional content hashes for high-value files.

**Performance Data** (EPFL):
| Method | Files | Time | Rate |
|--------|-------|------|------|
| md5sum (content) | 500 | 3.6s | 138/sec |
| stat (size+mtime) | 500 | 2.9s | 172/sec |
| Full scan (depth 4) | 7561 | 20.7s | 365/sec |

**Implementation**:
```python
# Lightweight fingerprint for all files (fast change detection)
file_fingerprint = f"{file_size}:{file_mtime}"

# Content hash only for high-value files (score > 0.7)
if interest_score > 0.7:
    content_hash = md5sum(file_path)
```

**Benefits**:
- Fast incremental discovery: Compare fingerprints to detect changes
- Content hashes for reproducibility on high-value code
- ~365 files/sec allows full facility scan in ~1 minute

### 2. Tree-sitter for Parsing ✅

**Decision**: Use **tree-sitter for all supported languages**, with regex fallback only for unsupported languages.

**Supported Languages** (via `tree-sitter-language-pack`):
| Language | Support | Notes |
|----------|---------|-------|
| Python | ✅ | Full AST |
| Fortran | ✅ | Full AST (both F77 and F90) |
| C/C++ | ✅ | Full AST |
| MATLAB | ✅ | Full AST |
| Julia | ✅ | Full AST |
| IDL (.pro) | ❌ | Use regex fallback |

**Implementation**:
```python
from tree_sitter_language_pack import get_language, get_parser

TREE_SITTER_LANGS = {'python', 'fortran', 'c', 'cpp', 'matlab', 'julia'}
REGEX_FALLBACK = {'idl', 'pro'}

def parse_file(path: str, language: str) -> list[Chunk]:
    if language in TREE_SITTER_LANGS:
        return tree_sitter_parse(path, language)
    else:
        return regex_chunker(path, language)
```

### 3. YAML Configuration Structure ✅

**Decision**: Consolidate all YAML into a **coherent directory structure** with clear purposes.

**Current Structure** (needs consolidation):
```
imas_codex/
├── config/              # Runtime configuration
│   ├── facilities/      # Per-facility config (public + private)
│   ├── patterns/        # Discovery patterns
│   └── tool_requirements.yaml
├── definitions/         # Static domain knowledge
│   ├── physics/         # Physics domains, units
│   └── clusters/        # Semantic cluster labels
└── schemas/             # LinkML data models (source of truth)
    ├── facility.yaml
    ├── imas_dd.yaml
    └── common.yaml
```

**Design Principles**:
| Directory | Purpose | Composable? | Version Controlled? |
|-----------|---------|-------------|---------------------|
| `schemas/` | Data model definitions | No (source of truth) | Yes |
| `definitions/` | Static domain knowledge | Yes (by physics domain) | Yes |
| `config/` | Runtime configuration | Yes (per-facility) | Partial (secrets gitignored) |

**Pattern Composition**:
```yaml
# config/patterns/discovery.yaml - Base patterns (all facilities)
# config/patterns/epfl.yaml - EPFL-specific overrides (optional)
# config/patterns/iter.yaml - ITER-specific overrides (optional)

# Loading order (merges with override):
patterns = load_patterns("discovery.yaml")
if facility_patterns := load_patterns(f"{facility}.yaml"):
    patterns = deep_merge(patterns, facility_patterns)
```

### 4. Incremental Discovery Schedule ✅

**Decision**: **Daily during initial exploration, then weekly, then on-demand**.

**Implementation**:
```python
# SourceFile properties for change tracking
{
    "fingerprint": "12345:1705312200",  # size:mtime
    "last_scanned": "2025-01-15T10:30:00Z",
    "scan_count": 3,
    "change_detected": false,
}

# Discovery schedule logic
def should_rescan(file: SourceFile, now: datetime) -> bool:
    age = now - file.last_scanned
    
    if file.scan_count < 7:  # First week: daily
        return age > timedelta(days=1)
    elif file.scan_count < 30:  # Month 1: weekly
        return age > timedelta(weeks=1)
    else:  # After: on-demand or monthly
        return age > timedelta(weeks=4)
```

**CLI Support**:
```bash
# Full discovery (first run or forced)
uv run imas-codex discover epfl --full

# Incremental (only changed files)
uv run imas-codex discover epfl --incremental

# Scheduled via cron
0 6 * * * uv run imas-codex discover epfl --incremental --quiet
```

### 5. Failed File Retry Strategy ✅

**Decision**: **Automatic retry with exponential backoff**, then mark as `failed` for manual review.

**Implementation**:
```python
# SourceFile retry properties
{
    "status": "failed",
    "retry_count": 3,
    "last_error": "UnicodeDecodeError: 'utf-8' codec can't decode...",
    "next_retry": "2025-01-16T10:30:00Z",
}

# Retry logic
MAX_RETRIES = 3
BACKOFF_BASE = 2  # hours

def handle_failure(file: SourceFile, error: Exception):
    file.retry_count += 1
    file.last_error = str(error)
    
    if file.retry_count >= MAX_RETRIES:
        file.status = "failed"
        file.next_retry = None
    else:
        backoff = BACKOFF_BASE ** file.retry_count
        file.next_retry = now() + timedelta(hours=backoff)
        file.status = "retry_pending"

# Retry schedule: 2h, 4h, 8h, then fail
```

**CLI Support**:
```bash
# Process retry queue
uv run imas-codex ingest retry epfl

# List failed files for review
uv run imas-codex ingest list epfl --status failed

# Force retry specific files
uv run imas-codex ingest retry epfl --file /path/to/file.py
```

## Additional Recommendations

### 1. Directory Fingerprinting

For quick change detection at directory level:

```python
def dir_fingerprint(path: str) -> str:
    """Generate fingerprint from all file metadata in directory."""
    result = ssh(f"""
        ~/bin/fd -t f . {path} | while read f; do
            stat -c '%s %Y' "$f" 2>/dev/null
        done | sort | md5sum | cut -d' ' -f1
    """)
    return result.strip()

# Store in FacilityPath
{
    "id": "epfl:/home/codes/liuqe",
    "dir_fingerprint": "a1b2c3d4...",
    "last_fingerprint_at": "2025-01-15T10:30:00Z",
}
```

### 2. Composable Pattern Library

Organize patterns as reusable modules:

```yaml
# config/patterns/modules/mdsplus.yaml
mdsplus:
  description: "MDSplus data access patterns"
  patterns:
    - pattern: "MDSplus|mdsplus"
      weight: 1.0
    - pattern: "Tree\\(|openTree"
      weight: 0.9

# config/patterns/modules/imas.yaml
imas:
  description: "IMAS/IDS patterns"
  patterns:
    - pattern: "imas\\.imasdef"
      weight: 1.0

# config/patterns/discovery.yaml
includes:
  - modules/mdsplus.yaml
  - modules/imas.yaml
  - modules/physics.yaml
```

### 3. SourceFile Status State Machine

```
                ┌─────────────────────────────────────┐
                │         STATE MACHINE               │
                │                                     │
discovered ────► scanned ────► queued ────► fetching ─┬──► parsing ────► embedding ────► ready
     │              │              │            │     │        │              │
     │              │              │            └─────┤        └──────────────┤
     │              │              │                  ▼                       ▼
     │              │              │           retry_pending ◄──────────  failed
     │              │              │                  │                       │
     └──────────────┴──────────────┴──────────────────┴───────────────────────┘
                                    (fingerprint change resets to scanned)
```

### 4. Configuration Validation

Add JSON Schema validation for YAML configs:

```python
# imas_codex/config/validate.py

def validate_patterns(config: dict) -> list[str]:
    """Validate pattern configuration."""
    errors = []
    
    for dim, dim_config in config.get("dimensions", {}).items():
        for pattern in dim_config.get("patterns", []):
            if "pattern" not in pattern:
                errors.append(f"Dimension {dim}: missing 'pattern' key")
            if "weight" in pattern and not 0 <= pattern["weight"] <= 1:
                errors.append(f"Dimension {dim}: weight must be 0-1")
    
    return errors
```

## Appendix: Current Schema (Relevant Nodes)

```yaml
FacilityPath:
  properties:
    - id (required)
    - path (required)
    - facility_id (required)
    - status: discovered|explored|skipped|stale
    - path_type: code_directory|data_directory|...
    - interest_score: 0.0-1.0
    - patterns_found: list[str]
    - notes: str
    - file_count: int
    - files_ingested: int
    - last_examined: datetime

SourceFile:
  properties:
    - id (required)
    - path (required)
    - facility_id (required)
    - status: discovered|ready|failed|fetching|embedding
    - interest_score: 0.0-1.0
    - patterns_matched: list[str]
    - code_example_id: str (link to CodeExample)
    - started_at: datetime
    - completed_at: datetime

CodeExample:
  properties:
    - id (required)
    - facility_id (required)
    - title: str
    - description: str
    - language: str
    - author: str
    - source_file: str
    - ingested_at: datetime
```
