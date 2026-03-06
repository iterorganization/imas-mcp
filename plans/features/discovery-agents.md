# Discovery Strategy

> **Goal**: Exhaustive, multi-dimensional file discovery across fusion facilities with flexible scoring, autonomous agents, and efficient ingestion.

## Executive Summary

This document outlines the complete discovery and ingestion architecture:

1. **Map Phase**: Exhaustive file enumeration (no LLM, fd/rg tools)
2. **Score Phase**: Multi-dimensional relevance scoring (ReAct agent, LLM-driven)
3. **Ingest Phase**: Optimized pipeline with streaming and local embedding
4. **Enrich Phase**: LLM-driven metadata and relationship discovery
5. **Map Phase**: IMAS mapping with confidence scoring

Key innovations:
- **Multi-dimensional scoring** instead of binary skip/include
- **Persistent pattern filters** in YAML for reproducible searches
- **Tool requirements** as project-level, not facility-specific
- **Graph-driven workflow** using existing SourceFile/FacilityPath nodes
- **Autonomous ReAct agents** with budget controls and checkpointing

---

## Architecture: LLM Context Split

The system distinguishes between two LLM execution contexts:

| Context | Token Source | Duration | Human Involvement |
|---------|--------------|----------|-------------------|
| **VS Code Chat** | Cursor subscription | Minutes | Continuous |
| **ReAct Agents** | OpenRouter (budgeted) | Hours/days | Launch and monitor |

Both contexts use the **same tools, schemas, and prompts**. The difference is orchestration:
ReAct agents loop autonomously with budget controls; VS Code chat requires human steering.

---

## Multi-Facility Strategy

### Target Facilities

| Facility | Machine | Data System | Status |
|----------|---------|-------------|--------|
| EPFL | TCV | MDSplus | ✅ First implementation |
| JET | JET | PPF + MDSplus | 🔜 Next target |
| DIII-D | DIII-D | MDSplus | 🔜 Planned |
| ITER | ITER | IMAS native | 🔜 Future |

### Facility Onboarding Workflow

```
1. Infrastructure Discovery (1 day)
   └── SSH access, tool availability, data system detection

2. Tree Structure Ingestion (1-2 days)
   └── MDSplus/PPF tree enumeration, DataNode creation

3. Code Discovery (1 week)
   └── Map → Score → Ingest pipeline execution

4. Wiki/Documentation (if available)
   └── Portal discovery, page evaluation, chunk embedding

5. IMAS Mapping (ongoing)
   └── Semantic search for equivalents, confidence scoring
```

### Facility-Agnostic Design Principles

1. **No hardcoded paths**: All facility paths in `config/facilities/<facility>.yaml`
2. **Tool abstraction**: Required tools defined at project level, not per-facility
3. **Pattern composition**: Base patterns + facility-specific overrides
4. **Graph isolation**: Each facility in separate AT_FACILITY namespace

---

## Facility Onboarding Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Facility Onboarding Pipeline                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 1: Infrastructure Discovery (minutes)                            │
│  ───────────────────────────────────────────                            │
│  imas-codex agent discover infra <facility>                             │
│  - OS, tools, Python environment                                        │
│  - Data system detection (MDSplus, PPF, UDA)                           │
│  - Schema compatibility check                                           │
│  Output: Facility node, infrastructure YAML                             │
│  STOP if schema gaps detected → update schema first                     │
│                                                                         │
│  Phase 2: Path Discovery (hours)                                        │
│  ──────────────────────────────                                         │
│  imas-codex agent discover paths <facility> --root /home --budget 4h    │
│  - Recursive directory scan                                            │
│  - Pattern matching for code directories                                │
│  - Interest scoring via LLM                                             │
│  Output: FacilityPath nodes with status                                 │
│                                                                         │
│  Phase 3: File Discovery (hours)                                        │
│  ─────────────────────────────                                          │
│  imas-codex agent discover files <facility> --budget 8h                 │
│  - Scan flagged paths for source files                                  │
│  - Queue SourceFile nodes                                               │
│  Output: SourceFile nodes (status=queued)                               │
│                                                                         │
│  Phase 4: TDI Ingestion (hours)                                         │
│  ──────────────────────────────                                         │
│  imas-codex agent discover tdi <facility> --budget 4h                   │
│  - Discover .fun files                                                  │
│  - Parse with LlamaIndex (chunk, embed)                                 │
│  - Create TDIFunction nodes                                             │
│  - Link to DataNodes                                                    │
│  Output: TDIFunction nodes, CodeChunk embeddings                        │
│                                                                         │
│  Phase 5: Code Ingestion (hours-days)                                   │
│  ─────────────────────────────────────                                  │
│  imas-codex ingest run <facility> --budget 24h                          │
│  - Fetch queued SourceFiles                                             │
│  - Chunk with LlamaIndex                                                │
│  - Generate embeddings                                                  │
│  - Extract DataReferences, link to DataNodes                            │
│  Output: CodeChunk nodes with embeddings                                │
│                                                                         │
│  Phase 6: Enrichment (hours-days)                                       │
│  ─────────────────────────────────                                      │
│  imas-codex agent enrich --tree results --budget 12h                    │
│  - LLM generates physics descriptions                                   │
│  - Physics domain classification                                        │
│  - Unit validation                                                      │
│  Output: Enriched DataNode metadata                                     │
│                                                                         │
│  Phase 7: Mapping Discovery (hours)                                     │
│  ──────────────────────────────────                                     │
│  imas-codex agent map <facility> --budget 8h                            │
│  - Semantic search IMAS DD for equivalents                              │
│  - Unit/dimension compatibility checking                                │
│  - Confidence scoring                                                   │
│  Output: IMASMapping nodes                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

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

exclude_paths:
  - ".git"
  - "__pycache__"
  - "node_modules"
  - ".venv"
  - "build"
```

### Implementation Sketch

```python
# imas_codex/discovery/map_agent.py

class MapAgent:
    """No-LLM file discovery agent using remote tools."""
    
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
```

---

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

---

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

### Streaming Architecture

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
    │  6. Link to DataNode/IDS paths                              │
    │  7. Update SourceFile status                                 │
    └─────────────────────────────────────────────────────────────┘
```

---

## CLI Structure

```
imas-codex
├── agent                              # ReAct agents (autonomous, long-running)
│   │
│   ├── discover                       # Discovery agent group
│   │   ├── infra <facility>           # Phase 1: Infrastructure
│   │   ├── paths <facility>           # Phase 2: Directory discovery
│   │   ├── files <facility>           # Phase 3: File discovery
│   │   ├── tdi <facility>             # Phase 4: TDI function ingestion
│   │   └── status <facility>          # Show discovery progress
│   │
│   ├── enrich [PATHS...]              # Phase 6: Metadata enrichment
│   ├── map <facility>                 # Phase 7: IMAS mapping discovery
│   ├── status <facility>              # Overall agent progress
│   └── resume <run-id>                # Resume interrupted run
│
├── ingest                             # Deterministic pipelines (no LLM reasoning)
│   ├── run <facility>                 # Phase 5: Process queued files
│   ├── queue <facility> [PATHS]       # Queue files for ingestion
│   ├── status <facility>              # Ingestion status
│   └── list <facility>                # List queued files
│
├── neo4j                              # Database operations
└── facilities                         # Configuration
```

---

## Tool Requirements

### Project-Level Definition

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
```

---

## Language Support

### Tree-sitter Languages

| Language | Support | Notes |
|----------|---------|-------|
| Python | ✅ | Full AST |
| Fortran | ✅ | Full AST (F77 and F90) |
| C/C++ | ✅ | Full AST |
| MATLAB | ✅ | Full AST |
| Julia | ✅ | Full AST |
| IDL/GDL (.pro) | 🔜 | See [tree-sitter-gdl](https://github.com/iterorganization/tree-sitter-gdl) |
| TDI (.fun) | 🔜 | Planned as tree-sitter-tdi |

---

## Implementation Status

### Phase 0: Foundation ✅
- [x] Core MCP server with tools (python, get_graph_schema, ingest_nodes, private)
- [x] Pattern configuration YAML design
- [x] Strategy document with resolved design decisions
- [x] CodeChunk embedding pipeline operational

### Phase 1: Map Agent 🔄
- [x] SourceFile node creation and status tracking
- [x] CLI: `uv run imas-codex ingest queue/status/run`
- [x] Pattern-based file discovery via SSH + rg
- [ ] Formal `MapAgent` class with fd/rg pipeline
- [ ] Multi-dimensional pattern scoring (config/patterns/*.yaml)

### Phase 2: Score Agent ⬜
- [ ] `ScoreAgent` ReAct loop for file analysis
- [ ] Batch prioritization with budget tracking
- [ ] Code summarization and relationship discovery

### Phase 3: Streaming Ingestion ✅
- [x] Streaming file fetch via SSH
- [x] Tree-sitter parsing for Python
- [x] Parallel embedding with batching
- [x] CodeChunk node creation with AT_FACILITY relationships

### Phase 4: Multi-Facility ⬜
- [ ] Incremental discovery scheduling
- [ ] Second facility onboarding (JET or DIII-D)
- [ ] Documentation and runbooks

---

## Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| TDI functions per facility | >100 | Parsed and linked to DataNodes |
| DataNodes per facility | >10,000 | From tree introspection |
| IMAS mappings per facility | >200 | With confidence scores |
| Code files ingested | >1,000 | Per facility |
| CodeChunks embedded | >5,000 | Per facility |
| Agent resume success | >95% | Crash recovery |
| Facilities onboarded | 3+ | EPFL, JET, DIII-D |

---

## Design Decisions (Resolved)

### File Hashing for Change Detection ✅
**Decision**: Use metadata fingerprinting (size + mtime) for all files, with optional content hashes for high-value files.

### Tree-sitter for Parsing ✅
**Decision**: Use tree-sitter for all supported languages, with regex fallback only for unsupported languages.

### YAML Configuration Structure ✅
**Decision**: Consolidate all YAML into coherent directory structure:
- `schemas/` - LinkML data model definitions (source of truth)
- `definitions/` - Static domain knowledge
- `config/` - Runtime configuration (per-facility)

### Incremental Discovery Schedule ✅
**Decision**: Daily during initial exploration, then weekly, then on-demand.

### Failed File Retry Strategy ✅
**Decision**: Automatic retry with fast exponential backoff (5s, 30s, 2min), then mark as `failed`.
