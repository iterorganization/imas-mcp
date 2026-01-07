# Autonomous Discovery Agents

> **Status**: Approved (January 2026)
> **Scope**: ReAct agents for autonomous facility discovery, TDI ingestion, and IMAS mapping

## Executive Summary

This document describes the architecture for autonomous ReAct agents that systematically
explore fusion facilities, ingest content, and build a unified knowledge graph. The key
insight is the distinction between two LLM contexts:

| Context | Token Source | Duration | Human Involvement |
|---------|--------------|----------|-------------------|
| **VS Code Chat** | Cursor subscription | Minutes | Continuous |
| **ReAct Agents** | OpenRouter (budgeted) | Hours/days | Launch and monitor |

Both contexts use the **same tools, schemas, and prompts**. The difference is orchestration:
ReAct agents loop autonomously with budget controls; VS Code chat requires human steering.

## Design Principles

1. **Equivalence** - A VS Code chat session following prompts achieves same result as ReAct agent
2. **Graph-driven** - All state in Neo4j; agents query graph to know what's done/pending
3. **Incremental** - Every operation persists immediately; crash-safe and resumable
4. **Budgeted** - Time and cost limits prevent runaway execution
5. **Hierarchical** - Ordered phases with dependencies; each phase completes before next begins

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
│  - Recursive directory crawl                                            │
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
│  - Link to TreeNodes                                                    │
│  Output: TDIFunction nodes, CodeChunk embeddings                        │
│                                                                         │
│  Phase 5: Code Ingestion (hours-days)                                   │
│  ─────────────────────────────────────                                  │
│  imas-codex ingest run <facility> --budget 24h                          │
│  - Fetch queued SourceFiles                                             │
│  - Chunk with LlamaIndex                                                │
│  - Generate embeddings                                                  │
│  - Extract DataReferences, link to TreeNodes                            │
│  Output: CodeChunk nodes with embeddings                                │
│                                                                         │
│  Phase 6: Enrichment (hours-days)                                       │
│  ─────────────────────────────────                                      │
│  imas-codex agent enrich --tree results --budget 12h                    │
│  - LLM generates physics descriptions                                   │
│  - Physics domain classification                                        │
│  - Unit validation                                                      │
│  Output: Enriched TreeNode metadata                                     │
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

## CLI Structure

```
imas-codex
├── agent                              # ReAct agents (autonomous, long-running)
│   │
│   ├── discover                       # Discovery agent group
│   │   ├── infra <facility>           # Phase 1: Infrastructure
│   │   │   --dry-run                  # Preview without changes
│   │   │   --verbose                  # Show agent reasoning
│   │   │
│   │   ├── paths <facility>           # Phase 2: Directory discovery
│   │   │   --root PATH                # Starting directory (required first run)
│   │   │   --max-depth INT            # Max recursion depth (default: 6)
│   │   │   --exclude PATTERN          # Glob patterns to skip (repeatable)
│   │   │   --budget DURATION          # Time budget (e.g., "4h", "30m")
│   │   │   --max-cost FLOAT           # Cost cap in USD
│   │   │   --resume                   # Continue from last checkpoint
│   │   │
│   │   ├── files <facility>           # Phase 3: File discovery
│   │   │   --patterns TEXT            # Content patterns (comma-separated)
│   │   │   --extensions TEXT          # File extensions (default: py,m,f90,fun)
│   │   │   --budget DURATION
│   │   │   --resume
│   │   │
│   │   ├── tdi <facility>             # Phase 4: TDI function ingestion
│   │   │   --tdi-root PATH            # TDI directory (or from facility config)
│   │   │   --link-trees               # Link to existing TreeNodes
│   │   │   --budget DURATION
│   │   │   --resume
│   │   │
│   │   └── status <facility>          # Show discovery progress
│   │
│   ├── enrich [PATHS...]              # Phase 6: Existing enrichment (enhanced)
│   │   --budget DURATION              # NEW: Time budget
│   │   --max-cost FLOAT               # NEW: Cost cap
│   │   --resume                       # NEW: Resume from checkpoint
│   │
│   ├── map <facility>                 # Phase 7: IMAS mapping discovery
│   │   --min-confidence FLOAT         # Threshold (default: 0.7)
│   │   --budget DURATION
│   │   --resume
│   │
│   ├── status <facility>              # Overall agent progress
│   ├── resume <run-id>                # Resume interrupted run
│   └── run <task>                     # Ad-hoc agent task (existing)
│
├── ingest                             # Deterministic pipelines (no LLM reasoning)
│   ├── run <facility>                 # Phase 5: Process queued files
│   ├── queue <facility> [PATHS]       # Existing
│   ├── status <facility>              # Existing
│   └── list <facility>                # Existing
│
├── neo4j                              # Database operations (existing)
└── facilities                         # Configuration (existing)
```

## New MCP Tools

These tools are used by both VS Code chat (via MCP) and ReAct agents:

### `get_exploration_state`
```python
def get_exploration_state(facility: str, root_path: str | None = None) -> dict:
    """Get current exploration state for a facility."""
    # Returns: paths_discovered, paths_complete, paths_pending, 
    #          files_queued, files_ingested, coverage_pct, recommended_next
```

### `mark_path_explored`
```python
def mark_path_explored(facility: str, path: str, status: str, 
                       file_count: int | None = None) -> dict:
    """Mark a FacilityPath as explored (prevents re-scanning)."""
```

### `get_agent_progress`
```python
def get_agent_progress(run_id: str | None = None, facility: str | None = None) -> dict:
    """Get progress for an agent run (or most recent if no run_id)."""
```

### `checkpoint_agent_run`
```python
def checkpoint_agent_run(run_id: str, last_item_id: str, 
                         items_processed: int, estimated_cost: float) -> dict:
    """Save checkpoint for resumability."""
```

## TDI Function Ingestion

TDI functions are ingested using LlamaIndex, following the same pattern as code ingestion:

1. **Discovery** - SSH find .fun files, create SourceFile nodes
2. **Fetch** - SSH cat to retrieve content
3. **Parse & Chunk** - Custom TDI parser for LlamaIndex
4. **Embed** - Generate embeddings for code chunks
5. **Extract Metadata** - Function name, parameters, tree paths accessed
6. **Create Nodes** - TDIFunction nodes linked to TreeNodes via ACCESSES

## Schema Updates Required

### New: AgentRun

```yaml
AgentRun:
  description: Tracks a long-running agent execution for resumability and audit
  attributes:
    id:
      identifier: true
    facility_id:
      range: Facility
      required: true
    agent_type:
      description: discover-paths, discover-files, discover-tdi, enrich, map
      required: true
    started_at:
      range: datetime
      required: true
    status:
      description: running, paused, completed, failed
      required: true
    budget_seconds:
      range: integer
    budget_cost:
      range: float
    items_processed:
      range: integer
    last_item_id:
      description: For resumption
    error_message:
      description: If status=failed
```

### PathStatus Addition

```yaml
# Add to PathStatus enum
complete:
  description: Fully explored, all files queued, ready for next phase
```

## Implementation Phases

### Phase 0: Schema & Foundation (2-3 days)
1. Add `AgentRun` class to `facility.yaml`
2. Add `complete` to `PathStatus` enum
3. Implement 4 new MCP tools
4. Regenerate Pydantic models
5. Add tools to MCP server

### Phase 1: Discovery Agents (1 week)
1. Create `imas_codex/agents/discover.py`
2. Add `agent discover` command group to CLI
3. Implement `agent discover infra/paths/files/status` commands
4. Budget controls and checkpoint/resume logic

### Phase 2: TDI Ingestion (1 week)
1. Create TDI parser for LlamaIndex
2. Implement `agent discover tdi` command
3. Link TDIFunctions to TreeNodes

### Phase 3: IMAS Mapping (1 week)
1. Implement `agent map` command
2. Confidence scoring function
3. Create IMASMapping nodes

### Phase 4: Budget & Monitoring (3-5 days)
1. Add `--budget` and `--max-cost` to `agent enrich`
2. Implement `agent status` and `agent resume`

### Phase 5: Multi-Facility (1 week)
1. Create JET facility configuration
2. Run Phase 1 infra discovery on JET
3. Validate pipeline on JET

## Success Metrics

| Metric | Target |
|--------|--------|
| TDI functions ingested | >500 |
| TreeNodes with accessor | >50% |
| IMAS mappings discovered | >200 |
| Code files queued | >2000 |
| Agent resume success | >95% |
| Multi-facility support | 2+ (JET + EPFL) |
