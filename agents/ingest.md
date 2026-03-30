# Ingestion Workflows

Code ingestion pipeline for processing discovered source files into the knowledge graph.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  SCOUT (LLM)    │     │   GRAPH (Neo4j) │     │    CLI (User)   │
│                 │     │                 │     │                 │
│  run() + rg/fd  │── ─▶│  SourceFile     │────▶│  imas-codex     │
│  ingest_nodes() │     │  status=        │     │  ingest run     │
│                 │     │  discovered     │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## SourceFile Lifecycle

```
discovered ──▶ ingested
    │            │
    │     ┌── failed ◀─┘
    ▼     ▼
stale ◀── (re-scan)
```

| Status | Meaning |
|--------|---------|
| `discovered` | Queued for ingestion |
| `ingested` | Successfully processed |
| `failed` | Processing error |
| `stale` | File may have changed |

## Queue Files for Ingestion

### Via MCP (Preferred)

```python
repl("""
ingest_nodes("SourceFile", [
    {"id": "tcv:/home/codes/liuqe.py", 
     "path": "/home/codes/liuqe.py",
     "facility_id": "tcv", 
     "status": "discovered", 
     "score_composite": 0.8}
])
""")
```

### Via CLI

```bash
# Queue specific files
uv run imas-codex ingest queue tcv /path/a.py /path/b.py

# Pipe from remote search (use run() in MCP or ssh in terminal)
uv run imas-codex tools check tcv  # Ensure rg is available
ssh tcv 'rg -l "equilibrium|IMAS" /home/codes -g "*.py"' | \
  uv run imas-codex ingest queue tcv --stdin

# From file list
uv run imas-codex ingest queue tcv -f files.txt
```

## Run Ingestion

```bash
# Check queue status
uv run imas-codex ingest status tcv

# Process discovered files
uv run imas-codex ingest run tcv

# Process high-priority only
uv run imas-codex ingest run tcv --min-score 0.7

# Process more files
uv run imas-codex ingest run tcv -n 500

# Preview without processing
uv run imas-codex ingest run tcv --dry-run

# Force re-ingestion
uv run imas-codex ingest run tcv --force
```

## Monitor Status

```bash
# List discovered files
uv run imas-codex ingest list tcv

# List failed files
uv run imas-codex ingest list tcv -s failed

# List ingested files
uv run imas-codex ingest list tcv -s ingested
```

## Key Features

- **Graph-driven**: Scouts discover files, CLI processes the queue
- **Deduplication**: Already-ingested files are automatically skipped
- **Interrupt-safe**: Partial ingestion can be resumed safely
- **MDSplus linking**: Extracted paths are linked to `SignalNode` entities

## Recovery from Failures

If ingestion is interrupted:
1. `SourceFile` nodes retain their status
2. Rerun `uv run imas-codex ingest run tcv` to continue
3. Already-ingested files are skipped automatically

For failed files:
```bash
# List failures
uv run imas-codex ingest list tcv -s failed

# Retry failed files
uv run imas-codex ingest run tcv --retry-failed
```

## What Gets Extracted

The pipeline automatically extracts:
- MDSplus tree paths (`\TREE::NODE`)
- TDI function calls
- IDS references (`core_profiles`, `equilibrium`, etc.)
- Import statements and dependencies

Do not fabricate `patterns_matched` metadata - let the pipeline do real extraction.
