# Ingestion Workflows

Code ingestion pipeline for processing discovered source files into the knowledge graph.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  SCOUT (LLM)    │     │   GRAPH (Neo4j) │     │    CLI (User)   │
│                 │     │                 │     │                 │
│  ssh + rg/fd    │────▶│  SourceFile     │────▶│  imas-codex     │
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
python("""
ingest_nodes("SourceFile", [
    {"id": "epfl:/home/codes/liuqe.py", 
     "path": "/home/codes/liuqe.py",
     "facility_id": "epfl", 
     "status": "discovered", 
     "interest_score": 0.8}
])
""")
```

### Via CLI

```bash
# Queue specific files
uv run imas-codex ingest queue epfl /path/a.py /path/b.py

# Pipe from SSH search
ssh epfl 'rg -l "equilibrium|IMAS" /home/codes -g "*.py"' | \
  uv run imas-codex ingest queue epfl --stdin

# From file list
uv run imas-codex ingest queue epfl -f files.txt
```

## Run Ingestion

```bash
# Check queue status
uv run imas-codex ingest status epfl

# Process discovered files
uv run imas-codex ingest run epfl

# Process high-priority only
uv run imas-codex ingest run epfl --min-score 0.7

# Process more files
uv run imas-codex ingest run epfl -n 500

# Preview without processing
uv run imas-codex ingest run epfl --dry-run

# Force re-ingestion
uv run imas-codex ingest run epfl --force
```

## Monitor Status

```bash
# List discovered files
uv run imas-codex ingest list epfl

# List failed files
uv run imas-codex ingest list epfl -s failed

# List ingested files
uv run imas-codex ingest list epfl -s ingested
```

## Key Features

- **Graph-driven**: Scouts discover files, CLI processes the queue
- **Deduplication**: Already-ingested files are automatically skipped
- **Interrupt-safe**: Partial ingestion can be resumed safely
- **MDSplus linking**: Extracted paths are linked to `TreeNode` entities

## Recovery from Failures

If ingestion is interrupted:
1. `SourceFile` nodes retain their status
2. Rerun `uv run imas-codex ingest run epfl` to continue
3. Already-ingested files are skipped automatically

For failed files:
```bash
# List failures
uv run imas-codex ingest list epfl -s failed

# Retry failed files
uv run imas-codex ingest run epfl --retry-failed
```

## What Gets Extracted

The pipeline automatically extracts:
- MDSplus tree paths (`\TREE::NODE`)
- TDI function calls
- IDS references (`core_profiles`, `equilibrium`, etc.)
- Import statements and dependencies

Do NOT fabricate `patterns_matched` metadata - let the pipeline do real extraction.
