# Discovery Pipeline

Graph-led facility discovery with parallel scan and score workers.

## Overview

The discovery pipeline provides unified exploration of remote facilities:

- **Paths**: Directory structure with LLM scoring for prioritization
- **Code**: Source files in high-value directories (Python, Fortran, etc.)
- **Docs**: Wiki pages, READMEs, and document artifacts (PDFs, etc.)
- **Data**: MDSplus trees, HDF5 datasets, IMAS databases

All state is stored in Neo4j with atomic state transitions via Cypher queries,
enabling crash recovery and idempotent operations.

## Quick Start

```bash
# Discover directory structure (foundation)
uv run imas-codex discover paths epfl --cost-limit 5.0

# Check discovery progress
uv run imas-codex discover status epfl

# Manage documentation sources
uv run imas-codex discover sources list
uv run imas-codex discover sources add --name "EPFL Wiki" --url https://... --facility epfl

# Clear all paths (for fresh start)
uv run imas-codex discover clear epfl --force
```

## CLI Structure

```
imas-codex discover
├── paths <facility>    # Scan and score directory structure
│   ├── --seed-only     # Only seed roots, don't scan
│   └── -p, --path      # Additional paths to seed
├── code <facility>     # Find source files [PLACEHOLDER]
├── docs <facility>     # Find documentation [PLACEHOLDER]
├── data <facility>     # Find data sources [PLACEHOLDER]
├── status <facility>   # Show discovery statistics
├── inspect <facility>  # Debug view of scanned/scored paths
├── clear <facility>    # Clear paths (reset discovery)
└── sources             # Manage documentation sources
    ├── list            # List all sources
    ├── add             # Add a new source
    ├── rm              # Remove a source
    ├── enable          # Enable a paused source
    └── disable         # Disable a source
```

## Discovery Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PATHS     │     │    CODE     │     │   INGEST    │
│  discover   │────>│  discover   │────>│   (code)    │
│   paths     │     │    code     │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
      │
      │ (parallel)
      ▼
┌─────────────┐     ┌─────────────┐
│    DOCS     │────>│   INGEST    │
│  discover   │     │   (docs)    │
│    docs     │     │             │
└─────────────┘     └─────────────┘
```

Prerequisites:
- `discover paths` must run first to identify high-value directories
- `discover docs` can run in parallel with paths (for wiki sources)

## Path Discovery State Machine

States are defined in the LinkML schema (`schemas/common.yaml`) as the `PathStatus` enum.
All code imports `PathStatus` from generated models - no hardcoded strings.

```python
from imas_codex.graph.models import PathStatus
# Use: PathStatus.discovered.value, PathStatus.listed.value, etc.
```

### States

| State | Type | Description |
|-------|------|-------------|
| `discovered` | Long-lived | Path found, awaiting enumeration |
| `listing` | Transient | Scanner worker active (fallback → discovered) |
| `listed` | Long-lived | Enumerated with file/dir counts, awaiting score |
| `scoring` | Transient | Scorer worker active (fallback → listed) |
| `scored` | Terminal | LLM scored, interest_score set |
| `skipped` | Terminal | Low value (score < 0.2) |
| `excluded` | Terminal | Matched exclusion pattern |
| `stale` | Long-lived | Path may have changed, needs re-discovery |

### State Transitions

```
discovered ──(claim)──> listing ──(complete)──> listed ──(claim)──> scoring
     ↑                     │                                          │
     │                 (timeout)                                      │
     │                     ↓                                          │
     └─────────────── discovered                            ┌─────────┴─────────┐
              │            │            │
         Low value    Good value     Error
              │            │            │
           skipped      scored       listed
                                   (retry)
```

### Orphan Recovery

Transient states (`listing`, `scoring`) have timeouts. When a worker crashes, orphaned
paths are recovered at next discovery run:

```cypher
-- Reset orphaned listing states (> 10 min old)
MATCH (p:FacilityPath {status: 'listing'})
WHERE p.claimed_at < datetime() - duration('PT10M')
SET p.status = 'discovered', p.claimed_at = null

-- Reset orphaned scoring states
MATCH (p:FacilityPath {status: 'scoring'})  
WHERE p.claimed_at < datetime() - duration('PT10M')
SET p.status = 'listed', p.claimed_at = null
```

## CLI Options

### discover paths

```bash
uv run imas-codex discover paths <facility> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--focus TEXT` | (all dims) | Natural language focus for scoring |
| `--cost-limit` | 10.0 | Maximum LLM cost in USD |
| `--limit N` | (none) | Maximum paths to process |
| `--threshold` | 0.7 | Minimum score to expand |
| `--scan-workers` | 1 | SSH scanner workers (single connection) |
| `--score-workers` | 4 | Parallel LLM scorer workers |

### discover status

```bash
uv run imas-codex discover status <facility> [options]
```

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--domain [paths\|code\|docs\|data]` | Show detailed status for specific domain |

### discover sources

```bash
uv run imas-codex discover sources add [options]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--name, -n` | Yes | Human-readable name |
| `--url, -u` | Yes | Base URL of the source |
| `--portal, -p` | No | Portal/starting page |
| `--type, -t` | No | Site type (mediawiki, confluence, readthedocs, etc.) |
| `--auth` | No | Auth method (none, ssh_proxy, basic, session) |
| `--facility, -f` | No | Link to facility |
| `--credential-service` | No | Keyring service name |

## Architecture

```
                    ┌─────────────┐
                    │  discover   │  Single CLI command
                    │ paths epfl  │  Auto-seeds on first run
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
        └──────────────────┬──────────────────┘
                           │
                    discovered → listed
                           │
        ┌──────────────────▼──────────────────┐
        │         3. SCORER WORKERS           │
        │  Claims listed paths atomically     │
        │  Batches for LLM scoring            │
        │  Sets interest_score                │
        └──────────────────┬──────────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         Low value    Good value     Error
              │            │            │
           skipped      scored       listed
                                   (retry)
```

## Progress Display

The discovery command shows live progress with scan/score rates:

```
SCAN  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  657 @ 1.6/s
SCORE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  157 @ 0.6/s
COST  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  $0.42/$1.00
STATS frontier=3527  depth=2  skipped=51  excluded=54
```

## Configuration

### Exclusion Patterns

Defined in `config/patterns/exclude.yaml`:

- VCS directories: `.git`, `.svn`, `.hg`
- Python: `__pycache__`, `.venv`, `*.egg-info`
- Build outputs: `build`, `dist`, `node_modules`
- System paths: `/proc`, `/sys`, `/dev`, `/tmp`

### Scoring Patterns

Defined in `config/patterns/scoring/`:

- `base.yaml` - thresholds, dimension weights
- `data_systems.yaml` - IMAS, MDSplus, HDF5, NetCDF patterns
- `physics.yaml` - equilibrium, transport, MHD patterns

## Performance

| Operation | Throughput | Notes |
|-----------|------------|-------|
| Light scan (enumeration only) | 65-100 paths/s | Batch 200 paths per SSH call |
| Scan with rg patterns | 6-8 paths/s | Pattern detection adds ~15s overhead |
| LLM scoring | 0.35-0.45 paths/s | Batch 100 dirs, Sonnet 4.5, 32k max_tokens |

### Batch Size Optimization

SSH overhead dominates scan time (~5s constant). Larger batches amortize this:

| Batch Size | Time (s) | Paths/s | Notes |
|------------|----------|---------|-------|
| 25 | 5.2 | 4.8 | Small batch |
| 100 | 4.8 | 20.9 | Good balance |
| 200 | 4.8 | 41.6 | Recommended |
| 500 | 5.0 | 100.3 | Maximum tested |

LLM scoring throughput is constant (~0.4/s) but cost scales with batch:

| Batch Size | Time (s) | Cost ($) | Tokens | Notes |
|------------|----------|----------|--------|-------|
| 10 | 34 | 0.046 | 8k | Small batch |
| 25 | 81 | 0.101 | 15k | Minimum viable |
| 50 | 132 | 0.182 | 26k | Good balance |
| 100 | 258 | 0.342 | 45k | New default |

Key optimizations:
- **Batch SSH calls**: Same ~5s overhead whether 1 or 500 paths
- **max_tokens=32000**: Supports 50+ paths per LLM call without truncation
- **Persistent workers**: No Python restart between batches
- **Atomic claims**: Cypher transactions prevent race conditions

## Module Structure

| File | Purpose |
|------|---------|
| `frontier.py` | Graph queries for stats and work queues |
| `parallel.py` | Async scanner/scorer workers |
| `parallel_progress.py` | Rich live progress display |
| `progress.py` | DiscoveryStats dataclass |
| `scanner.py` | SSH batch enumeration |
| `scorer.py` | LLM path evaluation |
| `config.py` | Pattern loading from YAML |
| `models.py` | Pydantic models for scoring |
| `executor.py` | Discovery loop orchestration |
| `facility.py` | Facility configuration access |

## Configuration Constants

Key constants in `parallel.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `ORPHAN_TIMEOUT_MINUTES` | 10 | Reset orphaned claims after 10 min |
| `scan_batch_size` | 200 | Paths per SSH call |
| `score_batch_size` | 100 | Paths per LLM call |

Key constants in `scorer.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_RETRIES` | 3 | Retry count for rate limits |
| `RETRY_BASE_DELAY` | 2.0 | Base delay in seconds (doubles each retry) |
| `max_tokens` | 32000 | LLM output token limit |
| `SCORE_WEIGHTS` | code=1.0, data=0.8, imas=1.2 | Dimension weights |

## Deprecation Notice

The following command groups are deprecated and will be removed:

- `imas-codex wiki` → Use `imas-codex discover docs` instead
- `imas-codex scout` → Use `imas-codex discover` instead
