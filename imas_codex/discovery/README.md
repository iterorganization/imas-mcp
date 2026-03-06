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
uv run imas-codex discover paths tcv --cost-limit 5.0

# Phase-separated discovery (scan requires SSH, score is offline)
uv run imas-codex discover paths tcv --scan-only    # Fast SSH enumeration
uv run imas-codex discover paths tcv --score-only   # Offline LLM scoring

# Check discovery progress
uv run imas-codex discover status tcv

# Manage documentation sources
uv run imas-codex discover sources list
uv run imas-codex discover sources add --name "EPFL Wiki" --url https://... --facility tcv

# Clear all paths (for fresh start)
uv run imas-codex discover clear tcv --force
```

## CLI Structure

```
imas-codex discover
├── paths <facility>    # Scan and score directory structure
│   ├── --scan-only     # SSH enumeration only (no LLM scoring)
│   ├── --score-only    # LLM scoring only (no SSH, offline)
│   ├── --cost-limit    # Maximum LLM spend in USD
│   ├── --focus         # Natural language focus for scoring
│   └── --threshold     # Minimum score to expand paths
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
- `discover wiki` can run in parallel with paths (for wiki sources)

## Path Discovery State Machine

States are defined in the LinkML schema (`schemas/common.yaml`) as the `PathStatus` enum.
All code imports `PathStatus` from generated models - no hardcoded strings.

```python
from imas_codex.graph.models import PathStatus
# Use: PathStatus.discovered.value, PathStatus.scanned.value, etc.
```

### Graph State Pattern

**Key principle**: Status enums represent durable states only. Transient worker coordination
uses the `claimed_at` timestamp property:

- **Claim**: Worker atomically sets `claimed_at = datetime()` on unclaimed/expired paths
- **Complete**: Worker updates status AND clears `claimed_at = null`
- **Orphan recovery**: Expired claims (`claimed_at < now - 10min`) become reclaimable

This pattern is consistent across all discovery pipelines (paths, wiki, signals).

### Score-Gated Expansion

Children are only created for paths that score highly. This prevents graph pollution
from low-value directories:

1. **First Scan**: Enumerate directory, set `status='scanned'`, no children created
2. **Score**: LLM evaluates path, sets `score` and `should_expand` flag
3. **Expansion Scan**: If `should_expand=true`, scanner re-claims path, creates children

```
                       FIRST DISCOVERY
                       
   [seed paths]                                 
        │                                        
        ▼                                        
   discovered ─────SCAN────► scanned ─────SCORE────► scored
   (score=NULL)              (score=NULL)           (should_expand=T/F)
                                                         │
                                                         │
             ┌───────────────────────────────────────────┘
             │
             ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                      POST-SCORE ROUTING                     │
   │                                                             │
   │  should_expand=true?                                        │
   │    YES → Scanner re-claims, creates children                │
   │          Parent stays 'scored', children → 'discovered'     │
   │    NO  → Terminal. No children created.                     │
   │                                                             │
   └─────────────────────────────────────────────────────────────┘
```

### States

| State | Description |
|-------|-------------|
| `discovered` | Path found, awaiting enumeration |
| `scanned` | Enumerated with file/dir counts, awaiting score |
| `scored` | LLM scored, may be re-claimed for expansion |
| `skipped` | Low value (score < 0.2) or excluded |
| `failed` | Error during processing |
| `stale` | Path may have changed, needs re-discovery |

### State Transitions with Claimed-At Coordination

```
                            SCAN WORKER
     ┌──────────────────────────────────────────────────────────┐
     │                                                          │
     │  Claims (sets claimed_at, status unchanged):             │
     │    1. status='discovered' AND score IS NULL (first scan) │
     │    2. status='scored' AND should_expand=true (expansion) │
     │                                                          │
     │  Order: First by unscored (breadth-first by depth)       │
     │         Then by expansion (score DESC, high-value first) │
     └──────────────────────────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
    First Scan            Expansion              Error
         │                     │                     │
      scanned            (stays scored)           skipped
   (no children)       (children created)
         │
         ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                       SCORE WORKER                          │
   │                                                             │
   │  Claims (sets claimed_at): status='scanned', score IS NULL  │
   │  Result: Sets score, should_expand → status='scored'        │
   │          Clears claimed_at on completion                    │
   │                                                             │
   │  If should_expand=true:                                     │
   │    → Path becomes eligible for expansion scan               │
   │    → Scanner picks up, creates children                     │
   │                                                             │
   └─────────────────────────────────────────────────────────────┘
```

### Orphan Recovery

Orphan recovery is built into the claim queries via timeout check. If a worker crashes,
its claimed paths automatically become available when `claimed_at` expires (10 min):

```cypher
-- Claim query includes timeout check (paths with expired claims are re-claimable)
MATCH (p:FacilityPath {status: 'discovered'})
WHERE p.claimed_at IS NULL 
   OR p.claimed_at < datetime() - duration('PT600S')
SET p.claimed_at = datetime()
RETURN p.id, p.path
```

On CLI startup, all orphaned claims are cleared immediately since only one CLI process
runs per facility:

```cypher
-- Reset all orphaned claims
MATCH (p:FacilityPath {facility_id: $facility})
WHERE p.claimed_at IS NOT NULL
SET p.claimed_at = null
```

### Deadlock Avoidance in Claim Queries

Concurrent workers claiming nodes from the same pool cause Neo4j deadlocks unless
three patterns are applied together. All claim functions across all pipelines (paths,
code, wiki, signals, mdsplus) must follow this pattern.

**Why deadlocks happen:** When multiple workers execute `SET n.claimed_at = datetime()`
on overlapping node sets, Neo4j acquires write locks in node-ID order. Deterministic
`ORDER BY` (e.g., `ORDER BY score DESC`) causes all workers to lock the same rows in the
same order, creating lock convoys that escalate to deadlocks.

**Required patterns** (all three, always together):

1. **`@retry_on_deadlock()`** — decorator from `discovery/base/claims.py`. Catches
   `TransientError` and retries with exponential backoff + jitter (up to 5 attempts).

2. **`ORDER BY rand()`** — randomizes which nodes each worker locks first, breaking
   the deterministic lock ordering that causes convoys.

3. **`claim_token` two-step verify** — SET a UUID token in step 1, read back by token
   in step 2. Prevents race conditions where two workers both think they claimed the
   same node.

**Correct pattern:**
```python
from imas_codex.discovery.base.claims import retry_on_deadlock

@retry_on_deadlock()
def claim_items(facility: str, limit: int = 10) -> list[dict]:
    token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query("""
            MATCH (n:MyNode {facility_id: $facility})
            WHERE n.status = 'discovered' AND n.claimed_at IS NULL
            WITH n ORDER BY rand() LIMIT $limit
            SET n.claimed_at = datetime(), n.claim_token = $token
        """, facility=facility, limit=limit, token=token)
        return list(gc.query("""
            MATCH (n:MyNode {claim_token: $token})
            RETURN n.id AS id, n.path AS path
        """, token=token))
```

**Anti-patterns (never do these):**
```python
# BAD: deterministic ordering causes lock convoys
ORDER BY score DESC LIMIT $limit

# BAD: manual retry loop instead of decorator
for attempt in range(3):
    try: ...
    except TransientError: time.sleep(1)

# BAD: no claim_token — two workers can both "claim" the same node
SET n.claimed_at = datetime()
RETURN n.id  # Race: another worker may have claimed between SET and RETURN
```

**Shared infrastructure:** `discovery/base/claims.py` provides `retry_on_deadlock()`,
`claim_items()` factory, `reset_stale_claims()`, `release_claim()`, and
`release_claims_batch()`. Use these instead of writing raw claim Cypher.

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
| `--scan-only` | False | SSH enumeration only, no LLM scoring |
| `--score-only` | False | LLM scoring only, no SSH (offline) |

#### Phase Separation

The `--scan-only` and `--score-only` flags enable phase-separated discovery:

| Phase | Requires | Cost | Use Case |
|-------|----------|------|----------|
| **Scan** | SSH access | Free | Enumerate directories during work hours |
| **Score** | Graph only | LLM $ | Score offline or from CI without SSH |

This is useful when:
- SSH access is intermittent or time-limited
- You want to batch-score overnight without facility access
- You need fast enumeration without waiting for LLM calls

**Workflow:**
```bash
# 1. Fast scan with SSH access (no LLM cost)
uv run imas-codex discover paths iter --scan-only

# 2. Later: score from graph (no SSH needed)
uv run imas-codex discover paths iter --score-only --cost-limit 20.0

# 3. Iterate: new scan expands scored paths above threshold
uv run imas-codex discover paths iter --scan-only
```

**Notes:**
- `--score-only` errors if graph is empty (must scan first)
- `--score-only` only expands paths already scored above threshold
- Flags are mutually exclusive

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
                    │ paths tcv  │  Auto-seeds on first run
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
        │  Sets score                       │
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

## LLM Structured Output

Path scoring uses Pydantic models with LiteLLM's `response_format` for reliable parsing.

### Schema Injection

Prompts in `imas_codex/agentic/prompts/discovery/` use Jinja2 templating with dynamic
schema injection. The scorer prompt includes:

```jinja
{{ scoring_schema_example }}    # JSON example from Pydantic model
{{ scoring_schema_fields }}     # Field descriptions
{{ path_purposes }}             # Enum values from LinkML schema
```

Schema context is loaded lazily via `get_schema_for_prompt()`:

```python
from imas_codex.agentic.prompt_loader import render_prompt

# Only loads what the prompt needs (9 keys for scorer, 5 for refiner)
prompt = render_prompt("paths/scorer", {"facility": "tcv", "paths": paths})
```

### Pydantic Models

LLM responses are validated against `DirectoryScoringBatch`:

```python
from imas_codex.discovery.paths.models import DirectoryScoringBatch

# In scorer.py:
response = litellm.completion(
    model=model_id,
    response_format=DirectoryScoringBatch,  # Enforced by LLM
    messages=[{"role": "system", "content": system_prompt}, ...],
)
batch = DirectoryScoringBatch.model_validate_json(response.content)
```

### Adding New Prompts

1. Create prompt in `imas_codex/agentic/prompts/` (use appropriate subdir: `paths/`, `code/`, `signals/`)
2. Define Pydantic model in `imas_codex/discovery/paths/models.py`
3. Add to `_DEFAULT_SCHEMA_NEEDS` in `prompt_loader.py`:
   ```python
   _DEFAULT_SCHEMA_NEEDS = {
       "paths/scorer": ["score_schema", "score_dimensions", ...],
       "paths/triage": ["scoring_schema", "score_dimensions", ...],
       "your_domain/your_new_prompt": ["your_schema_needs"],
   }
   ```
4. Use `response_format=YourModel` in LiteLLM call

**Never hardcode JSON examples** - use `get_pydantic_schema_json(Model)` to generate
examples from the Pydantic model. This ensures prompts stay in sync with schema changes.

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

## Shared Infrastructure (`discovery/base/`)

All discovery pipelines share common infrastructure in `discovery/base/`. New pipelines
must compose from these modules — never reimplement parallel workers, claim logic,
progress displays, or scoring from scratch.

### Engine (`base/engine.py`)

Single entry point for all parallel discovery pipelines:

```python
from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine

workers = [
    WorkerSpec(name="scan", phase_attr="scan", worker_fn=scan_worker, count=1),
    WorkerSpec(name="score", phase_attr="score", worker_fn=score_worker, count=4),
]
await run_discovery_engine(state, workers, orphan_recovery=orphan_spec)
```

`WorkerSpec` defines: name, phase attribute on state, async worker function, count,
enabled flag, should_stop predicate, on_progress callback, worker group, and extra kwargs.

`run_discovery_engine()` handles: phase disable logic, stop watcher task, worker
registration, orphan recovery setup, supervised loop execution, and cleanup.

### State (`base/state.py`)

Common state dataclass inherited by all pipeline states:

```python
from imas_codex.discovery.base.state import DiscoveryStateBase

@dataclass
class MyPipelineState(DiscoveryStateBase):
    my_custom_field: str = ""
```

Provides: `facility`, `cost_limit`, `deadline`, `stop_requested`, `service_monitor`,
plus properties `deadline_expired`, `total_cost`, `budget_exhausted`, `should_stop()`,
`all_phases`, `all_stats`, and `await_services()`.

### Claims (`base/claims.py`)

Anti-deadlock claim infrastructure (see Deadlock Avoidance section above):

- `retry_on_deadlock()` — decorator with exponential backoff + jitter
- `claim_items(label, facility, where, limit, return_props, token)` — generic claim factory
- `reset_stale_claims(label, facility)` — clear all claims on startup
- `release_claim(label, node_id)` / `release_claims_batch(label, node_ids)` — unclaim on error

### Supervision (`base/supervision.py`)

Worker lifecycle management with structured concurrency:

- `SupervisedWorkerGroup` — manages worker tasks with health monitoring
- `supervised_worker()` — wraps async workers with error handling and restart
- `PipelinePhase` — tracks phase progress (items processed, errors, cost)
- `run_supervised_loop()` — main event loop with graceful shutdown
- `OrphanRecoverySpec` — periodic orphan recovery configuration

### Scoring (`base/scoring.py`)

Shared scoring models and composite functions used by paths and code pipelines:

- `CODE_SCORE_DIMENSIONS` (9 dimensions): imas_relevance, mdsplus_usage, etc.
- `CONTENT_SCORE_DIMENSIONS` (6 dimensions): for wiki/document scoring
- `CodeScoreFields` / `ContentScoreFields` — Pydantic models for LLM structured output
- `max_composite(scores)` — max-based composite (rewards any high signal)
- `purpose_weighted_composite(scores, weights)` — weighted composite for ranking

### Progress (`base/progress.py`)

Rich terminal progress display:

- `BaseProgressDisplay` — live-updating progress bars with scan/score rates
- `WorkerStats` — per-worker statistics (processed, errors, rate)
- `ProgressConfig` — display configuration (refresh rate, bar width)
- Formatting utilities: `format_rate()`, `format_cost()`, `format_duration()`

### CLI Harness (`cli/discover/common.py`)

Shared CLI utilities that handle boilerplate for all `discover` subcommands:

```python
from imas_codex.cli.discover.common import DiscoveryConfig, run_discovery

config = DiscoveryConfig(domain="paths", facility="tcv", ...)
await run_discovery(config, run_pipeline_fn)
```

`DiscoveryConfig` fields: domain, facility, facility_config, service checks
(graph/embed/ssh/auth/model), display mode, refresh intervals, verbose flag.

`run_discovery()` handles: Rich vs plain mode selection, service monitor lifecycle,
SIGINT/SIGTERM shutdown handlers, background graph refresh + ticker tasks.

### Other Shared Modules

| Module | Purpose |
|--------|---------|
| `base/llm.py` | LLM infrastructure: retry with backoff, cost tracking, structured output, `suppress_litellm_noise()` |
| `base/image.py` | Image processing: downsample, content-addressed IDs, VLM scoring with claim patterns |
| `base/embed_worker.py` | Embedding worker for batch vector computation |
| `base/executor.py` | Parallel SSH execution with branch claiming |
| `base/facility.py` | Facility configuration loading via `get_facility()` |
| `base/services.py` | Service health monitoring (graph, embed server, SSH) |
| `base/transfer.py` | File transfer utilities (SCP, content extraction) |

## Deprecation Notice

The following command groups are deprecated and will be removed:

- `imas-codex wiki` → Use `imas-codex discover wiki` instead
- `imas-codex scout` → Use `imas-codex discover` instead
