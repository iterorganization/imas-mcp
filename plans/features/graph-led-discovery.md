# Graph-Led Discovery Pipeline

**Status**: Planning  
**Phase**: 5 (Discovery Automation)  
**Replaces**: `scout/` agent-based exploration  
**Priority**: High

## Overview

A two-phase discovery architecture that separates **deterministic scanning** (SSH, fast, cheap) from **semantic scoring** (LLM, batched, local). The graph is the single source of truth for exploration state, enabling resumable, idempotent operations.

### Core Principles

1. **Graph-led execution**: Commands query the graph for work, not CLI parameters
2. **Separation of concerns**: SSH ops in `scan`, LLM ops in `score` - no mixing unless necessary
3. **Idempotent operations**: Running `scan` twice with no `score` = second scan completes immediately
4. **Frontier-based expansion**: Score sets `expand_to`; scan discovers children
5. **Grounded scoring**: LLM collects evidence, deterministic function computes score
6. **Multi-faceted interest**: Code, data, docs dimensions scored independently
7. **Local + Remote transparency**: Same interface whether running on facility or via SSH

### Clean Break

Before implementation, **purge existing FacilityPath data**:
```cypher
MATCH (p:FacilityPath) DETACH DELETE p
```

No migration - start fresh with the new schema.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         GRAPH STATE                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  FacilityPath   │  │   SourceFile    │  │    Facility     │  │
│  │  status=scanned │  │  status=queued  │  │  ssh_host       │  │
│  │  expand_to=N    │  │  interest=0.8   │  │  is_local       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
           ▲                                          │
           │ persist (UNWIND)                         │ query
           │                                          ▼
┌──────────┴──────────┐                    ┌──────────────────────┐
│   SCAN PHASE        │                    │   SCORE PHASE        │
│   (SSH/Local)       │                    │   (LLM, Local Only)  │
│                     │                    │                      │
│ • Query frontier    │◄───── cycle ──────│ • Query scanned paths│
│ • run(): fd, rg     │                    │ • Batched LLM prompt │
│ • Collect DirStats  │                    │ • Grounded scoring   │
│ • Create children   │                    │ • Set expand_to      │
│ • Mark as 'scanned' │                    │ • Mark as 'scored'   │
└─────────────────────┘                    └──────────────────────┘
          │                                          │
          └──────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │   DISCOVER COMMAND   │
                    │   (Iterative Loop)   │
                    │                      │
                    │ • scan → score cycle │
                    │ • Budget control     │
                    │ • Rich progress UI   │
                    └──────────────────────┘
```

### State Machine (Simplified)

```
┌─────────┐    ┌──────────┐    ┌─────────┐
│ pending │───▶│ scanned  │───▶│ scored  │
│ (seed)  │    │          │    │         │
└─────────┘    └────┬─────┘    └────┬────┘
                    │               │
                    │               │ expand_to > depth
                    ▼               ▼
               ┌─────────┐    ┌──────────────┐
               │ skipped │    │ new children │
               │         │    │ (pending)    │
               └─────────┘    └──────────────┘
```

**States** (merged `discovered` and `listed` into `pending`):
- `pending`: Path known, awaiting scan (created by seed or expansion)
- `scanned`: Directory contents enumerated, DirStats populated
- `scored`: LLM has evaluated and scored
- `skipped`: Low value, dead-end, or error (with skip_reason)

**Why "scored" not "enriched"?** The LLM phase computes interest scores from evidence - "scored" is more precise than "enriched" which implies adding arbitrary metadata.

## CLI Interface

### Scan Command

```bash
# Graph-led scan - no path/depth parameters
imas-codex scan <facility>

# Options
--dry-run          # Show what would be scanned without executing
--limit N          # Max paths to process this run
--max-sessions N   # Concurrent SSH sessions (default: 4)
--timeout SEC      # Per-directory timeout (default: 30s)
```

**Behavior:**
1. Query graph for `FacilityPath` nodes with `status='pending'` OR `expand_to > depth`
2. If no nodes exist, seed with facility root paths
3. For each path: collect stats via `run()`, persist DirStats, create children, mark `scanned`
4. Exit when no more scannable paths (idempotent)

**Large Directories:** Never skip - they may contain important data. Use timeouts and file count limits instead:
- Count files with `fd -t f . | head -10001 | wc -l` (quick, bounded)
- Skip size calculation if >10k files (set `size_skipped=true`)
- Always persist what we can discover

### Score Command

```bash
# Graph-led scoring - LLM only, no SSH
imas-codex score <facility>

# Options
--dry-run          # Show what would be scored
--batch-size N     # Paths per LLM call (default: 25)
--budget DOLLARS   # Max spend before stopping
--focus QUERY      # Natural language focus (transmitted to LLM)
--threshold SCORE  # Min score to expand (default: 0.7)
--model MODEL      # LLM model (default: claude-sonnet-4-5)
```

**Behavior:**
1. Query graph for `status='scanned' AND score IS NULL`
2. Build batched prompt with context (parent, siblings, DirStats)
3. LLM collects evidence → grounded scorer computes scores
4. For paths with score ≥ threshold AND should_expand: set `expand_to = depth + 1`
5. Mark all as `scored`

**Note:** Score does NOT create children - it only sets `expand_to`. The next `scan` creates the actual children via SSH. This maintains strict separation between SSH and LLM phases.

### Discover Command (Iterative)

```bash
# Combined scan → score cycles with budget control
imas-codex discover <facility>

# Options
--budget DOLLARS       # Max LLM spend (required for safety)
--max-cycles N         # Max scan→score cycles (default: 10)
--focus QUERY          # Natural language focus
--threshold SCORE      # Expansion threshold (default: 0.7)
--max-sessions N       # Concurrent SSH sessions
```

**Behavior:**
1. Loop: scan → score → scan → score...
2. Stop when: budget exhausted OR no new frontier OR max cycles
3. Display rich progress with:
   - Current cycle number
   - Frontier size (pending paths)
   - Completion fraction (can decrease as frontier grows)
   - Accumulated spend / budget
   - Model being used

### Status Command

```bash
# Show discovery state for facility
imas-codex discover status <facility>

# Output:
# Facility: epfl
# Total paths: 1,234
# ├─ Pending:   45 (3.6%)
# ├─ Scanned:  189 (15.3%)
# ├─ Scored:   987 (80.0%)
# └─ Skipped:   13 (1.1%)
# 
# Frontier: 45 paths awaiting scan
# Coverage: 80.0% scored
# High-value paths (score > 0.7): 234
```

## Schema Changes

### FacilityPath Updates

```yaml
# In schemas/facility.yaml - FacilityPath modifications

attributes:
  # Status (simplified)
  status:
    description: Current discovery status
    range: DiscoveryStatus  # pending, scanned, scored, skipped
    required: true

  # Expansion control (set by score, read by scan)
  expand_to:
    description: >-
      Depth to expand children to. Set by score phase when path is valuable.
      Scan creates children when expand_to > depth. Reset to null after expansion.
    range: integer

  # DirStats (inline, not separate class)
  file_type_counts:
    description: JSON map of extension → count. E.g., {"py": 42, "f90": 12}
  total_files:
    description: Total file count
    range: integer
  total_dirs:
    description: Subdirectory count
    range: integer
  total_size_bytes:
    description: Size in bytes (null if skipped)
    range: integer
  size_skipped:
    description: True if size calc was skipped (large dir)
    range: boolean
  has_readme:
    range: boolean
  has_makefile:
    range: boolean
  has_git:
    range: boolean
  patterns_detected:
    description: IMAS/physics patterns found via quick rg search
    multivalued: true

  # Scoring (set by score phase)
  score:
    description: Combined interest score (0.0-1.0)
    range: float
  score_code:
    description: Code interest dimension
    range: float
  score_data:
    description: Data interest dimension
    range: float
  score_imas:
    description: IMAS relevance dimension
    range: float
  description:
    description: One-sentence LLM-generated description
  path_purpose:
    description: >-
      Classified purpose: physics_code, data_files, documentation,
      configuration, build_artifacts, test_files, user_home, system, unknown
  evidence:
    description: JSON evidence collected by LLM for grounded scoring

  # Metadata
  scanned_at:
    description: When directory was scanned
    range: datetime
  scored_at:
    description: When LLM scored this path
    range: datetime
  skip_reason:
    description: Why path was skipped (if status=skipped)
```

### DiscoveryStatus Enum

```yaml
enums:
  DiscoveryStatus:
    description: Discovery lifecycle status
    permissible_values:
      pending:
        description: Awaiting scan (newly seeded or from expansion)
      scanned:
        description: Directory enumerated, DirStats collected
      scored:
        description: LLM evaluated and scored
      skipped:
        description: Excluded (dead-end, error, or low value)
```

## Parallel SSH Execution

### Shared Executor Interface

The `run()` function in `imas_codex/remote/tools.py` already handles local vs SSH transparency. For parallel scanning, we add an async executor:

```python
# imas_codex/discovery/executor.py

"""
Parallel command execution for local and remote facilities.

This module extends remote.tools.run() with async parallel execution.
Commands run either locally or via SSH depending on is_local_facility().

IMPORTANT: This interface must work identically whether:
1. Running on the target facility (local execution)
2. Running from a different machine (SSH execution)

Agents and scripts should use this interface without knowing which mode is active.
The max_sessions parameter limits concurrent SSH connections to avoid overwhelming
remote systems.
"""

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator

from imas_codex.remote.tools import is_local_facility, run as sync_run


@dataclass
class CommandResult:
    """Result of a command execution."""
    path: str
    stdout: str
    stderr: str
    returncode: int
    duration_ms: int


@dataclass
class ParallelExecutor:
    """Execute commands in parallel with session limiting.
    
    Uses asyncio with a semaphore to limit concurrent SSH sessions.
    Works transparently for both local and remote execution.
    
    Args:
        facility: Target facility ID
        max_sessions: Max concurrent executions (default: 4)
        timeout: Per-command timeout in seconds
    
    Example:
        executor = ParallelExecutor(facility="epfl", max_sessions=4)
        async for result in executor.run_batch(commands):
            print(f"{result.path}: {result.returncode}")
    """
    facility: str
    max_sessions: int = 4
    timeout: int = 30
    _semaphore: asyncio.Semaphore = field(init=False)
    
    def __post_init__(self):
        self._semaphore = asyncio.Semaphore(self.max_sessions)
        self._is_local = is_local_facility(self.facility)
    
    async def run_one(self, cmd: str, path: str) -> CommandResult:
        """Execute single command with semaphore limiting."""
        import time
        
        async with self._semaphore:
            start = time.monotonic()
            
            # Use run_in_executor for the blocking sync_run call
            loop = asyncio.get_event_loop()
            try:
                output = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: sync_run(cmd, facility=self.facility, timeout=self.timeout)
                    ),
                    timeout=self.timeout + 5
                )
                returncode = 0
                stderr = ""
                if "[stderr]:" in output:
                    parts = output.split("[stderr]:", 1)
                    output = parts[0].strip()
                    stderr = parts[1].strip() if len(parts) > 1 else ""
                    
            except asyncio.TimeoutError:
                output = ""
                stderr = "Timeout"
                returncode = -1
            except Exception as e:
                output = ""
                stderr = str(e)
                returncode = -1
            
            duration_ms = int((time.monotonic() - start) * 1000)
            
            return CommandResult(
                path=path,
                stdout=output,
                stderr=stderr,
                returncode=returncode,
                duration_ms=duration_ms
            )
    
    async def run_batch(
        self,
        commands: list[tuple[str, str]]  # [(cmd, path), ...]
    ) -> AsyncIterator[CommandResult]:
        """Execute commands in parallel, yielding results as they complete."""
        tasks = [
            asyncio.create_task(self.run_one(cmd, path))
            for cmd, path in commands
        ]
        
        for coro in asyncio.as_completed(tasks):
            yield await coro
```

### Agent Documentation Requirement

Add to `AGENTS.md` under Critical Rules:

```markdown
### Local + Remote Execution Transparency

The discovery system works identically whether running:
1. **On the target facility** (direct local execution)
2. **From a different machine** (SSH execution)

Use `ParallelExecutor` from `imas_codex/discovery/executor.py`:

```python
from imas_codex.discovery.executor import ParallelExecutor

# Works the same locally or via SSH
executor = ParallelExecutor(facility="epfl", max_sessions=4)
async for result in executor.run_batch(commands):
    process(result)
```

**Never assume local execution** - always use the facility-aware interfaces.
```

## Rich Progress Display

Based on wiki prefetch pattern, implement comprehensive progress:

```python
# imas_codex/discovery/progress.py

from dataclasses import dataclass
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table


@dataclass
class DiscoveryStats:
    """Live statistics for discovery progress."""
    # Counts
    total_paths: int = 0
    pending: int = 0
    scanned: int = 0
    scored: int = 0
    skipped: int = 0
    
    # Current operation
    current_phase: str = "idle"  # scan, score, idle
    current_path: str = ""
    
    # Budget tracking (score phase)
    accumulated_cost: float = 0.0
    budget_limit: float | None = None
    model: str = ""
    
    # Cycle tracking (discover command)
    current_cycle: int = 0
    max_cycles: int = 0
    
    @property
    def frontier_size(self) -> int:
        """Paths awaiting scan."""
        return self.pending
    
    @property
    def completion_fraction(self) -> float:
        """Fraction of known paths that are scored."""
        if self.total_paths == 0:
            return 0.0
        return self.scored / self.total_paths
    
    @property
    def budget_fraction(self) -> float:
        """Fraction of budget used."""
        if not self.budget_limit:
            return 0.0
        return self.accumulated_cost / self.budget_limit


class DiscoveryProgressDisplay:
    """Rich progress display for discovery operations.
    
    Shows:
    - Overview panel with counts and coverage
    - Progress bar for current operation
    - Budget tracking (for score/discover)
    - Frontier size indicator
    """
    
    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.stats = DiscoveryStats()
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task_id = None
    
    def _build_overview_panel(self) -> Panel:
        """Build the overview statistics panel."""
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold")
        table.add_column(justify="left")
        table.add_column(justify="right", style="bold")
        table.add_column(justify="left")
        
        # Row 1: Counts
        table.add_row(
            "Total:", f"{self.stats.total_paths:,}",
            "Frontier:", f"[cyan]{self.stats.frontier_size:,}[/cyan]"
        )
        
        # Row 2: Status breakdown
        table.add_row(
            "Scanned:", f"{self.stats.scanned:,}",
            "Scored:", f"[green]{self.stats.scored:,}[/green]"
        )
        
        # Row 3: Coverage and skipped
        coverage_pct = self.stats.completion_fraction * 100
        table.add_row(
            "Skipped:", f"{self.stats.skipped:,}",
            "Coverage:", f"{coverage_pct:.1f}%"
        )
        
        # Row 4: Budget (if applicable)
        if self.stats.budget_limit:
            budget_pct = self.stats.budget_fraction * 100
            table.add_row(
                "Spent:", f"${self.stats.accumulated_cost:.2f}",
                "Budget:", f"${self.stats.budget_limit:.2f} ({budget_pct:.0f}%)"
            )
            table.add_row(
                "Model:", self.stats.model,
                "", ""
            )
        
        # Row 5: Cycle (if in discover mode)
        if self.stats.max_cycles > 0:
            table.add_row(
                "Cycle:", f"{self.stats.current_cycle}/{self.stats.max_cycles}",
                "Phase:", f"[yellow]{self.stats.current_phase}[/yellow]"
            )
        
        return Panel(table, title="Discovery Progress", border_style="blue")
    
    def _build_display(self) -> Group:
        """Build complete display."""
        return Group(
            self._build_overview_panel(),
            self._progress
        )
    
    def __enter__(self):
        """Start live display."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self._task_id = self._progress.add_task("Starting...", total=100)
        
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=4,
        )
        self._live.__enter__()
        return self
    
    def __exit__(self, *args):
        """Stop live display."""
        if self._live:
            self._live.__exit__(*args)
    
    def update(
        self,
        description: str | None = None,
        advance: int = 0,
        total: int | None = None,
        **stats_updates
    ):
        """Update progress and stats."""
        # Update stats
        for key, value in stats_updates.items():
            if hasattr(self.stats, key):
                setattr(self.stats, key, value)
        
        # Update progress bar
        if description:
            self._progress.update(self._task_id, description=description)
        if total is not None:
            self._progress.update(self._task_id, total=total)
        if advance:
            self._progress.advance(self._task_id, advance)
        
        # Refresh display
        if self._live:
            self._live.update(self._build_display())
    
    def refresh_from_graph(self, facility: str):
        """Refresh stats from graph state."""
        from imas_codex.discovery.frontier import get_discovery_stats
        stats = get_discovery_stats(facility)
        self.stats.total_paths = stats["total"]
        self.stats.pending = stats["pending"]
        self.stats.scanned = stats["scanned"]
        self.stats.scored = stats["scored"]
        self.stats.skipped = stats["skipped"]
```

## Prompts

Store prompts in `imas_codex/agentic/prompts/` with clear naming:

### `discovery-scorer.md`

```markdown
---
name: discovery-scorer
description: System prompt for directory scoring in discovery pipeline
used_by: imas_codex.discovery.scorer.DirectoryScorer.score_batch()
model: claude-sonnet-4-5
---

You are analyzing directories at a fusion research facility to determine their value for IMAS code discovery.

## Task

For each directory, collect evidence about its contents and purpose, then provide a structured assessment.

## Evidence to Collect

For each directory, determine:

1. **path_purpose**: Classification (one of):
   - `physics_code`: Simulation or analysis code
   - `data_files`: Scientific data storage
   - `documentation`: Docs, wikis, READMEs
   - `configuration`: Config files, settings
   - `build_artifacts`: Compiled outputs, caches
   - `test_files`: Test suites
   - `user_home`: Personal directories
   - `system`: OS or infrastructure
   - `unknown`: Cannot determine

2. **description**: One sentence describing the directory's likely contents

3. **evidence**: Specific observations:
   - `code_indicators`: Programming files present (list extensions)
   - `data_indicators`: Data files present (list extensions)
   - `imas_indicators`: IMAS-related patterns found
   - `physics_indicators`: Physics domain patterns
   - `quality_indicators`: Project quality signals (readme, makefile, git)

4. **should_expand**: Whether to explore children (true/false)

5. **expansion_reason** or **skip_reason**: Brief justification

{% if focus %}
## Focus Area

Prioritize paths related to: {{ focus }}
{% endif %}

## Response Format

Return a JSON array:
```json
[
  {
    "path": "/home/codes/liuqe",
    "path_purpose": "physics_code",
    "description": "LIUQE equilibrium reconstruction code with Fortran source",
    "evidence": {
      "code_indicators": ["f90", "py"],
      "data_indicators": [],
      "imas_indicators": ["put_slice pattern found"],
      "physics_indicators": ["equilibrium in path"],
      "quality_indicators": ["has_readme", "has_makefile"]
    },
    "should_expand": true,
    "expansion_reason": "High-value equilibrium code with IMAS integration"
  }
]
```
```

## Cross-Facility Learning

**Key benefit of unified graph**: Scores and patterns from one facility inform scoring at others.

### Exploitation Strategies

1. **Path Pattern Transfer**
   ```cypher
   -- Find high-value path patterns across facilities
   MATCH (p:FacilityPath)
   WHERE p.score > 0.8
   WITH split(p.path, '/') AS segments, p.path_purpose AS purpose
   UNWIND range(0, size(segments)-1) AS i
   WITH segments[i] AS segment, purpose, count(*) AS cnt
   WHERE cnt > 5
   RETURN segment, purpose, cnt
   ORDER BY cnt DESC
   ```
   
   Use these patterns to boost scores for new facilities.

2. **Purpose-Based Priors**
   ```cypher
   -- What extensions correlate with high scores?
   MATCH (p:FacilityPath)
   WHERE p.score > 0.7
   UNWIND keys(p.file_type_counts) AS ext
   RETURN ext, avg(p.score) AS avg_score, count(*) AS cnt
   ORDER BY avg_score DESC
   ```

3. **Description Similarity**
   Use embeddings of existing descriptions to find similar new paths:
   ```python
   # Score boost for paths similar to known high-value paths
   similar = semantic_search(
       path.description_context,
       index="facility_path_embedding",
       k=5
   )
   if any(s.score > 0.9 and s.node.score > 0.8 for s in similar):
       score_boost = 0.1
   ```

4. **Shared Code Patterns**
   ```cypher
   -- Cross-facility code with same IMAS patterns
   MATCH (p1:FacilityPath)-[:FACILITY_ID]->(f1:Facility {id: 'epfl'})
   MATCH (p2:FacilityPath)-[:FACILITY_ID]->(f2:Facility {id: 'iter'})
   WHERE p1.patterns_detected IS NOT NULL
     AND any(pat IN p1.patterns_detected WHERE pat IN p2.patterns_detected)
   RETURN p1.path, p2.path, 
          [pat IN p1.patterns_detected WHERE pat IN p2.patterns_detected] AS shared
   ```

## Future Considerations

### Stale Path Re-scanning

**Not a primary concern** - facility filesystems change slowly.

However, track `scanned_at` timestamp. Future enhancement could add:
```bash
imas-codex scan <facility> --refresh-stale DAYS
```

Flag paths where `scanned_at < now() - DAYS` for re-scan. Low priority.

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Scan throughput | 500 dirs/min | Time 500 path scan with parallel SSH |
| Score cost | <$0.01/path | Total LLM cost / paths scored |
| Coverage growth | 5x per cycle | Paths after / paths before |
| Idempotency | 100% | Second scan with no score = 0 work |
| Frontier accuracy | >80% | % of expanded paths yielding high-value children |
