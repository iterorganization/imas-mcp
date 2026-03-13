# Signal Scanner Diagnostics & MCP Logs Tool

## Context

The `imas-codex discover signals` CLI runs a multi-worker pipeline (scan → enrich → check)
with a Rich TUI dashboard. During long-running ingestion jobs (e.g. JET at 8+ hours),
operators need better diagnostic information to identify stalls, failures, and performance
bottlenecks. Additionally, since these processes run on remote hosts (iter-login, tcv, etc.),
there's no way to inspect logs without SSH access.

## Phase 1: Enhanced Scanner Progress Streaming (small, implement next)

### Problem
The SCAN phase shows a single "scanning ppf" or "connecting..." status. For facilities
with multiple scanners (JET has wiki, ppf, mdsplus, jpf, device_xml), there's no visibility
into which scanner is active, which have completed, or if one is blocked on SSH.

### Implementation
1. **Per-scanner status in seed_worker**: Track scanner start/end times in `discover_stats`
   metadata. Report per-scanner status through the `on_progress` callback:
   ```
   SCAN x5 ━━━━━━━━━━━━━━━━━  46,868  100%  2M/s
   wiki✓  ppf✓ 5,204  mdsplus✓  jpf: subsystem DA 3/26  device_xml✓
   ```

2. **SSH connection status**: The "connecting..." stream appears when the extract worker
   is waiting for SSH. Add a timing annotation: `connecting (3.2s)` → `connecting (12.5s)`
   so stalls are immediately visible. Threshold at 30s to flag: `connecting (45s ⚡)`.

3. **Scanner timing in stats**: Add per-scanner elapsed time to the STATS section:
   ```
   SCANNERS  wiki:2.1s  ppf:8.4s  mdsplus:0.3s  jpf:45.2s  device_xml:12.1s
   ```

### Files to modify
- `imas_codex/discovery/signals/parallel.py` — seed_worker progress callbacks
- `imas_codex/discovery/signals/progress.py` — scanner status display row
- `imas_codex/discovery/base/progress.py` — add scanner timing support to base

## Phase 2: Worker Health Indicators (medium)

### Problem
When a CHECK worker hits repeated SSH timeouts or MDSplus segfaults, the display shows
"failed" on the current signal but doesn't indicate systemic issues. An operator can't
tell the difference between "1 signal failed" and "every signal is failing".

### Implementation
1. **Error rate tracking**: Add rolling error rate to `WorkerStats`:
   - `error_rate_1m`: errors per second over last 60s
   - `consecutive_errors`: count of consecutive failures
   - Display as color coding: green (<5%), yellow (5-25%), red (>25%)

2. **Backoff visibility**: When supervised workers enter backoff, show remaining time:
   ```
   CHECK x4 ━━━━━━━━━  36,619  84%  1.2/s
   jet:magnetic_field_diagnostics/da_cqy-c402  failed: SSH timeout
   workers: 2 active, 1 backoff (23s), 1 failed
   ```

3. **SSH health summary**: Surface connection health from ServiceMonitor:
   ```
   SERVERS  graph:titan  embed:titan  models:iter  ssh:jet (avg 2.3s, fail 3/100)
   ```

### Files to modify
- `imas_codex/discovery/base/progress.py` — WorkerStats rolling error rate
- `imas_codex/discovery/base/supervision.py` — expose backoff remaining time
- `imas_codex/discovery/signals/progress.py` — render health indicators

## Phase 3: MCP Logs Tool (medium-large)

### Problem
Logs are stored at `~/.local/share/imas-codex/logs/<command>_<facility>.log` on whichever
host is running the CLI. When running from the ITER host, an operator on their laptop
has no way to inspect logs without SSH. MCP tools run in the same process as the agent,
which may or may not be the same host as the CLI process.

### Feasibility Assessment
**Strongly feasible.** The logging infrastructure is well-structured:
- Consistent naming: `<command>_<facility>.log` (e.g., `signals_jet.log`, `wiki_tcv.log`)
- Standard location: `~/.local/share/imas-codex/logs/`
- Rotating file handler with 10MB limit, 3 backups
- DEBUG level in files, WARNING on console
- Structured format: `%(asctime)s %(levelname)-8s %(name)s: %(message)s`

### Implementation Plan

#### Tool 1: `get_logs` (read logs)
Register on the `agents` MCP server (`imas_codex/llm/server.py`):
```python
@self.mcp.tool()
def get_logs(
    command: str = "signals",
    facility: str | None = None,
    lines: int = 100,
    level: str = "WARNING",
    grep: str | None = None,
    since: str | None = None,  # e.g., "1h", "30m", "2024-03-13T10:00"
) -> str:
    """Read imas-codex log files with filtering.
    
    Reads from ~/.local/share/imas-codex/logs/<command>_<facility>.log.
    Supports level filtering, grep, and time-based filtering.
    """
```

#### Tool 2: `list_logs` (discover available logs)
```python
@self.mcp.tool()
def list_logs() -> str:
    """List available log files with sizes and last-modified times."""
```

#### Tool 3: `tail_logs` (stream recent entries)
```python
@self.mcp.tool()
def tail_logs(
    command: str = "signals",
    facility: str | None = None,
    lines: int = 50,
) -> str:
    """Get the most recent log entries (tail -n)."""
```

### Remote Log Access
For logs on remote hosts, leverage the existing `run_python_script()` infrastructure:
```python
# When log file doesn't exist locally, try SSH
if not log_file.exists() and ssh_host:
    output = run_python_script(
        "read_logs.py",
        {"log_path": str(log_file), "lines": lines, "level": level},
        ssh_host=ssh_host,
    )
```

#### Files to create/modify
- `imas_codex/llm/server.py` — register new tools
- `imas_codex/cli/logging.py` — add `read_log()`, `list_logs()` utility functions
- `imas_codex/remote/scripts/read_logs.py` — remote log reader script

## Phase 4: Log Consistency Improvements (small, prerequisite for Phase 3)

### Current State
Log naming is consistent (`<command>_<facility>.log`) but log content quality varies:
- Workers log at different verbosity levels
- Some workers log batch IDs, others don't
- Error messages vary in structure (some include signal IDs, some don't)

### Improvements
1. **Structured log fields**: Add worker name and batch ID to log format:
   ```
   2026-03-13 10:15:23 INFO     check_worker_2 [batch=abc123]: checked 20 signals (18 success, 2 failed)
   ```

2. **Consistent error logging**: All workers should log errors with:
   - Worker name, batch ID, signal ID
   - Error classification (infrastructure vs application)
   - Retry count / max retries

3. **Graph state snapshots**: Periodically log graph counts (every 5 min):
   ```
   2026-03-13 10:20:00 INFO     signals_jet: SNAPSHOT total=46868 discovered=3410 enriched=43458 checked=36619 pending_check=6759
   ```

## Implementation Order

1. **Phase 1** (small) — Enhanced scanner progress. Immediate value for operators.
2. **Phase 4** (small) — Log consistency. Prerequisite for Phase 3.
3. **Phase 2** (medium) — Worker health indicators. Reduces debugging time.
4. **Phase 3** (medium-large) — MCP logs tool. Full remote observability.
