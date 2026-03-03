# Service Management

IMAS Codex runs three independent services that can be started, stopped, and
scaled independently. Each service has its own CLI group with a unified
`start` command that auto-detects the deployment mode.

## Services Overview

| Service | CLI | Scheduler | Resources | Purpose |
|---------|-----|-----------|-----------|---------|
| **Neo4j** | `imas-codex graph` | SLURM (compute) | 4 CPUs, 32 GB, no GPU | Knowledge graph database |
| **Embed** | `imas-codex embed` | SLURM (compute) | 2 CPUs, 16 GB, N GPUs | GPU-accelerated embedding |
| **LLM Proxy** | `imas-codex llm` | systemd (login) | CPU only, ~50 MB | LiteLLM routing + Langfuse tracking |

## Quick Start

```bash
# Start all services
imas-codex graph start           # Neo4j on SLURM
imas-codex embed start           # Embed server on SLURM (4 GPUs default)
imas-codex llm start             # LLM proxy on login node

# Check status
imas-codex graph status          # Neo4j health + SLURM resource usage
imas-codex embed status          # Embed health + GPU info

# Stop all
imas-codex embed stop
imas-codex graph stop
imas-codex llm stop
```

## Deployment Architecture

### Location Resolution

The deploy target is derived from configuration, not hardcoded:

1. `pyproject.toml` sets `[embedding].location = "titan"`
2. `resolve_location("titan")` searches facility YAMLs
3. `iter.yaml` defines `compute_locations.titan: {scheduler: slurm, partition: titan}`
4. Result: `LocationInfo(scheduler="slurm", partition="titan", is_compute=True)`

This means the same `start` command works across different facilities —
only the YAML configuration differs.

### SLURM Services (Neo4j, Embed)

When the location maps to a SLURM partition, `start` submits a batch job:

```
User: imas-codex embed start
  → _is_compute_target() → True (scheduler=slurm)
  → _submit_service_job("codex-embed", ...)
    → generates sbatch script
    → sbatch /tmp/codex-embed.sh
  → _wait_for_job("codex-embed")  ← rich spinner + countdown
    → polls squeue until RUNNING
    → health check: curl /health
  → ✓ codex-embed healthy
```

Inside the SLURM batch script, the process is the last command (`exec`),
so SLURM manages it directly — `scancel` stops it, cgroup enforcement
is automatic.

### Systemd Services (LLM Proxy)

The LLM proxy runs on the login node (needs outbound HTTPS for API
providers). `start` installs and starts a systemd user service:

```
User: imas-codex llm start
  → installs ~/.config/systemd/user/imas-codex-llm.service (if needed)
  → systemctl --user start imas-codex-llm
  → _wait_for_health("LLM proxy", ...)  ← rich spinner
  → ✓ LLM proxy healthy
```

### Foreground Mode

Use `--foreground` / `-f` to run a service directly in the terminal
(for debugging or when running inside a SLURM allocation):

```bash
imas-codex embed start -f                    # Run embed server directly
imas-codex embed start -f --gpu 1            # On specific GPU
imas-codex llm start -f                      # Run LLM proxy directly
imas-codex llm start -f --port 19000         # Custom port
```

When `SLURM_JOB_ID` is set (i.e., already inside a batch script), the
embed `start` command auto-detects foreground mode.

## GPU Management

The embed server supports dynamic GPU reallocation without manual SLURM
interaction. Reallocation cycle time is ~18 seconds.

```bash
# Start with specific GPU count
imas-codex embed start -g 2      # 2 GPUs

# Reallocate GPUs (stop + restart)
imas-codex embed restart -g 8    # Scale to 8 GPUs (~18s)
imas-codex embed restart -g 2    # Scale back to 2 GPUs (~18s)
imas-codex embed restart         # Reset to default (4 GPUs)
```

### Resource Budget (Titan Partition)

Single node: 20 CPUs, 250 GB RAM, 8× Tesla P100-16GB

| Config | Neo4j | Embed (4 GPU) | Available |
|--------|-------|---------------|-----------|
| CPUs | 4 | 2 | 14 |
| Memory | 32 GB | 16 GB | 202 GB |
| GPUs | 0 | 4 | 4 |

## Status & Monitoring

### Graph Status

```bash
$ imas-codex graph status
SLURM:
  neo4j: job 1017391 RUNNING on 98dci4-gpu-0002
    Allocated: CPUs: 4/20, Time: 44:09
    CPU:  [██░░░░░░░░░░░░░░░░░░] 10%  41% of 4 cores
    Mem:  [░░░░░░░░░░░░░░░░░░░░] 2%  6084 MB / 250 GB

Neo4j: running
  Graph: codex
  Location: titan
  Bolt: 7687, HTTP: 7474
```

### Embed Status

```bash
$ imas-codex embed status
  embed: job 1017494 RUNNING on 98dci4-gpu-0002
    Allocated: CPUs: 2/20, GPUs: 4/8, Time: 0:29
Server (http://98dci4-gpu-0002.iter.org:18765):
  ✓ Healthy
  Model: Qwen/Qwen3-Embedding-0.6B
  Device: Tesla P100-PCIE-16GB
  Dimension: 256
  GPU: Tesla P100-PCIE-16GB (16276 MB)
  Uptime: 136.9h
```

### Logs

```bash
imas-codex embed logs            # Last 50 lines
imas-codex embed logs -f         # Follow live
imas-codex embed logs -n 100     # Last 100 lines
imas-codex llm logs              # LLM proxy logs
```

## Systemd Service Management

For persistent services that survive user logout:

```bash
# Embed (login node fallback)
imas-codex embed service install
imas-codex embed service start
imas-codex embed service status

# LLM proxy
imas-codex llm service install
imas-codex llm service start
imas-codex llm service status
```

## Troubleshooting

### Embed server won't start

1. **SLURM resources**: Check `squeue -u $USER` — are other jobs consuming GPUs?
2. **stale jobs**: `scancel <job_id>` any stuck jobs
3. **Script errors**: `cat ~/.local/share/imas-codex/services/codex-embed.log`
4. **Health check**: `curl http://<compute-node>:18765/health`

### LLM proxy won't start

1. **Missing API key**: Ensure `OPENROUTER_API_KEY` is in `.env`
2. **Service logs**: `imas-codex llm logs`
3. **Manual test**: `imas-codex llm start -f` to see errors directly

### Neo4j won't start

1. **Lock files**: Check `graph status` for lock warnings
2. **Memory**: JVM heap defaults to 8-16 GB, check `neo4j.conf`
3. **Logs**: `cat ~/.local/share/imas-codex/services/codex-neo4j.log`

### Port conflicts

Default ports (configurable in `pyproject.toml`):

| Service | Port |
|---------|------|
| Neo4j Bolt | 7687 |
| Neo4j HTTP | 7474 |
| Embed | 18765 |
| LLM Proxy | 18790 |

### Rich output disabled

The spinner/countdown requires a TTY. In pipes, CI, or redirected
output it falls back to plain text. Force rich with `IMAS_CODEX_RICH=1`.
