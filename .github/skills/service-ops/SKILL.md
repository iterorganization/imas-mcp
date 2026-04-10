---
name: service-ops
description: How to manage imas-codex services (Neo4j graph, embedding server, LLM proxy, SSH tunnels). Use when starting, stopping, checking status, or troubleshooting services.
---

# Service Operations

## Neo4j Graph Database

```bash
uv run imas-codex graph start       # Start (auto-detects SLURM/SSH/local)
uv run imas-codex graph stop        # Stop
uv run imas-codex graph status      # Check status
uv run imas-codex graph shell       # Interactive Cypher shell
uv run imas-codex graph profiles    # List all profiles and ports
uv run imas-codex graph export      # Export graph archive
uv run imas-codex graph load <f>    # Load archive
uv run imas-codex graph pull        # Pull latest from GHCR
uv run imas-codex graph backup      # Create neo4j-admin dump
uv run imas-codex graph clear       # Clear graph (auto-backup first)
```

### Ports by Location

| Location | Bolt | HTTP |
|----------|------|------|
| iter | 7687 | 7474 |
| tcv | 7688 | 7475 |
| jt-60sa | 7689 | 7476 |

### Neo4j Safety Rules

- Never delete Lucene `write.lock` files — corrupts vector indexes
- Always backup before destructive operations
- Never use `DETACH DELETE` without confirmation
- Use `NOT (x IN [...])` instead of `x NOT IN [...]` (Cypher 5 syntax)

## Embedding Server

```bash
uv run imas-codex embed start         # Start via SLURM
uv run imas-codex embed start -g 2    # Start with 2 GPUs
uv run imas-codex embed start -f      # Foreground (debugging/inside SLURM batch)
uv run imas-codex embed stop          # Stop SLURM job + cleanup rogues
uv run imas-codex embed restart       # Stop + start
uv run imas-codex embed status        # Health + SLURM job + node state
uv run imas-codex embed logs          # View SLURM logs
uv run imas-codex embed service install  # Install systemd service
```

### Embedding Configuration

- Model: `Qwen/Qwen3-Embedding-0.6B` (dimension: 256)
- Default: 4× P100 GPUs on titan partition
- Port: 18765 (HTTP)
- SLURM script: `slurm/codex-embed.sh`

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `embed status` shows node draining | Ask admin to resume node |
| Server healthy but no SLURM job | `imas-codex embed stop` kills rogues |
| Embedding calls timeout | Check tunnel: `lsof -i :18765` |
| Rapid FAILED jobs | Check `imas-codex embed logs`, run `uv sync` on node |

## LLM Proxy (LiteLLM)

```bash
uv run imas-codex llm start           # Start LiteLLM proxy
uv run imas-codex llm stop            # Stop
uv run imas-codex llm restart         # Restart
uv run imas-codex llm status          # Check status
uv run imas-codex llm spend           # Cost tracking
uv run imas-codex llm logs            # View logs
```

`sn mint` and `sn benchmark` require the LLM proxy to be running. Model names must use the
`openrouter/` prefix (e.g. `openrouter/anthropic/claude-sonnet-4-5`) to preserve
`cache_control` blocks — prompt caching is handled provider-side by OpenRouter, not by this
codebase.

## SSH Tunnels

```bash
uv run imas-codex tunnel start iter   # Start SSH tunnel to remote host
uv run imas-codex tunnel stop         # Stop tunnel
uv run imas-codex tunnel status       # Show active tunnels
```

## HPC Job Management

```bash
uv run imas-codex hpc status          # Active SLURM allocations
uv run imas-codex hpc shell           # Interactive compute shell
uv run imas-codex hpc run <cmd>       # Execute on compute node
uv run imas-codex hpc submit <script> # Submit batch job
uv run imas-codex hpc attach <jobid>  # Attach to running job
uv run imas-codex hpc cancel <jobid>  # Cancel job
uv run imas-codex hpc info <jobid>    # Job details
```

## MCP Server

```bash
uv run imas-codex serve                                  # Full mode (all tools)
uv run imas-codex serve --read-only                      # Search/read only
uv run imas-codex serve --dd-only --transport streamable-http  # DD-only container
uv run imas-codex serve --transport stdio                 # For MCP clients
```

## Critical Rules

1. **All services MUST run as SLURM jobs** — never nohup, screen, tmux, or ssh &
2. **Check SLURM allocation** before starting services: `squeue -u $USER`
3. **Never start services directly on compute nodes via SSH** — use SLURM
4. **If a node is draining**, the fix is to get the node resumed — not work around SLURM
