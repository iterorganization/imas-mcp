# Titan Embed Server: Long-Lived Batch Node Deployment

Move the embedding server from the ITER login node onto a long-lived (unlimited) Titan batch node. Configure uniform access from WSL, the login node, and SLURM compute nodes so clients never need to know where the server runs.

## Context

The embedding server currently runs on the ITER login node (98dci4-srv-1001) as a systemd user service, using T4 GPU 1. This competes with ~64 users, NoMachine desktop sessions, and other GPU consumers. The Titan partition (98dci4-gpu-0002) has 8x P100 GPUs, 20 CPUs, 256 GB RAM, and is idle — we have 50% allocation (4 GPUs) available.

### Previous Analysis Results

| Finding | Result |
|---------|--------|
| P100 compatibility | Qwen3-Embedding-0.6B runs on P100 (compute cap 6.0), no code changes |
| P100 vs T4 at batch=100 | 448 texts/s vs 323 texts/s — **P100 39% faster** |
| First-request warmup | 4773ms (CUDA kernel compilation), then normal speed |
| Login → Titan network | Direct HTTP works (`http://10.154.84.245:18766`) — no tunnel needed |
| Health endpoint | Works from both local and cross-node |
| Real document embedding | 5 fusion-domain docs embedded correctly, vectors match T4 output |

## Design

### Uniform Access Pattern

The key requirement: **clients should not know or care where the embed server runs**. The same `remote-url = "http://localhost:18765"` in pyproject.toml must work from:

1. **WSL/workstation**: SSH tunnel from localhost:18765 → Titan:18765 via login node
2. **ITER login node**: SSH tunnel from localhost:18765 → Titan:18765 (loopback)
3. **SLURM compute nodes**: SSH tunnel from localhost:18765 → Titan:18765, OR direct network access

This means the embed server always binds `0.0.0.0:18765` on the Titan node, and every client location uses a tunnel or direct connection to make it appear as `localhost:18765`.

#### Access Matrix

| Client Location | Access Method | URL | Tunnel Required |
|----------------|---------------|-----|-----------------|
| Titan node (local) | Direct loopback | `http://localhost:18765` | No |
| Login node | Reverse tunnel from Titan, or forward to Titan IP | `http://localhost:18765` | Yes — `ssh -L 18765:localhost:18765 98dci4-gpu-0002` |
| Compute node (SLURM) | Direct network to Titan IP | `http://10.154.84.245:18765` or `http://98dci4-gpu-0002:18765` | No |
| WSL/workstation | SSH tunnel through login to Titan | `http://localhost:18765` | Yes — `ssh -L 18765:98dci4-gpu-0002:18765 iter` |

### Embed Server Location Config

Add `embed-host` to `[tool.imas-codex.embedding]` in pyproject.toml to specify where the embedding server runs. This tells tunnel commands and readiness checks where to reach the server:

```toml
[tool.imas-codex.embedding]
model = "Qwen/Qwen3-Embedding-0.6B"
dimension = 256
backend = "remote"
remote-url = "http://localhost:18765"
server-port = 18765
embed-host = "98dci4-gpu-0002"  # Titan node hostname
```

When `embed-host` is set:
- `imas-codex tunnel start iter --embed` forwards to `embed-host:server-port` instead of `127.0.0.1:server-port`
- `readiness.py` checks against the Titan hostname for compute node URL rewriting
- The systemd embed service on the login node is replaced by a SLURM batch job on Titan

### SLURM Batch Job (Long-Lived)

The Titan partition supports unlimited wall time. The embed server runs as a single-GPU SLURM job with `--time=UNLIMITED`:

```bash
#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=UNLIMITED
#SBATCH --job-name=codex-embed
#SBATCH --nodelist=98dci4-gpu-0002
#SBATCH --output=%h/.local/share/imas-codex/logs/embed-titan_%j.log

# Titan has 8x P100; use GPU 0
export CUDA_VISIBLE_DEVICES=0

cd ~/Code/imas-codex

# Start embed server on all interfaces
exec uv run --extra gpu imas-codex serve embed start \
    --host 0.0.0.0 \
    --port 18765 \
    --gpu 0
```

### Login Node Tunnel Service

A systemd service on the login node forwards `localhost:18765` to the Titan node. This makes the embed server appear local to any process on the login node:

```ini
[Unit]
Description=SSH tunnel to Titan embedding server
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/autossh -M 0 -N \
    -o "ServerAliveInterval=15" \
    -o "ServerAliveCountMax=3" \
    -o "ExitOnForwardFailure=yes" \
    -L 18765:localhost:18765 \
    98dci4-gpu-0002
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

### Watchdog Timer

A systemd timer on the login node checks the SLURM job and resubmits if it died:

```ini
[Timer]
OnCalendar=*:0/5
Persistent=true
```

The watcher script:

```bash
#!/bin/bash
if ! squeue -n codex-embed -h -u $USER | grep -q RUNNING; then
    sbatch ~/Code/imas-codex/slurm/codex-embed.sh
fi
```

### `serve embed start` CLI Changes

Add `--location` option to `serve embed start` that records the embed server location:

```
imas-codex serve embed start --host 0.0.0.0 --port 18765 --gpu 0 --location titan
```

The location is exposed in the `/health` and `/info` endpoints so clients can determine where the server is running. This enriches the `hostname` field already present in health responses.

### Tunnel CLI Changes

Update `imas-codex tunnel start` to use `embed-host` from config when forwarding the embed port:

```python
# Before: tunnel to login's localhost
# -L 18765:127.0.0.1:18765 iter
#
# After: tunnel through login to Titan
# -L 18765:98dci4-gpu-0002:18765 iter
```

The `_get_tunnel_ports()` function reads `embed-host` from settings and uses it as the remote bind address instead of `127.0.0.1`.

### Readiness Module Changes

Update `imas_codex/embeddings/readiness.py`:

1. Replace hardcoded `ITER_LOGIN_HOST` with dynamic resolution from settings
2. On compute nodes, redirect to Titan IP (not login) when `embed-host` is configured
3. When the embed server is on Titan, don't try `systemctl --user start` on login (no service there)

### Compute Module Changes

Update `imas_codex/cli/compute.py`:

1. `_build_env_exports()` uses `embed-host` to set `IMAS_CODEX_EMBED_REMOTE_URL`
2. Compute nodes reaching Titan directly: `http://98dci4-gpu-0002:18765`

## Implementation

### Phase 1: Settings & CLI (this PR)

- [x] Add `get_embed_host()` to `settings.py`
- [x] Add `embed-host` to `[tool.imas-codex.embedding]` in pyproject.toml
- [x] Add `--location` option to `serve embed start`
- [x] Update `_get_tunnel_ports()` in tunnel.py to use embed-host as remote bind
- [x] Update `readiness.py` to resolve embed host dynamically
- [x] Update `compute.py` to use embed-host for SLURM job env exports
- [x] Add `imas-codex-embed-tunnel.service` template for login→Titan tunnel
- [x] Update `imas-codex-embed.service` template for SLURM-based Titan deployment

### Phase 2: Deploy & Test

- [ ] Submit SLURM job on Titan: `sbatch slurm/codex-embed.sh`
- [ ] Install tunnel service on login: `imas-codex tunnel service install iter --embed`
- [ ] Test from WSL: `curl http://localhost:18765/health`
- [ ] Test from login: `curl http://localhost:18765/health`
- [ ] Test from compute node: `curl http://98dci4-gpu-0002:18765/health`
- [ ] Run real embedding from each location: `imas-codex embed update --label FacilitySignal --dry-run`
- [ ] Compare P100 Titan vs T4 login performance with production workload
- [ ] Stop old systemd service on login: `systemctl --user disable --now imas-codex-embed`

### Phase 3: Multi-GPU (future)

Reserve additional Titan GPUs for:
- Larger embedding models (4B/8B Qwen3)
- Vision model serving
- LLM inference (compaction/language tasks)

## Performance Expectations

Based on previous benchmarking (3-iteration average, post-warmup):

| Metric | T4 Login | P100 Titan | Improvement |
|--------|----------|------------|-------------|
| Throughput (batch=100) | 323 texts/s | 448 texts/s | +39% |
| Throughput (batch=50) | 370 texts/s | 417 texts/s | +13% |
| Throughput (batch=10) | 139 texts/s | 190 texts/s | +37% |
| GPU contention | Shared with NX desktop | Dedicated | Eliminated |
| CPU contention | 16 cores shared with 64 users | 4 dedicated cores | Eliminated |

## Risks

| Risk | Mitigation |
|------|-----------|
| Titan node reboot/maintenance | Watchdog timer resubmits job; login can run local embed as fallback |
| SLURM preemption | `--time=UNLIMITED` + singleton prevents preemption on idle partition |
| Network partition login↔Titan | Same InfiniBand fabric, <0.1ms latency; extremely unlikely |
| Tunnel to Titan dies | autossh auto-reconnects; systemd restarts on failure |
| Model download on Titan | GPFS home dir shared; model cached from login node already |

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02 | Fixed Titan node (nodelist) | Only 1 node in partition; stable hostname avoids service discovery |
| 2026-02 | Tunnel-based uniform access | Same `localhost:18765` URL works everywhere; zero client changes |
| 2026-02 | Single GPU for embed | 0.6B model uses <2GB VRAM; reserve others for future models |
| 2026-02 | Login tunnel service | Makes embed appear local on login; readiness module unchanged |
| 2026-02 | `embed-host` config key | Explicit location config; no magic hostname detection |
