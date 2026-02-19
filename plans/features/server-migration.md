# Server Migration to Compute Nodes

Move the embedding server and Neo4j off the ITER login node onto long-lived Slurm batch allocations, managed via systemd on GPFS.

## Context

The ITER login node (98dci4-srv-1001) hosts Neo4j, the embedding server, and discovery CLI processes. All compete for 16 cores and 2x T4 GPUs while ~64 users share the node. Moving persistent services to reserved compute nodes frees login resources and lets us use dedicated GPUs.

## Empirical Findings

### Titan Node Profile

Titan (98dci4-gpu-0002): 20 CPUs, 256 GB RAM, 8x Tesla P100-PCIE-16GB (compute capability 6.0, 16 GB each). Partition is idle — we are the sole users. 50% allocation (4 GPUs) confirmed available.

### Embedding Server on P100: Confirmed Working

Qwen3-Embedding-0.6B deploys successfully on P100 (compute capability 6.0) with no code changes. CUDA 12.2 on the node, model loaded via GPFS home directory.

**Startup time:** ~26s (model download from GPFS cache + CUDA kernel compilation)

### Performance Comparison (3-iteration average, post-warmup)

| Batch Size | T4 Login (texts/s) | T4 Server (ms) | P100 Titan (texts/s) | P100 Server (ms) | Speedup |
|-----------|--------------------:|----------------:|---------------------:|------------------:|--------:|
| 1 | 16.0 | 48.7 | 19.5 | 49.7 | 1.2x |
| 10 | 139.1 | 67.6 | 189.6 | 49.5 | 1.4x |
| 50 | 369.8 | 111.6 | 416.9 | 109.3 | 1.1x |
| 100 | 415.4 | 205.6 | 454.0 | 200.8 | 1.1x |
| 200 | 575.1 | 304.2 | 467.5 | 390.1 | 0.8x |
| 500 | 561.8 | 765.2 | 422.2 | 1085.5 | 0.8x |

**Analysis:** P100 matches or beats T4 at batch ≤100 (our typical working range — discovery embeds 10-50 texts per call). At larger batches (200+), T4's newer Turing architecture with tensor cores overtakes P100's Pascal. For our workload, P100 is equivalent or better.

The key win is not raw speed — it's moving the GPU workload off the shared login node entirely.

### Real Document Embedding: Verified

5 fusion-domain documents embedded correctly on P100. Vectors are normalized (norms 0.9995–1.0004), 256-dimensional, model=Qwen/Qwen3-Embedding-0.6B. Vector values match T4 output within float32 precision.

### Network Connectivity: All Paths Confirmed

| Path | Method | Status |
|------|--------|--------|
| Titan → Login (SSH) | `ssh 98dci4-srv-1001.iter.org` | **OK** |
| Titan → Login (ports 7687, 18765) | Direct TCP `/dev/tcp/` | **Blocked** (services bind 127.0.0.1) |
| Login → Titan (port 18766) | `curl http://10.154.84.245:18766/health` | **OK** (server binds 0.0.0.0) |
| Titan → Titan (localhost) | `curl http://localhost:18766/health` | **OK** |
| Compute → Login (ping) | `ping 98dci4-srv-1001.iter.org` | **OK** (0.068ms) |

**Key finding:** Login services (Neo4j, embed) bind to `127.0.0.1`, so compute nodes cannot reach them directly. But compute nodes CAN bind to `0.0.0.0` and be reached from anywhere on the internal network. This means the migration direction is correct: move services TO compute, have login/workstation connect TO compute.

### Port Strategy

When services move to Titan, clients need to reach them:

| Service | Current Location | Current Port | Titan Port | Access Method |
|---------|-----------------|-------------|------------|---------------|
| Embed server | Login (127.0.0.1) | 18765 | 18765 | Direct: `http://98dci4-gpu-0002:18765` |
| Neo4j Bolt | Login (127.0.0.1) | 7687 | 7687 | Direct: `bolt://98dci4-gpu-0002:7687` |
| Neo4j HTTP | Login (127.0.0.1) | 7474 | 7474 | Direct: `http://98dci4-gpu-0002:7474` |

From the workstation (WSL), SSH tunnels point to the Titan hostname instead of localhost:

```bash
# Before: tunnel to login
ssh -L 18765:127.0.0.1:18765 iter

# After: tunnel through login to Titan
ssh -L 18765:98dci4-gpu-0002:18765 -L 7687:98dci4-gpu-0002:7687 -L 7474:98dci4-gpu-0002:7474 iter
```

## Design

### Long-Lived Batch Job

A single Slurm batch job runs for the maximum wall time (varies by partition), hosting both servers:

```bash
#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=codex-services
#SBATCH --output=%x_%j.log

# Start Neo4j
$HOME/Code/imas-codex/scripts/start-neo4j.sh &

# Start embed server on GPU
cd $HOME/Code/imas-codex
uv run --extra gpu imas-codex serve embed start --host 0.0.0.0 --port 18765 --gpu 0

# Keep alive
wait
```

### Systemd on GPFS

Systemd user services cannot run on compute nodes (no lingering, no user sessions). But we can use Slurm's built-in job management:

1. **`sbatch` with `--dependency=singleton`**: Only one instance of the job runs at a time
2. **`scontrol requeue`**: Auto-restart on failure
3. **Slurm `RequeueExitCode`**: Re-submit on specific exit codes

For monitoring and restart, a small systemd timer on the login node checks the Slurm job and resubmits if it died:

```ini
[Unit]
Description=IMAS Codex service watchdog

[Timer]
OnCalendar=*:0/5
Persistent=true

[Install]
WantedBy=timers.target
```

The watcher script:

```bash
#!/bin/bash
# Check if codex-services job is running
if ! squeue -n codex-services -h -u $USER | grep -q RUNNING; then
    sbatch ~/Code/imas-codex/slurm/codex-services.sh
fi
```

### Service Discovery

After a batch job starts, the hostname/IP of the compute node is dynamic. Clients need to find the service.

**Option A: Fixed node reservation** (preferred for Titan)
Since Titan has only 1 node and is idle, request the specific node:
```bash
#SBATCH --nodelist=98dci4-gpu-0002
```
The hostname is then stable and can be hardcoded in config.

**Option B: Service advertisement file**
The batch script writes service endpoints to a GPFS file:
```bash
echo "EMBED_URL=http://$(hostname):18765" > ~/.codex-services.env
echo "NEO4J_URI=bolt://$(hostname):7687" >> ~/.codex-services.env
```
Clients read `~/.codex-services.env` at startup. The `imas_codex.settings` module auto-detects this file.

**Option C: Environment variable override**
For one-off use:
```bash
export IMAS_CODEX_EMBED_REMOTE_URL=http://98dci4-gpu-0002:18765
export NEO4J_URI=bolt://98dci4-gpu-0002:7687
```

### Multi-GPU Deployment

User requests 50% of Titan GPUs (4 of 8) for embedding. With a 0.6B model, one GPU is sufficient. Multi-GPU options:

1. **Replicated servers**: 4 independent embed servers on ports 18765-18768, load-balanced by client
2. **Single server, data-parallel**: Use `sentence-transformers` multi-GPU pool (not supported by our current encoder)
3. **Single GPU + spare capacity**: Use 1 GPU for embed, reserve others for future models (vision/language)

**Recommendation:** Option 3. The 0.6B model uses <2GB VRAM. Reserve additional GPUs for when we deploy larger embedding models (4B/8B) or vision models.

### Tunnel Automation

The `imas-codex tunnel` CLI already exists for SSH tunnels. Extend it:

```bash
# Current: tunnel to login
imas-codex tunnel start iter

# New: tunnel to services (wherever they run)
imas-codex tunnel start iter --services
# Reads ~/.codex-services.env or uses fixed Titan hostname
# Creates: ssh -L 18765:98dci4-gpu-0002:18765 -L 7687:98dci4-gpu-0002:7687 iter
```

## Implementation Phases

### Phase 0: Embed server migration (validated)

Move embedding server to Titan. Deploy via `sbatch` with fixed node. Update SSH tunnel config.

- [x] Confirm P100 compatibility (compute cap 6.0)
- [x] Benchmark P100 vs T4 (equivalent at operational batch sizes)
- [x] Verify network connectivity (login → Titan direct, workstation → Titan via tunnel)
- [x] Test health endpoint and real document embedding
- [ ] Write Slurm batch script
- [ ] Update `~/.ssh/config` tunnel targets
- [ ] Update `imas_codex tunnel` command to use Titan hostname
- [ ] Add systemd timer watchdog on login

### Phase 1: Neo4j migration

Move Neo4j to Titan (no GPU needed, but benefits from 256GB RAM and 20 CPUs).

- [ ] Test Neo4j startup on compute node (Java + GPFS data dir)
- [ ] Benchmark Neo4j on Titan vs login (CPU-bound queries)
- [ ] Update graph profile bolt/http URIs
- [ ] Test graph export/import with Titan-hosted Neo4j

### Phase 2: Service discovery

Implement `~/.codex-services.env` pattern:
- [ ] Batch script writes endpoint file on GPFS
- [ ] `imas_codex.settings` reads endpoint file as fallback
- [ ] `imas-codex tunnel start iter --services` reads endpoint file to build tunnel command
- [ ] `imas-codex serve status` shows where services are running

### Phase 3: Watchdog and auto-restart

- [ ] Systemd timer on login for Slurm job monitoring
- [ ] Slurm `--dependency=singleton` to prevent duplicates
- [ ] Alerting (optional): write health status to GPFS, login timer reads it

### Phase 4: Multi-partition readiness

If Titan goes offline or is claimed, fall back to other partitions:
- rigel: 28c/125GB, no GPU → Neo4j only
- sun: 36c/192GB, no GPU → Neo4j only
- sirius: 36c/512GB, no GPU → Neo4j + large graph operations

The batch script could accept a partition list and try each in order.

## Risks

| Risk | Mitigation |
|------|-----------|
| Slurm preemption / wall time limit | Watchdog timer resubmits; Neo4j WAL ensures crash recovery |
| Titan node maintenance | Fallback partition list; login services remain as emergency backup |
| Network partition (login ↔ compute) | Extremely unlikely on same InfiniBand fabric; <0.1ms latency |
| GPFS latency for Neo4j | Neo4j uses memory-mapped I/O; 256GB RAM means working set fits in page cache |
| Compute node reboot | Slurm requeues job automatically; Neo4j recovery from WAL |

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-07 | P100 viable for Qwen3-Embedding-0.6B | Benchmarked: equivalent throughput at batch ≤100, no code changes needed |
| 2025-07 | Direct network access preferred over tunnels | Login → Titan HTTP works without tunnels; simpler than SSH forwarding for intra-cluster |
| 2025-07 | Single GPU for embed, reserve rest | 0.6B model uses <2GB VRAM; save GPUs for future larger models |
| 2025-07 | Fixed node reservation for Titan | Only 1 node in partition; avoids dynamic service discovery complexity |
