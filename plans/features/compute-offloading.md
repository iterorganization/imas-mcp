# Compute Offloading

Route CPU-intensive CLI work to compute nodes to avoid saturating shared login nodes.

## Context

Discovery processes (`discover paths`, `discover wiki`, `discover signals`, `discover files`) collectively consume most of a 16-core ITER login node when run in parallel. Neo4j alone takes ~81% CPU. This impacts other users and throttles our own throughput. Facilities with job schedulers have idle compute capacity that can absorb this work.

## Facility Compute Landscape

| Facility | Scheduler | Nodes | Debug Alloc | GPU | Login Cores | Status |
|----------|-----------|-------|-------------|-----|-------------|--------|
| ITER | Slurm | 392 (5 partitions) | ~149ms | 2x T4 login, 8x P100 titan | 16 | **High priority** — login saturated |
| JET | SGE | 52 (heimdall101-152) | untested | none | 16 | Moderate — load 17.57 with 69 users |
| TCV | none (planned) | ~35 desktop PCs + 3 LAC nodes | — | spcgpu01/02 (specs unknown) | 24 (LAC) | Low — no scheduler, LAC process watcher |
| JT-60SA | PBS Professional | nakasvr23 (64c/896G) + falcon (PBS+Slurm) | — | none known | 1 vCPU | PBS on nakasvr23, Slurm on falcon |

### ITER Slurm Details

- **Partitions**: rigel (28c/125G), sun (36c/192G), vega (56c/512G), sirius (36c/512G), titan (20c/256G/8xP100)
- **Debug partitions**: 4 nodes each, 1h max time, near-instant allocation
- **Home via GPFS**: All tools, `uv`, configs available on compute without deployment
- **Tunnels verified**: compute → login SSH tunnels work for embed (18765), Neo4j bolt (7687), Neo4j HTTP (7474)
- **SSH routing**: Server-side SSH config routes compute node SSH to external hosts (TCV, JET) through login

### JET SGE Details

- **Queues**: 52 queues (heimdall101-152), 16 slots each, INFINITY time/mem limits
- **Load**: Many nodes near-idle (0.01-0.12 load) while login at 17.57
- **Access**: `qsub` for batch, `qrsh` for interactive
- **Home shared**: Yes, same filesystem on login and compute

### TCV Compute Details

- **No scheduler**: LAC wiki confirms "Currently there is no scheduler (such as Slurm) in lacs. There are plans to implement one."
- **LAC nodes** (spclac05/06/07+): Shared analysis servers, up to 48 cores, up to 200 GB RAM. Automated process watcher terminates resource-heavy jobs. Users directed to SCITAS for heavy work.
- **Desktop PC pool** (~35 nodes, spcpc*): Staff desktops configured as compute nodes during idle cycles. Managed by medusa.epfl.ch. SSH access for CRPP members (Gaspar auth). No scheduler — use `nice -20`. 16-128 GB RAM, 4-16 cores (Haswell i7 to Alder Lake). openSUSE Leap.
- **GPU nodes**: spcgpu01/02 exist (ping OK), SSH access not configured for codex user, specs unknown.
- **PPB110**: Former Slurm cluster (8x Haswell i7-4770, 32GB each). **Dismantled 2026**. Login node ppb110.epfl.ch still online for old data access.
- **External HPC**: SCITAS (Fidis/Helvetios) and RCP mentioned as options, but not reachable from TCV login node.
- **Implication**: No offloading possible at TCV. Commands run directly on LAC nodes with `nice`. Process watcher limits sustained high-CPU work.

### JT-60SA Compute Details

- **PBS Professional on nakasvr23**: Queue `workq`, max 64 cores, 896 GB memory. `qsub` command. Jobs run on the login node itself (no separate compute nodes). Policy: parallel/high-load jobs MUST use batch; frontend jobs may be killed without notice.
- **falcon**: Secondary compute system with both PBS (queue: `normal`) and Slurm (partition: `single`, compute node falcon02). Used for MUSCLE3, IMPACT, OFMC codes.
- **imaging**: Older compute server for MPI codes (Intel compiler 15/18).
- **helios, SGI8600**: JAEA HPC systems, separate infrastructure.
- **VPN required**: SSH access requires QST VPN, frequently unreachable.
- **Implication**: PBS offloading theoretically possible but limited — nakasvr23 is a single-node VM running PBS locally. falcon offers Slurm but access/configuration unknown.

## Design

### `--submit` CLI Flag

Add a `--submit/--no-submit` flag to discover commands. When enabled, the CLI submits work to the facility's scheduler instead of running locally.

```
imas-codex discover paths iter --submit
imas-codex discover wiki tcv   # no scheduler → runs locally (no-op)
```

Behavior:
1. Read `compute.scheduler` from facility private config
2. If no scheduler: warn and run locally
3. If scheduler: allocate interactive session, establish tunnels, run CLI on compute node
4. On completion or timeout: release allocation

The scheduler abstraction reads config from `ComputeConfig` in the private YAML, supporting Slurm (`salloc`/`sbatch`), SGE (`qrsh`/`qsub`), and PBS (`qsub`) with the same interface.

### Execution Patterns

Three patterns for different workloads — the right choice depends on task duration vs allocation overhead.

**Pattern A: Long-lived interactive allocation (recommended for ITER)**
- `salloc --partition=rigel --time=04:00:00` once per session
- Run multiple discover commands within the allocation
- SSH tunnels established once at allocation time
- Best when: tasks run for hours, scheduler has fast allocation

**Pattern B: Per-command batch submission**
- Each `--submit` invocation wraps the CLI call in `sbatch`/`qsub`
- Output goes to log file, progress via Neo4j status queries
- Best when: fire-and-forget overnight runs

**Pattern C: Debug queue for short tasks**
- `salloc --partition=rigel_debug --time=01:00:00`
- 1h window for quick operations (du, rg, tokei)
- Caution: heavy use of debug queues may violate site policy

Pattern A is the primary target. Pattern B is useful for unattended operation. Pattern C is a pragmatic escape but should not be the default.

### cx Integration

The `cx` script on ITER could gain a `--slurm` mode:

```bash
cx --slurm iter    # salloc + tunnels + zellij on compute
```

This wraps the interactive allocation pattern and attaches a zellij session on the compute node. All codex CLI commands then run on compute transparently.

### Tunnel Setup

When running on an ITER compute node, services on the login node need SSH tunnels:

```bash
ssh -f -N -L 18765:127.0.0.1:18765 -L 7687:127.0.0.1:7687 -L 7474:127.0.0.1:7474 98dci4-srv-1001
```

This should be automated as part of the allocation setup. The `compute.tunnel_required` config flag indicates when this is needed.

### Remote Facility Considerations

ITER discovery operates on remote facilities via SSH. From compute nodes:
- SSH to TCV/JET works (routed through login via server-side SSH Match blocks)
- No impact on the remote facility's compute — we only SSH and run lightweight commands
- The offloading benefit is entirely for the ITER login node where Neo4j, embed server, and Python processes run

## Implementation Phases

### Phase 0: Tunnel helper
Add `imas-codex compute tunnel <facility>` to establish SSH tunnels from compute to login services. Read ports from config.

### Phase 1: Manual allocation workflow
Document the `salloc` + tunnel + `uv run imas-codex discover ...` workflow. No CLI changes yet — users run commands in an interactive allocation. Add exploration notes confirming the workflow.

### Phase 2: `--submit` flag
Add the flag to discover commands via `common.py` shared options. Implement Slurm and SGE backends that read from `ComputeConfig`. Handle tunnel setup/teardown.

### Phase 3: cx --slurm
Extend the `cx` script to support Slurm interactive mode. Allocate, tunnel, attach zellij — transparent to the user.

### Phase 4: Titan GPU for embedding
Investigate why the embed server previously failed on P100 (compute capability 6.0). If viable, move embedding to titan to free a T4 on the login node.

## Schema

`ComputeConfig` added to `facility_config.yaml` LinkML schema with:
- `SchedulerConfig` (type, alloc/batch command, partitions, latency, shared home, tunnel required)
- `PartitionConfig` (name, max_time, nodes, cores, memory, is_debug, gpu info)
- `GPUResource` (location, model, count, memory, current use)
- `LoginNodeConfig` (hostname, cores, memory, shared users)
- `SchedulerType` enum (slurm, sge, pbs, lsf, condor)

All compute config lives in `<facility>_private.yaml` (private, gitignored).
