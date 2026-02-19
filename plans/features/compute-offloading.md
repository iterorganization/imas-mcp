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
| JT-60SA | PBS Professional | nakasvr23 (80c/160t/896G, single-node) | — | none known | 160t (1 via cgroup) | PBS on nakasvr23, falcon unreachable |

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

### JT-60SA Compute Details (verified 2026-02-19)

**Hardware (nakasvr23, HPE ProLiant DL360 Gen10 Plus, launched June 2023):**
- 2× Intel Xeon Platinum 8380 @ 2.30 GHz (40 physical cores each, 80 total, 160 HT threads)
- 1024 GB RAM (896 GB available to PBS), 2 NUMA nodes
- NAS: 90 TB (NFS4 via nakanas23_1). /home 72 TB (68% used), /analysis 186 GB, /analysis_DB 186 GB

**PBS Professional 2022.1.2 — single-node architecture:**
- Server name `licsvr23_1`. Single execution host = nakasvr23 itself.
- **All PBS jobs run on the login node.** There are no separate compute nodes. PBS provides resource isolation (cgroups) and scheduling, not physical offloading.
- Queue `workq`: max 64 ncpus (of 160 threads), max 1024 GB mem. PBS intentionally constrains to 64 CPUs to reserve 96 threads for interactive / system.
- Default allocation: 1 ncpu, 4 GB per job. No walltime limit (`resources_max.walltime` unset — jobs can run indefinitely; docs show example of 1536+ hours = 64 days).
- Total historical jobs: 39,732. License count: 2 available global, 2 in use.

**Interactive session cgroup limits:**
- `nproc` returns 1 for SSH sessions (cgroup `/user.slice/user-24056.slice/session-*.scope`).
- All 160 threads visible in `/proc/cpuinfo` and `/sys/fs/cgroup/cpuset/cpuset.cpus = 0-159`, but process launch is capped at 1 by `max user processes = 1024` and cgroup memory slice.
- **PBS jobs escape the 1-CPU cgroup** — confirmed by submitting a test job (allocated, ran, and deleted successfully).

**Current utilization (2026-02-19 18:19 JST):**
- 7 PBS jobs running from 2 users (MECS transport simulations), each 1 ncpu / 4 GB
- 7/64 PBS CPUs in use (11%), 28 GB / 896 GB RAM assigned (3%)
- System load average: 8.21 (5% of 160 threads)
- 15 users logged in, 819 GB RAM free (91%)
- **System is ~89% idle by CPU, ~97% idle by RAM**

**Policy (from server landing page):**
- "Use the batch queueing system and do not use the frontend for processing the parallel or high-load computation."
- "If the frontend proceeding is identified, it may be an unexpected cancellation."
- "The administrator will not control individual system load, so use it carefully."
- "Use JT-60SA Data Analysis Server only for developing advanced plasma research in JT-60SA."

**falcon:** DNS does not resolve from nakasvr23 (`ssh falcon` fails with "Name or service not known"). Previously documented as having PBS+Slurm, but not reachable from the primary analysis server. Likely on a different network segment or VPN.

**Implication for codex:**
- **PBS offloading has no physical offloading benefit** — jobs still run on nakasvr23. The value is: (1) escaping the 1-CPU cgroup limit, (2) complying with site policy for CPU-intensive work, and (3) proper resource accounting.
- For I/O-bound scanning (rg, fd, SSH, file reads): run directly in interactive session. 1 CPU is sufficient since these are I/O-throttled by NFS, not CPU-bound. `nice` is applied automatically via common infrastructure (see below).
- For CPU-intensive work (if ever needed here — LLM scoring runs on ITER, not JT-60SA): submit via `qsub` for policy compliance and to get more than 1 CPU.
- No tunnel infrastructure needed — services (Neo4j, embed) run on ITER, not on nakasvr23.
- **Recommended scanning policy:** Run lightweight commands directly. Be conservative with concurrency. This is a shared analysis server for plasma physicists — we are guests.

### Nice Level Infrastructure (Implemented)

Process priority is handled by **common infrastructure** — not per-facility ad-hoc policy. The `nice_level` attribute in `SchedulerConfig` (LinkML schema `facility_config.yaml`) configures a Unix nice level for each facility. When set, all remote SSH commands to that facility automatically run at the specified priority.

**How it works:**
1. Facility config declares `nice_level: 19` (private YAML, `is_private` attribute)
2. `tools.py::_resolve_ssh_host()` reads config and calls `configure_host_nice(ssh_host, level)` in `executor.py`
3. All 4 execution functions (`run_command`, `run_script_via_stdin`, `run_python_script`, `async_run_python_script`) check the per-host registry and wrap commands with `nice -n {level}`

**Currently configured:**
- JT-60SA: `nice_level: 19` (lowest priority — yields to all facility user processes)

No per-facility scanning policy sections are needed. Just set `nice_level` in private YAML and all SSH commands to that host inherit the priority.

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
