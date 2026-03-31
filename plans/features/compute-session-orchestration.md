# Compute Session Orchestration

Move interactive development sessions (zellij, LiteLLM, AI agents) from SDCC login
nodes to SLURM-managed compute nodes, eliminating login node saturation while
maintaining the reconnectable, persistent session experience.

---

## Problem Statement

All interactive development workloads currently run on shared login nodes (6 nodes,
16–48 cores each). A single user running 5 Copilot CLI agents, a LiteLLM proxy, a
zellij server, and compilation jobs can push login node load above 20× on 48 cores,
degrading performance for all ~30 concurrent users. The SDCC has 424 compute nodes
(157+ idle at any time) with UNLIMITED time limits, but no established pattern for
running persistent interactive sessions on them.

## Approach

Extend the existing `cx` session manager to support SLURM-backed compute sessions.
Each compute session is a SLURM job that hosts a zellij server on a compute node.
Users connect via `srun --overlap --pty` from the login node, and reconnect with the
same command after disconnections. LiteLLM moves to a compute node to keep it close
to the agents that consume it, with DNS workarounds for the partial internet access
on compute nodes.

---

## Network Topology (Empirically Verified)

All findings below are from live tests on 2026-03-31.

### Connectivity Matrix

| From → To | Method | Status | Evidence |
|-----------|--------|--------|----------|
| Login → Compute | Direct TCP | ✅ | `curl http://98dci4-gpu-0002:18765` returns HTTP 200 |
| Login → Compute | SSH | ⚠️ | `pam_slurm_adopt` blocks unless caller has active SLURM job on target |
| Login → Compute | `srun --jobid --overlap` | ✅ | Tested: `srun --jobid=X --overlap --pty hostname` works |
| Compute → Login | ICMP | ✅ | `ping 10.154.100.16` (login IP) succeeds |
| Compute → Login | TCP (ports) | ❌ | `curl http://10.154.100.16:18400` times out (firewall) |
| Compute → Compute | TCP | ✅ | From sun-3001: `curl http://98dci4-clu-2002:7474` (Neo4j on rigel) returns HTTP 200 |
| Compute → Internet | github.com | ✅ | HTTP 200, 0.21s |
| Compute → Internet | pypi.org | ✅ | HTTP 200, 0.11s |
| Compute → Internet | api.anthropic.com | ✅ | DNS resolves to 160.79.104.10 |
| Compute → Internet | api.openrouter.ai | ❌ | **NXDOMAIN** (DNS filtering — `openrouter.ai` resolves but `api.` subdomain doesn't) |
| Compute → Internet | External DNS (8.8.8.8) | ❌ | Connection timed out |
| Compute SSH tunneling | `-L` / `-R` forwarding | ❌ | `administratively prohibited` (AllowTcpForwarding=no) |

### Network Subnets

| Network | Subnet | Hosts |
|---------|--------|-------|
| Login nodes | 10.154.100.0/24 | sdcc-login-1001..1006 (srv names: 98dci4-srv-100x) |
| Rigel compute | 10.154.5.0/24 | 98dci4-clu-2001..2080 |
| Sun compute | 10.154.7.0/24 | 98dci4-clu-3001..3144 |
| Vega compute | 10.154.8.0/24 | 98dci4-clu-4001..4076 |
| Sirius compute | 10.154.9.0/24 | 98dci4-clu-5001..5140 |
| Titan GPU | 10.154.84.0/24 | 98dci4-gpu-0002 |
| GPFS storage | 10.153.100.0/24 | ems/ess servers |

### DNS Configuration

Compute nodes use ITER internal DNS servers (10.10.24x.x) with search domain `iter.org`.
External DNS servers (8.8.8.8) are unreachable. DNS resolution is selective — most
public domains resolve, but `api.openrouter.ai` specifically returns NXDOMAIN while
`openrouter.ai` and `api.anthropic.com` both resolve. This suggests either a DNS
filtering policy or a resolver issue with specific CNAME chains (api.openrouter.ai
likely CNAMEs to Cloudflare infrastructure).

### SLURM Configuration

| Partition | MaxTime | MaxNodes | Cores/Node | RAM | Idle | Notes |
|-----------|---------|----------|------------|-----|------|-------|
| all (default) | UNLIMITED | UNLIMITED | 28-56 | 128-512 GB | ~157 | Largest idle pool |
| sirius | UNLIMITED | UNLIMITED | 36 | 512 GB | ~45 | Newest hardware, high RAM |
| vega | UNLIMITED | UNLIMITED | 56 | 512 GB | ~3 | High core count |
| sun | UNLIMITED | UNLIMITED | 36 | 192 GB | ~36 | Good general-purpose |
| rigel | UNLIMITED | UNLIMITED | 28 | 128 GB | ~73 | Oldest, most available |
| titan | UNLIMITED | UNLIMITED | 20 | 256 GB | 1 | 8× P100 GPU |
| *_debug | 1h | 3 | varies | varies | ~15 | debug_high QOS, instant scheduling |

- **PreemptMode: OFF** — jobs are never preempted
- **JobRequeue: 1** — but with PreemptMode OFF, this doesn't trigger
- **PrologFlags: Alloc,Contain,X11** — X11 forwarding available
- **pam_slurm_adopt** — enforces that SSH to compute requires an active SLURM job

### Filesystem

GPFS home directories are shared across login and all compute nodes. Zellij binary
(`~/.local/bin/zellij`), configs (`~/.config/zellij/`), and all project files are
visible on compute. Zellij server sockets live in `/tmp/zellij-<uid>/` which is
**per-node** — sessions created on one node are not visible from another.

---

## Architecture

### Session Lifecycle

```
┌─ Windows Terminal ─┐     ┌─ Login Node ──────┐     ┌─ Compute Node ────────────┐
│                     │     │                    │     │                            │
│  cx iter work  ─────┼──SSH──► cx work  ────────┼──srun──► zellij server          │
│                     │     │   (detect SLURM    │  --pty   │  ├── agent-1 tab     │
│                     │     │    job, reconnect)  │  overlap │  ├── agent-2 tab     │
│                     │     │                    │     │     │  ├── shell tab        │
│  Disconnect ────────┼─────┼────────────────────┼─────┼─────│  └── mcp tab         │
│  (network drop)     │     │   srun step dies   │     │     │                      │
│                     │     │   SLURM job persists│     │     │  zellij server lives │
│  cx iter work  ─────┼──SSH──► cx work  ────────┼──srun──► zellij attach work     │
│  (reconnect)        │     │   (same job ID)    │  --pty   │  (all tabs restored) │
│                     │     │                    │  overlap │                        │
└─────────────────────┘     └────────────────────┘     └────────────────────────────┘
```

### Service Topology (Target State)

```
┌─ Login Node (gateway only) ──────────────────────────┐
│                                                       │
│  cx script ──► srun --overlap --pty ──────────────┐   │
│  SSH tunnels from Windows                         │   │
│  (no long-running compute workloads)              │   │
│                                                   │   │
└───────────────────────────────────────────────────┼───┘
                                                    │
                    ┌───────────────────────────────┘
                    ▼
┌─ Sirius Compute Node A (session host) ───────────────┐
│                                                       │
│  zellij "imas-codex" session                         │
│    ├── agent-1: Claude Code / Copilot CLI            │
│    ├── agent-2: Claude Code / Copilot CLI            │
│    ├── shell: interactive development                │
│    ├── mcp: imas-codex serve                         │
│    └── git: version control                          │
│                                                       │
│  SLURM job: cx-imas-codex (--time=7-0, 8 cores, 32G)│
│                                                       │
└───────────────────────────────────────────────────────┘

┌─ Sirius Compute Node B (services) ──────────────────┐
│                                                       │
│  LiteLLM proxy (:18400)                              │
│    → api.anthropic.com (direct)                      │
│    → api.openrouter.ai (via /etc/hosts workaround)   │
│                                                       │
│  SLURM job: codex-llm (--time=UNLIMITED)             │
│                                                       │
└───────────────────────────────────────────────────────┘

┌─ Rigel Compute Node (data) ──────────────────────────┐
│                                                       │
│  Neo4j graph database (:7687/:7474)                  │
│  SLURM job: codex-neo4j (running 3+ days)            │
│                                                       │
└───────────────────────────────────────────────────────┘

┌─ Titan GPU Node (inference) ─────────────────────────┐
│                                                       │
│  Embedding server (:18765) — 4× P100                 │
│  Ollama local LLM (:11434) — 1× P100                │
│  SLURM jobs: codex-embed, codex-llm-local            │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### Connection Chain

```
User (Windows) ──SSH──► Login Node ──srun──► Compute Node
                                              │
                                              ├── zellij tabs (interactive)
                                              │     └── agents connect to:
                                              │           ├── LiteLLM on Node B (:18400)
                                              │           ├── Neo4j on Rigel (:7687)
                                              │           └── Embed on Titan (:18765)
                                              │
                                              └── compute-local processes
                                                    ├── MCP server
                                                    ├── compilation
                                                    └── tests
```

---

## Implementation Phases

### Phase 1: SLURM Session Infrastructure

Create the foundational SLURM job management for persistent interactive sessions.

#### 1.1 Session Host Job Template

Create `slurm/session-host.sh`:

```bash
#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=cx-SESSION_NAME
#SBATCH --output=%x-%j.log
#
# Persistent session host. Does nothing except keep the allocation alive.
# Users connect via: srun --jobid=JOBID --overlap --pty zellij attach --create SESSION_NAME
#
# The job runs a keepalive loop that:
# 1. Periodically checks if zellij server is still running
# 2. Exits cleanly if zellij has been detached/killed (no orphan allocations)
# 3. Logs heartbeat for monitoring

set -euo pipefail

echo "Session host started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Session: ${SLURM_JOB_NAME#cx-}"

SESSION_NAME="${SLURM_JOB_NAME#cx-}"
IDLE_TIMEOUT=${CX_IDLE_TIMEOUT:-86400}  # 24h default before cleanup
LAST_ACTIVITY=$(date +%s)

cleanup() {
    echo "Session host shutting down at $(date)"
    # Kill any remaining zellij servers for this session
    pkill -u $(id -u) -f "zellij.*--server.*${SESSION_NAME}" 2>/dev/null || true
}
trap cleanup EXIT

while true; do
    # Check if zellij server is running for our session
    if pgrep -u $(id -u) -f "zellij.*server" >/dev/null 2>&1; then
        LAST_ACTIVITY=$(date +%s)
    fi

    # Check idle timeout (no zellij activity)
    IDLE_SECONDS=$(( $(date +%s) - LAST_ACTIVITY ))
    if [ $IDLE_SECONDS -gt $IDLE_TIMEOUT ]; then
        echo "No zellij activity for ${IDLE_SECONDS}s, exiting"
        exit 0
    fi

    sleep 60
done
```

#### 1.2 Session Management Functions

Create `~/.local/lib/cx-slurm.sh` (sourced by cx):

```bash
# cx-slurm.sh — SLURM session management functions for cx

# Find existing SLURM job for a session
cx_find_job() {
    local session="$1"
    squeue -u "$USER" -h -o "%i %j %t %N" 2>/dev/null | \
        awk -v name="cx-${session}" '$2 == name && $3 == "R" {print $1, $4; exit}'
}

# Submit a new session host job
cx_submit_job() {
    local session="$1"
    local partition="${2:-sirius}"
    local cpus="${3:-8}"
    local mem="${4:-32G}"
    local time="${5:-7-00:00:00}"

    sbatch \
        --partition="$partition" \
        --cpus-per-task="$cpus" \
        --mem="$mem" \
        --time="$time" \
        --job-name="cx-${session}" \
        --output="$HOME/.local/share/imas-codex/logs/cx-${session}-%j.log" \
        --parsable \
        ~/Code/imas-codex/slurm/session-host.sh
}

# Wait for job to start running
cx_wait_for_job() {
    local jobid="$1"
    local timeout="${2:-30}"
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        local state
        state=$(squeue -j "$jobid" -h -o "%t" 2>/dev/null)
        case "$state" in
            R)  return 0 ;;
            PD) ;;
            *)  return 1 ;;
        esac
        sleep 1
        elapsed=$((elapsed + 1))
        printf "\r  Waiting for resources... %ds" "$elapsed" >&2
    done
    printf "\n" >&2
    return 1
}

# Connect to zellij on compute via srun
cx_connect() {
    local jobid="$1"
    local session="$2"
    local layout="$3"

    local layout_flag=""
    [ -n "$layout" ] && layout_flag="-l $layout"

    # srun --overlap allows multiple steps on same allocation
    # --pty gives us a terminal for zellij
    exec srun \
        --jobid="$jobid" \
        --overlap \
        --pty \
        bash -c "exec zellij $layout_flag attach --create '$session'"
}

# Cancel a session's SLURM job
cx_cancel() {
    local session="$1"
    local info
    info=$(cx_find_job "$session")
    if [ -n "$info" ]; then
        local jobid
        jobid=$(echo "$info" | awk '{print $1}')
        scancel "$jobid"
        echo "Cancelled job $jobid for session $session"
    else
        echo "No active job found for session $session"
    fi
}

# List all active compute sessions
cx_list() {
    echo "Active compute sessions:"
    squeue -u "$USER" -h -o "%10i %20j %2t %10M %20R" 2>/dev/null | \
        grep "cx-" | while read -r jobid name state time node; do
            local session="${name#cx-}"
            echo "  $session  job=$jobid  node=$node  time=$time  state=$state"
        done
}
```

#### 1.3 Verification

Run these tests to verify Phase 1:

```bash
# Test 1: Submit a session host job
JOBID=$(cx_submit_job test-session sirius 4 16G 00:30:00)
echo "Submitted: $JOBID"

# Test 2: Wait for allocation
cx_wait_for_job $JOBID 30 && echo "Running" || echo "Failed to start"

# Test 3: Get job info
cx_find_job test-session

# Test 4: Connect and verify zellij works
srun --jobid=$JOBID --overlap --pty zellij attach --create test-session
# (inside zellij: run a command, then Ctrl-Q to detach)

# Test 5: Reconnect to same session
srun --jobid=$JOBID --overlap --pty zellij attach test-session
# (verify tabs and state are preserved)

# Test 6: Clean up
scancel $JOBID
```

---

### Phase 2: cx Script Extension

Extend the `cx` script to transparently support compute-backed sessions.

#### 2.1 New Invocation Patterns

| Command | Behavior |
|---------|----------|
| `cx iter work` | (existing) SSH to iter, local zellij session "work" |
| `cx compute work` | Submit/attach to SLURM-backed "work" session on compute |
| `cx compute imas-codex` | SLURM-backed "imas-codex" session with project layout |
| `cx compute stop work` | Cancel the SLURM job for "work" session |
| `cx compute list` | List active compute sessions |
| `cx compute status` | Show all SLURM jobs with session info |

The keyword `compute` triggers SLURM mode. Without it, behavior is unchanged.

#### 2.2 cx Compute Mode Logic

Add to `cx` after the existing argument parsing:

```bash
# ── Compute mode ──────────────────────────────────────────────────────
if [ "${1:-}" = "compute" ]; then
    shift
    source ~/.local/lib/cx-slurm.sh

    # Subcommands
    case "${1:-}" in
        stop|cancel)
            cx_cancel "${2:?Usage: cx compute stop <session>}"
            exit 0
            ;;
        list|ls)
            cx_list
            exit 0
            ;;
        status)
            squeue -u "$USER" -o "%.10i %.9P %.20j %.2t %.10M %.6D %R" | grep -E "JOBID|cx-"
            exit 0
            ;;
        *)
            SESSION="${1:-codex}"
            ;;
    esac

    LAYOUT=$(resolve_layout "$SESSION")

    # Check for existing job
    JOB_INFO=$(cx_find_job "$SESSION")

    if [ -n "$JOB_INFO" ]; then
        JOBID=$(echo "$JOB_INFO" | awk '{print $1}')
        NODE=$(echo "$JOB_INFO" | awk '{print $2}')
        echo "Reconnecting to $SESSION on $NODE (job $JOBID)"
    else
        echo "Starting compute session: $SESSION"
        JOBID=$(cx_submit_job "$SESSION")
        echo "Submitted job $JOBID, waiting for resources..."
        if ! cx_wait_for_job "$JOBID" 60; then
            echo "Job failed to start within 60s. Check: squeue -j $JOBID"
            exit 1
        fi
        NODE=$(squeue -j "$JOBID" -h -o "%R")
        echo "Allocated: $NODE"
    fi

    # Reset terminal before connecting
    reset_terminal

    # Connect to zellij on compute node
    cx_connect "$JOBID" "$SESSION" "$LAYOUT"
    exit $?
fi
```

#### 2.3 Session Cleanup on Terminate

Zellij's `on_force_close` config option is set to `detach`. When a user explicitly
closes all panes (not detaches), the zellij server exits. The session-host.sh
keepalive loop detects this and exits, releasing the SLURM allocation. This ensures
no orphan jobs persist after intentional session termination.

For involuntary disconnects (SSH drops, network issues), the zellij server persists
on the compute node, and the session-host keepalive loop keeps the SLURM job alive.
The user reconnects with the same `cx compute <session>` command.

For explicit cleanup: `cx compute stop <session>` cancels the SLURM job, which
triggers the cleanup trap in session-host.sh, killing any remaining zellij servers.

#### 2.4 Remote Compute Mode (from Windows)

When called as `cx iter compute work`, the cx script on the Windows/WSL side
should:

1. SSH to `iter` (login node) as normal
2. On the login node, `cx compute work` enters compute mode
3. The srun connects the PTY chain: Windows → SSH → login → srun → compute → zellij

This works because the existing remote mode already delegates to cx on the remote
side. We just need to pass the `compute` argument through:

```bash
# In the remote mode section, update the REMOTE_CMD:
REMOTE_CMD="CX_SESSION=1 ~/.local/bin/cx $SESSION"
# If $SESSION starts with "compute", it enters compute mode on the remote side
```

Actually, the cleaner approach: add `compute` as a recognized "host" prefix:

```bash
# cx iter compute imas-codex → SSH to iter, then cx compute imas-codex
# Parse: $1=iter (remote), $2=compute, $3=imas-codex
if [ $# -ge 3 ] && [ "$2" = "compute" ]; then
    REMOTE="$1"
    shift  # Remove host
    # Remaining args: "compute imas-codex" — passed to remote cx
    SESSION_ARGS="$*"
fi
```

#### 2.5 Verification

```bash
# Test 1: Submit and connect from login
cx compute test-phase2
# (verify zellij starts, create a file, detach with Ctrl-Q)

# Test 2: Reconnect
cx compute test-phase2
# (verify the file and session state are preserved)

# Test 3: List sessions
cx compute list

# Test 4: Remote from WSL/Windows
cx iter compute test-phase2
# (verify full chain: Windows → SSH → login → srun → compute → zellij)

# Test 5: Disconnect resilience
# While connected via cx iter compute test-phase2:
# Kill the SSH connection (close terminal window)
# Wait 10 seconds
# cx iter compute test-phase2
# Verify session is fully restored

# Test 6: Clean up
cx compute stop test-phase2
```

---

### Phase 3: LiteLLM Migration to Compute

Move the LiteLLM proxy from the login node to a compute node, eliminating the
last significant login-node workload.

#### 3.1 DNS Workaround for api.openrouter.ai

The `api.openrouter.ai` subdomain returns NXDOMAIN on compute nodes while
`openrouter.ai` resolves to Cloudflare IPs (104.18.2.115, 104.18.3.115). This is
likely a DNS resolver issue with CNAME chains rather than an intentional block,
since `api.anthropic.com` resolves correctly.

**Option A: /etc/hosts override (per-job, no root needed)**

```bash
# In the LiteLLM SLURM script, resolve on login and inject:
# (Login node can resolve api.openrouter.ai)
echo "104.18.2.115 api.openrouter.ai" >> /tmp/hosts-override
export HOSTALIASES=/tmp/hosts-override
```

Note: `HOSTALIASES` only works with glibc `getaddrinfo()`, which Python's
`socket.getaddrinfo()` uses. This should work for LiteLLM's HTTP client.

**Option B: Request IT DNS whitelist**

File a ticket requesting `api.openrouter.ai` be added to the DNS resolver
whitelist. Reference that `openrouter.ai`, `api.anthropic.com`, `github.com`,
and `pypi.org` already resolve correctly.

**Option C: Use Anthropic API directly (bypass OpenRouter for Anthropic models)**

Since `api.anthropic.com` resolves on compute nodes, configure LiteLLM with
a direct Anthropic API key for Claude models, and only use OpenRouter for
non-Anthropic models. This eliminates the OpenRouter dependency for the
primary model family.

**Recommended: Option C (immediate) + Option B (medium-term)**

Option C works today with no infrastructure changes. Option B provides full
OpenRouter access for model diversity.

#### 3.2 LiteLLM SLURM Job

Create `slurm/litellm-proxy.sh`:

```bash
#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=UNLIMITED
#SBATCH --job-name=codex-llm
#SBATCH --output=%x-%j.log
#
# LiteLLM proxy on compute node.
# Reachable from all compute nodes at $(hostname):18400
# Reachable from login nodes at $(hostname):18400
#
# DNS workaround: HOSTALIASES for api.openrouter.ai
# if ITER DNS doesn't resolve this subdomain.

set -euo pipefail

cd ~/Code/imas-codex

echo "Starting LiteLLM proxy on $(hostname) at $(date)"

# DNS workaround for api.openrouter.ai (if needed)
OPENROUTER_IP=$(getent hosts api.openrouter.ai 2>/dev/null | awk '{print $1}')
if [ -z "$OPENROUTER_IP" ]; then
    echo "WARNING: api.openrouter.ai not in DNS, using HOSTALIASES workaround"
    echo "104.18.2.115 api.openrouter.ai" > /tmp/hostaliases-litellm
    export HOSTALIASES=/tmp/hostaliases-litellm
fi

# Write node info for service discovery
mkdir -p ~/.local/share/imas-codex/services
echo "$(hostname):18400" > ~/.local/share/imas-codex/services/litellm-node.txt
echo "$SLURM_JOB_ID" >> ~/.local/share/imas-codex/services/litellm-node.txt

# Load environment
source ~/Code/imas-codex/.env

# Start PostgreSQL for team/key persistence
uv run pgserver start ~/.local/share/imas-codex/services/pgdata/ &
PG_PID=$!
sleep 3

# Start LiteLLM
exec uv run --offline litellm \
    --config ~/Code/imas-codex/imas_codex/config/litellm_config.yaml \
    --host 0.0.0.0 \
    --port 18400 \
    --drop_params
```

#### 3.3 Service Discovery

Since the LiteLLM compute node hostname varies, clients need service discovery.
The SLURM job writes its hostname to a well-known GPFS path:

```bash
# Any client can find LiteLLM:
LITELLM_HOST=$(head -1 ~/.local/share/imas-codex/services/litellm-node.txt)
export LITELLM_PROXY_URL="http://${LITELLM_HOST}"
```

Update `imas_codex/settings.py` `get_llm_proxy_url()` to check this file when
`location != "local"`.

#### 3.4 Verification

```bash
# Test 1: Submit LiteLLM job
sbatch slurm/litellm-proxy.sh
# Wait for start, get node

# Test 2: Health check from login
LLM_NODE=$(head -1 ~/.local/share/imas-codex/services/litellm-node.txt)
curl http://$LLM_NODE:18400/health

# Test 3: Health check from compute
srun --partition=all --time=00:05:00 --pty bash -c \
    "curl http://$LLM_NODE:18400/health"

# Test 4: LLM call from compute
srun --partition=all --time=00:05:00 --pty bash -c "
    export LITELLM_PROXY_URL=http://$LLM_NODE:18400
    cd ~/Code/imas-codex
    uv run python -c \"
from litellm import completion
r = completion(model='anthropic/claude-haiku-4-5', messages=[{'role':'user','content':'ping'}],
               api_base='http://$LLM_NODE:18400', api_key='sk-litellm-master')
print(r.choices[0].message.content)
\"
"

# Test 5: Verify DNS workaround
srun --partition=all --time=00:05:00 --pty bash -c "
    export HOSTALIASES=/tmp/hostaliases-litellm
    curl -s https://api.openrouter.ai/api/v1/models | head -c 200
"
```

---

### Phase 4: Tab-Level Compute Integration

Extend the architecture to support individual zellij tabs connecting to different
compute nodes, enabling mixed local/compute workflows.

#### 4.1 Compute Tab Script

Create `~/.local/bin/cx-tab` — a script designed to run inside a zellij tab
that connects to a SLURM job:

```bash
#!/bin/bash
# cx-tab — connect a zellij tab to a SLURM compute allocation
#
# Usage (inside a zellij tab):
#   cx-tab <session-job-name>     Connect to existing session's node
#   cx-tab --new <partition>      Allocate a new node for this tab
#
# This script runs INSIDE the tab. It:
# 1. Finds or creates a SLURM allocation
# 2. Connects via srun --overlap --pty
# 3. On disconnect, shows reconnect prompt
# 4. On job cancellation, shows cleanup message

set -euo pipefail

source ~/.local/lib/cx-slurm.sh

JOB_NAME="${1:?Usage: cx-tab <job-name>}"

while true; do
    JOB_INFO=$(cx_find_job "$JOB_NAME")
    if [ -z "$JOB_INFO" ]; then
        echo "No active job for $JOB_NAME"
        echo "Submit with: cx compute $JOB_NAME"
        read -p "Press Enter to retry, or Ctrl-C to exit..."
        continue
    fi

    JOBID=$(echo "$JOB_INFO" | awk '{print $1}')
    NODE=$(echo "$JOB_INFO" | awk '{print $2}')
    echo "Connecting to $NODE (job $JOBID)..."

    srun --jobid="$JOBID" --overlap --pty bash || true

    echo ""
    echo "Disconnected from $NODE."
    echo "Press Enter to reconnect, or Ctrl-C to exit..."
    read
done
```

#### 4.2 Compute-Aware Layouts

Create `~/.config/zellij/layouts/compute-project.kdl`:

```kdl
layout {
    default_tab_template {
        pane size=1 borderless=true {
            plugin location="tab-bar"
        }
        children
    }
    new_tab_template {
        pane size=1 borderless=true {
            plugin location="tab-bar"
        }
        pane
    }

    // Tab 1: Agent (connected to compute via srun)
    tab name="agent-1" focus=true {
        pane
        // After session starts, this pane runs on compute via srun --overlap
    }

    // Tab 2: Second agent
    tab name="agent-2" {
        pane
    }

    // Tab 3: Shell on compute
    tab name="shell" {
        pane
    }

    // Tab 4: MCP server (runs on compute, accesses Neo4j/Embed on other compute nodes)
    tab name="mcp" {
        pane
    }

    // Tab 5: Git (can run on login since it's lightweight)
    tab name="git" {
        pane
    }
}
```

#### 4.3 Multi-Node Tab Pattern

For advanced use cases where different tabs need different compute nodes:

```
Zellij session on Login Node
├── Tab "agent" → srun --jobid=JOB_A --overlap --pty bash (Sirius node)
├── Tab "build" → srun --jobid=JOB_B --overlap --pty bash (Sirius node, different allocation)
├── Tab "gpu"   → srun --jobid=JOB_C --overlap --pty bash (Titan GPU node)
└── Tab "local" → bash (login node, for lightweight tasks)
```

Each tab runs `cx-tab <job-name>` which handles the srun connection and
reconnection. This is the "script that gets run within the tab that launches
and connects to a SLURM job" pattern.

**Hybrid approach recommended**: The zellij server runs on the LOGIN node (for
stability and reconnection simplicity), but individual TABS connect to compute
nodes via srun. This gives:

- Reliable zellij server (login node, shared GPFS, no SLURM dependency)
- Compute resources per-tab (different nodes, different allocations)
- Graceful degradation (if a SLURM job dies, only that tab is affected)
- Simple reconnection (zellij session is always on login, tabs auto-reconnect)

This is a simpler and more robust pattern than running the zellij server itself on
a compute node, because:

1. No SLURM dependency for the session server
2. Login node zellij survives SLURM job failures
3. Different tabs can target different partitions/nodes
4. `cx iter <session>` just works (no compute orchestration for session attach)

#### 4.4 Verification

```bash
# Test 1: Start a compute allocation
sbatch --partition=sirius --time=01:00:00 --cpus-per-task=4 --mem=16G \
    --job-name=cx-test-tab --wrap="sleep 3600" --parsable

# Test 2: Inside a zellij tab, run cx-tab
cx-tab test-tab
# Verify: you get a shell on the compute node
hostname  # Should show compute node name

# Test 3: Open another tab, connect to same job
# (in new zellij tab)
cx-tab test-tab
# Both tabs share the same compute node

# Test 4: Disconnect resilience
# Kill the srun connection (Ctrl-C)
# cx-tab shows "Disconnected" and waits for Enter
# Press Enter → reconnects

# Test 5: Multi-node
sbatch --partition=titan --time=01:00:00 --gres=gpu:1 --mem=16G \
    --job-name=cx-gpu-tab --wrap="sleep 3600" --parsable
# In a different zellij tab:
cx-tab gpu-tab
# Verify: this tab is on the Titan GPU node
```

---

### Phase 5: Integration and Hardening

#### 5.1 Environment Setup on Compute

Compute nodes share GPFS but may not have identical environment setup. Ensure:

```bash
# In session-host.sh or cx-tab, before launching zellij:
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export TMPDIR="/tmp"  # Override SLURM TMPDIR issues

# Service discovery for agents running on compute
LITELLM_HOST=$(head -1 ~/.local/share/imas-codex/services/litellm-node.txt 2>/dev/null)
[ -n "$LITELLM_HOST" ] && export LITELLM_PROXY_URL="http://${LITELLM_HOST}"

NEO4J_HOST="98dci4-clu-2002"  # Or from service discovery
export NEO4J_URI="bolt://${NEO4J_HOST}:7687"
```

#### 5.2 cx sync Extension

Extend `cx sync` to also sync SLURM helpers:

```bash
# In cx sync subcommand:
scp ~/.local/lib/cx-slurm.sh "$host:~/.local/lib/cx-slurm.sh"
scp ~/.local/bin/cx-tab "$host:~/.local/bin/cx-tab"
```

#### 5.3 Monitoring Dashboard

Add a `cx compute status` command that shows:

```
Compute Sessions:
  imas-codex   sirius/98dci4-clu-5023   job=1139800   8 cores  32G   running 2d 4h
  efitpp       sirius/98dci4-clu-5044   job=1139801   18 cores 64G   running 6h

Services:
  LiteLLM      sirius/98dci4-clu-5010   job=1139799   :18400    running 5d
  Neo4j        rigel/98dci4-clu-2002    job=1132649   :7687     running 3d
  Embed        titan/98dci4-gpu-0002    job=1139179   :18765    running 1h
  Ollama       titan/98dci4-gpu-0002    job=1139180   :11434    running 1h

Login Node Load:
  1003: 2.3 (16 cores)  ← only SSH gateway + cx processes
  1006: 1.1 (48 cores)
```

#### 5.4 Comprehensive Verification

```bash
# === End-to-end test: Full chain from Windows ===

# 1. From Windows Terminal:
cx iter compute imas-codex
# Expected: SSH → login → sbatch → wait → srun → zellij with project layout

# 2. Inside zellij agent-1 tab:
# Start Claude Code or Copilot CLI
# Verify LLM calls work (through LiteLLM on compute)
# Verify MCP server works (connecting to Neo4j on rigel, embed on titan)

# 3. Disconnect test:
# Close Windows Terminal window
# Wait 30 seconds
# Open new Windows Terminal
cx iter compute imas-codex
# Expected: instant reconnect, all tabs preserved

# 4. Multi-session test:
cx iter compute efitpp
# Expected: separate SLURM job, separate compute node, separate zellij session

# 5. Cleanup test:
cx iter  # (regular login session)
cx compute stop imas-codex
cx compute stop efitpp
cx compute list  # Should be empty

# 6. Load verification:
# While compute sessions are running:
ssh iter "uptime"
# Expected: login node load < 5 (only SSH gateway)

# === Hybrid tab test ===

# 7. Inside a login-based zellij session:
cx iter imas-codex  # regular login zellij
# In one tab: cx-tab imas-codex  (connects to compute)
# In another tab: cx-tab gpu-tab (connects to Titan)
# In another tab: regular login shell
# Verify all three work simultaneously

# === Service connectivity from compute ===

# 8. From a compute zellij tab:
curl http://$(head -1 ~/.local/share/imas-codex/services/litellm-node.txt)/health
curl http://98dci4-clu-2002:7474  # Neo4j
curl http://98dci4-gpu-0002:18765/health  # Embed

# === Resilience tests ===

# 9. Kill and restart LiteLLM job:
scancel -n codex-llm
sbatch slurm/litellm-proxy.sh
# Verify agents reconnect after service discovery file updates

# 10. SLURM preemption (shouldn't happen with PreemptMode=OFF, but verify):
squeue -u $USER  # All jobs should remain RUNNING
```

---

## Recommended Architecture Decision

After investigating both patterns (zellij-on-compute vs. zellij-on-login with
compute tabs), **the hybrid approach (Phase 4) is recommended**:

| Pattern | Pros | Cons |
|---------|------|------|
| Zellij on compute (Phase 1-2) | Everything on compute, clean separation | SLURM dependency for session server; reconnect requires job lookup; session dies if job cancelled |
| **Zellij on login + compute tabs (Phase 4)** | **Robust session server; per-tab compute targeting; graceful degradation; simpler reconnect** | Login hosts lightweight zellij server; small login footprint remains |

The hybrid approach keeps the zellij server on the login node (tiny footprint: ~15MB
RSS) where it benefits from reliable uptime and simple `cx iter <session>` reconnection.
Heavy workloads (agents, compilation, MCP server) run in tabs connected to compute
nodes via srun. If a SLURM job dies, only that tab loses its connection — the session
and other tabs survive.

Implement Phase 1-2 first (full compute sessions) as a proof of concept, then
migrate to Phase 4 (hybrid) for production use.

---

## Future: GPU Cluster Considerations

When the proposed H200 GPU cluster is deployed, it will need internet access for:

1. **Training data download** — partner tokamak data from JET, TCV, JT-60SA
2. **Model weight download** — Hugging Face Hub, Ollama registry
3. **API access** — for hybrid local/cloud inference routing
4. **Package management** — PyPI, conda-forge for ML framework updates

The same DNS and firewall patterns observed on current compute nodes will apply.
See the companion report `plans/gpu-cluster-internet-justification.md` for the
detailed security/operational analysis.

---

## Dependencies and Prerequisites

| Dependency | Status | Action Required |
|------------|--------|-----------------|
| SLURM UNLIMITED time | ✅ Verified | None |
| srun --overlap | ✅ Verified | None |
| Zellij on GPFS | ✅ Verified (v0.44.0) | None |
| Compute → compute TCP | ✅ Verified | None |
| Login → compute TCP | ✅ Verified | None |
| Compute → api.anthropic.com | ✅ Verified | None |
| Compute → api.openrouter.ai | ❌ DNS fails | Use HOSTALIASES or direct Anthropic API |
| pam_slurm_adopt | ⚠️ Blocks SSH | Must use srun for compute access |
| TMPDIR on compute | ⚠️ Permission denied | Set `export TMPDIR=/tmp` in scripts |
