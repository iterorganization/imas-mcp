# Configuration Directory

This directory contains all runtime configuration for imas-codex.

## Directory Structure

```
config/
├── README.md                    # This file
├── facilities/                  # Per-facility configuration
│   ├── tcv.yaml               # Public (git tracked)
│   ├── tcv_private.yaml       # Private (gitignored)
│   ├── iter.yaml
│   ├── iter_private.yaml
│   ├── jet.yaml
│   └── jet_private.yaml
├── patterns/                    # Discovery patterns
│   ├── exclude.yaml            # Exclusion patterns (deterministic, fast)
│   ├── file_types.yaml         # File extension categories
│   ├── discovery.yaml          # (deprecated)
│   └── scoring/                # LLM scoring patterns
│       ├── base.yaml           # Dimension weights
│       ├── data_systems.yaml   # IMAS, MDSplus, etc.
│       └── physics.yaml        # Physics domain patterns
└── remote_tools.yaml              # Fast CLI tools configuration
```

## Fast CLI Tools

Fast Rust-based tools for code exploration are defined in `remote_tools.yaml`.

### CLI Commands

```bash
# Check tool availability
imas-codex tools check              # Local
imas-codex tools check iter         # ITER (auto-detects local)
imas-codex tools check tcv         # EPFL (via SSH)

# Install missing tools
imas-codex tools install            # Install locally
imas-codex tools install tcv       # Install on EPFL
imas-codex tools install --dry-run  # Show what would be installed

# List available tools
imas-codex tools list
```

### Python API

```python
# Via MCP python() REPL
python("print(check_tools('tcv'))")
python("print(run('rg pattern', facility='tcv'))")
python("result = setup_tools('tcv'); print(result.summary)")
```

### Auto-Detection

The `run()` function auto-detects whether to use SSH:
- Compares facility's `ssh_host` to current hostname
- If running on the target machine, executes locally
- Otherwise uses SSH

Example: On SDCC, `run('rg pattern', facility='iter')` runs locally.

## YAML Organization Principles

The project uses YAML files organized by purpose:

| Directory | Purpose | Composable? | Version Controlled? |
|-----------|---------|-------------|---------------------|
| `schemas/` | LinkML data models (source of truth) | No | Yes |
| `definitions/` | Static domain knowledge | Yes (by domain) | Yes |
| `config/` | Runtime configuration | Yes (per-facility) | Partial |

### What Goes Where?

- **`schemas/`**: Data model definitions - defines node types and relationships
- **`definitions/`**: Static knowledge - physics domains, cluster labels, units
- **`config/`**: Runtime config - facility connections, discovery patterns, tools

---

# Facility Configuration Guide

## File Structure

Each facility has two configuration files:

| File | Visibility | Content | Schema |
|------|------------|---------|--------|
| `<facility>.yaml` | ✅ Git tracked | Public data semantics | `facility.yaml` |
| `<facility>_infrastructure.yaml` | ❌ Gitignored | Sensitive infrastructure | `facility_infrastructure.yaml` |

**Example for EPFL:**
```
config/facilities/
├── tcv.yaml                    # Public - machine, data systems
└── tcv_infrastructure.yaml     # Private - paths, tools, OS, IPs
```

## Data Classification Policy

**Rule**: If the LinkML schema (`facility.yaml`) has a property for it, store it in the graph. Otherwise, use infrastructure files.

### Graph (Public) - Data access semantics

| Data Type | Schema Property | Purpose |
|-----------|-----------------|---------|
| MDSplus tree names | `MDSplusTree.name` | Data discovery |
| Diagnostic names | `Diagnostic.name` | Data discovery |
| Analysis code names | `AnalysisCode.name` | Code discovery |
| Code versions | `AnalysisCode.version` | Reproducibility |
| Code paths | `AnalysisCode.path` | Data access |
| TDI function names | `TDIFunction.name` | Data access |

### Infrastructure (Private) - Operational/security data

| Data Type | Why Private |
|-----------|-------------|
| Hostnames, IPs, NFS mounts | Network reconnaissance risk |
| OS/kernel versions | CVE matching risk |
| System tool availability | Reconnaissance risk |
| Rich directory paths | Exploration guidance |
| User home directories | Privacy |
| Credentials, tokens | Security |

## Public Facility File

Minimal configuration for graph building and connections:

```yaml
# tcv.yaml - PUBLIC (version controlled)
facility: tcv
name: École Polytechnique Fédérale de Lausanne
machine: TCV
description: Swiss Plasma Center - TCV Tokamak
location: Lausanne, Switzerland

# SSH connection alias (user configures ~/.ssh/config)
ssh_host: tcv

# Data systems at this facility
data_systems:
  - mdsplus
  - tdi
```

## Infrastructure File

Sensitive data for exploration agents:

```yaml
# tcv_infrastructure.yaml - PRIVATE (gitignored)
facility_id: tcv
last_explored: 2025-01-15T10:30:00Z

# Network infrastructure
nfs_mounts:
  - source: "10.27.128.167:/usr/local/CRPP/tdi"
    target: /usr/local/CRPP/tdi
    options: ro

# File paths
paths:
  tdi:
    root: /usr/local/CRPP/tdi
    tcv: /usr/local/CRPP/tdi/tcv

# Operating system (CVE-sensitive)
os:
  name: RHEL
  version: "9.6"
  kernel: 5.14.0-570.el9

# Tool availability
tools:
  rg: unavailable
  grep: available
  tree: available

# Python environment
python:
  version: 3.9.21
  path: /usr/bin/python3
  packages:
    - numpy==1.23.5
    - MDSplus

# Agent guidance notes
notes:
  - "No ripgrep - use grep -r instead"
  - "MDSplus config at /usr/local/mdsplus/local/mdsplus.conf"
```

## Exploration Workflow

### 1. Load Both Files

When exploring, agents should load and merge both files:

```python
# Load public config
with open(f"config/facilities/{facility}.yaml") as f:
    public = yaml.safe_load(f)

# Load infrastructure if exists
infra_path = f"config/facilities/{facility}_infrastructure.yaml"
if Path(infra_path).exists():
    with open(infra_path) as f:
        infrastructure = yaml.safe_load(f)
```

### 2. SSH Exploration

Use batched commands for efficiency:

```bash
# Read facility config first
cat imas_codex/config/facilities/tcv.yaml

# SSH using the host alias
ssh tcv "which python3; python3 --version; pip list | head -10"
```

### 3. Persist Findings

**After every exploration session, persist ALL discoveries:**

#### Persistence Checklist

| Discovery Type | Where to Persist | Tool |
|----------------|------------------|------|
| Analysis codes (name, version, type) | Graph | `ingest_nodes("AnalysisCode", [...])` |
| Directory paths for exploration | Graph | `ingest_nodes("FacilityPath", [...])` |
| Diagnostics | Graph | `ingest_nodes("Diagnostic", [...])` |
| MDSplus trees | Graph | `ingest_nodes("MDSplusTree", [...])` |
| TDI functions | Graph | `ingest_nodes("TDIFunction", [...])` |
| OS/kernel versions | Infrastructure | `update_infrastructure(...)` |
| Tool availability | Infrastructure | `update_infrastructure(...)` |
| Unstructured findings | `_Discovery` nodes | `cypher("CREATE (:_Discovery {...})")` |

#### Examples

Use the Agents MCP server tools:

```python
# For sensitive data (local only, never graphed)
update_infrastructure("tcv", {
    "tools": {
        "rg": {"status": "unavailable"},
        "grep": {"status": "available"}
    },
    "notes": ["MDSplus config at /usr/local/mdsplus/local/mdsplus.conf"]
})

# For public data semantics (goes to graph, always use list)
ingest_nodes("Diagnostic", [
    {"name": "XRCS", "facility_id": "tcv", "category": "spectroscopy"},
    {"name": "Thomson", "facility_id": "tcv", "category": "spectroscopy"},
])

# For unstructured discoveries (staging area)
cypher('''
    CREATE (d:_Discovery {
        facility: 'tcv',
        type: 'unknown_tree',
        name: 'tcv_raw',
        discovered_at: datetime()
    })
''')
```

## Safety Rules

**Safe operations only on remote facilities:**
- Reading: `cat`, `head`, `tail`, `less`, `grep`
- Listing: `ls`, `find`, `tree`, `du`, `df`
- System info: `uname`, `hostname`, `whoami`
- Package queries: `rpm -qa`, `pip list`

**Never run:**
- File modification: `rm`, `mv`, `chmod`
- Privilege escalation: `sudo`, `su`
- System control: `kill`, `shutdown`, `reboot`

## Structured Discovery Workflow

### FacilityPath Nodes

Use `FacilityPath` nodes to track discovery state in the graph.

### Two-Phase Discovery Pipeline

The discovery pipeline uses parallel workers for scanning (file enumeration) and scoring (LLM evaluation):

```
┌─────────────────────┐        ┌─────────────────────┐
│    SCAN WORKERS     │        │   SCORE WORKERS     │
│                     │        │                     │
│ discovered          │        │ listed              │
│     ↓ (listing)     │        │     ↓ (scoring)     │
│  listed             │───────>│  scored             │
│     or              │        │     or              │
│  excluded           │        │  skipped            │
└─────────────────────┘        └─────────────────────┘
```

**Scan Phase**: Enumerate directory contents
```python
# Start with discovered paths (seeded from facility config)
# Scanner worker claims paths, enumerates files/dirs, updates counts

# After enumeration:
add_to_graph("FacilityPath", [{
    "id": "tcv:/home/codes/transport",
    "status": "listed",  # Ready for scoring
    "file_count": 15,
    "dir_count": 3,
    "last_examined": "2025-01-15T10:30:00Z"
}])

# Paths matching exclusion patterns:
add_to_graph("FacilityPath", [{
    "id": "tcv:/home/user/.cache",
    "status": "excluded",  # Never scored
}])
```

**Score Phase**: LLM evaluates directory value
```python
# Scorer worker claims listed paths, evaluates with LLM

# After scoring:
add_to_graph("FacilityPath", [{
    "id": "tcv:/home/codes/transport",
    "status": "scored",
    "score": 0.85,
    "score_imas": 0.9,
    "score_code": 0.8,
    "notes": "IMAS integration, equilibrium code"
}])

# Low-value paths:
add_to_graph("FacilityPath", [{
    "id": "tcv:/home/user/random_scripts",
    "status": "skipped",
    "score": 0.15,
}])
```

### Path Status Values (LinkML Schema)

| Status | Phase | Meaning |
|--------|-------|---------|
| `discovered` | Seed | Found, awaiting enumeration |
| `listing` | Scan | Scanner worker active (transient) |
| `listed` | Scan→Score | Enumerated, awaiting LLM score |
| `scoring` | Score | Scorer worker active (transient) |
| `scored` | Score | LLM evaluated, has score |
| `skipped` | Score | Low value (score < 0.2) |
| `excluded` | Scan | Matched exclusion pattern |
| `stale` | Any | Path may have changed |

Transient states (`listing`, `scoring`) auto-recover to previous state on timeout.

### Score Guidelines

| Score | Use Case |
|-------|----------|
| 0.9+ | IMAS integration, IDS read/write |
| 0.7+ | MDSplus access, equilibrium codes |
| 0.5+ | General analysis codes |
| 0.3+ | Utilities, helpers |
| <0.3 | Config files, documentation → may be skipped |

### Tool Preferences

Check `get_facility(facility)` for available tools:
- `tools.rg` - ripgrep version (if installed)
- `tools.fd` - fd version (if installed)
- `paths.user_tools.bin` - where user tools are installed

Use fast tools when available:
```bash
# If rg available at ~/bin/rg
ssh tcv "~/bin/rg -l 'pattern' /path --max-depth 4 -g '*.py'"

# Fallback to grep
ssh tcv "grep -r 'pattern' /path --include='*.py'"
```

## Graph vs Local Storage

| Operation | Uses Public File | Uses Infrastructure File |
|-----------|------------------|--------------------------|
| Graph building | ✅ | ❌ Never |
| Recipe generation | ✅ | ❌ |
| SSH connection | ✅ (ssh_host) | ❌ |
| Agent exploration | ✅ | ✅ (local context) |
| LLM prompts | ✅ | ✅ (merged for context) |

The infrastructure file provides **agent guidance** for exploration but is never:
- Committed to git
- Loaded into the graph
- Distributed in OCI artifacts

---

# Discovery Exclusion Strategy

Discovery uses a **two-tier filtering** approach:

## Exclusion Philosophy

**Exclude only directories with NO scientific value.** Let the scanner discover 
everything else, and let the LLM scoring decide value. A high score doesn't mean 
we download everything - it just marks the directory as valuable.

### What to EXCLUDE (deterministic, before scanning)

| Category | Examples | Rationale |
|----------|----------|-----------|
| **System directories** | `/proc`, `/sys`, `/tmp`, `/var/log` | No user code |
| **Build artifacts** | `__pycache__`, `node_modules`, `build/` | Generated files |
| **Personal media** | `Desktop`, `Downloads`, `Music`, `Pictures`, `Videos` | XDG dirs |
| **Scratch/temp** | `scratch`, `tmp`, `temp` directories | Transient job files |
| **Caches** | `.cache`, `.local`, `.conda` | Environment state |
| **Archives** | `.tar.gz`, `.zip` | Opaque containers |

### What to NEVER EXCLUDE (let LLM score)

| Category | Examples | Why Valuable |
|----------|----------|--------------|
| **Video diagnostics** | `videodata`, `fast_camera` | Fusion diagnostic data! |
| **Simulation outputs** | `large_runs`, `run_outputs` | Physics results |
| **Documents** | `Documents` | May contain code/scripts |
| **Archives/backups** | `old`, `backup` | Legacy code with algorithms |
| **Data directories** | `data`, `shots`, `pulses` | Shot data files |

## Tier 1: Deterministic Exclusion (BEFORE scanning)

Fast, zero-cost, no SSH/LLM overhead. Defined in `patterns/exclude.yaml`
and merged with facility-specific excludes from `*_private.yaml`.

### Scratch Pattern Detection

Scratch directories are detected by:
1. **Name matching**: `scratch`, `tmp`, `temp`, `SCRATCH`, etc.
2. **Path patterns**: `*/scratch/*`, `*/SCRATCH/*`, `*/tmp/*`

## Tier 2: LLM Scoring (AFTER scanning)

For nuanced decisions requiring directory content context:
- "backup", "old" directories (may contain valuable code)
- Purpose classification: physics_code, data_files, test_suite

## Facility-Specific Exclusions

Each facility's `*_private.yaml` can define exclusions for true noise only:

```yaml
excludes:
  path_prefixes:
    - /mnt/HPC_T2/ITER/HPC/scratch  # HPC scratch (ephemeral)
  patterns:
    - "*.dat.old.bak"                # Double backup suffix
  # NOTE: Do NOT add videodata, large_runs, results, etc.
  # These may contain valuable fusion data!
```

These are merged via `get_exclusion_config_for_facility(facility)`.

## API Usage

```python
from imas_codex.config.discovery_config import (
    get_discovery_config,
    get_exclusion_config_for_facility,
)

# Base config (no facility-specific merging)
config = get_discovery_config()

# With facility-specific excludes merged
exclusions = get_exclusion_config_for_facility("iter")
should_exclude, reason = exclusions.should_exclude("/mnt/scratch/user")
is_scratch = exclusions.is_scratch_path("/mnt/HPC_T2/ITER/HPC/scratch/users")

# Test that scientific data is NOT excluded
exclusions.should_exclude("/data/videodata")  # (False, None) - diagnostic data!
exclusions.should_exclude("/work/large_runs") # (False, None) - simulation outputs
```

## Adding New Exclusions

1. **Global exclusions**: Add to `patterns/exclude.yaml`
2. **Facility-specific**: Add to `facilities/<facility>_private.yaml` under `excludes:`
3. **Clear cache after changes**: `clear_config_cache()`
