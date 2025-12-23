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
├── epfl.yaml                    # Public - machine, data systems
└── epfl_infrastructure.yaml     # Private - paths, tools, OS, IPs
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
# epfl.yaml - PUBLIC (version controlled)
facility: epfl
name: École Polytechnique Fédérale de Lausanne
machine: TCV
description: Swiss Plasma Center - TCV Tokamak
location: Lausanne, Switzerland

# SSH connection alias (user configures ~/.ssh/config)
ssh_host: epfl

# Data systems at this facility
data_systems:
  - mdsplus
  - tdi
```

## Infrastructure File

Sensitive data for exploration agents:

```yaml
# epfl_infrastructure.yaml - PRIVATE (gitignored)
facility_id: epfl
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
cat imas_codex/config/facilities/epfl.yaml

# SSH using the host alias
ssh epfl "which python3; python3 --version; pip list | head -10"
```

### 3. Persist Findings

**After every exploration session, persist ALL discoveries:**

#### Persistence Checklist

| Discovery Type | Where to Persist | Tool |
|----------------|------------------|------|
| Analysis codes (name, version, type) | Graph | `ingest_node("AnalysisCode", {...})` |
| Code installation paths | Graph | `ingest_node("AnalysisCode", {path: ...})` |
| Diagnostics | Graph | `ingest_node("Diagnostic", {...})` |
| MDSplus trees | Graph | `ingest_node("MDSplusTree", {...})` |
| TDI functions | Graph | `ingest_node("TDIFunction", {...})` |
| Rich directory paths (e.g., `/home/codes`) | Infrastructure | `update_infrastructure(...)` |
| OS/kernel versions | Infrastructure | `update_infrastructure(...)` |
| Tool availability | Infrastructure | `update_infrastructure(...)` |
| SVN/Git repos discovered | Infrastructure | `update_infrastructure(...)` |
| Unstructured findings | `_Discovery` nodes | `cypher("CREATE (:_Discovery {...})")` |

#### Examples

Use the Agents MCP server tools:

```python
# For sensitive data (local only, never graphed)
update_infrastructure("epfl", {
    "tools": {
        "rg": {"status": "unavailable"},
        "grep": {"status": "available"}
    },
    "notes": ["MDSplus config at /usr/local/mdsplus/local/mdsplus.conf"]
})

# For public data semantics (goes to graph)
ingest_node("Diagnostic", {
    "name": "XRCS",
    "facility_id": "epfl",
    "category": "spectroscopy"
})

# For unstructured discoveries (staging area)
cypher('''
    CREATE (d:_Discovery {
        facility: 'epfl',
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

## Structured Exploration Approach

### Tool Preferences

Before exploring, check infrastructure for available tools:

```yaml
# In infrastructure file
exploration:
  agent_instructions:
    search_tools:
      preferred: [rg, fd]  # Use these if available
      system: [grep, find]  # Fallback
      commands:
        - ~/bin/rg   # User-installed ripgrep
        - ~/bin/fd   # User-installed fd
      notes: "Use rg (ripgrep) and fd from $HOME/bin for fast searching"
```

Always read `infrastructure.knowledge.tools` first to know what's available:
- If `rg` is available, use `~/bin/rg` instead of `grep -r`
- If `fd` is available, use `~/bin/fd` instead of `find`

### Tracking Explored Paths

Avoid re-treading searched paths by recording them:

```yaml
# In infrastructure file
exploration:
  explored_paths:
    codes:
      searched: true
      equilibrium:
        paths: [/home/codes/liuqe, /home/codes/helena]
        searched: true
  search_queries_run:
    - "rg -l 'equilibrium' /home/codes --max-depth 4 -g '*.py'"
```

Before starting exploration:
1. Read infrastructure to see what's already been explored
2. Check `explored_paths` for directories already searched
3. Check `search_queries_run` for queries already executed
4. Focus on unexplored areas

After exploration:
1. Update `explored_paths` with new directories searched
2. Append new queries to `search_queries_run`
3. Persist all discoveries immediately (don't batch for later)

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
