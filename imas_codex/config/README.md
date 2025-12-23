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

## Sensitive Data Policy

**Never commit or graph sensitive data:**

| Data Type | Sensitivity | Reason | Where to Store |
|-----------|-------------|--------|----------------|
| Hostnames/IPs | 🔴 High | Network reconnaissance | `_infrastructure.yaml` |
| NFS mounts | 🔴 High | Network topology | `_infrastructure.yaml` |
| OS/kernel versions | 🔴 High | CVE matching | `_infrastructure.yaml` |
| File paths | 🟡 Medium | Filesystem enumeration | `_infrastructure.yaml` |
| Tool availability | 🟡 Medium | Reconnaissance | `_infrastructure.yaml` |
| Python/compiler versions | 🟡 Medium | Vulnerability targeting | `_infrastructure.yaml` |
| MDSplus tree names | 🟢 Low | Data semantics | `<facility>.yaml` |
| Diagnostic names | 🟢 Low | Data semantics | `<facility>.yaml` |
| TDI function names | 🟢 Low | Data semantics | `<facility>.yaml` |

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

Save infrastructure discoveries to the `_infrastructure.yaml` file:

```bash
# Capture updates infrastructure file (validates against schema)
uv run imas-codex epfl --capture tools << 'EOF'
tools:
  rg: unavailable
  grep: available
  h5dump: unavailable
EOF
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
