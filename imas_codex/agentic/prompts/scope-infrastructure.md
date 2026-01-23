---
name: scope-infrastructure
description: Facility-agnostic infrastructure discovery for fusion data systems
---

You are scoping a new fusion facility's infrastructure. Your goal is to gather comprehensive information about the computing environment and data systems so it can be persisted to the knowledge graph.

## Mission

Systematically discover infrastructure at {{ facility | default("the target facility") }} to enable:
- Fast tool installation and verification
- Data system integration (IMAS, MDSplus, UDA, PPF/JPF, etc.)
- Code discovery and ingestion pipelines

## Discovery Phases

### Phase 1: SSH Connectivity & Basic Environment (2 min)

```bash
# Verify connection and basic info
hostname && uname -a
cat /etc/*release* 2>/dev/null | head -10

# Python environment
which python3 && python3 --version
which python && python --version 2>/dev/null

# Shell and PATH
echo $SHELL
echo $PATH | tr ':' '\n' | head -10
```

### Phase 2: Module System & Data Systems (5 min)

Check for each data system. Not all facilities have all systems.

```bash
# Module system
module avail 2>&1 | head -50

# IMAS (ITER standard)
module avail 2>&1 | grep -i imas
module avail 2>&1 | grep -i imasdd
which imasdb 2>/dev/null

# MDSplus
module avail 2>&1 | grep -i mdsplus
which mdstcl 2>/dev/null
echo '$MDS_PATH' | tr ':' '\n' | head -5

# UDA (Universal Data Access - UKAEA/MAST)
module avail 2>&1 | grep -i uda
which uda_cli 2>/dev/null

# PPF/JPF (JET legacy)
module avail 2>&1 | grep -i ppf
which ppfget 2>/dev/null

# WEST/Tore Supra data
module avail 2>&1 | grep -i west
module avail 2>&1 | grep -i tsbase
```

### Phase 3: Storage Layout (5 min)

```bash
# Disk usage overview (if dust available)
dust -d 2 /work 2>/dev/null || du -h --max-depth=2 /work 2>/dev/null | sort -rh | head -20

# Common code locations
ls -la /work 2>/dev/null
ls -la /home/$USER 2>/dev/null
ls -la /common 2>/dev/null
ls -la /usr/local 2>/dev/null

# Find potential code directories
find /work -maxdepth 3 -type d -name 'codes' 2>/dev/null
find /work -maxdepth 3 -type d -name 'scripts' 2>/dev/null
find /home -maxdepth 3 -type d -name 'codes' 2>/dev/null
```

### Phase 4: Fast Tools Availability (2 min)

```bash
# Required tools
rg --version 2>/dev/null
fd --version 2>/dev/null

# Optional tools
tokei --version 2>/dev/null
scc --version 2>/dev/null
dust --version 2>/dev/null
bat --version 2>/dev/null
fzf --version 2>/dev/null
yq --version 2>/dev/null
jq --version 2>/dev/null
eza --version 2>/dev/null
delta --version 2>/dev/null

# User bin directory for tool installation
ls -la ~/bin 2>/dev/null
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc  # If ~/bin not in PATH
```

### Phase 5: Environment Persistence (2 min)

```bash
# Check what persists across sessions
cat ~/.bashrc | grep -E 'module load|PATH|export' | head -20
cat ~/.bash_profile 2>/dev/null | grep -E 'module load|PATH|export' | head -10

# Check for login scripts
ls -la ~/.bash* ~/.profile 2>/dev/null
```

## Data Systems Reference

| System | Facility | Key Commands | Check For |
|--------|----------|--------------|-----------|
| IMAS | ITER, EUROfusion | `module load imas`, `imasdb` | `imas.DBEntry` |
| MDSplus | Most US/EU facilities | `module load mdsplus`, `mdstcl` | `MDSplus.Tree` |
| UDA | MAST-U, UKAEA | `uda_cli`, `pyuda` | `pyuda.Client` |
| PPF/JPF | JET | `ppfget`, `jpfget` | Legacy binary format |
| WEST DB | WEST | `module load west` | Tore Supra heritage |
| D3D GA | DIII-D | `gadat`, MDSplus | General Atomics |
| ASDEX-U | IPP | MDSplus + IMAS | Shotfile system |

## Persistence Requirements

After discovery, persist using MCP tools:

### 1. Update Infrastructure
```python
update_facility_infrastructure("FACILITY", {
    "os": {"name": "...", "version": "..."},
    "python": {"version": "...", "path": "..."},
    "module_system": True,  # or False
    "shell": "/bin/bash",
    "home_in_path": True  # ~/bin in PATH
})
```

### 2. Update Data Systems
```python
update_facility_infrastructure("FACILITY", {
    "data_systems": {
        "imas": {"available": True, "module": "imas/3.42.0"},
        "mdsplus": {"available": True, "module": "mdsplus/..."},
        "uda": {"available": False},
        "ppf": {"available": True, "commands": ["ppfget", "jpfget"]}
    }
})
```

### 3. Update Paths
```python
update_facility_paths("FACILITY", {
    "codes": {
        "main": "/work/codes",
        "user": "/home/user/codes"
    },
    "data": {
        "shots": "/common/shots",
        "imas": "/common/imas/shared"
    }
})
```

### 4. Update Tools
```python
update_facility_tools("FACILITY", {
    "rg": {"version": "14.1.1", "path": "/usr/bin/rg"},
    "fd": {"version": "10.2.0", "path": "~/bin/fd"},
    # ... etc
})
```

### 5. Add Exploration Notes
```python
add_exploration_note("FACILITY", "Initial infrastructure scoping complete")
add_exploration_note("FACILITY", "Found legacy PPF data at /common/ppf")
```

## Output Format

Summarize findings for human review:

```
## Infrastructure Summary: [FACILITY]

### Environment
- **OS**: [name] [version]
- **Python**: [version] at [path]
- **Shell**: [shell]
- **Module System**: [yes/no]

### Data Systems
| System | Available | Module/Path | Notes |
|--------|-----------|-------------|-------|
| IMAS | ✓/✗ | ... | ... |
| MDSplus | ✓/✗ | ... | ... |
| UDA | ✓/✗ | ... | ... |
| PPF/JPF | ✓/✗ | ... | ... |

### Storage Layout
- Code directories: [paths]
- Data directories: [paths]
- User home: [path]

### Fast Tools
| Tool | Available | Version | Path |
|------|-----------|---------|------|
| rg | ✓/✗ | ... | ... |
| fd | ✓/✗ | ... | ... |
| ... | ... | ... | ... |

### Next Steps
1. [Install missing fast tools: `uv run imas-codex tools install FACILITY`]
2. [Run code discovery: scout-facility prompt]
3. [Configure data system access]
```

{% include "safety.md" %}
