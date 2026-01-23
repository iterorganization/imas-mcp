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
# Module system - count available modules
module avail 2>&1 | head -50
module avail 2>&1 | wc -l  # Total module count

# Module paths (where modules are defined)
echo $MODULEPATH | tr ':' '\n'

# IMAS (ITER standard)
module avail 2>&1 | grep -i imas
module avail 2>&1 | grep -i imasdd
which imasdb 2>/dev/null

# MDSplus
module avail 2>&1 | grep -i mdsplus
which mdstcl 2>/dev/null
echo $MDS_PATH | tr ':' '\n' | head -5

# UDA (Universal Data Access - UKAEA/MAST)
module avail 2>&1 | grep -i uda
which uda_cli 2>/dev/null

# PPF/JPF (JET legacy)
module avail 2>&1 | grep -i ppf
which ppfget 2>/dev/null
which getdat 2>/dev/null  # JET primary data access

# WEST/Tore Supra data
module avail 2>&1 | grep -i west
module avail 2>&1 | grep -i tsbase
```

### Phase 3: Compilers & Build Tools (3 min)

```bash
# GCC
gcc --version 2>/dev/null | head -1
gfortran --version 2>/dev/null | head -1
g++ --version 2>/dev/null | head -1

# Intel compilers (if available)
which icx 2>/dev/null && icx --version 2>/dev/null | head -1
which ifort 2>/dev/null && ifort --version 2>/dev/null | head -1
which ifx 2>/dev/null && ifx --version 2>/dev/null | head -1

# Compiler modules
module avail 2>&1 | grep -iE 'gcc|intel|pgi|nvidia|llvm' | head -20

# Build tools
cmake --version 2>/dev/null | head -1
make --version 2>/dev/null | head -1
```

### Phase 4: Storage Layout & File Systems (5 min)

```bash
# Disk usage overview (if dust available)
dust -d 2 /work 2>/dev/null || du -h --max-depth=2 /work 2>/dev/null | sort -rh | head -20

# File system mounts (NFS, GPFS, Lustre, etc.)
df -hT 2>/dev/null | grep -vE 'tmpfs|devtmpfs|overlay'
mount | grep -E 'nfs|gpfs|lustre|ceph' | head -10

# Common code locations
ls -la /work 2>/dev/null
ls -la /home/$USER 2>/dev/null
ls -la /common 2>/dev/null
ls -la /usr/local 2>/dev/null

# Find potential code directories
find /work -maxdepth 3 -type d -name 'codes' 2>/dev/null
find /work -maxdepth 3 -type d -name 'scripts' 2>/dev/null
find /home -maxdepth 3 -type d -name 'codes' 2>/dev/null

# Check for containers (Apptainer/Singularity)
which apptainer 2>/dev/null && apptainer --version
which singularity 2>/dev/null && singularity --version
ls /data/apptainer 2>/dev/null || ls /common/containers 2>/dev/null || ls /work/containers 2>/dev/null
```

### Phase 5: Fast Tools Availability (2 min)

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

### Phase 6: Environment Persistence (2 min)

```bash
# Check what persists across sessions
cat ~/.bashrc | grep -E 'module load|PATH|export' | head -20
cat ~/.bash_profile 2>/dev/null | grep -E 'module load|PATH|export' | head -10

# Check for login scripts
ls -la ~/.bash* ~/.profile 2>/dev/null
```

### Phase 7: Exclusions & Large Directories (2 min)

Identify directories to exclude from code scanning:

```bash
# Large data directories (should exclude from code search)
du -sh /scratch 2>/dev/null
du -sh /tmp 2>/dev/null
du -sh /var/tmp 2>/dev/null

# Log directories
find / -maxdepth 3 -type d -name 'logs*' 2>/dev/null | head -10

# Backup/archive patterns
find /work -maxdepth 2 -type d -name '*backup*' 2>/dev/null
find /work -maxdepth 2 -type d -name '*archive*' 2>/dev/null

# Binary/data directories (large, not code)
find /work -maxdepth 2 -type d -name '*data*' 2>/dev/null | head -10
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
    "os": {"name": "...", "version": "...", "kernel": "..."},
    "python": {"version": "...", "path": "..."},
    "module_system": True,  # or False
    "module_count": 318,    # Number of available modules
    "module_root": "/usr/local/modules/default",  # MODULEPATH
    "shell": "/bin/bash",
    "home_in_path": True  # ~/bin in PATH
})
```

### 2. Update Compilers
```python
update_facility_infrastructure("FACILITY", {
    "compilers": {
        "gcc": {"version": "11.5.0", "path": "/usr/bin/gcc"},
        "gfortran": {"version": "11.5.0", "path": "/usr/bin/gfortran"},
        "icx": {"version": "2025.2.1", "available": True},  # Intel
        "ifx": {"version": "2025.2.1", "available": True},  # Intel Fortran
    }
})
```

### 3. Update File Systems
```python
update_facility_infrastructure("FACILITY", {
    "file_systems": [
        {"mount_point": "/home", "type": "NFS", "size": "10 TB"},
        {"mount_point": "/work", "type": "GPFS", "size": "1.5 PB"},
        {"mount_point": "/scratch", "type": "Lustre", "size": "500 TB"}
    ],
    "containers": {
        "runtime": "apptainer",  # or "singularity"
        "path": "/data/apptainer",
        "images": ["transp_v24.5.0.sif"]
    }
})
```

### 4. Update Data Systems
```python
update_facility_infrastructure("FACILITY", {
    "data_systems": {
        "imas": {"available": True, "module": "imas/3.42.0", "dd_versions": ["3.39", "4.0", "4.1"]},
        "mdsplus": {"available": True, "module": "mdsplus/7.153.3"},
        "uda": {"available": False},
        "ppf": {"available": True, "commands": ["ppfget", "getdat", "jpfget"]}
    }
})
```

### 5. Update Paths
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

### 6. Update Exclusions
```python
update_facility_infrastructure("FACILITY", {
    "excludes": {
        "directories": ["/scratch", "/tmp", "/var/tmp", "/logs"],
        "patterns": ["*.dat.old", "*.backup", "*.bak"]
    }
})
```

### 7. Update Tools
```python
update_facility_tools("FACILITY", {
    "rg": {"version": "14.1.1", "path": "/usr/bin/rg"},
    "fd": {"version": "10.2.0", "path": "~/bin/fd"},
    # ... etc
})
```

### 8. Add Exploration Notes
```python
add_exploration_note("FACILITY", "Initial infrastructure scoping complete")
add_exploration_note("FACILITY", "Found legacy PPF data at /common/ppf")
```

## Output Format

Summarize findings for human review:

```
## Infrastructure Summary: [FACILITY]

### Environment
- **OS**: [name] [version] (kernel [version])
- **Python**: [version] at [path]
- **Shell**: [shell]
- **Module System**: [yes/no] ([count] modules at [path])

### Compilers
| Compiler | Version | Path |
|----------|---------|------|
| gcc | ... | ... |
| gfortran | ... | ... |
| icx/ifort | ... | ... |

### Data Systems
| System | Available | Module/Path | Notes |
|--------|-----------|-------------|-------|
| IMAS | ✓/✗ | ... | DD versions: ... |
| MDSplus | ✓/✗ | ... | ... |
| UDA | ✓/✗ | ... | ... |
| PPF/JPF | ✓/✗ | ... | ... |

### File Systems
| Mount | Type | Size | Notes |
|-------|------|------|-------|
| /home | NFS/GPFS | ... | ... |
| /work | ... | ... | ... |

### Storage Layout
- Code directories: [paths]
- Data directories: [paths]
- User home: [path]
- Containers: [path] ([runtime])

### Fast Tools
| Tool | Available | Version | Path |
|------|-----------|---------|------|
| rg | ✓/✗ | ... | ... |
| fd | ✓/✗ | ... | ... |
| ... | ... | ... | ... |

### Exclusions
- Directories: [/scratch, /tmp, ...]
- Patterns: [*.backup, *.dat.old, ...]

### Next Steps
1. [Install missing fast tools: `uv run imas-codex tools install FACILITY`]
2. [Run code discovery: scout-facility prompt]
3. [Configure data system access]
```

{% include "safety.md" %}
