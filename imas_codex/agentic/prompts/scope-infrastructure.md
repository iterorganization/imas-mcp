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

### Phase 2.5: Data Storage & Access Methods (5 min)

Discover where shot data is stored and how it's accessed. This is critical for data integration.

```bash
# Check for IMAS databases (shared and user)
ls -la /work/imas/shared/imasdb 2>/dev/null | head -20    # ITER shared
ls -la /common/imas 2>/dev/null | head -10                # JET/EUROfusion
ls -la ~/public/imasdb 2>/dev/null | head -10             # User databases

# After loading IMAS module (if available):
# env | grep -iE 'imas|imasdb' | head -20

# Check for MDSplus tree paths
env | grep -iE 'mds|tree|_path' | head -30
echo $MDS_PATH | tr ';' '\n' | head -10

# Check for PPF/JPF (JET legacy)
env | grep -iE 'ppf|jpf' | head -10
ls -la /common/EFDA-DATA-PPF-JPF 2>/dev/null | head -10

# Check MATLAB data access paths
echo $MATLABPATH | tr ':' '\n' | grep -iE 'ppf|jpf|imas' | head -10
```

**Document for each data system:**
1. **Storage location**: Where is shot/pulse data physically stored?
2. **Access method**: API calls (imas.DBEntry, MDSplus.Tree, getdat)
3. **File format**: HDF5, MDSplus trees, binary (PPF/JPF)
4. **Server**: Remote data servers (e.g., tcvdata::, ppfhost.jetdata.eu)

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

**CRITICAL**: Fusion facilities have petabyte-scale storage. NEVER run `dust`, `du`, or recursive `find` on `/work`, `/scratch`, or shared filesystems - these will hang indefinitely.

```bash
# File system mounts - use df (instant, no recursion)
df -hT 2>/dev/null | grep -vE 'tmpfs|devtmpfs|overlay'
mount | grep -E 'nfs|gpfs|lustre|ceph' | head -10

# Directory structure - use ls, NOT du/dust/find on large dirs
ls -la /work 2>/dev/null | head -20          # Top-level only
ls -la /home/$USER 2>/dev/null | head -20    # User home
ls -la /common 2>/dev/null | head -20        # Common area
ls -la /usr/local 2>/dev/null | head -20     # Local installs

# Check for containers (Apptainer/Singularity)
which apptainer 2>/dev/null && apptainer --version
which singularity 2>/dev/null && singularity --version
ls /data/apptainer 2>/dev/null | head -10
ls /common/containers 2>/dev/null | head -10
```

**Safe code discovery** - only in user home or known small directories:
```bash
# Only search in user home (bounded size)
find ~/ -maxdepth 3 -type d -name 'codes' 2>/dev/null | head -10
ls -la ~/codes 2>/dev/null
ls -la ~/projects 2>/dev/null
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

### Phase 6.5: User Home Directory Pattern (1 min)

Discover the home directory structure for user identification. This is critical
for linking FacilityPath nodes to FacilityUser nodes via OWNS relationships.

```bash
# Check home directory structure
ls -d /home/*/ 2>/dev/null | head -10

# Some facilities use nested structure like /home/ITER/username
# Check for nested patterns:
ls -d /home/*/*/ 2>/dev/null | head -5

# Standard pattern: /home/username         → "^/home/([a-z0-9_-]+)(?:/|$)"
# ITER pattern:     /home/ITER/username    → "^/home/ITER/([a-z0-9_-]+)(?:/|$)"
# EPFL pattern:     /home/username         → standard
# Some use:         /users/username        → "^/users/([a-z0-9_-]+)(?:/|$)"

# Verify by checking a known home directory
echo ~  # Current user's home path
ls -d ~/../*/ 2>/dev/null | head -5  # Peer directories
```

**Important**: If `/home/<facility>/username` is used instead of `/home/username`,
you must configure `user_info.home_path_pattern` in the facility YAML. See Phase 10.

### Phase 7: Internal Git Servers (1 min)

Discover internal git servers (GitLab, GitHub Enterprise). Code on these servers
will have children skipped during discovery (content is version-controlled).

```bash
# Check git config for internal remotes
git config --global -l 2>/dev/null | grep -i url
cat ~/.gitconfig 2>/dev/null | grep -iE 'url|insteadof' | head -10

# Check for known internal server patterns (substitute facility-specific domains)
# ITER: git.iter.org
# EPFL: gitlab.epfl.ch  
# IPP: gitlab.mpcdf.mpg.de
# EUROfusion: git.euro-fusion.org

# Sample git remotes from code directories (if any exist)
find ~/codes ~/projects ~/*codes* -maxdepth 3 -name '.git' 2>/dev/null | head -5 | while read gitdir; do
  git -C "$(dirname "$gitdir")" remote -v 2>/dev/null | grep fetch | head -1
done
```

### Phase 8: Exclusions & Large Directories (2 min)

Identify directories to exclude from code scanning. **Do NOT run du/dust on shared filesystems.**

```bash
# Check if common exclusion paths exist (don't measure size)
ls -d /scratch 2>/dev/null && echo "EXCLUDE: /scratch exists"
ls -d /tmp 2>/dev/null && echo "EXCLUDE: /tmp exists"  
ls -d /var/tmp 2>/dev/null && echo "EXCLUDE: /var/tmp exists"

# Only measure small local directories
du -sh /tmp 2>/dev/null   # Usually local, bounded

# Standard exclusions (don't search, just document)
# - /scratch, /tmp, /var/tmp (temporary files)
# - */logs*, */log (log files)
# - *backup*, *archive* (backups)
# - *.dat, *.h5, *.nc (data files, not code)
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
    },
    "data_storage": {
        "imas_databases": {
            "shared": "/work/imas/shared/imasdb",  # Shared databases
            "user": "~/public/imasdb",              # User databases
            "machines": ["ITER_SCENARIOS", "JET", "AMNS"]  # Available machines
        },
        "mdsplus_trees": {
            "server": "tcvdata::",                  # Remote server
            "paths": ["/tcvssd/trees", "/Terra16/mdsplus/trees"],
            "tree_types": ["tcv_shot", "results", "magnetics"]
        },
        "ppf_jpf": {
            "server": "ppfhost.jetdata.eu",
            "matlab_path": "/jet/share32/matlab/ppf",
            "access_command": "getdat"
        }
    },
    "file_formats": {
        "primary": "HDF5",           # or "MDSplus", "binary"
        "backends": ["HDF5_BACKEND", "MDSPLUS_BACKEND"],  # IMAS backends
        "extensions": [".h5", ".hdf5", ".nc", ".tree"]
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

### 6. Update Discovery Roots
Explicit list of root paths that seed the discovery frontier. When set, these
paths are used directly instead of extracting roots from the nested `paths:` structure.
This gives you precise control over where automated discovery starts.

```python
update_facility_infrastructure("FACILITY", {
    "discovery_roots": [
        "/home/codes",           # Primary code repository
        "/usr/local/CRPP/tdi",   # TDI function libraries
        "/common/imas/shared"    # Shared IMAS workflows
    ]
})
```

**When to use discovery_roots:**
- To limit discovery to specific high-value directories
- When the nested `paths:` structure doesn't reflect actual discovery priorities
- To add roots not captured in other path categories

If `discovery_roots` is empty or unset, seeding falls back to extracting paths
from the `paths:` structure (codes.root, tdi.root, etc.).

### 7. Update Exclusions
```python
update_facility_infrastructure("FACILITY", {
    "excludes": {
        "directories": ["/scratch", "/tmp", "/var/tmp", "/logs"],
        "patterns": ["*.dat.old", "*.backup", "*.bak"]
    }
})
```

### 9. Update Internal Git Servers
If the facility has internal GitLab/GitHub servers that we can access, add them
so code discovery skips walking their children (content is version-controlled):
```python
update_facility_infrastructure("FACILITY", {
    "internal_git_servers": ["git.iter.org", "gitlab.epfl.ch"]
})
```

### 10. Update User Info (Home Directory Pattern)
**CRITICAL** for user identification and OWNS relationships. If the facility uses
a non-standard home directory structure, you must configure the pattern:

```python
# Standard: /home/username (no config needed - this is the default)
# Non-standard examples requiring config:

# ITER uses /home/ITER/username:
update_facility_infrastructure("ITER", {
    "user_info": {
        "name_format": "last_first",  # GECOS format: "Last First [EXT]"
        "gecos_suffix_pattern": "\\s+EXT$",  # Strip " EXT" from names
        "home_path_pattern": "^/home/ITER/([a-z0-9_-]+)(?:/|$)"
    }
})

# If /users/username instead of /home/username:
update_facility_infrastructure("FACILITY", {
    "user_info": {
        "home_path_pattern": "^/users/([a-z0-9_-]+)(?:/|$)"
    }
})
```

The pattern must:
- Match the full path from start (`^`)
- Capture the username in group 1 `([a-z0-9_-]+)`
- End with optional `/` or end-of-string `(?:/|$)`

**Default patterns (no config needed):**
- `/home/username`
- `/users/username`
- `/u/username`
- `/work/username`

### 10. Update Tools
```python
update_facility_tools("FACILITY", {
    "rg": {"version": "14.1.1", "path": "/usr/bin/rg"},
    "fd": {"version": "10.2.0", "path": "~/bin/fd"},
    # ... etc
})
```

### 11. Add Exploration Notes
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
