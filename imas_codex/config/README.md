# Facility Exploration Guide

This guide teaches how to explore remote fusion facilities via SSH.

## Quick Start

1. Read the facility config to get the SSH host alias
2. SSH directly and run commands  
3. Capture findings using the CLI

```bash
# Read facility config
cat imas_codex/config/facilities/epfl.yaml

# SSH using the host alias from your ~/.ssh/config
ssh epfl "which python3; python3 --version"

# Capture findings when done
uv run imas-codex epfl --capture environment << 'EOF'
python:
  version: "3.9.21"
  path: "/usr/bin/python3"
EOF
```

## Batch Commands (Critical for Performance)

Each SSH round-trip takes ~200ms. **Batch commands to minimize latency.**

Use `;`, `&&`, `||`, and pipes:

```bash
# Good: batch multiple checks (200ms total)
ssh epfl "which python3; python3 --version; pip list 2>/dev/null | head -20"

# Good: check with fallbacks
ssh epfl "which rg || which grep; cat /etc/os-release | head -5"

# Good: environment + packages together  
ssh epfl "cat /etc/os-release; rpm -qa | grep -E 'python|numpy' | head -20"

# Bad: separate calls (600ms total)
ssh epfl "which python3"
ssh epfl "python3 --version"  
ssh epfl "pip list | head -20"
```

## Exploration Categories

### Environment

Discover Python, OS, compilers, and module systems:

```bash
ssh epfl << 'EOF'
echo "=== Python ==="
which python python3 2>/dev/null
python3 --version 2>/dev/null
pip list 2>/dev/null | head -20

echo "=== OS ==="
cat /etc/os-release | head -10
uname -a

echo "=== Compilers ==="
gcc --version 2>/dev/null | head -1
gfortran --version 2>/dev/null | head -1
which icx ifx icc icpc 2>/dev/null
icx --version 2>/dev/null | head -1

echo "=== Module System ==="
type module 2>/dev/null && module avail 2>&1 | head -30 || echo "No module system"
EOF
```

### Tools

Check CLI tool availability:

```bash
ssh epfl << 'EOF'
echo "=== Search Tools ==="
for tool in rg ag grep find; do
    loc=$(which $tool 2>/dev/null)
    if [ -n "$loc" ]; then echo "$tool: $loc"; else echo "$tool: not found"; fi
done

echo "=== Text Processing ==="
for tool in awk sed jq; do
    which $tool 2>/dev/null && echo "  -> available" || echo "$tool: not found"
done

echo "=== File Tools ==="
for tool in tree file stat; do
    which $tool 2>/dev/null && echo "  -> available" || echo "$tool: not found"
done

echo "=== Editors ==="
for tool in vim nano emacs; do
    which $tool 2>/dev/null && echo "  -> available" || echo "$tool: not found"
done

echo "=== Build Tools ==="
for tool in make cmake ninja; do
    which $tool 2>/dev/null && echo "  -> available" || echo "$tool: not found"
done

echo "=== Data Tools ==="
for tool in h5dump ncdump mdstcl; do
    which $tool 2>/dev/null && echo "  -> available" || echo "$tool: not found"
done

echo "=== Job Schedulers ==="
for tool in sbatch srun qsub bsub; do
    which $tool 2>/dev/null && echo "  -> available" || echo "$tool: not found"
done
EOF
```

### Filesystem

Explore directory structure:

```bash
ssh epfl << 'EOF'
echo "=== Common Paths ==="
for path in /common /opt /usr/local; do
    ls -la $path 2>/dev/null | head -10 && echo "---" || echo "$path: not accessible"
done

echo "=== File Type Summary ==="
for ext in py m sh h5 nc mat; do
    count=$(find /common -name "*.$ext" 2>/dev/null | wc -l)
    echo "  .$ext files: $count"
done

echo "=== Tree Structure (if available) ==="
tree -L 2 /common 2>/dev/null || find /common -maxdepth 2 -type d 2>/dev/null | head -30
EOF
```

### Data Systems

Discover MDSplus, HDF5, NetCDF:

```bash
ssh epfl << 'EOF'
echo "=== MDSplus ==="
python3 -c "import MDSplus; print('Python bindings: OK')" 2>/dev/null || echo "MDSplus Python: not available"
which mdstcl 2>/dev/null && echo "mdstcl: available" || echo "mdstcl: not available"

echo "=== HDF5 ==="
which h5dump 2>/dev/null && echo "h5dump: available" || echo "h5dump: not available"
echo "HDF5 files found:"
find /common -name '*.h5' 2>/dev/null | wc -l

echo "=== NetCDF ==="
which ncdump 2>/dev/null && echo "ncdump: available" || echo "ncdump: not available"
echo "NetCDF files found:"
find /common -name '*.nc' 2>/dev/null | wc -l

echo "=== Other Formats ==="
for ext in mat csv json; do
    count=$(find /common -name "*.$ext" 2>/dev/null | wc -l)
    echo "  .$ext files: $count"
done
EOF
```

## Safety Rules

**DO NOT run these commands on remote facilities:**

- `rm`, `rmdir` — no file deletion
- `mv` — no file renaming/moving  
- `chmod`, `chown`, `chgrp` — no permission changes
- `sudo`, `su` — no privilege escalation
- `kill`, `pkill`, `killall` — no process termination
- `shutdown`, `reboot`, `halt` — no system control
- `mkfs`, `fdisk`, `dd` — no disk operations
- `iptables`, `firewall-cmd` — no network changes

**Safe operations:**

- Reading files: `cat`, `head`, `tail`, `less`, `grep`
- Listing: `ls`, `find`, `tree`, `du`, `df`
- System info: `uname`, `hostname`, `whoami`, `id`
- Environment: `env`, `printenv`, `echo $VAR`
- Package queries: `rpm -qa`, `pip list`, `conda list`
- Module system: `module avail/list/load/show`

## Capturing Findings

Persist exploration results using typed artifacts. The system validates against Pydantic models.

### Artifact Types

| Type | Schema | Purpose |
|------|--------|---------|
| `environment` | `imas_codex/discovery/models/environment.py` | Python, OS, compilers, modules |
| `tools` | `imas_codex/discovery/models/tools.py` | CLI tool availability |
| `filesystem` | `imas_codex/discovery/models/filesystem.py` | Directory structure, paths |
| `data` | `imas_codex/discovery/models/data.py` | MDSplus, HDF5, NetCDF patterns |

### Capture Command

```bash
uv run imas-codex <facility> --capture <type> << 'EOF'
# YAML content matching the schema
EOF
```

### Environment Artifact Example

```bash
uv run imas-codex epfl --capture environment << 'EOF'
python:
  version: "3.9.21"
  path: "/usr/bin/python3"
  packages:
    - "numpy==1.23.5"
    - "matplotlib==3.4.3"
os:
  name: "RHEL"
  version: "9.6"
  kernel: "5.14.0-427.el9.x86_64"
compilers:
  - name: "gcc"
    version: "11.5.0"
  - name: "icx"
    version: "2025.2.1"
module_system:
  available: false
notes:
  - "Intel oneAPI 2025.2 auto-loaded via /etc/profile.d"
EOF
```

### Tools Artifact Example

```bash
uv run imas-codex epfl --capture tools << 'EOF'
tools:
  - name: "rg"
    available: false
  - name: "grep"
    available: true
    path: "/usr/bin/grep"
  - name: "tree"
    available: true
    path: "/usr/bin/tree"
  - name: "h5dump"
    available: false
notes:
  - "No ripgrep, use grep -r instead"
EOF
```

### Filesystem Artifact Example

```bash
uv run imas-codex epfl --capture filesystem << 'EOF'
important_paths:
  - path: "/common/tcv/data"
    purpose: "TCV shot data"
    path_type: "data"
  - path: "/common/tcv/codes"
    purpose: "Analysis codes"
    path_type: "code"
notes:
  - "Codes directory doesn't exist, data in /common/tcv/results"
EOF
```

### Data Artifact Example

```bash
uv run imas-codex epfl --capture data << 'EOF'
mdsplus:
  available: true
  server: "tcvdata.epfl.ch"
  trees:
    - "tcv"
    - "tcv_shot"
    - "results"
  python_bindings: true
hdf5:
  available: false
  h5dump_available: false
notes:
  - "MDSplus is primary data system"
EOF
```

### View Captured Artifacts

```bash
# List all artifacts for a facility
uv run imas-codex epfl --artifacts

# View a specific artifact
uv run imas-codex epfl --artifact environment
```

## Tool Preferences

When searching, prefer in order:
1. `rg` (ripgrep) — fastest, if available
2. `ag` (silver searcher) — fast alternative
3. `grep -r` — universal fallback

When listing directories:
1. `tree` — structured output, if available
2. `find` — POSIX compatible
3. `ls -la` — basic fallback

## Facility Configs

Facility configurations are in `imas_codex/config/facilities/`:

```yaml
# Example: epfl.yaml
facility: epfl
ssh_host: epfl          # SSH alias from ~/.ssh/config
description: Swiss Plasma Center - TCV Tokamak
paths:
  data:
    - /common/tcv/data
  code:
    - /common/tcv/codes
known_systems:
  mdsplus:
    server: tcvdata.epfl.ch
    trees: [tcv, tcv_shot, results]
knowledge:              # Accumulated exploration findings
  tools:
    rg: unavailable
    grep: available
  python:
    version: "3.9.21"
```

Read the config before exploring to see what's already known.
