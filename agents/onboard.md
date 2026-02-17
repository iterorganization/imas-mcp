# Facility Onboarding

Step-by-step guide for adding a new fusion facility to IMAS Codex.

## Prerequisites

Before starting:
1. SSH access configured (`~/.ssh/config` with host alias)
2. Neo4j running locally (`uv run imas-codex neo4j start`)
3. API keys set (`OPENROUTER_API_KEY` in `.env`)

## Quick Start (5-minute bootstrap)

```bash
# 1. Create facility config file
cat > imas_codex/config/facilities/myfacility.yaml << 'EOF'
facility: myfacility
name: My Facility Full Name
machine: TOKAMAK
description: Brief description
location: City, Country
ssh_host: myfacility
data_systems:
  - imas
  - mdsplus
EOF

# 2. Verify SSH access
ssh myfacility "hostname && whoami && pwd"

# 3. Check and install remote tools
uv run imas-codex tools check myfacility
uv run imas-codex tools install myfacility

# 4. Populate infrastructure via MCP (from python() REPL)
# See Phase 2 below

# 5. Run initial discovery
uv run imas-codex discover myfacility --cost-limit 1.0 --limit 50
```

## Phase 1: Configuration Setup

### Step 1.1: Create Public Config

Create `imas_codex/config/facilities/<facility>.yaml`:

```yaml
# <FACILITY> - <Full Name>
#
# Public facility configuration for graph and mappings.
# Infrastructure data is in <facility>_private.yaml (gitignored).

facility: jet  # Must match filename
name: Joint European Torus
machine: JET
description: Brief description of the facility
location: City, Country

# SSH connection alias (user configures ~/.ssh/config)
ssh_host: jet

# Data systems available at this facility
data_systems:
  - ppf      # JET Processed data Files
  - jpf      # JET Pulse Files
  - mdsplus  # If applicable

# Wiki/documentation sites (optional)
wiki_sites:
  - url: https://wiki.example.org
    portal_page: Main_Page
    site_type: mediawiki  # or "confluence"
    auth_type: none       # or "ssh_proxy", "basic"
```

### Step 1.2: Configure SSH Access

Add to `~/.ssh/config`:

```
Host myfacility
    HostName login.facility.org
    User your-username
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
```

For facilities behind jump hosts (ProxyJump):
```
Host myfacility
    HostName localhost
    Port 12345
    User your-username
    ProxyJump gateway.facility.org
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
```

Create the socket directory:
```bash
mkdir -p ~/.ssh/sockets
chmod 700 ~/.ssh/sockets
```

### Step 1.3: Verify Connectivity

```bash
# Test SSH
ssh myfacility "hostname && pwd"

# Check remote tools status
uv run imas-codex tools check myfacility
```

## Phase 2: Remote Tools Installation

### Step 2.1: Install Required Tools

The CLI handles all installation automatically:

```bash
# Check current status
uv run imas-codex tools check myfacility

# Install required tools only (rg, fd)
uv run imas-codex tools install myfacility --required-only

# Install all tools (recommended)
uv run imas-codex tools install myfacility
```

Tools are installed to `~/bin` on the remote system. The installer:
- Auto-detects architecture (x86_64/aarch64)
- Downloads pre-built binaries from GitHub releases
- Verifies installation with version check

### Step 2.2: Verify PATH Configuration

The tools require `~/bin` to be in PATH. Check:

```bash
ssh myfacility 'echo $PATH | grep -o "$HOME/bin" || echo "~/bin NOT in PATH"'
```

If not in PATH, add to `~/.bashrc`:
```bash
ssh myfacility 'echo "export PATH=\$HOME/bin:\$PATH" >> ~/.bashrc'
```

### Step 2.3: Handle Module Systems

Many HPC facilities use environment modules. If Python or other tools need modules:

```bash
# Check if modules persist over SSH
ssh -T myfacility 'module list 2>&1'

# If not, add to ~/.bashrc
ssh myfacility 'echo "module load python/3.11" >> ~/.bashrc'
```

**Note:** Bash reads `.bashrc` even with `ssh -T` for non-login remote commands.

## Phase 3: Infrastructure Discovery

### Step 3.1: Gather System Information

Use SSH to gather grounded information:

```bash
# Gather OS info
ssh myfacility "uname -a; cat /etc/os-release | head -5"

# Gather Python info
ssh myfacility "which python3; python3 --version"

# Find key paths
ssh myfacility "ls -la /common /work /home 2>/dev/null | head -30"
```

### Step 3.2: Persist via MCP Tools

Use the MCP tools to persist infrastructure (creates private YAML automatically):

```python
# Update infrastructure (private, gitignored)
update_facility_infrastructure("myfacility", {
    "hostname": "compute-node-001",
    "os": {
        "name": "RHEL",
        "version": "8.9",
        "kernel": "4.18.0-513.el8.x86_64"
    },
    "python": {
        "version": "3.11.5",
        "path": "/usr/bin/python3"
    },
    "paths": {
        "imas": {"root": "/common/IMAS"},
        "codes": {"shared": "/common/codes"}
    }
})

# Update tool versions
update_facility_tools("myfacility", {
    "rg": {"version": "14.1.1", "path": "/home/user/bin/rg", "purpose": "Fast pattern search"},
    "fd": {"version": "10.2.0", "path": "/home/user/bin/fd", "purpose": "Fast file finder"}
})

# Add exploration notes
add_exploration_note("myfacility", "Initial setup complete, IMAS at /common/IMAS")
```

### Step 3.3: Verify Configuration

```python
# Read back infrastructure
get_facility_infrastructure("myfacility")
```

## Phase 4: Discovery Pipeline

### Step 4.1: Seed Initial Paths

```python
python("""
add_to_graph('FacilityPath', [
    {'id': 'myfacility:/common/codes', 'path': '/common/codes',
     'facility_id': 'myfacility', 'path_type': 'code_directory', 
     'status': 'discovered', 'interest_score': 0.8}
])
""")
```

### Step 4.2: Run Discovery

```bash
# Discovery with cost limit
uv run imas-codex discover myfacility --cost-limit 10.0

# Monitor progress
uv run imas-codex discovery status myfacility
```

### Step 4.3: Handle Timeouts

If paths cause timeouts, persist constraints immediately:

```python
python("""
# Record problematic path
update_infrastructure('jet', {
    'excludes': {
        'large_dirs': ['/archive', '/raw_data'],
        'depth_limits': {
            '/home': 3,
            '/work': 2
        }
    }
})
add_exploration_note('jet', '/archive too large - excluded from discovery')
""")
```

## Phase 4: Data System Ingestion

### Step 4.1: MDSplus Trees (if applicable)

```bash
# Discover MDSplus trees
uv run imas-codex mdsplus discover jet

# Ingest tree structure
uv run imas-codex mdsplus ingest jet magnetics --shot 12345
```

### Step 4.2: PPF/JPF (JET-specific)

For JET, PPF and JPF require custom ingestion:

```python
python("""
# Query available diagnostics
result = run('ls /common/ppf/diagnostics', facility='jet')
print(result)

# Add discovered diagnostics
add_to_graph('Diagnostic', [
    {'id': 'jet:hrts', 'name': 'HRTS', 'facility_id': 'jet',
     'category': 'spectroscopy', 'description': 'High Resolution Thomson Scattering'}
])
""")
```

### Step 4.3: Code Ingestion

After discovery finds high-value files:

```bash
# Queue discovered files
uv run imas-codex ingest queue jet

# Run ingestion
uv run imas-codex ingest run jet --limit 100
```

## Phase 5: Enrichment & Mapping

### Step 5.1: Enrich High-Value Nodes

```bash
# Run enrichment agent on TreeNodes
uv run imas-codex enrich jet --filter "score > 0.7"
```

### Step 5.2: Generate IMAS Mappings

```python
python("""
# Find potential IMAS mappings
candidates = search_code('equilibrium psi axis', facility='jet')
for c in candidates[:5]:
    print(f'{c.path}: {c.description}')
""")
```

## Configuration Reference

### Public Config (`<facility>.yaml`)

| Field | Required | Description |
|-------|----------|-------------|
| `facility` | ✅ | Short ID (must match filename) |
| `name` | ✅ | Full institution name |
| `machine` | ✅ | Tokamak/device name |
| `description` | ✅ | Brief description |
| `location` | ✅ | City, Country |
| `ssh_host` | ✅ | SSH host alias |
| `data_systems` | ✅ | List of data systems |
| `wiki_sites` | ❌ | List of wiki configurations |

### Private Config (`<facility>_private.yaml`)

| Field | Description |
|-------|-------------|
| `paths.codes.*` | Code directory paths |
| `paths.data.*` | Data directory paths |
| `tools.*` | Tool availability and versions |
| `os_info` | Operating system details |
| `python_info` | Python environment |
| `excludes.large_dirs` | Paths to skip entirely |
| `excludes.depth_limits` | Max depth per path |
| `exploration_notes` | Timestamped observations |
| `actionable_paths` | Seed paths for discovery |

## Troubleshooting

### SSH Connection Issues

```bash
# Debug SSH
ssh -v jet "hostname"

# Check SSH config
grep -A5 "Host jet" ~/.ssh/config
```

### Discovery Not Finding Files

```bash
# Check facility is configured
uv run python -c "from imas_codex.discovery.facility import get_facility; print(get_facility('jet'))"

# Verify SSH works through the tool
uv run python -c "from imas_codex.remote.tools import run; print(run('ls -la', facility='jet'))"
```

### Remote Tools Not Working

```bash
# Check installation
ssh jet "ls -la ~/bin/rg ~/bin/fd"

# Verify PATH
ssh jet "echo \$PATH | tr ':' '\n' | grep bin"

# Use absolute path
ssh jet "~/bin/rg --version"
```

## Checklist

- [ ] Public config created (`<facility>.yaml`)
- [ ] SSH access verified
- [ ] Remote tools installed (optional but recommended)
- [ ] Infrastructure discovered and persisted
- [ ] Initial paths seeded
- [ ] Discovery run with cost limit
- [ ] Timeout constraints recorded
- [ ] Data systems catalogued
- [ ] High-value code ingested
- [ ] IMAS mappings identified

## Related Documentation

- [Explore Workflows](explore.md) - Detailed exploration patterns
- [Ingest Workflows](ingest.md) - Code ingestion pipeline
- [Graph Operations](graph.md) - Knowledge graph queries
- [Facility Access Architecture](../docs/architecture/facility-access.md) - Technical deep-dive
