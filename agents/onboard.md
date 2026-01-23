# Facility Onboarding

Step-by-step guide for adding a new fusion facility to IMAS Codex.

## Prerequisites

Before starting:
1. SSH access configured (`~/.ssh/config` with host alias)
2. Neo4j running locally (`uv run imas-codex neo4j start`)
3. API keys set (`OPENROUTER_API_KEY` in `.env`)

## Quick Start (5-minute bootstrap)

```bash
# 1. Create facility config files
cat > imas_codex/config/facilities/jet.yaml << 'EOF'
# JET - Joint European Torus
#
# Public facility configuration for graph and mappings.
# Infrastructure data is in jet_private.yaml (gitignored).

facility: jet
name: Joint European Torus
machine: JET
description: Joint European Torus at Culham
location: Culham, United Kingdom

# SSH connection alias (configure in ~/.ssh/config)
ssh_host: jet

# Data systems available at this facility
data_systems:
  - ppf
  - jpf
  - mdsplus
EOF

# 2. Verify SSH access
ssh jet "hostname && whoami && pwd"

# 3. Run initial discovery
uv run imas-codex discover jet --cost-limit 1.0 --limit 50

# 4. Check progress
uv run imas-codex discovery status jet
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
Host jet
    HostName login.jet.example.org
    User your-username
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
ssh jet "hostname && pwd"

# Test fast tools (may not be installed yet)
ssh jet "which rg fd || echo 'Fast tools not available'"
```

## Phase 2: Infrastructure Discovery

### Step 2.1: Check Available Tools

```bash
uv run imas-codex tools check jet
```

If fast tools are missing:
```bash
# Install to ~/bin on remote system
uv run imas-codex tools install jet
```

### Step 2.2: Discover System Environment

Use MCP REPL to explore and persist:

```python
# Check OS, Python, and available tools
python("""
import socket
result = run('uname -a; python3 --version; which rg fd tree', facility='jet')
print(result)
""")

# Persist infrastructure findings
python("""
update_infrastructure('jet', {
    'os_info': {'name': 'RHEL', 'version': '8.9'},
    'python_info': {'version': '3.9.16'},
    'tools': {
        'rg': {'version': '14.1.1', 'path': '~/bin/rg'},
        'fd': {'version': '10.2.0', 'path': '~/bin/fd'}
    }
})
""")
```

### Step 2.3: Identify Key Paths

```bash
# Find code directories
ssh jet "ls -la /home; ls -la /work 2>/dev/null || echo 'No /work'"
ssh jet "find /home -maxdepth 2 -type d -name '*.py' 2>/dev/null | head -20"
```

Persist discovered paths:

```python
python("""
update_infrastructure('jet', {
    'paths': {
        'codes': {
            'user_scripts': '/home/users',
            'shared_codes': '/common/codes',
            'diagnostics': '/diagnostics/scripts'
        }
    },
    'actionable_paths': [
        {'path': '/common/codes', 'priority': 'high'},
        {'path': '/diagnostics/scripts', 'priority': 'medium'}
    ]
})
""")
```

## Phase 3: Discovery Pipeline

### Step 3.1: Seed Initial Paths

The discovery CLI auto-seeds from `actionable_paths` on first run:

```bash
# Run discovery with small budget to test
uv run imas-codex discover jet --cost-limit 1.0 --limit 50 -v
```

Or manually seed specific paths:

```python
python("""
add_to_graph('FacilityPath', [
    {'id': 'jet:/common/codes', 'path': '/common/codes',
     'facility_id': 'jet', 'path_type': 'code_directory', 
     'status': 'discovered', 'interest_score': 0.8},
    {'id': 'jet:/diagnostics', 'path': '/diagnostics',
     'facility_id': 'jet', 'path_type': 'code_directory',
     'status': 'discovered', 'interest_score': 0.7}
])
""")
```

### Step 3.2: Run Discovery

```bash
# Full discovery run
uv run imas-codex discover jet --cost-limit 10.0

# Monitor progress
uv run imas-codex discovery status jet
```

### Step 3.3: Handle Timeouts

If paths cause timeouts, persist the constraint immediately:

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

### Fast Tools Not Working

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
- [ ] Fast tools installed (optional but recommended)
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
