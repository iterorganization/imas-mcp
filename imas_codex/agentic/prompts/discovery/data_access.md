---
name: data_access
description: Discover all data access methods at a facility
dynamic: true
---

# Data Access Discovery for {{ facility | default("the target facility") }}

## Objective

Discover **all ways to programmatically access experimental data** at this facility.
Create `AccessMethod` graph nodes for agent discoverability and update infrastructure
with operational setup details.

## Design Principle: Graph as Single Source of Truth

AccessMethod nodes in the graph contain **everything** needed to load data:
- Import statements
- Module/environment setup commands
- Connection and data retrieval templates
- Test shots for validation

**Why?** The target architecture has `imas-ambix` serving deterministic mappings 
from facility signals → IMAS. These mappings must be self-contained and executable
from the graph alone, without reading infrastructure files.

Infrastructure YAML is for internal operational config (SSH hosts, paths) - 
never for data access patterns that external tools need.

{% if existing_access_methods %}
## Reference: Existing AccessMethod Nodes

Use these as templates for new facilities:

{% for method in existing_access_methods %}
### {{ method.name }} ({{ method.facility }})

```
ID: {{ method.id }}
Type: {{ method.method_type }}
Library: {{ method.library }}
```

{% if method.data_template %}
**Data template:** `{{ method.data_template }}`
{% endif %}

{% endfor %}
{% endif %}

## Discovery Checklist

### 1. Check Module System

```bash
ssh {{ ssh_host | default('{facility}') }} "source /etc/profile.d/modules.sh 2>/dev/null && module avail 2>&1 | grep -iE 'python|mdsplus|imas|uda|ppf|sal|hdf5|matlab|idl' | head -30"
```

**Record:** Module names and versions for each data system.

### 2. Probe Python Packages Per Module

For each Python module version found:

```bash
ssh {{ ssh_host | default('{facility}') }} 'source /etc/profile.d/modules.sh; module purge; module load python/X.Y; python3 << "EOF"
import sys
print(f"Python: {sys.version}")
for pkg in ["MDSplus", "imas", "pyuda", "h5py"]:
    try:
        m = __import__(pkg)
        print(f"{pkg}: AVAILABLE - {getattr(m, '__version__', 'unknown')}")
    except ImportError:
        print(f"{pkg}: not found")

# Check facility-specific packages
for pkg in ["jet.data.sal", "tcv", "aug"]:
    try:
        m = __import__(pkg, fromlist=[""])
        print(f"{pkg}: AVAILABLE")
    except ImportError:
        pass
EOF'
```

### 3. Find Data Access Libraries (All Languages)

```bash
# MATLAB
ssh {{ ssh_host | default('{facility}') }} "find /usr/local /opt /common /jet 2>/dev/null -path '*matlab*ppf*' -o -path '*matlab*mds*' 2>/dev/null | head -10"

# IDL
ssh {{ ssh_host | default('{facility}') }} "find /usr/local /opt /common /jet 2>/dev/null -name '*idl*' -type d | grep -iE 'ppf|mds|data' | head -10"

# Python packages
ssh {{ ssh_host | default('{facility}') }} "pip3 show MDSplus imas pyuda 2>/dev/null | grep -E 'Name|Version|Location'"

# CLI tools
ssh {{ ssh_host | default('{facility}') }} "which getdat mdsvalue ppfget 2>/dev/null"
```

### 4. Test Each Method

For each discovered method, validate with a test shot:

```python
# Example validation pattern
ssh {{ ssh_host | default('{facility}') }} 'source /etc/profile.d/modules.sh; module load python/3.9; python3 << "EOF"
{imports_template}
{connection_template}
data = {data_template}  # Use known test shot
print(f"Success: type={type(data).__name__}, shape={getattr(data, 'shape', 'N/A')}")
EOF'
```

### 5. Find Documentation

```bash
ssh {{ ssh_host | default('{facility}') }} "find /common /usr/local -name '*guide*' -o -name '*manual*' -o -name '*tutorial*' 2>/dev/null | grep -iE 'data|ppf|mds|imas' | head -10"
```

## Output Requirements

### Create Graph Nodes (Self-Contained)

{% include "schema/access-method.md" %}

Each AccessMethod must contain **all information** needed to execute data retrieval:

```python
mcp_codex_add_to_graph(node_type="AccessMethod", data=[
    {
        "id": "{{ facility | default('{facility}') }}:{method_type}:{variant}",
        "facility_id": "{{ facility | default('{facility}') }}",
        "name": "{Human Readable Name}",
        "method_type": "{mdsplus|rest|uda|imas|hdf5|matlab|idl|cli}",
        "library": "{import target or path}",
        "access_type": "local",
        
        # Environment setup (previously in infrastructure)
        "setup_commands": ["source /etc/profile.d/modules.sh", "module load python/3.9"],
        "environment_variables": {"MDSPLUS_SERVER": "server.facility.edu"},
        
        # Code templates
        "imports_template": "{import statements}",
        "connection_template": "{connection/setup code}",
        "data_template": "{data retrieval with {shot}, {signal} placeholders}",
        "time_template": "{time axis retrieval if applicable}",
        
        # Validation
        "data_source": "{default data source name}",
        "discovery_shot": {valid_test_shot_number},
        "full_example": "{complete working code example}",
        "verified_date": "{YYYY-MM-DD}",
        
        # Documentation
        "documentation_url": "{wiki or doc URL}",
        "documentation_local": "{path on facility}"
    }
])
```

**Critical:** The `setup_commands` field must include module loads - this is what
makes the AccessMethod self-contained and usable by imas-ambix mappings.

## Known Patterns by Data System

### MDSplus Facilities (TCV, DIII-D, NSTX-U, W7-X)

**Python:**
```python
import MDSplus
tree = MDSplus.Tree('{tree_name}', {shot}, 'readonly')
data = tree.tdiExecute('data({path})').data()
time = tree.tdiExecute('dim_of({path})').data()
```

**MATLAB:**
```matlab
mdsconnect('server.facility.edu')
mdsopen('{tree_name}', {shot})
data = mdsvalue('\\{path}')
```

**TDI functions** provide semantic wrappers (e.g., `tcv_eq('i_p')`, `efit_*`).

### PPF/SAL Facilities (JET)

**Python (SAL - recommended):**
```python
from jet.data import sal
data = sal.get('/pulse/{shot}/ppf/signal/{owner}/{dda}/{dtype}')
```

**MATLAB:**
```matlab
addpath('/jet/share32/matlab/ppf')
ppfgo
[data, x, t, units] = ppfget({shot}, '{dda}', '{dtype}');
```

**IDL:**
```idl
@idlppf
ppfgo
ppfget, {shot}, '{dda}', '{dtype}', data, x, t
```

### IMAS Facilities (ITER, emerging standard)

**Python:**
```python
import imas
entry = imas.DBEntry(imas.imasdef.MDSPLUS_BACKEND, '{database}', {shot}, 1)
entry.open()
ids = entry.get('{ids_name}')
```

### UDA Facilities (MAST-U)

**Python:**
```python
import pyuda
client = pyuda.Client()
data = client.get('{signal}', {shot})
```

## Relationship to Other Discovery Phases

| Phase | Focus | Data Access Role |
|-------|-------|------------------|
| **Infrastructure Scoping** | OS, paths, SSH config | Internal operational setup |
| **Discovery Roots** | Directory seeding | Identify `data_access` category paths |
| **Data Access Discovery** | Full method templates | Create self-contained graph nodes |

**Important:** Infrastructure is for internal SSH/operational config only.
All data access patterns go into AccessMethod graph nodes so that external tools
(like imas-ambix) can query and execute them without reading our config files.

Add an exploration note after completion:

```python
mcp_codex_add_exploration_note(
    facility="{{ facility | default('{facility}') }}",
    note="Data access methods documented: {list methods}. Primary: {recommended}. Validated with shot {shot}."
)
```

## Completion Criteria

- [ ] All language bindings discovered (Python, MATLAB, IDL, Fortran, CLI)
- [ ] Each method tested with a valid shot
- [ ] AccessMethod nodes created with **complete** setup_commands
- [ ] Documentation URLs recorded in nodes
- [ ] Exploration notes added
