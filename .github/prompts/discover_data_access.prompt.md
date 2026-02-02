---
agent: agent
---

# Data Access Discovery

Discover all data access methods at {facility} and create graph nodes.

## Objective

Find every way to programmatically access experimental data at this facility. Create `AccessMethod` graph nodes and update facility infrastructure.

## Discovery Checklist

### 1. Check Module System

```bash
ssh {facility} "source /etc/profile.d/modules.sh 2>/dev/null && module avail 2>&1 | grep -iE 'python|mdsplus|imas|uda|ppf|sal|hdf5|matlab|idl' | head -30"
```

**Record:** Module names and versions for each data system.

### 2. Probe Python Packages Per Module

For each Python module version found:

```bash
ssh {facility} 'source /etc/profile.d/modules.sh; module purge; module load python/X.Y; python3 << "EOF"
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

### 3. Find Data Access Libraries

Search for language-specific bindings:

```bash
# MATLAB
ssh {facility} "find /usr/local /opt /common -name '*ppf*' -o -name '*mdsplus*' 2>/dev/null | grep -i matlab | head -10"

# IDL  
ssh {facility} "find /usr/local /opt /common -name '*idl*' -type d 2>/dev/null | head -10"

# Python packages
ssh {facility} "pip3 show MDSplus imas pyuda 2>/dev/null | grep -E 'Name|Version|Location'"
```

### 4. Test Each Method

For each discovered method, validate with a test shot:

```python
# Example test pattern
ssh {facility} 'source /etc/profile.d/modules.sh; module load python/3.9; python3 << "EOF"
{imports_template}
{connection_template}
data = {data_template}  # Use known test shot
print(f"Success: {type(data)}, shape: {getattr(data, 'shape', len(data))}")
EOF'
```

### 5. Document Discovery Path

Find how new users learn about data access:

```bash
ssh {facility} "find /common /usr/local -name '*guide*' -o -name '*manual*' -o -name '*tutorial*' 2>/dev/null | grep -iE 'data|ppf|mds|imas' | head -10"
```

## Output Requirements

### 1. Create Graph Nodes

```python
mcp_codex_add_to_graph(node_type="AccessMethod", data=[
    {
        "id": "{facility}:{method_type}:{variant}",
        "facility_id": "{facility}",
        "name": "{Human Readable Name}",
        "method_type": "{mdsplus|rest|uda|imas|hdf5|matlab|idl|cli}",
        "library": "{import target or path}",
        "access_type": "local",
        "imports_template": "{import statements}",
        "connection_template": "{connection/setup code}",
        "data_template": "{data retrieval code with {shot}, {signal} placeholders}",
        "time_template": "{time axis retrieval if applicable}",
        "data_source": "{default data source name}",
        "discovery_shot": {valid_test_shot_number},
        "full_example": "{complete working code example}"
    }
])
```

### 2. Update Infrastructure

```python
mcp_codex_update_facility_infrastructure(facility="{facility}", data={
    "data_access_methods": {
        "{method_key}": {
            "id": "{facility}:{method_type}:{variant}",
            "status": "verified|available|legacy",
            "setup_commands": ["module load X", ...],
            "probe_script": "{one-liner to test availability}",
            "verified_date": "{YYYY-MM-DD}",
            "documentation": {"local": "/path", "wiki": "url"}
        }
    }
})
```

## Known Patterns by Facility Type

### MDSplus Facilities (TCV, DIII-D, NSTX-U)
- Python: `import MDSplus; tree = MDSplus.Tree(name, shot)`
- TDI layer with semantic functions (tcv_eq, efit_*)
- MATLAB: mdsconnect/mdsopen/mdsvalue

### PPF/SAL Facilities (JET)
- Python: `from jet.data import sal; sal.get(path)`
- MATLAB: ppfgo/ppfget
- IDL: @idlppf, ppfgo, ppfget
- Legacy Fortran: PPFGO, PPFGET, PPFDAT

### IMAS Facilities (ITER, emerging standard)
- Python: `import imas; entry = imas.DBEntry(...)`
- Standard IDS structure

### UDA Facilities (MAST-U)
- Python: `import pyuda; client = pyuda.Client()`

## Completion Criteria

- [ ] All language bindings discovered (Python, MATLAB, IDL, Fortran, CLI)
- [ ] Each method tested with a valid shot
- [ ] AccessMethod nodes created in graph
- [ ] Infrastructure updated with setup_commands and probe_scripts
- [ ] Documentation paths recorded
- [ ] Discovery notes added to exploration_notes
