---
name: explore_data
description: Discover data systems, MDSplus, HDF5, and data organization
tags: [mcp, exploration, cursor, cli, data]
artifact_type: data
---

{% include "explore_core.md" %}

---

## Data Exploration

Focus on discovering:
- **MDSplus**: Server, trees, signal structure
- **HDF5**: File locations, naming patterns
- **NetCDF**: File locations, naming patterns
- **Other formats**: MAT files, CSV, JSON
- **Shot ranges**: What data is available

## Recommended Commands

### MDSplus

```bash
# Check if MDSplus is available
uv run imas-codex {{ facility }} "python3 -c 'import MDSplus; print(MDSplus.__version__)' 2>/dev/null || echo 'MDSplus not available'"

# List trees (if mdstcl available)
uv run imas-codex {{ facility }} "mdstcl 2>/dev/null << 'MDSCMD'
set tree tcv
show db
MDSCMD"

# Test Python bindings
uv run imas-codex {{ facility }} "python3 -c \"
import MDSplus
c = MDSplus.Connection('{{ known_systems.mdsplus.server | default(\"localhost\") }}')
print('Connection OK')
\""
```

### HDF5

```bash
# Find HDF5 files
uv run imas-codex {{ facility }} "find /common -name '*.h5' -o -name '*.hdf5' 2>/dev/null | head -20"

# Check h5dump availability
uv run imas-codex {{ facility }} "which h5dump && h5dump --version"

# Inspect a file structure (if h5dump available)
uv run imas-codex {{ facility }} "h5dump -H /path/to/file.h5 2>/dev/null | head -50"
```

### NetCDF

```bash
# Find NetCDF files
uv run imas-codex {{ facility }} "find /common -name '*.nc' 2>/dev/null | head -20"

# Check ncdump availability
uv run imas-codex {{ facility }} "which ncdump && ncdump --version 2>&1 | head -1"
```

## Comprehensive Script

```bash
uv run imas-codex {{ facility }} << 'EOF'
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

## MDSplus Tips

{% if known_systems.mdsplus %}
This facility uses MDSplus with server **{{ known_systems.mdsplus.server }}**.

Known trees:
{% for tree in known_systems.mdsplus.trees %}
- `{{ tree }}`
{% endfor %}
{% else %}
Check if MDSplus is configured for this facility.
{% endif %}

{% include "schemas/data_schema.md" %}

