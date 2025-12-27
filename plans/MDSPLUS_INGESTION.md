# MDSplus Tree Ingestion

> LLM-driven workflow for ingesting MDSplus tree structures into the knowledge graph.

## Overview

MDSplus trees are hierarchical data containers used at fusion facilities (TCV, DIII-D, NSTX, etc.). This document describes how to ingest tree node definitions into the imas-codex knowledge graph for:

1. **Discoverability** - Search nodes by physics domain, units, description
2. **IMAS Mapping** - Link facility data to IMAS paths for Ambix Recipe generation
3. **Documentation** - Provide context for data access patterns

## Key Insight: No Scripts Needed

Tree structures are **very stable** (years between changes). Rather than maintaining ingestion scripts, we use an **LLM-driven on-the-fly approach**:

1. SSH to facility, introspect tree structure
2. Parse output dynamically
3. Use MCP tools (`ingest_nodes`, `cypher`) to batch insert
4. LLM adapts to different tree structures without code changes

## Workflow

### Step 1: List Tree Structure

SSH to the facility and use MDSplus commands to list the tree:

```bash
# TCV example - list results tree structure
ssh epfl "mdstcl -c 'set tree results /shot=80000; show structure'"

# Alternative: use Python MDSplus
ssh epfl "python3 -c \"
import MDSplus
tree = MDSplus.Tree('results', 80000, 'readonly')
for node in tree.getNodeWild('***'):
    print(f'{node.path}|{node.usage}|{node.units or \"\"}')
\""
```

### Step 2: Parse and Enrich

For each node, extract:
- `path` - Full node path (e.g., `\RESULTS::LIUQE:PSI`)
- `node_type` - MDSplus usage (SIGNAL, NUMERIC, STRUCTURE, etc.)
- `units` - Physical units (must be pint-compatible)
- `physics_domain` - Inferred from path/context (equilibrium, transport, etc.)
- `variants` - For multi-source nodes like PSI, PSI_2, PSI_3

### Step 3: Batch Ingest via MCP

Use the `ingest_nodes` MCP tool:

```python
ingest_nodes("TreeNode", [
    {
        "path": "\\RESULTS::LIUQE:PSI",
        "tree_name": "results",
        "facility_id": "epfl",
        "node_type": "SIGNAL",
        "units": "Wb",
        "physics_domain": "equilibrium",
        "description": "Poloidal flux from LIUQE equilibrium reconstruction",
        "variants": ["", "_2", "_3"],  # PSI, PSI_2, PSI_3
        "example_shot": 80000
    },
    # ... more nodes
])
```

### Step 4: Assign Physics Domains

Use the project's `PhysicsDomain` enum for categorization:

| Domain | Example Nodes |
|--------|---------------|
| `equilibrium` | PSI, IP, Q, BPHI, RAXIS, ZAXIS |
| `transport` | TE, TI, NE, PRAD |
| `magnetohydrodynamics` | BETA, LI, WMHD |
| `auxiliary_heating` | PECRH, PNBI, PICRH |
| `magnetic_field_diagnostics` | BTOR, BPOL, FLUX_LOOP |

Query domains:
```python
from imas_codex.core.data_model import PhysicsDomain
print([d.value for d in PhysicsDomain])
```

## Handling Variants

Many facilities have multiple sources for the same quantity (e.g., different equilibrium codes). Store as a single node with `variants` array:

| Node Path | Variants | Meaning |
|-----------|----------|---------|
| `\RESULTS::LIUQE:PSI` | `["", "_2", "_3"]` | PSI from LIUQE, LIUQE02, LIUQE03 |
| `\RESULTS::FBTE:PSI` | `[""]` | PSI from FBTE (no variants) |

The first variant is the canonical/default source.

## TDI Functions

High-level accessor functions abstract tree navigation. Ingest these separately:

```python
ingest_nodes("TDIFunction", [
    {
        "name": "tcv_eq",
        "facility_id": "epfl",
        "description": "Get equilibrium quantities with source selection",
        "signature": "tcv_eq(signal, source='LIUQE', shot=None)",
        "parameters": ["signal", "source", "shot"],
        "physics_domain": "equilibrium"
    }
])
```

## IMAS Mapping

After ingesting nodes, create mappings to IMAS paths:

```python
ingest_nodes("IMASMapping", [
    {
        "id": "epfl:psi_to_imas",
        "source_path": "\\RESULTS::LIUQE:PSI",
        "target_path": "equilibrium/time_slice/profiles_2d/0/psi",
        "facility_id": "epfl",
        "driver": "mdsplus",
        "driver_args": "{\"tree\": \"results\", \"path\": \"\\\\RESULTS::LIUQE:PSI\"}",
        "units_in": "Wb",
        "units_out": "Wb",
        "scale": 1.0
    }
])
```

## Chunking Strategy

For large trees (>1000 nodes), ingest in chunks:

1. **By physics domain** - Ingest equilibrium nodes, then transport, etc.
2. **By tree subtree** - Ingest LIUQE subtree, then FBTE, etc.
3. **Batch size** - Use 100-node batches to avoid timeout

## TCV Specifics (EPFL)

Key trees at TCV:

| Tree | Nodes | Purpose |
|------|-------|---------|
| `results` | ~11k | Processed/analyzed data (start here) |
| `tcv_shot` | ~84k | Raw diagnostic data |
| `magnetics` | ~500 | Magnetic measurements |
| `base` | ~1.3k | Machine configuration |

TDI function locations:
- `/home/VMS/XYZ/MDSMGR/TDI/` (~442 functions)
- `/home/VMS/XYZ/MDSMGR/TCV_TDI_FUNCTIONS/` (~253 functions)

Data server: `tcvdata.epfl.ch`

## See Also

- [facility.yaml](../imas_codex/schemas/facility.yaml) - Schema definitions
- [AMBIX_PLAN.md](AMBIX_PLAN.md) - Recipe structure for data pumping
- [FACILITY_KNOWLEDGE.md](FACILITY_KNOWLEDGE.md) - Exploration workflow
