# MDSplus Tree Ingestion

> CLI-driven workflow for ingesting MDSplus tree structures into the knowledge graph.

## Overview

MDSplus trees are hierarchical data containers used at fusion facilities (TCV, DIII-D, NSTX, etc.). This document describes how to ingest tree node definitions into the imas-codex knowledge graph for:

1. **Discoverability** - Search nodes by physics domain, units, description
2. **IMAS Mapping** - Link facility data to IMAS paths for Ambix Recipe generation
3. **Documentation** - Provide context for data access patterns
4. **Versioning** - Track shot-range validity for evolving analysis codes

## Quick Start: Batch Ingestion

For complete tree ingestion across all TCV trees:

```bash
cd /path/to/imas-codex
uv run imas-codex neo4j start
./scripts/ingest_all_trees.sh 2>&1 | tee ingest_trees.log
```

For individual trees:

```bash
# Ingest a single tree with epoch discovery and metadata extraction
uv run discover-mdsplus epfl results -v

# Include legacy node cleanup (merge metadata then delete superseded nodes)
uv run discover-mdsplus epfl magnetics -v --clean

# Dry run to preview
uv run discover-mdsplus epfl base --dry-run
```

## CLI Options

| Option | Purpose |
|--------|---------|
| `-v, --verbose` | Detailed logging |
| `--clean` | Merge legacy metadata and cleanup superseded nodes |
| `--skip-metadata` | Skip units/description extraction (faster) |
| `--dry-run` | Preview without writing to graph |
| `--full` | Force full scan, ignoring existing epochs |
| `--refine` | Refine existing rough epoch boundaries |

## Available Trees (TCV/EPFL)

```
apcs atlas base diag_act diagz ecrh hybrid magnetics manual pcs 
power raw_bolo raw_ece raw_fild raw_mag raw_mhd results tcv_shot thomson vsystem
```

## Priority: TDI Functions First

TDI functions are the recommended starting point because:

1. **They abstract complexity** - Handle source selection, variant mapping, version lookup
2. **Fewer items** - ~700 functions vs ~100k tree nodes
3. **Rich semantics** - Signature, parameters, physics domain are explicit
4. **Version tracking** - VERSION.FUN provides shot-range validity

**Recommended order:**
1. TDI functions (especially VERSION.FUN pattern)
2. Analysis codes with version/variant info
3. Tree nodes (starting with `results` tree equilibrium subtree)
4. IMAS mappings

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
        "physics_domain": "equilibrium",
        "default_source": "LIUQE",
        "available_sources": ["LIUQE", "LIUQE2", "LIUQE.M", "FBTE", "FLAT", "RAMP"]
    }
])
```

### TCV VERSION.FUN Pattern

TCV implements shot-range versioning via `VERSION.FUN`. This function returns:
- **`-1`**: Program not applicable for this shot range
- **`-2`**: Program doesn't exist (undefined)
- **`> 0`**: Version number valid from that shot

**Internal structure:**
```tdi
PUBLIC FUN VERSION(OPTIONAL IN _program, OPTIONAL IN _shot) {
    /* Arrays of versions and starting shot numbers */
    _equil_1_verss = [1.0, ...];
    _equil_1_shots = [0, ...];
    
    /* bsearch to find version for shot */
    _loc = BSEARCH(_shot, _shots);
    RETURN(_verss[_loc]);
}
```

**Ingest version history:**
```python
ingest_nodes("TDIFunction", [
    {
        "name": "version",
        "facility_id": "epfl",
        "description": "Master version lookup for analysis codes",
        "signature": "version(program, shot)",
        "parameters": ["program", "shot"],
        "physics_domain": "general",
        "return_type": "float"
    }
])
```

### TCV Variant Mapping (tcv_eq)

The `tcv_eq` function handles source selection and variant mapping:

```tdi
/* Source to variant suffix mapping */
_mode = _mode == "LIUQE" ? "" : "_" // extract(5,1,_mode)
/* LIUQE -> "", LIUQE2 -> "_2", LIUQE.M -> ".M" */

/* Path construction */
_path = "\\RESULTS::" // _var // _mode
/* e.g., \\RESULTS::PSI_2 for LIUQE2 */
```

| Source | Variant Suffix | Example Path |
|--------|---------------|--------------|
| LIUQE | `` | `\RESULTS::PSI` |
| LIUQE2 | `_2` | `\RESULTS::PSI_2` |
| LIUQE3 | `_3` | `\RESULTS::PSI_3` |
| LIUQE.M | `.M` | `\RESULTS::PSI.M` |
| FBTE | `` | `\RESULTS::FBTE:PSI` |

### Analysis Code Versioning

Link analysis codes to the variants they produce:

```python
ingest_nodes("AnalysisCode", [
    {
        "name": "LIUQE",
        "facility_id": "epfl",
        "code_type": "equilibrium",
        "description": "Primary equilibrium reconstruction",
        "output_variant": "",
        "writes_to_tree": "results"
    },
    {
        "name": "LIUQE02",
        "facility_id": "epfl",
        "code_type": "equilibrium",
        "output_variant": "_2",
        "writes_to_tree": "results",
        "supersedes": "LIUQE"
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

| Tree | Nodes | Population | Purpose |
|------|-------|------------|---------|
| `results` | ~11k | dynamic | Processed/analyzed data (start here) |
| `tcv_shot` | ~84k | dynamic | Raw diagnostic data |
| `magnetics` | ~500 | static | Magnetic measurements |
| `base` | ~1.3k | static | Machine configuration |
| `thomson` | ~1k | hybrid | Thomson scattering data |

**Population types:**
- **static**: Structure fixed, always present
- **dynamic**: Nodes created at runtime by analysis codes
- **hybrid**: Fixed structure with dynamic population

TDI function locations:
- `/usr/local/mdsplus/tdi/` (~672 .FUN files total)
- Key functions: `VERSION.FUN`, `TCV_EQ.FUN`, `TCV_IP.FUN`, `TCV_PSITBX.FUN`

Data server: `tcvdata.epfl.ch`

### TDI Function Categories (from scouting)

| Category | Count | Examples |
|----------|-------|----------|
| Equilibrium | ~50 | `tcv_eq`, `tcv_psitbx`, `setequil` |
| Magnetics | ~30 | `tcv_ip`, `tcv_bphi`, `tcv_mhd_env` |
| Thomson | ~20 | `ts_te`, `ts_ne`, `thomson_merge` |
| ECE | ~15 | `ece_te`, `ece_freq` |
| General | ~50 | `version`, `tcv_get`, `tcv_time` |

### Current Graph State

Run `get_exploration_progress("epfl")` to see:
- `mdsplus_coverage`: Per-tree ingestion status and node counts
- `tdi_coverage`: TDI functions by physics domain
- `code_coverage`: Analysis codes by type

## See Also

- [facility.yaml](../imas_codex/schemas/facility.yaml) - Schema definitions
- [AMBIX_PLAN.md](AMBIX_PLAN.md) - Recipe structure for data pumping
- [FACILITY_KNOWLEDGE.md](FACILITY_KNOWLEDGE.md) - Exploration workflow
