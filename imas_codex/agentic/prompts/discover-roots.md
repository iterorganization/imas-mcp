# Discover Root Paths for {facility}

## Objective

Identify high-value root directories for seeding the discovery pipeline. Good roots
ensure balanced coverage across **simulation** (forward modeling) and **experimental**
(measurement analysis) domains.

## Discovery Root Categories

The taxonomy maintains duality between forward modeling (prediction) and experimental
analysis (measurement). Categories are generic and apply across all facilities.

### Forward Modeling Domain (Prediction)

| Category | Purpose | Examples |
|----------|---------|----------|
| `modeling_code` | Physics simulation source code | ASTRA, JOREK, DREAM, JINTRAC, TRANSP, GENE |
| `modeling_data` | Simulation outputs and results | Parameter scans, scenario DBs, turbulence data |

### Experimental Analysis Domain (Measurement)

| Category | Purpose | Examples |
|----------|---------|----------|
| `analysis_code` | Shot/pulse processing code | LIUQE, EFIT, Thomson processing, profile fitting |
| `experimental_data` | Measurement data from shots | MDSplus trees, PPF databases, EDAS, shot archives |

### Shared Infrastructure

| Category | Purpose | Examples |
|----------|---------|----------|
| `data_access` | Data access layers | TDI functions, IDL SAL, IMAS wrappers |
| `workflow` | User analysis environments | Jupyter notebooks, batch scripts, user workspaces |
| `documentation` | Reference materials | Manuals, papers, tutorials, READMEs |

## Exploration Commands

Use these commands to discover candidate roots:

```bash
# Find top-level code directories
ssh {ssh_host} "ls -la /home/codes /work/codes 2>/dev/null | head -50"

# Find shot data locations (MDSplus, PPF, EDAS, etc.)
ssh {ssh_host} "df -h | grep -E 'data|shots|mds|ppf'"
ssh {ssh_host} "ls -la /tcvssd /data /work 2>/dev/null | head -30"

# Find data access layer (TDI, IDL, SAL)
ssh {ssh_host} "ls -la /usr/local/*/tdi /usr/local/idl 2>/dev/null"

# Find user workspaces
ssh {ssh_host} "ls -la /home/users /work/analysis 2>/dev/null | head -20"

# Find documentation directories
ssh {ssh_host} "fd -t d -d 2 'Docs|docs|doc|manual' /home/codes 2>/dev/null | head -20"
```

## Output Format

After exploration, update the facility's private YAML with discovered roots:

```python
update_facility_infrastructure("{facility}", {
    "discovery_roots": [
        # === FORWARD MODELING (Prediction) ===
        {"path": "/home/codes/jorek", "category": "modeling_code", "description": "JOREK MHD modeling"},
        {"path": "/scratch/simulations", "category": "modeling_data", "description": "Simulation outputs"},
        
        # === EXPERIMENTAL ANALYSIS (Measurement) ===
        {"path": "/home/codes/liuqe", "category": "analysis_code", "description": "Equilibrium reconstruction"},
        {"path": "/tcvssd/trees", "category": "experimental_data", "description": "MDSplus shot data"},
        
        # === SHARED INFRASTRUCTURE ===
        {"path": "/usr/local/CRPP/tdi", "category": "data_access", "description": "TDI data access functions"},
        {"path": "/home/users", "category": "workflow", "description": "User analysis scripts"},
        {"path": "/home/codes/*/Docs", "category": "documentation", "description": "Code documentation"}
    ]
})
```

## Prioritization Guidelines

1. **Maintain duality**: Ensure roots in BOTH simulation AND experimental domains
2. **Avoid overlap**: Don't add `/home/codes` AND `/home/codes/astra` (parent covers child)
3. **Experimental data first**: Common blind spot - actively seek shot/pulse data stores
4. **Include data access**: Critical for understanding semantic mappings
5. **User workspaces**: Where active analysis happens (often missed)

## After Discovery

Run targeted deep dives on newly discovered roots:

```bash
# Deep dive into experimental data
imas-codex discover paths {facility} -r /tcvssd/trees -c 5.0

# Deep dive with multiple roots (mixing domains)
imas-codex discover paths {facility} -r /home/codes/liuqe -r /home/codes/jorek -c 10.0

# Explore user workspaces
imas-codex discover paths {facility} -r /home/users -c 5.0
```

## Current Discovery Roots

Check existing configuration:

```python
print(get_facility_infrastructure("{facility}").get("discovery_roots", []))
```
