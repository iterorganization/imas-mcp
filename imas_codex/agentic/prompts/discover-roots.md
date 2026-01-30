# Discover Root Paths for {facility}

## Objective

Identify high-value root directories for seeding the discovery pipeline. Good roots
ensure balanced coverage across **simulation** and **experimental** domains.

## Discovery Root Categories

Each root should be classified into one of these categories:

### Simulation Domain (predictive, offline)
- **simulation_code**: Physics modeling codes (ASTRA, JOREK, DREAM, JINTRAC)
- **simulation_output**: Results from simulation runs (HDF5, NetCDF parameter scans)

### Experimental Domain (shot-based, measurements)
- **experimental_code**: Code that processes actual shots (real-time analysis)
- **shot_archive**: Raw shot databases (MDSplus servers, shot files)
- **diagnostic_archive**: Raw diagnostic files (Thomson, bolometry, interferometry)
- **reconstruction**: Derived quantities (equilibrium fits, profile reconstructions)

### Cross-cutting
- **tdi_functions**: TDI function libraries for MDSplus access
- **analysis_workflows**: User analysis scripts, Jupyter notebooks
- **documentation**: Manuals, papers, READMEs

## Exploration Commands

Use these commands to discover candidate roots:

```bash
# Find top-level code directories
ssh {ssh_host} "ls -la /home/codes /work/codes 2>/dev/null | head -50"

# Find MDSplus/shot data locations
ssh {ssh_host} "locate -i mdsplus 2>/dev/null | head -20"
ssh {ssh_host} "df -h | grep -E 'data|shots|mds'"

# Find diagnostic data archives
ssh {ssh_host} "fd -t d -d 3 'thomson|bolom|interferom|ece' /data 2>/dev/null"

# Find TDI function locations
ssh {ssh_host} "ls -la /usr/local/*/tdi 2>/dev/null"

# Find documentation directories
ssh {ssh_host} "fd -t d -d 2 'Docs|docs|doc|manual' /home/codes 2>/dev/null | head -20"
```

## Output Format

After exploration, update the facility's private YAML with discovered roots:

```python
update_facility_infrastructure("{facility}", {
    "discovery_roots": [
        # === SIMULATION DOMAIN ===
        {"path": "/home/codes", "category": "simulation_code", "description": "Central code repository"},
        {"path": "/work/modeling/outputs", "category": "simulation_output", "description": "Simulation results"},
        
        # === EXPERIMENTAL DOMAIN ===
        {"path": "/data/shots", "category": "shot_archive", "description": "MDSplus shot database"},
        {"path": "/diagnostic/raw", "category": "diagnostic_archive", "description": "Raw diagnostic data"},
        {"path": "/work/equilibrium", "category": "reconstruction", "description": "LIUQE/CHEASE outputs"},
        
        # === CROSS-CUTTING ===
        {"path": "/usr/local/CRPP/tdi", "category": "tdi_functions", "description": "TDI function library"},
        {"path": "/home/codes/*/Docs", "category": "documentation", "description": "Code documentation"}
    ]
})
```

## Prioritization Guidelines

1. **Avoid overlap**: Don't add `/home/codes` AND `/home/codes/astra` (parent covers child)
2. **Balance domains**: Ensure at least 1-2 roots per major category
3. **Prefer specific over general**: `/data/shots/tcv` over `/data` if TCV-specific
4. **Include experimental data**: Common blind spot - actively seek shot archives
5. **Add TDI roots**: Critical for IMAS mapping, often in `/usr/local/*/tdi`

## After Discovery

Run targeted deep dives on newly discovered roots:

```bash
# Deep dive into specific experimental data area
imas-codex discover paths {facility} -r /data/shots/tcv -c 5.0

# Deep dive with multiple roots
imas-codex discover paths {facility} -r /diagnostic/raw -r /work/equilibrium -c 10.0
```

## Current Discovery Roots

Check existing configuration:

```python
print(get_facility_infrastructure("{facility}").get("discovery_roots", []))
```
