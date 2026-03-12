---
name: roots
description: Identify high-value root directories for seeding the discovery pipeline
dynamic: true
---

# Discover Root Paths for {{ facility | default("the target facility") }}

## Objective

Identify high-value root directories for seeding the discovery pipeline. Good roots
ensure balanced coverage across **simulation** (forward modeling) and **experimental**
(measurement analysis) domains.

{% include "schema/discovery-categories.md" %}

## CRITICAL: Workflow for Modifying Discovery Roots

**IMPORTANT**: Always use the MCP `update_facility_infrastructure()` tool to modify discovery roots.
This ensures schema validation and proper persistence.

**DO NOT directly edit private YAML files** - the MCP tool handles persistence correctly.

**Workflow:**

1. **Use MCP to update discovery_roots** (validates against schema):
   ```python
   update_facility_infrastructure("{{ facility | default('FACILITY') }}", {
       "discovery_roots": [
           {"path": "/path/to/codes", "category": "modeling_code", "description": "Physics codes"},
           # ... more roots
       ]
   })
   ```

2. **The MCP tool will**:
   - Validate category values against the schema
   - Persist to the private YAML file automatically
   - Merge with existing infrastructure data

3. **After updating via MCP**, commit the changes:
   ```bash
   git add -f imas_codex/config/facilities/{{ facility | default('FACILITY') }}_private.yaml
   uv run git commit -m "feat: update {{ facility | default('FACILITY') }} discovery_roots"
   git push origin main
   ```

Note: Private YAML files are gitignored by default. Use `git add -f` to force-add
when discovery_roots need to be version-controlled.

## Step 1: Check Current Discovery State

**Before exploring, query the graph to see what's already been discovered:**

```python
# Check configured discovery roots (what WILL be seeded)
print(get_facility_infrastructure("{{ facility | default('FACILITY') }}").get("discovery_roots", []))

# Check what's CURRENTLY in the graph
print(query("""
    MATCH (p:FacilityPath {facility_id: '{{ facility | default('FACILITY') }}'})
    WHERE p.depth = 0
    RETURN p.path AS root, p.status AS status, p.path_purpose AS purpose
    ORDER BY p.path
"""))

# See coverage by category
print(query("""
    MATCH (p:FacilityPath {facility_id: '{{ facility | default('FACILITY') }}'})
    WHERE p.status = 'scored' AND p.path_purpose IS NOT NULL
    RETURN p.path_purpose AS purpose, count(*) AS count
    ORDER BY count DESC
"""))
```

Use this information to identify **gaps** in coverage.

## Step 2: Exploration Commands

Use these commands to discover candidate roots:

```bash
# Find top-level code directories
ssh {{ ssh_host | default('{ssh_host}') }} "ls -la /home/codes /work/codes 2>/dev/null | head -50"

# Find shot data locations (MDSplus, PPF, EDAS, etc.)
ssh {{ ssh_host | default('{ssh_host}') }} "df -h | grep -E 'data|shots|mds|ppf'"
ssh {{ ssh_host | default('{ssh_host}') }} "ls -la /tcvssd /data /work 2>/dev/null | head -30"

# Find data access layer (TDI, IDL, SAL)
ssh {{ ssh_host | default('{ssh_host}') }} "ls -la /usr/local/*/tdi /usr/local/idl 2>/dev/null"

# Find user workspaces
ssh {{ ssh_host | default('{ssh_host}') }} "ls -la /home/users /work/analysis 2>/dev/null | head -20"

# Find documentation directories
ssh {{ ssh_host | default('{ssh_host}') }} "fd -t d -d 2 'Docs|docs|doc|manual' /home/codes 2>/dev/null | head -20"
```

## Step 3: Persist Discovered Roots

After exploration, update the facility's private YAML with discovered roots:

```python
update_facility_infrastructure("{{ facility | default('FACILITY') }}", {
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

### Valid Category Values

Use ONLY these category values (from schema):

{% for cat in discovery_categories %}
- `{{ cat.value }}`: {{ cat.description }}
{% endfor %}

## Prioritization Guidelines

1. **Maintain duality**: Ensure roots in BOTH simulation AND experimental domains
2. **Check gaps**: Use graph queries to find underrepresented categories
3. **Avoid overlap**: Don't add `/home/codes` AND `/home/codes/astra` (parent covers child)
4. **Experimental data first**: Common blind spot - actively seek shot/pulse data stores
5. **Include data access**: Critical for understanding semantic mappings
6. **User workspaces**: Where active analysis happens (often missed)

## User Discovery

**To discover FacilityUser nodes, first identify where user home directories are located.**

User homes may be in multiple locations depending on the facility:
- `/home` (common default)
- `/users`, `/home/users`
- NFS mounts like `/export/homes`, `/net/homes`
- Facility-specific paths like `/solhome`, `/JAChome`

### Step 1: Discover Home Directory Locations

```bash
# Check common locations and NFS mounts
ssh {{ ssh_host | default('{ssh_host}') }} "df -h | grep -iE 'home|user|export'"
ssh {{ ssh_host | default('{ssh_host}') }} "ls -la /home /users /export 2>/dev/null | head -30"

# Check for automounted home directories
ssh {{ ssh_host | default('{ssh_host}') }} "cat /etc/auto.master 2>/dev/null | grep -i home"

# Sample user home to verify structure
ssh {{ ssh_host | default('{ssh_host}') }} "getent passwd | head -5"
```

### Step 2: Add Discovered Home Locations as Container Roots

For each discovered home location, add as a container root with `expand_depth: 1`:

```python
update_facility_infrastructure("{{ facility | default('FACILITY') }}", {
    "discovery_roots": [
        # ... existing roots ...
        {
            "path": "/home",  # Replace with discovered path
            "category": "container",
            "description": "User home directories - discover facility users",
            "expand_depth": 1,
            "priority": "high"
        },
        # Add additional home locations if facility uses multiple
        # {"path": "/solhome", "category": "container", ...}
    ]
})
```

### What Happens During User Discovery

1. List all directories in the home container (e.g., `/home/duval`, `/home/sauter`)
2. Extract usernames from paths matching `{home_root}/{username}`
3. Query `getent passwd` to get full names (GECOS field)
4. Create `FacilityUser` nodes linked to `Person` nodes
5. Create `HAS_HOME` relationships to home directory paths

**Check existing user coverage:**
```python
print(query("""
    MATCH (u:FacilityUser)-[:AT_FACILITY]->(f:Facility {id: '{{ facility | default('FACILITY') }}'})
    RETURN count(u) AS users
"""))
```

## Step 4: Trigger Discovery

**CLI Flags Explained:**

| Flag | Purpose |
|------|---------|
| `--seed` | Adds discovery_roots from config that are NOT already in graph (additive, idempotent) |
| `-r/--root` | Restricts discovery to specific paths AND adds them to graph if missing |
| `--scan-only` | SSH enumeration only, no LLM scoring (fast, requires SSH) |
| `--score-only` | Score existing paths in graph (offline, no SSH) |

**Seeding workflow:**

```bash
# 1. First time: seed from config (creates FacilityPath nodes with status=discovered)
uv run imas-codex discover paths {{ facility | default('FACILITY') }} --seed

# 2. After modifying discovery_roots in private YAML:
uv run imas-codex discover paths {{ facility | default('FACILITY') }} --seed
# (Only adds NEW roots - existing paths are preserved)

# 3. After the scan has run, re-run to continue (idempotent):
uv run imas-codex discover paths {{ facility | default('FACILITY') }}
```

**Targeted deep dives (adds roots AND restricts scope):**

```bash
# Deep dive into experimental data
uv run imas-codex discover paths {{ facility | default('FACILITY') }} -r /tcvssd/trees -c 5.0

# Deep dive with multiple roots (mixing domains)
uv run imas-codex discover paths {{ facility | default('FACILITY') }} -r /home/codes/liuqe -r /home/codes/jorek -c 10.0

# Explore user workspaces
uv run imas-codex discover paths {{ facility | default('FACILITY') }} -r /home/users -c 5.0
```

## Horizontal Breakout: Finding New Areas

If discovery has stalled, use graph queries to identify opportunities:

```python
# Find scored containers that weren't expanded (potential new roots)
print(query("""
    MATCH (p:FacilityPath {facility_id: '{{ facility | default('FACILITY') }}'})
    WHERE p.path_purpose = 'container' 
          AND p.score > 0.5 
          AND p.should_expand = false
          AND p.terminal_reason IS NULL
    RETURN p.path AS path, p.score AS score, p.description AS description
    ORDER BY p.score DESC LIMIT 10
"""))

# Find categories with no discoveries yet
print(query("""
    WITH ['modeling_code', 'modeling_data', 'analysis_code', 'experimental_data', 
          'data_access', 'workflow', 'documentation'] AS expected_categories
    MATCH (p:FacilityPath {facility_id: '{{ facility | default('FACILITY') }}'})
    WHERE p.status = 'scored'
    WITH expected_categories, collect(DISTINCT p.path_purpose) AS found
    UNWIND expected_categories AS cat
    WITH cat WHERE NOT cat IN found
    RETURN cat AS missing_category
"""))
```

{% include "safety.md" %}
