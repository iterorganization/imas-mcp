# Graph Profile Configuration

Graph profiles configure **what** data is in a Neo4j instance and **where**
it runs. These are two orthogonal concerns:

| Concept | Config key | Env override | Example |
|---------|-----------|--------------|---------|
| **Name** | `name` | `IMAS_CODEX_GRAPH` | `"codex"` (all facilities + IMAS DD) |
| **Location** | `location` | `IMAS_CODEX_GRAPH_LOCATION` | `"iter"` (ITER login node) |

The default graph is **`codex`** — a single graph containing all facilities
and the IMAS data dictionary. It runs at the **`iter`** location (the ITER
HPC cluster).

## Configuration

All graph settings live in `pyproject.toml`:

```toml
[tool.imas-codex.graph]
name = "codex"          # What data (override: IMAS_CODEX_GRAPH=tcv)
location = "iter"       # Where it runs (override: IMAS_CODEX_GRAPH_LOCATION=local)
username = "neo4j"
password = "imas-codex"

# Port slots — position = port offset
# iter=0 (7687/7474), tcv=1 (7688/7475), jt-60sa=2 (7689/7476), ...
locations = ["iter", "tcv", "jt-60sa", "jet", "west", "mast-u", "asdex-u", "east", "diii-d", "kstar"]
```

## Key Concepts

### Name (what data)

The graph **name** identifies what data lives in the Neo4j instance:

- `"codex"` — the main graph with all facilities + IMAS data dictionary
- `"tcv"` — a single-facility graph (when name matches a location, ports
  come from that location's slot — see below)
- `"sandbox"` — any arbitrary name for experimentation

### Location (where it runs)

The **location** determines where Neo4j physically runs. Each location maps
to an SSH alias and a port slot:

| Location | Bolt Port | HTTP Port | SSH alias |
|----------|-----------|-----------|-----------|
| iter | 7687 | 7474 | `iter` |
| tcv | 7688 | 7475 | `tcv` |
| jt-60sa | 7689 | 7476 | `jt-60sa` |
| jet | 7690 | 7477 | `jet` |

Port formula: `bolt = 7687 + index`, `http = 7474 + index` (index from the
`locations` list).

### How Name and Location Interact

When resolving a graph profile:

1. **Name matches a location** (e.g. `name = "tcv"`) → ports come from that
   location's slot. Useful for per-facility graphs.
2. **Name does not match any location** (e.g. `name = "codex"`) → ports come
   from the configured `location` key (default: `"iter"`).

This means **`codex`** at location `"iter"` uses the same ports as an
explicit `"iter"` graph — they share the Neo4j instance.

### SSH hosts

By default, each location's name doubles as its SSH alias (e.g. location
`"tcv"` → `ssh tcv`). Only add explicit entries in `[graph.hosts]` when
the SSH alias differs from the location name:

```toml
[tool.imas-codex.graph.hosts]
# Only needed when SSH alias ≠ location name:
# custom-location = "my-ssh-alias"
```

## URI Resolution

```
Name → Location → is_local_host(location) → URI
                       ↓ (remote)
                 auto-tunnel → bolt://localhost:{port+10000}
```

1. Graph name `"codex"` doesn't match a location → uses configured
   location `"iter"` → ports 7687/7474, SSH to `iter`
2. `is_local_host("iter")` checks facility private YAML:
   - On ITER login node: True → `bolt://localhost:7687`
   - Elsewhere: False → auto-tunnel → `bolt://localhost:17687`

## Auto-Tunneling

When connecting to a **remote** location, the profile resolver automatically
establishes an SSH tunnel with a +10000 offset:

```
Direct (on the host):    bolt = 7687 + offset
Tunneled (remote):       bolt = 17687 + offset
```

Override with env var: `IMAS_CODEX_TUNNEL_BOLT_ITER=17687`

Manual tunnel management:
```bash
imas-codex tunnel start iter         # Start tunnel
imas-codex tunnel status             # Show active tunnels
imas-codex tunnel stop iter          # Stop tunnel
```

## Configuration Scenarios

### 1. Default: Codex Graph on ITER

```toml
[tool.imas-codex.graph]
name = "codex"       # All facilities + IMAS DD
location = "iter"    # Runs on ITER login node
```

**From ITER login node:** Direct access at `bolt://localhost:7687`
**From WSL:** Auto-tunnels to `bolt://localhost:17687`

### 2. Per-Facility Graph

When `name` matches a location, it uses that location's port slot:

```bash
# Run a TCV-only graph (port 7688)
IMAS_CODEX_GRAPH=tcv imas-codex graph start
```

### 3. Local Development

```toml
[tool.imas-codex.graph]
name = "codex"
location = "local"    # No SSH, direct localhost access
```

Or via env var:
```bash
export IMAS_CODEX_GRAPH_LOCATION=local
```

### 4. Explicit Profile Override

For non-standard setups, define an explicit profile:

```toml
[tool.imas-codex.graph.profiles.staging]
location = "staging-server"   # Where Neo4j runs
bolt-port = 7700              # Custom ports
http-port = 7701
data-dir = "/custom/path/neo4j-staging"
```

```bash
IMAS_CODEX_GRAPH=staging imas-codex graph status
```

### 5. Multiple Tunnels

Access multiple facility graphs simultaneously from WSL:

```bash
# Each location gets its own tunnel port:
#   iter:   17687/17474
#   tcv:    17688/17475
#   jt-60sa: 17689/17476

IMAS_CODEX_GRAPH=codex  imas-codex graph status  # iter ports
IMAS_CODEX_GRAPH=tcv    imas-codex graph status  # tcv ports
```

## Data Directory Convention

| Graph name | Directory |
|-----------|-----------|
| `codex` (default) | `~/.local/share/imas-codex/.neo4j/codex/` |
| `tcv` | `~/.local/share/imas-codex/.neo4j/tcv/` |
| `sandbox` | `~/.local/share/imas-codex/.neo4j/sandbox/` |

## Quick Reference

| Env Var | Purpose |
|---------|---------|
| `IMAS_CODEX_GRAPH` | Select graph name |
| `IMAS_CODEX_GRAPH_LOCATION` | Override where Neo4j runs |
| `IMAS_CODEX_TUNNEL_BOLT_ITER` | Override tunnel port |
| `NEO4J_URI` | Override URI completely |

| CLI Command | Purpose |
|-------------|---------|
| `graph profiles` | List all profiles and status |
| `graph status` | Check active graph |
| `graph start` | Start active graph |
| `graph stop` | Stop active graph |
| `graph shell` | Interactive Cypher shell |
| `graph secure` | Rotate Neo4j password |
| `graph tags` | List GHCR tags |
| `graph prune` | Prune old GHCR tags |
| `tunnel start iter` | Manual tunnel to iter |
| `tunnel status` | Show active tunnels |
| `graph push` | Push graph to GHCR |
| `graph pull` | Fetch + load from GHCR |
