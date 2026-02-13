# Graph Profile Configuration

Graph profiles configure where Neo4j runs and how to connect. The system uses
**convention-based** port allocation with **location-aware** URI resolution.

## Key Concepts

| Concept | Description | Example |
|---------|-------------|---------|
| **Profile name** | Logical graph identifier | `"iter"`, `"tcv"` |
| **Host** | Where Neo4j physically runs | `"iter"` (SSH alias), `None` (local) |
| **Ports** | Bolt and HTTP ports at the host | 7687/7474 (iter), 7688/7475 (tcv) |
| **Data dir** | Neo4j data directory | `~/.local/share/imas-codex/neo4j` |

## Port Convention

Each facility uses a unique port offset (convention, no config needed):

| Profile | Bolt Port | HTTP Port | Default Host |
|---------|-----------|-----------|--------------|
| iter | 7687 | 7474 | `"iter"` (ITER cluster) |
| tcv | 7688 | 7475 | `"tcv"` (TCV machine) |
| jt60sa | 7689 | 7476 | `"jt60sa"` (JT-60SA) |
| jet | 7690 | 7477 | None (local) |

## How URI Resolution Works

```
Profile name → Host → is_local_host(host) → URI
```

1. Profile "iter" has host="iter" (from `FACILITY_HOST_DEFAULTS`)
2. `is_local_host("iter")` checks facility private YAML:
   - On ITER login node: True (matches `login_nodes` pattern)
   - Elsewhere: False
3. URI resolved:
   - Local: `bolt://localhost:{port}` (direct)
   - Remote: `bolt://localhost:{port}` (assumes same-port SSH tunnel)

## Configuration Scenarios

### 1. Access ITER Graph (default)

**From ITER login node:**
```bash
# No config needed — is_local_host("iter") returns True
imas-codex graph db status
```

**From WSL (requires SSH tunnel):**
```bash
# Establish tunnel first
ssh -f -N -L 7687:localhost:7687 iter

# Then use normally
imas-codex graph db status
```

### 2. Local-only Graph (no remote)

Create an explicit local profile in `pyproject.toml`:

```toml
[tool.imas-codex.graph]
default = "local"  # Use local profile by default

[tool.imas-codex.graph.profiles.local]
# No host = local execution
bolt-port = 7687
http-port = 7474
data-dir = "/home/user/.local/share/imas-codex/neo4j-local"
```

Or use env var override:
```bash
export IMAS_CODEX_GRAPH=local
```

### 3. Local + Facility-specific

Run Neo4j locally for TCV-only data on different ports:

```toml
[tool.imas-codex.graph]
default = "tcv-local"

[tool.imas-codex.graph.profiles.tcv-local]
# No host = runs locally
bolt-port = 7688    # TCV convention port
http-port = 7475
data-dir = "/home/user/.local/share/imas-codex/neo4j-tcv"
```

```bash
# Start TCV-specific local graph
IMAS_CODEX_GRAPH=tcv-local imas-codex graph db start
```

### 4. ITER + Facility-specific

Access facility-specific graph on ITER (different port, same host):

```bash
# On ITER: switch to TCV graph (runs on ITER, port 7688)
IMAS_CODEX_GRAPH=tcv imas-codex graph db status

# From WSL: tunnel to TCV graph on ITER
ssh -f -N -L 7688:localhost:7688 iter
IMAS_CODEX_GRAPH=tcv imas-codex graph db status
```

### 5. Multiple Tunnels

For simultaneous access to multiple facility graphs from WSL:

```bash
# SSH config (~/.ssh/config)
Host iter
  LocalForward 7687 localhost:7687  # iter graph
  LocalForward 7688 localhost:7688  # tcv graph
  LocalForward 7689 localhost:7689  # jt60sa graph

# Start all tunnels
ssh -f -N iter

# Switch between graphs
IMAS_CODEX_GRAPH=iter imas-codex graph db status
IMAS_CODEX_GRAPH=tcv imas-codex graph db status
```

## Avoiding Port Conflicts

**Problem:** Local Neo4j and SSH tunnel both on port 7687.

**Solutions:**

1. **Don't run local Neo4j when using tunnel**
   ```bash
   imas-codex graph db stop  # Stop local
   ssh -f -N iter            # Start tunnel
   ```

2. **Use different port for local development**
   ```toml
   [tool.imas-codex.graph.profiles.dev]
   bolt-port = 7697  # Unused port
   http-port = 7484
   ```

3. **Use tunnel port override**
   ```bash
   # Tunnel on different local port
   ssh -f -N -L 17687:localhost:7687 iter
   export IMAS_CODEX_TUNNEL_BOLT_ITER=17687
   ```

## Quick Reference

| Env Var | Purpose |
|---------|---------|
| `IMAS_CODEX_GRAPH` | Select active profile |
| `IMAS_CODEX_TUNNEL_BOLT_ITER` | Override tunnel port for iter |
| `NEO4J_URI` | Override URI completely |

| CLI Command | Purpose |
|-------------|---------|
| `graph db profiles` | List all profiles and status |
| `graph db status` | Check active graph |
| `graph db status -g tcv` | Check specific profile |
| `graph db start -g tcv` | Start specific profile |
