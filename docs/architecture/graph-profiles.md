# Graph Profile Configuration

Graph profiles configure where Neo4j runs and how to connect. The system uses
**convention-based** port allocation with **location-aware** URI resolution
and **automatic tunneling** for remote graphs.

## Key Concepts

| Concept | Description | Example |
|---------|-------------|---------|
| **Profile name** | Logical graph identifier | `"iter"`, `"tcv"` |
| **Host** | Where Neo4j physically runs | `"iter"` (SSH alias), `None` (local) |
| **Ports** | Bolt and HTTP ports at the host | 7687/7474 (iter), 7688/7475 (tcv) |
| **Tunnel offset** | +10000 to local port for tunneled access | 17687 (tunneled iter) |
| **Data dir** | Neo4j data directory | `~/.local/share/imas-codex/neo4j` |

## Port Convention

Port offsets and host defaults are managed in `pyproject.toml` under
`[tool.imas-codex.graph.locations]` and `[tool.imas-codex.graph.hosts]`:

```toml
[tool.imas-codex.graph.locations]
iter = 0       # bolt 7687, http 7474
tcv = 1        # bolt 7688, http 7475
jt60sa = 2     # bolt 7689, http 7476
jet = 3        # bolt 7690, http 7477
# ... more facilities

[tool.imas-codex.graph.hosts]
iter = "iter"      # SSH alias
tcv = "tcv"
jt60sa = "jt60sa"
```

| Profile | Bolt Port | HTTP Port | Default Host |
|---------|-----------|-----------|--------------|
| iter | 7687 | 7474 | `"iter"` (ITER cluster) |
| tcv | 7688 | 7475 | `"tcv"` (TCV machine) |
| jt60sa | 7689 | 7476 | `"jt60sa"` (JT-60SA) |
| jet | 7690 | 7477 | None (local) |

## How URI Resolution Works

```
Profile name → Host → is_local_host(host) → URI
                  ↓ (remote)
            auto-tunnel → bolt://localhost:{port+10000}
```

1. Profile "iter" has host="iter" (from `[tool.imas-codex.graph.hosts]`)
2. `is_local_host("iter")` checks facility private YAML:
   - On ITER login node: True (matches `login_nodes` pattern)
   - Elsewhere: False
3. URI resolved:
   - Local: `bolt://localhost:{port}` (direct)
   - Remote: auto-tunnel establishes `ssh -L {port+10000}:127.0.0.1:{port}`,
     then `bolt://localhost:{port+10000}`

## Auto-Tunneling

When connecting to a **remote** host, the profile resolver automatically
establishes an SSH tunnel with a +10000 offset:

```
Direct (on the host):    bolt = 7687 + offset
Tunneled (remote):       bolt = 17687 + offset
```

Override with env var: `IMAS_CODEX_TUNNEL_BOLT_ITER=17687`

Manual tunnel management:
```bash
imas-codex graph tunnel start iter   # Start tunnel
imas-codex graph tunnel status       # Show active tunnels
imas-codex graph tunnel stop iter    # Stop tunnel
```

## Graph Commands

```bash
imas-codex graph push          # Push graph to GHCR
imas-codex graph fetch         # Download archive (no load)
imas-codex graph pull          # Fetch + load (convenience)
imas-codex graph list          # List available versions in GHCR
imas-codex graph dump          # Export to archive
imas-codex graph load <file>   # Load archive into Neo4j
```

## Configuration Scenarios

### 1. Access ITER Graph (default)

**From ITER login node:**
```bash
# No config needed — is_local_host("iter") returns True
imas-codex graph db status
```

**From WSL (auto-tunnel):**
```bash
# Auto-tunnel kicks in automatically when resolving profile
imas-codex graph db status  # tunnels to bolt://localhost:17687
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

# From WSL: auto-tunnels to TCV graph on ITER (port 17688)
IMAS_CODEX_GRAPH=tcv imas-codex graph db status
```

### 5. Multiple Tunnels

For simultaneous access to multiple facility graphs from WSL:

```bash
# Auto-tunnels handle this — each graph gets its own port:
#   iter:  17687/17474
#   tcv:   17688/17475
#   jt60sa: 17689/17476

# Switch between graphs
IMAS_CODEX_GRAPH=iter imas-codex graph db status
IMAS_CODEX_GRAPH=tcv imas-codex graph db status
```

Or configure manual tunnels in SSH config:
```bash
# SSH config (~/.ssh/config)
Host iter
  LocalForward 17687 localhost:7687  # iter graph
  LocalForward 17688 localhost:7688  # tcv graph
  LocalForward 17689 localhost:7689  # jt60sa graph
```

## Avoiding Port Conflicts

**Problem:** Local Neo4j and SSH tunnel both on port 7687.

The tunnel offset (+10000) prevents this by default. Local Neo4j uses
convention ports (7687+) while tunnels use offset ports (17687+).

**Custom overrides for edge cases:**

1. **Use different port for local development**
   ```toml
   [tool.imas-codex.graph.profiles.dev]
   bolt-port = 7697  # Unused port
   http-port = 7484
   ```

2. **Use tunnel port override**
   ```bash
   export IMAS_CODEX_TUNNEL_BOLT_ITER=17687
   ```

## Quick Reference

| Env Var | Purpose |
|---------|---------|
| `IMAS_CODEX_GRAPH` | Select active profile |
| `IMAS_CODEX_GRAPH_LOCATION` | Override where Neo4j runs |
| `IMAS_CODEX_TUNNEL_BOLT_ITER` | Override tunnel port for iter |
| `NEO4J_URI` | Override URI completely |

| CLI Command | Purpose |
|-------------|---------|
| `graph db profiles` | List all profiles and status |
| `graph db status` | Check active graph |
| `graph db status -g tcv` | Check specific profile |
| `graph db start -g tcv` | Start specific profile |
| `graph tunnel start iter` | Manual tunnel to iter |
| `graph tunnel status` | Show active tunnels |
| `graph fetch` | Download archive from GHCR |
| `graph pull` | Fetch + load convenience |
| `graph list` | List versions in GHCR |
