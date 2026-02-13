# Graph Identity, Auto-Tunnels & Port Clash Prevention

## Problem Statement

The current graph profile system conflates three independent concerns:

1. **Graph identity** — what data is in it (facilities, DD version)
2. **Location** — where Neo4j physically runs (local, iter, tcv)
3. **Connection** — how to reach it (direct, tunnel, port)

This causes:

- **Silent port clashes**: local Neo4j on 7687 masks a tunnel to ITER on 7687
- **Manual tunnel management**: user must `ssh -f -N -L ...` before connecting
- **Naming confusion**: `"iter"` means both "the ITER facility" and "the graph
  that happens to run on ITER" — but that graph contains ALL facilities

## Design Decisions

### Graph Naming Convention

Graph names describe **content**, not location. A graph's name should tell you
what's inside without looking at where it runs.

| Name Pattern | Description | Example |
|-------------|-------------|---------|
| `<facility>` | Single-facility graph | `tcv`, `iter`, `jt60sa` |
| `<f1>-<f2>` | Multi-facility (alpha-sorted) | `iter-tcv`, `iter-jt60sa-tcv` |
| `codex` | All facilities — the canonical graph | `codex` |
| `sandbox` | Experimental/scratch | `sandbox` |

**Why not `full`?** It's relative — "full" compared to what? `codex` is the
project name and unambiguous. Single facility names are self-describing.

**Definition drift**: The graph stores a `GraphMeta` node with `facilities`
list. Ingestion checks this before writing — you can't accidentally ingest
TCV data into an `iter`-only graph. The meta node is the source of truth;
the name is a label that must match.

### Architectural Constraint: One Neo4j Per Host

**Each physical host runs exactly ONE Neo4j instance on the standard ports
(bolt 7687, http 7474).** You switch between graphs by swapping the data
loaded into that instance, not by running multiple Neo4j containers.

This simplification means:
- No per-graph-name port offsets needed
- Port allocation depends only on LOCATION (which host you're connecting to)
- Graph management commands swap content, not Neo4j instances

### Graph Storage Model

Two tiers of graph storage:

| Tier | Location | Versioned? | Purpose |
|------|----------|------------|---------|
| **GHCR** | `ghcr.io/iterorganization/imas-codex-graph` | Yes (tags) | Canonical versioned catalog |
| **Local Store** | `~/.local/share/imas-codex/graphs/` | No (named only) | Fast switching between graphs |

**GHCR** is the source of truth. Published graphs have version tags (`codex:v4.0.0`,
`tcv:v4.0.0`). The `latest` tag tracks the most recent release.

**Local Store** holds named graph dumps for fast switching. NOT versioned —
each graph name has exactly one stored copy. This avoids disk bloat from
storing multiple similar versions locally. If you need version history,
fetch older versions from GHCR on demand.

```
~/.local/share/imas-codex/graphs/
├── codex.tar.gz      # Dump of "codex" graph
├── tcv.tar.gz        # Dump of "tcv" graph
└── sandbox.tar.gz    # Dump of "sandbox" graph
```

### Graph Management Commands

**Data preservation principle**: Graph data is never lost unless explicitly
requested. Switching graphs saves the current graph first. Destructive
operations require confirmation.

| Command | Action | Data Loss? |
|---------|--------|------------|
| `graph fetch <name>` | Download from GHCR to local store | No |
| `graph switch <name>` | Dump current → load target from store | No |
| `graph new <name> --facilities f1,f2` | Create empty graph with metadata | No |
| `graph clear` | Remove all nodes/relationships, keep GraphMeta | Yes (explicit) |
| `graph dump [name]` | Export current graph to local store | No |
| `graph push` | Upload current graph to GHCR | No |

**Why `fetch` not `pull`?** Git's `pull` means fetch+merge. Our `fetch`
only downloads — it never automatically loads or modifies the active graph.

**To work with a different graph**:
```bash
imas-codex graph fetch codex       # Download codex from GHCR to local store
imas-codex graph switch codex      # Dump current graph, load codex

imas-codex graph fetch tcv         # Download tcv from GHCR
imas-codex graph switch tcv        # Dump current (codex), load tcv

imas-codex graph switch codex      # Dump current (tcv), load codex from store
# No re-fetch needed — codex is already in local store
```

**Switch workflow**:
1. Read `GraphMeta` from active graph → get current name
2. Dump active graph to local store under that name
3. Load target graph from local store
4. Verify `GraphMeta` in newly loaded graph

```python
def switch_graph(target_name: str) -> None:
    current = get_current_graph_name()  # Read GraphMeta
    if current:
        dump_to_store(current)          # Save current first
    load_from_store(target_name)        # Load target
    verify_graph_meta(target_name)      # Confirm identity
```

**Creating a new graph**:
```bash
imas-codex graph new sandbox --facilities iter,tcv
# Creates empty graph with GraphMeta: name=sandbox, facilities=[iter, tcv]

imas-codex graph new tcv-dev --facilities tcv
# Single-facility development graph
```

**Clearing graph data**:
```bash
imas-codex graph clear
# Prompts: "This will delete all nodes and relationships. Continue? [y/N]"
# Preserves GraphMeta (name, facilities, dd_version)

imas-codex graph clear --force
# Skip confirmation (for scripts)
```

### Port Allocation: Location-Based Offsets

Ports are determined by **location** (where Neo4j runs), not graph name.
Each known location gets a fixed offset:

```
LOCATION OFFSETS (defined in pyproject.toml):
  iter   = 0    → bolt 7687, http 7474
  tcv    = 1    → bolt 7688, http 7475
  jt60sa = 2    → bolt 7689, http 7476
  local  = 0    → bolt 7687, http 7474  (your machine, no offset)

TUNNEL OFFSET = +10000 (separates direct from tunneled connections)
```

**Port calculation**:
```
Direct (you're on the host):     bolt = 7687 + location_offset
Tunneled (remote host):          bolt = 17687 + location_offset
```

**Where offsets are defined** — `pyproject.toml`:
```toml
[tool.imas-codex.graph.locations]
# SSH alias → offset mapping
iter = 0
tcv = 1
jt60sa = 2
# 'local' is implicit offset 0
```

**Why location offsets exist**: You may have tunnels to MULTIPLE remote
hosts simultaneously. Each tunnel needs a unique local port:

```
From WSL with tunnels to both iter and tcv:

  Tunnel to iter:7687    →  localhost:17687  (17687 + 0)
  Tunnel to tcv:7688     →  localhost:17688  (17687 + 1)
  Local Neo4j            →  localhost:7687   (direct, no tunnel)
```

### Port Allocation Examples

**Scenario: On WSL, want to query graphs on different hosts**

| Graph | Location | How | Local Port |
|-------|----------|-----|------------|
| codex | iter | tunnel | 17687 (= 17687 + iter_offset=0) |
| tcv | tcv | tunnel | 17688 (= 17687 + tcv_offset=1) |
| sandbox | local | direct | 7687 |

```bash
# Query codex graph on iter
IMAS_CODEX_GRAPH=codex IMAS_CODEX_GRAPH_LOCATION=iter imas-codex graph db status
# → Auto-tunnel to iter:7687, connect via localhost:17687

# Query tcv graph on tcv
IMAS_CODEX_GRAPH=tcv IMAS_CODEX_GRAPH_LOCATION=tcv imas-codex graph db status
# → Auto-tunnel to tcv:7688, connect via localhost:17688

# Query local sandbox
IMAS_CODEX_GRAPH=sandbox IMAS_CODEX_GRAPH_LOCATION=local imas-codex graph db status
# → Direct connect to localhost:7687
```

**Scenario: On ITER login node, want local graph**

| Graph | Location | How | Port |
|-------|----------|-----|------|
| codex | iter | direct (you're on iter) | 7687 |

No tunnel needed — `is_local_host("iter")` returns True when you're on ITER.

**Scenario: What if iter hosts multiple graphs?**

It doesn't — one Neo4j per host. If you want to switch from codex to tcv
on the iter host:
```bash
ssh iter "imas-codex graph switch tcv"
# Automatically: dump codex → load tcv
```

Now iter's Neo4j contains tcv data. The port (7687) stays the same; only
the GraphMeta node changes to reflect the new content. The previous codex
graph is preserved in iter's local store — `switch codex` restores it.

### Automatic Tunnel Management

When `_resolve_uri()` detects a remote host:

1. Look up location offset for the host
2. Calculate tunnel port: `17687 + location_offset`
3. Check if tunnel port is already bound → use it
4. If not bound → auto-launch: `ssh -f -N -L {tunnel_port}:localhost:{remote_port} {host}`
5. Verify tunnel came up (socket probe with timeout)
6. Connect to `bolt://localhost:{tunnel_port}`

The remote port uses the same location offset: `7687 + location_offset`.
This handles the case where tcv runs Neo4j on 7688 locally (its own offset).

```python
TUNNEL_OFFSET = 10000
LOCATION_OFFSETS = {"iter": 0, "tcv": 1, "jt60sa": 2}  # from pyproject.toml

def get_ports(location: str) -> tuple[int, int]:
    """Return (remote_bolt_port, tunnel_bolt_port) for a location."""
    offset = LOCATION_OFFSETS.get(location, 0)
    remote = 7687 + offset
    tunnel = remote + TUNNEL_OFFSET
    return remote, tunnel
```

Existing code in `embeddings/readiness.py::_ensure_ssh_tunnel()` provides
the pattern — extract to shared utility.

### GraphMeta Node

Every graph instance contains exactly one `(:GraphMeta)` node:

```cypher
CREATE CONSTRAINT graph_meta_singleton IF NOT EXISTS
FOR (m:GraphMeta) REQUIRE m.id IS UNIQUE;

MERGE (m:GraphMeta {id: "meta"})
SET m.name = "codex",
    m.facilities = ["iter", "tcv", "jt60sa"],
    m.dd_version = "4.1.0",
    m.created_at = datetime(),
    m.updated_at = datetime()
```

This enables:
- **Identity check**: `graph db status` shows graph name + facilities
- **Ingestion gating**: before writing TCV data, verify "tcv" in facilities
- **Version tracking**: DD version the graph was built against
- **Drift detection**: warn if graph name doesn't match meta.name

### Configuration Model

Replace `[tool.imas-codex.graph].default` with three concerns:

```toml
[tool.imas-codex.graph]
name = "codex"          # Graph identity (what data is loaded)
location = "iter"       # Where Neo4j runs (SSH alias)
username = "neo4j"
password = "imas-codex"

# Location offset registry — determines ports per host
[tool.imas-codex.graph.locations]
iter = 0      # iter Neo4j on 7687
tcv = 1       # tcv Neo4j on 7688
jt60sa = 2    # jt60sa Neo4j on 7689
# 'local' always uses offset 0 (implicit)
```

**Resolution logic**:
1. `location` → look up offset → determine remote port (7687 + offset)
2. `is_local_host(location)?` → direct connect, else auto-tunnel
3. If tunneling → tunnel port = remote port + 10000
4. `name` → used only for GraphMeta validation, not port selection

**Key semantic change**: `name` and `location` are **orthogonal**.
- `name` = what data is in the graph (codex, tcv, sandbox)
- `location` = where Neo4j physically runs (iter, tcv, local)

Same graph name can exist at different locations (you pulled codex to your
local machine for development). Different graph names can be loaded at the
same location (you switched iter from codex to tcv for testing).

### Default Behavior Matrix

| You're on | name | location | Connection | Port |
|-----------|------|----------|------------|------|
| ITER | codex | iter | Direct | 7687 |
| WSL | codex | iter | Tunnel → iter:7687 | 17687 |
| WSL | codex | local | Direct | 7687 |
| TCV | tcv | tcv | Direct | 7688 |
| WSL | tcv | tcv | Tunnel → tcv:7688 | 17688 |
| WSL | tcv | local | Direct | 7687* |
| WSL | sandbox | local | Direct | 7687* |

*Local always uses 7687 regardless of graph name — one Neo4j per host.

**Multiple simultaneous connections from WSL**:
```bash
# Terminal 1: query codex on iter
IMAS_CODEX_GRAPH_LOCATION=iter imas-codex graph db shell
# → bolt://localhost:17687 (tunnel to iter:7687)

# Terminal 2: query tcv on tcv  
IMAS_CODEX_GRAPH_LOCATION=tcv imas-codex graph db shell
# → bolt://localhost:17688 (tunnel to tcv:7688)

# Terminal 3: query local sandbox
IMAS_CODEX_GRAPH_LOCATION=local imas-codex graph db shell
# → bolt://localhost:7687 (direct)
```

All three connections can be active simultaneously because each uses a
unique local port.

## Implementation Plan

### Phase 1: Port Clash Prevention (zero user friction)

Add tunnel offset and location registry. Update `_resolve_uri()`:

```python
TUNNEL_OFFSET = 10000

def get_location_offset(location: str) -> int:
    """Get port offset for a location from pyproject.toml."""
    from imas_codex.settings import get_graph_locations
    locations = get_graph_locations()  # {"iter": 0, "tcv": 1, ...}
    return locations.get(location, 0)

def _resolve_uri(location: str) -> str:
    offset = get_location_offset(location)
    remote_port = 7687 + offset
    
    if is_local_host(location):
        return f"bolt://localhost:{remote_port}"
    
    tunnel_port = remote_port + TUNNEL_OFFSET
    _ensure_graph_tunnel(location, remote_port, tunnel_port)
    return f"bolt://localhost:{tunnel_port}"
```

Extract `_ensure_ssh_tunnel()` from `embeddings/readiness.py` to
`imas_codex/remote/tunnel.py` as shared utility. Both embeddings and
graph code use the same tunnel logic.

**Files**: `profiles.py`, `settings.py`, new `remote/tunnel.py`, `embeddings/readiness.py`

### Phase 2: GraphMeta Node

Add `GraphMeta` to facility schema. Create on `graph db start` or first
write. Check on `graph db status`.

```python
def ensure_graph_meta(client, name: str, facilities: list[str]) -> None:
    client.query("""
        MERGE (m:GraphMeta {id: "meta"})
        ON CREATE SET m.name = $name, m.facilities = $facilities,
                      m.created_at = datetime(), m.updated_at = datetime()
        ON MATCH SET m.updated_at = datetime()
    """, name=name, facilities=facilities)

def check_graph_identity(client, expected_name: str) -> str | None:
    """Return warning if graph name doesn't match, None if OK."""
    result = client.query("MATCH (m:GraphMeta) RETURN m.name AS name")
    if not result:
        return None  # No meta yet — first use
    actual = result[0]["name"]
    if actual != expected_name:
        return f"Graph identity mismatch: expected '{expected_name}', found '{actual}'"
    return None
```

**Files**: new `graph/meta.py`, `graph/client.py`, schema update

### Phase 3: Name/Location Split

Replace `default = "iter"` with `name` + `location` + `locations` registry
in pyproject.toml. Update `resolve_graph()` to use the new model.

```toml
# New style
[tool.imas-codex.graph]
name = "codex"
location = "iter"

[tool.imas-codex.graph.locations]
iter = 0
tcv = 1
jt60sa = 2

# Backward compatible — treated as name="codex", location="iter"
[tool.imas-codex.graph]
default = "iter"
```

Migration path: if only `default` is set, infer `location = default` and
`name = "codex"` (or the facility name if it's a single-facility profile).

**Files**: `profiles.py`, `settings.py`, `pyproject.toml`

### Phase 4: Ingestion Gating

Before writing to graph, check `GraphMeta.facilities`:

```python
def gate_ingestion(client, facility_id: str) -> None:
    meta = client.query("MATCH (m:GraphMeta) RETURN m.facilities")
    if meta and facility_id not in meta[0]["facilities"]:
        raise ValueError(
            f"Graph '{meta[0].get('name', '?')}' does not include "
            f"facility '{facility_id}'. Add it to GraphMeta.facilities first."
        )
```

Wire into `add_to_graph()` and ingestion pipeline entry points.

**Files**: `graph/meta.py`, `agentic/server.py`, `ingestion/`

### Phase 5: Local Graph Store

Implement the local store for fast graph switching without re-fetching.

```python
from pathlib import Path

STORE_DIR = Path.home() / ".local/share/imas-codex/graphs"

def dump_to_store(name: str) -> Path:
    """Dump current graph to local store under given name."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    path = STORE_DIR / f"{name}.tar.gz"
    neo4j_admin_dump(path)  # Use neo4j-admin dump
    return path

def load_from_store(name: str) -> None:
    """Load graph from local store into Neo4j."""
    path = STORE_DIR / f"{name}.tar.gz"
    if not path.exists():
        raise FileNotFoundError(f"Graph '{name}' not in local store. Run: graph fetch {name}")
    neo4j_admin_load(path)  # Use neo4j-admin load

def list_store() -> list[dict]:
    """List graphs in local store with metadata."""
    graphs = []
    for path in STORE_DIR.glob("*.tar.gz"):
        meta = extract_graph_meta(path)  # Read GraphMeta from archive
        graphs.append({
            "name": path.stem,
            "size": path.stat().st_size,
            "facilities": meta.get("facilities", []),
        })
    return graphs

def switch_graph(target_name: str) -> None:
    """Switch to a different graph, preserving current."""
    current = get_current_graph_name()
    if current and current != target_name:
        dump_to_store(current)
    load_from_store(target_name)
    verify_graph_meta(target_name)
```

**Files**: new `graph/store.py`, `cli/graph_cli.py`

### Phase 6: GHCR Integration

Update fetch/push to use new naming conventions. Fetch downloads to local
store, push uploads from active graph.

```python
def fetch_graph(name: str, version: str = "latest") -> Path:
    """Download graph from GHCR to local store."""
    tag = f"{name}:{version}" if version else f"{name}:latest"
    image = f"ghcr.io/iterorganization/imas-codex-graph:{tag}"
    
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    path = STORE_DIR / f"{name}.tar.gz"
    
    # Download OCI artifact to local path
    oras_pull(image, path)
    return path

def push_graph(version: str | None = None) -> str:
    """Push current graph to GHCR."""
    name = get_current_graph_name()
    tag = f"{name}:{version}" if version else f"{name}:latest"
    image = f"ghcr.io/iterorganization/imas-codex-graph:{tag}"
    
    # Dump current graph and push
    path = dump_to_store(name)
    oras_push(image, path)
    return image
```

**Files**: `graph/store.py`, `graph/oci.py`

### Phase 7: CLI Updates

```bash
# Show graph identity
imas-codex graph db status
# → Neo4j [codex] running at iter (bolt://localhost:17687 via tunnel)
#   Facilities: iter, tcv, jt60sa
#   DD version: 4.1.0
#   Nodes: 244,344  Relationships: 581,000

# Override location for this session
IMAS_CODEX_GRAPH_LOCATION=local imas-codex graph db status
# → Neo4j [codex] running locally (bolt://localhost:7687)

# Fetch graph from GHCR (does not modify active graph)
imas-codex graph fetch tcv
# → Downloaded tcv:latest to ~/.local/share/imas-codex/graphs/tcv.tar.gz

imas-codex graph fetch codex:v4.0.0
# → Downloaded codex:v4.0.0 to ~/.local/share/imas-codex/graphs/codex.tar.gz

# Switch graphs (auto-dumps current first)
imas-codex graph switch tcv
# → Dumped current graph (codex) to store
# → Loaded tcv from store
# → Active graph: tcv (facilities: tcv)

# Create new empty graph
imas-codex graph new sandbox --facilities iter,tcv
# → Created GraphMeta: name=sandbox, facilities=[iter, tcv]

# Clear graph data (keeps GraphMeta)
imas-codex graph clear
# → This will delete all nodes and relationships. Continue? [y/N]

# Add facility to existing graph
imas-codex graph meta add-facility jt60sa

# List graphs in local store
imas-codex graph store list
# → codex.tar.gz (2.1 GB, facilities: iter, tcv, jt60sa)
# → tcv.tar.gz (340 MB, facilities: tcv)
# → sandbox.tar.gz (empty)

# Push current graph to GHCR
imas-codex graph push
# → Pushed codex to ghcr.io/iterorganization/imas-codex-graph:codex-latest
```

**Files**: `cli/graph_cli.py`

## Migration

1. **Phase 1**: Auto-tunnels with port offsets — zero user friction, works immediately
2. **Phase 2**: GraphMeta node — added on first `graph db status`, non-breaking
3. **Phase 3**: Name/location split — deprecates `default` key with warning
4. **Phase 4**: Ingestion gating — prevents writes to wrong graph
5. **Phase 5**: Local store — enables fast switching without re-fetching
6. **Phase 6**: GHCR integration — `fetch`/`push` with versioned tags
7. **Phase 7**: CLI updates — `fetch`, `switch`, `new`, `clear`, `store list`

**Data safety**: No migration step loses data. The local store is additive.
Existing `graph dump/load` commands continue to work during transition.

**Backward compatibility**: Existing `profiles.iter`, `profiles.tcv` sections
continue to work. The new `[graph.locations]` table only adds port offset
lookups; explicit profile overrides take precedence.

## FAQ: How It All Fits Together

### Q: How are port offsets calculated?

Port offsets are per-**location**, not per-graph-name. Defined in
`pyproject.toml` under `[tool.imas-codex.graph.locations]`:

```
Location   Offset   Local Port   Tunnel Port
iter       0        7687         17687
tcv        1        7688         17688
jt60sa     2        7689         17689
local      0        7687         (no tunnel)
```

Formula: `port = 7687 + location_offset` (add +10000 for tunnel).

### Q: Can I run multiple Neo4j instances on one host?

**No — by design.** Each host runs ONE Neo4j. You switch graphs using
`graph switch <name>`, which automatically dumps the current graph before
loading the target. No data is lost. This keeps port management simple.

### Q: I'm on WSL and want to access codex on iter, tcv on tcv, and sandbox locally. How?

Each uses a different location, so each gets a unique port:

```bash
# All three can be active simultaneously

# codex on iter → tunnel to iter:7687 → localhost:17687
IMAS_CODEX_GRAPH=codex IMAS_CODEX_GRAPH_LOCATION=iter imas-codex graph db status

# tcv on tcv → tunnel to tcv:7688 → localhost:17688  
IMAS_CODEX_GRAPH=tcv IMAS_CODEX_GRAPH_LOCATION=tcv imas-codex graph db status

# sandbox locally → direct localhost:7687
IMAS_CODEX_GRAPH=sandbox IMAS_CODEX_GRAPH_LOCATION=local imas-codex graph db status
```

### Q: How do I avoid losing data when switching graphs?

You don't need to worry — `switch` automatically dumps the current graph
before loading the target:

```bash
imas-codex graph switch tcv
# 1. Reads GraphMeta → current graph is "codex"
# 2. Dumps codex to ~/.local/share/imas-codex/graphs/codex.tar.gz
# 3. Loads tcv from local store
# 4. Verifies GraphMeta shows "tcv"
```

The only data-destructive command is `graph clear`, which requires
confirmation.

### Q: What's the difference between `fetch` and `switch`?

- **`fetch`** downloads from GHCR to local store. Does NOT modify the
  active graph. Safe to run anytime.
- **`switch`** swaps the active graph. Dumps current → loads target from
  local store. Requires target to exist in local store (fetch first).

```bash
imas-codex graph fetch codex:v4.0.0   # Download specific version
imas-codex graph switch codex         # Load it (dumps current first)
```

### Q: How do I create a new empty graph?

```bash
imas-codex graph new my-sandbox --facilities iter,tcv
# Creates empty graph with GraphMeta: name=my-sandbox, facilities=[iter, tcv]
```

This does NOT switch to the new graph — it only creates the metadata.
Use `switch` afterward if you want to work with it.

### Q: What if I want both codex AND tcv graphs accessible on iter?

You can't run two Neo4j instances on iter. Instead:
1. Load codex on iter (the canonical case)
2. Load tcv on tcv (where it belongs)
3. Query each via its respective tunnel

Or, if you must have both at the same location:  
Load the multi-facility graph (codex) which includes tcv data.

### Q: How do I permanently delete a graph?

Local store graphs are just files — delete them directly:
```bash
rm ~/.local/share/imas-codex/graphs/sandbox.tar.gz
```

To delete a graph from GHCR, use `graph remove`:
```bash
imas-codex graph remove sandbox --from-ghcr
```

The `clear` command only clears the **active** graph's data, it doesn't
delete the graph itself. After `clear`, the graph still exists with its
GraphMeta intact — it's just empty.

### Q: Where is the "single Neo4j per host" constraint enforced?

It's not hard-enforced in code — it's a design convention. The port
allocation scheme assumes it. If you manually start a second Neo4j on a
non-standard port, you'd need to configure it explicitly in a profile
override.

### Q: Are auto-tunnels linked to location offsets?

Yes. When you specify `location=iter` and you're not on iter:
1. Look up iter's offset (0) → remote port = 7687
2. Calculate tunnel port = 7687 + 10000 = 17687
3. Auto-launch: `ssh -f -N -L 17687:localhost:7687 iter`
4. Connect to `bolt://localhost:17687`

The offset cascade: `location → offset → remote_port → tunnel_port`.

## Open Questions

- Should `codex` be a reserved name, or should any multi-facility graph use
  alphabetical facility lists? Recommendation: `codex` is reserved for
  "all known facilities" — the canonical development graph. Custom subsets
  use explicit facility lists.
- Should `graph push` embed GraphMeta in the OCI artifact? Yes — the
  GHCR tag should include the graph name for discoverability.
- Should location offsets be hardcoded defaults or require explicit config?
  Recommendation: hardcode defaults for known facilities (iter=0, tcv=1,
  jt60sa=2) in code; `pyproject.toml` registry is for overrides/additions.
- Should we support "multi-instance" mode for power users who want multiple
  Neo4j on one host? Probably not — adds complexity for little benefit.
  Power users can manually configure profile overrides with explicit ports.
- Should `fetch` support version tags (`codex:v4.0.0`) or only `latest`?
  Recommendation: support both. Tags for reproducibility, `latest` for
  convenience. Version tag is optional, defaults to `latest`.
