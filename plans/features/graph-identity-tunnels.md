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

### Port Allocation: Tunnel Offset

Replace same-port tunneling with a **deterministic offset**:

```
LOCAL instances (Neo4j runs on this machine):
  bolt = 7687 + facility_offset     (7687, 7688, 7689, ...)
  http = 7474 + facility_offset     (7474, 7475, 7476, ...)

TUNNEL instances (SSH tunnel to remote Neo4j):
  bolt = 17687 + facility_offset    (17687, 17688, 17689, ...)
  http = 17474 + facility_offset    (17474, 17475, 17476, ...)
```

Tunnel offset is +10000. This is:
- **Deterministic** — no configuration needed
- **Clash-proof** — local and tunnel ranges never overlap
- **Unique per facility** — two facilities can't share a tunnel port

Example on WSL:
```
Local Neo4j (codex, development):   bolt://localhost:7687
Tunnel to ITER (codex, production): bolt://localhost:17687
Tunnel to TCV (tcv):                bolt://localhost:17688
```

### Automatic Tunnel Management

When `_resolve_uri()` detects a remote host:

1. Calculate tunnel port: `bolt_port + TUNNEL_OFFSET`
2. Check if tunnel port is already bound → use it
3. If not bound → auto-launch: `ssh -f -N -L {tunnel_port}:localhost:{bolt_port} {host}`
4. Verify tunnel came up (socket probe with timeout)
5. Connect to `bolt://localhost:{tunnel_port}`

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

Replace `[tool.imas-codex.graph].default` with two fields:

```toml
[tool.imas-codex.graph]
name = "codex"          # Graph identity (what data)
location = "iter"       # Where Neo4j runs (where)
username = "neo4j"
password = "imas-codex"
```

Resolution:
- `name` selects ports via facility-offset convention (or explicit profile)
- `location` selects host (SSH alias) for locality detection
- Together they determine: which ports at which host via which transport

**Key semantic change**: `name` and `location` are orthogonal. "codex" at
"iter" uses iter's bolt port (7687) on the iter host. "codex" at "local"
uses the same bolt port (7687) locally.

For facility-specific graphs that run at their own facility:
```bash
# TCV graph running on TCV machine
IMAS_CODEX_GRAPH=tcv IMAS_CODEX_GRAPH_LOCATION=tcv imas-codex graph db status
```

### Default Behavior Matrix

| You're on | name | location | Result |
|-----------|------|----------|--------|
| ITER | codex | iter (default) | Direct bolt://localhost:7687 |
| WSL | codex | iter (default) | Auto-tunnel bolt://localhost:17687 |
| WSL | codex | local | Direct bolt://localhost:7687 (local Neo4j) |
| TCV | tcv | tcv | Direct bolt://localhost:7688 |
| WSL | tcv | tcv | Auto-tunnel bolt://localhost:17688 |
| WSL | tcv | local | Direct bolt://localhost:7688 (local Neo4j) |

## Implementation Plan

### Phase 1: Port Clash Prevention (zero user friction)

Add `TUNNEL_OFFSET = 10000` constant. Update `_resolve_uri()`:

```python
TUNNEL_OFFSET = 10000

def _resolve_uri(host: str | None, bolt_port: int) -> str:
    if host is None or is_local_host(host):
        return f"bolt://localhost:{bolt_port}"
    tunnel_port = bolt_port + TUNNEL_OFFSET
    _ensure_graph_tunnel(host, bolt_port, tunnel_port)
    return f"bolt://localhost:{tunnel_port}"
```

Extract `_ensure_ssh_tunnel()` from `embeddings/readiness.py` to
`imas_codex/remote/tunnel.py` as shared utility. Both embeddings and
graph code use the same tunnel logic.

**Files**: `profiles.py`, new `remote/tunnel.py`, `embeddings/readiness.py`

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

Replace `default = "iter"` with `name` + `location` in pyproject.toml.
Update `resolve_graph()` to use the two-field model. Backward-compatible:
if only `default` is set, infer name from location for known facilities,
or use "codex" if location is a known facility host.

```toml
# New style
[tool.imas-codex.graph]
name = "codex"
location = "iter"

# Backward compatible — treated as name="iter", location="iter"
[tool.imas-codex.graph]
default = "iter"
```

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

### Phase 5: CLI Updates

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

# Initialize a new facility-specific graph
imas-codex graph init tcv --location local
# → Created GraphMeta: name=tcv, facilities=[tcv]

# Add facility to existing graph
imas-codex graph meta add-facility jt60sa
```

**Files**: `cli/graph_cli.py`

## Migration

1. Deploy Phase 1 first — existing users get auto-tunnels with no config change
2. Phase 2 adds GraphMeta on next `graph db status` (non-breaking)
3. Phase 3 deprecates `default` key with a warning (reads old config fine)
4. No port assignments change — existing tunnels/local instances keep working

## Open Questions

- Should `codex` be a reserved name, or should any multi-facility graph use
  alphabetical facility lists? Recommendation: `codex` is reserved for
  "all known facilities" — the canonical development graph. Custom subsets
  use explicit facility lists.
- Should `graph push/pull` embed GraphMeta in the OCI artifact? Yes — the
  GHCR tag should include the graph name for discoverability.
