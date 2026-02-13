# Knowledge Graph Architecture

> **Module**: `imas_codex.graph`

## Overview

The imas-codex knowledge graph is a Neo4j-based store that unifies:
- **Facility knowledge**: TreeNodes, CodeChunks, Diagnostics, Analysis Codes, Wiki content
- **IMAS Data Dictionary**: IMASPath nodes with version tracking and embeddings
- **Cross-facility data**: Shared IMAS mappings, semantic clusters

All schema definitions live in **LinkML** (`schemas/*.yaml`) as the single source of truth.

## Graph Profiles

Named graph profiles allow switching between Neo4j instances at runtime.
Each profile maps to a specific host, port, and data directory.

### Port Convention

| Facility | Bolt | HTTP | Data Dir |
|----------|------|------|----------|
| iter | 7687 | 7474 | `neo4j/` |
| tcv | 7688 | 7475 | `neo4j-tcv/` |
| jt60sa | 7689 | 7476 | `neo4j-jt60sa/` |

### Location-Aware Resolution

The `host` field on each profile records where Neo4j physically runs:

- `host="iter"` — Neo4j runs on the ITER login node
- `host=None` — Neo4j runs locally
- At connection time, `is_local_host(host)` determines direct vs tunnel access

**From ITER** (host matches local machine):
```
resolve_graph("iter") → bolt://localhost:7687  (direct)
```

**From WSL** (host is remote):
```
resolve_graph("iter") → bolt://localhost:7687  (via SSH tunnel)
```

**Dual-instance** (local + tunneled, conflicting ports):
```bash
# .env
IMAS_CODEX_TUNNEL_BOLT_ITER=17687
# Then: ssh -f -N -L 17687:localhost:7687 iter
```

### Profile Resolution Priority

1. `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` env vars (escape hatch)
2. Explicit `[tool.imas-codex.graph.profiles.<name>]` in pyproject.toml
3. Convention-based port mapping for known facility names
4. `IMAS_CODEX_GRAPH` env var selects the active profile

### Configuration

```toml
# pyproject.toml
[tool.imas-codex.graph]
default = "iter"
username = "neo4j"
password = "imas-codex"

# Optional explicit profile override
[tool.imas-codex.graph.profiles.staging]
host = "staging-server"
bolt-port = 7700
http-port = 7701
```

## Graph Client

```python
from imas_codex.graph import GraphClient

# Use active profile
with GraphClient() as client:
    result = client.query("MATCH (n:Facility) RETURN n.id")

# Use specific profile
with GraphClient.from_profile("tcv") as client:
    print(client.get_stats())

# Runtime switching
import os
os.environ["IMAS_CODEX_GRAPH"] = "tcv"
# Next GraphClient() will connect to tcv
```

## Graph Management CLI

### Database Operations

```bash
# Start/stop/status
imas-codex graph db start              # Start active profile
imas-codex graph db start -g tcv       # Start specific profile
imas-codex graph db stop -g tcv
imas-codex graph db status -g tcv
imas-codex graph db profiles           # List all profiles
imas-codex graph db shell              # Interactive Cypher shell
imas-codex graph db service install    # Install systemd service
```

### Graph Lifecycle

```bash
# Dump and load
imas-codex graph dump                  # Full graph dump
imas-codex graph dump --facility tcv   # Per-facility dump (filtered)
imas-codex graph load archive.tar.gz   # Load from archive

# GHCR registry
imas-codex graph push                  # Push release to GHCR
imas-codex graph push --dev            # Push dev build
imas-codex graph push --facility tcv   # Push per-facility graph
imas-codex graph pull                  # Pull latest from GHCR
imas-codex graph pull --facility tcv   # Pull per-facility graph
imas-codex graph list                  # List GHCR versions
imas-codex graph list --facility tcv   # List per-facility versions

# Backup and restore
imas-codex graph backup                # Create neo4j-admin dump backup
imas-codex graph restore               # Restore from backup (interactive)
imas-codex graph restore backup.dump   # Restore specific backup
imas-codex graph clear                 # Clear graph (auto-backup first)

# Cleanup
imas-codex graph remove tag1 tag2      # Delete GHCR tags
imas-codex graph remove --dev          # Remove all dev tags
imas-codex graph remove --backups --older-than 30d  # Clean old backups
```

### SSH Tunnels

```bash
imas-codex graph tunnel start          # Start tunnel to active profile's host
imas-codex graph tunnel start iter     # Start tunnel to specific host
imas-codex graph tunnel stop iter
imas-codex graph tunnel status         # Show active tunnels
```

## Per-Facility Federation

Full graph contains all facilities. Per-facility graphs are extracted via dump-and-clean:

1. Dump the full graph via `neo4j-admin database dump`
2. Load into a temporary Neo4j instance
3. Delete nodes with `facility_id != target_facility`
4. Delete orphaned non-DD nodes (no relationships)
5. Re-dump the cleaned graph

This preserves the full IMAS Data Dictionary (shared across facilities) while isolating facility-specific data.

```bash
# Create and push per-facility graph
imas-codex graph dump --facility tcv
imas-codex graph push --facility tcv --dev

# End user pulls only their facility
export IMAS_CODEX_GRAPH=tcv
imas-codex graph pull --facility tcv
imas-codex graph db start
```

## GHCR Package Naming

| Package | Content |
|---------|---------|
| `imas-codex-graph` | Full unified graph (all facilities) |
| `imas-codex-graph-tcv` | TCV-only graph + IMAS DD |
| `imas-codex-graph-jt60sa` | JT-60SA-only graph + IMAS DD |

## Schema Management

### LinkML as Single Source of Truth

```
imas_codex/schemas/
├── common.yaml      # Shared enums, PhysicsDomain
├── facility.yaml    # Facility nodes (TreeNode, CodeChunk, etc.)
└── imas_dd.yaml     # IMAS DD nodes (IMASPath, DDVersion, etc.)
```

Models auto-generated during `uv sync` via build hook. Regenerate manually:
```bash
uv run build-models --force
```

## Vector Indexes

| Index | Content |
|-------|---------|
| `imas_path_embedding` | IMASPath nodes |
| `cluster_centroid` | IMASSemanticCluster centroids |
| `code_chunk_embedding` | CodeChunk nodes |
| `wiki_chunk_embedding` | WikiChunk nodes |
| `facility_signal_desc_embedding` | FacilitySignal descriptions |
| `facility_path_desc_embedding` | FacilityPath descriptions |
| `tree_node_desc_embedding` | TreeNode descriptions |
| `wiki_artifact_desc_embedding` | WikiArtifact descriptions |

## Docker Compose

```bash
# Default ports (iter convention)
docker compose --profile graph up

# Custom ports for a different facility
BOLT_PORT=7688 HTTP_PORT=7475 docker compose --profile graph up
```
