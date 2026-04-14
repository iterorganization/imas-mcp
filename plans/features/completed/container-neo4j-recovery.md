# Fix: Neo4j Database Unavailable in Container

## Problem

The Azure-deployed container serves `/health` with HTTP 200, but all graph
statistics are `null`. Neo4j logs show persistent
`Neo.TransientError.General.DatabaseUnavailable` — the DBMS is running but the
`neo4j` database is stuck OFFLINE.

### Root Cause

The container's `docker-entrypoint.sh` checks **HTTP readiness**
(`curl localhost:7474`), not **database readiness**. In Neo4j 2026, HTTP
responds as soon as the JVM binds the connector — before the user database
finishes recovery. The MCP server starts immediately and every graph query
fails.

On CI runners (fast SSD, 7 GB+ RAM) recovery completes in ~30 s and the
database is ONLINE before the first graph query. On Azure App Service
(network-attached storage, constrained memory) recovery takes much longer or
**never completes** — the database stays OFFLINE permanently.

### Contributing Factors

| Factor | CI Runner | Azure App Service |
|--------|-----------|-------------------|
| Storage I/O | Local SSD | Network-attached (slow) |
| RAM available | 7+ GB | 1.75 GB (B1) / 3.5 GB (B2) |
| Neo4j heap | 256–512 MB | 256–512 MB |
| Graph dump size | 1.9 GB | 1.9 GB |
| Recovery time | ~30 s | 4+ min / stuck |
| `system` DB exists | No (first start) | No (first start) |

The 256–512 MB heap may be insufficient for WAL replay on a 1.9 GB graph,
especially on slow I/O where GC pressure amplifies. The `system` database
must also bootstrap from scratch on every container start since `neo4j-admin
database load` doesn't create it.

### Auth is NOT the Issue

`dbms.security.auth_enabled=false` is set in neo4j.conf. The Python driver
sends `auth=("neo4j", "neo4j")` which Neo4j silently ignores. This is
harmless.

### Graph Data IS Shipped

The multi-stage build correctly pulls the graph dump from GHCR, loads it with
`neo4j-admin database load`, and copies `/data` → `/opt/neo4j/data`. CI smoke
tests verify `node_count > 0` (84,166 nodes). No additional GHCR pull is
needed at runtime.

---

## Fix: Build-Time Recovery + Runtime Readiness Gate

Two complementary changes that together eliminate the problem:

### Fix 1 — Pre-start Neo4j in the `graph-loader` build stage

Complete database recovery at **build time** on fast CI runners, so the
container ships a fully-recovered, clean database that opens instantly at
runtime.

**File: `Dockerfile` (graph-loader stage, after `neo4j-admin database load`)**

After the existing `neo4j-admin database load` step (line ~207), add:

```dockerfile
# Pre-start Neo4j to complete WAL recovery and create system DB.
# This shifts the expensive recovery from runtime (slow Azure I/O)
# to build time (fast CI SSD). The database ships in ONLINE state.
RUN if [ ! -f /tmp/graph-pull/.no-graph ]; then \
    echo "Pre-starting Neo4j for database recovery..." && \
    /var/lib/neo4j/bin/neo4j-admin server start && \
    for i in $(seq 1 120); do \
        if /var/lib/neo4j/bin/cypher-shell "RETURN 1" > /dev/null 2>&1; then \
            echo "✓ Database recovered (${i}s)"; \
            break; \
        fi; \
        sleep 1; \
    done && \
    /var/lib/neo4j/bin/neo4j stop && \
    echo "✓ Neo4j shut down cleanly — database is recovery-free"; \
fi
```

**What this achieves:**
- WAL replay happens on CI runner (SSD, 7+ GB RAM) — fast
- The `system` database is created and properly registered
- Clean shutdown writes a checkpoint — no recovery needed on next start
- Container startup becomes near-instant (~5–10 s instead of 30+ s)

**Trade-offs:**
- Build time increases by ~30–60 s (recovery + clean shutdown)
- Image size increases slightly (system DB metadata files, ~10 MB)
- Both are acceptable given the reliability gain

**Investigation needed:** Verify that `neo4j-admin server start` is the
correct command in Neo4j 2026 for starting the server non-interactively in a
build stage. Alternatives: `neo4j console &` with background + wait + stop.
Also verify that `cypher-shell` works without auth when
`dbms.security.auth_enabled` hasn't been set in this stage (the conf
modifications happen in the final stage). May need to set
`--env NEO4J_AUTH=none` or pass a conf flag.

### Fix 2 — Bolt-level readiness gate in `docker-entrypoint.sh`

Defense-in-depth: even with build-time recovery, always verify the database
is queryable before starting the MCP server. This handles edge cases (empty
graph fallback, Azure filesystem quirks, future regressions).

**File: `docker-entrypoint.sh`**

Replace the current HTTP-only check with a two-phase check:

```bash
# Phase 1: Wait for Neo4j DBMS (HTTP)
echo "Waiting for Neo4j DBMS..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:7474/ > /dev/null 2>&1; then
        echo "Neo4j DBMS ready (${i}s)"
        break
    fi
    if ! kill -0 "$NEO4J_PID" 2>/dev/null; then
        echo "Neo4j process exited unexpectedly. Log:"
        cat "$NEO4J_LOG" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

if ! curl -sf http://localhost:7474/ > /dev/null 2>&1; then
    echo "Neo4j DBMS failed to start within 120s. Log:"
    tail -50 "$NEO4J_LOG" 2>/dev/null || true
    exit 1
fi

# Phase 2: Wait for neo4j database to be ONLINE (queryable via Bolt)
echo "Waiting for neo4j database..."
DB_READY=0
for i in $(seq 1 180); do
    if "${NEO4J_HOME}/bin/cypher-shell" -a bolt://127.0.0.1:7687 \
        "RETURN 1" > /dev/null 2>&1; then
        echo "Neo4j database ready (${i}s)"
        DB_READY=1
        break
    fi
    if ! kill -0 "$NEO4J_PID" 2>/dev/null; then
        echo "Neo4j process exited during database recovery. Log:"
        tail -100 "$NEO4J_LOG" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

if [ "$DB_READY" -eq 0 ]; then
    echo "WARNING: Neo4j database not ready after 180s — starting MCP server anyway"
    echo "Neo4j log tail:"
    tail -30 "$NEO4J_LOG" 2>/dev/null || true
fi
```

**Key design decisions:**
- Phase 1 timeout: 120 s (DBMS startup, same as current)
- Phase 2 timeout: 180 s (database recovery — generous for slow Azure I/O)
- Non-fatal Phase 2 timeout: MCP server starts anyway with degraded graph,
  rather than crashing the container (Azure would restart loop otherwise)
- Uses `cypher-shell` which is bundled in the Neo4j image — no additional
  dependencies

**Note on `cypher-shell` and auth:** When `dbms.security.auth_enabled=false`,
`cypher-shell` works without `-u`/`-p` flags. However, the conf is set in
the final stage, not the graph-loader stage. Verify that `cypher-shell`
connects correctly.

### Fix 3 — Increase Neo4j heap for recovery (optional)

If build-time recovery (Fix 1) is implemented, the runtime heap can stay
small. But as defense-in-depth, bumping the heap slightly helps edge cases:

**File: `Dockerfile` (line 244–245)**

```dockerfile
echo "server.memory.heap.initial_size=384m" >> /opt/neo4j/conf/neo4j.conf && \
echo "server.memory.heap.max_size=768m" >> /opt/neo4j/conf/neo4j.conf && \
```

This is conservative — the container has ~2.5 GB total memory budget
(Qwen model takes ~1.5 GB). Only increase if Azure plan provides ≥ 3.5 GB.

### Fix 4 — Health endpoint reports database status (optional)

Make the health response clearly indicate when the database is recovering vs
permanently unavailable:

**File: `imas_codex/llm/server.py` (`_query_graph` function)**

Add a `graph.status` field: `"online"`, `"recovering"`, or `"unavailable"`.
Currently returns nulls silently. With an explicit status field, monitoring
can distinguish "still starting" from "permanently broken".

---

## Implementation Order

1. **Fix 1** (build-time recovery) — eliminates the root cause
2. **Fix 2** (entrypoint readiness gate) — defense-in-depth
3. Test locally with `docker build` + `docker run` — verify database is
   queryable within 10 s of container start
4. **Fix 3** (heap bump) — only if testing shows recovery still slow
5. **Fix 4** (health status) — nice-to-have for monitoring
6. Release RC, monitor Azure health endpoint

## Validation

- `docker build -t imas-codex-test .` succeeds (build-time recovery completes)
- `docker run -p 8000:8000 imas-codex-test` → `/health` returns `node_count > 0`
  within 15 s
- Neo4j log shows no WAL recovery at runtime (already done at build time)
- Azure deployment returns `/health` with graph stats populated
