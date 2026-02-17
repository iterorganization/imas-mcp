# Graph Merge

Transfer facility data between named graph instances (e.g. new TCV updates from `codex` → `tcv-imas`).

## Context

Each named graph (`neo4j-codex/`, `neo4j-tcv-imas/`) is a self-contained Neo4j data directory with its own `(:GraphMeta)` node declaring name + facilities. Merge enables composing graphs from independent facility pipelines without rebuilding from scratch.

## Architecture

### Merge direction

```
source graph  ──[facility filter]──►  target graph
(codex)           (tcv nodes)         (tcv-imas)
```

### What travels

1. **Facility-scoped nodes** — all nodes where `facility_id = <facility>` (FacilityPath, SourceFile, CodeChunk, FacilitySignal, TreeNode, etc.)
2. **Relationships** from/to those nodes
3. **Shared DD nodes** — IMASPath, DDVersion, Unit, IMASCoordinateSpec — needed by cross-facility relationships (MAPS_TO_IMAS, etc.)
4. **GraphMeta** — target's facility list is updated to include the merged facility

### What does NOT travel

- Nodes belonging to other facilities
- Orphaned nodes with no relationships
- Vector indexes (must be recreated on target)
- Constraints/indexes (must exist on target already)

## Approach

### Option A: Cypher-level merge (dual-driver)

Connect to both graphs simultaneously. Read source nodes/rels, MERGE into target.

- **Pro**: No stop/start cycle, works while target is live, incremental
- **Con**: Requires two Neo4j instances running simultaneously (different ports), more code, large graphs may be slow node-by-node

### Option B: Export-filter-load

Use existing `graph export -f <facility>` to create a per-facility dump, then load into target.

- **Pro**: Uses existing infrastructure, battle-tested dump/load, handles large graphs efficiently
- **Con**: Requires stop/start of target, full facility replacement (not incremental merge)

### Option C: APOC-based (if available)

Use `apoc.export.cypher.all()` to generate Cypher scripts from source, replay on target.

- **Pro**: Portable, human-readable intermediate format
- **Con**: APOC not installed in our Apptainer image, slow for large graphs

### Recommended: Option A for incremental, Option B for full replacement

Implement Option B first (simpler, reuses existing code), Option A later for live incremental merges.

## Implementation Outline

### Phase 1: Full facility transfer (Option B)

```bash
# Export TCV data from codex
imas-codex graph export -f tcv -g codex -o tcv-export.tar.gz

# Load into tcv-imas (replaces existing TCV data)
imas-codex graph load tcv-export.tar.gz -g tcv-imas
```

Already works today. Missing piece: `graph load` should update GraphMeta on target to add the facility.

### Phase 2: Incremental merge (Option A)

```bash
# Merge TCV updates from codex into tcv-imas (live, no restart)
imas-codex graph merge --from codex --to tcv-imas --facility tcv
```

Implementation:
1. Resolve source and target profiles → two `GraphClient` instances
2. Query source: `MATCH (n {facility_id: $fac}) RETURN n` in batches
3. For each batch, `UNWIND $nodes AS node MERGE (n:Label {id: node.id}) SET n += node` on target
4. Query source relationships: `MATCH (a {facility_id: $fac})-[r]->(b) RETURN type(r), a.id, b.id, properties(r)`
5. Replay relationships on target via MERGE
6. Transfer DD nodes referenced by merged facility nodes
7. Update target GraphMeta: `add_facility_to_meta(target_client, facility)`

### Edge cases

- **Embedding vectors**: Large (256-1024 dim). Transfer them or re-embed on target? Transfer is simpler but bloats Cypher traffic. Could use a side-channel (export embeddings as numpy, bulk-set via parameter).
- **Conflicting IDs**: Not expected — IDs are `facility:path` scoped. DD nodes use deterministic IDs.
- **Schema drift**: Source and target may have different schema versions. Validate before merge.
- **Partial merge**: What if source has updated some TCV nodes but not others? MERGE handles this — only changed properties get updated.

## Complexity

Medium (~2-3 days for Phase 1 polish, ~1 week for Phase 2 dual-driver merge).

## Dependencies

- Named graph directories (symlink architecture) — so we can run two instances simultaneously for Option A
- GraphMeta with facility tracking — for post-merge identity update
- Per-facility export (already exists)

## Not in scope

- Cross-facility relationship merging (e.g. merging a relationship that spans two facilities)
- Conflict resolution UI (if both graphs modified the same node differently)
- Automated merge triggers (CI/CD)
