# Export + Rebuild Pipeline for Filtered Graph Dumps

## Problem

The release workflow creates three graph dump variants: **full** (all nodes),
**dd-only** (IMAS Data Dictionary only), and **per-facility** (DD + one
facility). The current filtering approach in `temp_neo4j.py` is:

1. Stop production Neo4j
2. `neo4j-admin database dump` the entire graph (~1.9 GB)
3. Load the dump into a temporary Neo4j instance
4. Start the temp instance and wait for readiness
5. Run Cypher `DELETE` queries to remove unwanted nodes/relationships
6. Stop the temp instance
7. `neo4j-admin database dump` the filtered result
8. Restart production Neo4j

### Measured Costs

| Step | Duration | Notes |
|------|----------|-------|
| Stop production | ~5 s | Causes downtime for MCP server |
| Full dump (1.9 GB) | ~30 s | Entire graph serialized |
| Load into temp | ~20 s | Full dump loaded |
| Temp Neo4j startup + recovery | ~60–120 s | WAL replay on 1.9 GB |
| Cypher DELETE (dd-only) | ~300–600 s | `MATCH (n) WHERE NOT ... DETACH DELETE n` in batches |
| Cypher DELETE (facility) | ~120–300 s | More selective |
| Stop temp + dump filtered | ~30 s | |
| **Total per variant** | **~10–20 min** | × 3 variants = 25–50 min |

### Specific Failures

- **Production downtime**: ~30 s per dump cycle while Neo4j is stopped
- **OOM on CI**: Full graph load + Cypher DELETE + transaction log = memory
  spike. 4 GB SLURM jobs often fail; 8–16 GB required
- **Bloated output**: The filtered dump retains free-space from deleted nodes.
  DD-only dump (~200 MB of data) weighs ~900 MB because it was carved from a
  1.9 GB store
- **Slow filtering**: Cypher `DETACH DELETE` on 1.3 M nodes (for dd-only) is
  O(n) in the *removed* set, not the *kept* set

---

## Solution: Query Live → CSV → `neo4j-admin import` → Dump

Build filtered graphs from scratch rather than carving them from the full
dump. Query the **live** production Neo4j via Cypher, export matching nodes
and relationships to CSV, use `neo4j-admin database import full` to construct
a fresh compact database, create indexes, then dump.

### Architecture Overview

```
┌─────────────────────────────────────┐
│  LIVE Production Neo4j              │
│  bolt://98dci4-clu-2001:7687        │
│  ~1.5M nodes, ~4.4M relationships   │
└───────────┬─────────────────────────┘
            │ Cypher read queries
            │ (keyset pagination)
            ▼
┌─────────────────────────────────────┐
│  Phase 1: Export to CSV             │
│  • Per-label node CSVs              │
│  • Per-type relationship CSVs       │
│  • Index/constraint DDL captured    │
│  Timing: ~22 s (sequential)         │
│  Output: ~200 MB CSV (DD-only)      │
└───────────┬─────────────────────────┘
            │ CSV files on local disk
            ▼
┌─────────────────────────────────────┐
│  Phase 2: neo4j-admin import full   │
│  • Reads CSV directly (no Neo4j)    │
│  • Creates compact store files      │
│  • ID groups prevent cross-label    │
│    collisions                       │
│  Timing: ~40 s (projected)          │
│  Output: ~50 MB database store      │
└───────────┬─────────────────────────┘
            │ /data/databases/neo4j/
            ▼
┌─────────────────────────────────────┐
│  Phase 3: Create Indexes            │
│  • Start temp Neo4j (brief)         │
│  • CREATE CONSTRAINT / INDEX        │
│  • Wait for ONLINE state            │
│  • Stop temp Neo4j                  │
│  Timing: ~30–45 s                   │
└───────────┬─────────────────────────┘
            │ /data/databases/neo4j/
            ▼
┌─────────────────────────────────────┐
│  Phase 4: neo4j-admin dump          │
│  • Serializes compact store to dump │
│  Timing: ~15–25 s                   │
│  Output: ~25 MB .dump file          │
└─────────────────────────────────────┘
```

### Key Benefits

| Metric | Current | New Pipeline |
|--------|---------|-------------|
| Production downtime | ~30 s per variant | **Zero** (read-only queries) |
| DD-only total time | 10–20 min | **~1.5 min** |
| Per-facility total time | 5–10 min | **~2 min** |
| DD-only dump size | ~900 MB (bloated) | **~25 MB** (compact) |
| Memory requirement | 8–16 GB | **4 GB** sufficient |
| Failure mode | Corrupted temp store | CSV on disk (restartable) |

---

## Benchmarks

All benchmarks run against the live production graph on `98dci4-clu-2001`.
Neo4j 2026.01.4 Community in Apptainer. Measurements are wall-clock times
including network latency.

### DD Subgraph Statistics

| Label | Nodes | With Embeddings | CSV Size |
|-------|-------|-----------------|----------|
| IMASNode | 61,366 | 20,037 | 89.4 MB |
| IMASNodeChange | 137,310 | 0 | 22.5 MB |
| IMASSemanticCluster | 3,912 | 3,912 (×3 cols) | 36.6 MB |
| DDVersion | 35 | 0 | < 1 KB |
| Unit | 182 | 0 | < 1 KB |
| IMASCoordinateSpec | 109 | 0 | < 1 KB |
| IdentifierSchema | 62 | 62 | < 1 KB |
| IDS | 87 | 87 | < 1 KB |
| COCOS | 18 | 0 | < 1 KB |
| GraphMeta | 1 | 0 | < 1 KB |
| **Total** | **205,047** | **~24,100** | **~150 MB** |

Note: Embeddings are 256-dimensional float arrays. PhysicsDomain,
SignConvention, CoordinateRelationship, and ClusterMembership labels have
0 nodes currently and are excluded.

### Node Export Performance

Sequential export using keyset pagination (`WHERE n.id > $last LIMIT 5000`):

| Label | Batches | Time | Throughput |
|-------|---------|------|------------|
| IMASNode | 13 | 6.55 s | 9,369 nodes/s |
| IMASNodeChange | 28 | 3.35 s | 40,937 nodes/s |
| IMASSemanticCluster | 1 | 1.65 s | 2,372 nodes/s |
| Others (small) | 1 each | < 1 s | — |
| **Total** | — | **12.8 s** | — |

Throughput for IMASNode is lower because embedding serialization dominates
(~4.5 KB per 256-dim vector in CSV text form).

### Relationship Export Performance

| Relationship Type | Count | Source → Target |
|---|---|---|
| IN_VERSION | 137,310 | IMASNodeChange → DDVersion |
| FOR_IMAS_PATH | 94,158 | IMASNodeChange → IMASNode |
| INTRODUCED_IN | 61,583 | IMASNode/IDS → DDVersion |
| IN_IDS | 61,366 | IMASNode → IDS |
| HAS_PARENT | 60,334 | IMASNode → IMASNode |
| IN_CLUSTER | 33,873 | IMASNode → IMASSemanticCluster |
| HAS_ERROR | 31,281 | IMASNode → IMASNode |
| HAS_UNIT | 25,270 | IMASNode → Unit |
| HAS_COORDINATE (→Spec) | 13,769 | IMASNode → IMASCoordinateSpec |
| HAS_COORDINATE (→Node) | 12,467 | IMASNode → IMASNode |
| DEPRECATED_IN | 17,324 | IMASNode → DDVersion |
| COORDINATE_SAME_AS | 7,439 | IMASNode → IMASNode |
| RENAMED_TO | 2,696 | IMASNode → IMASNode |
| HAS_IDENTIFIER_SCHEMA | 327 | IMASNode → IdentifierSchema |
| HAS_PREDECESSOR | 34 | DDVersion → DDVersion |
| HAS_SUCCESSOR | 34 | DDVersion → DDVersion |
| HAS_COCOS | 17 | DDVersion → COCOS |
| **Total** | **559,282** | **8.9 s** |

Three relationship types carry properties:
- `HAS_ERROR`: `error_type` (string)
- `HAS_COORDINATE`: `dimension` (integer)
- `COORDINATE_SAME_AS`: `dimension` (integer)

### Parallel Export

Tested 4 concurrent threads exporting different labels simultaneously:

| Configuration | Time | Speedup |
|---|---|---|
| Sequential (1 thread) | 12.8 s | 1.0× |
| 4 threads | 9.7 s | 1.3× |

Bottleneck is the Neo4j server, not the client. Parallel export provides
marginal benefit and adds complexity. **Recommendation: sequential export.**

### neo4j-admin Import Performance

Tested with actual data structure (nodes with 256-dim embeddings,
relationships, label groups):

| Test Scale | Import Time | DB Size | Dump Size | Dump Time |
|---|---|---|---|---|
| 1K nodes + 999 rels | 2.5 s (+7 s JVM) | 2.3 MB | 1.0 MB | 8 s |
| 120K nodes + 200K rels | 22.5 s | 48.8 MB | 22.8 MB | 12 s |

**Projected for DD data (205K nodes, 559K rels):**
- Import: ~40 s
- Dump: ~25 s

### `--schema` Flag Incompatibility

`neo4j-admin database import full --schema=<path>` **fails** with:

> Record format batch import does not support schema changes

Indexes and constraints must be created **post-import** by briefly starting a
temp Neo4j instance. This adds ~30–45 s (Neo4j startup + DDL execution +
index population) but is the only supported path.

### Projected End-to-End Timing

| Phase | DD-Only | Per-Facility |
|---|---|---|
| Export nodes to CSV | 13 s | 15 s (+facility nodes) |
| Export relationships to CSV | 9 s | 12 s (+facility rels) |
| `neo4j-admin import full` | 40 s | 50 s |
| Start temp Neo4j | 20 s | 20 s |
| Create indexes + wait ONLINE | 15 s | 20 s |
| Stop temp Neo4j | 5 s | 5 s |
| `neo4j-admin dump` | 25 s | 30 s |
| **Total** | **~2 min** | **~2.5 min** |

vs current: **10–20 min** (dd-only), **5–10 min** (per-facility).

---

## Design Decisions

### CSV Format (not Parquet, not JSONL)

- **Human-readable** for debugging — `head -5 nodes_IMASNode.csv`
- **Native vector support** — `--vector-delimiter=;` handles float arrays
- **~200 MB total** for DD — well within memory/disk constraints
- Parquet would save ~30% on disk but adds a build dependency and loses
  readability. The total CSV volume is trivially small.

### Label-Specific ID Groups

COCOS nodes use integer IDs (1–18), all other labels use string IDs. Without
ID groups, `neo4j-admin import` treats IDs as globally unique, causing
collisions or mismatched relationships.

Solution: each label gets its own ID namespace:

```csv
# nodes_COCOS.csv
id:ID(COCOS),convention:int,...
1,1,...
```

```csv
# rels_HAS_COCOS.csv
:START_ID(DDVersion),:END_ID(COCOS)
3.39.0,11
```

This maps exactly to neo4j-admin's `--id-type=string` with group syntax.

### Keyset Pagination (not SKIP/LIMIT)

Standard SKIP/LIMIT can produce inconsistent snapshots if nodes are
added/modified during export. Keyset pagination guarantees each node is
exported exactly once:

```cypher
MATCH (n:IMASNode)
WHERE n.id > $last_id
RETURN n.id AS id, n.name AS name, ...
ORDER BY n.id ASC
LIMIT 5000
```

For the DD subgraph (which changes only during DD ingestion, not
continuously), this is extra safety that costs nothing.

### Sequential Export (not Parallel)

Benchmarked 4 threads → only 1.3× speedup. The Neo4j server is the
bottleneck. Sequential export is simpler, deterministic, and easier to debug.
Total export time is ~22 s regardless.

### Post-Import Index Creation (not --schema)

`neo4j-admin import full --schema` fails on Neo4j 2026.01.4 with "Record
format batch import does not support schema changes". The workaround:

1. Import CSV data (no indexes)
2. Start a temp Neo4j instance pointing at the imported data dir
3. Execute `CREATE CONSTRAINT` and `CREATE INDEX` statements
4. Wait for all indexes to reach `ONLINE` state
5. Stop the temp instance
6. Dump the database

This adds ~30 s but is reliable and uses the same temp Neo4j lifecycle
management already implemented in `temp_neo4j.py`.

### Separate Module (not extending temp_neo4j.py)

The new pipeline has a fundamentally different approach (build vs carve) with
different failure modes, dependencies, and lifecycle. It should live in a
new module alongside `temp_neo4j.py` rather than extending it:

- `imas_codex/graph/export_rebuild.py` — the new pipeline
- `imas_codex/graph/temp_neo4j.py` — retained for backward compatibility
  until the new pipeline is proven

### APOC Evaluation

APOC provides `apoc.export.csv.*` procedures that could simplify the export
phase. However:
- APOC is **not installed** in the production Apptainer image
- Adding APOC requires rebuilding the image, testing compatibility, and
  managing plugin versions across Neo4j upgrades
- The native Cypher export (22 s total) is already fast enough
- APOC adds a runtime dependency for a build-time operation

**Recommendation: do not use APOC.** The Cypher+Python CSV writer is simpler,
faster to develop, and has zero additional dependencies.

### Pipe/stdin Import

`neo4j-admin database import full` does **not** support reading from stdin
or named pipes. All input must be regular files on disk. This is a non-issue
since the total CSV volume is ~200 MB and the export phase writes directly to
the temp directory used by the import phase.

### SLURM Execution

The existing SLURM dispatch pattern in `temp_neo4j.py` (`_run_filter_via_slurm`,
`_should_use_slurm`) should be reused. The new pipeline's resource requirements
are actually **lower** than the current approach:

| Resource | Current | New Pipeline |
|---|---|---|
| Memory | 8–16 GB (Neo4j + Cypher DELETE) | 4 GB (neo4j-admin import) |
| Disk | ~3 GB (full dump + temp store) | ~300 MB (CSV + temp store) |
| Time | 10–20 min | ~2 min |

A 4 GB / 30 min SLURM allocation is conservative and sufficient.

---

## DD Index Inventory

Indexes that must be recreated in the filtered dump, captured from production:

### Constraints (10)

| Label | Properties | Type |
|-------|-----------|------|
| COCOS | id | UNIQUENESS |
| DDVersion | id | UNIQUENESS |
| IDS | id | UNIQUENESS |
| IMASCoordinateSpec | id | UNIQUENESS |
| IMASNode | id | UNIQUENESS |
| IMASNodeChange | id | UNIQUENESS |
| IMASSemanticCluster | id | UNIQUENESS |
| IdentifierSchema | id | UNIQUENESS |
| SignConvention | id, facility_id | UNIQUENESS |
| Unit | id | UNIQUENESS |

### Range Indexes (13 non-constraint)

| Label | Properties |
|-------|-----------|
| DDVersion | status |
| IDS | name |
| IMASNode | node_category |
| IMASNode | node_category, ids |
| IMASNode | is_leaf |
| IMASNode | path_lower |
| IMASNode | ids |
| IMASNode | status |
| IMASNode | url |
| SignConvention | facility_id |
| SignConvention | id |
| Unit | symbol |

### Vector Indexes (6)

| Label | Property | Dimensions | Similarity | Quantization |
|-------|----------|-----------|------------|--------------|
| IDS | embedding | 256 | COSINE | true |
| IMASNode | embedding | 256 | COSINE | true |
| IMASSemanticCluster | embedding | 256 | COSINE | true |
| IMASSemanticCluster | label_embedding | 256 | COSINE | true |
| IMASSemanticCluster | description_embedding | 256 | COSINE | true |
| IdentifierSchema | embedding | 256 | COSINE | true |

### Fulltext Indexes (1)

| Name | Label | Properties | Analyzer |
|------|-------|-----------|----------|
| imas_node_text | IMASNode | documentation, name, id, description, keywords | standard-no-stop-words |

**Total: 30 indexes** (10 constraint-backed + 13 range + 6 vector + 1 fulltext)

---

## Implementation Plan

### Phase 1: Core Export Module

**File: `imas_codex/graph/export_rebuild.py`**

```python
"""Export + rebuild pipeline for creating filtered graph dumps.

Queries the live production graph via Cypher, exports nodes and relationships
to CSV, builds a fresh database with neo4j-admin import, creates indexes,
and produces a compact dump file.
"""
```

Key components:

1. **`ExportConfig` dataclass** — holds label lists, relationship specs,
   facility filter, temp directory, batch size (5000), GraphClient reference

2. **`export_nodes_csv(config, label, property_keys, output_dir)`** —
   Exports all nodes of a given label to CSV using keyset pagination.
   Handles embedding serialization (`";".join(f"{v:.8g}" for v in vec)`).
   Returns row count and file path.

3. **`export_relationships_csv(config, rel_type, start_label, end_label, output_dir)`** —
   Exports relationships to CSV. Splits types with multiple target labels
   (HAS_COORDINATE) into separate files. Includes property columns where
   applicable.

4. **`capture_index_ddl(config, labels)`** — Queries `SHOW INDEXES` and
   `SHOW CONSTRAINTS` for matching labels and returns a list of Cypher
   CREATE statements for post-import replay.

5. **`run_import(csv_dir, data_dir, neo4j_image)`** — Assembles the
   `neo4j-admin database import full` command with proper `--nodes`,
   `--relationships`, `--id-type=string`, `--vector-delimiter=;` flags.
   Runs via `subprocess.run()` (or `srun` if on SLURM).

6. **`create_indexes_post_import(data_dir, ddl_statements, neo4j_image)`** —
   Starts a temp Neo4j pointed at the imported data dir, executes CREATE
   CONSTRAINT/INDEX statements, polls `SHOW INDEXES` until all are ONLINE,
   then stops.

7. **`run_dump(data_dir, output_path, neo4j_image)`** — Executes
   `neo4j-admin database dump` on the built database.

8. **`export_rebuild_dd_only(output_path)`** — Top-level orchestrator for
   the DD-only variant. Calls phases 1–4 in sequence.

9. **`export_rebuild_facility(facility_id, output_path)`** — Top-level
   orchestrator for per-facility variant. Exports DD nodes + facility-specific
   nodes and their inter-relationships.

### Phase 2: DD Label and Relationship Registry

Extract the DD subgraph specification from hardcoded constants into a
queryable registry:

```python
# Relationship types internal to DD subgraph
DD_RELATIONSHIPS = [
    RelSpec("IN_VERSION", "IMASNodeChange", "DDVersion"),
    RelSpec("FOR_IMAS_PATH", "IMASNodeChange", "IMASNode"),
    RelSpec("INTRODUCED_IN", "IMASNode", "DDVersion"),
    RelSpec("INTRODUCED_IN", "IDS", "DDVersion"),
    RelSpec("IN_IDS", "IMASNode", "IDS"),
    RelSpec("HAS_PARENT", "IMASNode", "IMASNode"),
    RelSpec("IN_CLUSTER", "IMASNode", "IMASSemanticCluster"),
    RelSpec("HAS_ERROR", "IMASNode", "IMASNode", props=["error_type"]),
    RelSpec("HAS_UNIT", "IMASNode", "Unit"),
    RelSpec("HAS_COORDINATE", "IMASNode", "IMASCoordinateSpec", props=["dimension"]),
    RelSpec("HAS_COORDINATE", "IMASNode", "IMASNode", props=["dimension"]),
    RelSpec("DEPRECATED_IN", "IMASNode", "DDVersion"),
    RelSpec("COORDINATE_SAME_AS", "IMASNode", "IMASNode", props=["dimension"]),
    RelSpec("RENAMED_TO", "IMASNode", "IMASNode"),
    RelSpec("HAS_IDENTIFIER_SCHEMA", "IMASNode", "IdentifierSchema"),
    RelSpec("HAS_PREDECESSOR", "DDVersion", "DDVersion"),
    RelSpec("HAS_SUCCESSOR", "DDVersion", "DDVersion"),
    RelSpec("HAS_COCOS", "DDVersion", "COCOS"),
]
```

Important: `HAS_COORDINATE` has **two target label types** (IMASCoordinateSpec
and IMASNode), which requires two separate relationship CSV files since
`neo4j-admin import` uses `:START_ID(Group)` and `:END_ID(Group)` syntax
that is per-file.

### Phase 3: Facility Filter Variant

The per-facility variant exports:
1. All DD nodes and relationships (same as dd-only)
2. All nodes with `facility_id = $facility` for the specified facility
3. Facility-independent nodes: `Facility`, `GraphMeta`, `DiscoveryRoot`, etc.
4. All relationships **between exported nodes** (closed set)

The relationship closure is key: we cannot blindly export all relationships
from facility nodes, because some may reference nodes in other facilities.
Strategy:

```python
# Phase A: Export all DD nodes → node_id set
# Phase B: Export facility nodes → extend node_id set
# Phase C: For each relationship type, export only where BOTH endpoints
#          are in the node_id set
```

For the export query:
```cypher
MATCH (a)-[r:MAPS_TO]->(b)
WHERE a.facility_id = $facility OR a.facility_id IS NULL
  AND b.facility_id = $facility OR b.facility_id IS NULL
RETURN ...
```

### Phase 4: Integration with Release Workflow

Replace the call path in `release.py`:

```python
# Current (in _push_graph_variant):
#   temp_neo4j.create_filtered_dump(source_dump, output, filter_type, ...)

# New:
#   export_rebuild.export_rebuild_dd_only(output_path)
#   export_rebuild.export_rebuild_facility(facility, output_path)
```

The new functions read from the **live graph** and don't need a source dump
parameter. This also eliminates the need to stop production Neo4j at all —
the release workflow can create filtered variants without any downtime.

### Phase 5: SLURM Integration

Reuse the existing SLURM dispatch pattern:

```python
def _should_use_slurm() -> bool:
    """Check if running on a SLURM-managed cluster."""
    # Same logic as temp_neo4j._should_use_slurm()
    ...

def export_rebuild_via_slurm(variant, output_path, **kwargs):
    """Dispatch export-rebuild to a SLURM compute node."""
    # srun --mem=4G --time=00:30:00 --partition=rigel
    # python -m imas_codex.graph.export_rebuild --variant=dd-only --output=...
    ...
```

Resource request: `--mem=4G --time=00:30:00` (conservative for a 2-min job).

### Phase 6: Verification and Testing

1. **Count verification** — after import, start the temp Neo4j and compare
   node/relationship counts against the export CSVs:
   ```cypher
   MATCH (n:IMASNode) RETURN count(n)  -- should equal CSV row count
   ```

2. **Spot-check queries** — run a few known queries (e.g., "find paths in
   equilibrium IDS") against the rebuilt database and verify results match
   production.

3. **Integration test** — add a test that:
   - Exports a small subset (e.g., 1 IDS worth of nodes)
   - Imports into a temp Neo4j
   - Verifies counts and a semantic query

4. **Dump size regression** — assert the DD-only dump is < 100 MB (currently
   projected at ~25 MB, vs ~900 MB with the old approach).

---

## CSV File Format

### Node CSV Headers

Each label gets one CSV file: `nodes_{Label}.csv`

```csv
# nodes_IMASNode.csv
id:ID(IMASNode),name:string,ids:string,path_lower:string,description:string,documentation:string,keywords:string[],data_type:string,units:string,url:string,status:string,is_leaf:boolean,node_category:string,dd_version:string,lifecycle_status:string,cocos_label_transformation:string,cocos_replace:string,node_type:string,embedding:float[]
equilibrium/time_slice/profiles_1d/psi,psi,equilibrium,equilibrium/time_slice/profiles_1d/psi,"Poloidal flux","Full docs...",...,true,data,3.41.0,active,psi_like,psi_like,dynamic,-0.123;0.456;...
```

For the `embedding:float[]` column, the vector delimiter is `;`:
`-0.12345678;0.98765432;...` (256 values, 8 significant digits each).

String arrays (e.g., `keywords:string[]`) also use `;` as the array delimiter
(same flag: `--array-delimiter=;`). This works because the column type
annotation (`float[]` vs `string[]`) tells the importer which delimiter
semantics to apply.

### Relationship CSV Headers

Each (type, start_label, end_label) triple gets one CSV file:
`rels_{TYPE}_{StartLabel}_{EndLabel}.csv`

```csv
# rels_HAS_COORDINATE_IMASNode_IMASCoordinateSpec.csv
:START_ID(IMASNode),:END_ID(IMASCoordinateSpec),dimension:int

# rels_HAS_COORDINATE_IMASNode_IMASNode.csv
:START_ID(IMASNode),:END_ID(IMASNode),dimension:int

# rels_IN_VERSION_IMASNodeChange_DDVersion.csv (no properties)
:START_ID(IMASNodeChange),:END_ID(DDVersion)
```

### GraphMeta Node

The `GraphMeta` singleton node (`id: "meta"`) must be included in every
variant. It carries `name`, `facilities`, and `updated_at` properties. For
DD-only dumps, the `facilities` list should be set to `[]` (or omitted)
since no facility data is present.

---

## Edge Cases and Gotchas

### COCOS Integer IDs

COCOS nodes use integer IDs (1–18) in Neo4j, while the schema defines
`id: string`. During export, stringify them: `str(node["id"])`. The ID group
`ID(COCOS)` prevents collision with other labels' IDs.

### Empty Labels

PhysicsDomain, SignConvention (DD-only), CoordinateRelationship, and
ClusterMembership currently have 0 nodes. The export should handle empty
labels gracefully (skip CSV generation, omit from import command).

### SignConvention (Facility-Dependent)

SignConvention uses a composite key `(id, facility_id)`. In DD-only dumps,
no SignConvention nodes exist (they're facility-specific). In per-facility
dumps, only SignConvention nodes matching the facility are included.

### HAS_COORDINATE Dual Targets

`HAS_COORDINATE` relationships connect to **both** `IMASCoordinateSpec` and
`IMASNode`. Since `neo4j-admin import` requires consistent `:END_ID(Group)`
per file, these must be exported as two separate CSV files (one per target
label). The export query uses explicit label filtering:

```cypher
MATCH (a:IMASNode)-[r:HAS_COORDINATE]->(b:IMASCoordinateSpec)
RETURN a.id, b.id, r.dimension
```

### INTRODUCED_IN / DEPRECATED_IN Multi-Source

`INTRODUCED_IN` has both `IMASNode` and `IDS` as source labels. Similarly
requires separate CSV files per (source_label, target_label) combination.

### Connection Resilience

The Neo4j bolt connection can drop during long exports (observed during
benchmarking). The export code should:
- Use the existing `GraphClient` with retry logic
- Checkpoint progress (last exported ID) to allow resume
- Set a per-batch timeout (30 s) to detect stale connections

### Embedding Precision

256-dim embeddings are 32-bit floats. CSV serialization uses `f"{v:.8g}"`
(8 significant digits) which preserves full float32 precision. The
`--vector-delimiter=;` flag tells `neo4j-admin` to parse these correctly.

---

## Migration Path

### Step 1: Implement and Validate (this plan)

Build `export_rebuild.py` alongside the existing `temp_neo4j.py`. Both
approaches remain available.

### Step 2: A/B Comparison

During the next release, run both pipelines and compare:
- Dump file sizes
- Query results on sample queries
- Node/relationship counts

### Step 3: Replace Default

Once validated, update `release.py` to use `export_rebuild` by default. Keep
`temp_neo4j.py` as a fallback for one release cycle.

### Step 4: Remove Old Code

After one successful release cycle, remove the filtering code from
`temp_neo4j.py` (the temp Neo4j lifecycle management remains useful for other
purposes).

---

## File Layout

```
imas_codex/graph/
├── export_rebuild.py      # NEW: export + rebuild pipeline
├── temp_neo4j.py          # EXISTING: retained, eventually simplified
├── neo4j_ops.py           # EXISTING: dump/load operations (reused)
├── client.py              # EXISTING: GraphClient (used for export queries)
└── schema_context_data.py # EXISTING: index definitions (reference, not imported)
```

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Neo4j connection drops during export | Medium | Low | Keyset pagination + retry via GraphClient; each batch is independent |
| neo4j-admin import format changes in future Neo4j versions | Low | High | Pin to CSV format; test in CI against the Apptainer image |
| Missing relationship type in DD_RELATIONSHIPS registry | Low | Medium | Verify counts post-import match a control query on production |
| SLURM job preemption during pipeline | Low | Low | Pipeline completes in ~2 min; well within time limits |
| Concurrent DD ingestion during export | Low | Medium | Keyset pagination guarantees consistency within each label; run exports during quiet periods or add advisory lock |
