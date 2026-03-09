# Signal Discovery Pipeline

> Discover, enrich, and validate facility-specific data signals from
> MDSplus trees, TDI functions, PPF databases, and EDAS catalogs.

## CLI Reference

### `imas-codex discover signals <facility>`

Main entry point for signal discovery, enrichment, and validation.

```bash
# Full pipeline — scan, enrich, and check
imas-codex discover signals tcv

# Scan only — discover signals without LLM enrichment
imas-codex discover signals tcv --scan-only

# Enrich only — classify already-discovered signals
imas-codex discover signals tcv --enrich-only

# Specific scanners with cost cap
imas-codex discover signals tcv -s tdi,mdsplus -c 2.0

# JET PPF signals with signal limit
imas-codex discover signals jet -s ppf -n 200

# JT-60SA EDAS signals
imas-codex discover signals jt-60sa -s edas --scan-only

# Focus on equilibrium signals
imas-codex discover signals tcv -f equilibrium
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `FACILITY` | | arg | required | Facility ID (e.g., `tcv`, `jet`) |
| `--scanners` | `-s` | str | auto | Comma-separated scanner types |
| `--scan-only` | | flag | `false` | Only scan, skip enrichment |
| `--enrich-only` | | flag | `false` | Only enrich discovered signals |
| `--signal-limit` | `-n` | int | none | Maximum signals to process |
| `--cost-limit` | `-c` | float | `5.0` | Maximum LLM spend in USD |
| `--time` | | int | none | Maximum runtime in minutes |
| `--focus` | `-f` | str | none | Focus on signal patterns |
| `--enrich-workers` | | int | `2` | Parallel enrichment workers |
| `--check-workers` | | int | `4` | Parallel check workers |
| `--reference-shot` | | int | auto | Override reference shot for validation |

### `imas-codex discover clear <facility> -d signals`

Delete all signal discovery data for a facility (FacilitySignal,
DataAccess, StructuralEpoch, Diagnostic nodes and relationships).

### `imas-codex discover status <facility> -d signals`

Show pipeline progress: scanned, enriched, checked counts per scanner
type, accumulated LLM cost, and check success rates.

---

## Pipeline Architecture

### Three-Phase Pipeline

```
SCAN ──► ENRICH ──► CHECK
 │          │          │
 ▼          ▼          ▼
signals   +desc     +validated
(graph)   +domain   (access ok)
          +diag
```

1. **Scan** — Extract signal metadata from data sources via scanner
   plugins. Creates `FacilitySignal` nodes with `status=discovered`.
2. **Enrich** — LLM classifies each signal with physics domain,
   description, diagnostic assignment, and keywords. Status transitions
   to `enriched`.
3. **Check** — Validate data access by executing accessor expressions
   against the facility. Creates `CHECKED_WITH` relationships recording
   success/failure with shape, dtype, and error details. Status
   transitions to `checked`.

### Scanner Plugin System

Each data source type implements the `DataSourceScanner` protocol:

```python
class DataSourceScanner(Protocol):
    scanner_type: str
    def scan(self, facility, ssh_host, config, reference_shot) -> ScanResult: ...
    def check(self, facility, ssh_host, signals, config, reference_shot) -> list[dict]: ...
```

**Registered scanners:**

| Scanner | Type | Facility | Config Key |
|---------|------|----------|------------|
| MDSplus | `mdsplus` | Any | `data_systems.mdsplus` |
| TDI | `tdi` | TCV | `data_systems.tdi` |
| PPF | `ppf` | JET | `data_systems.ppf` |
| EDAS | `edas` | JT-60SA | `data_systems.edas` |
| Wiki | `wiki` | Any (with wiki_sites) | auto-detected |
| IMAS | `imas` | scaffold | `data_systems.imas` |

Scanner auto-detection reads `data_systems` from facility YAML config.
The wiki scanner always runs first when wiki data exists in the graph.

### Worker Coordination

The pipeline runs as supervised async workers within a
`SupervisedWorkerGroup`:

- **1 scan worker** iterates scanner plugins sequentially
- **N enrich workers** (default 2) claim batches of 50 discovered signals
- **N check workers** (default 4) claim batches of 100 enriched signals
- **1 embed worker** embeds signal descriptions for vector search

Workers coordinate via graph-based claims (`claimed_at` timestamps).
Orphaned claims are auto-recovered after 5-minute timeout.

### Enrichment Context Injection

The enrich worker injects **five levels of context** into LLM prompts:

1. **Facility wiki context** — sign conventions, COCOS definitions
   (cached per facility from `WikiChunk` nodes)
2. **Group wiki context** — diagnostic-specific documentation
   (per signal group from `wiki_chunk_embedding` vector search)
3. **Per-signal wiki context** — exact path matches from wiki scanner
4. **Code context** — source code usage patterns from
   `code_chunk_embedding` vector search
5. **TDI source code** — full `.fun` file content from `TDIFunction`
   graph nodes

This context layering significantly reduces LLM hallucination rates.

### Tree Extraction

MDSplus trees are extracted in three modes:

- **Versioned (static)** — trees with `static_trees` config. Multiple
  structural versions compared to build a _super tree_ with
  `INTRODUCED_IN` / `REMOVED_IN` epoch edges.
- **Epoched** — trees with shot-scoped structure changes. Epochs
  detected via binary search on structure fingerprints.
- **Shot-scoped** — single-shot extraction for live trees without
  versioning.

### TDI Linkage

After MDSplus and TDI scanners both complete, TDI linkage resolves
`build_path()` references in TDI functions to their underlying
`DataNode` nodes, creating `RESOLVES_TO_NODE` edges.

---

## Graph Schema

### Node Types

| Node Type | Description |
|-----------|-------------|
| `FacilitySignal` | A facility-specific data signal |
| `DataNode` | A node within an MDSplus tree |
| `StructuralEpoch` | Structural version of a tree model |
| `DataNodePattern` | Repeated structural pattern across indexed tree components |
| `TDIFunction` | TDI function providing high-level access abstraction |
| `DataAccess` | Code template for data retrieval |
| `Diagnostic` | Plasma diagnostic system |
| `DataSource` | An MDSplus tree instance |
| `Unit` | Physical unit (e.g., Tesla, eV) |

### Key Relationships

| From | Relationship | To | Description |
|------|-------------|-----|-------------|
| FacilitySignal | `HAS_DATA_SOURCE_NODE` | DataNode | Backing data location |
| FacilitySignal | `DATA_ACCESS` | DataAccess | How to access the data |
| FacilitySignal | `BELONGS_TO_DIAGNOSTIC` | Diagnostic | Diagnostic system |
| FacilitySignal | `CHECKED_WITH` | DataAccess | Validation result |
| FacilitySignal | `AT_FACILITY` | Facility | Owning facility |
| DataNode | `HAS_NODE` | DataNode | Parent-child hierarchy |
| DataNode | `INTRODUCED_IN` | StructuralEpoch | When node appeared |
| DataNode | `REMOVED_IN` | StructuralEpoch | When node disappeared |
| DataNode | `FOLLOWS_PATTERN` | DataNodePattern | Indexed pattern group |
| DataNode | `HAS_UNIT` | Unit | Physical unit |
| TDIFunction | `AT_FACILITY` | Facility | Owning facility |
| TDIFunction | `RESOLVES_TO_NODE` | DataNode | Underlying data path |
| Diagnostic | `AT_FACILITY` | Facility | Owning facility |
| StructuralEpoch | `HAS_PREDECESSOR` | StructuralEpoch | Version chain |

---

## Example Cypher Queries

### 1. Semantic Search — Find Signals by Physics Meaning

```cypher
CALL db.index.vector.queryNodes(
    'facility_signal_desc_embedding', 10, $embedding
) YIELD node AS signal, score
WHERE signal.facility_id = 'tcv'
RETURN signal.id, signal.accessor, signal.description,
       signal.physics_domain, signal.diagnostic, score
ORDER BY score DESC
```

### 2. Path Access — Navigate from Tree Path to Signal

```cypher
MATCH (tn:DataNode {path: '\\RESULTS::LIUQE:IP', facility_id: 'tcv'})
OPTIONAL MATCH (fs:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(tn)
OPTIONAL MATCH (tn)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (fs)-[:BELONGS_TO_DIAGNOSTIC]->(d:Diagnostic)
OPTIONAL MATCH (fs)-[:DATA_ACCESS]->(da:DataAccess)
OPTIONAL MATCH (tn)-[:INTRODUCED_IN]->(ver:StructuralEpoch)
RETURN tn.path, fs.id, fs.description, fs.physics_domain,
       u.name AS unit, d.name AS diagnostic,
       da.method_type, ver.first_shot
```

### 3. Preferential Accessor Selection

For signals accessible via both raw MDSplus path and TDI function,
query both access methods:

```cypher
MATCH (fs:FacilitySignal {facility_id: 'tcv'})
WHERE fs.tdi_function IS NOT NULL
MATCH (fs)-[:HAS_DATA_SOURCE_NODE]->(tn:DataNode)
MATCH (fs)-[:DATA_ACCESS]->(da:DataAccess)
RETURN fs.id, fs.accessor AS tdi_accessor,
       tn.path AS raw_path, da.method_type,
       fs.physics_domain
ORDER BY fs.accessor
LIMIT 20
```

### 4. Data Access Pattern Resolution

From a `DataAccess` node, find all signals and their access templates:

```cypher
MATCH (da:DataAccess {id: 'tcv:tdi:functions'})
MATCH (fs:FacilitySignal)-[:DATA_ACCESS]->(da)
RETURN da.data_template, da.connection_template,
       collect({
           id: fs.id,
           accessor: fs.accessor,
           domain: fs.physics_domain
       })[..10] AS sample_signals,
       count(fs) AS total_signals
```

### 5. Epoch-Aware Queries

Find signals that exist at a given shot number via `StructuralEpoch`
applicability ranges:

```cypher
MATCH (fs:FacilitySignal {facility_id: 'tcv'})-[:HAS_DATA_SOURCE_NODE]->(tn:DataNode)
MATCH (tn)-[:INTRODUCED_IN]->(v:StructuralEpoch)
WHERE v.first_shot <= $shot
  AND (v.last_shot IS NULL OR v.last_shot >= $shot)
  AND (tn.removed_version IS NULL)
RETURN fs.id, fs.accessor, tn.path, v.version,
       v.first_shot, v.last_shot
ORDER BY tn.path
LIMIT 50
```

### 6. Cross-Domain Traversal

From a signal, traverse to related wiki documentation, code chunks,
and IMAS paths:

```cypher
MATCH (fs:FacilitySignal {id: $signal_id})
// Wiki documentation
OPTIONAL MATCH (fs)-[:BELONGS_TO_DIAGNOSTIC]->(d:Diagnostic)
OPTIONAL MATCH (d)-[:DOCUMENTED_BY]->(wp:WikiPage)
OPTIONAL MATCH (wp)-[:HAS_CHUNK]->(wc:WikiChunk)
// Code usage
OPTIONAL MATCH (tn:DataNode)<-[:HAS_DATA_SOURCE_NODE]-(fs)
// IMAS mapping
OPTIONAL MATCH (tn)<-[:SOURCE_PATH]-(m:IMASMapping)-[:TARGET_PATH]->(imas:IMASPath)
RETURN fs.id, fs.description,
       collect(DISTINCT wp.title) AS wiki_pages,
       collect(DISTINCT wc.text)[..3] AS wiki_excerpts,
       collect(DISTINCT imas.id) AS imas_paths
```

### 7. Diagnostic Inventory

List all diagnostics and their signal counts:

```cypher
MATCH (d:Diagnostic)-[:AT_FACILITY]->(f:Facility {id: 'tcv'})
OPTIONAL MATCH (fs:FacilitySignal)-[:BELONGS_TO_DIAGNOSTIC]->(d)
RETURN d.name, d.category, d.description,
       count(fs) AS signal_count
ORDER BY signal_count DESC
```

### 8. TDI Function Resolution

From a TDI function, follow `RESOLVES_TO_NODE` edges to the
underlying `DataNode` nodes and their signals:

```cypher
MATCH (tdi:TDIFunction {name: 'tcv_eq', facility_id: 'tcv'})
OPTIONAL MATCH (tdi)-[:RESOLVES_TO_NODE]->(tn:DataNode)
OPTIONAL MATCH (fs:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(tn)
RETURN tdi.name, tdi.signature,
       collect(DISTINCT {
           tree_path: tn.path,
           signal: fs.id,
           accessor: fs.accessor,
           domain: fs.physics_domain
       })[..20] AS resolved_signals,
       tdi.quantity_count AS total_quantities
```

---

## Cross-Pipeline Dependencies

Signal enrichment quality improves significantly when wiki and code
discovery have already populated the graph:

```
1. imas dd build          → IMASPath, DDVersion, Unit
2. discover wiki          → WikiPage, WikiChunk (sign conventions, COCOS)
2. discover paths/code    → FacilityPath, SourceFile, CodeChunk
3. discover signals       → FacilitySignal, DataAccess (benefits from above)
```

Run wiki and code discovery before signals for best enrichment context.
