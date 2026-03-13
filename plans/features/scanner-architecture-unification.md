# Scanner Architecture: Static Source Handler Unification

## Context

The discovery pipeline has two divergent scanner patterns:

1. **Standard scanners** (ppf, tdi, edas, imas, wiki): `scan()` returns `ScanResult` with a list of `FacilitySignal` objects. The shared `seed_worker` → `ingest_discovered_signals()` pipeline creates `FacilitySignal` + `AT_FACILITY` + `DATA_ACCESS` + `INTRODUCED_IN` edges.

2. **device_xml scanner** (2,618 lines): `scan()` returns `ScanResult(signals=[])` and persists everything directly via 7 internal handler pairs (`_scan_*` async + `_persist_*` sync), each creating `DataSource`, `SignalEpoch`, `SignalNode`, `FacilitySignal`, and cross-reference relationships via raw Cypher.

### Why device_xml diverged

The standard pattern creates only `FacilitySignal` + 3 edges. The device_xml scanner creates a **much richer** graph structure:

- **DataSource** nodes (one per data source type)
- **SignalEpoch** nodes with epoch-specific properties (probes_enabled, pf_configuration, wall_configuration)
- **SignalNode** geometry nodes with arrays (r, z, angle, r_contour, z_contour)
- **Cross-reference relationships**: `USES_LIMITER`, `MATCHES_SENSOR`, `SAME_GEOMETRY`, `SAME_COMPONENT`, `IN_CIRCUIT`, `USES_GREENS`, `ACCESSES_GEOMETRY`, `IN_DATA_SOURCE`

The standard `ingest_discovered_signals()` function cannot express any of this — it only knows about `FacilitySignal` nodes. The divergence is **inherent to the data model** and not a design mistake.

### The actual problem

The divergence at the **scanner-level** interface is correct and should stay. The problem is at the **internal handler level**: the 7 `_persist_*` functions (1,884 lines total) share ~60% structural boilerplate:

| Handler | Lines | Creates |
|---------|-------|---------|
| `_persist_graph_nodes` | 414 | DataSource, SignalEpoch, SignalNode, FacilitySignal, USES_LIMITER |
| `_persist_jec2020_nodes` | 439 | DataSource, SignalNode, FacilitySignal, MATCHES_SENSOR |
| `_persist_mcfg_nodes` | 213 | DataSource, SignalNode, MATCHES_SENSOR |
| `_persist_magnetics_config_nodes` | 287 | DataSource, SignalEpoch, SignalNode, FacilitySignal |
| `_persist_pf_coil_turns_nodes` | 111 | DataSource, SignalNode, SAME_COMPONENT |
| `_persist_greens_table_nodes` | 118 | DataSource, SignalNode, USES_GREENS |
| `_persist_ppf_static_nodes` | 302 | DataSource, DataAccess, ACCESSES_GEOMETRY |

Similarly, the 6 `_scan_*` methods (483 lines) follow identical patterns:
1. Look up config from `static_sources` by name
2. Build `script_input` dict
3. `run_python_script("parse_*.py", script_input, ssh_host, timeout)`
4. Parse JSON from last line of output
5. Call corresponding `_persist_*` function
6. Log stats and return

### Shared boilerplate in `_persist_*`

Every handler repeats:

```python
# 1. DataSource creation — identical Cypher across all 7 handlers
gc.query("""
    MERGE (ds:DataSource {id: $facility + ':' + $name})
    ON CREATE SET
        ds.name = $name,
        ds.facility_id = $facility,
        ds.source_type = $source_type,
        ds.source_format = $source_format,
        ds.description = $description,
        ds.shot_dependent = false
    WITH ds
    MATCH (f:Facility {id: $facility})
    MERGE (ds)-[:AT_FACILITY]->(f)
""", ...)

# 2. SignalNode bulk creation — same loop pattern, different properties
for item in parsed_items:
    dn = {"path": ..., "data_source_name": ..., "facility_id": ..., ...}
    data_nodes.append(dn)
for dn in data_nodes:
    dn["id"] = dn["path"]
gc.create_nodes("SignalNode", data_nodes, batch_size=N)

# 3. FacilitySignal creation — same dict/dedup pattern, different fields
all_signals[sig_id] = FacilitySignal(id=..., facility_id=..., ...)
gc.create_nodes("FacilitySignal", signal_dicts, batch_size=100)
```

## Proposal

Extract a `StaticSourceHandler` base class that provides the shared scaffold, letting each handler define only what is unique.

### What does NOT change

- The `DataSourceScanner` Protocol and registry in `base.py` — no changes
- The `ScanResult(signals=[])` return pattern for device_xml — correct as-is
- The `seed_worker` → `ingest_discovered_signals()` pipeline for standard scanners
- The `DeviceXMLScanner` class — remains the single registered scanner
- The remote scripts (`parse_*.py`) — untouched

### What changes

1. **New base class** `StaticSourceHandler` in `device_xml.py` (~80 lines):
   - Provides `ensure_data_source()` — shared DataSource creation
   - Provides `persist_signal_nodes()` — batch SignalNode creation with id assignment
   - Provides `persist_facility_signals()` — batch FacilitySignal creation with dedup
   - Provides `persist_epochs()` — batch SignalEpoch creation with standard edges
   - Template method `run()` that calls `lookup_config()` → `run_remote()` → `persist()`
   - Each handler overrides `persist(parsed_data) → stats`

2. **7 handler subclasses** replace the 7 function pairs:
   - `DeviceXMLHandler(StaticSourceHandler)` replaces `_persist_graph_nodes` + main scan block
   - `JEC2020Handler(StaticSourceHandler)` replaces `_scan_jec2020` + `_persist_jec2020_nodes`
   - `MCFGHandler(StaticSourceHandler)` replaces `_scan_mcfg` + `_persist_mcfg_nodes`
   - `MagneticsConfigHandler(StaticSourceHandler)` replaces `_scan_magnetics_config` + `_persist_magnetics_config_nodes`
   - `PFCoilTurnsHandler(StaticSourceHandler)` replaces `_scan_pf_coil_turns` + `_persist_pf_coil_turns_nodes`
   - `GreensTableHandler(StaticSourceHandler)` replaces `_scan_greens_table` + `_persist_greens_table_nodes`
   - `PPFStaticHandler(StaticSourceHandler)` replaces `_scan_ppf_static` + `_persist_ppf_static_nodes`

3. **DeviceXMLScanner.scan()** becomes a simple loop over registered handlers.

### Expected impact

- **Lines eliminated**: ~400-500 lines of repeated DataSource creation, node bulk-create boilerplate, and config-lookup patterns
- **New handler template**: Adding a new static data source requires ~40-60 lines (config name + remote script name + persist method), vs ~250+ lines today
- **No graph schema changes**: Same nodes, same relationships, same Cypher patterns
- **No behavioral changes**: Identical graph output — pure refactor

## Phases

### Phase 1: Extract `StaticSourceHandler` base class

**Scope**: Define the base class with shared methods. No functional changes yet.

```python
class StaticSourceHandler:
    """Base class for static data source handlers within DeviceXMLScanner."""

    source_name: str           # e.g., "jec2020_geometry"
    config_key: str            # Key in static_sources list to match by name
    remote_script: str         # e.g., "parse_jec2020.py"
    timeout: int = 120

    def ensure_data_source(self, gc, facility, source_type, source_format, description):
        """Create/update DataSource node — shared Cypher."""

    def persist_signal_nodes(self, gc, label, nodes, batch_size=100):
        """Batch create nodes with id=path assignment."""

    def persist_facility_signals(self, gc, signals_dict, batch_size=100):
        """Batch create FacilitySignal from dedup dict."""

    def persist_epochs(self, gc, epoch_records):
        """Batch create SignalEpoch with AT_FACILITY + IN_DATA_SOURCE."""

    async def run(self, facility, ssh_host, config) -> dict | None:
        """Template: lookup config → run remote → persist → return stats."""

    def persist(self, gc, facility, source_config, parsed) -> dict:
        """Override in subclass: handler-specific graph persistence."""
```

**Files**: `imas_codex/discovery/signals/scanners/device_xml.py`
**Tests**: Existing tests must pass unchanged.

### Phase 2: Migrate handlers one at a time

Convert each `_scan_*` + `_persist_*` pair to a handler subclass, starting with the simplest:

1. `GreensTableHandler` (118 + 129 = 247 lines → ~80 lines)
2. `PFCoilTurnsHandler` (111 + 71 = 182 lines → ~70 lines)
3. `PPFStaticHandler` (302 + 49 = 351 lines → ~120 lines)
4. `MCFGHandler` (213 + 76 = 289 lines → ~100 lines)
5. `MagneticsConfigHandler` (287 + 76 = 363 lines → ~130 lines)
6. `JEC2020Handler` (439 + 82 = 521 lines → ~200 lines)
7. `DeviceXMLHandler` (414 + main scan block = ~500 lines → ~250 lines)

Each migration is a standalone commit with test validation.

### Phase 3: Handler registry in DeviceXMLScanner

Replace the hardcoded `_scan_*` call sequence in `DeviceXMLScanner.scan()` with a handler list:

```python
class DeviceXMLScanner:
    handlers = [
        DeviceXMLHandler(),
        JEC2020Handler(),
        MCFGHandler(),
        PPFStaticHandler(),
        MagneticsConfigHandler(),
        PFCoilTurnsHandler(),
        GreensTableHandler(),
    ]

    async def scan(self, facility, ssh_host, config, reference_shot=None):
        stats = {}
        for handler in self.handlers:
            result = await handler.run(facility, ssh_host, config)
            if result:
                stats[handler.source_name] = result
        return ScanResult(signals=[], data_access=..., stats=stats)
```

## Verdict

The `signals=[]` pattern is **correct** — the device_xml scanner creates structural graph data (DataSource, SignalEpoch, SignalNode, cross-references) that the standard `ingest_discovered_signals()` pipeline cannot express. Trying to force this into the standard return-signals pattern would require either:
- Massively expanding `ScanResult` to carry graph topology (wrong — scanner results shouldn't encode graph structure)
- Making `ingest_discovered_signals()` understand every possible node type and relationship (wrong — violates single responsibility)

The refactor opportunity is **internal to device_xml.py**: the 7 handler pairs share substantial boilerplate that a base class can eliminate, making it straightforward to add new static data sources.

## Priority

**Medium** — the current code works correctly and is well-documented. This is a developer experience improvement that becomes valuable when adding new static data sources (e.g., additional JET geometry files, or extending the pattern to other facilities).
