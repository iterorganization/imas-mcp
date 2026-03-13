# JET Ingestion Pipeline — Remaining Completion Items

> **Status**: Partial  
> **Priority**: Low — incremental improvements to fully-operational pipelines  
> **Consolidated from**: signal-scanner-diagnostics (Phase 4), jet-magnetics-quality-remediation (all done except this carry-forward), jet-machine-description-completion (Phases 1–2)

## Context

Three earlier plans drove the JET ingestion pipeline to operational maturity:

- **Signal scanner diagnostics** — scanner progress streaming, worker health
  indicators, and MCP log tools are all implemented. Only structured log field
  consistency remains.
- **JET magnetics quality** — all six root causes (RC-1 through RC-6) are
  resolved: `node_description` injection, section sub-grouping,
  individualized propagation via `SignalSourceCodeUnwind`, `SECTION_METADATA`
  in group headers, `is_static` flag, and `context_quality` persistence.
- **JET machine description** — the device_xml scanner handles 7 data sources,
  all limiter contour versions, and cross-reference relationships. PF circuit
  JPF addressing (Phase 3) is done. Two incremental enrichments remain.

This plan captures the residual items. None block core pipeline operation or
downstream IMAS mapping — they add provenance depth and operational polish.

## Completed Work (from source plans)

For reference, these items are done and do not appear below:

| Source Plan | Item | Evidence |
|-------------|------|----------|
| Scanner diagnostics Phase 1 | Per-scanner progress streaming | `ScannerProgress` in `progress.py` |
| Scanner diagnostics Phase 2 | Worker health indicators | `error_rate_1m`, `consecutive_errors` in `progress.py` |
| Scanner diagnostics Phase 3 | MCP log tools | `list_logs`, `get_logs`, `tail_logs` in `llm/server.py` |
| Magnetics RC-1 | `node_description` in enrichment | `fetch_tree_context()` returns `node_description` |
| Magnetics RC-2 | device_xml section sub-grouping | `_signal_context_key()` groups by `device_xml:{section}` |
| Magnetics RC-3 | Individualized propagation | `individualize_source_descriptions()` + `SignalSourceCodeUnwind` + `source_unwind.md` |
| Magnetics RC-4 | `SECTION_METADATA` in group headers | Used in enrichment prompt construction |
| Magnetics RC-5 | `is_static` flag | Schema, scanner (8 call sites), enrichment routing |
| Magnetics RC-6 | `context_quality` persistence | Persisted in `mark_signals_enriched()` and `propagate_source_enrichment()` |
| Machine desc Phase 3 | PF circuit JPF addresses | `jpf_address` extracted in `device_xml.py` |

---

## Phase 1: Calibration Epoch Graph Nodes

**Priority**: Low | **Effort**: Small | **Source**: machine-description Phase 2

### Justification

The MCFG.ix parser (`parse_mcfg_sensors.py`) already parses 77 calibration
changes from 2006–2019 (gain corrections, probe replacements, TF compensation
updates). This data is currently **discarded** — only the aggregate count
(`calibration_epoch_count`, `first_calibration_shot`, `last_calibration_shot`)
is persisted as properties on the `DataSource` node.

Persisting individual epochs as graph nodes enables shot-to-epoch lookups
("which calibration was active for shot 87000?") and provides provenance for
data quality assessment. This is the lowest-hanging fruit: the parser exists,
the data is structured, only persistence is missing.

### Implementation

1. **Schema**: Add `CalibrationEpoch` class to `facility.yaml` with properties:
   `date`, `first_shot`, `config_id`, `config_type` (MCFG/MCAL/MOFF/MTFC),
   `user`, `description`, `facility_id`

2. **Relationship**: `(DataSource)-[:HAS_CALIBRATION_EPOCH]->(CalibrationEpoch)`

3. **Scanner**: In `device_xml.py`, forward parsed epoch data from
   `parse_mcfg_sensors.py` to `create_nodes()` instead of discarding individual
   entries

4. **Query support**: Enable Cypher lookups:
   ```cypher
   MATCH (e:CalibrationEpoch)
   WHERE e.first_shot <= $shot
   RETURN e ORDER BY e.first_shot DESC LIMIT 1
   ```

### Files
- `imas_codex/schemas/facility.yaml` — add `CalibrationEpoch` class
- `imas_codex/discovery/signals/scanners/device_xml.py` — persist epochs
- `imas_codex/remote/scripts/parse_mcfg_sensors.py` — already parses epochs (no change needed)

---

## Phase 2: Historical Sensor File Versions

**Priority**: Low | **Effort**: Small-Medium | **Source**: machine-description Phase 1

### Justification

The MCFG scanner ingests the 2019 production sensor file
(`sensors_200c_2019-03-11.txt`, 238+ sensors). Three earlier versions exist on
the JET filesystem (2005, two from 2015). Ingesting all versions enables
tracking of sensor position refinements over 14 years and cross-validation of
EFIT reconstructions against the correct sensor geometry for a given shot range.

This depends loosely on Phase 1 — calibration epochs reference specific sensor
file versions. With both phases done, the graph captures which sensor positions
were active for which calibration configuration.

### Implementation

1. **Config**: Add historical file paths to `jet.yaml` under
   `data_systems.mdsplus.static_sources` or similar:
   ```yaml
   sensor_versions:
     - file: sensors-200c-12-05-04.txt
       path: /home/chain1/input/magn90/
       date: 2005-10-12
     - file: sensors_200c_15-09-04_3.txt
       path: /home/MAGNW/chain1/input/PPFcfg/
       date: 2015-09-04
     - file: sensors_200c_15-10-05.txt
       path: /home/MAGNW/chain1/input/PPFcfg/
       date: 2015-10-05
     - file: sensors_200c_2019-03-11.txt
       path: /home/MAGNW/chain1/input/PPFcfg/
       date: 2019-03-11
   ```

2. **Parser**: Extend `parse_mcfg_sensors.py` to accept versioned file list and
   tag each sensor node with its file version date

3. **Graph**: Create `SUPERSEDES` relationships between sensor versions
   (2019 supersedes 2015 supersedes 2005)

4. **Cross-reference**: Link `CalibrationEpoch` to the sensor file version
   it references (when parseable from MCFG.ix description field)

### Files
- `imas_codex/config/facilities/jet.yaml` — add versioned sensor file list
- `imas_codex/remote/scripts/parse_mcfg_sensors.py` — multi-file + versioning
- `imas_codex/discovery/signals/scanners/device_xml.py` — forward version info

---

## Phase 3: Structured Ingestion Logging

**Priority**: Low | **Effort**: Small | **Source**: scanner-diagnostics Phase 4

### Justification

The ingestion pipeline logs to `~/.local/share/imas-codex/logs/` with rotating
file handlers. Periodic graph state snapshots are already implemented via
`make_snapshot_logger()` in `supervision.py`. The MCP `get_logs` tool supports
level filtering and grep.

What's missing is structured log fields on worker log entries — `worker_name`,
`batch_id`, `signal_id`, and error classification are available in context but
logged via plain string interpolation. Adding structured fields would make log
filtering by the MCP `get_logs` tool more reliable and enable automated
anomaly detection on log streams.

This is the lowest-priority item — the pipeline is fully observable without it.
It improves machine-readability of logs for tooling, not human operators.

### Implementation

1. **Log format**: Add structured fields to worker log records via Python
   `logging` extras or a lightweight wrapper:
   ```python
   logger.info(
       "checked %d signals (%d success, %d failed)",
       total, ok, failed,
       extra={"worker": worker_name, "batch": batch_id}
   )
   ```

2. **Formatter**: Update the file handler formatter to include extras when
   present:
   ```
   2026-03-13 10:15:23 INFO check_worker_2 [batch=abc123]: checked 20 signals
   ```

3. **Consistent error logging**: Ensure all worker error paths include
   `worker_name`, `signal_id`, and error classification (infrastructure vs
   application) in structured fields

### Files
- `imas_codex/discovery/base/supervision.py` — structured log wrapper
- `imas_codex/discovery/signals/parallel.py` — worker log call sites
- `imas_codex/cli/logging.py` — formatter update for extras
