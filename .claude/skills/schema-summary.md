# Graph Schema Summary

Condensed schema for the imas-codex Neo4j knowledge graph.

## Core Node Types

| Type | Key Fields | Description |
|------|-----------|-------------|
| Facility | id, name, machine | Root node per facility (tcv, jet, iter, jt-60sa) |
| FacilitySignal | id, physics_domain, accessor, status | Signal with access info |
| DataAccess | id, method_type, data_template, time_template | Code generation template |
| TreeNode | path, tree_name, node_type, description | MDSplus tree node |
| TDIFunction | id, name, supported_quantities | TDI accessor function |
| IMASPath | id (DD path), description, units | IMAS Data Dictionary path |
| IMASMapping | id, source_path, target_path, driver, status | Facility→IMAS mapping |
| MappingEvidence | id, evidence_type, content | Evidence supporting a mapping |
| WikiPage | id, url, title, status, score | Wiki documentation page |
| WikiChunk | id, content, embedding | Searchable wiki text chunk |
| CodeChunk | id, content, embedding | Searchable code chunk |
| AgentSession | id, facility_id, status | Agent team session tracking |

## Key Relationships

| From | Relationship | To |
|------|-------------|-----|
| FacilitySignal | DATA_ACCESS | DataAccess |
| FacilitySignal | MAPS_TO_IMAS | IMASPath |
| FacilitySignal | AT_FACILITY | Facility |
| TreeNode | AT_FACILITY | Facility |
| IMASMapping | AT_FACILITY | Facility |
| IMASMapping | MAPS_TO_SOURCE | TreeNode / FacilitySignal |
| IMASMapping | MAPS_TO_TARGET | IMASPath |
| IMASMapping | HAS_EVIDENCE | MappingEvidence |
| WikiChunk | HAS_CHUNK ← | WikiPage |
| WikiChunk | NEXT_CHUNK | WikiChunk |
| CodeChunk | HAS_CHUNK ← | CodeExample |

## Status Enums

| Type | States |
|------|--------|
| FacilitySignal | discovered → enriched → checked / skipped / failed |
| WikiPage | scanned → scored → ingested / skipped / failed |
| IMASMapping | proposed → endorsed / contested → validated / rejected |
| AgentSession | running → completed / failed |

## Physics Domains (PhysicsDomain enum)

equilibrium, magnetics, electron_density, electron_temperature,
ion_temperature, radiation, neutron, spectroscopy, mhd,
heating_current_drive, fueling, vacuum, wall, divertor,
current_drive, transport, turbulence, pedestal, core_profiles
