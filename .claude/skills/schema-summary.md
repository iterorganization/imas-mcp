# Graph Schema Summary

Condensed schema for the imas-codex Neo4j knowledge graph.

**Before writing Cypher queries**, verify property names against this reference.
For the full schema (53 node types), see `agents/schema-reference.md` or call `get_graph_schema()`.

## Core Node Types

| Type | Key Fields | Description |
|------|-----------|-------------|
| Facility | id, name, machine | Root node per facility (tcv, jet, iter, jt-60sa) |
| FacilitySignal | id, physics_domain, accessor, status | Signal with access info |
| DataAccess | id, method_type, data_template, time_template | Code generation template |
| DataNode | path, tree_name, node_type, description | MDSplus tree node |
| TDIFunction | id, name, supported_quantities | TDI accessor function |
| IMASPath | id (DD path), description, units | IMAS Data Dictionary path |
| IMASMapping | id, source_path, target_path, driver, status | Facility→IMAS mapping |
| MappingEvidence | id, evidence_type, content | Evidence supporting a mapping |
| WikiPage | id, url, title, status, score | Wiki documentation page |
| WikiChunk | id, **text**, embedding | Searchable wiki text chunk |
| CodeChunk | id, **text**, embedding | Searchable code chunk |
| Image | id, source_url, description, embedding | Visual resource from wiki/docs |
| AgentSession | id, facility_id, status | Agent team session tracking |

## WikiChunk Properties (complete)

| Property | Type | Notes |
|----------|------|-------|
| id | string | e.g. "tcv:Thomson:chunk_0" |
| wiki_page_id | string | Parent WikiPage ID |
| artifact_id | string | Parent WikiArtifact ID (for doc chunks) |
| facility_id | string | Facility filter |
| chunk_index | integer | Position in document (0-based) |
| section | string | Section heading |
| **text** | string | **The actual text content** |
| embedding | float[] | Vector for semantic search |
| mdsplus_paths_mentioned | string[] | e.g. ["\\ATLAS::IP"] |
| imas_paths_mentioned | string[] | e.g. ["equilibrium/time_slice/profiles_1d/psi"] |
| ppf_paths_mentioned | string[] | JET PPF paths |
| units_mentioned | string[] | e.g. ["eV", "Tesla"] |
| conventions_mentioned | string[] | e.g. ["COCOS 11"] |
| tool_mentions | string[] | e.g. ["tdiExecute", "ppfget"] |

## Canonical Query Patterns

### Wiki semantic search
```cypher
CALL db.index.vector.queryNodes('wiki_chunk_embedding', $k, $embedding)
YIELD node, score
MATCH (p:WikiPage)-[:HAS_CHUNK]->(node)
WHERE p.facility_id = $facility
RETURN p.title AS page_title, p.url AS url, node.text AS text, score
ORDER BY score DESC
```

### Wiki chunks for a path
```cypher
MATCH (wc:WikiChunk)-[:DOCUMENTS]->(t:DataNode)
WHERE t.path CONTAINS $path
MATCH (wp:WikiPage)-[:HAS_CHUNK]->(wc)
RETURN wp.title AS page_title, wc.text AS text
LIMIT 5
```

### Signal semantic search
```cypher
CALL db.index.vector.queryNodes('facility_signal_desc_embedding', $k, $embedding)
YIELD node AS signal, score
MATCH (signal)-[:DATA_ACCESS]->(da:DataAccess)
OPTIONAL MATCH (signal)-[:MAPS_TO_IMAS]->(imas:IMASPath)
RETURN signal.id, signal.description, da.data_template,
       collect(imas.id) AS imas_paths, score
ORDER BY score DESC
```

### Image search
```cypher
CALL db.index.vector.queryNodes('image_desc_embedding', $k, $embedding)
YIELD node, score
RETURN node.description, node.source_url, node.keywords, score
ORDER BY score DESC
```

## Vector Indexes

| Index | Node | Search By |
|-------|------|-----------|
| wiki_chunk_embedding | WikiChunk | Wiki text content |
| code_chunk_embedding | CodeChunk | Code content |
| image_desc_embedding | Image | Image description/caption |
| facility_signal_desc_embedding | FacilitySignal | Signal description |
| facility_path_desc_embedding | FacilityPath | Path description |
| imas_path_embedding | IMASPath | DD path description |
| data_node_desc_embedding | DataNode | Data node description |
| wiki_artifact_desc_embedding | WikiArtifact | Artifact description |
| cluster_embedding | IMASSemanticCluster | Cluster content |

## Key Relationships

| From | Relationship | To |
|------|-------------|-----|
| FacilitySignal | DATA_ACCESS | DataAccess |
| FacilitySignal | MAPS_TO_IMAS | IMASPath |
| FacilitySignal | AT_FACILITY | Facility |
| DataNode | AT_FACILITY | Facility |
| IMASMapping | AT_FACILITY | Facility |
| IMASMapping | MAPS_TO_SOURCE | DataNode / FacilitySignal |
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
