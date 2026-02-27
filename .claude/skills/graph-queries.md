# Graph Query Patterns

Common Cypher patterns for the imas-codex Neo4j knowledge graph.

## Node Counts
```cypher
MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC
```

## Facility Signals by Domain
```cypher
MATCH (s:FacilitySignal {facility_id: $facility})
WHERE s.physics_domain = $domain AND s.status = 'checked'
OPTIONAL MATCH (s)-[:DATA_ACCESS]->(da:DataAccess)
RETURN s.id, s.name, s.accessor, da.data_template
ORDER BY s.name
```

## Semantic Search + Link Traversal
```cypher
CALL db.index.vector.queryNodes($index, $k, $embedding)
YIELD node AS signal, score
MATCH (signal)-[:DATA_ACCESS]->(da:DataAccess)
OPTIONAL MATCH (signal)-[:MAPS_TO_IMAS]->(imas:IMASPath)
RETURN signal.id, signal.description, da.data_template,
       collect(imas.id) AS imas_paths, score
ORDER BY score DESC
```

## IMAS Mappings
```cypher
-- Create mapping (status lifecycle: proposed → endorsed → validated | rejected)
MERGE (m:IMASMapping {id: $id})
SET m += $props, m.status = 'proposed', m.proposed_at = datetime()
WITH m
MATCH (s:FacilitySignal {id: $signal_id})
MATCH (t:IMASPath {id: $imas_path_id})
MERGE (m)-[:MAPS_TO_SOURCE]->(s)
MERGE (m)-[:MAPS_TO_TARGET]->(t)

-- Find mappings needing validation
MATCH (m:IMASMapping {status: 'proposed', facility_id: $facility})
RETURN m.id, m.confidence, m.proposed_at
ORDER BY m.confidence DESC
```

## Batch Writes (UNWIND)
```cypher
UNWIND $items AS item
MERGE (n:Tool {id: item.id})
SET n += item
WITH n
MATCH (f:Facility {id: $facility})
MERGE (n)-[:AT_FACILITY]->(f)
```

## Vector Indexes

See [agents/schema-reference.md](../../agents/schema-reference.md) for the full list of vector indexes derived from the LinkML schema.
