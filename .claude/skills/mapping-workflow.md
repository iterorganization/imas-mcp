# IMAS Mapping Workflow

How agents discover, propose, validate, and persist IMAS mappings.

## Lifecycle

IMASMapping nodes carry the full status lifecycle — no separate proposal node.

```
FacilitySignal/TreeNode → IMASMapping(proposed) → MappingEvidence → IMASMapping(validated)
```

### 1. Signal Selection
Query checked FacilitySignals for the target physics domain:
```cypher
MATCH (s:FacilitySignal {facility_id: $facility, status: 'checked'})
WHERE s.physics_domain = $domain
RETURN s.id, s.name, s.accessor, s.units, s.description
ORDER BY s.name
```

### 2. Candidate Discovery
Use semantic search on IMAS path descriptions to find candidates:
```python
semantic_search("plasma current measurement", index="imas_path_embedding", k=10)
```

Cross-reference with structured DD lookup:
```python
search_imas("equilibrium/time_slice/global_quantities/ip")
```

### 3. Mapping Creation
Create IMASMapping with status=proposed:
```cypher
MERGE (m:IMASMapping {id: $id})
SET m.status = 'proposed',
    m.facility_id = $facility,
    m.source_path = $signal_id,
    m.target_path = $imas_path,
    m.driver = $driver,
    m.confidence = $initial_confidence,
    m.proposed_at = datetime()
WITH m
MATCH (s:FacilitySignal {id: $signal_id})
MATCH (t:IMASPath {id: $imas_path})
MERGE (m)-[:MAPS_TO_SOURCE]->(s)
MERGE (m)-[:MAPS_TO_TARGET]->(t)
```

### 4. Evidence Collection
Gather evidence from multiple sources:

| Evidence Type | Source | Query |
|---------------|--------|-------|
| wiki_documentation | WikiChunk search | `semantic_search("signal mapping", "wiki_chunk_embedding", 5)` |
| code_reference | CodeChunk search | `semantic_search("read plasma current", "code_chunk_embedding", 5)` |
| data_validation | SSH to facility | Python script testing data access |
| unit_analysis | IMAS DD + signal | Compare units_in vs units_out |
| expert_knowledge | Agent reasoning | Domain expertise on physics equivalence |

Each evidence item becomes a MappingEvidence node linked to the mapping.

### 5. Validation Testing
Write Python scripts to test the mapping against real data:
```python
# SSH to facility, read the signal
data = tree.tdiExecute('tcv_eq("I_P")').data()
time = tree.tdiExecute('dim_of(tcv_eq("I_P"))').data()

# Check: units match? shape reasonable? sign correct?
print(f"Shape: {data.shape}, Units: A, Range: [{data.min():.1f}, {data.max():.1f}]")
```

### 6. Status Progression
- **proposed**: Initial mapping from LLM with evidence
- **endorsed**: Multiple evidence sources agree, tests pass
- **contested**: Conflicting evidence or failed tests
- **validated**: Human or lead agent approved
- **rejected**: Incorrect mapping (persisted with rejection reason)

### 7. Finalization
Update mapping status when validated:
```cypher
MATCH (m:IMASMapping {id: $id})
SET m.status = 'validated',
    m.validated = true,
    m.validated_at = datetime(),
    m.validated_shot = $shot
```

## Key Checks

- **Units**: Must be pint-compatible. Use `pint.UnitRegistry()` to verify conversion.
- **Sign convention**: Check COCOS. TCV uses COCOS 17 for equilibrium.
- **Array shape**: IMAS expects specific dimension ordering (time last in many cases).
- **NaN handling**: Check for NaN/inf in source data before mapping.
