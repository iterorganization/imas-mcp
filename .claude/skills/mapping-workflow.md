# IMAS Mapping Workflow

How agent teams discover, propose, validate, and persist IMAS mappings.

## Lifecycle

```
FacilitySignal → MappingProposal → MappingEvidence → validation → IMASMapping
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

### 3. Proposal Creation
Create MappingProposal with initial confidence:
```cypher
MERGE (mp:MappingProposal {id: $id})
SET mp.status = 'proposed',
    mp.facility_id = $facility,
    mp.signal_id = $signal_id,
    mp.imas_path_id = $imas_path,
    mp.confidence = $initial_confidence,
    mp.proposed_by = $session_id,
    mp.proposed_at = datetime()
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

Each evidence item becomes a MappingEvidence node linked to the proposal.

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
- **proposed**: Initial proposal with evidence
- **endorsed**: Multiple evidence sources agree, tests pass
- **contested**: Conflicting evidence or failed tests
- **validated**: Human or lead agent approved
- **rejected**: Incorrect mapping (persisted with rejection reason)

### 7. Persistence
Validated mappings become IMASMapping nodes:
```cypher
MATCH (mp:MappingProposal {id: $id, status: 'validated'})
MATCH (s:FacilitySignal {id: mp.signal_id})
MATCH (t:IMASPath {id: mp.imas_path_id})
MERGE (m:IMASMapping {id: $mapping_id})
SET m.source_path = s.id,
    m.target_path = t.id,
    m.facility_id = mp.facility_id,
    m.driver = $driver,
    m.units_in = $units_in,
    m.units_out = $units_out,
    m.scale = $scale,
    m.confidence = mp.confidence,
    m.validated = true,
    m.validated_shot = $shot
MERGE (m)-[:FROM_PROPOSAL]->(mp)
```

## Key Checks

- **Units**: Must be pint-compatible. Use `pint.UnitRegistry()` to verify conversion.
- **Sign convention**: Check COCOS. TCV uses COCOS 17 for equilibrium.
- **Array shape**: IMAS expects specific dimension ordering (time last in many cases).
- **NaN handling**: Check for NaN/inf in source data before mapping.
