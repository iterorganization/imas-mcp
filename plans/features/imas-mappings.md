# IMAS Mapping Discovery

Status: **Planning**
Priority: High — core deliverable for imas-ambix

## Summary

Discover and persist mappings between facility data (TreeNode/FacilitySignal)
and IMAS Data Dictionary paths (IMASPath). Each mapping connects a facility
signal to its IMAS equivalent with transformation metadata (unit conversion,
coordinate transforms, COCOS).

## Prerequisites

Before mapping discovery can begin, three foundation steps must complete:

1. **DataAccess nodes** — deterministic, generated from facility YAML config
2. **FacilitySignal→TreeNode links** — batch Cypher from existing TreeNodes
3. **MCP catalog tool** — agents need filtered graph views without context flooding

## Graph Schema

Already defined in `facility.yaml`:

```
IMASMapping
├── id: "facility:signal_id→imas_path"
├── facility_signal → FacilitySignal (MAPS_FROM)
├── imas_path → IMASPath (MAPS_TO)
├── mapping_type: direct | transform | composite | unavailable
├── expression: Python transform expression
├── unit_conversion: pint expression (e.g., "kA → A")
├── cocos_source / cocos_target: integer
├── time_interpolation: none | linear | cubic
├── confidence: float 0-1
├── status: proposed | validated | rejected | stale
├── evidence → MappingEvidence[] (HAS_EVIDENCE)
└── created_by_session, validated_by, validation_shot
```

## Agent Teams: Evaluation

The user asked for "a considered evaluation for the need of agent teams."

### What agent teams would do

A team of 3-5 Claude Code agents working in parallel, each:
1. Takes a physics domain (equilibrium, magnetics, edge_plasma, etc.)
2. Queries TreeNodes in that domain via MCP
3. Searches for matching IMASPaths via semantic search
4. Proposes IMASMapping nodes with evidence
5. Validates by checking unit compatibility and COCOS transforms

### Why agent teams are NOT necessary

| Factor | Assessment |
|--------|------------|
| **Scale** | TCV has ~11,596 leaf TreeNodes across ~15 physics domains. A single agent can process 50-100 mappings per session. At 3-4 sessions, coverage is achievable without parallelism. |
| **Determinism** | Most mappings are discoverable via embedding similarity (TreeNode.embedding ↔ IMASPath.embedding). This is a vector search + heuristic pipeline, not a reasoning task requiring agent autonomy. |
| **Coordination overhead** | Agent teams need a proxy server (LiteLLM), Langfuse observability, budget enforcement, conflict resolution for concurrent graph writes. This infrastructure doesn't exist yet. |
| **Quality vs speed** | Mapping quality depends on domain expertise (COCOS, unit conventions, coordinate systems). A focused single agent with good prompts produces better mappings than 5 parallel agents with shallow context. |
| **Cost** | Agent teams would cost ~$15-50 per facility (5 agents × ~$3-10 each). A deterministic pipeline with LLM fallback for ambiguous cases costs ~$0.50-2. |
| **Existing infrastructure** | The discovery pipeline already has the batch worker pattern (scan→enrich→check). Mapping fits this pattern without new infrastructure. |

### Recommendation: Pipeline-first, agent-assist for edge cases

Use agent teams only if the pipeline approach fails to achieve >80% coverage.

## Architecture: Three-Phase Pipeline

### Phase 1: Deterministic Matching (no LLM)

Vector similarity between TreeNode and IMASPath embeddings, filtered by
physics domain alignment.

```python
# Pseudocode for deterministic matching
for domain in physics_domains:
    tree_nodes = query("""
        MATCH (tn:TreeNode {facility_id: $facility, physics_domain: $domain})
        WHERE tn.node_type IN ['SIGNAL', 'NUMERIC']
        AND tn.embedding IS NOT NULL
        RETURN tn
    """)
    
    for tn in tree_nodes:
        # Semantic search against IMASPath embeddings
        candidates = semantic_search(
            tn.description,
            index="imas_path_embedding",
            k=5
        )
        
        # Filter by unit compatibility
        candidates = [c for c in candidates if units_compatible(tn.unit, c.unit)]
        
        if candidates and candidates[0].score > 0.85:
            create_mapping(tn, candidates[0], mapping_type="direct", confidence=score)
```

**Expected yield**: 40-60% of leaf TreeNodes map directly (equilibrium,
magnetics, and wall signals have near-identical naming).

### Phase 2: LLM-Assisted Matching (structured output)

For TreeNodes without a high-confidence deterministic match, use an LLM
with the TreeNode description + top-5 IMASPath candidates:

```python
# Input to LLM
{
    "tree_node": {
        "path": "\\RESULTS::LIUQE:PSI",
        "description": "Poloidal flux from LIUQE equilibrium reconstruction",
        "unit": "Wb",
        "physics_domain": "equilibrium"
    },
    "candidates": [
        {"path": "equilibrium/time_slice/profiles_2d/0/psi", "description": "...", "unit": "Wb"},
        {"path": "equilibrium/time_slice/profiles_1d/psi", "description": "...", "unit": "Wb"},
        ...
    ]
}

# Structured output
class MappingProposal(BaseModel):
    imas_path: str
    mapping_type: Literal["direct", "transform", "composite", "unavailable"]
    expression: str | None  # Transform if needed
    unit_conversion: str | None
    confidence: float
    reasoning: str
```

**Expected yield**: additional 20-30%, covering cases where naming differs
but physics meaning matches.

### Phase 3: Validation

For all proposed mappings with confidence > 0.5:

1. **Unit check** — pint-based: `ureg(tn.unit).dimensionality == ureg(imas.unit).dimensionality`
2. **Shape check** — TreeNode.shape vs IMASPath coordinate specs (1D time series → 1D, 2D profile → 2D)
3. **COCOS check** — if both have sign_convention, verify COCOS compatibility
4. **Access check** — use DataAccess to fetch sample data and verify it's physically reasonable

Validated mappings: `status = "validated"`. Failed: `status = "rejected"` with reason.

## CLI Integration

```bash
# Run mapping pipeline
imas-codex discover mappings tcv
imas-codex discover mappings tcv --domain equilibrium
imas-codex discover mappings tcv --dry-run

# Status
imas-codex discover mappings tcv --status

# Validate existing proposals
imas-codex discover mappings tcv --validate --shot 84000
```

Follows the same pattern as `discover static` and `discover signals`.

## DataAccess Generation (Prerequisite)

Deterministic step added to `discover static` CLI. After tree extraction,
for each tree, create a DataAccess node:

```python
DataAccess(
    id=f"{facility}:mdsplus:{tree_name}",
    facility_id=facility,
    name=f"MDSplus {tree_name} (local)",
    method_type="mdsplus",
    library="MDSplus",
    access_type="local",
    connection_template="tree = MDSplus.Tree('{data_source}', {shot}, 'readonly')",
    data_template="data = tree.getNode('{accessor}').data()",
    time_template="time = tree.tdiExecute('dim_of({accessor})').data()",
    cleanup_template="tree.close()",
    data_source=tree_name,
    setup_commands=facility_config.mdsplus.setup_commands,
    discovery_shot=static_tree.first_shot,
)
```

No SSH, no LLM. Reads facility YAML, writes to graph.

## FacilitySignal Generation (Prerequisite)

Batch Cypher creates FacilitySignal nodes from leaf TreeNodes:

```cypher
MATCH (tn:TreeNode {facility_id: $facility})
WHERE tn.node_type IN ['SIGNAL', 'NUMERIC']
AND NOT EXISTS { MATCH (fs:FacilitySignal)-[:SOURCE_NODE]->(tn) }
WITH tn
MATCH (da:DataAccess {id: $facility + ':mdsplus:' + tn.tree_name})
CREATE (fs:FacilitySignal {
    id: tn.facility_id + ':' + coalesce(tn.physics_domain, 'unknown') + '/' + 
        replace(replace(tn.path, '\\', ''), '::', '/'),
    facility_id: tn.facility_id,
    accessor: tn.path,
    tree_name: tn.tree_name,
    name: tn.description,
    physics_domain: tn.physics_domain,
    unit: tn.unit,
    status: 'discovered'
})
CREATE (fs)-[:SOURCE_NODE]->(tn)
CREATE (fs)-[:DATA_ACCESS]->(da)
CREATE (fs)-[:AT_FACILITY]->(:Facility {id: tn.facility_id})
```

This preserves all existing TreeNode enrichment. FacilitySignal is a thin
access wrapper — all physics metadata stays on TreeNode and is reached via
the `SOURCE_NODE` edge.

## Migration Safety

| Existing Data | Impact | Action |
|---------------|--------|--------|
| 47,976 TreeNodes | Zero modification | Untouched — FacilitySignal links TO them |
| 34,082 embeddings | Preserved | Used for similarity search in Phase 1 |
| 11,596 unit values | Preserved | Inherited by FacilitySignal via SOURCE_NODE |
| enrichment_* fields | Preserved | All stay on TreeNode |
| TreeModelVersion epochs | Preserved | INTRODUCED_IN/REMOVED_IN edges unchanged |
| TreeNodePattern | Preserved | FOLLOWS_PATTERN edges unchanged |

## Cost Estimate

| Phase | Method | Est. Cost | Nodes |
|-------|--------|-----------|-------|
| DataAccess generation | Deterministic | $0 | ~10 nodes |
| FacilitySignal generation | Batch Cypher | $0 | ~11,596 |
| Phase 1: Deterministic matching | Vector search | $0 | ~5,000-7,000 mapped |
| Phase 2: LLM-assisted matching | Structured output | ~$1-2 | ~2,000-3,500 |
| Phase 3: Validation | Pint + COCOS | $0 | All proposed |
| **Total** | | **~$1-2** | |

## When to Escalate to Agent Teams

If the pipeline achieves <60% mapping coverage after Phase 2, consider:

1. **Single focused agent session** — an MCP-equipped agent spends 1-2 hours
   on unmapped signals, using wiki chunks and code context for evidence
2. **Agent teams** — only if multiple facilities need simultaneous mapping
   AND the infrastructure (LiteLLM proxy, Langfuse) is already deployed

The threshold is pragmatic: for TCV's ~11,596 signals, even 60% coverage =
~7,000 mappings. The remaining 40% likely includes structural nodes
(containers, metadata) that genuinely have no IMAS equivalent.

## Implementation Order

1. [ ] Add `SOURCE_NODE` relationship to `facility.yaml` schema
2. [ ] Create `get_catalog()` MCP tool
3. [ ] Add DataAccess generation to `discover static` CLI
4. [ ] Add FacilitySignal batch generation to `discover static` CLI
5. [ ] Implement Phase 1 deterministic matching
6. [ ] Implement Phase 2 LLM-assisted matching
7. [ ] Implement Phase 3 validation
8. [ ] Add `discover mappings` CLI command
9. [ ] Run on TCV, evaluate coverage
10. [ ] Decide on agent teams based on coverage results
