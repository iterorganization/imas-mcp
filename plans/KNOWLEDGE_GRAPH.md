# Federated Fusion Knowledge Graph - Implementation Plan

**Status**: Draft  
**Date**: 2024-12-22  
**Based on**: EPFL/TCV exploration findings

---

## 1. Executive Summary

Build a Neo4j-based knowledge graph that captures facility-specific knowledge discovered through exploration. Start with EPFL/TCV as the proving ground, then generalize.

**Key insight from EPFL exploration**: The real value isn't in file paths—it's in the semantic relationships between:
- MDSplus trees and their node hierarchies
- TDI functions and the data they access
- Analysis codes (LIUQE, ASTRA, TORAY) and their inputs/outputs
- Diagnostic systems and their signals

---

## 2. Discovered Data Model (from EPFL)

### 2.1 What We Found

```
TCV Data Access Architecture
============================

User Query (Python)
    ↓
TDI Functions (213 .fun files)
    ├── tcv_ip()          → Plasma current
    ├── tcv_eq("kappa")   → Equilibrium via LIUQE
    ├── ts_te()           → Thomson Te
    └── ...
    ↓
MDSplus Trees
    ├── tcv_shot (primary)
    │   ├── ATLAS, BASE, DIAGZ, MAGNETICS, ECRH, HYBRID...
    │   └── RAW_TREES → links to raw data
    │
    ├── results (analysis outputs)
    │   ├── LIUQE (EQ_RECON, EQ_RECON_2, EQ_RECON_3)
    │   ├── ASTRA, CQL3D, TORAY (transport codes)
    │   ├── THOMSON, ECE, FIR (diagnostics)
    │   └── ...
    │
    └── (distributed across servers)
        ├── tcvdata     - Main data
        ├── spcsrv1     - Legacy
        ├── spcsrv8     - HXR, video
        ├── mantis4     - Thomson scattering
        └── ...
```

### 2.2 Sample Data Retrieved

```python
# Shot 89031 with LIUQE reconstruction
tcv_ip()           → 62,718 points, t=[-2.2, 4.0]s
tcv_eq("kappa")    → 500 points, t=[0.0, 1.0]s
tcv_eq("r_contour")→ (500, 129) boundary shape
```

---

## 3. Proposed Node Types

### Phase 1: Core Nodes (Week 1-2)

| Node Type | Properties | Example |
|-----------|------------|---------|
| `Facility` | name, description, ssh_host | EPFL/TCV |
| `MDSplusServer` | hostname, role | tcvdata (main), spcsrv8 (video) |
| `MDSplusTree` | name, description | tcv_shot, results, magnetics |
| `TreeNode` | path, type, units, description | \RESULTS::LIUQE |
| `TDIFunction` | name, file_path, description | tcv_eq, tcv_ip |
| `Diagnostic` | name, category | Thomson, ECE, Bolometry |
| `AnalysisCode` | name, type | LIUQE, ASTRA, TORAY |

### Phase 2: Relationships

```cypher
(TDIFunction)-[:ACCESSES]->(TreeNode)
(TDIFunction)-[:CALLS]->(TDIFunction)
(TreeNode)-[:CHILD_OF]->(TreeNode)
(TreeNode)-[:STORED_ON]->(MDSplusServer)
(AnalysisCode)-[:PRODUCES]->(TreeNode)
(Diagnostic)-[:WRITES_TO]->(TreeNode)
```

### Phase 3: IMAS Mapping

| Relationship | Description |
|--------------|-------------|
| `(TreeNode)-[:MAPS_TO]->(IMASPath)` | Source→Target mapping |
| `(TDIFunction)-[:EQUIVALENT_TO]->(IMASAccessor)` | Function equivalence |

---

## 4. Implementation Phases

### Phase 1: Local Neo4j Spike (2-3 days)

**Goal**: Prove we can ingest EPFL data into Neo4j and query it.

```bash
# Start Neo4j
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:latest

# Create Python ingest script
uv run python scripts/ingest_facility.py epfl
```

**Deliverables**:
- `scripts/ingest_facility.py` - reads YAML, writes Cypher
- Sample queries working in Neo4j browser

### Phase 2: MDSplus Tree Ingestion (1 week)

**Goal**: Automatically discover and ingest MDSplus tree structure.

```python
# Pseudo-code for tree walker
def walk_tree(connection, tree_name, shot):
    nodes = connection.get(f'getnci("\\\\{tree_name}::*", "FULLPATH")')
    for node in nodes:
        # Create TreeNode in Neo4j
        # Recurse into children
```

**Deliverables**:
- `imas_codex/discovery/mdsplus_walker.py`
- TreeNode population for TCV shot

### Phase 3: TDI Function Analysis (1 week)

**Goal**: Parse TDI functions to extract data access patterns.

```
Input:  /usr/local/CRPP/tdi/tcv/tcv_eq.fun
Output: 
  - Function: tcv_eq
  - Parameters: _var, _mode, _keyword
  - Accesses: \RESULTS::EQ_RECON, \RESULTS::LIUQE, ...
  - Calls: tcv_eq_profile, ...
```

**Deliverables**:
- `imas_codex/discovery/tdi_parser.py`
- TDIFunction and relationship edges

### Phase 4: MCP Tools (3-5 days)

**Goal**: Query the graph from agents.

```python
@server.tool()
async def facility_config(
    action: Literal["read", "write", "validate", "list", "query"],
    facility: str | None = None,
    cypher: str | None = None,  # For action="query"
    ...
):
    """
    Manage facility knowledge graph.
    
    Actions:
        query - Execute Cypher query against facility subgraph
    
    Example:
        facility_config(action="query", facility="epfl",
            cypher="MATCH (t:TDIFunction)-[:ACCESSES]->(n:TreeNode) RETURN t.name, n.path")
    """
```

---

## 5. Schema Evolution Strategy

### LinkML-Driven

```yaml
# ontology/facility.yaml
classes:
  Facility:
    attributes:
      name: {identifier: true}
      ssh_host: string
      
  MDSplusServer:
    attributes:
      hostname: {identifier: true}
      facility: {range: Facility}
      role: string
      
  TreeNode:
    attributes:
      path: {identifier: true}
      tree: {range: MDSplusTree}
      node_type: {enum: [STRUCTURE, SIGNAL, AXIS, ...]}
```

### Validation Pipeline

```
Discovery → JSON → Pydantic (from LinkML) → Valid? → Neo4j Ingest
                        ↓ Invalid
                   Flag for schema extension
```

---

## 6. Questions to Resolve

1. **Granularity**: Do we ingest every TreeNode or just "interesting" ones?
   - Full ingest: millions of nodes
   - Curated: hundreds, manually selected

2. **Shot-independence**: Tree structure is shot-dependent. Do we:
   - Pick a reference shot per facility?
   - Store multiple shot versions?
   - Only store model trees?

3. **TDI parsing**: TDI is a complex language. Do we:
   - Parse fully (hard)
   - Extract patterns with regex (fragile)
   - Use LLM to summarize (expensive but flexible)

4. **Graph vs YAML**: During transition, do we:
   - Keep YAML as source of truth, graph as view?
   - Make graph authoritative, generate YAML for humans?

---

## 7. Immediate Next Steps

1. **Set up Neo4j locally** (30 min)
2. **Write minimal ingest script** for current epfl.yaml (2 hrs)
3. **Test queries** in Neo4j browser (1 hr)
4. **Walk one MDSplus tree** from EPFL and ingest (4 hrs)
5. **Evaluate and decide** on Phase 2 scope

---

## Appendix: EPFL Discovery Artifacts

### MDSplus Configuration
- Server: tcvdata (main)
- Config: `/usr/local/mdsplus/local/mdsplus.conf`
- Trees: 14 TCV shot trees + raw diagnostics + results

### TDI Functions
- Location: `/usr/local/CRPP/tdi/tcv/` (NFS from 10.27.128.167)
- Count: 213 functions
- Key functions: tcv_eq, tcv_ip, ts_te, etc.

### Analysis Codes (in RESULTS tree)
- LIUQE (EQ_RECON) - Equilibrium reconstruction
- ASTRA - Transport
- CQL3D - Fokker-Planck
- TORAY - Ray tracing
- PROFFIT - Profile fitting

### Data Servers
| Server | Role |
|--------|------|
| tcvdata | Main MDSplus server |
| spcsrv1 | Legacy pre-Linux trees |
| spcsrv8 | Large data (HXR), video |
| crppsrv1 | tcv_day, lupin, rga |
| mantis4 | Thomson scattering |
| falcondata | Falcon camera |
| raiddata | RAID storage |
| scd | Real-Time Control |

### Container
- TRANSP v24.5.0 at `/data/apptainer/transp_v24.5.0_noimas_epfl.sif`
