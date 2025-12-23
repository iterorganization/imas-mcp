# Federated Fusion Knowledge Graph - Implementation Plan

**Status**: Active  
**Date**: 2024-12-23  
**Based on**: EPFL/TCV exploration findings, LinkML ecosystem research

---

## 1. Executive Summary

Build a Neo4j-based knowledge graph that captures facility-specific knowledge discovered through exploration. The graph schema is defined in **LinkML** (`schemas/facility.yaml`) as the single source of truth.

**Architecture Decision**: Use LinkML schema to auto-generate all graph artifacts:
- Pydantic models via `gen-pydantic`
- Neo4j node labels derived from class names
- Relationships derived from slots with class ranges

**Key insight from EPFL exploration**: The real value isn't in file paths—it's in the semantic relationships between:
- MDSplus trees and their node hierarchies
- TDI functions and the data they access
- Analysis codes (LIUQE, ASTRA, TORAY) and their inputs/outputs
- Diagnostic systems and their signals

---

## 2. Schema-First Approach

### 2.1 LinkML as Single Source of Truth

The schema at `imas_codex/schemas/facility.yaml` defines:

```yaml
classes:
  Facility:           # → Neo4j label: Facility
    attributes:
      id: ...
      name: ...
      
  MDSplusServer:      # → Neo4j label: MDSplusServer
    attributes:
      hostname: ...
      facility_id:    # → Relationship: -[:FACILITY_ID]->
        range: Facility
        
  TreeNode:           # → Neo4j label: TreeNode
    attributes:
      path: ...
      tree_name:      # → Relationship: -[:TREE_NAME]->
        range: MDSplusTree
```

### 2.2 Deriving Graph Structure

**Node Labels** = Class names from LinkML:
- `Facility`, `MDSplusServer`, `MDSplusTree`, `TreeNode`, `TDIFunction`, etc.

**Relationships** = Slots with class ranges:
- `facility_id: Facility` → Creates relationship to Facility node
- `tree_name: MDSplusTree` → Creates relationship to Tree node
- `accesses: TreeNode[]` → Creates multiple ACCESSES relationships

### 2.3 Current vs Target State

| Component | Current | Target |
|-----------|---------|--------|
| Schema | `schemas/facility.yaml` ✅ | Same |
| Pydantic models | Auto-generated ✅ | Same |
| Neo4j labels | Hard-coded in `cypher.py` ⚠️ | Derived from schema |
| Relationships | Hard-coded in `cypher.py` ⚠️ | Derived from schema |
| Client | Manual CRUD ✅ | Consider linkml-store |

---

## 3. Recommended Tools

### Option A: linkml-store (Preferred for Future)

[linkml-store](https://github.com/linkml/linkml-store) provides unified abstraction:

```python
from linkml_store import Client

client = Client()
db = client.attach_database(
    "neo4j://localhost:7687",
    schema="schemas/facility.yaml"
)

# Schema-aware operations
facilities = db.get_collection("Facility")
facilities.insert({"id": "epfl", "name": "EPFL/TCV", "machine": "TCV"})
facilities.query(where={"machine": "TCV"})
```

**Benefits**:
- Schema validation built-in
- Backend-agnostic (swap Neo4j for DuckDB in dev)
- Semantic search with embeddings
- No manual label/relationship mapping

### Option B: SchemaView Introspection (Current Direction)

Use `linkml_runtime` to derive structure at runtime:

```python
from linkml_runtime.utils.schemaview import SchemaView

sv = SchemaView("schemas/facility.yaml")

# Derive labels
node_labels = list(sv.all_classes().keys())

# Derive relationships
for cls_name, cls_def in sv.all_classes().items():
    for slot_name in sv.class_induced_slots(cls_name):
        slot = sv.get_slot(slot_name)
        if slot.range in node_labels:
            print(f"{cls_name} -[:{slot_name.upper()}]-> {slot.range}")
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
