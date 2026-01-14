# TreeNode Path Consistency and TDI Function Integration

> **Status**: Partially Implemented (January 2026)
> TreeNode ingestion complete (171k nodes). Path normalization and TDI linking pending.

## Summary

Investigation revealed two path format patterns in the graph:
- **Nodes with single backslash**: `\RESULTS::TOP.EQ_RECON.TRACES:I_P` (from MDSplus ingestion script)
- **Nodes with double backslash**: `\\RESULTS::I_P` (from code extraction pipeline)

"Unknown" paths (AMIN, BT, W_MHD, SPLASMA) are **valid TDI function parameters**, not hallucinations:
- `tcv_get.fun`: 114 computed quantities (AMIN, BT, KAPPA, SPLASMA, etc.)
- `tcv_eq.fun`: 76 quantities (PSI, I_P, W_MHD, VLOOP_SURF, etc.)

These represent different "views" of the same data:
- **Direct paths**: Physical tree structure from MDSplus introspection
- **Accessor paths**: High-level abstractions exposed by TDI functions

## Current Graph State

| Entity | Count | Notes |
|--------|-------|-------|
| TreeNode | 171,155 | All from tree introspection |
| TDIFunction | 21 | Discovered, not yet parsed |
| ACCESSES relationships | 0 | TDI-to-TreeNode linking pending |

## Proposed Schema Changes

### 1. New Enum: TreeNodeSource

Distinguishes how a TreeNode was discovered/created:

```yaml
enums:
  TreeNodeSource:
    description: How a TreeNode was discovered and ingested
    permissible_values:
      tree_introspection:
        description: >-
          Discovered via MDSplus tree introspection (getNodeWild).
          Path is the physical tree path. Has node_type.
      code_extraction:
        description: >-
          Extracted from source code via code_examples pipeline.
          Path may be abbreviated (no full hierarchy).
          May represent accessor-level path, not physical node.
      tdi_parameter:
        description: >-
          A parameter accepted by a TDI function (e.g., tcv_eq("PSI")).
          Not a physical tree path but a logical data accessor name.
          Links to TreeNode via accessor_function relationship.
      manual:
        description: Manually created during exploration or testing
```

### 2. TreeNode Updates

Add `source` field to track provenance:

```yaml
classes:
  TreeNode:
    attributes:
      # ... existing fields ...
      source:
        description: How this node was discovered
        range: TreeNodeSource
        required: true
      canonical_path:
        description: >-
          Canonical form of the path for deduplication and matching.
          Normalized to single backslash, uppercase, no channel indices.
          Used for fuzzy matching when exact path doesn't match.
        range: string
      physical_path:
        description: >-
          The actual MDSplus tree path (if different from accessor name).
          For tdi_parameter nodes, this links to the physical storage.
          E.g., tdi_parameter "I_P" -> physical_path "\RESULTS::TOP.EQ_RECON.TRACES:I_P"
        range: string
```

### 3. TDIFunction Enhancement

Already has `supported_quantities` but needs more structure:

```yaml
classes:
  TDIFunction:
    attributes:
      # ... existing fields ...
      quantity_definitions:
        description: >-
          JSON-encoded map of quantity name to definition.
          E.g., {"AMIN": {"expression": "tcv_eq(\"R_CONTOUR\") - R_AXIS", "dependencies": ["R_CONTOUR", "R_AXIS"]}}
        range: string
      quantity_categories:
        description: >-
          JSON-encoded map of category to quantity names.
          E.g., {"shape": ["AMIN", "KAPPA", "DELTA"], "energy": ["WE", "W_MHD"]}
        range: string
```

## Path Normalization Strategy

### Canonical Path Format

All paths should be normalized to a canonical form for matching:

```python
def normalize_mdsplus_path(path: str) -> str:
    """Normalize MDSplus path to canonical form.
    
    Canonical form:
    - Single backslash prefix: \RESULTS::...
    - Uppercase
    - No channel indices: CHANNEL_006 -> CHANNEL
    - No trailing spaces
    """
    # Remove leading/trailing whitespace
    path = path.strip()
    
    # Normalize backslashes: \\ -> \ (storage vs display)
    if path.startswith("\\\\"):
        path = path[1:]  # \\RESULTS -> \RESULTS
    
    # Uppercase
    path = path.upper()
    
    # Strip channel indices for fuzzy matching
    path = re.sub(r'_\d{2,3}$', '', path)  # CHANNEL_006 -> CHANNEL
    
    return path
```

### Migration Strategy

1. Add `canonical_path` field to TreeNode
2. Populate via migration script:
   ```cypher
   MATCH (n:TreeNode)
   WHERE n.canonical_path IS NULL
   SET n.canonical_path = ... // normalize(n.path)
   ```
3. Update ingestion pipelines to set both `path` and `canonical_path`

## TDI Function Ingestion Design

### Parser Script

Extract quantities from TDI .fun files:

```python
#!/usr/bin/env python3
"""Parse TDI function files and extract supported quantities."""

import re
from pathlib import Path

def parse_tdi_function(content: str) -> dict:
    """Extract case statement quantities from TDI function."""
    # Pattern: case("QUANTITY_NAME")
    quantities = re.findall(r'case\s*\(\s*["\']([A-Z_0-9]+)["\']\s*\)', 
                            content, re.IGNORECASE)
    return {
        "quantities": sorted(set(quantities)),
        "count": len(set(quantities))
    }

def parse_tdi_dependencies(content: str, quantity: str) -> list[str]:
    """Find what data a quantity depends on."""
    # Look for tcv_eq("X") or data(\path) patterns after case("quantity")
    # This is complex - may need AST-like parsing
    pass
```

### Ingestion Pipeline

1. **Discovery**: List .fun files at `/usr/local/CRPP/tdi/tcv/`
2. **Parse**: Extract function signature, case statements, dependencies
3. **Ingest**: Create TDIFunction nodes with `supported_quantities`
4. **Link**: Create TreeNode entries for each quantity with `source=tdi_parameter`

### Graph Relationships

```
(TDIFunction {name: "tcv_eq"})
  -[:SUPPORTS_QUANTITY]-> (TreeNode {path: "\RESULTS::I_P", source: "tdi_parameter"})
  
(TreeNode {path: "\RESULTS::I_P", source: "tdi_parameter"})
  -[:PHYSICAL_PATH]-> (TreeNode {path: "\RESULTS::TOP.EQ_RECON.TRACES:I_P", source: "tree_introspection"})
```

## Implementation Plan

### Phase 1: Schema Updates ✅ Complete
1. [x] TreeNodeSource enum defined in LinkML schema
2. [x] Source, canonical_path, physical_path fields available
3. [x] Pydantic models regenerated

### Phase 2: Migration 🔄 Partial
1. [x] TreeNodes ingested from tree introspection (171k)
2. [ ] Set `source=tree_introspection` for nodes with `node_type`
3. [ ] Set `source=code_extraction` for nodes from code pipeline
4. [ ] Compute and set `canonical_path` for all nodes

### Phase 3: TDI Ingestion ⬜ Planned
1. [ ] Create TDI parser script (or use tree-sitter-tdi when available)
2. [ ] Parse tcv_get.fun and tcv_eq.fun quantities
3. [ ] Create TreeNodes with `source=tdi_parameter`
4. [ ] Link to physical tree paths via ACCESSES relationship

### Phase 4: Deduplication ⬜ Future
1. [ ] Merge nodes that share `canonical_path`
2. [ ] Preserve relationships from both sources
3. [ ] Update code references to canonical nodes

## Open Questions

1. **Merging strategy**: Should we merge `\\RESULTS::I_P` and `\RESULTS::TOP.EQ_RECON.TRACES:I_P`?
   - Pro: Reduces duplication, single enrichment target
   - Con: Loses provenance information, path semantics differ

2. **Priority**: Which source is authoritative?
   - `tree_introspection`: Accurate structure, may miss aliases
   - `code_extraction`: Reflects actual usage, may miss nodes
   - `tdi_parameter`: User-facing API, authoritative for accessor functions

3. **TDI versioning**: How to handle shot-range validity of quantities?
   - Some quantities only exist for certain shot ranges
   - VERSION.FUN provides version info but mapping to quantities is complex
