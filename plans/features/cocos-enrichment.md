# COCOS Enrichment

> **Status**: Pending  
> **Dependencies**: `imas_codex.cocos` module (implemented)

## Overview

Extend COCOS identification to graph enrichment and imas-python integration.
The core calculator module is complete - this plan covers remaining work.

## Remaining Work

### 1. imas-python Path Transform Integration

Integrate with imas-python's `_3to4_sign_flip_paths` for path-level COCOS transforms:

```python
def path_needs_cocos_transform(ids_name: str, path: str) -> bool:
    """Check if path requires COCOS sign flip between DD3/DD4."""
    from imas.ids_convert import _3to4_sign_flip_paths
    return path in _3to4_sign_flip_paths.get(ids_name, [])
```

**Use case**: When mapping code outputs to IMAS IDS, identify which paths
need sign flips based on source/target DD versions.

### 2. TreeNode COCOS Enrichment

Enrich TreeNodes from MDSplus COCOS nodes (e.g., `\RESULTS::*.COCOS`):

```python
async def enrich_cocos_from_mdsplus(facility: str):
    """Fetch COCOS values from MDSplus and update TreeNodes."""
    cocos_nodes = query("""
        MATCH (t:TreeNode)
        WHERE t.path ENDS WITH ':COCOS'
        RETURN t.path, t.facility_id
    """)
    # Fetch actual values via SSH/MDSplus and link to equilibrium nodes
```

### 3. Agent Tools

Create MCP tools for automated COCOS determination:

- `determine_cocos_from_equilibrium(shot, tree)` - Calculate from MDSplus data
- `validate_code_cocos(code_name)` - Verify documented COCOS against data

## Success Criteria

- [ ] `path_needs_cocos_transform()` function with tests
- [ ] TreeNode enrichment for facilities with COCOS MDSplus nodes
- [ ] Agent tool for runtime COCOS determination
