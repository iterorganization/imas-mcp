# COCOS Enrichment

> **Status**: In Progress (4/5 items complete)  
> **Dependencies**: `imas_codex.cocos` module (implemented)

## Overview

Extend COCOS identification to graph enrichment and imas-python integration.
The core calculator module is complete - this plan covers remaining work.

## Architecture Clarification

**COCOS determination is deterministic physics** - Eq. 23 from Sauter paper.
No LLM involvement needed for calculation. The existing `determine_cocos()` 
function takes physics values directly:

```python
# CORRECT: Data-agnostic, works with any source
determine_cocos(
    psi_axis=...,    # Required: float [Wb]
    psi_edge=...,    # Required: float [Wb]  
    ip=...,          # Required: float [A]
    b0=...,          # Required: float [T]
    q=...,           # Optional: float (improves confidence)
    dp_dpsi=...,     # Optional: float (validates σBp)
)
```

**Data loading is a separate concern** - Facilities use different formats:
- EPFL/TCV: MDSplus
- ITER: IMAS IDS (HDF5 backend)
- Other: EQDSK files, custom formats

The caller is responsible for loading data from their source.

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

### 3. Validation Script (Deterministic)

Create a CLI script for batch COCOS validation - **no LLM needed**:

```bash
# Validate COCOS for a code's outputs against physics data
uv run imas-codex cocos validate --code LIUQE --shot 12345

# Check if declared COCOS matches calculated COCOS from data
uv run imas-codex cocos check equilibrium.h5
```

Implementation:

```python
def validate_cocos_from_data(
    declared_cocos: int,
    *,
    # Required physics quantities
    psi_axis: float,
    psi_edge: float,
    ip: float,
    b0: float,
    # Optional - improve confidence
    q: float | None = None,
    dp_dpsi: float | None = None,
    psi_boundary: float | None = None,
) -> ValidationResult:
    """Validate declared COCOS against physics data.
    
    This is deterministic - uses Eq. 23 from Sauter paper.
    No LLM involvement.
    
    Returns:
        ValidationResult with:
        - is_consistent: bool
        - calculated_cocos: int
        - confidence: float
        - inconsistencies: list[str]
    """
    calculated, confidence = determine_cocos(
        psi_axis=psi_axis,
        psi_edge=psi_edge,
        ip=ip,
        b0=b0,
        q=q,
        dp_dpsi=dp_dpsi,
    )
    
    errors = validate_cocos_consistency(
        cocos=declared_cocos,
        psi_axis=psi_axis,
        psi_edge=psi_edge,
        ip=ip,
        b0=b0,
        q=q,
        dp_dpsi=dp_dpsi,
    )
    
    return ValidationResult(
        is_consistent=len(errors) == 0,
        declared_cocos=declared_cocos,
        calculated_cocos=calculated,
        confidence=confidence,
        inconsistencies=errors,
    )
```

### 4. Data Loaders (Facility-Specific)

Separate data loading from COCOS calculation:

```python
# imas_codex/cocos/loaders.py

@dataclass
class EquilibriumData:
    """Physics quantities needed for COCOS determination."""
    psi_axis: float
    psi_edge: float
    ip: float
    b0: float
    q: float | None = None
    dp_dpsi: float | None = None
    source: str = ""  # e.g., "mdsplus:tcv:12345"


def load_from_imas_ids(equilibrium_ids) -> EquilibriumData:
    """Load from IMAS equilibrium IDS."""
    ts = equilibrium_ids.time_slice[0]
    return EquilibriumData(
        psi_axis=ts.global_quantities.psi_axis,
        psi_edge=ts.global_quantities.psi_boundary,
        ip=ts.global_quantities.ip,
        b0=ts.global_quantities.magnetic_axis.b_field_tor,
        q=ts.profiles_1d.q[len(ts.profiles_1d.q)//2] if ts.profiles_1d.q else None,
    )


def load_from_eqdsk(filepath: Path) -> EquilibriumData:
    """Load from G-EQDSK file."""
    # Parse EQDSK format
    ...
```

## Where LLM Agents ARE Useful

LLM agents are valuable for **orchestration and interpretation**, not calculation:

1. **Finding data sources** - "Which MDSplus nodes contain equilibrium data for LIUQE?"
2. **Interpreting results** - "The calculated COCOS is 11 but declared is 17 - what does this mean?"
3. **Suggesting fixes** - "This code likely needs a sign flip on ψ outputs"
4. **Graph enrichment** - Deciding which codes to validate, prioritizing work

## Success Criteria

- [x] `path_needs_cocos_transform()` function with tests
- [x] `ValidationResult` dataclass and `validate_cocos_from_data()` function
- [x] CLI commands: `cocos validate`, `cocos check`
- [x] Data loaders for IMAS IDS and EQDSK formats
- [ ] TreeNode enrichment for facilities with COCOS MDSplus nodes
