# COCOS Identification Strategy

> **Status**: Planning  
> **Last Updated**: 2025-01-19  
> **Dependencies**: [COCOS Paper](https://doi.org/10.1016/j.cpc.2012.09.010), imas-python ids_convert.py

## Executive Summary

This document outlines a strategy for **grounded COCOS identification** - determining coordinate conventions from actual data rather than metadata inference. The approach prioritizes composable routines that calculate COCOS directly from physics quantities.

## Background

### What is COCOS?

COCOS (COordinate COnventions for Specifying tokamak orientations) is a single integer (1-8 or 11-18) that fully encodes:
- Cylindrical coordinate handedness (σRφZ)
- Poloidal coordinate handedness (σρθφ)
- Poloidal flux sign convention (σBp)
- Whether ψ is divided by 2π (eBp)

### Known COCOS Values from Paper

| Code/System | COCOS | Source |
|-------------|-------|--------|
| CHEASE (default) | 2 | Sauter paper Table IV |
| ONETWO | 2 | Sauter paper Table IV |
| ITER/IMAS DD v3.x | 11 | IMAS standard |
| IMAS DD v4.x | 17 | DD version change |
| TCV/LIUQE | 17 | TCV facility standard |
| EU-ITM (pre-2012) | 13 | Sauter paper |
| ORB5 | 3 | Sauter paper (uses -Ip) |

### IMAS DD COCOS Change (DD3 → DD4)

The IMAS Data Dictionary changed from COCOS 11 (v3.x) to COCOS 17 (v4.x). This is handled by **imas-python** via:

1. **`cocos_label_transformation` attribute** in DD XML for affected paths:
   - `psi_like` - requires sign flip
   - `dodpsi_like` - derivative paths, also flip

2. **Hardcoded paths in `_3to4_sign_flip_paths`** for paths missing the attribute:
   - `equilibrium/time_slice/boundary/psi`
   - `core_profiles/profiles_1d/grid/psi_magnetic_axis`
   - etc.

3. **Sign flip function `_cocos_change()`**:
```python
def _cocos_change(node: IDSBase) -> None:
    """Handle COCOS definition change: multiply values by -1."""
    node.value = -node.value
```

**Critical Insight**: The DD doesn't store a COCOS field - it's implicit in the version. The COCOS is DD-level metadata, not path-level.

## Design Principles

### 1. Data-Grounded Determination

**Don't rely on metadata alone**. Calculate COCOS from physics:

| Observable | Calculation | COCOS Implication |
|------------|-------------|-------------------|
| sign(ψ_edge - ψ_axis) | Load psi profile | σIp × σBp |
| sign(dp/dψ) | Load pressure gradient | -σIp × σBp |
| sign(q) | Load q profile | σIp × σB0 × σρθφ |
| sign(Ip), sign(B0) | Load global quantities | σIp, σB0 |

### 2. Composable Routines

Design small, focused functions that can be combined:

```python
# Core determination functions (mixin pattern)
class COCOSCalculator:
    """Mixin providing COCOS calculation from equilibrium data."""
    
    @staticmethod
    def sigma_bp_from_psi(psi_axis: float, psi_edge: float, ip_sign: int) -> int:
        """Determine σBp from poloidal flux gradient direction."""
        psi_increasing = psi_edge > psi_axis
        return ip_sign if psi_increasing else -ip_sign
    
    @staticmethod
    def sigma_rho_theta_phi_from_q(q_sign: int, ip_sign: int, b0_sign: int) -> int:
        """Determine σρθφ from safety factor sign."""
        return q_sign * ip_sign * b0_sign
    
    @staticmethod
    def e_bp_from_psi_magnitude(psi: float) -> int:
        """Determine eBp (0 or 1) from ψ magnitude."""
        # If |ψ| > ~1 Weber, probably not divided by 2π
        return 1 if abs(psi) > 0.5 else 0
```

### 3. Multiple COCOS Support

Many codes support both input and output COCOS with internal transformations:

```python
@dataclass
class CodeCOCOSCapability:
    """COCOS capabilities for an analysis code."""
    native_cocos: int | None  # Internal working convention
    input_cocos: list[int]    # Accepted input conventions
    output_cocos: list[int]   # Available output conventions
    auto_transform: bool       # Does code auto-transform?
```

CHEASE example:
- Native: COCOS 2
- Input: Any (via transformation from paper Eq. 21)
- Output: Any (via transformation from paper Eq. 15)

### 4. Mixin Pattern for Schema

Follow existing `LLMProvenance` mixin pattern:

```yaml
# In schemas/common.yaml
classes:
  COCOSMixin:
    description: >-
      Mixin providing COCOS-related attributes for entities that handle
      equilibrium data. Use on AnalysisCode, Dataset, TreeNode.
    mixin: true
    class_uri: common:COCOSMixin
    attributes:
      cocos_native:
        description: Native COCOS convention (single value, e.g., 2 for CHEASE)
        range: integer
      cocos_input:
        description: Accepted input COCOS values (if code auto-transforms)
        multivalued: true
        range: integer
      cocos_output:
        description: Available output COCOS values
        multivalued: true
        range: integer
      cocos_determination:
        description: How COCOS was determined (metadata, calculated, documented)
        range: COCOSDetermination
      cocos_confidence:
        description: Confidence in COCOS assignment (0.0-1.0)
        range: float
```

## Implementation Strategy

### Phase 1: Schema Foundation

**Objective**: Define COCOS-aware schema with mixin pattern

1. Add `COCOSDetermination` enum to `common.yaml`:
   - `metadata` - From DD version, file header, MDSplus node
   - `calculated` - Computed from physics quantities
   - `documented` - From code documentation or wiki
   - `asserted` - User-provided, not verified

2. Add `COCOSMixin` class to `common.yaml`

3. Apply mixin to:
   - `AnalysisCode` (facility.yaml)
   - `SignConvention` (common.yaml) - already has `cocos_index`
   - Future: `Dataset` when we add data lineage

### Phase 2: Calculation Routines

**Objective**: Create composable COCOS calculators

Location: `imas_codex/cocos/calculator.py`

```python
"""COCOS calculation from equilibrium quantities.

Based on Sauter & Medvedev, CPC 184 (2013) 293-302.
Table I defines the 16 COCOS values via four parameters.
"""

from dataclasses import dataclass
from typing import Literal

@dataclass
class COCOSParameters:
    """The four COCOS-determining parameters from Table I."""
    sigma_bp: Literal[-1, 1]      # Poloidal flux sign
    e_bp: Literal[0, 1]           # ψ/(2π) factor (0 or 1)
    sigma_r_phi_z: Literal[-1, 1] # Cylindrical handedness
    sigma_rho_theta_phi: Literal[-1, 1]  # Poloidal handedness
    
    @property
    def cocos(self) -> int:
        """Compute COCOS value from parameters."""
        # Table I mapping
        base = 10 if self.e_bp == 1 else 0
        # σRφZ determines odd/even
        offset = 1 if self.sigma_r_phi_z == 1 else 2
        # σBp determines 1-4 vs 5-8 range  
        if self.sigma_bp == 1:
            if self.sigma_rho_theta_phi == 1:
                return base + offset  # 1,2 or 11,12
            else:
                return base + offset + 2  # 3,4 or 13,14
        else:
            if self.sigma_rho_theta_phi == 1:
                return base + 8 - offset + 1  # 7,8 or 17,18
            else:
                return base + 8 - offset - 1  # 5,6 or 15,16


def determine_cocos_from_equilibrium(
    psi_axis: float,
    psi_edge: float, 
    ip: float,
    b0: float,
    q: float | None = None,
    dp_dpsi: float | None = None,
) -> tuple[int | None, float]:
    """
    Determine COCOS from equilibrium quantities.
    
    Returns:
        (cocos, confidence) - COCOS value and confidence 0.0-1.0
    
    Based on Eq. 23 from Sauter & Medvedev:
    - sign(ψ_edge - ψ_axis) = σIp × σBp
    - sign(dp/dψ) = -σIp × σBp  
    - sign(q) = σIp × σB0 × σρθφ
    """
    import math
    
    sigma_ip = 1 if ip >= 0 else -1
    sigma_b0 = 1 if b0 >= 0 else -1
    
    # Determine σBp from psi gradient
    psi_diff = psi_edge - psi_axis
    sigma_bp = sigma_ip if psi_diff >= 0 else -sigma_ip
    
    # Determine eBp from magnitude (heuristic)
    max_psi = max(abs(psi_axis), abs(psi_edge))
    e_bp = 1 if max_psi > 0.5 else 0  # >0.5 Wb suggests full ψ
    
    confidence = 0.5  # Base confidence with just psi
    
    # Validate with dp/dpsi if available
    if dp_dpsi is not None:
        expected_sign = -sigma_ip * sigma_bp
        if math.copysign(1, dp_dpsi) == expected_sign:
            confidence += 0.2
        else:
            confidence -= 0.3  # Inconsistent
    
    # Determine σρθφ from q if available
    if q is not None and not math.isnan(q):
        sigma_rho_theta_phi = 1 if (q * sigma_ip * sigma_b0) >= 0 else -1
        confidence += 0.2
    else:
        # Default to right-handed (most common)
        sigma_rho_theta_phi = 1
    
    # Determine σRφZ - harder without explicit geometry
    # Default to (R,φ,Z) right-handed which is most common
    sigma_r_phi_z = 1
    
    params = COCOSParameters(
        sigma_bp=sigma_bp,
        e_bp=e_bp,
        sigma_r_phi_z=sigma_r_phi_z,
        sigma_rho_theta_phi=sigma_rho_theta_phi,
    )
    
    return params.cocos, min(1.0, max(0.0, confidence))
```

### Phase 3: IMAS Integration

**Objective**: Integrate with imas-python patterns

1. **DD Version → COCOS mapping**:
```python
def cocos_from_dd_version(version: str) -> int:
    """Get COCOS from IMAS DD version."""
    from packaging.version import Version
    v = Version(version)
    return 17 if v >= Version("4.0.0") else 11
```

2. **Path-level COCOS transformation detection**:
```python
def path_needs_cocos_transform(ids_name: str, path: str) -> bool:
    """Check if path requires COCOS sign flip between DD3/DD4."""
    # Use imas-python's _3to4_sign_flip_paths
    from imas.ids_convert import _3to4_sign_flip_paths
    return path in _3to4_sign_flip_paths.get(ids_name, [])
```

### Phase 4: Graph Enrichment

**Objective**: Populate COCOS for existing graph entities

1. **AnalysisCode enrichment** from known values:
```cypher
// Set known COCOS from paper
MATCH (c:AnalysisCode {name: 'CHEASE'})
SET c.cocos_native = 2,
    c.cocos_input = [1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18],
    c.cocos_output = [1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18],
    c.cocos_determination = 'documented'

MATCH (c:AnalysisCode {name: 'LIUQE'})
SET c.cocos_native = 17,
    c.cocos_determination = 'documented'
```

2. **TreeNode enrichment** from MDSplus values:
```python
# For nodes like \RESULTS::TOP.EQUIL_*.RESULTS:COCOS
async def enrich_cocos_from_mdsplus(facility: str):
    """Fetch COCOS values from MDSplus and update TreeNodes."""
    # Query graph for COCOS nodes
    cocos_nodes = query("""
        MATCH (t:TreeNode)
        WHERE t.path ENDS WITH ':COCOS'
        RETURN t.path, t.facility_id
    """)
    
    for node in cocos_nodes:
        # Fetch actual value via SSH/MDSplus
        value = await fetch_mdsplus_value(facility, node['t.path'])
        # Update related equilibrium nodes
        ...
```

### Phase 5: Validation & Testing

**Objective**: Ensure COCOS assignments are correct

1. **Cross-validation checks**:
```python
def validate_cocos_consistency(
    cocos: int,
    psi_axis: float, psi_edge: float,
    ip: float, b0: float, q: float,
) -> list[str]:
    """Validate COCOS against physics quantities."""
    errors = []
    
    params = cocos_to_parameters(cocos)
    sigma_ip = 1 if ip >= 0 else -1
    sigma_b0 = 1 if b0 >= 0 else -1
    
    # Check psi gradient
    expected_psi_sign = sigma_ip * params.sigma_bp
    actual_psi_sign = 1 if (psi_edge - psi_axis) >= 0 else -1
    if expected_psi_sign != actual_psi_sign:
        errors.append(f"ψ gradient inconsistent: expected {expected_psi_sign}, got {actual_psi_sign}")
    
    # Check q sign
    expected_q_sign = sigma_ip * sigma_b0 * params.sigma_rho_theta_phi
    actual_q_sign = 1 if q >= 0 else -1
    if expected_q_sign != actual_q_sign:
        errors.append(f"q sign inconsistent: expected {expected_q_sign}, got {actual_q_sign}")
    
    return errors
```

2. **Test fixtures** with known COCOS data from facilities

## Data Sources for COCOS

### Direct Sources (Highest Confidence)

| Source | Location | Confidence |
|--------|----------|------------|
| MDSplus COCOS node | `\RESULTS::*.COCOS` | 0.95 |
| IMAS DD version | `ids_properties` | 0.9 |
| Code output file | EQDSK header | 0.85 |

### Calculated Sources (Medium Confidence)

| Method | Required Data | Confidence |
|--------|--------------|------------|
| ψ gradient + Ip sign | psi profile, Ip | 0.6-0.8 |
| q sign validation | q profile, Ip, B0 | +0.1-0.2 |
| dp/dψ validation | pressure, psi | +0.1-0.2 |

### Documented Sources (Variable Confidence)

| Source | Confidence |
|--------|------------|
| COCOS paper Table IV | 0.95 |
| Code documentation | 0.7-0.9 |
| Wiki mentions | 0.5-0.7 |

## Current Graph State

### Existing Entities

- **SignConvention nodes**: 29 nodes, mostly sign conventions
- **COCOS-specific**: `epfl:cocos:17`, `epfl:cocos:2`, `imas:cocos:11`
- **AnalysisCode.cocos**: All NULL currently

### Missing

- AnalysisCode.cocos_native not populated
- No COCOSMixin on schema classes
- No calculation routines
- DD version → COCOS not captured

## Open Questions

1. **COCOS 20 in graph**: Found `epfl:cocos:ef2166ec` with cocos_index=20. 
   This is **invalid** per paper (only 1-8, 11-18). Investigation shows this was
   incorrectly extracted from wiki page `Boundary_physics`. The description field
   contains HTML fragments suggesting parsing error. **Action**: Delete this node
   and improve wiki extraction to validate COCOS values against valid range.

2. **Code-specific variations**: Some codes (like ORB5) use non-standard 
   sign conventions. How to capture?

3. **Time-varying COCOS**: Can a dataset change COCOS mid-shot? Probably not,
   but should we allow per-time-slice COCOS?

## Next Steps

1. [ ] Add `COCOSMixin` to common.yaml schema
2. [ ] Create `imas_codex/cocos/` module with calculator
3. [ ] Populate known COCOS values for AnalysisCodes
4. [ ] Create agent tool for COCOS determination from data
5. [ ] Add COCOS validation to enrichment pipeline

## References

- Sauter, O. and Medvedev, S.Yu., "Tokamak coordinate conventions: COCOS", 
  Computer Physics Communications, 184(2):293–302, February 2013.
  DOI: [10.1016/j.cpc.2012.09.010](https://doi.org/10.1016/j.cpc.2012.09.010)

- imas-python `ids_convert.py` - DD3↔DD4 COCOS transformation logic

- IMAS Data Dictionary homepage - COCOS declaration in header metadata
