# COCOS Identification

> Coordinate convention determination from equilibrium data.

## Overview

The `imas_codex.cocos` module provides utilities for working with tokamak coordinate conventions (COCOS) as defined in Sauter & Medvedev, CPC 184 (2013) 293-302.

**Key capabilities:**
- Calculate COCOS from equilibrium physics quantities (ψ, Ip, B0, q)
- Convert between COCOS representations
- Validate COCOS consistency against data
- Map IMAS DD versions to COCOS (3.x→11, 4.x→17)

## COCOS Fundamentals

COCOS is a single integer (1-8 or 11-18) encoding four parameters:

| Parameter | Symbol | Values | Meaning |
|-----------|--------|--------|---------|
| Poloidal flux sign | σBp | ±1 | Whether ψ increases from axis to edge with positive Ip |
| ψ normalization | eBp | 0 or 1 | 0 = ψ/(2π), 1 = full ψ |
| Cylindrical handedness | σRφZ | ±1 | +1 for (R,φ,Z) right-handed |
| Poloidal handedness | σρθφ | ±1 | +1 for (ρ,θ,φ) right-handed |

**COCOS ranges:**
- 1-8: ψ/(2π) normalization (eBp=0)
- 11-18: Full ψ (eBp=1)

## Known Code Conventions

| Code/System | COCOS | Source |
|-------------|-------|--------|
| CHEASE | 2 | Sauter paper Table IV |
| ONETWO | 2 | Sauter paper Table IV |
| LIUQE/TCV | 17 | TCV facility standard |
| ITER/IMAS DD v3.x | 11 | IMAS standard |
| IMAS DD v4.x | 17 | DD version change |
| ORB5 | 3 | Sauter paper |

## API Reference

### Core Functions

```python
from imas_codex.cocos import (
    VALID_COCOS,           # frozenset of valid values {1-8, 11-18}
    COCOSParameters,       # Dataclass with 4 parameters
    cocos_to_parameters,   # int → COCOSParameters
    cocos_from_dd_version, # "3.41.0" → 11, "4.0.0" → 17
    determine_cocos,       # Physics quantities → (cocos, confidence)
    validate_cocos_consistency,  # Check COCOS against data
)
```

### Determining COCOS from Data

```python
from imas_codex.cocos import determine_cocos

# From equilibrium quantities (Eq. 23 from Sauter paper)
cocos, confidence = determine_cocos(
    psi_axis=0.5,      # Poloidal flux at magnetic axis [Wb]
    psi_edge=-0.2,     # Poloidal flux at plasma edge [Wb]
    ip=-1e6,           # Plasma current [A]
    b0=-5.0,           # Toroidal field at axis [T]
    q=3.0,             # Safety factor (optional, improves confidence)
    dp_dpsi=-1e3,      # Pressure gradient (optional, validates)
)
print(f"COCOS={cocos} (confidence={confidence:.2f})")
```

**Determination logic (Eq. 23):**
- sign(ψ_edge - ψ_axis) = σIp × σBp
- sign(dp/dψ) = -σIp × σBp
- sign(q) = σIp × σB0 × σρθφ

### Validating COCOS

```python
from imas_codex.cocos import validate_cocos_consistency

errors = validate_cocos_consistency(
    cocos=11,
    psi_axis=0.5, psi_edge=-0.2,
    ip=-1e6, b0=-5.0, q=3.0,
)
if errors:
    print("Inconsistencies:", errors)
```

### DD Version Mapping

```python
from imas_codex.cocos import cocos_from_dd_version

cocos_from_dd_version("3.41.0")  # → 11
cocos_from_dd_version("4.0.0")   # → 17
```

## Schema Integration

The `COCOSMixin` in `common.yaml` provides COCOS attributes for graph entities:

```yaml
COCOSMixin:
  mixin: true
  attributes:
    cocos_native:     # Native working convention (int)
    cocos_input:      # Accepted input conventions (list[int])
    cocos_output:     # Available output conventions (list[int])
    cocos_determination:  # How determined (COCOSDetermination enum)
    cocos_confidence:     # Confidence 0.0-1.0
```

**Applied to:** `AnalysisCode`

**COCOSDetermination values:**
- `metadata` - From DD version, file header, MDSplus node
- `calculated` - Computed from physics quantities
- `documented` - From code documentation
- `asserted` - User-provided, not verified

## Graph Data

AnalysisCodes with known COCOS are enriched via `COCOSMixin` attributes.
Query the graph for current state:

```cypher
MATCH (ac:AnalysisCode)
WHERE ac.cocos_native IS NOT NULL
RETURN ac.name, ac.cocos_native, ac.cocos_determination
```

## References

- Sauter, O. and Medvedev, S.Yu., "Tokamak coordinate conventions: COCOS",
  Computer Physics Communications, 184(2):293–302, February 2013.
  DOI: [10.1016/j.cpc.2012.09.010](https://doi.org/10.1016/j.cpc.2012.09.010)
