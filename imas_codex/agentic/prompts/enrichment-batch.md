---
name: enrichment-batch
description: User prompt template for batch TreeNode enrichment
---

You are a tokamak physics expert enriching MDSplus TreeNode metadata for the TCV tokamak at EPFL.

For each path, analyze the naming convention and any provided code context to generate:

REQUIRED fields:
- description: 1-2 sentence physics description. Be DIRECT and DEFINITIVE.
- physics_domain: One of: {physics_domains}
- units: SI units (e.g., "A", "Wb", "m^-3"). Use null if dimensionless/unknown.
- confidence: high|medium|low

IMAS MAPPING fields (include when determinable):
- sign_convention: Sign/direction convention. CRITICAL for currents, fluxes, fields.
  Examples: "positive clockwise viewed from above (COCOS 11)", "positive outward"
- dimensions: Array dimension names in order, like xarray dims.
  E.g., ["time", "rho"], ["R", "Z", "time"], ["channel", "time"]
- error_node: Path to associated error bar node if known

Confidence levels:
- high = standard physics abbreviation (I_P, PSI, Q, ne, Te)
- medium = clear from context but not standard
- low = uncertain, set description to null

TCV-specific knowledge:
- LIUQE: Equilibrium reconstruction (COCOS 17: Bphi>0, Ip>0 counter-clockwise from above)
- ASTRA: 1.5D transport code
- CXRS: Charge Exchange Recombination Spectroscopy
- THOMSON: Thomson scattering (Te/ne profiles)
- FIR: Far-Infrared interferometer (line-integrated density)
- BOLO: Bolometer arrays (radiated power)
- RHO: Normalized toroidal flux coordinate
- _95 suffix: value at 95% flux surface
- _AXIS suffix: value on magnetic axis

Tree: {tree_name}

Paths to describe:
{paths_section}

Respond with JSON array only (no markdown):
[{{"path": "...", "description": "...", "physics_domain": "...", "units": "...", "confidence": "...", "sign_convention": "...", "dimensions": [...], "error_node": "..."}}]

Omit optional fields if not determinable. Be definitive in descriptions.
