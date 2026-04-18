## Controlled Tag Vocabulary (`tags` field — SECONDARY ONLY)

The `tags` field accepts **secondary tags only**. Physics domain (``edge-physics``,
``transport``, ``mhd``, ``equilibrium``, ``magnetics``, ``core-physics``, etc.)
belongs in the separate `physics_domain` field, **never** in `tags`.

Do not invent tags. Non-conforming tags are dropped silently by the pipeline.

### Allowed `tags` values (secondary — cross-cutting descriptors)

`auxiliary-heated`, `benchmark-quantity`, `broadband-fluctuation`, `calibrated`,
`cartesian-coordinates`, `coherent-mode`, `current-drive-efficiency`,
`cylindrical-coordinates`, `data-archival`, `dc-component`, `derived`,
`disruption-prediction`, `divertor-configuration`, `energy-balance`,
`equilibrium-reconstruction`, `field-aligned`, `flux-coordinates`,
`flux-surface-average`, `fourier-component`, `global-quantity`,
`heating-deposition`, `high-confinement-mode`, `high-frequency`,
`impurity-transport`, `integrated-modeling`, `inverse-problem`,
`limiter-configuration`, `line-integrated`, `local-measurement`,
`long-timescale`, `low-confinement-mode`, `low-frequency`, `machine-learning`,
`measured`, `mhd-stability-analysis`, `momentum-balance`, `ohmic-heating`,
`particle-balance`, `performance-metric`, `post-shot-analysis`,
`published-result`, `raw-data`, `real-time`, `real-time-control`,
`reconstructed`, `scenario-planning`, `simulated`, `spatial-profile`,
`steady-state`, `synthetic-diagnostic`, `time-dependent`, `transient`,
`transport-modeling`, `uncertainty-quantification`, `validated`,
`volume-average`.

### Common mistakes (rejected by validator)

- `tags: ["edge-physics", "measured"]` — WRONG: `edge-physics` is a primary tag.
  Put it in `physics_domain`. Tags become `["measured"]`.
- `tags: ["equilibrium", "spatial-profile"]` — WRONG: use
  `physics_domain: equilibrium` and `tags: ["spatial-profile"]`.

