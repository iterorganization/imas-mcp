# Research: Geometric Quantity Taxonomy

**Date:** 2025-07-23
**Status:** Complete — findings feed into `dd-unified-classification.md` plan update

## Problem Statement

The `dd-unified-classification.md` plan splits `quantity` into `physical_quantity` + `geometric_quantity`, defining "geometric" via a 17-name leaf-name set (elongation, triangularity, minor_radius, etc.). This captures only 36 nodes (0.36% of quantities) and conflates plasma shape parameters (physics outputs) with machine geometry (engineering inputs).

The user's instinct: "geometries in my mind means coil positions, vessel outlines, lines of sight of diagnostics." Is this too narrow, or is it actually the correct definition?

## Findings

### Three Natural Groupings Within "quantity" Nodes

| Grouping | Count | Examples | Nature |
|----------|-------|----------|--------|
| Physical quantity | ~9,090 | T_e, n_e, j_tor, β, q, B_tor, pressure | Physics measurements and derived quantities |
| Hardware geometry | ~870 | coil r/z, vessel outline, LoS, aperture, detector position | Engineering inputs — machine and diagnostic structure |
| Plasma shape descriptor | ~33 | elongation, triangularity, squareness, aspect_ratio | Equilibrium-derived boundary coefficients |

### Plasma Shape Parameters Are Physics Outputs, Not Geometry

Elongation (κ), triangularity (δ), squareness, and tilt are **computed by equilibrium reconstruction codes** from magnetic measurements. They are:

- Dimensionless (unit = `-`)
- Derived from the separatrix contour (itself an equilibrium output)
- Key physics parameters affecting MHD stability and confinement
- Already assigned standard names via ISN's `physical_base` segment (e.g., `elongation_of_plasma_boundary`)

Minor radius and major radius have unit `m` but are still equilibrium outputs — they describe the plasma, not the machine.

These are **physical quantities with geometric character**, not geometry in the engineering/structural sense.

### Hardware Geometry Is Structurally Identifiable

Hardware geometry nodes cluster under specific DD subtree patterns:

| Path pattern | Count | What it captures |
|-------------|-------|------------------|
| `/geometry/` | ~282 | Coil geometry, iron core, optical element shapes |
| `/outline/` | ~75 | Wall, vessel, coil cross-sections |
| `/aperture/` | ~150 | Diagnostic aperture positions and dimensions |
| `/line_of_sight/` | ~80 | Diagnostic viewing geometry |
| `/position/` | ~200 | Gauge, probe, channel positions |
| `/detector/` | ~100 | Detector hardware dimensions |
| `/mirror/`, `/waveguide/`, `/launcher/`, `/antenna/` | ~80 | Heating/current drive hardware geometry |

All require a spatial unit (`m`, `rad`, `m^2`, `m^-1`, `deg`) to avoid false positives.

### IDS Distribution of Hardware Geometry

| IDS | HW geometry nodes | % of IDS quantities |
|-----|-------------------|---------------------|
| bolometer | 88 | 89% |
| camera_visible | 58 | 70% |
| pf_active | 48 | 55% |
| soft_x_rays | 41 | 56% |
| hard_x_rays | 41 | 53% |
| spectrometer_uv | 41 | 49% |
| spectrometer_visible | 80 | 45% |
| neutron_diagnostic | 76 | 42% |

Diagnostic IDS are dominated by hardware geometry — this is exactly the kind of structural information users need when setting up a diagnostic model.

### Exclusion Rules (False Positive Prevention)

Not all `/outline/` or `/position/` paths are hardware:

- `equilibrium/time_slice/boundary/outline/r` — plasma boundary contour (physics output, not hardware)
- `equilibrium/time_slice/boundary_separatrix/outline/z` — separatrix shape (same)
- `/grid/`, `/ggd/` paths — computational mesh (numerical artifact, not hardware)

Exclusion: paths containing `boundary`, `separatrix`, `grid`, `ggd` are NOT hardware geometry.

### "Other Length-Unit" Quantities

160 nodes have unit `m` but don't fit cleanly into hardware geometry:

- Curvature radii of optical elements (`x1_curvature`, `x2_curvature`)
- Reference major radius (`r0`) — a physics reference point
- Beam spot size, pixel size — instrument scale properties
- Debye length, skin depth — physics quantities with length units

These remain `physical_quantity`. The length unit alone does NOT make something "geometry."

## Recommendation

**Drop `geometric_quantity` entirely. Replace with `hardware_geometry`.**

| Aspect | Plan's `geometric_quantity` | Recommended `hardware_geometry` |
|--------|---------------------------|-------------------------------|
| Definition | 17 leaf-name plasma shape params | Machine/diagnostic structural positions |
| Count | 36 nodes (0.36%) | ~870 nodes (8.7%) |
| Detection | Leaf name matching | Structural path pattern + unit |
| Semantic | Ambiguous — "shape" vs "structure" | Clear — "engineering inputs" |
| SN grammar | physical_base (already works) | geometric_base (correct alignment) |
| User value | Low — niche equilibrium filter | High — "show me machine setup" |

### Revised NodeCategory Enum (8 values)

```
physical_quantity   (~9,090)  — Physics measurements and derived quantities
hardware_geometry   (  ~870)  — Machine and diagnostic structural positions
coordinate          ( 1,739)  — Independent variables / axes
structural          ( 7,395)  — Container / grouping nodes
identifier          (   243)  — Named type selectors
error               (31,281)  — Uncertainty fields
metadata            (10,715)  — Provenance, time references
constant            (     ?)  — Fixed parameters (future, if needed)
```

### Classifier Rules for `hardware_geometry`

```python
HARDWARE_GEOMETRY_PARENTS = frozenset({
    "geometry", "outline", "aperture", "line_of_sight",
    "sightline", "position", "detector", "mirror",
    "waveguide", "launcher", "antenna",
})

HARDWARE_GEOMETRY_EXCLUDE = frozenset({
    "boundary", "separatrix", "grid", "ggd",
})

SPATIAL_UNITS = frozenset({"m", "rad", "m^2", "m^-1", "m^-2", "m^3", "deg"})

def is_hardware_geometry(path: str, unit: str, ancestors: list[str]) -> bool:
    if unit not in SPATIAL_UNITS:
        return False
    if any(excl in path for excl in HARDWARE_GEOMETRY_EXCLUDE):
        return False
    return any(parent in HARDWARE_GEOMETRY_PARENTS
               for parent in ancestors)
```

This is a **structural classification** (parent path patterns), not a leaf-name classification. It's more robust because:

1. New DD paths under `/geometry/` automatically get classified correctly
2. No hardcoded name list to maintain
3. Exclusions prevent false positives from plasma boundary and grid paths

### Impact on Category Sets

```python
EMBEDDABLE_CATEGORIES  = frozenset({"physical_quantity", "hardware_geometry"})
SEARCHABLE_CATEGORIES  = frozenset({"physical_quantity", "hardware_geometry", "coordinate"})
SN_SOURCE_CATEGORIES   = frozenset({"physical_quantity", "hardware_geometry"})
ENRICHABLE_CATEGORIES  = frozenset({"physical_quantity", "hardware_geometry", "coordinate"})
```

### Impact on Standard Names

Both `physical_quantity` and `hardware_geometry` nodes are SN sources:

- `physical_quantity` → ISN `physical_base` segment (electron_temperature, plasma_current)
- `hardware_geometry` → ISN `geometric_base` segment (radial_position_of_pf_coil_outline)

The compose prompt should include `node_category` so the LLM selects the appropriate grammar branch.

### Impact on MCP Tools

`node_category` filter in `search_dd_paths` and `list_dd_paths` enables:

```
"Show me all hardware geometry in the bolometer IDS"
→ node_category=hardware_geometry, ids_filter=bolometer

"What physics quantities does equilibrium compute?"
→ node_category=physical_quantity, ids_filter=equilibrium
```

## ISN StandardNameKind Assessment

**Finding:** `kind=vector` is effectively unused — 0 instances in graph, all 20 "component_of" names use `kind=scalar`.

This is correct behavior: DD stores individual scalar components (B_r, B_z, B_phi), not vector-valued fields. The `vector` kind serves a grammatical validation purpose (enforcing component segment presence) but is never needed in practice because IMAS doesn't store multi-component vector quantities as single paths.

**Recommendation:** Keep scalar/vector/metadata as-is. No ISN changes needed. Richer classification belongs in imas-codex's `node_category`, not in ISN's `Kind` enum.

## Enrichment Model Benchmark

All current DD enrichment was done with `google/gemini-3.1-flash-lite-preview` (cheapest model).

### 5-Node Pilot Results

| Model | Time | Cost | Quality |
|-------|------|------|---------|
| gemini-3.1-flash-lite | 2.6s | $0.019 | Adequate — generic, thin descriptions |
| claude-sonnet-4.6 | 14.4s | $0.021 | Excellent — precise physics context, practical |
| claude-opus-4.6 | 25.7s | $0.021 | Excellent — most detailed, includes definitions |

### Quality Comparison: triangularity

- **flash-lite:** "measures the degree of plasma indentation" — vague
- **sonnet:** "quantifying the D-shape asymmetry... key shaping parameter influencing MHD stability, confinement, and power exhaust" — precise
- **opus:** "defined as the horizontal offset of the maximum vertical extent from the geometric axis normalized by the minor radius" — includes formula

### Cost Projection

| Scenario | Nodes | Cost | Time |
|----------|-------|------|------|
| Sonnet (all enrichable) | 11,699 | ~$50 | ~3 hours |
| Flash-lite (all enrichable) | 11,699 | ~$45 | ~1 hour |
| Marginal cost of sonnet | — | ~$5 | ~2 hours |

**Recommendation:** Switch to `claude-sonnet-4.6` for DD enrichment. The $5 marginal cost buys dramatically better descriptions that directly improve:

1. Embedding quality (better descriptions → better vectors → better search)
2. Standard name generation (enriched descriptions are compose prompt context)
3. MCP tool responses (descriptions surfaced to users)

Opus 4.6 produces marginally better descriptions but at 2× the latency. Not justified given the quality/cost of sonnet is already excellent.
