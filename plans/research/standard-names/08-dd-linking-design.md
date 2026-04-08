# Design: Linking IMAS Data Dictionary Paths to Standard Names

**Status:** Proposal  
**Date:** 2025-07-18  
**DD Version Analyzed:** 4.1.0  

---

## Executive Summary

Standard names currently have an `ids_paths` field (`list[str]`, default empty) but no entries populate it. This design establishes the mapping model, schema decisions, privacy boundaries, and phased rollout strategy for linking DD paths to standard names.

**Core principle:** Standard names represent *physics concepts*; DD paths represent *storage locations*. The mapping is many-to-many: one standard name maps to multiple DD paths (same concept across IDSs), and occasionally one DD path relates to multiple standard names (a profile field and its derived quantities).

---

## 1. Mapping Model

### 1.1 Relationship Cardinality

**One standard name → many DD paths** (dominant pattern):

| Standard Name | DD Paths | IDSs Spanned |
|---|---|---|
| `electron_temperature` | 10 paths | 6 IDSs (core_profiles, edge_profiles, plasma_profiles, mhd, turbulence, core_instant_changes) |
| `poloidal_flux` (concept) | 20+ paths | 12+ IDSs |
| `toroidal_magnetic_field` (concept) | 50 paths | 28 IDSs |
| `plasma_current` (concept) | 38 paths | 7 IDSs |

**Many standard names → one DD path** (rare):
- A DD path like `equilibrium/time_slice/profiles_1d/psi` maps only to the poloidal flux standard name
- Edge cases: `summary/volume_average/t_e` could relate to both `electron_temperature` and a volume-averaged variant

### 1.2 What Goes in `ids_paths`

The `ids_paths` field stores **exact-equivalence** mappings only — DD paths where the physical quantity is the same concept as the standard name, just stored in a different IDS context.

**Include:**
- `core_profiles/profiles_1d/electrons/temperature` → `electron_temperature` (same quantity, core context)
- `edge_profiles/profiles_1d/electrons/temperature` → `electron_temperature` (same quantity, edge context)
- `plasma_profiles/ggd/electrons/temperature` → `electron_temperature` (same quantity, GGD representation)

**Exclude:**
- `summary/volume_average/t_e` — this is a *derived aggregation*, not the same quantity
- `core_profiles/profiles_1d/electrons/temperature_fit` — this is fit metadata, not the temperature itself
- Error fields (`_error_upper`, `_error_lower`) — tracked separately via DD error field tools

### 1.3 Clusters as Discovery Mechanism, Not Contract

DD semantic clusters (e.g., the "electron temperature" global cluster) group paths representing the same concept across IDSs. This aligns naturally with standard names, but **clusters are not the authoritative mapping**:

- Clusters are ML-derived artifacts whose labels and membership may change
- Clusters sometimes conflate local vs aggregated quantities, or profile vs summary data
- Clusters can seed candidate mappings, but human/tool review assigns the final `ids_paths`

**Workflow:** `cluster → candidate DD paths → review/validate → ids_paths`

---

## 2. Schema Extension Proposal

### 2.1 Keep `ids_paths` As-Is

The existing field is well-designed:

```python
# field_types.py (already exists)
IdsPaths = Annotated[
    list[str],
    Field(
        description="IMAS Data Dictionary paths that map to this standard name. "
        "Format: 'ids_name/path/to/leaf' (e.g., 'equilibrium/time_slice/profiles_1d/elongation'). "
        "Paths can be validated via check_imas_paths tool.",
    ),
]

# models.py (already exists)
ids_paths: IdsPaths = Field(default_factory=list)
```

**No type change needed.** The flat `list[str]` is correct for exact-equivalence paths.

### 2.2 Do Not Add `dd_clusters` Inline

Clusters are unstable derived artifacts. Storing cluster labels in YAML couples standard name entries to clustering algorithm output, causing unnecessary YAML churn when clusters are rebuilt.

**Instead:** Use clusters programmatically in mapping tools and workflows. The graph already stores `(IMASNode)-[:IN_CLUSTER]->(IMASSemanticCluster)` relationships for this purpose.

### 2.3 DD Version Tracking: Mapping Manifest, Not Per-Entry

A per-entry `dd_version` field implies all linked paths were validated together against one release — misleading when mappings are updated incrementally.

**Recommended approach:** A repository-level mapping manifest:

```yaml
# catalog-root/dd_mapping_manifest.yml
dd_version: "4.1.0"
validated_at: "2025-07-18"
validation_tool: "check_imas_paths"
entry_count: 309
mapped_count: 0  # entries with non-empty ids_paths
coverage_stats:
  total_dd_leaf_paths: ~30000
  mapped_paths: 0
  unmapped_physics_paths: ~5000  # estimated
```

This manifest is updated when bulk revalidation runs, providing a single point of truth for "which DD version are our mappings valid against?"

### 2.4 Structured Mapping Sidecar (Phase 2)

For non-exact relations that `ids_paths` cannot capture, a separate mapping artifact (generated, not hand-curated):

```yaml
# Generated file: catalog-root/.mapping/electron_temperature.yml
standard_name: electron_temperature
exact_paths:  # mirrors ids_paths (generated from YAML)
  - core_profiles/profiles_1d/electrons/temperature
  - edge_profiles/profiles_1d/electrons/temperature
  - plasma_profiles/profiles_1d/electrons/temperature
  # ... 7 more
related_mappings:
  - path: summary/volume_average/t_e
    relation: aggregated
    target_name: volume_averaged_electron_temperature  # different SN
  - path: core_profiles/profiles_1d/electrons/temperature_fit
    relation: fit_metadata
    target_name: null  # no SN needed
source_clusters:
  - label: "electron temperature"
    scope: global
    dd_version: "4.1.0"
```

This is a **generated artifact** (from YAML + graph queries), not a hand-maintained file. It provides richer context for tools and documentation without polluting the core catalog schema.

---

## 3. Relationship Types to Track

### 3.1 In `ids_paths` (authoritative, in YAML)

| Type | Description | Example |
|---|---|---|
| **Exact equivalence** | Same physical quantity in different IDS | `core_profiles/.../electrons/temperature` and `edge_profiles/.../electrons/temperature` |

Only this type belongs in `ids_paths`. All paths in the list are interchangeable representations of the same concept.

### 3.2 In Graph Edges (derived, generated)

| Relationship | From | To | Purpose |
|---|---|---|---|
| `MAPS_TO_DD` | StandardName | IMASNode | Direct exact mapping |
| `RELATED_DD` | StandardName | IMASNode | Non-exact relation (aggregated, diagnostic-specific) |
| `IN_CLUSTER` | IMASNode | IMASSemanticCluster | Existing cluster membership |

Graph edges are **generated from YAML** during catalog build. YAML is the single source of truth.

### 3.3 Path Classification Rules

When populating `ids_paths`, apply these rules:

1. **Same leaf name, different IDS context** → exact (include)
   - `core_profiles/profiles_1d/electrons/temperature` ✓
   - `edge_profiles/ggd/electrons/temperature` ✓

2. **Aggregated/reduced variant** → different standard name
   - `summary/volume_average/t_e` → `volume_averaged_electron_temperature`
   - `summary/local/magnetic_axis/t_e` → `electron_temperature_at_magnetic_axis`

3. **Fit/reconstruction metadata** → exclude
   - `*_fit`, `*_fit/measured`, `*_fit/reconstructed`

4. **Error/uncertainty companions** → exclude (use DD error field tools)
   - `*_error_upper`, `*_error_lower`, `*_error_index`

5. **Grid/coordinate paths** → separate standard names for grid concepts
   - `*/grid/psi` → standard name for the grid coordinate, not the physics quantity

---

## 4. Privacy and Visibility Design

### 4.1 Requirement Clarification

"DD path info should not be on user-facing websites but should be stored."

This means: **not rendered on documentation sites**, not "cryptographically hidden." The YAML files are in a Git repository and are accessible to anyone with repo access.

### 4.2 Implementation

| Layer | `ids_paths` Visible? | Mechanism |
|---|---|---|
| **YAML files** | Yes | Source of truth, version-controlled |
| **MCP tools** (`fetch_standard_names`) | Yes | Internal tooling needs full data |
| **Documentation site** (mkdocs) | No | Exclude from rendered output |
| **Public API** (if any) | Configurable | Allowlist of public fields |
| **Graph database** | Yes | `MAPS_TO_DD` edges for cross-referencing |

### 4.3 Implementation Notes

- The HTML renderer (`rendering/html.py`) already excludes `ids_paths` from output
- `fetch_standard_names` currently does **not** include `ids_paths` in its output — this needs to be fixed for MCP tool consumers
- Add an explicit **public field allowlist** for any future API layer rather than relying on per-renderer deny lists
- Test coverage: add a test asserting that public serializers exclude `ids_paths`

---

## 5. Cluster-Based Naming Strategy

### 5.1 Clusters as Bootstrap, Not Authority

Global semantic clusters provide an efficient way to identify candidate standard names:

```
Cluster: "electron temperature" (global, cross-IDS)
  → 10 DD paths across 6 IDSs
  → Maps cleanly to standard name: electron_temperature
  → All 10 paths go into ids_paths
```

### 5.2 When Clusters Match Standard Names Well

| Cluster Pattern | Standard Name Strategy |
|---|---|
| Global cluster, single physics concept | One standard name, all cluster paths in `ids_paths` |
| Cluster with local + aggregated variants | Multiple standard names (base + qualified variants) |
| COCOS-dependent cluster (e.g., `psi_like`) | Shared COCOS annotation, individual standard names per concept |
| Diagnostic-specific cluster | Standard name per diagnostic quantity |

### 5.3 When Clusters Don't Map Cleanly

- **Structural clusters** (grid subsets, error fields) → no standard name
- **Mixed-concept clusters** (momentum flux components) → multiple standard names
- **Summary/aggregation clusters** → separate standard names with position/reduction qualifiers
- **IDS-scoped clusters** → may duplicate a global cluster; prefer global

### 5.4 Cluster-Driven Batch Workflow

```
1. Query all global clusters
2. For each cluster:
   a. Check if a matching standard name exists (by label similarity)
   b. If yes: validate and populate ids_paths
   c. If no: evaluate whether a new standard name is needed
   d. Filter out structural/metadata clusters
3. Validate all ids_paths against current DD version
4. Generate mapping sidecar artifacts
```

---

## 6. Scale Assessment

### 6.1 DD Inventory (v4.1.0)

| Metric | Count |
|---|---|
| Total IDSs | 87 |
| Total DD paths | 61,366 |
| Estimated leaf physics/geometry paths | ~15,000–20,000 |
| Estimated global clusters (physics-relevant) | ~500–800 |
| Current standard names | 309 |

### 6.2 Coverage Phases

| Phase | Target | Estimated Names | Focus |
|---|---|---|---|
| **Phase 1** | Exact mappings for existing 309 names | 309 | Populate `ids_paths` for current catalog |
| **Phase 2** | High-value cluster coverage | ~450–550 | Add names for major unmapped physics concepts |
| **Phase 3** | Diagnostic + qualified variants | ~600–900 | Position-qualified, aggregated, species-specific |
| **Full coverage** | All physics-relevant DD concepts | ~800–1200 | Including edge cases, rare diagnostics |

### 6.3 Why Full Coverage Exceeds Initial Estimates

The initial estimate of 400–600 is likely too low because:
- **Diagnostic-specific quantities**: bolometer power, Thomson scattering profiles, ECE channels — each diagnostic family adds 5–15 names
- **Geometry-qualified variants**: `_at_magnetic_axis`, `_at_separatrix`, `_at_midplane` — each base quantity multiplied by ~5 positions
- **Species-qualified variants**: electron, ion, deuterium, tritium, helium, impurity — ~6× multiplier for particle-dependent quantities
- **Process-qualified variants**: ohmic, bootstrap, neoclassical, turbulent — ~4× for transport quantities
- **Component variants**: radial, toroidal, poloidal, parallel — ~4× for vector quantities

Not all combinations exist, but the combinatorial space is larger than concept-query intuition suggests.

---

## 7. Implementation Roadmap

### 7.1 Immediate (No Schema Changes)

1. **Populate `ids_paths` for existing entries**: Use cluster-based discovery + `check_imas_paths` validation
2. **Fix `fetch_standard_names`**: Include `ids_paths` in MCP tool output
3. **Add validation**: MCP tool or CLI that checks all `ids_paths` against current DD version
4. **Add tests**: Ensure public renderers exclude `ids_paths`

### 7.2 Near-Term (Minor Schema Addition)

1. **Add mapping manifest**: Repository-level `dd_mapping_manifest.yml` tracking DD version and coverage stats
2. **Generate graph edges**: `(StandardName)-[:MAPS_TO_DD]->(IMASNode)` from YAML during catalog build
3. **Add reverse-lookup MCP tool**: "Given a DD path, which standard name(s) map to it?"

### 7.3 Future (Phase 2+)

1. **Structured mapping sidecar**: Generated artifacts with relation types for non-exact mappings
2. **Batch name generation**: Cluster-driven workflow for identifying gaps and generating candidates
3. **Migration tooling**: When DD version changes, detect renamed/moved paths and update `ids_paths`
4. **Coverage dashboard**: Track what percentage of physics-relevant DD paths have standard name mappings

---

## 8. Source of Truth Policy

| Artifact | Role | Authoritative? |
|---|---|---|
| **YAML entry files** (with `ids_paths`) | Primary store | Yes — single source of truth |
| **Mapping manifest** | Version/coverage metadata | Yes — for DD version info |
| **Graph edges** (`MAPS_TO_DD`) | Query/discovery index | No — generated from YAML |
| **Mapping sidecar** (Phase 2) | Rich relation context | No — generated from YAML + graph |
| **Semantic clusters** | Discovery aid | No — unstable derived artifacts |

**Rule:** Never edit graph edges or sidecar files directly. All mapping changes flow through YAML → build pipeline → derived artifacts.

---

## 9. Open Questions

1. **Granularity of GGD vs 1D paths**: Should `edge_profiles/ggd/electrons/temperature` and `edge_profiles/profiles_1d/electrons/temperature` both map to `electron_temperature`, or does the GGD representation warrant a separate name?
   - **Recommendation:** Same name. The representation format (GGD vs 1D) is a storage detail, not a physics distinction.

2. **Summary IDS paths**: The `summary` IDS contains scalar reductions of profile quantities. Should `summary/volume_average/t_e` map to `electron_temperature` or to a qualified variant?
   - **Recommendation:** Qualified variant (`volume_averaged_electron_temperature`) since it's a different physical quantity (a spatial average vs a profile).

3. **Coordinate grid paths**: `core_profiles/profiles_1d/grid/psi` is a coordinate grid, not a measurement. Should it get a standard name?
   - **Recommendation:** Yes, but as a coordinate-type name (e.g., `poloidal_flux` as a coordinate concept), not mixed with the measurement concept.

4. **COCOS-sensitive paths**: 35 `b0_like` fields, 38 `ip_like` fields, 14 `q_like` fields. Should COCOS sensitivity be annotated on standard names?
   - **Recommendation:** Not in Phase 1. COCOS is a DD-level concern. Standard names describe concepts, not sign conventions. The graph already tracks COCOS via `cocos_label_transformation` on `IMASNode`.

---

## Appendix: Representative Mappings

### Example: `electron_temperature`

```yaml
name: electron_temperature
kind: scalar
unit: eV
ids_paths:
  - core_profiles/profiles_1d/electrons/temperature
  - edge_profiles/profiles_1d/electrons/temperature
  - edge_profiles/ggd/electrons/temperature
  - edge_profiles/ggd_fast/electrons/temperature
  - plasma_profiles/profiles_1d/electrons/temperature
  - plasma_profiles/ggd/electrons/temperature
  - plasma_profiles/ggd_fast/electrons/temperature
  - core_instant_changes/change/profiles_1d/electrons/temperature
  - mhd/ggd/electrons/temperature
  - turbulence/profiles_2d/electrons/temperature
```

### Example: `poloidal_flux` (if created)

```yaml
name: poloidal_flux
kind: scalar
unit: Wb
ids_paths:
  - equilibrium/time_slice/profiles_1d/psi
  - core_profiles/profiles_1d/grid/psi
  - edge_profiles/profiles_1d/grid/psi
  - plasma_profiles/profiles_1d/grid/psi
  - core_sources/source/profiles_1d/grid/psi
  - core_transport/model/profiles_1d/grid_flux/psi
  - core_transport/model/profiles_1d/grid_d/psi
  - core_transport/model/profiles_1d/grid_v/psi
  - disruption/profiles_1d/grid/psi
  - distribution_sources/source/profiles_1d/grid/psi
  - distributions/distribution/profiles_1d/grid/psi
  - distributions/distribution/profiles_2d/grid/psi
  # ... 8+ more paths
```

### Example: `safety_factor` (if created)

```yaml
name: safety_factor
kind: scalar
unit: "1"
ids_paths:
  - equilibrium/time_slice/profiles_1d/q
  - core_profiles/profiles_1d/q
  - edge_profiles/profiles_1d/q
  - plasma_profiles/profiles_1d/q
  - core_instant_changes/change/profiles_1d/q
```
