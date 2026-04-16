# SN Vector Hierarchy & Kind Assessment

## Problem Statement

The standard name pipeline generates only `kind=scalar` names because it processes
individual DD paths one at a time. This misses vector physics concepts — `magnetic_field`,
`electric_field`, `ion_velocity` — that exist as the parent grouping of their scalar
components across coordinate systems and data sources.

117 scalar component SNs already exist (e.g., `toroidal_component_of_magnetic_field`)
implying 37 core physics vector parents that don't exist yet. Standard names are
**source-independent** — a vector SN like `magnetic_field` doesn't need a DD path
mapping; it represents the physics concept that links components across DD paths,
facility signals, and simulation codes.

Separately, `kind=metadata` has no pipeline support and no demonstrated use case.
Two orphan metadata SNs exist in the graph. This plan resolves both the vector
gap and the metadata question.

## Research Findings

### Vector SNs: ISN Already Supports Them

Verified experimentally — ISN requires **no changes**:

| Check | Result |
|-------|--------|
| `StandardNameVectorEntry(name="magnetic_field", kind="vector", ...)` | Pydantic validates ✓ |
| `parse_standard_name("magnetic_field")` | `physical_base='magnetic_field'` ✓ |
| `compose_standard_name(parse_standard_name("magnetic_field"))` | Round-trip `"magnetic_field"` ✓ |
| VectorEntry has `magnitude` property | `"magnitude_of_magnetic_field"` auto-derived ✓ |
| Grammar: component segment on children | `{coord}_component_of_{parent}` ✓ |

### ISN Model Differences by Kind

| Field | Scalar | Vector | Metadata |
|-------|--------|--------|----------|
| name, description, documentation | ✓ | ✓ | ✓ |
| unit | Required | Required | **Absent** |
| provenance (operator/reduction) | Optional | Optional | **Forbidden** |
| physics_domain | Required | Required | Required |
| `magnitude` property | — | Auto: `magnitude_of_{name}` | — |
| Grammar parse | Identical | Identical | Identical |
| Component support | Via `_component_of_` | Decomposes into scalar components | Semantically forbidden |

**Key:** Vector and scalar are structurally identical except for the `magnitude` property.
Metadata is structurally different — no unit, no provenance.

### 37 Core Physics Vector Concepts

Identified by parsing existing `*_component_of_*` scalar SNs. Interpolation/numerical
artifacts (23 additional) excluded.

| Vector concept | Components | Coordinate bases |
|----------------|------------|-----------------|
| magnetic_field | 3 | toroidal, poloidal, parallel |
| electric_field | 5 | radial, toroidal, poloidal, parallel, diamagnetic |
| ion_velocity | 3 | parallel, radial, toroidal |
| electron_velocity | 3 | parallel, radial, toroidal |
| neutral_velocity | 3 | parallel, radial, toroidal |
| current_density | 2 | diamagnetic, toroidal |
| plasma_current_density | 3 | parallel, toroidal, vertical |
| magnetic_vector_potential | 2 | radial, vertical |
| exb_drift_velocity | 3 | diamagnetic, parallel, radial |
| fast_ion_pressure | 2 | parallel, perpendicular |
| diamagnetic_current_density | 2 | poloidal, toroidal |
| pfirsch_schlueter_current_density | 2 | poloidal, toroidal |
| magnetization | 2 | radial, vertical |
| ... | ... | (24 more, see research report) |

9 coordinate bases represented: toroidal (38), radial (30), poloidal (19),
parallel (18), diamagnetic (14), vertical (11), z (5), perpendicular (4), x (1).

### Full Hierarchy Per Vector

Each vector parent creates a complete family:

```
magnetic_field                           (kind=vector, unit=T)     ← NEW
├── magnitude_of_magnetic_field          (kind=scalar, unit=T)     ← NEW
├── toroidal_component_of_magnetic_field (kind=scalar, unit=T)     ← EXISTS
├── poloidal_component_of_magnetic_field (kind=scalar, unit=T)     ← EXISTS
└── parallel_component_of_magnetic_field (kind=scalar, unit=T)     ← EXISTS
```

Scope: **74 new SNs** (37 vector parents + 37 magnitude scalars) + **117 linked** existing
component scalars.

### Metadata Kind Assessment

**Keep in ISN, but do not invest in pipeline support now.**

| Aspect | Finding |
|--------|---------|
| Existing metadata SNs | 2 orphans (`atoms_per_molecule`, `grid_object_boundary_neighbour_index`) |
| Potential sources | 243 identifier DD nodes (types, materials, classifications) |
| Pipeline support | None — `SN_SOURCE_CATEGORIES` excludes identifier/metadata nodes |
| Use cases | Ontology alignment, type vocabulary standardization |
| User demand | None demonstrated |
| Cost to keep | Zero — 1 enum value, 1 model class in ISN |
| Cost to remove | Breaking change for no gain |

**Decision: Keep `kind=metadata` as reserved.** Clean up the 2 orphan metadata SNs
(delete or reclassify). Document metadata as "future ontology alignment capability."
Do not build metadata SN generation infrastructure until a concrete use case emerges.

**Sanctioned future path for metadata SNs:**
1. **Now:** manual/reference catalog import via `sn import` (for curated metadata entries)
2. **Future:** optional allowlisted metadata pipeline mode that processes identifier DD
   nodes (`node_category=identifier`). Gated behind a `--include-metadata` flag in
   `sn generate`. Not built until a concrete use case materializes.
3. **Key difference:** metadata SNs have no `unit` or `provenance` — they're definitional.
   The existing EXTRACT filter (`SN_SOURCE_CATEGORIES`) must be kept as-is for default
   runs; the allowlist mode would add `"identifier"` to the source set.

**Why not remove it?** Metadata has a coherent definition: "definitional entries without
units or provenance." The ISN model enforces this structurally (no `unit` field, no
`provenance` field). If we ever need to name categorical concepts (confinement modes,
grid types, species classifications), the infrastructure exists. Removing it costs more
than keeping it.

## Design

### Graph Schema Changes

Add to `imas_codex/schemas/standard_name.yaml`:

```yaml
# New relationship: vector → scalar component children
has_component:
  description: >-
    Links a vector StandardName to its scalar component children.
    E.g., (magnetic_field)-[:HAS_COMPONENT]->(toroidal_component_of_magnetic_field)
  range: StandardName
  annotations:
    relationship_type: HAS_COMPONENT

# New relationship: vector → scalar magnitude companion
has_magnitude:
  description: >-
    Links a vector StandardName to its magnitude scalar companion.
    Magnitude is NOT a component — it is the norm of the vector.
    E.g., (magnetic_field)-[:HAS_MAGNITUDE]->(magnitude_of_magnetic_field)
  range: StandardName
  annotations:
    relationship_type: HAS_MAGNITUDE
```

**Relationships created per vector parent:**
- `(magnetic_field)-[:HAS_COMPONENT]->(toroidal_component_of_...)` (existing scalar)
- `(magnetic_field)-[:HAS_COMPONENT]->(poloidal_component_of_...)` (existing scalar)
- `(magnetic_field)-[:HAS_MAGNITUDE]->(magnitude_of_magnetic_field)` (new scalar)

**Why two relationship types?** Magnitude is the *norm* of a vector, not a coordinate
component. Treating it as a component would break semantic queries ("list all coordinate
decompositions of magnetic_field") and violate the physics: magnitude is derived from
ALL components, not a projection onto a single coordinate axis.

### Pipeline: Component Grouping Pass

A new worker in the SN pipeline that runs **after consolidation, before embedding**.
(In the current code, names are persisted during COMPOSE; PERSIST handles embedding.
GROUP runs post-consolidation when all scalar names are final.)

```
EXTRACT → COMPOSE → VALIDATE → CONSOLIDATE → **GROUP** → PERSIST (embed)
```

The GROUP phase:

1. **Query** all existing `*_component_of_*` scalar SNs from the graph
2. **Parse** each name via ISN grammar → extract full parsed segments
3. **Reconstruct parent signature** by removing only the `component` segment from the
   parsed name. This preserves ALL qualifiers (subject, process, position, etc.).
   E.g., `toroidal_component_of_impurity_species_velocity_at_inner_midplane`
   → parent = `impurity_species_velocity_at_inner_midplane` (not bare `velocity`)
4. **Group** by reconstructed parent signature → candidate vector parents
5. **Eligibility check**: exclude non-vector groupings (see below)
6. **Create** vector parent SN (inherits unit from components, must be consistent)
7. **Create** magnitude scalar SN (`magnitude_of_{parent}`) — only for eligible vectors
8. **Validate** all new names via ISN grammar round-trip
9. **Link** vector → components via `HAS_COMPONENT`, vector → magnitude via `HAS_MAGNITUDE`

**Naming is deterministic, zero-LLM.** The vector parent name IS the reconstructed parent
signature. The magnitude name IS `magnitude_of_{parent}`. LLM is not needed for naming, kind,
or unit — only for description and documentation enrichment (via `sn enrich`).

Metadata fields (`physics_domain`, `tags`) are deterministic-from-children:
- `physics_domain` = majority vote from components (all should agree for true vectors)
- `tags` = union of component tags
- `links` = empty (populated during enrichment)

### Grouping: Full Parent Signature, Not Just physical_base

**Critical:** grouping by `physical_base` alone would wrongly collapse qualified names:

```
toroidal_component_of_magnetic_field              → parent: magnetic_field
toroidal_component_of_magnetic_field_at_magnetic_axis → parent: magnetic_field_at_magnetic_axis
toroidal_component_of_impurity_species_velocity_at_inner_midplane → parent: impurity_species_velocity_at_inner_midplane
```

These are THREE distinct vector parents, not one. The grouping key is the full recomposed
name with the component segment stripped — preserving subject, process, position, and all
other qualifiers. ISN's `parse_standard_name()` returns all segments; we remove only
`component` and `compose_standard_name()` the remainder.

### Vector Eligibility Check

Not every `_component_of_` name implies a true vector. Exclusion criteria:

1. **Interpolation/numerical artifacts:** parent contains `interpolation` or `finite_element`
2. **Tensor/anisotropy indicators:** parent has exactly 2 components with bases
   `parallel`+`perpendicular` and represents a pressure/temperature/viscosity (these are
   diagonal tensor elements, not vector components). Check: if all component bases are
   {parallel, perpendicular} AND the physical_base is in TENSOR_CANDIDATES, skip.
3. **Single-component groups:** if only 1 component exists, the parent may not be a genuine
   vector. Flag for manual review rather than auto-creating.

```python
TENSOR_CANDIDATES = {"pressure", "temperature", "viscosity", "diffusivity"}
EXCLUDE_PARENTS = {"interpolation", "finite_element", "fitting_weight"}

def is_eligible_vector(parent_name: str, components: list[str], bases: set[str]) -> bool:
    """Check if a grouped parent is a true physics vector."""
    if any(exc in parent_name for exc in EXCLUDE_PARENTS):
        return False
    if bases == {"parallel", "perpendicular"} and any(
        t in parent_name for t in TENSOR_CANDIDATES
    ):
        return False  # Anisotropic tensor, not vector
    if len(components) < 2:
        return False  # Single component — not enough evidence
    return True
```

**Golden test list:** The exclusion rules and the resulting 37 (or adjusted) core vector
set must be captured as a parametrized test fixture, not prose. The test enumerates ALL
`_component_of_` names, runs the grouping + eligibility logic, and asserts exact counts
and parent names against a frozen golden set.

### Unit Consistency Rule

A vector parent's unit must match all its components. If components have inconsistent
units (shouldn't happen for real physics vectors, but must be checked), flag for review
rather than auto-creating.

```python
component_units = {sn.unit for sn in components}
if len(component_units) != 1:
    log.warning(f"Skipping {parent}: inconsistent units {component_units}")
    continue
vector_unit = component_units.pop()
```

### MCP Tool Enhancements

`search_standard_names` should surface vector hierarchy:

```
Query: "magnetic field"
Result:
  magnetic_field (vector, T)
    Components: toroidal, poloidal, parallel
    Magnitude: magnitude_of_magnetic_field
    Sources: 12 DD paths, 3 TCV signals, 2 JET signals
```

Implementation: when a result has `kind=vector`, follow `HAS_COMPONENT` and `HAS_MAGNITUDE`
to list children. When a result is a scalar component, mention its vector parent.

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| **Catalog round-trip loss:** `sn publish` / `sn import` would lose graph-only `HAS_COMPONENT`/`HAS_MAGNITUDE` edges | GROUP is idempotent — re-run after import to reconstruct edges. Document this in `sn import` help text. |
| **Source-link semantics:** Vector parents are source-independent; DD paths/signals should NOT auto-attach to them | Vector parents get NO `FROM_DD_PATH` or `FROM_SIGNAL` relationships. Their provenance is "grouped from children." A new `provenance_method=grouped` field distinguishes them. |
| **Plan consistency:** This supersedes `isn-standard-name-kind.md` vector assessment | ISN boundary preserved: ISN unchanged, imas-codex owns grouping/storage/tooling. Cross-reference added. |
| **False vector detection:** Some `_component_of_` names may represent tensor/anisotropy, not vector | Eligibility check (see above) + golden test fixture + manual review for edge cases |

### Orphan Cleanup

Delete or reclassify the 2 existing metadata SNs:

- `atoms_per_molecule` — no DD mapping (orphan). **Delete.**
- `grid_object_boundary_neighbour_index` — maps to structural DD node. **Delete.**

Both were generated by early pipeline runs before proper classification.

## Implementation Phases

### Phase 1: Schema & Grouping Logic

- Add `HAS_COMPONENT` relationship to SN schema
- Add `HAS_COMPONENT` and `HAS_MAGNITUDE` relationships to SN schema
- Implement component grouping function:
  - Parse `_component_of_` SNs → reconstruct full parent signature (not just physical_base)
  - Group by reconstructed parent, validate unit consistency
  - Run vector eligibility check (tensor/anisotropy exclusion)
  - Generate vector parent + magnitude names for eligible groups only
  - ISN grammar validation for all new names
- TDD tests: golden test fixture with ALL `_component_of_` names → exact parent set
- TDD tests: eligibility edge cases (parallel+perpendicular pressure → tensor, skip)

### Phase 2: Pipeline Integration

- Add GROUP worker to SN pipeline DAG (post-consolidation, pre-embedding)
- Wire into `sn generate` CLI (new `--vectors` flag or automatic after scalar pass)
- Persist vector SNs, HAS_COMPONENT and HAS_MAGNITUDE relationships
- Metadata fields from children: physics_domain (majority), tags (union)
- Generate descriptions/documentation via LLM enrichment (`sn enrich`)

### Phase 3: MCP Tool Integration

- Surface vector hierarchy in `search_standard_names` results
- Show `HAS_COMPONENT` children and `HAS_MAGNITUDE` companion for vector results
- Show vector parent for scalar component results
- Add `kind` filter to MCP tools (already planned in dd-unified-classification)
- Enable "find all components of magnetic_field" type queries

### Phase 4: Orphan Cleanup

- Delete 2 metadata orphan SNs
- Document metadata kind as reserved for future use
- Document GROUP idempotency requirement for catalog round-trip

## ISN Changes Required

**None.** ISN grammar, models, and validation all support vector names today.
The only changes are in imas-codex's pipeline and graph schema.

Future ISN enhancement (not blocking): structural validation that checks whether
a vector SN's `magnitude_of_{name}` companion exists in the catalog. This validation
already exists in ISN but is optional (non-blocking check).

## Relationship to Other Plans

| Plan | Interaction |
|------|-------------|
| `dd-unified-classification.md` | Provides `node_category` labels that the SN pipeline uses for source filtering. Geometry nodes → geometry SNs. Vector grouping is orthogonal. |
| `isn-standard-name-kind.md` | This plan **supersedes** the ISN kind assessment. Conclusion changed from "keep as-is, vector unused" to "keep as-is, vector has concrete near-term value via pipeline grouping." ISN boundary preserved: ISN unchanged, imas-codex owns grouping/storage/tooling. |
| `sn-bootstrap-loop.md` | Vector parents could participate in the bootstrap quality loop if needed. |

## Cost Estimate

| Item | Count | Cost |
|------|-------|------|
| Vector parent creation | ~37 | $0 (deterministic) |
| Magnitude SN creation | ~37 | $0 (deterministic) |
| LLM enrichment (descriptions) | ~74 | ~$1 (sonnet, small batch) |
| HAS_COMPONENT relationships | ~117 | $0 (graph writes) |
| HAS_MAGNITUDE relationships | ~37 | $0 (graph writes) |
| **Total** | | **~$1** |

## RD Review History

### Round 1 (pre-RD — research phase findings)

| # | Finding | Resolution |
|---|---------|------------|
| 1 | Original analysis treated DD as only SN source — missed vector value | Corrected: SNs are source-independent, vector parents are physics concepts |
| 2 | Claimed "ISN grammar changes needed" for vector | Disproved: ISN validates vector names today (verified round-trip) |
| 3 | Metadata SNs have no pipeline or use case | Confirmed: keep kind, clean orphans, document as reserved |
| 4 | VectorEntry has magnitude property — must create companions | Added magnitude creation to pipeline design |
| 5 | 23 interpolation vectors should be excluded from core set | Added filter: exclude `interpolation` and `finite_element` parents |

### Round 2 (rubber-duck critique)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | GROUP placement uses stale DAG semantics | Minor | Fixed: clarified as "post-consolidation, pre-embedding" |
| 2 | "Zero-LLM" claim overstated — metadata fields need spec | Minor | Fixed: documented deterministic derivation (physics_domain=majority, tags=union) |
| 3 | **Magnitude is NOT a component** — wrong relationship type | **Blocking** | Fixed: split into `HAS_COMPONENT` (true components) + `HAS_MAGNITUDE` (norm) |
| 4 | **Grouping by `physical_base` alone collapses qualified names** | **Blocking** | Fixed: group by full reconstructed parent signature minus component segment |
| 5 | Interpolation exclusion not reproducible | Moderate | Fixed: added golden test fixture requirement + `is_eligible_vector()` spec |
| 6 | Auto-create magnitudes for non-vectors (tensor/anisotropy) | Moderate | Fixed: added eligibility check (TENSOR_CANDIDATES, min 2 components) |
| 7 | Metadata future path undocumented | Minor | Fixed: documented sanctioned path (manual import → future allowlist mode) |
| 8 | Missing risks: catalog round-trip, source-link semantics | Moderate | Fixed: added Risks & Mitigations table |
