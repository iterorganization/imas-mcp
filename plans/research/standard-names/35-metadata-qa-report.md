# Metadata QA Report — 2025-07-17

## Audit scope

- **1083** StandardName nodes audited
- DD version: 4.1.0 (uniform across all nodes)
- Relationship types verified: HAS_UNIT, HAS_COCOS, HAS_STANDARD_NAME, PRODUCED_NAME, HAS_SEGMENT, FROM_DD_PATH

## Findings

| # | Severity | Category | Count | Fix applied |
|---|----------|----------|------:|-------------|
| 1 | HIGH | Unit mismatch vs DD (SN unit wrong) | 5 | Cypher fix: `boron_ion_number_density` (1→m⁻³), `hydrogen_ion_density` (1→m⁻³), `major_radius_of_magnetic_axis` (1→m), `poloidal_angle_grid_of_flux_tube` (1→rad), `poloidal_steering_angle_of_electron_cyclotron_beam` (1→rad) |
| 2 | HIGH | Unit convention mismatch (charge unit) | 4 | Cypher fix: `atomic_number`, `ion_average_charge_of_ion_state`, `ion_average_square_charge_of_ion_state`, `ion_charge_number_at_ion_state` — all 1→e |
| 3 | MED | Unit dimensional equivalents (DD convention preferred) | 2 | Cypher fix: `fast_ion_pressure` (J.m⁻³→Pa), `toroidal_torque_inside_flux_surface` (kg.m².s⁻²→m.N) |
| 4 | MED | Unit ambiguity (DD path has dual units) | 2 | Cypher fix: `binormal_component_of_wave_magnetic_field` (m⁻¹.V→T), `parallel_current_due_to_non_inductive_current_drive` (A.m→A) |
| 5 | LOW | Residual unit mismatches (HAS_STANDARD_NAME mislinks) | 7 | No SN fix — mislinked DD paths with dimensionally equivalent or wrong DD units |
| 6 | MED | Empty descriptions | 136 | Expected: all have `review_status=named` (pre-enrichment stage) |
| 7 | LOW | Orphaned SNs (no PRODUCED_NAME source) | 140 | No fix — from enrichment pipeline, `model=NULL` |
| 8 | LOW | Missing cocos_transformation_type warning guard | 1 | Code fix: added warning in `write_standard_names()` |
| 9 | INFO | physics_domain NULL | 1 | Acceptable — single edge case |

### Unit fix detail

13 StandardName HAS_UNIT relationships corrected via inline Cypher:

```cypher
-- For each (sn_id, new_unit):
MATCH (sn:StandardName {id: $sn_id})-[old_r:HAS_UNIT]->(:Unit)
DELETE old_r
WITH sn
MATCH (u:Unit {id: $new_unit})
MERGE (sn)-[:HAS_UNIT]->(u)
```

### Clean findings (no action needed)

| Category | Result |
|----------|--------|
| Missing HAS_UNIT (non-metadata) | 0 ✓ |
| physics_domain values vs PhysicsDomain enum | All valid ✓ |
| physics_domain mismatch vs DD source | 0 ✓ |
| COCOS transformation without HAS_COCOS link | 0 ✓ |
| Partial grammar population | 0 ✓ |
| Tag normalisation (case, format) | 48 distinct tags, all lowercase-hyphenated ✓ |
| Kind values | 5 valid kinds (scalar, vector_component, complex_part, tensor_component, spectrum) ✓ |
| dd_version consistency | All 4.1.0 ✓ |

### Distributions

**Validation status:** valid=766, pending=131, quarantined=102, needs_revision=84

**Review status:** enriched=742, named=328, drafted=13

**Kind:** scalar=892, vector_component=179, complex_part=9, tensor_component=2, spectrum=1

**COCOS transformation types:** one_like=614, NULL=457, psi_like=4, b0_like=3, ip_like=2, q_like=1, grid_type_tensor_contravariant_like=1, tor_angle_like=1

**Grammar coverage:** grammar_physical_base=1066/1083, grammar_subject=286, grammar_component=161

**HAS_SEGMENT edges:** 14 SNs (grammar edge writing is optional, runs on-demand)

**Confidence:** min=0.20, max=1.00, avg=0.81 (N=943)

### Reviewer fields (OBSERVE ONLY — c4-multi-reviewer-2 restructuring)

- 763 SNs have reviewer_score/model/tier
- 0 Review nodes exist (migration to Review node schema pending)
- **No modifications made** to reviewer_* fields per task constraint

## Pipeline guards

### Existing (verified)
- `unit` injected from `source_item.get("unit")` in compose worker (line 1137)
- `physics_domain` injected from `source_item.get("physics_domain")` (line 1146)
- `cocos_transformation_type` injected from `source_item.get("cocos_label")` (line 1150)
- Unit conflict detection in `write_standard_names()` (line 556-585)
- `coalesce()` used for physics_domain, cocos_transformation_type, dd_version in MERGE Cypher

### Added
- **cocos_transformation_type without cocos integer warning** in `write_standard_names()` — logs a WARNING when `cocos_transformation_type` is set but `cocos` integer is `None`, since the HAS_COCOS edge won't be created

## Invariant tests (30 total)

### Pre-existing (16)
- TestSchemaInvariants: unit_slot_has_unit_range, cocos_slot_has_cocos_range, kind_enum_values, validation_status_enum
- TestWriteCypherInvariants: has_unit_relationship, has_cocos_uses_match, physics_domain_coalesce, cocos_transformation_type_coalesce, dd_version_coalesce, cocos_missing_integer_warning, unit_in_batch_payload
- TestGrammarDecomposition: field_set, returns_all_fields
- TestComposeWorkerInjection: unit_from_source, physics_domain_from_source, cocos_from_source

### Existing (4)
- TestPhysicsDomainEnum: importable, has_general, has_equilibrium, no_mhd_alias

### Added (10 new)
- TestStandardNameKindEnum: kind_enum_no_unknown_values, kind_enum_includes_metadata
- TestTagNormalisation: tags_lowercase_hyphenated, write_cypher_preserves_tags
- TestDDVersionConsistency: dd_version_uses_coalesce_in_write
- TestWritePathRelationships: has_unit_uses_merge, has_standard_name_for_dd, unit_batch_includes_all
- TestUnitOverrideEngine: override_engine_importable, override_config_exists

## Residual (not fixable in this audit)

1. **136 empty descriptions** — expected for `named` status; will be populated by `sn enrich` pipeline
2. **140 orphaned SNs** (no PRODUCED_NAME source) — created by enrichment pipeline without source provenance; needs pipeline investigation
3. **7 residual unit mismatches** via HAS_STANDARD_NAME — caused by DD paths with dual units or mislinked entity→concept edges; fixing requires HAS_STANDARD_NAME edge cleanup
4. **0 links populated** — link resolution (`sn.links`) not yet run on any SN
5. **14/1083 HAS_SEGMENT edges** — grammar token edge writing is on-demand; most SNs lack segment edges
6. **Quarantine rate 9.4%** (102/1083) — exceeds 5% test threshold; requires vocab update + regen cycle
7. **ISN rc17 regression** — `test_enrich_document` and `test_vector_not_defaulted` failures are pre-existing in ISN v0.7.0rc17
