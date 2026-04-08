# Maarten Feedback Gap Analysis

## Sources

- PR #2: Guidelines documentation
- PR #3, #11: Review comments
- Issues #4–8: Concrete standard name proposals
- Inline comments on naming conventions, units, and sign handling

## Feedback by Theme

### 1. Missing Transformations

Maarten's guidelines describe transformation patterns that are not implemented in the grammar:

| Transformation | Pattern | Status | Example |
|---------------|---------|--------|---------|
| `square_of_X` | Unary transform of physical base | Not implemented | `square_of_electron_temperature` |
| `change_over_time_in_X` | Time derivative transform | Not implemented | `change_over_time_in_magnetic_flux` |
| `product_of_X_and_Y` | Binary operator | Not implemented | `product_of_density_and_temperature` |
| `ratio_of_X_to_Y` | Binary operator | Not implemented | `ratio_of_electron_temperature_to_ion_temperature` |

**Implementation status**: The grammar currently supports simple `component_of_X` and `X_at_position` patterns but has no mechanism for mathematical transformations of physical bases or binary operations over two quantities.

**Action items**:
1. Design grammar extension for unary transformations (`square_of`, `change_over_time_in`, `logarithm_of`, `inverse_of`)
2. Design grammar extension for binary operators (`product_of_X_and_Y`, `ratio_of_X_to_Y`, `difference_of_X_and_Y`)
3. Binary operators are a significant grammar change — they require two `physical_base` arguments, which the current single-base grammar cannot represent
4. Add vocabulary entries for transformation keywords
5. Update `compose_standard_name` tool to support transformation parameters

**Priority**: High — blocks correct naming of derived quantities in several IDS domains

### 2. Missing Standard Name Entries

Specific name entries identified in Maarten's proposals that are absent from the catalog:

| Name | Domain | Issue/PR | Status |
|------|--------|----------|--------|
| `flux_loop_name` | magnetics | Issues #4–8 | Not created |
| `coil_current` | magnetics | Issues #4–8 | Not created |
| `passive_current` | magnetics | Issues #4–8 | Not created |

**Implementation status**: These entries require both grammar support (some may need new vocabulary tokens) and catalog entries.

**Action items**:
1. Validate each proposed name against current grammar
2. Identify vocabulary gaps (if `flux_loop` or `passive` need to be added as tokens)
3. Create entries using MCP tools (`create_standard_names`)
4. Ensure provenance references the originating issue

**Priority**: Medium — these are concrete deliverables that can be created once grammar support exists

### 3. Units: None vs Dimensionless

Maarten's feedback distinguishes between:
- **`None`**: The quantity has no unit (e.g., a name or identifier string)
- **`dimensionless`**: The quantity is a ratio or pure number with physical meaning (e.g., safety factor, beta)

**Implementation status**: The current validation does not enforce this distinction. Both values are accepted but the semantic difference is not documented or validated.

**Action items**:
1. Document the distinction in the grammar reference (`docs/grammar-reference.md`)
2. Add validation rule: `unit: None` only valid for `kind: metadata` entries
3. Add validation rule: dimensionless physical quantities must use `unit: dimensionless`
4. Audit existing catalog entries for incorrect unit assignments
5. Update `get_schema()` tool output to reflect this rule

**Priority**: Medium — affects data quality but not blocking

### 4. Passive/Active Conductor Naming Ambiguity

Maarten raised a naming policy question about distinguishing passive conductors (vessel, structures) from active conductors (coils) in standard names.

**Implementation status**: No documented naming convention for this distinction. The grammar has vocabulary for specific devices (coils, loops) but no systematic way to distinguish passive vs active.

**Action items**:
1. Document the naming convention for conductors in guidelines
2. Determine if `passive` and `active` should be vocabulary tokens (subjects? processes?)
3. Add examples to the grammar reference showing correct naming for both types
4. Consider whether this maps to existing `object` vocabulary or needs new tokens

**Priority**: Low — design decision needed before implementation

### 5. Sign Convention Documentation

Maarten's feedback notes that sign conventions (COCOS) are partially documented. The IMAS DD tools provide COCOS field information, but standard names do not systematically encode or validate sign conventions.

**Implementation status**: The IMAS DD MCP tools have `get_dd_cocos_fields()` which returns all COCOS-dependent paths grouped by transformation type. Standard names grammar has no sign convention representation.

**Action items**:
1. Document which standard names are COCOS-sensitive (cross-reference with DD COCOS fields)
2. Add optional `cocos_transform` field to standard name entries for COCOS-dependent quantities
3. Link to DD version context for sign convention changes
4. Consider adding a validation check that COCOS-sensitive names have documented sign behavior

**Priority**: Low — informational/documentation improvement, not blocking

### 6. Tag Vocabulary Divergence

Maarten proposed an IDS-aligned tag vocabulary where primary tags map to IDS names (e.g., `equilibrium`, `magnetics`). The current implementation uses a different but overlapping approach.

**Implementation status**: The tag system is implemented and functional. The vocabulary has diverged from Maarten's original proposal but serves a similar purpose. The current tags are physics-domain-oriented rather than strictly IDS-aligned.

**Action items**:
1. Document the rationale for the current tag vocabulary design
2. Create a mapping table showing the relationship between current tags and Maarten's IDS-aligned proposal
3. Evaluate whether a reconciliation is beneficial or if the divergence is acceptable
4. If reconciliation is needed, plan it as a separate effort (not blocking batch generation)

**Priority**: Low — track but do not block. The current system works; alignment is a long-term refinement.

### 7. xarray Dimension Usage Pattern

Maarten's comments reference an xarray-based usage pattern for standard names that is not documented.

**Implementation status**: No documentation exists for how standard names map to xarray dimensions or coordinates.

**Action items**:
1. Document the intended xarray integration pattern
2. Add examples showing how standard names are used as xarray dimension labels
3. Consider whether this affects name design (e.g., length constraints for dimension names)

**Priority**: Low — documentation improvement, non-blocking

## Priority Summary

### High Priority (blocks batch generation or grammar correctness)

1. **Grammar extensions for unary transformations** — `square_of`, `change_over_time_in`
2. **Grammar extensions for binary operators** — `product_of_X_and_Y`, `ratio_of_X_to_Y`

### Medium Priority (improves quality, can be addressed in parallel)

3. **Missing standard name entries** — `flux_loop_name`, `coil_current`, `passive_current`
4. **Units None vs dimensionless validation** — enforce semantic distinction
5. **Conductor naming convention** — document passive/active distinction

### Low Priority (documentation and refinement)

6. **Sign convention documentation** — cross-reference with COCOS fields
7. **Tag vocabulary reconciliation** — document rationale, evaluate alignment
8. **xarray dimension pattern** — documentation only

## Relationship to Feature Plans

| Gap | Related Feature | Integration Point |
|-----|----------------|-------------------|
| Unary transformations | Feature 06 (new) | Grammar extension |
| Binary operators | Feature 06 (new) | Grammar extension (significant) |
| Missing entries | Feature 06 (new) | Catalog entries after grammar work |
| None vs dimensionless | Feature 06 (new) | Validation rule addition |
| Conductor naming | Guidelines documentation | Policy decision, then vocabulary |
| Sign conventions | Future work | DD cross-reference tooling |
| Tag vocabulary | Future work | Reconciliation assessment |
| xarray pattern | Documentation | Non-code change |
