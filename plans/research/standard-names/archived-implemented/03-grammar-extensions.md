# Feature Plan: Grammar Extensions

## Problem

Maarten's feedback (PR #2, Issues #4-8) identifies grammar patterns and standard name entries that the current system cannot represent. These include mathematical transformations of physical bases (`square_of_X`), time derivatives (`change_over_time_in_X`), binary operations (`product_of_X_and_Y`, `ratio_of_X_to_Y`), and a validation gap between `None` and `dimensionless` units.

Without these extensions, the batch generation pipeline cannot produce correct names for derived quantities in several IDS domains.

## Approach

Extend the grammar in phases: unary transformations first (simpler, single-base), then binary operators (complex, requires two-base parsing), then validation rules and missing entries.

## Deliverables

### Phase 1: Unary Transformations

**Owner**: Agent A (grammar)
**Wave**: 4 (parallel with pipeline polish)

Add support for mathematical transformations that take a single physical base:

| Transformation | Pattern | Example |
|---------------|---------|---------|
| `square_of` | `square_of_{physical_base}` | `square_of_electron_temperature` |
| `change_over_time_in` | `change_over_time_in_{physical_base}` | `change_over_time_in_magnetic_flux` |
| `logarithm_of` | `logarithm_of_{physical_base}` | `logarithm_of_density` |
| `inverse_of` | `inverse_of_{physical_base}` | `inverse_of_safety_factor` |

Implementation:

1. Add `transformation` segment to grammar specification:
   ```yaml
   transformation:
     position: prefix  # Appears before physical_base
     tokens:
       - square_of
       - change_over_time_in
       - logarithm_of
       - inverse_of
     connector: null  # Directly prepended: "square_of_electron_temperature"
   ```

2. Update `grammar.constants`:
   - Add `TRANSFORMATION_TOKENS` to `SEGMENT_TOKEN_MAP`
   - Update `SEGMENT_ORDER` to include transformation before physical_base
   - Update `SEGMENT_RULES` with transformation constraints

3. Update grammar parser:
   - Extend `parse_standard_name()` to recognize transformation prefixes
   - Extend `compose_standard_name()` to accept `transformation` parameter

4. Update vocabulary:
   - Add transformation tokens to vocabulary YAML
   - Run codegen to regenerate constants

5. Tests: `tests/grammar/test_transformations.py`
   - Test parsing of transformation names
   - Test composition with transformation parameter
   - Test that transformation + component is invalid (mutual exclusivity)
   - Test round-trip: compose → parse → compose

**Dependencies**: None (grammar is independent)
**Files**: Grammar specification, `grammar/constants.py` (regenerated), parser updates

### Phase 2: Binary Operators

**Owner**: Agent A (grammar, continued)
**Wave**: 4+ (may extend beyond Wave 4)

Binary operators are a significant grammar extension. They require two physical base arguments, which the current single-base grammar cannot represent.

| Operator | Pattern | Example |
|----------|---------|---------|
| `product_of_X_and_Y` | `product_of_{base1}_and_{base2}` | `product_of_density_and_temperature` |
| `ratio_of_X_to_Y` | `ratio_of_{base1}_to_{base2}` | `ratio_of_electron_temperature_to_ion_temperature` |
| `difference_of_X_and_Y` | `difference_of_{base1}_and_{base2}` | `difference_of_total_pressure_and_electron_pressure` |

**Design considerations**:

This is the most complex grammar extension. Key decisions:

1. **Two-base parsing**: The parser must recognize `_and_` or `_to_` as separators between two physical bases. Since physical bases are open vocabulary (any snake_case string), the separator detection must be unambiguous.

2. **Recursion depth**: Can operands themselves be transformed? E.g., `ratio_of_square_of_X_to_Y`. Recommend limiting to depth 1 (no nested transformations) for initial implementation.

3. **Connector words**: `and` for symmetric operations (product), `to` for asymmetric (ratio). These must be reserved words that cannot appear in physical bases.

4. **Compose API**: The `compose_standard_name()` tool needs two base parameters:
   ```python
   compose_standard_name(
       binary_operator="ratio_of",
       physical_base="electron_temperature",
       secondary_base="ion_temperature",
   )
   ```

Implementation:

1. Design the two-base grammar extension:
   - Define `binary_operator` segment with tokens (`product_of`, `ratio_of`, `difference_of`)
   - Define connector words (`and`, `to`) per operator
   - Define `secondary_base` as a new segment following the connector
   - Document mutual exclusivity: `binary_operator` is incompatible with `component`, `transformation`

2. Update grammar specification with binary operator rules

3. Update parser:
   - Detect binary operator prefix
   - Split on connector word to extract both bases
   - Validate both bases independently

4. Update `compose_standard_name()`:
   - Add `binary_operator` and `secondary_base` parameters
   - Validate mutual exclusivity with other segments

5. Tests: `tests/grammar/test_binary_operators.py`
   - Test parsing with various base combinations
   - Test connector word detection (and vs to)
   - Test mutual exclusivity constraints
   - Test that bases with `and` or `to` substrings parse correctly
   - Test error messages for ambiguous parses

**Dependencies**: Phase 1 (builds on transformation infrastructure)
**Risk**: High complexity. Recommend a design review (rubber-duck critique) before implementation.

### Phase 3: Validation Enhancements

**Owner**: Agent B (validation)
**Wave**: 4

1. **Units: None vs dimensionless**

   Add validation rule enforcing the semantic distinction:
   - `unit: None` — valid only for entries where the quantity has no physical unit (identifiers, names, labels). Typically `kind: metadata`.
   - `unit: dimensionless` — valid for physical quantities that are pure ratios or numbers (safety factor, beta, normalized quantities).

   Implementation:
   - Add validation rule to the validation pipeline
   - If `unit` is `None` and `kind` is `scalar` or `vector`, warn (may be incorrect)
   - If `unit` is `dimensionless`, ensure `kind` is `scalar` or `vector` (not metadata)
   - Document the distinction in `docs/grammar-reference.md`

2. **Audit existing entries**

   Check all current catalog entries for incorrect unit assignments:
   ```python
   # Pseudo-code for audit
   for entry in catalog:
       if entry.unit is None and entry.kind in ("scalar", "vector"):
           report_warning(f"{entry.name}: unit=None with kind={entry.kind}")
   ```

3. Tests: `tests/validation/test_unit_rules.py`
   - Test None unit with metadata kind (valid)
   - Test None unit with scalar kind (warning)
   - Test dimensionless unit with scalar kind (valid)
   - Test dimensionless unit with metadata kind (invalid)

**Dependencies**: None
**Files**: Validation pipeline, `docs/grammar-reference.md`

### Phase 4: Missing Standard Name Entries

**Owner**: Agent B (catalog)
**Wave**: 4 (after grammar extensions land)

Create the specific entries identified in Maarten's feedback:

| Name | Kind | Unit | Tags | Source |
|------|------|------|------|--------|
| `flux_loop_name` | metadata | None | [magnetics] | Issues #4-8 |
| `coil_current` | scalar | A | [magnetics] | Issues #4-8 |
| `passive_current` | scalar | A | [magnetics] | Issues #4-8 |

Implementation:
1. Validate each name against grammar (including any new transformation rules)
2. Check for vocabulary gaps — add tokens if needed via `manage_vocabulary`
3. Create entries using `create_standard_names` MCP tool
4. Include provenance referencing the originating GitHub issue
5. Persist using `write_standard_names` (with user approval)

**Dependencies**: Phases 1-3 (grammar must support any required patterns)
**Files**: Catalog YAML entries (created via MCP tools)

## Acceptance Criteria

- [ ] Unary transformations parse and compose correctly
- [ ] Binary operators parse and compose correctly (if implemented)
- [ ] `compose_standard_name` supports `transformation` parameter
- [ ] `compose_standard_name` supports `binary_operator` and `secondary_base` parameters
- [ ] None vs dimensionless validation enforced
- [ ] Missing entries created with correct grammar and provenance
- [ ] All grammar changes are backward compatible (existing names still parse)
- [ ] Grammar reference documentation updated
- [ ] 100% test coverage on code

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Binary operator parsing ambiguity | Medium | High | Design review before implementation; restrict to depth 1 |
| `and`/`to` in physical base names | Low | High | Reserved word list; validate bases don't contain connectors |
| Backward compatibility break | Low | Critical | Existing names must continue parsing; add, never remove |
| Scope creep (more transformations) | Medium | Low | Start with Maarten's list; extend later as needed |
