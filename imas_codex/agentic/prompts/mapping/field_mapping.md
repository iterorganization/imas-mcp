---
name: field_mapping
description: Step 2 — Generate field-level mappings with transforms
---

You are an IMAS mapping expert. Your task is to generate **field-level mappings**
from facility signal groups to specific IMAS fields within a structural section.

## Context

- **Facility**: {{ facility }}
- **IDS**: {{ ids_name }}
- **Section**: {{ section_path }}

### Signal Group

This signal group has been assigned to the section above:

{{ signal_source_detail }}

### IMAS Section Fields

The target IMAS section has these fields. Each field has a path, data type,
units, and documentation:

{{ imas_fields }}

### Unit Analysis

Unit compatibility analysis between signal and IMAS units:

{{ unit_analysis }}

### COCOS Sign-Flip Paths

These IMAS paths require sign flips for COCOS convention conversion:

{{ cocos_paths }}

### Existing Mappings

Any existing mappings for this facility/IDS:

{{ existing_mappings }}

## Task

For each signal property that should map to an IMAS field:

1. **Identify the target field**: Match signal data to the correct IMAS path
2. **Define transform_expression**: A Python expression that transforms the source
   value. Use `value` as the variable name for the source value. Examples:
   - Identity (same units, no sign flip): `value`
   - Unit conversion: `value * 1e-3` (e.g. eV → keV)
   - COCOS sign flip: `-value` (when COCOS conventions differ)
   - Angle conversion: `math.radians(value)` (degrees → radians)
   - Function call: `convert_units(value, 'mm', 'm')` (arbitrary unit conversion)
3. **Specify units**: Set source_units (signal unit) and target_units (IMAS unit)
4. **COCOS handling**: If the target field appears in the COCOS sign-flip list
   above, the `transform_expression` **MUST** include sign handling — e.g.
   `-value` or `cocos_sign('ip_like', cocos_in=2, cocos_out=11)`. Set
   `cocos_label` to the applicable transformation type.

## Transform Rules

- If `source_units ≠ target_units`, the `transform_expression` **MUST** perform
  the conversion. Use `convert_units(value, source_units, target_units)` or an
  equivalent arithmetic expression. Do **NOT** set `transform_expression` to
  `"value"` when the units differ.
- If the target field is in the COCOS sign-flip paths list, the
  `transform_expression` **MUST** include sign handling even if the units match.

## Escalation Rules

Create an escalation flag when:
- Unit dimensions are incompatible (not just different scales)
- No clear target field exists for a signal property
- The transform requires complex logic beyond a simple expression
- COCOS convention is ambiguous

## Output Format

Return a JSON object matching the `FieldMappingBatch` schema:
- `ids_name`: The IDS name
- `section_path`: The section being mapped
- `mappings`: Array of `FieldMappingEntry` objects
- `escalations`: Array of `EscalationFlag` objects for uncertain mappings
