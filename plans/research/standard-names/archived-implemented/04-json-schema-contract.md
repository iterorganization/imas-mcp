# Feature 04: JSON Schema Contract

**Repository:** imas-standard-names  
**Wave:** 2 (after Feature 01)  
**Depends on:** Feature 01 (Grammar API Exports)  
**Enables:** Feature 05 (codex SN Build Pipeline — validation at mint boundary)  

---

## Goal

Publish a versioned JSON schema that defines the contract between imas-codex (producer) and imas-standard-names (validator). Codex generates candidate names as structured data matching this schema; standard-names validates and persists them.

## Deliverables

### Phase 1: Extract and version the schema

- Export `StandardNameEntry` Pydantic JSON schema as a static file
- Include all discriminated union variants (scalar, vector, metadata)
- Version the schema with semver, linked to grammar version
- Store at `imas_standard_names/schemas/entry_schema.json`

### Phase 2: Validation utility

- Create `validate_against_schema(data: dict) -> list[str]` utility
- Validates arbitrary JSON/YAML against the published schema
- Returns list of validation errors (empty = valid)
- Usable by codex's mint phase without importing full standard-names

### Phase 3: YAML file format contract

- Document the exact YAML structure codex must produce for minted names
- Include: required fields, tag ordering rules, provenance format
- Include: `ids_paths` population expectations
- Publish as part of the schema artifact

## Acceptance Criteria

- JSON schema file generated and included in package distribution
- Schema versioned and linked to grammar version
- Validation utility works standalone
- Contract documentation covers all edge cases
