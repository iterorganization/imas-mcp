# Feature 01: Grammar API Exports

**Repository:** imas-standard-names  
**Wave:** 1 (parallel with 02, 03)  
**Enables:** Feature 04 (JSON Schema Contract), Feature 05 (codex SN Build Pipeline)  

---

## Goal

Ensure the grammar module is cleanly importable as a library by imas-codex. The grammar system already has zero external coupling — this feature audits and formalizes the public API.

## Deliverables

### Phase 1: Audit and formalize `__all__` exports

- Verify `grammar/__init__.py` exports all needed symbols
- Ensure all enums are accessible: `Component`, `Position`, `Subject`, `Object`, `GeometricBase`, `Process`, `GenericPhysicalBase`
- Ensure core functions exported: `compose_standard_name`, `parse_standard_name`, `compose_name`, `parse_name`
- Add `py.typed` marker file for type-checking support

### Phase 2: Export validation functions

- Make `services.validate_models()` a documented public API
- Ensure it can be called with just a `dict[str, StandardNameEntry]` — no catalog required
- Document the function signature and return type in docstring

### Phase 3: Zero-side-effect import verification

- Write a test that imports `imas_standard_names.grammar` in isolation
- Verify no file I/O, no network calls, no catalog loading on import
- Verify grammar module works without optional dependencies (spacy, proselint)

### Phase 4: Package extras (optional)

- Consider `imas-standard-names[grammar]` extra that installs only grammar deps
- This allows codex to depend on a minimal footprint

## Acceptance Criteria

- `from imas_standard_names.grammar import compose_standard_name, parse_standard_name` works
- All enums importable
- `validate_models()` callable without catalog
- No side effects on import
- 100% test coverage on new/modified code
