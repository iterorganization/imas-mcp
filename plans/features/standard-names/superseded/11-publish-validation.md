# 11: Publish Validation + Graph-Backed Staging

**Status:** Pending
**Priority:** High — prevents publishing invalid entries to catalog
**Depends on:** 09 (schema providers + LLM compose), 10 (pipeline fixes)
**Effort:** 1-2 days

## Problem

Two issues with the current write/publish flow:

### A. No validation gate

The `sn publish` command converts StandardName graph nodes to YAML catalog
files, but only checks for filename collisions. It does NOT validate that
the generated entries conform to the catalog's grammar, schema, or semantic rules.

### B. No staging or rollback

The pipeline holds results in plain Python lists (`state.candidates`,
`state.composed`) with no transactional semantics. If the compose worker
crashes midway through a batch, partial results are lost. There's no
"validate-before-commit" gate — names go straight from the LLM to the
graph via `write_standard_names()`.

The `imas-standard-names` UnitOfWork pattern provides exactly this:
- In-memory staging with add/update/remove/rename operations
- Undo stack with typed operations (UndoOpAdd, UndoOpDelete, etc.)
- Validation gate before commit
- Rollback on failure

## Approach

### Phase 4a: Validation gate for publish

Import validation functions from `imas_standard_names` and run them
against generated entries before writing YAML.

```python
from imas_standard_names.grammar import compose_standard_name, parse_standard_name
from imas_standard_names.grammar.field_schemas import FIELD_GUIDANCE
from imas_standard_names.services import validate_models
```

### Phase 4b: Graph-backed staging

Adapt the UnitOfWork pattern for graph-backed persistence. The key insight:

**The YAML UnitOfWork commits by writing YAML files. The graph UnitOfWork
commits by writing to Neo4j.** The staging, validation, undo, and rollback
mechanics are identical — only the persistence backend changes.

```python
class GraphUnitOfWork:
    """In-memory staging with graph persistence commit boundary."""

    def __init__(self):
        self._staged: dict[str, StandardNameEntry] = {}
        self._undo: list[UndoOp] = []

    def add(self, entry: StandardNameEntry) -> None:
        if entry.name in self._staged:
            raise ValueError(f"'{entry.name}' already staged")
        self._staged[entry.name] = entry
        self._undo.append(UndoOpAdd(entry.name))

    def validate(self) -> list[str]:
        """Run all validation checks. Returns list of error strings."""
        errors = []
        for entry in self._staged.values():
            # Grammar round-trip
            errors.extend(self._check_grammar(entry))
            # Schema conformance
            errors.extend(self._check_schema(entry))
        # Cross-entry semantic checks
        errors.extend(self._check_semantic_conflicts())
        return errors

    def commit(self) -> int:
        """Validate and write all staged entries to graph.

        Returns the number of entries written. Raises ValueError if
        validation fails (entries remain staged for correction).
        """
        issues = self.validate()
        if issues:
            raise ValueError("Validation failed:\n" + "\n".join(issues))
        written = write_standard_names([e.model_dump() for e in self._staged.values()])
        self._staged.clear()
        self._undo.clear()
        return written

    def rollback(self) -> None:
        """Discard all staged entries."""
        self._staged.clear()
        self._undo.clear()
```

The compose worker stages results instead of accumulating in plain lists:

```python
# Before (no staging):
state.composed.append(result.model_dump())

# After (staged):
state.staging.add(StandardNameEntry(**result.model_dump()))
```

The validate worker calls `state.staging.validate()` and only commits
entries that pass all checks.

## Files to Create/Modify

### New: `imas_codex/sn/staging.py`

Graph-backed UnitOfWork with:
- `add()`, `update()`, `remove()` with undo stack
- `validate()` importing `imas_standard_names` validation
- `commit()` calling `write_standard_names()`
- `rollback()` discarding staged entries

### Modify: `imas_codex/sn/publish.py`

Add validation step between entry generation and YAML writing:

```python
for entry in entries:
    # Grammar round-trip
    parsed = parse_standard_name(entry.name)
    recomposed = parsed.compose()
    if recomposed != entry.name:
        report_warning(f"Round-trip mismatch: {entry.name} → {recomposed}")

    # Schema validation
    try:
        create_standard_name_entry(entry.to_dict())
    except ValidationError as e:
        report_error(f"Schema-invalid: {entry.name}: {e}")
```

### Modify: `imas_codex/sn/state.py`

Add `staging: GraphUnitOfWork` field, replacing plain list accumulation.

### Modify: `imas_codex/sn/workers.py`

- Compose worker stages results via `state.staging.add()`
- Validate worker calls `state.staging.validate()`
- Final write uses `state.staging.commit()`

## Acceptance Criteria

- `sn publish --dry-run` shows validation summary:
  ```
  Publish Validation:
    Grammar: 42/45 passed (3 failed)
    Schema: 41/45 passed (4 failed)
    Publishable: 40 entries
  ```
- Grammar-invalid names are excluded from output
- Schema-invalid entries are excluded from output
- Compose worker uses staging instead of plain lists
- If compose worker crashes mid-batch, `rollback()` clears partial results
- `--force` flag overrides warnings (but not errors)

## Testing

- Unit test: GraphUnitOfWork add/validate/commit/rollback cycle
- Unit test: validation catches invalid grammar, missing fields, bad tags
- Unit test: undo stack correctly reverts operations
- Integration: `sn publish --dry-run` with known-bad entries
- Integration: `sn build` with crash simulation (verify rollback)

## Design Decision: UnitOfWork vs Plain Lists

**Why keep UnitOfWork even with Neo4j?**

Neo4j has ACID transactions, but that only covers the *write* boundary.
The UnitOfWork pattern adds value at the *staging* boundary:

1. **Validate-before-commit** — catch grammar/schema errors BEFORE touching
   the graph, not after. Failed validation doesn't create orphan nodes.
2. **Batch rollback** — if the LLM produces garbage for one IDS batch,
   discard just that batch without affecting earlier successes.
3. **Undo stack** — the review worker can flag entries for removal;
   undo preserves the original for debugging.
4. **Clean separation** — "proposed" entries live in staging until explicitly
   committed. The graph only contains validated, committed names.
