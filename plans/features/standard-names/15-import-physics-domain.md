# Import PhysicsDomain from imas-standard-names

> **Repo**: imas-codex
> **Status**: planned
> **Depends on**: imas-standard-names `01-rename-tags-to-physics-domain.md` (Phase 6 RC release)

## Problem

imas-codex maintains its own `PhysicsDomain` enum (22 values) codegen'd from
a 250-line LinkML schema (`physics_domains.yaml`). This duplicates the physics
domain vocabulary that imas-standard-names now owns as the canonical source.

After imas-standard-names publishes `PhysicsDomain` as a `str, Enum` (31 values),
imas-codex should import it rather than maintain a parallel definition.

## Current Architecture

```
imas_codex/schemas/physics_domains.yaml    ← 250-line LinkML schema (SOURCE)
    ↓ gen_physics_domains.py (codegen)
imas_codex/core/physics_domain.py          ← PhysicsDomain(str, Enum) 22 values
    ↓ imported by
imas_codex/schemas/common.yaml             ← `imports: - physics_domains`
    ↓ used by
facility.yaml, standard_name.yaml          ← `range: PhysicsDomain`
    ↓ used by
15+ Python modules                         ← runtime enum validation
```

## Target Architecture

```
imas-standard-names (PyPI)
    ↓ exports
PhysicsDomain(str, Enum) 31 values
    ↓ imported by
imas_codex/core/physics_domain.py          ← re-export (one-line change)
    ↓ imported by (unchanged)
15+ Python modules                         ← same import path, no changes
```

## Phase 1: Replace codegen with import

### 1a. Update `imas_codex/core/physics_domain.py`

Replace the entire codegen'd file with a re-export:

```python
"""Physics domain enum — canonical source is imas-standard-names.

This module re-exports PhysicsDomain from imas-standard-names so that
all imas-codex code continues to import from the same path:
    from imas_codex.core.physics_domain import PhysicsDomain
"""

from imas_standard_names.grammar.tag_types import PhysicsDomain

__all__ = ["PhysicsDomain"]
```

This file is currently auto-generated and gitignored. Change it to a
hand-written re-export and **remove from .gitignore**.

### 1b. Delete `imas_codex/schemas/physics_domains.yaml`

The 250-line LinkML schema is no longer needed. The enum source of truth
is now in imas-standard-names.

### 1c. Delete `scripts/gen_physics_domains.py`

The codegen script is no longer needed.

### 1d. Update `scripts/build_models.py`

Remove the physics_domains codegen step. The build hook currently:
1. Generates physics_domain.py from physics_domains.yaml
2. Generates graph models from facility.yaml/common.yaml/etc

Remove step 1. Step 2 continues unchanged.

### 1e. Update `hatch_build_hooks.py`

Remove physics_domain from the build hook's generated file list.

### 1f. Update `.gitignore`

Remove `imas_codex/core/physics_domain.py` from gitignore — it's now a
hand-written file that should be tracked.

### 1g. Update `imas_codex/schemas/common.yaml`

Remove `imports: - physics_domains` from the imports list. The LinkML schema
no longer needs to reference the physics_domains schema since the enum now
comes from Python at runtime. The `range: PhysicsDomain` annotations in
facility.yaml and standard_name.yaml become string-validated at the LinkML
level but enum-validated at the Python level.

**Option A** (clean): Remove `range: PhysicsDomain` from LinkML, use
`range: string` — runtime Python code does enum validation via the imported
enum. This avoids needing a phantom LinkML enum.

**Option B** (keep schema validation): Create a minimal LinkML enum stub
that lists just the values (no descriptions/categories) and is auto-synced
from imas-standard-names. This preserves `range: PhysicsDomain` in schema.

**Decision**: Option A is cleaner. The PhysicsDomain values appear in the
generated schema_context_data.py regardless (built from Python enum, not
LinkML). Schema compliance tests validate against the Python enum.

### 1h. Update `pyproject.toml`

Add `imas-standard-names >= 0.8.0` to dependencies (the version with
PhysicsDomain export).

## Phase 2: Add new enum values to graph

The unified enum has 9 new values not currently in the graph. These are
additive — no existing data needs migration.

New values: `core_plasma_physics`, `fast_particles`, `runaway_electrons`,
`waves`, `fueling`, `plasma_initiation`, `spectroscopy`, `neutronics`,
`gyrokinetics`.

No graph migration needed — new values become available for new data
automatically. Existing classification prompts will start using them
as they appear in the enum.

## Phase 3: Update tests

### 3a. `tests/core/test_physics_categorization.py`

Update test expectations for the new 31-value enum. Add tests for
new values.

### 3b. `tests/graph/test_schema_compliance.py`

Verify schema compliance tests pass with the new enum source.
The tests should work unchanged since they validate against the
Python enum, not the LinkML schema directly.

### 3c. Run full test suite

```bash
uv run build-models --force
uv run pytest tests/ -x -q
```

## Phase 4: Clean up and commit

```bash
# Lint
uv run ruff check --fix .
uv run ruff format .

# Stage changes (NOT auto-generated files)
git add imas_codex/core/physics_domain.py  # Now hand-written, tracked
git add -u  # Stage deletions and modifications
# DO NOT stage: graph/models.py, graph/dd_models.py, config/models.py

uv run git commit -m "refactor: import PhysicsDomain from imas-standard-names

BREAKING CHANGE: PhysicsDomain enum now has 31 values (was 22).
Nine new values added: core_plasma_physics, fast_particles,
runaway_electrons, waves, fueling, plasma_initiation, spectroscopy,
neutronics, gyrokinetics.

Removed physics_domains.yaml LinkML schema and gen_physics_domains.py
codegen script. PhysicsDomain is now imported from imas-standard-names
and re-exported from imas_codex.core.physics_domain."

git pull --no-rebase origin develop
git push origin develop
```

## Documentation Updates

| Target | Changes |
|--------|---------|
| `AGENTS.md` | Update PhysicsDomain section: note it's imported from imas-standard-names |
| `AGENTS.md` | Remove physics_domains.yaml from schema files list |
| `AGENTS.md` | Remove gen_physics_domains.py from build pipeline |

## Files Changed

| Action | File | Notes |
|--------|------|-------|
| REWRITE | `imas_codex/core/physics_domain.py` | Codegen → hand-written re-export |
| DELETE | `imas_codex/schemas/physics_domains.yaml` | No longer source of truth |
| DELETE | `scripts/gen_physics_domains.py` | No longer needed |
| MODIFY | `scripts/build_models.py` | Remove physics_domains step |
| MODIFY | `hatch_build_hooks.py` | Remove from generated files |
| MODIFY | `.gitignore` | Un-ignore physics_domain.py |
| MODIFY | `imas_codex/schemas/common.yaml` | Remove physics_domains import |
| MODIFY | `pyproject.toml` | Add imas-standard-names >= 0.8.0 dep |
| MODIFY | `tests/core/test_physics_categorization.py` | Update for 31 values |
| MODIFY | `AGENTS.md` | Update documentation |

## Risks

- **Version pinning**: If imas-standard-names adds/removes PhysicsDomain
  values in a future release, imas-codex graph data could become inconsistent.
  Mitigation: pin to `>= 0.8.0, < 1.0.0` and review on major bumps.
- **Build order**: `uv sync` must install imas-standard-names before the
  build hook runs. Since physics_domain.py is now hand-written (not codegen'd),
  this is only a runtime concern, not a build-time concern.
- **LinkML validation gap**: With Option A, LinkML schemas lose
  `range: PhysicsDomain` validation. Schema compliance tests still work
  because they validate against the Python enum. The gap is only in the
  LinkML schema itself (used for documentation, not runtime).
