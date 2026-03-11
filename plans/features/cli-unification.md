# CLI Unification: 19 → 13 Top-Level Commands

Status: **Approved**
Supersedes: CLI sections in `imas-mapping-combined.md` Phase 5, `imas-mapping-pipeline.md` Phase 1c

## Summary

Consolidate the CLI from 19 top-level commands to 13 by removing dead tools,
absorbing related functionality, and unifying the IMAS domain under a single
`imas` command. No backwards compatibility.

## Motivation

The CLI grew organically. Three separate commands (`ids`, `imas`, `map`) all deal
with IMAS but expose different facets. Two commands (`enrich`, `ingest`) are zombies —
their functionality now lives inside the `discover` pipeline workers. `setup-age`
generates encryption keys for a feature that was never implemented. The `clusters`
subgroup exposes individual steps of what `imas build` already orchestrates.

A user wanting to "work with IMAS mappings for JET pf_active" has to know that
`map run` creates the mapping, `map show` displays it, but `ids export` exports it.
The mental model is fragmented.

## Analysis of Removals

### `enrich` — Dead

| Command | Status | Replacement |
|---------|--------|-------------|
| `enrich nodes` | Dead | `discover signals --enrich-only` handles all signal enrichment via parallel async workers. The old `enrich nodes` uses a CodeAgent-based approach that pre-dates the discovery pipeline. |
| `enrich run` | Dead | Wraps `quick_task_sync()` — a smolagents CodeAgent runner. Not used by any workflow. If needed, the agentic library is importable directly. |
| `enrich mark-stale` | Dead | Sets `enrichment_status='stale'` on SignalNodes. This property doesn't exist in the current schema — signals use `status` from `FacilitySignalStatus`. The query targets a non-existent property. |

**Verdict**: Delete `imas_codex/cli/enrich.py`. No replacement needed.

### `ingest` — Dead

| Command | Status | Replacement |
|---------|--------|-------------|
| `ingest run` | Dead | Code ingestion is a worker stage within `discover code`. Files are fetched, chunked, and embedded as part of the code discovery pipeline. |
| `ingest status` | Dead | `discover status <facility> --domain code` shows CodeFile queue stats. |
| `ingest queue` | Dead | Code file discovery creates CodeFile nodes automatically during `discover code`. No manual queueing needed. |
| `ingest list` | Dead | `discover inspect <facility>` shows discovered items. Graph queries via MCP for specific status filtering. |

**Verdict**: Delete `imas_codex/cli/ingest.py`. No replacement needed.

### `setup-age` — Never implemented

Generates an `age` encryption key pair for "encrypting private facility YAML files."
The encryption/decryption code was never written — there is no code anywhere in the
codebase that reads `IMAS_AGE_KEY_FILE`, encrypts files with age, or decrypts them.
Private YAML files are protected by `.gitignore` and synced via GitHub Gist
(`config private push/pull`). The age key serves no purpose.

**Verdict**: Delete the `setup_age` command from `utils.py`. Remove registration from
`__init__.py`.

### `clusters` — Absorbed into `imas build`

The `clusters` subgroup exposes 5 individual steps:

| Command | Already called by `imas build`? |
|---------|-------------------------------|
| `clusters build` | Yes — `imas build` calls it unless `--skip-clusters` |
| `clusters label` | Yes — `imas build` labels after building |
| `clusters sync` | Yes — `imas build` syncs to graph |
| `clusters embed` | Yes — `imas build` embeds cluster text |
| `clusters status` | No — diagnostic only |

The only command not covered by `imas build` is `clusters status`, which is a
diagnostic that can be folded into `imas status --verbose` or `imas dd status`.

**Verdict**: Remove `clusters` as a separate CLI subgroup. Add `--skip-cluster-labels`
flag to `imas build` for the only case where individual step control matters (skipping
the LLM labeling cost). Fold status into `imas dd status`.

## Target Structure

### Before (19 top-level commands, ~97 leaf commands)

```
imas-codex
├── serve           ✅ Keep
├── graph           ✅ Keep
├── llm             ✅ Keep
├── tunnel          ✅ Keep
├── embed           ✅ Keep
├── config          ✅ Keep
├── hpc             ✅ Keep
├── credentials     ✅ Keep
├── hosts           ✅ Keep
├── facilities      ✅ Keep
├── tools           ✅ Keep
├── release         ✅ Keep
├── discover        ✅ Keep
├── setup-age       ❌ Delete (never implemented)
├── enrich          ❌ Delete (zombie)
├── ingest          ❌ Delete (zombie)
├── ids             ❌ Merge → imas
├── imas            ⚠️ Restructure
└── map             ❌ Merge → imas
```

### After (13 top-level commands)

```
imas-codex
├── serve              No change
├── graph              No change
├── llm                No change
├── tunnel             No change
├── embed              No change
├── config             No change
├── hpc                No change
├── credentials        No change
├── hosts              No change
├── facilities         No change
├── tools              No change
├── release            No change
├── discover           No change
│
└── imas               Unified IMAS command
    │
    ├── dd              DD graph management (was top-level imas build/status/etc.)
    │   ├── build       Build/update IMAS DD graph (clusters included by default)
    │   ├── status      DD statistics + cluster status
    │   ├── search      Semantic search for IMAS paths
    │   ├── version     Show/set DD version
    │   ├── clear       Delete DD nodes
    │   └── path-history  Path version history across DD versions
    │
    ├── map             Mapping pipeline (was top-level map)
    │   ├── run         Run LLM mapping pipeline: imas map run <facility> <ids>
    │   ├── show        Show detailed mapping
    │   ├── status      Show mapping status
    │   ├── validate    Validate mapping paths
    │   └── clear       Remove mapping
    │
    ├── list [facility]            List IDS mappings/recipes (from ids)
    ├── show <facility> <ids>      Show IDS details + epochs (from ids)
    ├── export <facility> <ids>    Export IDS to file (from ids)
    └── epochs <facility>          List structural epochs (from ids)
```

### Command Path Changes

| Before | After |
|--------|-------|
| `imas-codex imas build` | `imas-codex imas dd build` |
| `imas-codex imas status` | `imas-codex imas dd status` |
| `imas-codex imas search <q>` | `imas-codex imas dd search <q>` |
| `imas-codex imas version` | `imas-codex imas dd version` |
| `imas-codex imas clear` | `imas-codex imas dd clear` |
| `imas-codex imas path-history <p>` | `imas-codex imas dd path-history <p>` |
| `imas-codex imas clusters build` | Removed (use `imas dd build`) |
| `imas-codex imas clusters label` | Removed (use `imas dd build`) |
| `imas-codex imas clusters sync` | Removed (use `imas dd build`) |
| `imas-codex imas clusters embed` | Removed (use `imas dd build`) |
| `imas-codex imas clusters status` | `imas-codex imas dd status` (folded in) |
| `imas-codex map run <f> <i>` | `imas-codex imas map run <f> <i>` |
| `imas-codex map show <f> <i>` | `imas-codex imas map show <f> <i>` |
| `imas-codex map status <f>` | `imas-codex imas map status <f>` |
| `imas-codex map validate <f> <i>` | `imas-codex imas map validate <f> <i>` |
| `imas-codex map clear <f> <i>` | `imas-codex imas map clear <f> <i>` |
| `imas-codex ids list` | `imas-codex imas list` |
| `imas-codex ids show <f> <i>` | `imas-codex imas show <f> <i>` |
| `imas-codex ids export <f> <i>` | `imas-codex imas export <f> <i>` |
| `imas-codex ids epochs <f>` | `imas-codex imas epochs <f>` |
| `imas-codex enrich *` | Removed |
| `imas-codex ingest *` | Removed |
| `imas-codex setup-age` | Removed |

---

## Phase 1: Remove Dead Tools

**Goal**: Delete `enrich`, `ingest`, `setup-age`. Clean break.

### 1a. Delete CLI files

| Action | File |
|--------|------|
| Delete | `imas_codex/cli/enrich.py` |
| Delete | `imas_codex/cli/ingest.py` |

### 1b. Remove `setup-age` command

- Delete the `setup_age` function from `imas_codex/cli/utils.py` (lines 187–254)
- Remove `from imas_codex.cli.utils import setup_age` from `__init__.py`
- Remove `main.add_command(setup_age)` from `__init__.py`

### 1c. Update `__init__.py` registration

Remove three registrations:
```python
# DELETE these lines:
from imas_codex.cli.enrich import enrich
from imas_codex.cli.ingest import ingest
main.add_command(enrich)
main.add_command(ingest)
main.add_command(setup_age)
```

### 1d. Update tests

- Remove/update tests referencing `enrich`, `ingest`, `setup-age` commands
- In `tests/test_cli.py`: remove `test_setup_age_help`, update command registration
  assertions

### 1e. Update AGENTS.md

- Remove `## Ingestion` section showing `imas-codex ingest` commands
- Remove any references to `enrich` CLI commands
- Update the help text example in the main CLI group docstring

### 1f. Clean up dead imports

Check if removing `enrich.py` and `ingest.py` leaves orphan code in:
- `imas_codex/agentic/enrich.py` — `discover_nodes_to_enrich()`,
  `batch_enrich_paths()` — these may still be used by the agentic library
  itself, so keep the module but note it's no longer CLI-exposed
- `imas_codex/ingestion/` — check if anything else imports from here

### Verification

```bash
uv run imas-codex --help          # No enrich, ingest, setup-age
uv run imas-codex enrich --help   # Error: No such command 'enrich'
uv run imas-codex ingest --help   # Error: No such command 'ingest'
uv run imas-codex setup-age       # Error: No such command 'setup-age'
uv run pytest tests/test_cli.py   # All pass
```

---

## Phase 2: Absorb `clusters` into `imas build`

**Goal**: Remove `clusters` as a separate subgroup. `imas build` handles everything.

### 2a. Fold cluster status into `imas status`

Add cluster statistics to the `imas status` output:
- Number of IMASSemanticCluster nodes
- Number with labels
- Number with embeddings

This data is already queryable — just add 3 Cypher counts to the existing
`imas status` command.

### 2b. Add `--skip-cluster-labels` flag to `imas build`

The existing `--skip-clusters` flag skips cluster import entirely. Add a finer
control: `--skip-cluster-labels` to skip the LLM labeling step (the only expensive
individual step users might want to skip).

### 2c. Remove clusters registration

- Remove `from imas_codex.cli.clusters import clusters` from `imas_dd.py`
- Remove `imas.add_command(clusters)` from `imas_dd.py`
- Keep `imas_codex/cli/clusters.py` file temporarily (Phase 2 only) — the
  build functions it calls (`ClusterLabeler`, `import_semantic_clusters`, etc.)
  live in `imas_codex/graph/build_dd.py` and `imas_codex/clusters/`, not in
  the CLI file
- Delete `imas_codex/cli/clusters.py` after verifying no imports

### Verification

```bash
uv run imas-codex imas --help            # No clusters subgroup
uv run imas-codex imas build --help      # Has --skip-cluster-labels
uv run imas-codex imas status            # Shows cluster stats
uv run pytest tests/ -k cluster          # All pass
```

---

## Phase 3: Restructure `imas` — Create `dd` Subgroup

**Goal**: Current `imas` commands become `imas dd` subgroup. Clear the `imas`
top-level for IDS assembly commands.

### 3a. Create `dd` subgroup in `imas_dd.py`

```python
@click.group()
def imas():
    """IMAS data dictionary, mappings, and IDS assembly."""

@click.group("dd")
def dd():
    """IMAS Data Dictionary graph management."""

imas.add_command(dd)
```

Move existing commands (`build`, `status`, `search`, `version`, `clear`,
`path-history`) from `imas` group to `dd` subgroup.

### 3b. File restructure

The current `imas_dd.py` (894 lines) becomes the container for both the `imas`
group and the `dd` subgroup. No file split needed — the `dd` subgroup is
defined in the same file.

### 3c. Update help text

```
imas-codex imas --help

  IMAS data dictionary, mappings, and IDS assembly.

  Data Dictionary:
    dd          Manage the IMAS Data Dictionary graph

  Mapping:
    map         Run and manage IMAS field mappings

  Assembly:
    list        List IDS with mappings
    show        Show IDS mapping details
    export      Export IDS to file
    epochs      List structural epochs
```

### Verification

```bash
uv run imas-codex imas --help            # Shows dd, map, list, show, export, epochs
uv run imas-codex imas dd build --help   # Same as old imas build
uv run imas-codex imas dd search core_profiles  # Works
uv run pytest tests/ -k "imas"           # All pass
```

---

## Phase 4: Merge `ids` → `imas` and `map` → `imas map`

**Goal**: Unified `imas` command. Delete `ids.py`, re-register `map.py`.

### 4a. Move `ids` commands to `imas` group

Move four commands from `ids.py` into `imas_dd.py` (the `imas` group):
- `ids list` → `imas list`
- `ids show` → `imas show`
- `ids export` → `imas export`
- `ids epochs` → `imas epochs`

These are simple click commands — copy the function definitions, adjust imports.
The actual logic lives in `imas_codex/ids/assembler.py` (unchanged).

### 4b. Re-register `map` as subgroup of `imas`

In `imas_dd.py`:
```python
from imas_codex.cli.map import map_cmd
imas.add_command(map_cmd, "map")
```

Remove the top-level registration from `__init__.py`:
```python
# DELETE:
from imas_codex.cli.map import map_cmd
main.add_command(map_cmd, "map")
```

### 4c. Delete `ids.py`

After moving commands, delete `imas_codex/cli/ids.py`.

Remove from `__init__.py`:
```python
# DELETE:
from imas_codex.cli.ids import ids
main.add_command(ids)
```

### 4d. Update all references

- `AGENTS.md`: Update command examples
- `docs/`: Update any architecture docs referencing `ids` commands
- `plans/`: Note superseded sections
- Help text in main CLI group docstring: update command list
- MCP tool docstrings that reference CLI commands

### 4e. Update tests

- `tests/test_cli.py`: Update command registration assertions (14 → 13 commands)
- `tests/ids/test_mapping_e2e.py`: Update any CLI invocations that reference
  `map` as top-level (change to `imas map run ...`)
- Search for any test that invokes `ids` CLI commands

### Verification

```bash
uv run imas-codex imas --help                    # Full unified output
uv run imas-codex imas list                      # Works (was: ids list)
uv run imas-codex imas show jet pf_active        # Works (was: ids show)
uv run imas-codex imas export jet pf_active -e p68613  # Works
uv run imas-codex imas epochs jet                # Works
uv run imas-codex imas map run jet pf_active     # Works (was: map run)
uv run imas-codex imas map show jet pf_active    # Works
uv run imas-codex imas dd build --help           # Works
uv run imas-codex imas dd search "electron temperature"  # Works
uv run imas-codex map --help                     # Error: No such command
uv run imas-codex ids --help                     # Error: No such command
uv run pytest tests/ -x -q --ignore=tests/features --ignore=tests/graph_mcp --ignore=tests/agentic
```

---

## Phase 5: Documentation and Cleanup

**Goal**: Update all documentation, remove stale plan references.

### 5a. Update AGENTS.md

- Remove `## Ingestion` section entirely
- Update main CLI help text example to show new structure
- Update any command examples that use old paths

### 5b. Update docs/

- `docs/architecture/ids-mapping.md` — update CLI examples
- `docs/architecture/llamaindex-agents.md` — remove `enrich mark-stale` reference
- Any other docs referencing removed commands

### 5c. Mark plans as superseded

In the following files, add `Status: **Superseded by cli-unification.md**`:
- `plans/features/imas-mapping-combined.md` (Phase 5 CLI section)
- `plans/features/imas-mapping-pipeline.md` (Phase 1c CLI section)

### 5d. Final test pass

```bash
uv run pytest tests/ -x -q --ignore=tests/features --ignore=tests/graph_mcp --ignore=tests/agentic
```

---

## Files

### Delete

| File | Phase | Reason |
|------|-------|--------|
| `imas_codex/cli/enrich.py` | 1 | Zombie — functionality in discover signals |
| `imas_codex/cli/ingest.py` | 1 | Zombie — functionality in discover code |
| `imas_codex/cli/clusters.py` | 2 | Absorbed into imas build |
| `imas_codex/cli/ids.py` | 4 | Merged into imas |

### Modify

| File | Phase | Changes |
|------|-------|---------|
| `imas_codex/cli/__init__.py` | 1, 4 | Remove enrich/ingest/setup-age/ids/map registrations |
| `imas_codex/cli/utils.py` | 1 | Remove setup_age command |
| `imas_codex/cli/imas_dd.py` | 2, 3, 4 | Remove clusters, create dd subgroup, add ids+map commands |
| `imas_codex/cli/map.py` | 4 | No changes (re-registered under imas) |
| `tests/test_cli.py` | 1, 4 | Update command assertions |
| `tests/ids/test_mapping_e2e.py` | 4 | Update CLI invocations |
| `AGENTS.md` | 5 | Remove ingestion section, update examples |

### No Change

All other CLI files (`serve.py`, `graph_cli.py`, `llm_cli.py`, `tunnel.py`,
`config_cli.py`, `compute.py`, `discover/`, `embed.py`, `hosts.py`,
`facilities.py`, `release.py`, `credentials.py`) are untouched.

---

## Open Questions

None. The approach is clear:
- Dead tools get deleted
- Cluster steps get absorbed
- IMAS tools get unified
- No backwards compatibility
