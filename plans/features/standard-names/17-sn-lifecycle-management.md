# 17: Standard Name Lifecycle Management (Reset / Clear)

**Status:** Ready to implement
**Depends on:** None (standalone)
**Agent:** engineer

## Problem

All discovery domains (signals, paths, wiki, code, documents) have `--reset-to`
infrastructure via the shared `ResetSpec` / `reset_to_status()` pattern. Standard
names have none. This makes iterative development painful — after changing prompts
or models, there's no way to re-run the pipeline without manual Cypher cleanup.

### Key difference from other domains

StandardName nodes are **cross-facility** — they have no `facility_id` property.
The standard `reset_to_status()` function requires a facility parameter. We need
an adapted approach.

### StandardName lifecycle

```
drafted → published → accepted
                    ↘ rejected
                    ↘ skipped
```

## Phase 1: Scoped reset command

**Files:**
- `imas_codex/sn/graph_ops.py` — add `reset_standard_names()` and `clear_standard_names()`
- `imas_codex/cli/sn.py` — add `sn reset` subcommand

### `sn reset` command

```bash
# Reset all drafted names back for re-composition (clears LLM output, keeps nodes)
imas-codex sn reset --status drafted

# Reset published names back to drafted (e.g., after prompt change)
imas-codex sn reset --status published --to drafted

# Reset only DD-sourced names
imas-codex sn reset --status drafted --source dd

# Reset names from a specific IDS
imas-codex sn reset --status drafted --ids equilibrium

# Dry run
imas-codex sn reset --status drafted --dry-run
```

### Graph operation

```python
def reset_standard_names(
    *,
    target_status: str = "drafted",
    source_statuses: list[str] | None = None,
    source_filter: str | None = None,  # "dd" or "signals"
    ids_filter: str | None = None,
    clear_embeddings: bool = True,
) -> int:
    """Reset StandardName nodes to a target review_status.

    Unlike facility-scoped domains, StandardName has no facility_id.
    Filtering is by review_status, source, and IDS.
    """
```

Fields to clear on reset to `drafted`:
- `embedding`, `embedded_at` (will be regenerated)
- `model`, `generated_at` (provenance of old generation)
- `confidence` (will be re-scored)

Fields to preserve:
- `id` (the standard name itself)
- `source`, `source_path` (how it was sourced)
- `created_at` (first creation time)

Relationships to clean:
- `HAS_STANDARD_NAME` — remove (will be re-created on persist)
- `CANONICAL_UNITS` — remove (will be re-created)

## Phase 2: Clear command

**Files:** `imas_codex/sn/graph_ops.py`, `imas_codex/cli/sn.py`

```bash
# Delete all drafted standard names (not accepted/imported ones)
imas-codex sn clear --status drafted

# Delete ALL standard names (requires --confirm)
imas-codex sn clear --all --confirm

# Delete names from a specific source
imas-codex sn clear --source dd --status drafted
```

### Graph operation

```python
def clear_standard_names(
    *,
    status_filter: list[str] | None = None,
    source_filter: str | None = None,
    ids_filter: str | None = None,
    confirm_all: bool = False,
) -> int:
    """Delete StandardName nodes and their relationships.

    Refuses to delete accepted/imported names unless confirm_all=True.
    """
```

Safety rules:
- Default: only delete `drafted` names
- `accepted` and `imported` names require `--confirm` flag
- Always log count before deletion
- DETACH DELETE (removes all relationships)

## Phase 3: Wire into build command

**Files:** `imas_codex/cli/sn.py`

Add `--reset-to` option to `sn mint`:

```bash
# Re-compose all drafted names from scratch (reset + build)
imas-codex sn mint --source dd --reset-to drafted

# Full pipeline from extraction (clears and rebuilds)
imas-codex sn mint --source dd --reset-to extracted --ids equilibrium
```

Reset targets for `sn mint`:
- `extracted` — clear all SN nodes for matching source, re-run full pipeline
- `drafted` — reset existing drafted names, re-compose them

```python
@click.option(
    "--reset-to",
    type=click.Choice(["extracted", "drafted"]),
    default=None,
    help="Reset standard names to target state before building.",
)
```

## Acceptance criteria

1. `sn reset --status drafted` resets nodes and clears embeddings
2. `sn clear --status drafted` deletes only drafted names
3. `sn clear --all` requires `--confirm` flag
4. `sn mint --reset-to drafted` resets then rebuilds
5. Accepted/imported names are never touched without explicit confirmation
6. `sn status` shows correct counts after reset/clear

## Test plan

- Unit test: `reset_standard_names()` clears correct fields
- Unit test: `clear_standard_names()` refuses to delete accepted without confirm
- Unit test: `--reset-to` on build triggers reset before pipeline
- Integration: build → reset → rebuild cycle produces valid results
