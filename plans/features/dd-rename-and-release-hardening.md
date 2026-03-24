# Feature: dd-only Rename, Release Hardening, and Documentation

## Overview

Three features to prepare for v5.0.0 final release:

1. Rename `imas-only` → `dd-only` across all packages, CLI flags, and docs
2. Harden the release CLI — no silent failures anywhere
3. Fix outdated documentation and add missing content

**Design principles:**

- No backward compatibility for `--imas-only` — clean break
- No silent skips or warnings that swallow failures — every error must be loud
- `--skip-graph` is the only way to bypass graph operations

---

## Background: Artifact Landscape

### GHCR Graph Packages (OCI artifacts — Neo4j dump files)

| Package (after rename) | Contents | Visibility |
|------------------------|----------|------------|
| `imas-codex-graph` | Full graph (all facilities + DD) | Private |
| `imas-codex-graph-dd` | DD-only (no facility data) | Public |
| `imas-codex-graph-tcv` | TCV facility + DD | Private |
| `imas-codex-graph-jet` | JET facility + DD | Private |
| `imas-codex-graph-iter` | ITER facility + DD | Private |

### Docker Container Images (MCP server + embedded graph)

The full graph is the default (no suffix). DD-only gets a `-dd` suffix.

| Variant (after rename) | Graph embedded | RC tags | Release tags |
|------------------------|---------------|---------|-------------|
| **full** (default) | `imas-codex-graph` | `latest-streamable-http` | `prod-streamable-http` |
| **dd-only** | `imas-codex-graph-dd` | `latest-dd-streamable-http` | `prod-dd-streamable-http` |

Both → ACR. Releases also → GHCR.

### Why `-graph` stays in package names

GHCR shares one namespace for Docker images and OCI artifacts. The Docker image
is `ghcr.io/iterorganization/imas-codex`. The full graph OCI artifact is
`ghcr.io/iterorganization/imas-codex-graph`. Dropping `-graph` from the full
variant would collide with the Docker image.

### Azure test server

The Azure test server (`app-imas-mcp-server-test-frc`) should use the **DD
variant** (`latest-dd-streamable-http`). The DD-only graph is public and
sufficient for the test endpoint. The full graph contains private facility data.

**Action:** Update Azure App Service container configuration:
- Image: `crcommonallfrc.azurecr.io/iterorganization/imas-codex`
- Tag: `latest-dd-streamable-http`

This can be done via Azure Portal → App Service → Deployment Center, or:

```bash
az webapp config container set \
  --name app-imas-mcp-server-test-frc \
  --resource-group <rg-name> \
  --container-image-name crcommonallfrc.azurecr.io/iterorganization/imas-codex:latest-dd-streamable-http
```

---

## Feature 1: Remove Flaky Benchmark

**File:** `benchmarks/bench_mcp_facility_tools.py`

### Problem

`time_fetch_by_id` queries for `WikiPage` nodes that don't exist in the DD-only
graph dump used by CI. `self._fetch_id` is always `None`, so the benchmark
silently returns without measuring anything. It adds setup overhead for zero
value and causes CI flakiness.

### Changes

**Remove lines 37–42** (the `_fetch_id` setup in `setup()`):

```python
# DELETE:
        # Find a fetchable content ID (WikiPage or CodeChunk)
        pages = _fixture.graph_client.query(
            "MATCH (w:WikiPage {facility_id: $fac}) RETURN w.id AS id LIMIT 1",
            fac=self.facility,
        )
        self._fetch_id = pages[0]["id"] if pages else None
```

**Remove lines 95–101** (the entire method + section comment):

```python
# DELETE:
    # -- fetch ---------------------------------------------------------------

    def time_fetch_by_id(self):
        """Content retrieval by ID."""
        if not self._fetch_id:
            return  # No wiki pages in dump — measure nothing
        run_tool("fetch", {"id": self._fetch_id})
```

**No other files reference `_fetch_id` or `time_fetch_by_id`.**

---

## Feature 2: Release CLI Hardening

**File:** `imas_codex/cli/release.py`

**Principle:** Every function in the graph pipeline must either succeed or
raise. `--skip-graph` is the only bypass. No `except: pass`, no `⚠ warn +
continue`, no silent degradation.

### 2a: `_get_graph_facilities()` — raise on error (lines 457–469)

**Current code** silently swallows all exceptions and returns `[]`:

```python
def _get_graph_facilities() -> list[str]:
    """Read facility list from GraphMeta to determine if full variant is needed."""
    try:
        from imas_codex.graph import GraphClient
        from imas_codex.graph.meta import get_graph_meta

        with GraphClient() as client:
            meta = get_graph_meta(client)
            if meta:
                return list(meta.get("facilities") or [])
    except Exception:
        pass
    return []
```

**Replace with:**

```python
def _get_graph_facilities() -> list[str]:
    """Read facility list from GraphMeta.

    Raises click.ClickException if Neo4j is unreachable or GraphMeta is missing.
    """
    from imas_codex.graph import GraphClient
    from imas_codex.graph.meta import get_graph_meta

    try:
        with GraphClient() as client:
            meta = get_graph_meta(client)
    except Exception as e:
        raise click.ClickException(
            f"Cannot read graph facilities: {e}\n"
            "  Is Neo4j running? Check: imas-codex graph status\n"
            "  To release without graph: --skip-graph"
        ) from e
    if not meta:
        raise click.ClickException(
            "GraphMeta node not found — graph has no metadata.\n"
            "  Run 'imas-codex graph status' first, or use --skip-graph."
        )
    return list(meta.get("facilities") or [])
```

### 2b: `_validate_graph_privacy()` — raise on error (lines 428–432)

**Current code** warns and continues:

```python
    except Exception as e:
        click.echo(f"  ⚠ Could not validate graph: {e}", err=True)
        click.echo("    Is Neo4j running? Check with: imas-codex graph status")
```

**Replace with:**

```python
    except Exception as e:
        raise click.ClickException(
            f"Graph privacy validation failed: {e}\n"
            "  Is Neo4j running? Check: imas-codex graph status\n"
            "  To release without graph: --skip-graph"
        ) from e
```

### 2c: `_tag_dd_version()` — raise on error (lines 452–454)

**Current code** warns and continues:

```python
    except Exception as e:
        click.echo(f"  ⚠ Could not tag DDVersion: {e}", err=True)
        click.echo("    Is Neo4j running? Check with: imas-codex graph status")
```

**Replace with:**

```python
    except Exception as e:
        raise click.ClickException(
            f"Failed to tag DDVersion: {e}\n"
            "  Is Neo4j running? Check: imas-codex graph status\n"
            "  To release without graph: --skip-graph"
        ) from e
```

### 2d: `_push_graph_variant()` — include stderr in failure (lines 511–514)

**Current code** doesn't capture subprocess output:

```python
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        click.echo(f"  ✗ Failed to push {pkg_name}", err=True)
        return False
```

**Replace with:**

```python
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(
            f"  ✗ Failed to push {pkg_name}: {result.stderr.strip()}", err=True
        )
        return False
```

### 2e: `_push_all_graph_variants()` — track all failures, fail at end (lines 583–684)

This is the core fix. Three current inconsistencies:

| Variant | Current behaviour | Should be |
|---------|------------------|-----------|
| Full graph (line 641–644) | `⚠ warn`, continue | Track failure |
| DD-only (line 657–659) | `raise ClickException`, abort | Track failure |
| Per-facility (line 673) | `⚠ warn`, continue | Track failure |
| Shared dump (line 621–628) | `⚠ warn`, fall back (doomed) | Raise immediately |

**Replace the full function body (lines 583–684) with:**

```python
def _push_all_graph_variants(
    message: str, remote: str, dry_run: bool, git_tag: str | None = None
) -> None:
    """Push all graph variants: dd-only, full, and per-facility.

    Dumps the graph once and reuses the dump for filtered variants.
    Raises click.ClickException if any variant fails.
    """
    registry = _resolve_target_registry(remote)
    if registry:
        click.echo(f"  Target registry: {registry}")

    facilities = _get_graph_facilities()
    failed: list[str] = []

    if not facilities:
        click.echo("\n  Variant 1: IMAS Data Dictionary only")
        if not _push_graph_variant(
            dd_only=True,
            message=message,
            registry=registry,
            version_tag=git_tag,
            dry_run=dry_run,
        ):
            raise click.ClickException(
                "DD-only graph push failed. Check: GHCR_TOKEN set, Neo4j running."
            )
        return

    click.echo("\n  Creating shared graph dump (stops Neo4j once)...")

    if dry_run:
        cached_dump = None
    else:
        cached_dump = _create_shared_dump()
        if not cached_dump:
            raise click.ClickException(
                "Failed to create shared graph dump.\n"
                "  Is Neo4j running? Check: imas-codex graph status\n"
                "  To release without graph: --skip-graph"
            )

    variant = 0

    # Full graph
    variant += 1
    click.echo(
        f"\n  Variant {variant}: Full graph "
        f"(facilities: {', '.join(facilities)})"
    )
    if not _push_graph_variant(
        message=message,
        registry=registry,
        version_tag=git_tag,
        source_dump=cached_dump,
        dry_run=dry_run,
    ):
        failed.append("full")

    # DD-only
    variant += 1
    click.echo(f"\n  Variant {variant}: IMAS Data Dictionary only")
    if not _push_graph_variant(
        dd_only=True,
        message=message,
        registry=registry,
        version_tag=git_tag,
        source_dump=cached_dump,
        dry_run=dry_run,
    ):
        failed.append("dd-only")

    # Per-facility
    for fac in facilities:
        variant += 1
        click.echo(f"\n  Variant {variant}: {fac} + IMAS DD")
        if not _push_graph_variant(
            facility=fac,
            message=message,
            registry=registry,
            version_tag=git_tag,
            source_dump=cached_dump,
            dry_run=dry_run,
        ):
            failed.append(fac)

    # Cleanup
    if cached_dump:
        try:
            Path(cached_dump).unlink(missing_ok=True)
            click.echo("\n  Cleaned up shared dump cache.")
        except OSError:
            pass

    if failed:
        raise click.ClickException(
            f"Graph push failed for {len(failed)} variant(s): "
            f"{', '.join(failed)}.\n"
            "  Check: GHCR_TOKEN set, Neo4j running, network access."
        )
```

### 2f: Module docstring — line 11

**Old:** `3. Push all graph variants (imas-only + full) to GHCR`
**New:** `3. Push all graph variants (dd-only, full, per-facility) to GHCR`

---

## Feature 3: Documentation Updates

### 3a: Outdated command names

| File | Line | Current | Replacement |
|------|------|---------|-------------|
| `AGENTS.md` | 497 | `graph clean --dev` | `graph prune --dev` |
| `README.md` | 704 | `graph list` | `graph tags` |
| `README.md` | 705 | `graph remove --dev` | `graph prune --dev` |
| `README.md` | 706 | `graph remove --backups --older-than 30d` | `graph prune --backups --older-than 30d` |
| `README.md` | 715 | `graph dump --facility tcv` | `graph export --facility tcv` |
| `docs/architecture/graph.md` | 122 | `graph list` | `graph tags` |
| `docs/architecture/graph.md` | 123 | `graph list --facility tcv` | `graph tags --facility tcv` |
| `docs/architecture/graph.md` | 132 | `graph clean tag1 tag2` | `graph prune tag1 tag2` |
| `docs/architecture/graph.md` | 133 | `graph clean --dev` | `graph prune --dev` |
| `docs/architecture/graph.md` | 134 | `graph clean --backups --older-than 30d` | `graph prune --backups --older-than 30d` |

### 3b: Add release workflow section to README

Add after the existing graph CLI documentation section:

```markdown
### Release Workflow

```bash
# Check current release state and permitted commands
imas-codex release status

# Start a major release candidate
imas-codex release --bump major -m "IMAS DD 4.1.0 support"

# Iterate on the RC after fixes
imas-codex release -m "Fix signal mapping edge case"

# Finalize: promote RC to stable release
imas-codex release --final -m "Production release"

# Abandon current RC, start a different bump level
imas-codex release --bump minor -m "New approach"

# Direct release (skip RC)
imas-codex release --bump patch --final -m "Hotfix"

# Preview without executing
imas-codex release --bump major --dry-run -m "Test"
```
```

---

## Feature 4: Rename imas-only → dd-only

**No backward compatibility.** The `--imas-only` flag is removed entirely.

### Inventory: 77 functional occurrences across 14 files

---

### Step A: Core Python source (sequential — call chain dependencies)

#### A1. `imas_codex/graph/ghcr.py` — 4 changes

| Line | Old | New |
|------|-----|-----|
| 170 | `imas_only: bool = False,` | `dd_only: bool = False,` |
| 177 | `imas_only: If True, uses ``imas-codex-graph-imas``...` | `dd_only: If True, uses ``imas-codex-graph-dd``...` |
| 182 | `if imas_only:` | `if dd_only:` |
| 183 | `return "imas-codex-graph-imas"` | `return "imas-codex-graph-dd"` |

#### A2. `imas_codex/graph/temp_neo4j.py` — 3 changes

| Line | Old | New |
|------|-----|-----|
| 22 | `# DD node labels to keep for --imas-only exports` | `# DD node labels to keep for --dd-only exports` |
| 218 | `def create_imas_only_dump(...)` | `def create_dd_only_dump(...)` |
| 262 | `# Update GraphMeta to reflect imas-only content` | `# Update GraphMeta to reflect dd-only content` |

#### A3. `imas_codex/graph/meta.py` — 3 changes

| Line | Old | New |
|------|-----|-----|
| 203 | `imas_only: bool = False,` | `dd_only: bool = False,` |
| 214 | `imas_only: Whether pulling the IMAS-only package.` | `dd_only: Whether pulling the DD-only package.` |
| 226 | `if imas_only:` | `if dd_only:` |

#### A4. `imas_codex/graph/remote.py` — 11 changes

| Line | Old | New |
|------|-----|-----|
| 1021 | `def _build_remote_imas_only_push_script(` | `def _build_remote_dd_only_push_script(` |
| 1029 | `imas-codex graph export --imas-only` | `--dd-only` |
| 1039 | `ARCHIVE="$EXPORTS/imas-codex-graph-imas-push-$$.tar.gz"` | `imas-codex-graph-dd-push-$$.tar.gz` |
| 1047 | `graph export --imas-only -o "$ARCHIVE"` | `--dd-only` |
| 1077 | `imas_only: bool = False,` | `dd_only: bool = False,` |
| 1086 | `When ``imas_only`` is True, delegates...` | `When ``dd_only`` is True...` |
| 1087 | `imas-codex graph export --imas-only` | `--dd-only` |
| 1113 | `# When imas_only, delegate to...` | `# When dd_only...` |
| 1115 | `if imas_only:` | `if dd_only:` |
| 1116 | `return _build_remote_imas_only_push_script(` | `return _build_remote_dd_only_push_script(` |

#### A5. `imas_codex/cli/graph/data.py` — 7 changes

| Line | Old | New |
|------|-----|-----|
| 33 | `create_imas_only_dump as _create_imas_only_dump,` | `create_dd_only_dump as _create_dd_only_dump,` |
| 241 | `"--imas-only",` | `"--dd-only",` |
| 272 | `imas_only: bool,` | `dd_only: bool,` |
| 292 | `imas_only=imas_only` | `dd_only=dd_only` |
| 422 | `# If imas-only, remove all facility nodes` | `# If dd-only, remove all facility nodes` |
| 423 | `if imas_only:` | `if dd_only:` |
| 424 | `_create_imas_only_dump(` | `_create_dd_only_dump(` |

#### A6. `imas_codex/cli/graph/registry.py` — 16 changes

Three Click commands define `--imas-only` options. Each has the option string,
function parameter, docstring, and usage sites.

**`graph_push` command (~line 58):**

| Line | Old | New |
|------|-----|-----|
| 58 | `"--imas-only",` | `"--dd-only",` |
| 93 | `imas_only: bool,` | `dd_only: bool,` |
| 102 | `Use --imas-only to push only IMAS...` | `Use --dd-only...` |
| 122 | `imas_only=imas_only` | `dd_only=dd_only` |
| 156 | `if imas_only:` | `if dd_only:` |
| 200 | `imas_only=imas_only,` | `dd_only=dd_only,` |
| 262 | `if imas_only:` | `if dd_only:` |
| 263 | `dump_args.append("--imas-only")` | `dump_args.append("--dd-only")` |

**`graph_pull` command (~line 375):**

| Line | Old | New |
|------|-----|-----|
| 375 | `"--imas-only",` | `"--dd-only",` |
| 391 | `imas_only: bool,` | `dd_only: bool,` |
| 418 | `imas_only=imas_only` | `dd_only=dd_only` |

**`graph_fetch` command (~line 558):**

| Line | Old | New |
|------|-----|-----|
| 558 | `"--dd-only",` | (already correct after rename) |
| 570 | `imas_only: bool,` | `dd_only: bool,` |
| 598 | `imas_only=imas_only` | `dd_only=dd_only` |
| 620 | `imas_only=imas_only,` | `dd_only=dd_only,` |

#### A7. `imas_codex/cli/release.py` — 12 changes

| Line | Old | New |
|------|-----|-----|
| 11 | `(imas-only + full)` | `(dd-only, full, per-facility)` |
| 474 | `imas_only: bool = False,` | `dd_only: bool = False,` |
| 489 | `imas_only=imas_only` | `dd_only=dd_only` |
| 497 | `if imas_only:` | `if dd_only:` |
| 498 | `cmd.append("--imas-only")` | `cmd.append("--dd-only")` |
| 586 | `imas-only, full, and per-facility` | `dd-only, full, and per-facility` |
| 590 | `imas-only and per-facility` | `dd-only and per-facility` |
| 600 | `# No facilities — just push imas-only` | `dd-only` |
| 603 | `imas_only=True,` | `dd_only=True,` |
| 617 | `# Step 3: Push imas-only and...` | `dd-only` |
| 649 | `# Push imas-only (filtered...)` | `dd-only` |
| 653 | `imas_only=True,` | `dd_only=True,` |

---

### Step B: CI/Docker (independent — can run in parallel with A)

#### B1. `.github/workflows/docker-build-push.yml` — 11 changes

**Matrix suffix swap:** Full becomes the default (no suffix), DD gets `-dd`.

| Line | Old | New |
|------|-----|-----|
| 77 | `IMAS_PACKAGE="imas-codex-graph-imas"` | `IMAS_PACKAGE="imas-codex-graph-dd"` |
| 84 | `echo "No 'latest' tag for imas-codex-graph-imas..."` | `imas-codex-graph-dd` |
| 115 | `ARTIFACT="${REGISTRY}/imas-codex-graph-imas:..."` | `imas-codex-graph-dd` |
| 311 | `GRAPH_PACKAGE=imas-codex-graph-imas` | `imas-codex-graph-dd` |
| 416 | `- name: imas-only` | `- name: dd-only` |
| 417 | `package: imas-codex-graph-imas` | `package: imas-codex-graph-dd` |
| 418 | `suffix: ""` | `suffix: "-dd"` |
| 420 | `package: imas-codex-graph` | (unchanged) |
| 421 | `suffix: "-full"` | `suffix: ""` |
| 448 | `= "imas-only"` | `= "dd-only"` |

This changes the generated Docker tags as follows:

| Variant | Old RC tag | New RC tag |
|---------|-----------|-----------|
| dd-only | `latest-streamable-http` | `latest-dd-streamable-http` |
| full | `latest-full-streamable-http` | `latest-streamable-http` |

| Variant | Old release tags | New release tags |
|---------|-----------------|-----------------|
| dd-only | `5.0.0-streamable-http`, `prod-streamable-http` | `5.0.0-dd-streamable-http`, `prod-dd-streamable-http` |
| full | `5.0.0-full-streamable-http`, `prod-full-streamable-http` | `5.0.0-streamable-http`, `prod-streamable-http` |

#### B2. `Dockerfile` — 1 change

The default `GRAPH_PACKAGE` stays as the DD variant since that's what gets
built when no build arg is provided (public/lightweight default):

| Line | Old | New |
|------|-----|-----|
| 131 | `ARG GRAPH_PACKAGE="imas-codex-graph-imas"` | `ARG GRAPH_PACKAGE="imas-codex-graph-dd"` |

#### B3. `docker-compose.yml` — 1 change

| Line | Old | New |
|------|-----|-----|
| 38 | `GRAPH_PACKAGE=${GRAPH_PACKAGE:-imas-codex-graph-imas}` | `${GRAPH_PACKAGE:-imas-codex-graph-dd}` |

#### B4. `.github/workflows/graph-quality.yml` — 0 changes (verified)

#### B5. `.github/workflows/benchmark.yml` — 0 changes (verified)

---

### Step C: Documentation (independent — can run in parallel with A)

#### C1. `README.md` — 2 changes

| Line | Old | New |
|------|-----|-----|
| 640 | `graph pull --imas-only` | `graph pull --dd-only` |
| 741 | `imas-codex-graph-imas` | `imas-codex-graph-dd` |

#### C2. `AGENTS.md` — 1 change

| Line | Old | New |
|------|-----|-----|
| 647 | `(imas-only + full + per-facility)` | `(dd-only + full + per-facility)` |

#### C3. `plans/features/release-benchmark-pipeline.md` — 2 changes

| Line | Old | New |
|------|-----|-----|
| 173 | `imas-codex-graph-imas` | `imas-codex-graph-dd` |
| 434 | `PACKAGE="imas-codex-graph-imas"` | `PACKAGE="imas-codex-graph-dd"` |

---

### Step D: Tests (depends on A — param names must match)

#### D1. `tests/core/test_ghcr.py` — 2 changes

| Line | Old | New |
|------|-----|-----|
| 180 | `def test_imas_only(self):` | `def test_dd_only(self):` |
| 181 | `get_package_name(imas_only=True) == "imas-codex-graph-imas"` | `get_package_name(dd_only=True) == "imas-codex-graph-dd"` |

#### D2. `tests/llm/test_health_endpoint.py` — 4 changes

| Line | Old | New |
|------|-----|-----|
| 205 | `def test_imas_only_graph_no_facilities(...)` | `def test_dd_only_graph_no_facilities(...)` |
| 206 | `"""IMAS-only graph reports...` | `"""DD-only graph reports...` |
| 211 | `def imas_only_query(query, **kwargs):` | `def dd_only_query(query, **kwargs):` |
| 228 | `gc.query.side_effect = imas_only_query` | `gc.query.side_effect = dd_only_query` |

#### D3. `tests/graph_mcp/test_docker_embedded.py` — 3 changes

| Line | Old | New |
|------|-----|-----|
| 51 | `def test_oras_pull_imas_only(self):` | `def test_oras_pull_dd_only(self):` |
| 52 | `"""Pulls IMAS-only graph package...` | `"""Pulls DD-only graph package...` |
| 53 | `"imas-codex-graph-imas" in self.content` | `"imas-codex-graph-dd" in self.content` |

---

### Parallelization plan for agents

```
Agent A: Steps A1–A7 + D1–D3 (Python source + tests)
         Sequential within — the import chain is:
         ghcr.py ← meta.py ← temp_neo4j.py ← data.py ← registry.py ← release.py
         Tests depend on A1 (get_package_name signature)

Agent B: Steps B1–B5 (CI/Docker)
         All YAML/Dockerfile changes. No Python dependencies.

Agent C: Steps C1–C3 (Documentation)
         All markdown changes. No code dependencies.
```

All three agents can run in parallel. Each agent gets the full inventory
table for their steps and can execute without ambiguity.

---

## Implementation Order

| # | Feature | Dependency | Files changed |
|---|---------|-----------|---------------|
| 1 | Remove flaky benchmark | None | 1 file |
| 2 | Release CLI hardening (2a–2f) | None | 1 file |
| 3 | Documentation fixes (3a–3b) | None | 4 files |
| 4 | dd-only rename (A+B+C+D parallel) | After 2 (release.py changes overlap) | 14 files |

Features 1, 2, and 3 have no interdependencies and can run in parallel.
Feature 4 depends on Feature 2 completing first (both modify `release.py`).
