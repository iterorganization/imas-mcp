# Feature: dd-only Rename, Release Hardening, and Documentation

## Overview

Three related features to prepare for v5.0.0 final release:
1. Rename `imas-only` → `dd-only` across all packages, CLI flags, and docs
2. Harden the release CLI against silent failures
3. Fix outdated documentation and add missing content

## Background: Current Artifact Landscape

### GHCR Graph Packages (OCI artifacts — Neo4j dump files)

| Package | Contents | Visibility |
|---------|----------|------------|
| `imas-codex-graph` | Full graph (all facilities + DD) | Private |
| `imas-codex-graph-dd` | DD-only (no facility data) | Public |
| `imas-codex-graph-tcv` | TCV facility + DD | Private |
| `imas-codex-graph-jet` | JET facility + DD | Private |
| `imas-codex-graph-iter` | ITER facility + DD | Private |

### Docker Container Images (full application + embedded graph)

The CI builds **two Docker container variants**, each containing the full
`imas-codex` application with an embedded graph pre-loaded into Neo4j:

| Variant | Graph data | Image tags (RC) | Image tags (release) |
|---------|-----------|-----------------|---------------------|
| **dd-only** (default) | `imas-codex-graph-dd` | `latest-streamable-http` | `prod-streamable-http`, `5.0.0-streamable-http` |
| **full** | `imas-codex-graph` | `latest-full-streamable-http` | `prod-full-streamable-http`, `5.0.0-full-streamable-http` |

Both are pushed to ACR (`crcommonallfrc.azurecr.io/iterorganization/imas-codex`).
Release builds also push to GHCR (`ghcr.io/iterorganization/imas-codex`).

The Azure test server should use the **dd-only** variant (`latest-streamable-http`)
— the public IMAS DD graph is sufficient for the test endpoint, and the full
graph contains private facility data.

**Key insight:** The Docker containers are NOT just the graph — they are the
complete `imas-codex` MCP server with Neo4j + graph data baked in.

### Naming convention: `-graph` is required for the full variant

GHCR uses a single namespace for both Docker images and OCI artifacts.
The Docker image is `ghcr.io/iterorganization/imas-codex`. The full graph
OCI artifact is `ghcr.io/iterorganization/imas-codex-graph`. Removing `-graph`
from the full variant would collide with the Docker image — both would resolve
to the same GHCR package. **Keep `-graph` for the full variant.**

DD-only and per-facility packages (`imas-codex-graph-dd`, `imas-codex-graph-tcv`)
have no collision risk. The `-graph` infix also serves as self-documentation —
any `imas-codex-graph-*` package is a Neo4j dump, not a Docker image.

### Azure test server

The Azure test server (`app-imas-mcp-server-test-frc`) should use the **DD
variant** (`latest-streamable-http`), not the full variant. The DD-only graph
is public and contains the IMAS Data Dictionary — sufficient for the public
test endpoint. The full graph contains private facility data and should only
run in authenticated environments.

**Action:** Update the Azure App Service container configuration to pull the
DD variant tag (`latest-streamable-http`) instead of `latest-full-streamable-http`.

---

## Feature 1: Remove Flaky Benchmark

**File:** `benchmarks/bench_mcp_facility_tools.py`

`FacilityToolBenchmarks.time_fetch_by_id` silently returns `None` when WikiPage
data is absent in the graph dump. It only measures Neo4j query latency (trivial,
already covered by `bench_graph_queries.py`).

### Steps
- [ ] Remove `time_fetch_by_id` method
- [ ] Remove `_fetch_id` setup from the class `setup()` method

**Effort:** 1 file, ~10 lines removed.

---

## Feature 2: Release CLI Hardening

**File:** `imas_codex/cli/release.py`

### 2a: Warn on empty facility list

`_get_graph_facilities()` (line 457-469) silently returns `[]` on any exception.
If Neo4j is unreachable, the release pushes only the dd-only variant without
warning — an incomplete release.

**Fix:** Log a warning and require `--skip-graph` to proceed without facilities.

### 2b: Track and report failed variants

Per-facility push failures log inline warnings but `release` reports full success
at the end. Users may miss the warnings.

**Fix:** Count failures in `_push_all_graph_variants()`, print summary at end.

### 2c: Make dd-only failure non-fatal

dd-only push failure currently raises `click.ClickException` (line 660-662),
aborting the entire release. The full graph already contains all DD data.

**Fix:** Change to warning, continue like per-facility failures.

### 2d: Fix module docstring

Line 3 says "imas-only + full" but the CLI also pushes per-facility variants.

**Effort:** ~30 lines changed in release.py.

---

## Feature 3: Documentation Updates

### 3a: Fix outdated command names

| Location | Wrong | Correct |
|----------|-------|---------|
| README.md ~line 704 | `graph list` | `graph tags` |
| README.md ~line 705 | `graph remove --dev` | `graph prune` |
| README.md ~line 715 | `graph dump` | `graph export` |
| docs/architecture/graph.md ~line 122 | `graph list` | `graph tags` |
| docs/architecture/graph.md ~line 132 | `graph clean` | `graph prune` |

### 3b: Add missing content to README

- [ ] `release status` subcommand example
- [ ] `-m/--message` requirement in release examples
- [ ] Dump-once optimization description
- [ ] Full release workflow example (start RC → iterate → finalize)

### 3c: Release CLI usage examples

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
imas-codex release --bump minor -m "New approach for signal discovery"

# Direct release (skip RC entirely)
imas-codex release --bump patch --final -m "Hotfix for COCOS transform"

# Preview what would happen
imas-codex release --bump major --dry-run -m "Test"
```

**Effort:** ~50 lines across 3 files.

---

## Feature 4: Rename imas-only → dd-only

### Scope: ~90 line changes across 18 files

#### Step A: Core Python source (must be sequential)

| File | Changes |
|------|---------|
| `imas_codex/graph/ghcr.py` | `get_package_name()`: param `imas_only→dd_only`, return `"imas-codex-graph-dd"` |
| `imas_codex/graph/temp_neo4j.py` | Rename `create_imas_only_dump()` → `create_dd_only_dump()` |
| `imas_codex/graph/meta.py` | Rename `imas_only` param → `dd_only` |
| `imas_codex/cli/graph/registry.py` | `--imas-only` → `--dd-only` on push/fetch/pull (6 locations) |
| `imas_codex/cli/graph/data.py` | `--imas-only` → `--dd-only` on export |
| `imas_codex/cli/release.py` | Update params and CLI arg forwarding |
| `imas_codex/graph/remote.py` | Rename `_build_remote_imas_only_push_script()`, update archive names |

#### Step B: CI/Docker (parallel with A)

| File | Changes |
|------|---------|
| `.github/workflows/docker-build-push.yml` | Package `imas-codex-graph-imas` → `imas-codex-graph-dd`, matrix name `imas-only` → `dd-only` |
| `.github/workflows/graph-quality.yml` | Update package name in artifact ref |
| `.github/workflows/benchmark.yml` | Update if referencing package name |
| `Dockerfile` | Default `GRAPH_PACKAGE` ARG |
| `docker-compose.yml` | Default `GRAPH_PACKAGE` |

#### Step C: Documentation (parallel with A)

| File | Changes |
|------|---------|
| `README.md` | Package table, CLI examples, setup instructions |
| `DOCKER.md` | Build examples |
| `docs/architecture/graph.md` | Package table |
| `AGENTS.md` | References |

#### Step D: Tests + backward compat (depends on A)

| File | Changes |
|------|---------|
| `tests/core/test_ghcr.py` | Assertions and param names |
| `tests/llm/test_health_endpoint.py` | Test method names |
| `tests/graph_mcp/test_docker_embedded.py` | Assertion string |

**Backward compatibility:** Add hidden `--imas-only` Click alias that maps to
`--dd-only` with a deprecation warning. Remove after v6.0.0.

### Parallelization for agents

```
Agent A: Steps A + D (Python + tests) — sequential, core logic
Agent B: Step B (CI/Docker) — independent
Agent C: Step C (Documentation) — independent
```

---

## Implementation Order

| # | Feature | Dependency | Effort |
|---|---------|-----------|--------|
| 1 | Remove flaky benchmark | None | 10 min |
| 2 | Release CLI hardening | After 1 | 30 min |
| 3 | Documentation updates | After 1 | 30 min |
| 4 | dd-only rename (3 parallel tracks) | After 2, 3 | 60 min |
| 5 | v5.0.0 finalization | After 4 | 15 min |

Features 2 and 3 can run in parallel after Feature 1.
Feature 4 Steps B and C can run in parallel with Step A.
