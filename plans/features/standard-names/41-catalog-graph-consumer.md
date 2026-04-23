# Plan 41 — Lightweight Catalog Graph Consumer (NetworkX + Renderer)

> Follow-up to plan 40. Deferred from 40 per reviewer feedback to
> isolate a three-repo format migration from downstream consumer
> work.

## Problem

Plan 40 migrates the ISNC catalog to one-file-per-domain with inline
hierarchy fields (`parent`, `components`, `real_part`, `imag_part`,
`uncertainty_siblings`). These fields exist so downstream consumers
can reconstruct a local graph from YAML alone — but plan 40 does not
ship such a consumer.

Today:
- The ISN library has no in-memory graph; only
  `graphlib.TopologicalSorter` for load ordering.
- The ISN MCP server exists (`standard-names-mcp` entry point) but
  has no graph-traversal tools.
- The catalog-site mkdocs renderer ignores the `links:` field, does
  not emit `cocos_transformation_type`, has no per-entry hierarchy
  display, and produces a flat alphabetic nav.
- An external tool wanting to answer "what are the components of
  this vector?" must either parse name grammar or spin up a Neo4j
  instance against the imas-codex graph.

## Approach

Add a minimal NetworkX-based graph module and MCP tools to the ISN
library. Build the graph from YAML alone — no Neo4j dependency.
Extend the catalog-site renderer to display hierarchy blocks
(Mermaid), resolve `links:` as hyperlinks, and emit
`cocos_transformation_type`. Generate a structured `nav:` so sibling
names appear adjacent in the sidebar.

Plan 41 depends on plan 40 being merged (the inline hierarchy fields
and per-domain file layout must be in place before NetworkX can
consume them).

## Scope

### A — imas-standard-names library

#### 1. NetworkX local graph — `imas_standard_names/graph/local_graph.py` (new)

```python
def build_catalog_graph(catalog_root: Path) -> nx.DiGraph: ...
def get_neighbours(g, name, rel_type=None) -> list[str]: ...
def get_ancestors(g, name) -> list[str]: ...
def get_descendants(g, name) -> list[str]: ...
def shortest_path(g, a, b) -> list[str]: ...
```

- Reads all `standard_names/<domain>.yml` files.
- Adds one node per entry with attrs
  `{kind, unit, domain, description, …}`.
- Adds typed edges:
  - `parent` / `components` → `COMPONENT_OF`
  - `real_part` / `imag_part` → `REAL_PART_OF` / `IMAGINARY_PART_OF`
  - `uncertainty_siblings` → `UNCERTAINTY_OF`
  - `deprecates` / `superseded_by` → corresponding edge types
  - `links` → `REFERENCES`
- Edge type stored as attribute `kind`; helpers filter on it.
- Graph is in-memory; ~480 nodes today — trivial footprint.

Dependency: `networkx >= 3.0` added as
`[project.optional-dependencies].graph-local`. No version conflict
expected; ISN runtime has no numerical dependencies.

#### 2. MCP tool wiring — `imas_standard_names/cli/server.py`

Add four new tools:

- `get_neighbours(name: str, rel_type: str | None = None)` →
  list of neighbours (optionally filtered by edge type).
- `get_ancestors(name: str)` → all ancestors via `COMPONENT_OF` ∪
  `REAL_PART_OF` ∪ `IMAGINARY_PART_OF` ∪ `UNCERTAINTY_OF`.
- `get_descendants(name: str)` → inverse.
- `shortest_path(a: str, b: str)` → shortest structural path.

Guarded by the `[graph-local]` extra — tools are registered only if
networkx is importable. Server startup logs whether the graph is
loaded.

#### 3. Catalog-site renderer upgrades — `imas_standard_names/rendering/catalog.py`

- Emit `links: [name:X]` as anchor hyperlinks `[X](#X)` (the audit
  found the field is already loaded but never rendered).
- Emit `cocos_transformation_type: <type>` as an inline line in the
  entry block.
- Emit per-entry Mermaid block when any hierarchy field is present:

  ```mermaid
  graph LR
    parent_name --> this_name
    this_name --> component_r
    this_name --> component_z
  ```

- Entry sibling navigation: at the bottom of each entry page, list
  `parent`, `components`, `real_part` / `imag_part`,
  `uncertainty_siblings` as clickable links.

#### 4. Structured `nav:` generation — `imas_standard_names/cli/catalog_site.py`

Replace the flat alphabetic nav with a structured nav generated
from the per-domain YAML files. For each domain, list entries in
the same order they appear in the file (which plan 40 has made
hierarchy-aware). Vectors and their components appear adjacent in
the sidebar.

#### 5. Mermaid plugin

Add `mkdocs-mermaid2-plugin` to the ISN docs-extras group and to
the inline `MKDOCS_DEPLOY_TEMPLATE`. Verify rendering on the test
catalog-site deploy.

#### 6. Tests

- `build_catalog_graph` against a fixture catalog returns the
  expected node + edge set.
- `get_ancestors` on a grandchild returns the full chain.
- `shortest_path` returns the right hop sequence.
- MCP tool integration test (in-process server + tool call).
- Renderer snapshot: one domain page with Mermaid blocks, one
  entry with `links`, one entry with `cocos_transformation_type`.
- Structured-nav snapshot: sibling names adjacent.

#### 7. Documentation

- `AGENTS.md` (ISN) — new section on the NetworkX local graph +
  MCP tools.
- `CONTRIBUTING.md` — how to use `get_ancestors` /
  `get_descendants` locally for quick graph inspection.
- `mkdocs.yml` — mermaid plugin; structured nav entry point.

### B — imas-codex repo

No codex-side changes. Plan 41 is entirely downstream of plan 40.

## Out of scope

- **Cluster-membership manifest.** NetworkX graph does not need
  cluster info for the four shipped tools. A separate manifest
  file + loader extension is a future plan if cluster navigation
  becomes needed catalog-side.
- **Interactive D3.js graph widgets.** Mermaid is enough for v1;
  D3 is a separate future plan.
- **Cross-domain navigation / search UI.** mkdocs builtin search
  is adequate.

## Rollout

1. **NetworkX module + tests.** Land in ISN.
2. **MCP tool wiring + tests.**
3. **Catalog-site renderer upgrades + snapshot tests.**
4. **Structured nav generator.**
5. **mkdocs plugin + docs commits.**
6. **Release ISN version bump** (`v0.8.x → v0.9.0` or equivalent).
7. **Catalog-site deploy** via `mike` to the `latest` channel.

## Revision notes

Split from plan 40 per Opus review (#5). Original plan 40 bundled a
three-repo format migration with a new downstream consumer; the
format migration has higher blast radius and different rollback
semantics, so it ships first. Plan 41 can then be developed and
released independently without gating on 40's stability.

## Documentation updates

- `AGENTS.md` (ISN) — NetworkX module + MCP tools section.
- `CONTRIBUTING.md` (ISN) — graph-local workflow.
- `mkdocs.yml` — mermaid + structured nav.
- `plans/README.md` — add plan 41.
