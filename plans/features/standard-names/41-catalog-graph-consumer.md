# Plan 41 — Lightweight Catalog Graph Consumer (NetworkX + Renderer)

> **Rewritten** after plan 39 shipped (commit `0d5674a3`).  The catalog
> edges to surface are `HAS_ARGUMENT` / `HAS_ERROR` /
> `HAS_PREDECESSOR` / `HAS_SUCCESSOR` — as emitted by plan 40's
> per-domain YAML files (`arguments` and `error_variants` inline
> fields plus scalar `deprecates` / `superseded_by`).

## Problem

Plan 40 migrates the ISNC catalog to one-file-per-domain with inline
hierarchy fields (`arguments`, `error_variants`) plus the existing
scalar deprecation fields (`deprecates`, `superseded_by`).  These
fields exist so downstream consumers can reconstruct a local graph
from YAML alone — but plan 40 does not ship such a consumer.

Today:

- The ISN library has no in-memory graph; only
  `graphlib.TopologicalSorter` for load ordering.
- The ISN MCP server (`standard-names-mcp` entry point) has no
  graph-traversal tools.
- The catalog-site mkdocs renderer ignores the `links:` field, does
  not emit `cocos_transformation_type`, has no per-entry hierarchy
  display, and produces a flat alphabetic nav.
- An external tool wanting to answer "what arguments does this name
  take?" or "which names wrap this base as an argument?" must either
  re-parse name grammar or spin up a Neo4j instance against the
  imas-codex graph.

## Approach

Add a minimal NetworkX-based graph module to the ISN library.  Build
the graph from YAML alone — no Neo4j dependency, no live grammar
parsing.  Wire new MCP traversal tools.  Extend the catalog-site
renderer to display hierarchy blocks (Mermaid), resolve `links:` as
hyperlinks, emit `cocos_transformation_type`, and generate a
structured `nav:` so sibling names appear adjacent.

Plan 41 depends on plan 40 being merged — the inline hierarchy fields
and per-domain file layout must be in place before NetworkX can
consume them.

## Scope

### A — imas-standard-names library

#### 1. NetworkX local graph — `imas_standard_names/graph/local_graph.py` (new)

```python
import networkx as nx
from pathlib import Path
from typing import Literal

EdgeType = Literal[
    "HAS_ARGUMENT",
    "HAS_ERROR",
    "HAS_PREDECESSOR",
    "HAS_SUCCESSOR",
    "REFERENCES",
]

def build_catalog_graph(catalog_root: Path) -> nx.DiGraph: ...
def get_neighbours(
    g: nx.DiGraph,
    name: str,
    edge_types: set[EdgeType] | None = None,
    direction: Literal["out", "in", "both"] = "both",
) -> list[dict]: ...
def get_ancestors(
    g: nx.DiGraph, name: str, edge_types: set[EdgeType] | None = None,
) -> list[str]: ...
def get_descendants(
    g: nx.DiGraph, name: str, edge_types: set[EdgeType] | None = None,
) -> list[str]: ...
def shortest_path(
    g: nx.DiGraph, a: str, b: str,
) -> list[dict]: ...
```

- Reads all `standard_names/<domain>.yml` files.
- Adds one node per entry with attrs
  `{kind, unit, domain, description, …}`.
- Adds typed edges with full property dict, mirroring the codex
  Neo4j edge schema:

  | YAML source | Edge | From | To | Properties emitted |
  |-------------|------|------|-----|--------------------|
  | `arguments: [{name, operator, …}]` | `HAS_ARGUMENT` | entry | `name` | `{operator, operator_kind, role?, separator?, axis?, shape?}` |
  | `error_variants: {upper, lower, index}` | `HAS_ERROR` | entry | variant | `{error_type}` |
  | `deprecates: X` | `HAS_PREDECESSOR` | entry | `X` | `{}` |
  | `superseded_by: X` | `HAS_SUCCESSOR` | entry | `X` | `{}` |
  | `links: [name:X]` | `REFERENCES` | entry | `X` | `{}` |

- `DiGraph` matches the shipped Neo4j semantics: the codex writer
  issues `MERGE (src)-[:HAS_ARGUMENT]->(tgt)` with no edge key, so
  the graph already collapses duplicate `(src, tgt)` edges to a
  single last-write-wins edge.  Binary operators emit two
  `HAS_ARGUMENT` edges with **distinct targets** (role=a, role=b),
  so `DiGraph` represents them faithfully without multi-edge
  support.  The pathological `ratio_of_X_to_X` case collapses to a
  single edge in both Neo4j and the NetworkX representation —
  accepted.
- Edge type is stored under edge attribute `edge_type`; helpers
  filter on it.
- Forward-reference targets (`arguments[].name` pointing at a name
  absent from the catalog) are added as **stub nodes** with attr
  `stub=True`, matching the codex Neo4j behaviour where
  `_write_standard_name_edges` MERGEs bare placeholders for
  forward-refs.
- Graph is in-memory; ~480 nodes today — trivial footprint.

Dependency: `networkx >= 3.0` added as
`[project.optional-dependencies].graph-local`.  No version conflict
expected; ISN runtime has no numerical dependencies.

**`HAS_ERROR` direction note.**  Per the shipped derivation module,
`HAS_ERROR` is emitted with direction `inner → uncertainty_form`
(i.e.  base → variant).  The YAML layout matches: `error_variants`
appears on the base's entry, not on the variant entry, so the edge
goes base-entry → variant.  Tool documentation and catalog-site
copy must state this convention plainly — the naming is
base-centric, not variant-centric.

#### 2. MCP tool wiring — `imas_standard_names/cli/server.py`

Add four new tools:

- `get_neighbours(name: str, edge_types: list[str] | None = None,
  direction: str = "both")` — list of `{neighbour, edge_type,
  direction, props}` dicts.  Optional filter on edge type set and
  direction.
- `get_ancestors(name: str)` — all ancestors via
  `HAS_ARGUMENT`-outgoing (transitive: my arg's arg's arg…) ∪
  `HAS_ERROR`-incoming (the base I'm an uncertainty-variant of).
  Returns the unified ordering-parent closure.
- `get_descendants(name: str)` — inverse.  `HAS_ARGUMENT`-incoming
  (everything that wraps me) ∪ `HAS_ERROR`-outgoing (my
  uncertainty variants).
- `shortest_path(a: str, b: str)` — shortest directed path over all
  structural edges; returns `list[dict]` where each element is
  `{name, edge_type_in}`.  The first element has `edge_type_in:
  None` (the source); each subsequent element records the edge type
  taken to reach it from the previous hop.

All tools are guarded by the `[graph-local]` extra — registered only
if networkx is importable.  Server startup logs whether the graph
loaded, how many nodes and edges, and whether any stub nodes were
created.

Tool docstrings must explicitly document the edge-direction
conventions (both `HAS_ARGUMENT` wrapped-form → arg, and `HAS_ERROR`
base → variant) so MCP clients construct correct traversals.

#### 3. Catalog-site renderer upgrades — `imas_standard_names/rendering/catalog.py`

- Emit `links: [name:X]` as anchor hyperlinks `[X](#X)` (the audit
  found the field is already loaded but never rendered).
- Emit `cocos_transformation_type: <type>` as an inline line in the
  entry block.
- **Per-entry Mermaid hierarchy block** when any of `arguments`,
  `error_variants`, `deprecates`, `superseded_by` is present.
  Example for `x_component_of_magnetic_field`:

  ```mermaid
  graph LR
    x_component_of_magnetic_field -- "component axis=x" --> magnetic_field
  ```

  Example for `temperature` (has error_variants):

  ```mermaid
  graph LR
    temperature -- "error upper" --> upper_uncertainty_of_temperature
    temperature -- "error lower" --> lower_uncertainty_of_temperature
    temperature -- "error index" --> uncertainty_index_of_temperature
  ```

  Example for `ratio_of_pressure_to_density`:

  ```mermaid
  graph LR
    ratio_of_pressure_to_density -- "ratio role=a" --> pressure
    ratio_of_pressure_to_density -- "ratio role=b" --> density
  ```

  **Edge labels derive from edge properties**, not hardcoded
  strings.  Label template:
  - `HAS_ARGUMENT` unary: `"{operator}"`
  - `HAS_ARGUMENT` binary: `"{operator} role={role}"`
  - `HAS_ARGUMENT` projection: `"{operator} axis={axis} shape={shape}"`
  - `HAS_ERROR`: `"error {error_type}"`
  - `HAS_PREDECESSOR`: `"deprecates"`
  - `HAS_SUCCESSOR`: `"superseded by"`

- Entry sibling navigation: at the bottom of each entry page, list
  neighbours grouped by edge type — `Arguments:`, `Wrapped by:`,
  `Error variants:`, `Deprecates:`, `Superseded by:` — as clickable
  internal links.  `Wrapped by:` is computed by the renderer
  (inverse of `HAS_ARGUMENT`) from the loaded graph; it is not
  emitted inline in YAML per plan 40.

#### 4. Structured `nav:` generation — `imas_standard_names/cli/catalog_site.py`

Replace the flat alphabetic nav with a structured nav generated from
the per-domain YAML files.  For each domain, list entries in the
same order they appear in the file (plan 40's hierarchy traversal).
Vectors and their components appear adjacent in the sidebar.  Nav
entries respect pre-order DFS grouping produced by plan 40.

#### 5. Mermaid plugin

Add `mkdocs-mermaid2-plugin` to the ISN docs-extras group and to the
inline `MKDOCS_DEPLOY_TEMPLATE`.  Verify rendering on the test
catalog-site deploy.  Confirm the plugin handles the edge-label
escape cases (quoted labels with equals signs) used in §3 examples.

#### 6. Tests

- `build_catalog_graph` against a fixture catalog returns the
  expected node + edge set with full property dicts on each edge.
- `get_ancestors` on a component returns the base (outgoing
  `HAS_ARGUMENT`) and — if the component has its own wrappings —
  the full chain.
- `get_descendants` on a base returns all wrappings and uncertainty
  variants.
- `shortest_path` across mixed `HAS_ARGUMENT` + `HAS_ERROR` returns
  the correct hop sequence with edge-type labels.
- Stub-node handling: an entry with `arguments: [{name:
  nonexistent_base, …}]` produces a `stub=True` node;
  `get_neighbours` still returns it; renderer links the stub name
  without an anchor (because no entry page exists for it).
- MCP tool integration test: in-process server + tool call for each
  of the four tools.
- Renderer snapshot tests — six cases:
  1. Entry with unary-prefix `arguments` (1 edge).
  2. Entry with binary `arguments` (2 edges).
  3. Entry with projection `arguments` (1 edge, axis+shape label).
  4. Base entry with `error_variants` (up to 3 edges).
  5. Entry with `links: [name:X]` + `cocos_transformation_type`.
  6. Entry with `deprecates` + `superseded_by`.
- Structured-nav snapshot: sibling names adjacent in the nav order.

#### 7. Documentation

- ISN `AGENTS.md` — new section on the NetworkX local graph + MCP
  tools; state the `HAS_ERROR` direction convention explicitly;
  mirror the codex `AGENTS.md` "StandardName Graph Edges" table
  with ISN-side column naming (YAML field → edge type).
- `CONTRIBUTING.md` — how to use `get_ancestors` / `get_descendants`
  locally for quick graph inspection.
- `mkdocs.yml` — mermaid plugin; structured nav entry point.

### B — imas-codex repo

No codex-side changes.  Plan 41 is entirely downstream of plan 40.

## Out of scope

- **Cluster-membership exposure in local graph.**  Cluster is not
  in inline YAML per plan 40.  NetworkX graph cannot reconstruct
  it without a separate manifest file; deferred to a future plan if
  cluster navigation becomes needed catalog-side.
- **`HAS_PHYSICS_DOMAIN` / `IN_CLUSTER` edges in local graph.**
  Physics domain is a scalar property; cluster is graph-only.
- **Interactive D3.js graph widgets.**  Mermaid is enough for v1.
- **Cross-domain navigation / search UI.**  mkdocs builtin search
  is adequate.
- **Live grammar parsing** in the NetworkX builder.  Plan 41 reads
  structure from the YAML fields only — it does not re-parse names.

## Rollout

1. **NetworkX module + tests.**  Land in ISN.
2. **MCP tool wiring + tests.**
3. **Catalog-site renderer upgrades + snapshot tests.**
4. **Structured nav generator.**
5. **mkdocs plugin + docs commits.**
6. **Release ISN version bump** (`v0.8.x → v0.9.0`).
7. **Catalog-site deploy** via `mike` to the `latest` channel.

Prerequisite: plan 40 must be shipped.  NetworkX module depends on
the inline `arguments` / `error_variants` fields, which only exist
once plan 40 has migrated the catalog.

## Revision notes

**Rewrite from original draft** to align with the shipped plan-39
edge model.  Key changes:

- **Edge-type union replaced.**  Drafted `COMPONENT_OF` /
  `REAL_PART_OF` / `IMAGINARY_PART_OF` / `UNCERTAINTY_OF` replaced
  by `HAS_ARGUMENT` / `HAS_ERROR`.  Both the NetworkX graph edge
  types and the MCP tool `edge_types` parameter enum updated
  accordingly.
- **Edge-property fidelity.**  NetworkX `HAS_ARGUMENT` edges carry
  the full `{operator, operator_kind, role, separator, axis,
  shape}` property dict — not just the edge-type label.  The
  renderer derives Mermaid labels from these properties instead of
  hardcoding labels per edge-type.
- **`HAS_ERROR` direction inversion documented.**  Unlike the
  drafted `UNCERTAINTY_OF` (child → parent), the shipped
  `HAS_ERROR` points base → variant.  MCP tool docstrings and the
  renderer's "Error variants:" section must treat the base as the
  owner.
- **Stub node handling** matches codex Neo4j semantics: unknown
  forward-references become `stub=True` nodes, not errors.
- **`DiGraph` chosen for shipped-graph fidelity.**  The codex Neo4j
  writer collapses duplicate `(src, tgt, edge_type)` edges via
  `MERGE`, so the shipped structural graph is already a simple
  digraph.  `DiGraph` matches this exactly; binary operators work
  because their two edges have distinct targets.  Earlier draft
  used `MultiDiGraph`; that added NetworkX-only fidelity beyond
  what the shipped graph can represent.
- **`shortest_path` return shape pinned** to
  `list[dict]` with `edge_type_in` per hop (first hop `None`).
- **`wrapped_by` computed by renderer**, not emitted inline (plan
  40 decision).  Plan 41 builds it from forward edges at graph
  load time.

## Documentation updates

- ISN `AGENTS.md` — NetworkX module + MCP tools section; edge
  conventions table; `HAS_ERROR` direction note.
- ISN `CONTRIBUTING.md` — graph-local workflow.
- ISN `mkdocs.yml` — mermaid plugin + structured nav.
- `plans/README.md` — update P1g pointer.
