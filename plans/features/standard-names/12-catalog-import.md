# 12: Catalog Feedback Import

**Status:** Ready to implement
**Depends on:** 11 (rich schema — Phase 1 only)
**Enables:** 13 (publish dedup), 14 (SN MCP tools — richer context)
**Agent:** engineer (well-defined: CLI command + graph_ops function)

## Problem

After `sn publish` exports StandardName entries to YAML and a human reviews them
via a catalog PR, those edits need to flow back to the graph. Without this
feedback loop, the graph diverges from the reviewed catalog.

**Clean break:** We are NOT importing the previous 309 catalog entries. The
`imas-standard-names-catalog` repo will be rebuilt from graph-generated data.
This plan implements the feedback import for the publish → review → import cycle.

## Design: Option B+ Authority Model

The catalog YAML is authoritative for **reviewed entries**. The graph is authoritative
for **drafted candidates** and operational metadata (embeddings, generation batches).

**Authority boundaries:**

| Owner | Fields |
|-------|--------|
| Catalog (YAML) | name, description, documentation, kind, unit, tags, links, ids_paths, status, constraints, validity_domain, provenance |
| Graph only | embedding, embedded_at, model, generated_at, review_status, confidence, source, source_path |
| Derived on import | grammar fields (from name parse), HAS_UNIT edge (from unit string) |

**Import rule:** Whole-entry import. Catalog fields always win. Graph-only fields
are preserved via coalesce. Imported entries get `review_status: accepted`.

## Phase 1: `sn import-catalog` CLI command

**Files:**
- `imas_codex/cli/sn.py` — add `import-catalog` subcommand
- `imas_codex/standard_names/catalog_import.py` — new module

### CLI interface

```bash
# Import reviewed entries from catalog checkout
imas-codex sn import-catalog --catalog-dir ../imas-standard-names-catalog/standard_names

# Dry run — show what would be imported
imas-codex sn import-catalog --catalog-dir <path> --dry-run

# Import only specific tags
imas-codex sn import-catalog --catalog-dir <path> --tags equilibrium,core-physics
```

### Import logic (`catalog_import.py`)

```python
def import_catalog(
    catalog_dir: Path,
    dry_run: bool = False,
    tag_filter: list[str] | None = None,
) -> ImportResult:
    """Import YAML catalog entries into graph as accepted StandardName nodes.

    Reads all *.yml files from catalog_dir (recursive), parses each entry,
    derives grammar fields via name parsing, and MERGEs into the graph.

    Imported entries get review_status='accepted'. Catalog fields overwrite
    graph fields (catalog is authoritative). Graph-only fields preserved.
    """
```

Steps:
1. Walk `catalog_dir` recursively for `*.yml` files
2. Parse each YAML → validate against Pydantic import model
3. Derive grammar fields by parsing the standard name
4. MERGE StandardName nodes — catalog fields overwrite, graph-only fields preserved
5. Set `review_status = 'accepted'`, `imported_at = datetime()`
6. Derive `HAS_UNIT` relationships from `unit` field
7. Derive `HAS_STANDARD_NAME` relationships from `ids_paths` field
   (link to existing IMASNode nodes in the DD graph)
8. Report: imported, updated, skipped, errors

### Graph write — coalesce for graph-only fields

```cypher
UNWIND $items AS item
MERGE (sn:StandardName {id: item.name})
SET sn.description = item.description,
    sn.documentation = item.documentation,
    sn.kind = item.kind,
    sn.unit = item.unit,
    sn.tags = item.tags,
    sn.links = item.links,
    sn.ids_paths = item.ids_paths,
    sn.constraints = item.constraints,
    sn.validity_domain = item.validity_domain,
    sn.review_status = 'accepted',
    sn.imported_at = datetime(),
    sn.physical_base = item.physical_base,
    sn.subject = item.subject,
    sn.component = item.component,
    sn.coordinate = item.coordinate,
    sn.position = item.position,
    sn.process = item.process,
    sn.created_at = coalesce(sn.created_at, datetime()),
    sn.embedding = coalesce(sn.embedding, null),
    sn.embedded_at = coalesce(sn.embedded_at, null)

WITH sn, item
WHERE item.unit IS NOT NULL
MERGE (u:Unit {id: item.unit})
MERGE (sn)-[:HAS_UNIT]->(u)
```

Note: Uses `HAS_UNIT` not `HAS_UNIT` per schema convention (range: Unit
on `unit` slot auto-generates the relationship type).

**Acceptance:**
- `sn import-catalog --catalog-dir <path> --dry-run` reports entries found
- After import, nodes have `review_status: accepted`
- HAS_UNIT relationships exist for entries with units
- Grammar fields derived from name parsing
- Graph-only fields (embedding, model, generated_at) preserved

## Phase 2: Version tracking

**Files:** `imas_codex/standard_names/catalog_import.py`

Add version tracking to imported entries:
- `catalog_commit_sha` — git rev-parse HEAD of catalog repo at import time
- `imported_at` — timestamp

Add `sn import-catalog --check` mode that reports whether graph entries
match current catalog without importing.

**Acceptance:**
- Imported nodes have `catalog_commit_sha` property
- `sn import-catalog --check` reports sync status

## Test Plan

- Unit test: YAML parsing and grammar field derivation
- Unit test: import idempotency (re-import same catalog → no changes)
- Unit test: import preserves graph-only fields (embedding, model)
- Unit test: catalog fields overwrite graph fields (authority model)
- Integration test: publish → import round-trip preserves all data
