# Standard Names — Implementation Order

**Approach:** Iterative. Build the simplest working pipeline first, test it
on real data, then extend based on observed quality gaps.

**Clean break:** The previous 309 catalog entries are discarded. All standard
names will be generated fresh from the graph. The catalog repo will be rebuilt.

## Authority Model: Option B+

**Catalog YAML is authoritative** for reviewed entries. Graph is authoritative
for drafted candidates and operational metadata. See 12-catalog-import.md for
the full authority boundary design.

| Owner | Scope |
|-------|-------|
| Catalog YAML | name, description, documentation, kind, unit, tags, links, ids_paths, status, constraints, validity_domain, provenance |
| Graph only | embedding, embedded_at, model, generated_at, review_status, confidence, source, source_path |
| Derived on import | grammar fields (from name parse), CANONICAL_UNITS edge (from unit string) |

## Lifecycle

```
DRAFTED (graph, LLM-generated) → PUBLISHED (catalog PR) → ACCEPTED (merged PR, imported back)
```

All status values use past tense: drafted, published, accepted, rejected, skipped.

## Consistency Notes

- **Relationship direction:** `(entity)-[:HAS_STANDARD_NAME]->(sn:StandardName)`
  for ALL entity types (IMASNode, FacilitySignal). Single relationship name.
  The schema doc `(FacilitySignal)-[:MEASURES]->(StandardName)` is WRONG — fix in Plan 11.
- **Unit linking:** `(sn:StandardName)-[:CANONICAL_UNITS]->(u:Unit)` via canonical_units
  range declaration (relationship type: CANONICAL_UNITS per schema convention).
- **Embedding:** StandardName.embedding exists in schema with vector index
  `standard_name_desc_embedding`. Persist worker must call embed after write.
- **Coalesce bug:** `write_standard_names()` uses unconditional SET — re-runs
  erase existing data. MUST fix before any production use. Plan 11 Phase 4a.

## Implementation Status

| # | Plan | Description | Status | Depends On | Enables |
|---|------|-------------|--------|------------|---------|
| 09 | sn-generate | Core pipeline EXTRACT→COMPOSE→VALIDATE→PERSIST | ✅ Done | — | 11 |
| 11 | rich-compose | Full catalog fields, schema extension, coalesce fix, tests | 📋 Ready | 09 | 12, 13, 14 |
| 12 | catalog-import | Feedback import from reviewed catalog PRs | 📋 Ready | 11 P1 | 13 P4 |
| 13 | publish-pipeline | Lossless YAML export, batched PRs | 📋 Ready | 11 (all) | 12 (feedback loop) |
| 14 | mcp-tools-benchmark | SN search/fetch/list MCP tools + benchmark quality | 📋 Ready | 11 (embedding) | — |

## Deployment Waves

### Wave 1: Schema + Persist Fix + Tests (Plan 11 Phases 1+4)

Fix the foundation before building on it:
- Extend StandardName schema with ~12 rich fields + 2 enums (kind, review_status)
- Rename review_status enum: candidate → drafted (past tense)
- Fix schema doc: FacilitySignal uses HAS_STANDARD_NAME (not MEASURES)
- Fix signals.py MEASURES query → HAS_STANDARD_NAME
- Fix unconditional SET overwrite bug (use coalesce)
- Wire CANONICAL_UNITS relationship creation
- Wire embedding generation into persist worker
- Add graph_ops unit tests (currently 0 tests)
- Add conftest.py with shared fixtures

**Agent:** architect (cross-module: schema + graph_ops + workers + tests)

### Wave 2: LLM Compose Upgrade (Plan 11 Phases 2+3+5) + Catalog Import (Plan 12)

Run in parallel — compose upgrade and catalog import are independent after schema:
- **Agent A (architect):** Extend SNCandidate model, update prompts for rich docs, update validate worker
- **Agent B (engineer):** Build `sn import-catalog` CLI + catalog_import.py + version tracking

### Wave 3: Publish (Plan 13) + MCP Tools + Benchmark (Plan 14)

Run in parallel — export and tools are independent:
- **Agent A (engineer):** Fix lossy publish export, update graph query, PR workflow, dedup
- **Agent B (engineer):** 3 MCP tools (search/fetch/list) + benchmark quality tiers + reviewer

### Wave 4: Integration Testing

- End-to-end: build → publish → review → import
- Round-trip idempotence verification
- Embedding coverage for all StandardName nodes
- Documentation updates (AGENTS.md, README)

**Agent:** architect (integration testing + documentation)

## Archived Plans

Previous plans 09 (schema providers), 10 (pipeline fixes) are superseded.
Research material in `plans/research/standard-names/`.
Archived v1 plans in `plans/research/standard-names/archived-v1/`.
