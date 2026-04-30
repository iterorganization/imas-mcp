# Plan 44 — SN graph renames, multi-valued domains, batch embedding, sn clear

## Problem

Four cross-cutting issues surfaced from the post-Plan-43 validation
`sn run -c 10`:

1. **Sequential single-text embedding** is the dominant runtime cost.
   `_hybrid_search_neighbours` is called per source path during compose,
   issuing 2 single-text remote embed round-trips per item. For a 25-path
   batch that is 50 round-trips — ~3 s of wall time before the LLM call,
   sequential. The discover and DD-build pipelines already solved this
   with `embed_descriptions_batch` (one round-trip for N texts) — SN must
   adopt the same pattern.

2. **`physics_domain` is overwritten** when the same `StandardName` id
   surfaces in a second domain turn (e.g. `plasma_current` produced first
   in `equilibrium`, then re-touched in `transport` — the SN ends up
   tagged `physics_domain=transport`, losing equilibrium attribution).
   The user wants the same multi-source topology that `IMASNode` and
   `StandardNameSource` already use: a name can belong to **many**
   physics domains.

3. **`Review` is a poor class name and `REVIEWS` is a poor edge name.**
   They collide with semantic intent in queries (`MATCH (r:Review)` reads
   like a noun, but `Review` as a node label conflicts with the verb
   `r.score`/`r.tier`). Rename to `StandardNameReview` (matches the
   `StandardName*` family — `StandardNameSource`, `StandardNameReview`)
   and `HAS_REVIEW` (matches the `HAS_*` ownership pattern used
   throughout: `HAS_UNIT`, `HAS_COCOS`, `HAS_PHYSICS_DOMAIN`).

4. **`sn clear` does not delete `LLMCost`.** A clean reset for re-running
   the pipeline must wipe the cost ledger — otherwise stale rows
   accumulate.

## Approach

Six work streams, dependency-ordered. Streams A/D run first because
they touch only schema + CLI. Stream B (multi-valued domain) gates the
write path. Stream C (batch embed) is pure refactor inside compose.
Stream E migrates existing graph data. Stream F validates end-to-end.

| Stream | Topic | Owner | Depends on |
|---|---|---|---|
| A | Rename `Review` → `StandardNameReview`, `REVIEWS` → `HAS_REVIEW` | engineer | — |
| B | Multi-valued `physics_domain` on StandardName (list, append-only) | engineer | A (model rebuild) |
| C | Batch hybrid-search embedding (kill N+1 in `_hybrid_search_neighbours`) | engineer | — |
| D | `sn clear` includes `LLMCost`; rebuild gap doc | engineer | — |
| E | Graph migration: rename labels/edges, convert domain scalar → list, drop hanging LLMCost | engineer | A, B, D |
| F | Tests + validation `sn run -c 10` (cleared graph), assert correctness, fix bugs | engineer | E |

## Stream A — Rename Review → StandardNameReview, REVIEWS → HAS_REVIEW

### Schema changes (`imas_codex/schemas/standard_name.yaml`)

```yaml
StandardNameReview:                 # was: Review
  description: >-
    ... A StandardName may have one (canonical) or many (multi-reviewer)
    StandardNameReview nodes attached via HAS_REVIEW.
  class_uri: sn:StandardNameReview
  attributes:
    standard_name_id:
      annotations:
        relationship_type: HAS_REVIEW   # was: REVIEWS, direction reversed
    ...
```

**Edge direction note.** Today: `(Review)-[:REVIEWS]->(StandardName)`.
After: `(StandardName)-[:HAS_REVIEW]->(StandardNameReview)`. The
`HAS_*` ownership convention (matches `HAS_UNIT`, `HAS_COCOS`,
`HAS_PHYSICS_DOMAIN`, `HAS_ERROR`) requires the parent to be the
StandardName. Migration must reverse direction.

### Code-side rename surface

`grep -rn "REVIEWS\|class Review\|:Review\b" imas_codex/ tests/` returns
~15 files. Touch points (verified):

- `imas_codex/standard_names/graph_ops.py` (write_reviews, queries)
- `imas_codex/standard_names/review/pipeline.py` (consolidator,
  `(sn:StandardName)<-[:HAS_STANDARD_NAME]-(node:IMASNode)` is
  unrelated; the `:REVIEWS` reads need updating)
- `imas_codex/standard_names/review/audits.py`
- `imas_codex/standard_names/review/persist.py`
- `imas_codex/cli/sn.py` (sn clear queries)
- All `tests/standard_names/test_review_*.py` (mock fixtures, queries)

A find-replace is appropriate: every `:Review)` becomes
`:StandardNameReview)`, every `[:REVIEWS]` becomes `[:HAS_REVIEW]`,
every `<-[:REVIEWS]-` becomes `-[:HAS_REVIEW]->` (direction reversal).

### Tests

- `tests/graph/test_schema_compliance.py` already validates labels +
  rel types from the LinkML schema — passing this is the gate.
- Add `tests/standard_names/test_review_label_migration.py` with a
  fake-graph fixture that asserts the migration query renames labels
  and reverses the edge correctly.

## Stream B — Multi-valued `physics_domain` on StandardName

### Schema change

```yaml
StandardName:
  attributes:
    ...
    physics_domain:
      description: >-
        Physics domain classifications. A StandardName may belong to multiple
        domains when its source DD paths span domains (e.g.
        `plasma_current` appears in equilibrium, transport, magnetics).
        Append-only at write time — a domain seen once is recorded
        forever, never overwritten.
      multivalued: true
      range: PhysicsDomain                    # references the enum
      annotations:
        relationship_type: HAS_PHYSICS_DOMAIN
    # DROP `physics_domain_ref` (was singular ref). The multivalued
    # `physics_domain` slot now does the dual-property + relationship
    # job for both directions.
```

The graph already has `(:StandardName)-[:HAS_PHYSICS_DOMAIN]->(:PhysicsDomain)`
edges — they just need to become a true 1-to-many.

### Write semantics — append + write-time dedupe (no APOC)

**RD finding (blocking):** APOC is **not** installed on this Neo4j —
`apoc.version()` is unknown. Use pure Cypher with write-time set
semantics (no read-time dedupe — keeps invariants on the node).

Today (`graph_ops.write_standard_names`):

```cypher
MERGE (sn:StandardName {id: b.id})
SET sn.physics_domain = coalesce(b.physics_domain, sn.physics_domain)
```

After (pure Cypher set semantics, dedupe at write time):

```cypher
MERGE (sn:StandardName {id: b.id})
WITH sn, b, coalesce(sn.physics_domain, []) AS existing,
     coalesce(b.physics_domain, []) AS incoming
WITH sn, b, existing,
     [d IN incoming WHERE d IS NOT NULL AND NOT (d IN existing) | d] AS new_domains
SET sn.physics_domain = existing + new_domains
WITH sn, b, new_domains
UNWIND new_domains AS domain
MATCH (pd:PhysicsDomain {id: domain})
MERGE (sn)-[:HAS_PHYSICS_DOMAIN]->(pd)
```

`b.physics_domain` is always sent as a list from the writer
(scalar→list conversion happens in the python builder, not Cypher).
Existing-domain check on the python side is unnecessary — Cypher
filter handles it.

### Read-side updates — full audit

**RD finding (medium):** loop/status/display readers expect a scalar.
Audit list (verified call sites):

| File:line | Today | After |
|---|---|---|
| `loop.py:93-100` | `RETURN sn.physics_domain AS domain` | `UNWIND sn.physics_domain AS domain RETURN domain` (rotation visits each domain once per name) |
| `context.py:236-252` | `WHERE sn.physics_domain = $domain` | `WHERE $domain IN sn.physics_domain` |
| `cli/sn.py:968-970` | `WHERE sn.physics_domain = $domain` | `WHERE $domain IN sn.physics_domain` |
| `graph_ops.py:335-336` | filter equality | `IN` predicate |
| `export.py:147-149` | filter equality | `IN` predicate |

`grep -rn "\.physics_domain" imas_codex/ tests/` is the gate — every
hit must be inspected. Filtering predicates always become
`$domain IN sn.physics_domain`. Aggregations always become
`UNWIND sn.physics_domain AS d`.

### Catalog YAML round-trip — design (RD blocking)

**Problem:** `export.py:949-975` groups by `entry_domain =
cand.get("physics_domain")` and writes one file per (domain, name).
A list breaks this; a roundtrip via `catalog_import.py:115-129`
collapses to scalar.

**Decision:** Keep the catalog file path single-domain (one YAML per
(primary_domain, name)) but preserve the **full list** in the YAML
body. Specifically:

1. **Export.** When SN has `physics_domain=[a, b, c]`, write the file
   under `standard_names/<primary>/<name>.yml` where `primary` is
   chosen by **priority-first sort** (see §Domain priority below)
   with alphabetical tie-break for determinism. Body includes
   `physics_domain: [a, b, c]` as a YAML list, sorted alphabetically
   (independent of primary choice — the body is a set, the file
   path is just one of them).
2. **Import.** Parse `physics_domain` as a list. No legacy
   bare-string handling — the catalog will be cleared and rebuilt
   from fresh, so import requires the list form. Reject (validation
   error) any YAML where `physics_domain` is not a list.
3. **Round-trip invariant.** The catalog write then re-import must
   preserve the full list, not collapse to the directory-derived
   primary.

The "primary" choice (priority-first) is a display/file-path
convention — never authoritative.

### Domain priority — derived from cluster mapping relevance

The existing notion of importance lives at the **cluster** level via
`Cluster.mapping_relevance` (`high` / `medium` / `low`, see
`imas_codex/definitions/clusters/mapping_relevance.yaml`). HIGH
clusters are core physics quantities (Te, Ti, ne, Ip, Bt, q-profile,
boundary shape, plasma current, heating power) — exactly the
domains we want to surface as "primary" when a name spans several.

A `Cluster` node also carries `physics_domain` (primary physics
domain of its members). Domain importance therefore lifts naturally
from cluster importance:

```cypher
MATCH (c:Cluster)
WHERE c.physics_domain IS NOT NULL
  AND c.mapping_relevance IS NOT NULL
WITH c.physics_domain AS domain,
     CASE c.mapping_relevance
       WHEN 'high' THEN 100
       WHEN 'medium' THEN 10
       ELSE 1
     END AS weight
RETURN domain, sum(weight) AS importance
ORDER BY importance DESC
```

A domain whose clusters are mostly HIGH-relevance scores high; one
whose clusters are mostly LOW-relevance scores low. This is
DD-derived (no hardcoded list), survives DD version changes, and
extends naturally if `MappingRelevance` gets new tiers.

**Implementation.** Add a module-level cached helper:

```python
# imas_codex/standard_names/domain_priority.py (new)
@functools.lru_cache(maxsize=1)
def get_domain_priority_index() -> dict[str, int]:
    """Return rank (0-based, lower = more important) per physics
    domain. Computed once per process from Cluster.mapping_relevance
    weighted counts. Domains absent from the index get rank 999."""
    ...

def pick_primary_domain(domains: list[str]) -> str:
    """Pick the highest-priority domain; alphabetical tie-break."""
    ranks = get_domain_priority_index()
    return sorted(domains, key=lambda d: (ranks.get(d, 999), d))[0]
```

Use in `export.py:953-954`:

```python
from imas_codex.standard_names.domain_priority import pick_primary_domain
primary = pick_primary_domain(physics_domain_list) if physics_domain_list else "unscoped"
```

**Tests** (`tests/standard_names/test_domain_priority.py`):
- Mock graph with 3 high+5 medium clusters in `equilibrium`, 0
  high+1 medium in `transport` → equilibrium ranks above transport
- Empty cluster set → fall back to alphabetical
- Tie scores → alphabetical tie-break preserved
- Cache: subsequent calls don't re-query the graph

### Tests

- `tests/standard_names/test_physics_domain_multi.py`:
  - Write SN with domain=[equilibrium], assert list stored
  - Re-write same id with domain=[transport], assert list grew to
    [equilibrium, transport] (order-insensitive)
  - Assert HAS_PHYSICS_DOMAIN edges = 2

## Stream C — Batch hybrid-search embedding

### Root cause

`imas_codex/standard_names/workers.py:_hybrid_search_neighbours` calls
`hybrid_dd_search(gc, text, ...)` twice per source path (description
text + path text). Each call hits `_embed(text)` →
`encoder.embed_texts([text], prompt_name="query")` — a 1-text remote
round-trip with the **query** prompt flavour.

For a batch of N items: 2N round-trips × ~50 ms = N × 100 ms
sequential overhead before any LLM call.

**RD finding (blocking):** `embed_descriptions_batch` does **not**
pass `prompt_name="query"` — it embeds in document mode. Reusing it
unchanged would silently change retrieval semantics. We must add a
query-flavoured batch helper.

### Fix

**1. Add query-batch helper.** New function in
`imas_codex/embeddings/description.py` (or new module
`imas_codex/embeddings/query.py`):

```python
def embed_query_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed N texts with prompt_name='query'. One round-trip."""
    encoder = _get_encoder()
    if not texts:
        return []
    return encoder.embed_texts(texts, prompt_name="query")
```

This preserves the `prompt_name="query"` semantics that
`hybrid_dd_search._embed` uses today.

**2. Add `embedding` parameter to `hybrid_dd_search`.** When provided,
skip the internal `_embed()` call and use the caller's pre-computed
embedding. When not provided, fall back to existing single-text embed
(for MCP tool callers).

`imas_codex/graph/dd_search.py`:
```python
def hybrid_dd_search(
    gc, text, *,
    embedding: list[float] | None = None,
    ...
) -> list[SearchHit]:
    if embedding is None:
        embedding = _embed(text)
    # rest unchanged
```

**3. Pre-batch query texts in `_hybrid_search_neighbours`.** Push the
batch boundary down to the helper itself so all callers benefit
without duplicating cache logic. The helper accepts a list of
`(path, description)` tuples and returns a list-of-results in the
same order. Internally it builds the unique text set, calls
`embed_query_texts()` once, and threads the cache through
`hybrid_dd_search(..., embedding=...)`.

```python
# imas_codex/standard_names/workers.py
def _hybrid_search_neighbours_batch(
    gc, items: list[tuple[str, str | None]], *, k: int = 5,
) -> list[HybridSearchResult]:
    # 1. Collect unique query texts
    texts: set[str] = set()
    for path, desc in items:
        if desc:
            texts.add(desc.strip()[:200])
        texts.add(path)
    texts.discard("")
    # 2. One remote round-trip
    text_list = sorted(texts)
    embeddings = embed_query_texts(text_list)
    cache = dict(zip(text_list, embeddings, strict=True))
    # 3. Per-item enrichment uses cache
    return [
        _hybrid_search_neighbours_one(gc, path, desc, cache, k=k)
        for path, desc in items
    ]
```

**4. Cover ALL callers (RD blocking).** The N+1 helper is also
called from:
- `imas_codex/standard_names/workers.py:1642` (compose retry)
- `imas_codex/standard_names/enrich_workers.py:676` (link candidates)
- `imas_codex/standard_names/review/pipeline.py:678` (review context)

Refactor each to call `_hybrid_search_neighbours_batch` once with
all items in scope rather than per-item in a loop.

### Performance target

For a 25-item batch:
- **Before**: 50 round-trips × ~50 ms = ~2.5 s embedding overhead
- **After**: 1 round-trip × ~80 ms (50 texts) = ~80 ms overhead
- **Speedup**: ~30× on embedding stage

### Tests

- `tests/standard_names/test_compose_batch_embed.py`:
  - Mock encoder with call counter
  - Run a 5-item batch through compose-prep
  - Assert `encoder.embed_texts` called once (not 10×)
  - Assert `prompt_name="query"` was passed
- `tests/graph/test_dd_search.py`:
  - Add test for `hybrid_dd_search(text, embedding=...)` skipping
    internal embed call

## Stream D — `sn clear` includes LLMCost

### Code change

`imas_codex/cli/sn.py` `_run_sn_clear_cmd` — add `LLMCost` to the
list of labels deleted. Verify via:

```cypher
MATCH (n:LLMCost) RETURN count(n) AS leftover
```

after `sn clear --force`. Should be 0.

### Tests

- `tests/standard_names/test_sn_clear.py`:
  - Pre-populate LLMCost + StandardName + StandardNameReview
  - Run `sn clear --force`
  - Assert all four labels gone

## Stream E — Graph migration

Run as a one-off Cypher script (per AGENTS.md: never persist as
`scripts/migrate_*.py` — execute inline via `graph shell` or python).
The migration is destructive of `Review` label and `REVIEWS` edges
but is safe because we will re-run `sn clear --force` immediately
afterwards as part of validation.

**RD finding (blocking):** Run in a single explicit write transaction
and use `valueType()` (Cypher 5) for type-safe idempotency check —
not string heuristics like `CONTAINS '['` (which fails after first
run when the value is already a list).

```python
# Inline — single transaction, idempotent, no APOC.
gc.query("""
    // 1. Rename Review → StandardNameReview (idempotent: REMOVE on
    //    a label that's not present is a no-op)
    MATCH (r:Review)
    REMOVE r:Review
    SET r:StandardNameReview
    WITH count(*) AS _renamed

    // 2. Reverse + rename REVIEWS → HAS_REVIEW
    MATCH (r:StandardNameReview)-[old:REVIEWS]->(sn:StandardName)
    MERGE (sn)-[:HAS_REVIEW]->(r)
    DELETE old
    WITH count(*) AS _redirected

    // 3. Convert physics_domain string → list (idempotent via valueType)
    MATCH (sn:StandardName)
    WHERE sn.physics_domain IS NOT NULL
      AND valueType(sn.physics_domain) = 'STRING NOT NULL'
    SET sn.physics_domain = [sn.physics_domain]
    WITH count(*) AS _converted

    // 4. Drop hanging LLMCost rows
    MATCH (n:LLMCost) DETACH DELETE n
    RETURN count(*) AS _llmcost_deleted
""")
```

We can skip migration in this case (Stream F clears the graph
anyway), but the script is documented for users with existing data.

## Stream F — Validation

1. `sn clear --force` then verify:
   ```cypher
   MATCH (n) WHERE labels(n)[0] STARTS WITH 'StandardName' OR labels(n)[0] = 'LLMCost' OR labels(n)[0] = 'SNRun'
   RETURN labels(n)[0] AS label, count(*) AS n
   ```
   All zero.

2. `imas-codex sn run -c 10` (no domain filter, no extra flags)

3. Pass criteria:
   - SNRun.status='completed', cost_is_exact=true
   - LLMCost row count matches LLM call count from logs
   - At least one StandardName has `physics_domain` containing 2+
     entries (cross-domain attribution preserved)
   - All `:StandardNameReview` nodes (zero `:Review` nodes)
   - All `[:HAS_REVIEW]` edges (zero `[:REVIEWS]` edges)
   - Embedding round-trip count per batch ≤ 2 (one for query texts,
     one for produced names) — verify via embed server logs or
     `Remote embedding: N texts` log lines

4. Iterate fixes until passing.

## Build & test order (RD medium finding)

After **any** LinkML schema edit, the pipeline is:

1. Edit `imas_codex/schemas/standard_name.yaml`
2. `uv run build-models --force` — regenerates `models.py`,
   `schema_context_data.py`, `schema-reference.md`,
   `EXPECTED_RELATIONSHIP_TYPES`. Generated files are gitignored.
3. `uv run pytest tests/graph/test_schema_compliance.py` — schema
   gate. This must pass before any other test or Python import that
   depends on the new model.
4. Code edits, then `uv run pytest tests/standard_names/`.
5. Commit.

Stream A and Stream B both edit the schema — they MUST share one
`build-models --force` run (sequential), or merge their schema edits
before rebuilding. Dispatch order: A first (rename), B second
(multi-domain) — B builds on A's regenerated models.

## Stream D — `sn clear` includes LLMCost (extended scope)

**RD finding (nit):** `sn clear` is not the only stale-cost path.
`sn run --reset-to extracted` goes through `clear_standard_names()`
(`graph_ops.py:2672-2865`) which also leaves `LLMCost` orphaned.
Both code paths must be fixed.

### Code change

1. `imas_codex/cli/sn.py` `_run_sn_clear_cmd` — add `LLMCost` to the
   label list.
2. `imas_codex/standard_names/graph_ops.py` `clear_standard_names`
   — same label addition.

### Test

- `tests/standard_names/test_sn_clear.py`:
  - Pre-populate LLMCost + StandardName + StandardNameReview
  - Run `sn clear --force` → assert all gone
  - Pre-populate again, run `clear_standard_names()` (via reset
    code-path) → assert LLMCost also gone

## Out of scope

- Other pre-existing bugs from the validate-10usd-run inspection
  (reviewer_tier null, vocab gap classifier issues) — file as
  separate todos.
- Renaming `VocabGap` → `StandardNameVocabGap` (consistent but not
  requested; defer).

## Documentation updates

- `AGENTS.md` table mentioning Review under "key relationships" —
  update to `HAS_REVIEW` + StandardNameReview.
- `agents/schema-reference.md` — auto-regenerated by build hook.
- Delete this plan when fully implemented.
