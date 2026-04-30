# Plan 40 — Grammar-aware SN Search & Fetch Facility

> **Status**: DRAFT v1 — plan only; no code in this task.
> **Branch**: `main`
> **Owner area**: Standard Names — read & write paths around grammar decomposition.
> **Parent / context**: AGENTS.md "Standard Names" → "Schema", "MCP Tools", "Write Semantics".
> **Related plans**:
> - **Plan 39** (structured fan-out) — *underwritten by this plan*. Plan 39's
>   `search_existing_names` runner calls into the helpers redesigned here;
>   plan 39 owns the dispatcher / proposer / synthesizer pattern. See §15 for
>   the boundary.
> - Plan 37 (grammar identity prefix) — established the ISN parser as the
>   sole source of grammar truth; this plan unblocks its graph persistence.
> - Plan 32 / Plan 31 / Plan 26 — compose, retry chain, review pipeline (consumers).

---

## 1. Executive summary

Three mechanisms currently store decomposed grammar segments on
`StandardName` nodes — the legacy `grammar_fields` JSON blob, typed
edges to `GrammarToken` (`HAS_PHYSICAL_BASE` etc.), and the per-segment
column properties declared in the LinkML schema (`grammar_physical_base`
etc.).  All three are **partially populated**, **none is canonical**,
and the most discriminating segment for cross-name grouping
(`physical_base`, an *open-vocabulary* slot by ISN design) is silently
dropped on every write because the writer requires a matching
`GrammarToken` node which does not exist for open-vocab values.

This plan does two things:

1. **Write-side fix** — pick **per-segment column properties** as the
   canonical store, populate them from the deterministic ISN parser on
   every write path (`compose_worker`, `catalog_import`, `refine`),
   and keep typed edges only as a *secondary* index for closed-vocab
   segments.  Drop the `grammar_fields` JSON blob from new writes.
2. **Read-side redesign** — rewrite the `_search_standard_names` MCP
   tool as a true hybrid: the query is parsed by the ISN parser into
   per-segment tokens, and three streams (vector, grammar-segment,
   keyword) are retrieved in parallel and **fused with Reciprocal
   Rank Fusion (RRF)**.  Add a focused `siblings_by_segment` mode for
   the user's "all names sharing the same base name" requirement and
   a group-by-base mode for catalog audits.

**Non-backward-compatible policy (per AGENTS.md "Reset and Clear
Semantics"):** the graph is small (22 SNs at HEAD `5e4d3bbe`) and is
expected to be wiped via `sn clear` before this lands.  No backfill
migration; no compatibility shim for the legacy blob.

| Property                   | Today                                                  | After plan 40                                              |
|----------------------------|--------------------------------------------------------|------------------------------------------------------------|
| Canonical grammar storage  | None — three partial mechanisms                        | Per-segment columns on `StandardName`                      |
| Open-vocab `physical_base` | Silently dropped (no `GrammarToken` exists)            | Stored as raw string on `sn.grammar_physical_base`         |
| Search modes               | Mutually exclusive (vector XOR keyword XOR segment)    | Three parallel streams fused via RRF                       |
| Sibling lookup             | Not exposed                                            | `siblings_by_segment(gc, value, segment)` + MCP tool       |
| Provenance in hits         | Single opaque score                                    | Per-stream rank vector + fused score                       |
| `grammar_fields` JSON blob | Written on 10/22 SNs, read by no production code path  | Removed from new writes; deleted in follow-up cleanup      |

---

## 2. Goals & non-goals

### Goals

- **G1.** One canonical write target for decomposed grammar segments,
  populated by every writer that creates or refines an SN.
- **G2.** Open-vocab `physical_base` round-trips: parse → column →
  search → fetch → reviewer prompt context.
- **G3.** Hybrid search that surfaces siblings sharing a `physical_base`
  even when only one of them has a vector embedding hit.
- **G4.** Deterministic, low-cost, no LLM call on the search hot path.
- **G5.** Sibling and group-by-base lookups exposed as first-class
  helpers (Python API + MCP tool).
- **G6.** Idempotent writers — re-running on the same SN produces an
  identical column state, even if the parser output narrows over time.

### Non-goals

- ❌ Backfilling existing SNs in deployed graphs (graph is reset before
  this lands; see §12).
- ❌ Schema changes to `GrammarToken` / `GrammarSegment` /
  `ISNGrammarVersion`.  The only schema delta is the slot canonicalisation
  documented in §4.1.
- ❌ A new vector index.  We reuse `standard_name_desc_embedding`.
- ❌ Replacing the ISN parser or duplicating its rules.  All decomposition
  goes through `imas_standard_names.grammar.parse_standard_name`.
- ❌ Free agentic tool-calling, multi-model debate, or any change to the
  fan-out dispatcher (owned by plan 39).
- ❌ Search-result re-ranking with LLM judges.  The fused score is
  deterministic.
- ❌ Cross-version GrammarToken reconciliation (e.g. `0.7.0rc35` vs
  earlier).  Existing `_resolve_grammar_token_version` policy is reused
  unchanged.

---

## 3. Empirical findings (verbatim from orchestrator audit at `5e4d3bbe`)

> Credit: orchestrator's empirical pass against the live `imas-codex`
> MCP server (development graph) at HEAD `5e4d3bbe`. Re-verified by
> the author against the same graph immediately before drafting.

### 3.1 Three parallel grammar storage mechanisms

| Mechanism | Where | 22-SN coverage | Used by |
|---|---|---|---|
| `grammar_fields` JSON blob (legacy keys: `phenomenon`, `attribute`, `process`, `feature`, `source`, `quantity`, `component`) | property on SN | **10 / 22** | Nothing in production read paths — orphaned |
| `HAS_SEGMENT` edge → `GrammarToken` (with `position`, `segment` rel-props) | edge | **7 / 22** SNs, 11 edges | Fallback inside `_segment_filter_search_sn` |
| Typed edges (`HAS_PHYSICAL_BASE`, `HAS_SUBJECT`, `HAS_TRANSFORMATION`, `HAS_COMPONENT`, `HAS_COORDINATE`, `HAS_PROCESS`, `HAS_POSITION`, `HAS_REGION`, `HAS_DEVICE`, `HAS_GEOMETRIC_BASE`) → `GrammarToken` | edge | 6 typed edges total across 22 SNs (0 `HAS_PHYSICAL_BASE`) | `imas_codex/llm/sn_tools.py::_segment_filter_search_sn` (line 137) |
| Per-segment column properties (`grammar_physical_base`, `grammar_subject`, `grammar_component`, `grammar_geometric_base`, …) | property on SN | **0 / 22** | Schema declares them at `imas_codex/schemas/standard_name.yaml:803+`; writer at `catalog_import.py:428` only writes them on the import path; live pipeline never populates them |

Live re-verification (today, against the development graph):

```
sn_count                = 22
sns_with_HAS_SEGMENT    = 7  (11 edges total)
sns_with_grammar_col    = 0
sns_with_grammar_blob   = 10
typed_edge_breakdown    = HAS_COORDINATE:3, HAS_GEOMETRIC_BASE:2, HAS_COMPONENT:1
```

### 3.2 Root cause of the data loss

`imas_codex/standard_names/graph_ops.py::_write_segment_edges` (lines
1976-2110) does:

```cypher
OPTIONAL MATCH (t:GrammarToken {value: edge.token, segment: edge.segment, version: $token_version})
WITH sn, edge, t
FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
    MERGE (sn)-[r:HAS_SEGMENT]->(t) ...)
```

When the GrammarToken does not exist for that `(value, segment, version)`
triple, the edge is silently dropped.  Per AGENTS.md ("Open vs closed
grammar segments"): **only `physical_base` is open-vocabulary by design**
— the ISN v0.7 `SEGMENT_TOKEN_MAP` exposes an empty token tuple for it.
So the parser yields `physical_base = "major_radius"` /
`"temperature"` / `"plasma_current"` / etc. as raw strings, but those
strings are never materialised as `GrammarToken` nodes.  Result: the
most discriminating segment for cross-name grouping is **unrecoverable
from the graph for any open-vocab value**.

### 3.3 Verified parser output (sample)

`imas_standard_names.grammar.parse_standard_name` decomposes:

| Name | Parser output |
|---|---|
| `major_radius_of_strike_point` | `physical_base='major_radius'`, `geometry='strike_point'` |
| `vertical_coordinate_of_geometric_axis` | `coordinate='vertical'`, `geometric_base='coordinate'`, `geometry='geometric_axis'` |
| `electron_temperature` | `subject='electron'`, `physical_base='temperature'` |
| `plasma_current` | `physical_base='plasma_current'` |
| `plasma_geometric_axis_vertical_centroid_position` | `physical_base='plasma_geometric_axis_vertical_centroid_position'` (whole string — parse-fallback) |

The parser is reliable; the writer drops `physical_base` because it is
open-vocab.

### 3.4 Existing search code paths

- `imas_codex/standard_names/search.py::search_similar_names(query, k)` —
  vector-only on `standard_name_desc_embedding`. No grammar awareness.
  Used by compose worker for collision avoidance.
- `imas_codex/standard_names/search.py::search_similar_sns_with_full_docs(...)`
  — vector-only, returns full docs as exemplars. No grammar awareness.
- `imas_codex/llm/sn_tools.py::_search_standard_names(...)` — MCP tool
  entry point. Three branches:
  - `_segment_filter_search_sn` if any grammar-segment kwarg is supplied
    (uses typed edges; requires them to exist).
  - `_vector_search_sn` if embedding succeeds.
  - `_keyword_search_sn` (substring `CONTAINS` on `id` / `description`
    / `documentation`) as fallback.
  - **Branches are mutually exclusive — there is no fusion.**  The
    docstring's "hybrid (vector + keyword)" claim is a misnomer.

### 3.5 The MCP servers

- **imas-codex** MCP server binds to the local development graph (where
  SN content actually lives).  `imas_codex/llm/server.py` lines
  3087, 3172, 3190 register `search_standard_names`, `fetch_standard_names`,
  `list_standard_names` against `_search_standard_names`,
  `_fetch_standard_names`, `_list_standard_names` from `sn_tools.py`.
  Registration is gated by `if not self.dd_only:` (line 3055)
  AND `if self.include_standard_names:` (line 3084).
- **imas-dd** MCP server points at a frozen DD-only remote graph that
  predates this work — it has no SN data and is irrelevant here.

### 3.6 GrammarToken vocabulary in graph

- 9 850 GrammarToken nodes
- 198 GrammarSegment nodes
- 108 GrammarTemplate nodes
- 18 ISNGrammarVersion nodes (latest active: `0.7.0rc35`)
- Token id format: `{version}:{segment}:{value}` (e.g.
  `0.7.0rc35:geometry:strike_point`).
- Per AGENTS.md: `physical_base` is open-vocab; all other segments are
  closed against ISN token lists.

### 3.7 Plan 39 already covers (do NOT duplicate here)

- Fan-out dispatcher / proposer / synthesizer pattern.
- The MVP catalog (`search_existing_names`, `search_dd_paths`,
  `find_related_dd_paths`, `search_dd_clusters`).
- `search_similar_names(gc=None, ...)` overload — Phase 0(a) of plan 39.
- `cluster_search(gc, query, ...)` extraction — Phase 0(c) of plan 39.
- `related_dd_search(gc, path, ...)` already exists at
  `graph/dd_search.py:590` — plan 39 Phase 0(b) is a no-op.

---

## 4. Data model fix (write side)

### 4.1 Canonical storage decision

**Per-segment column properties on `StandardName`** are canonical.

Concretely, the canonical slots (matching what `catalog_import.py:428`
already writes and what `_segment_filter_search_sn` already expects):

```
sn.grammar_physical_base    : string | None   # OPEN-VOCAB
sn.grammar_subject          : string | None
sn.grammar_transformation   : string | None
sn.grammar_component        : string | None
sn.grammar_coordinate       : string | None
sn.grammar_process          : string | None
sn.grammar_position         : string | None
sn.grammar_region           : string | None
sn.grammar_device           : string | None
sn.grammar_geometric_base   : string | None
sn.grammar_object           : string | None
sn.grammar_geometry         : string | None
sn.grammar_secondary_base   : string | None
sn.grammar_binary_operator  : string | None
```

Notes:

- Type is **string** (the raw token value as the parser yields it),
  *not* the GrammarToken `id` (which is `{version}:{segment}:{value}`).
  Storing the raw value is what makes open-vocab work and what makes
  sibling lookups version-stable across rotations.
- Multivalued segments (today: none in the parser API) would store a
  `;`-separated string OR a list — not in scope.  ISN parser is
  single-token-per-segment in v0.7.
- These slots **already exist in the LinkML schema** as per-name
  properties (the schema additionally declares
  `grammar_<segment>_token` *edge* slots at `standard_name.yaml:803+`
  for the typed edges).  Schema rebuild is a no-op for the column
  slots; the schema-compliance test gains the new write expectations
  but no new declarations.

**Typed edges are kept as a secondary index** for closed-vocab segments
only.  They are `MERGE`-d when (and only when) the matching
`GrammarToken` exists — same OPTIONAL MATCH guard, no behaviour change
on misses.  They exist to support graph-traversal queries from other
plans (e.g. "all SNs that share a GrammarToken with this DD path's
inferred segment"); they are **not** the read source for
`siblings_by_segment` or hybrid search.

The legacy `grammar_fields` JSON blob is **removed from new writes**
(see §13 for the removal call-out).  `grammar_fields` continues to be
emitted by `models.StandardNameAttachment` until the follow-up
cleanup commit; the writer drops it on the floor when persisting.

### 4.2 Writer changes

#### 4.2.1 Before (current `_write_segment_edges`)

```python
# imas_codex/standard_names/graph_ops.py:1976-2120
def _write_segment_edges(gc, name_ids):
    # ... resolves token_version ...
    for sn_id in name_ids:
        parsed = parse_standard_name(sn_id)
        edge_specs = segment_edge_specs(parsed)
        # DELETE old edges
        gc.query("MATCH ... DELETE r", sn_id=sn_id)
        # OPTIONAL MATCH GrammarToken — drops on miss
        gc.query("""
            UNWIND $edges AS edge
            OPTIONAL MATCH (t:GrammarToken {value: edge.token, ...})
            FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
                MERGE (sn)-[r:HAS_SEGMENT]->(t) ...)
            ...
        """, sn_id=sn_id, edges=edges_param, ...)
```

The two writes happen back-to-back per SN.  No column properties are
ever set.  Open-vocab values disappear on the OPTIONAL MATCH miss.

#### 4.2.2 After

Rename to `_write_grammar_decomposition`. Two-phase, batched:

1. **Phase 4.2.A — parse all names, write columns (UNWIND batch).**
   Parse failures recorded but do not abort the batch.  This phase is
   **always idempotent and lossless**: every parser output goes
   straight to columns.

2. **Phase 4.2.B — write typed edges + HAS_SEGMENT for closed-vocab
   tokens (UNWIND batch).**  OPTIONAL MATCH guard preserved.  Misses
   are still logged as `VocabGap` candidates **only for closed-vocab
   segments** (the existing `filter_closed_segment_gaps` helper) — no
   change to the gap-reporting contract.

Sketch:

```python
def _write_grammar_decomposition(gc: GraphClient, name_ids: list[str]) -> list[VocabGap]:
    parser_rows: list[dict] = []
    edge_rows: list[dict] = []
    parse_failures: list[str] = []

    for sn_id in name_ids:
        try:
            parsed = parse_standard_name(sn_id)
        except Exception:
            parse_failures.append(sn_id)
            logger.warning("Grammar parse failed for '%s'", sn_id, exc_info=True)
            continue

        # Phase A: column row (always written, even on parse-fallback).
        parser_rows.append({
            "id": sn_id,
            "grammar_physical_base":   parsed.physical_base,    # may be raw open-vocab
            "grammar_subject":         parsed.subject,
            "grammar_transformation":  parsed.transformation,
            "grammar_component":       parsed.component,
            "grammar_coordinate":      parsed.coordinate,
            "grammar_process":         parsed.process,
            "grammar_position":        parsed.position,
            "grammar_region":          parsed.region,
            "grammar_device":          parsed.device,
            "grammar_geometric_base":  parsed.geometric_base,
            "grammar_object":          parsed.object,
            "grammar_geometry":        parsed.geometry,
            "grammar_secondary_base":  parsed.secondary_base,
            "grammar_binary_operator": parsed.binary_operator,
        })

        # Phase B: edge specs (closed-vocab only — open-vocab segments
        # are skipped at spec time, not at MERGE time).
        for spec in segment_edge_specs(parsed):
            edge_rows.append({
                "sn_id":   sn_id,
                "segment": spec.segment,
                "token":   spec.token,
                "position": spec.position,
            })

    # ---- Phase A: bulk SET columns (single round-trip) ----
    if parser_rows:
        gc.query(
            """
            UNWIND $rows AS r
            MATCH (sn:StandardName {id: r.id})
            SET sn.grammar_physical_base    = r.grammar_physical_base,
                sn.grammar_subject          = r.grammar_subject,
                sn.grammar_transformation   = r.grammar_transformation,
                sn.grammar_component        = r.grammar_component,
                sn.grammar_coordinate       = r.grammar_coordinate,
                sn.grammar_process          = r.grammar_process,
                sn.grammar_position         = r.grammar_position,
                sn.grammar_region           = r.grammar_region,
                sn.grammar_device           = r.grammar_device,
                sn.grammar_geometric_base   = r.grammar_geometric_base,
                sn.grammar_object           = r.grammar_object,
                sn.grammar_geometry         = r.grammar_geometry,
                sn.grammar_secondary_base   = r.grammar_secondary_base,
                sn.grammar_binary_operator  = r.grammar_binary_operator
            """,
            rows=parser_rows,
        )

    # ---- Phase B: bulk DELETE-then-MERGE typed edges (idempotent) ----
    # ... unchanged from current writer except hoisted out of per-SN loop ...

    return _collect_vocab_gaps(edge_rows, results)
```

#### 4.2.3 Call-site updates

| Call-site | File / line | Change |
|---|---|---|
| `write_standard_names()` (build path) | `graph_ops.py` (search "_write_segment_edges") | Replace call with `_write_grammar_decomposition` |
| `_write_catalog_entries()` (import path) | `catalog_import.py:424-443` | Delete the inline grammar-column SET block (now subsumed by `_write_grammar_decomposition`); call the helper instead |
| `persist_refined_name_batch` | `graph_ops.py` (find by name) | Call `_write_grammar_decomposition` for each newly-created refined SN; predecessor's columns are left intact (it remains queryable) |
| `compose_worker` candidate persistence | `workers.py` (around lines 2120, 3501, 3911) | Stop passing `grammar_fields=...` into `write_standard_names`; the helper does its own parse from the canonical SN id |
| `_write_segment_edges` | `graph_ops.py:1976` | Renamed and rewritten as above |

The compose worker's `StandardNameAttachment.grammar_fields` field
remains in the Pydantic model for now (LLM still emits it; we just
stop persisting it).  Removing it from the model is a follow-up.

### 4.3 Open vs closed segment handling

The decision matrix in the new writer:

| Segment | Closed in ISN v0.7? | Column write | Typed edge write | VocabGap on miss? |
|---|---|---|---|---|
| `physical_base` | **No** (open-vocab) | always | only if GrammarToken exists (rarely) | **No** (open-vocab gaps are not gaps) |
| `subject`, `transformation`, `component`, `coordinate`, `process`, `position`, `region`, `device`, `geometric_base`, `object`, `geometry`, `secondary_base`, `binary_operator` | Yes | always | guarded by OPTIONAL MATCH | yes (existing `filter_closed_segment_gaps`) |

The "open vs closed" decision is **not hard-coded in the writer**.  It
is sourced at module load from
`imas_standard_names.grammar.get_grammar_context()` — same pattern as
`imas_codex/standard_names/audits.py`.  This is required by AGENTS.md
"Architecture Boundary":

> Never hardcode grammar rules — get them from `get_grammar_context()`.

### 4.4 Idempotency

- Phase A: `MATCH … SET` is naturally idempotent.  Re-running on the
  same `sn_id` overwrites with the same parser output.
- Phase B: existing DELETE-then-MERGE pattern preserved; idempotent by
  construction.
- Parse-fallback case (parser returns whole-string `physical_base`) is
  **stable across re-runs** as long as the ISN version is unchanged —
  the parser is deterministic.  ISN version rotation may shift segment
  assignments; that is expected and handled by the pipeline's existing
  rotation workflow (AGENTS.md "Vocabulary Rotation: ISN Fork RC
  Workflow").

A re-run on a graph with stale `grammar_fields` blobs leaves the blobs
untouched (they continue to live as orphaned data) until the follow-up
cleanup commit drops them via:

```cypher
MATCH (sn:StandardName) REMOVE sn.grammar_fields
```

(That cleanup is **not** in this plan; see §13.)

---

## 5. Search redesign (read side)

### 5.1 Query parsing (deterministic ISN call, no LLM)

Add a thin helper in `imas_codex/standard_names/search.py`:

```python
def parse_query_segments(query: str) -> dict[str, str]:
    """Best-effort decomposition of *query* into ISN grammar segments.

    Returns an empty dict if the query is plainly natural language
    (e.g. contains spaces, punctuation other than '_', or is shorter
    than 3 chars).  Otherwise calls
    ``imas_standard_names.grammar.parse_standard_name`` and returns the
    populated segments only (drops None).

    Pure / deterministic.  Never raises — parse failures yield {}.
    """
```

The "is this an SN candidate?" predicate is intentionally cheap — we
only invoke the parser when the query *looks like* a candidate
(`re.fullmatch(r"[a-z][a-z0-9_]+", query.strip())`).  Callers may
override with `as_candidate=True` to force a parse.

### 5.2 Three streams

Each stream is an independently-bounded helper that returns
`list[StreamHit]`:

```python
class StreamHit(BaseModel):
    sn_id:   str
    rank:    int           # 1-based within this stream
    raw:     float | None  # native score (vector cosine, count, …)
```

| Stream | Helper | Source | k_default |
|---|---|---|---|
| **Vector** | `_vector_stream(gc, query, k)` | `db.index.vector.queryNodes('standard_name_desc_embedding', …)` (existing) | 20 |
| **Grammar** | `_grammar_stream(gc, segments, k)` | `MATCH (sn:StandardName) WHERE sn.grammar_physical_base = $val OR …` over the **column properties** | 20 |
| **Keyword** | `_keyword_stream(gc, query, k)` | `WHERE toLower(sn.id) CONTAINS $q OR toLower(sn.description) CONTAINS $q` (existing logic) | 20 |

The grammar stream is **column-based, not edge-based**, so it surfaces
SNs across open-vocab tokens too.  Concretely:

```cypher
// Worked example — query = "parallel_current_density_weight" parses to
// physical_base='current_density', component='parallel', secondary_base='weight'
// (if parser produces that decomposition).
MATCH (sn:StandardName)
WHERE sn.grammar_physical_base = $physical_base
   OR sn.grammar_component     = $component
   OR sn.grammar_secondary_base = $secondary_base
WITH sn,
     (CASE WHEN sn.grammar_physical_base    = $physical_base    THEN 3 ELSE 0 END +
      CASE WHEN sn.grammar_component        = $component        THEN 1 ELSE 0 END +
      CASE WHEN sn.grammar_secondary_base   = $secondary_base   THEN 1 ELSE 0 END) AS hits
WHERE hits > 0
RETURN sn.id AS sn_id, hits AS raw
ORDER BY hits DESC, sn.id ASC
LIMIT $k
```

Per-segment weighting (1 / 1 / 3 / 1 …) is encoded in the helper as
a constant dict, **not** in user-tunable settings.  The sort tie-breaker
is `sn.id ASC` for determinism.

`physical_base` carries weight 3 because it is the most discriminating
slot empirically (open-vocab values are highly distinctive: only one
SN has `physical_base='temperature_electron_average'`, etc.).
`subject` carries weight 2; all others 1.  Final weights chosen for
plan 40:

```
physical_base    : 3
subject          : 2
component        : 1
coordinate       : 1
geometric_base   : 1
secondary_base   : 1
process          : 1
position         : 1
region           : 1
device           : 1
transformation   : 1
object           : 1
geometry         : 1
binary_operator  : 1
```

### 5.3 RRF fusion math

Reciprocal Rank Fusion as defined by Cormack, Clarke, Buettcher (SIGIR
2009):

```
RRF(d) = Σ_streams  1 / (k_rrf + rank_stream(d))
```

with `k_rrf = 60` (the canonical TREC default).  For absent streams,
the term is 0.

```python
K_RRF = 60

def rrf_fuse(streams: dict[str, list[StreamHit]]) -> list[FusedHit]:
    score: dict[str, float] = defaultdict(float)
    contrib: dict[str, dict[str, int]] = defaultdict(dict)
    for stream_name, hits in streams.items():
        for h in hits:
            score[h.sn_id] += 1.0 / (K_RRF + h.rank)
            contrib[h.sn_id][stream_name] = h.rank
    fused = [
        FusedHit(sn_id=sid, score=score[sid], stream_ranks=contrib[sid])
        for sid in score
    ]
    fused.sort(key=lambda x: (-x.score, x.sn_id))  # deterministic
    return fused
```

`k_rrf=60` is a **hard-coded constant**, not a setting (per
AGENTS.md "Configuration" — defaults bake in when there's no operator
need to tune them).

**Worked unit-test fixture (used in §9):**

| sn_id   | vector rank | grammar rank | keyword rank | RRF score        |
|---------|:-----------:|:------------:|:------------:|------------------|
| `electron_temperature`   | 1 | 1 | 2 | 1/61 + 1/61 + 1/62 ≈ 0.0489 |
| `ion_temperature`        | 2 | 1 | 4 | 1/62 + 1/61 + 1/64 ≈ 0.0481 |
| `temperature`            | 5 | 1 | 1 | 1/65 + 1/61 + 1/61 ≈ 0.0482 |

Order: `electron_temperature` > `temperature` > `ion_temperature`.  All
three streams contribute; the test asserts the order and the per-hit
provenance.

### 5.4 Output schema (hits with stream provenance)

```python
class FusedHit(BaseModel):
    sn_id:        str
    score:        float                   # fused RRF score
    stream_ranks: dict[str, int]          # {"vector": 1, "grammar": 3, "keyword": None}
    sn:           StandardNameSummary     # joined fields (id, description, kind, unit, …)

class HybridSearchReport(BaseModel):
    query:        str
    parsed:       dict[str, str]          # {} when not parseable
    streams_used: list[str]
    hits:         list[FusedHit]
```

The MCP-tool string formatter renders `stream_ranks` as `[V:1 G:3 K:-]`
inline next to each hit — short, scannable, and makes the new behaviour
visible to the operator.

### 5.5 Siblings mode signature

```python
def siblings_by_segment(
    gc: GraphClient,
    *,
    value: str,
    segment: str,                # one of the canonical segment names
    exclude: list[str] | None = None,
    k: int = 50,
) -> list[StandardNameSummary]:
    """Return all SNs whose ``sn.grammar_<segment>`` equals *value*.

    Pure column lookup.  No vector index, no LLM, no parsing.  Used by:
      - the catalog-audit "find me all SNs sharing this base" workflow,
      - the reviewer prompt context (§7), to show siblings in the
        decomposition rule (sn_review_criteria.yaml I4.6).

    *exclude* removes specific ids (typically the candidate SN itself).
    Returns up to *k* rows ordered by ``sn.id`` ascending (deterministic).
    """
```

Cypher:

```cypher
MATCH (sn:StandardName)
WHERE sn.grammar_<segment> = $value
  AND NOT sn.id IN $exclude
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
RETURN sn.id AS id, sn.description AS description, sn.kind AS kind,
       coalesce(u.id, sn.unit) AS unit
ORDER BY sn.id ASC
LIMIT $k
```

The `<segment>` interpolation is **not** user-supplied SQL — `segment`
is validated against a hard-coded `Literal[...]` set at the API
boundary, and the Cypher template is built from that allow-list.

### 5.6 Group-by-base mode signature

```python
def group_by_base(
    gc: GraphClient,
    *,
    physics_domain: str | None = None,
    min_group_size: int = 1,
) -> list[BaseGroup]:
    """Return SN groups keyed by (physical_base, subject).

    Used by catalog audits to surface families and to spot orphans
    (groups of size 1 are likely candidates for renaming or for a
    new sibling).
    """
```

Cypher:

```cypher
MATCH (sn:StandardName)
WHERE ($pd IS NULL OR sn.physics_domain = $pd)
  AND sn.grammar_physical_base IS NOT NULL
WITH sn.grammar_physical_base AS base,
     coalesce(sn.grammar_subject, '') AS subject,
     collect(sn.id) AS members
WHERE size(members) >= $min_group_size
RETURN base, subject, members, size(members) AS n
ORDER BY n DESC, base ASC, subject ASC
```

### 5.7 Worked end-to-end example

Query `parallel_current_density_weight` (a hypothetical SN candidate
the reviewer is considering):

1. **Parse**: `physical_base='current_density'`, `component='parallel'`,
   `secondary_base='weight'` (assuming parser supports the suffix).
2. **Vector stream**: top-20 from
   `standard_name_desc_embedding` against the raw query string.
3. **Grammar stream** (column-MATCH, weighted):
   - hits SNs sharing `physical_base='current_density'` (weight 3 each),
   - hits SNs sharing `component='parallel'` (weight 1),
   - hits SNs sharing `secondary_base='weight'` (weight 1).
4. **Keyword stream**: substring on the raw string.
5. **RRF fuse** the three streams.

A pre-existing SN `equilibrium_reconstruction_parallel_current_density_weight`
that did not embed well (under-described in `description`) but shares
the `physical_base` and `component` is now visible — pre-plan-40 it
would have been invisible to a vector-only search and the segment
filter would have required the operator to type
`physical_base=current_density&component=parallel` by hand.

---

## 6. Fetch facility extensions

`fetch_standard_names(names: str, *, with_grammar=False, with_neighbours=False)`:

```python
def _fetch_standard_names(
    names: str,
    *,
    with_grammar: bool = False,
    with_neighbours: bool = False,
    gc: GraphClient | None = None,
) -> str:
    """Fetch full entries for known standard names.

    Args:
        names: Space- or comma-separated standard name IDs.
        with_grammar: If True, include parsed grammar segments
            (column properties) and the typed-edge tokens (where they
            exist) in the output report.
        with_neighbours: If True, for each name, run
            ``siblings_by_segment(value=sn.grammar_physical_base,
            segment='physical_base')`` (and ``subject`` if non-null)
            and report the count + first 5 sibling ids.
    """
```

`with_neighbours=True` is the lever for the user's workflow:

> "this is the only SN with this base — investigate"

For the deployed graph today, every SN with no `grammar_physical_base`
column will report `with_neighbours: <unknown — column not populated>`.
After §4 lands, that becomes a real count.

---

## 7. MCP exposure

### 7.1 Verify existing registration works

The orchestrator's audit notes that the user's MCP-client palette did
not list `search_standard_names` etc.  Two gates exist in
`imas_codex/llm/server.py`:

1. `if not self.dd_only:` (line 3055) — wraps `fetch_content`. (The
   SN block is **outside** this guard, so SN tools are *not* dd_only-gated
   today.)
2. `if self.include_standard_names:` (line 3084) — wraps the SN block.
   Default value of `include_standard_names` must be confirmed at
   `MCPServer.__init__`; if the default is `False` for some launch
   modes, that explains the missing palette entry.

**Plan 40 action (Phase 3 of §8):** verify the default in the
`imas-codex` launcher config and document the expected truth-table:

| Launch mode      | `dd_only` | `include_standard_names` | SN tools registered? |
|------------------|-----------|--------------------------|-----------------------|
| `imas-codex`     | False     | True                     | yes                   |
| `imas-codex --dd-only` | True | False                  | no                    |
| `imas-dd` (frozen remote) | True | n/a (no SN data) | no                    |

If the truth-table reveals that `include_standard_names` defaults to
`False` for the dev launch, fix the default and add a regression test
in `tests/llm/test_server_registration.py`.  No source change is
in this plan if the truth-table is already correct.

### 7.2 New `siblings_by_segment` MCP tool

Add a single new MCP tool, registered in the same SN block:

```python
@self.mcp.tool()
def siblings_by_segment(
    value: str,
    segment: str,
    exclude: list[str] | None = None,
    k: int = 50,
) -> str:
    """List all standard names sharing a given grammar segment value.

    Use to inspect families:
        siblings_by_segment(value="temperature", segment="physical_base")
        siblings_by_segment(value="electron",   segment="subject")

    Args:
        value: Token value (raw string — case-sensitive, as parsed
            by ISN).  For closed-vocab segments, see
            ``list_grammar_vocabulary`` for valid tokens.
        segment: One of physical_base, subject, transformation,
            component, coordinate, process, position, region, device,
            geometric_base, object, geometry, secondary_base,
            binary_operator.
        exclude: SN ids to omit (typically the focal candidate).
        k: Maximum rows (default 50).

    Returns:
        Markdown table of SN id, description, kind, unit.
    """
    from imas_codex.llm.sn_tools import _siblings_by_segment as _sbs
    return _sbs(value=value, segment=segment, exclude=exclude or [], k=k)
```

Reasoning for "new tool" vs "kwarg fold": the search tool already
carries 16 kwargs.  A focused tool is more discoverable and the
implementation is a 5-line column-lookup that does not justify a new
branch in `_search_standard_names`.

### 7.3 DD-only mode behaviour

`siblings_by_segment` is registered inside the existing
`if self.include_standard_names:` block, **not** inside
`if not self.dd_only:`.  The two flags are independent (SN data is
local-graph, but it does not require facility data).  Adding a
defensive runtime check at the entry of `_siblings_by_segment` that
returns "Standard Names not available on this server" when the SN
namespace is empty is worthwhile and cheap.

---

## 8. Phased rollout

| Phase | Scope | Verification gate |
|-------|-------|---------------------|
| **1. Writer fix** | §4 — rename + rewrite `_write_segment_edges` → `_write_grammar_decomposition`; replace the inline column SET in `catalog_import.py`; remove `grammar_fields=` propagation from `compose_worker`. | Unit: parser → column round-trip; integration: `sn run --paths` on a 5-SN seed produces 5 SNs with all parser-emitted columns populated; `sns_with_grammar_col == sns_total`. |
| **2. Read-side** | §5 — `parse_query_segments`, three stream helpers, `rrf_fuse`, refactor `_search_standard_names` to call them; add `siblings_by_segment` and `group_by_base` Python helpers. | Unit: RRF math test (worked fixture from §5.3); unit: stream helpers return deterministic order; integration: hybrid search on a seeded 10-SN graph surfaces siblings that vector-only misses. |
| **3. MCP exposure** | §7 — verify registration truth-table; add `siblings_by_segment` MCP tool; `with_grammar` / `with_neighbours` kwargs on `fetch_standard_names`. | Tool-availability test (asserts the four SN tools register under non-dd-only mode); end-to-end manual smoke against the dev `imas-codex` MCP server. |

Each phase commits and pushes independently.  Phase 2 hard-depends on
Phase 1 (the grammar stream is column-based; columns must be populated
first).  Phase 3 has only soft dependencies.

---

## 9. Test strategy (per AGENTS.md "Schema-Driven Testing")

**Discipline reminder:** AGENTS.md forbids adding LinkML schema
declarations *just to make tests green*.  The slots used in this plan
already exist in the schema (§4.1).  No schema additions.

### 9.1 Phase 1 — writer

| Test | File | Kind | Asserts |
|------|------|------|---------|
| `test_write_grammar_columns_populated` | `tests/standard_names/test_graph_ops.py` | unit (mocked `gc.query`) | every parser-emitted segment lands on the SET payload |
| `test_open_vocab_physical_base_kept` | same | unit | `physical_base="major_radius"` (no GrammarToken) survives to `sn.grammar_physical_base` |
| `test_writer_idempotent` | same | unit | running twice with same input produces identical SET payloads |
| `test_grammar_columns_e2e_compose` | `tests/integration/test_compose_persistence.py` | integration (live test graph) | after `compose_batch` of 5 fixtures, `MATCH (sn) WHERE sn.grammar_physical_base IS NULL RETURN count(sn) → 0` |
| `test_no_grammar_fields_blob_on_new_writes` | same | integration | `sn.grammar_fields IS NULL` for all newly-persisted SNs |

### 9.2 Phase 2 — search

| Test | File | Kind | Asserts |
|------|------|------|---------|
| `test_parse_query_segments` | `tests/standard_names/test_search.py` | unit | natural-language queries → `{}`; SN-shaped queries → parsed dict |
| `test_rrf_fuse_worked_example` | same | unit | the §5.3 fixture produces exact ordering and scores within 1e-6 |
| `test_grammar_stream_column_match` | same | unit (mocked `gc.query`) | weighted hit count matches §5.2 weights |
| `test_siblings_by_segment` | same | unit | column-MATCH Cypher template renders for each allowed segment |
| `test_hybrid_search_surfaces_siblings` | `tests/integration/test_search.py` | integration | seeded graph: 5 SNs share `physical_base='temperature'`, only 1 has a strong vector hit; hybrid returns all 5 |

### 9.3 Phase 3 — MCP

| Test | File | Kind | Asserts |
|------|------|------|---------|
| `test_sn_tools_registered_when_include_standard_names` | `tests/llm/test_server_registration.py` | unit | `MCPServer(include_standard_names=True, dd_only=False)` exposes `search_standard_names`, `fetch_standard_names`, `list_standard_names`, `siblings_by_segment` |
| `test_sn_tools_hidden_under_dd_only` | same | unit | `dd_only=True` (or `include_standard_names=False`) ⇒ none registered |
| `test_fetch_with_neighbours` | `tests/llm/test_sn_tools.py` | integration | `_fetch_standard_names("electron_temperature", with_neighbours=True)` returns sibling counts derived from columns |

### 9.4 Schema-compliance

`tests/graph/test_schema_compliance.py` already asserts that every
property declared in the LinkML schema for `StandardName` is either
`None` or a string on persisted nodes.  After Phase 1, the
column-population assertion strengthens to: for SNs with `name_stage IN
('drafted', 'accepted', 'reviewed', 'superseded')`, every parser-emitted
segment column is non-null when the parser emits a value for it.

This is a **derived assertion**, not a new schema declaration — it
respects the AGENTS.md ban on using schema as a test crutch.

---

## 10. Configuration

No new settings.  Constants baked in:

| Constant | Where | Value |
|---|---|---|
| `K_RRF` | `imas_codex/standard_names/search.py` | `60` |
| Per-segment weights | `imas_codex/standard_names/search.py` | dict in §5.2 |
| Stream `k_default` | `imas_codex/standard_names/search.py` | `20` |
| Sibling default `k` | `imas_codex/standard_names/search.py` | `50` |

If operator-tuning ever becomes necessary (telemetry-gated, not
speculative), promote into `[tool.imas-codex.sn.search]` in
`pyproject.toml`.  Out of scope today.

---

## 11. Acceptance criteria

The plan is "done" when **all** of the following hold against the live
`imas-codex` MCP server, against a freshly-cleared-and-seeded graph
of ≥ 30 SNs spanning at least 3 distinct `grammar_physical_base`
values:

1. **A1.** `MATCH (sn:StandardName) WHERE sn.grammar_physical_base IS NULL RETURN count(sn)` → `0` for SNs whose parser emitted a `physical_base`.
2. **A2.** `MATCH (sn:StandardName) WHERE sn.grammar_fields IS NOT NULL RETURN count(sn)` → `0` (new writes do not emit the legacy blob).
3. **A3.** Calling `siblings_by_segment(value="temperature", segment="physical_base")` returns ≥ 2 rows (assuming the seed has them).
4. **A4.** A vector-only search for `"plasma stored thermal energy"` and a hybrid search for the same string both succeed; the hybrid result includes ≥ 1 SN sharing the candidate's `physical_base` that is **not** in the vector top-20 (proven by per-stream rank provenance).
5. **A5.** `fetch_standard_names("electron_temperature", with_neighbours=True)` returns a sibling count derived from the column lookup, not from the typed-edge fallback.
6. **A6.** `tests/graph/test_schema_compliance.py` and the new tests in §9 all pass.
7. **A7.** Running `_write_grammar_decomposition` twice on the same `name_ids` produces no graph diff.
8. **A8.** Server-registration test asserts the four SN tools register under `dd_only=False` and none under `dd_only=True`.

---

## 12. Migration / clear policy

Per the user's policy (AGENTS.md "Reset and Clear Semantics" + the
direct guidance: *"we have a non-backward compatible policy to reduce
code clutter"*):

- **No backfill.**  The graph is small (22 SNs) and will be cleared
  via `sn clear` before this lands.
- **`sn clear` continues to work** (it deletes all SNs anyway; no
  change required).
- **`grammar_fields` blob removal** lives in a follow-up cleanup
  commit on the same feature branch:
  ```cypher
  MATCH (sn:StandardName) REMOVE sn.grammar_fields
  ```
  plus deletion of the `grammar_fields` slot in
  `imas_codex/standard_names/models.py` (`StandardNameAttachment`).
- **Catalog round-trip safety**: if a published catalog YAML carries
  `grammar_fields:` (legacy), `_write_catalog_entries` should silent-
  ignore (mirrors the existing `confidence:` tolerance pattern at
  AGENTS.md "Write Semantics").

---

## 13. Documentation updates required

| File | Change |
|---|---|
| `AGENTS.md` § "Standard Names" → "Schema" | Add a "Grammar decomposition storage" sub-section: per-segment columns are canonical; typed edges are a secondary index; `physical_base` is open-vocab and stored as raw string. |
| `AGENTS.md` § "Standard Names" → "MCP Tools" | Add `siblings_by_segment`; clarify that `search_standard_names` now uses RRF and is no longer mutually exclusive across modes. Update the docstring claim "Hybrid search (vector + keyword)" to match reality. |
| `AGENTS.md` § "Write Semantics" | Note the unified write helper `_write_grammar_decomposition` and that legacy `grammar_fields` is dropped on new writes. |
| `docs/architecture/standard-names.md` | Append the worked example from §5.7 + the §5.3 RRF fixture. |
| `imas_codex/standard_names/search.py` (module docstring) | Document streams, fusion math, and the `parse_query_segments` heuristic. |
| `imas_codex/standard_names/graph_ops.py` (module docstring) | Document the writer's two-phase contract. |
| `imas_codex/llm/sn_tools.py` | Update `_search_standard_names` docstring; remove the "branches are mutually exclusive" wording. |
| `imas_codex/schemas/standard_name.yaml` | No change. The column slots already exist; the typed-edge slots already exist. |
| `agents/schema-reference.md` | Auto-rebuilds from LinkML; not touched by hand. |
| `plans/features/standard-names/40-...md` | This plan; tick boxes as work proceeds. |

---

## 14. What this plan does NOT do

- ❌ No backfill of existing SNs in deployed graphs.
- ❌ No removal of `grammar_fields` from `StandardNameAttachment` (the
  Pydantic model the LLM emits) — that is a follow-up cleanup commit.
- ❌ No change to the ISN parser, vocabulary, or grammar context.
- ❌ No new vector index; no re-embedding of descriptions.
- ❌ No schema additions to `GrammarToken` / `GrammarSegment` /
  `ISNGrammarVersion`.
- ❌ No multi-token-per-segment storage.  The ISN parser is
  single-token-per-segment in v0.7; multi-token would change the
  column type to a list and is out of scope.
- ❌ No LLM-judged re-ranking of fused results.  Fusion is deterministic.
- ❌ No new graph schema fields on the existing typed edges (e.g.
  storing parser confidence on `HAS_PHYSICAL_BASE` rel-props).
- ❌ No telemetry node for search calls (`Fanout`-style).  Use the
  existing pipeline log lines; promote to a node only if a follow-up
  plan needs it.
- ❌ No change to the `compose_worker` LLM prompts.  The reviewer-side
  prompt-context refactor (use `siblings_by_segment` to enrich
  decomposition-rule context) is left for a follow-up plan that
  consumes this plan's helpers.

---

## 15. Interaction with plan 39

Plan 39 introduces a structured fan-out dispatcher around the
**refine_name** prompt with a closed catalog of four backing helpers:

```
search_existing_names | search_dd_paths | find_related_dd_paths | search_dd_clusters
```

Plan 39 owns:

- The proposer LLM, the discriminated-union schema, and the executor.
- Phase 0 helper extractions in `imas_codex/graph/dd_search.py`.
- The Phase 1 telemetry gate.

**Plan 40 underwrites plan 39's `search_existing_names` runner.**
After plan 40 lands:

- `search_existing_names(query, k, *, gc=None, …)` calls into
  `search.py::search_similar_names`, which is *itself* enhanced by
  plan 40 to route through the new hybrid pipeline (vector + grammar +
  keyword) when the query parses as an SN candidate.  The signature is
  unchanged; only the internals improve.
- The fan-out planner doesn't need a new `fn_id` — `search_existing_names`
  is now grammar-aware "for free".
- The fan-out hits-shape (`FanoutHit.payload`) gains an optional
  `stream_ranks: dict[str, int]` carry-through, mirroring §5.4.  Plan 39's
  Pydantic model is `dict[str, Any]` so this is purely additive.
- `siblings_by_segment` is **not** added to plan 39's catalog — it is
  not a generic fan-out helper; it is a focused MCP tool / Python
  helper for direct use by reviewers and audits.

**Order of merge:** plan 40 Phase 1 (writer) and Phase 2 (search
helpers) should land **before** plan 39 Phase 1 (refine_name fan-out
pilot).  Phase 3 (MCP) is order-independent.

---

## 16. Open questions for further RD review

1. **Per-segment weight tuning** (§5.2): the 3 / 2 / 1 weighting is
   ergonomic but unmeasured.  Once telemetry shows the grammar-stream
   contribution distribution, we may tune.  Defer until Phase 2 lands
   and produces real ranks.
2. **Multi-token segments**: ISN v0.7 is single-token-per-segment.  If
   v0.8 introduces multi-token segments (e.g. multiple `process`
   prefixes), the column type becomes `list[str]` and the grammar
   stream's `MATCH … = $val` becomes `$val IN sn.grammar_<seg>`.
   Tracked as a follow-up; not blocking.
3. **`group_by_base` exposure as MCP tool**: only added as a Python
   helper in this plan.  Promote to MCP tool if reviewers ask.
4. **Decommissioning the typed edges entirely**: once the column path
   has stable telemetry, the typed edges become pure overhead.  A
   future plan may delete them.  Not in scope here.

---

*End of plan 40.*
