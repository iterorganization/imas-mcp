# Plan 40 — Grammar-aware SN Search & Fetch Facility

> **Status**: DRAFT v2 — RD findings F1–F9 addressed; plan only; no code in this task.
> **Version history**:
> - v1 — initial draft.
> - **v2 (this revision)** — addresses RD critique F1–F9: free-text grammar
>   stream via tokenisation fallback (F1); per-segment streams replace
>   weighted-count fusion so RRF naturally weighs `physical_base` higher
>   (F2); idempotent self-backfill in Phase 1 deploy (F3); explicit
>   parser-narrowing test (F4); `mode={"hybrid","vector"}` kwarg on
>   `search_similar_names` to preserve plan-39 collision-avoidance
>   semantics (F5); `grammar_parse_fallback` flag for `group_by_base`
>   audits (F6); concurrent stream execution + early-return (F7);
>   schema-compliance assertion clarified to re-parse, not read schema
>   (F8); confirm `StandardNameAttachment` payload coexistence (F9).
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
        # F6: detect parse-fallback so group_by_base can exclude these
        # from the orphan-flag bucket.  Definition: physical_base equals
        # the whole sn_id AND every other segment is None — i.e. the
        # parser found no closed-vocab token to peel.
        is_fallback = (
            parsed.physical_base == sn_id
            and all(
                getattr(parsed, seg) is None
                for seg in (
                    "subject", "transformation", "component", "coordinate",
                    "process", "position", "region", "device",
                    "geometric_base", "object", "geometry",
                    "secondary_base", "binary_operator",
                )
            )
        )
        parser_rows.append({
            "id": sn_id,
            "grammar_physical_base":    parsed.physical_base,    # may be raw open-vocab
            "grammar_subject":          parsed.subject,
            "grammar_transformation":   parsed.transformation,
            "grammar_component":        parsed.component,
            "grammar_coordinate":       parsed.coordinate,
            "grammar_process":          parsed.process,
            "grammar_position":         parsed.position,
            "grammar_region":           parsed.region,
            "grammar_device":           parsed.device,
            "grammar_geometric_base":   parsed.geometric_base,
            "grammar_object":           parsed.object,
            "grammar_geometry":         parsed.geometry,
            "grammar_secondary_base":   parsed.secondary_base,
            "grammar_binary_operator":  parsed.binary_operator,
            "grammar_parse_fallback":   is_fallback,             # F6
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
                sn.grammar_binary_operator  = r.grammar_binary_operator,
                sn.grammar_parse_fallback   = r.grammar_parse_fallback
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

**F9 — verify Pydantic payload coexistence.** Before Phase 1 lands,
confirm `StandardNameAttachment.model_config.extra` (and the parent
`StandardNameComposeBatch`) do not raise on the now-orphaned field:
the field stays declared on the model so existing payloads validate;
production read paths do not consume `grammar_fields` (confirmed in
§3.1 — only `benchmark.py` reads it, which is offline tooling, and
the `workers.py` references are write-side propagation that this plan
removes).  The verification is a one-grep + one-test step in Phase 1
and is non-blocking: if a downstream reader is found, it is migrated
to read from the new columns in the same commit.

**Schema slot for `grammar_parse_fallback` (F6).** This is a new
boolean property on `StandardName`.  It is added to
`imas_codex/schemas/standard_name.yaml` as a single boolean slot
(no relationship, no enum).  Schema rebuild happens in Phase 1; the
auto-generated `models.py` and `schema-reference.md` rebuild
automatically and are NOT staged manually (per AGENTS.md "Schema
System").  This is a legitimate schema addition (storing real
provenance), not a test-driven crutch addition (per AGENTS.md
"Schema-Driven Testing").

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

- Phase A: `MATCH … SET prop = r.prop` is naturally idempotent.
  Re-running on the same `sn_id` overwrites with the same parser
  output.  **Critical Cypher semantics (F4):** `SET prop = null`
  *removes* the property, so when a parser-narrowing event leaves a
  segment unset (e.g. ISN rotation reclassifies `'electron'` from
  `subject` to `species`), re-running the writer **clears stale
  columns**.  This is the desired behaviour and is asserted by the
  Phase 1 test suite (§9.1, `test_writer_clears_segments_when_parser_narrows`).
- Phase B: existing DELETE-then-MERGE pattern preserved; idempotent by
  construction.
- Parse-fallback case (parser returns whole-string `physical_base` and
  no other segments) is **stable across re-runs** as long as the ISN
  version is unchanged — the parser is deterministic.  ISN version
  rotation may shift segment assignments; that is expected and handled
  by the pipeline's existing rotation workflow (AGENTS.md "Vocabulary
  Rotation: ISN Fork RC Workflow").

A re-run on a graph with stale `grammar_fields` blobs leaves the blobs
untouched (they continue to live as orphaned data) until the follow-up
cleanup commit drops them via:

```cypher
MATCH (sn:StandardName) REMOVE sn.grammar_fields
```

(That cleanup is **not** in this plan; see §13.)

**Self-backfill on Phase 1 deploy (F3).** Because the writer is
idempotent and lossless, the Phase 1 deploy script invokes it once
over **every existing SN id** as the *first* step:

```python
# scripts/sn_deploy_phase1.py (single-shot, runs once during deploy)
with GraphClient() as gc:
    all_ids = [r["id"] for r in gc.query("MATCH (sn:StandardName) RETURN sn.id AS id")]
    _write_grammar_decomposition(gc, all_ids)
```

This replaces "backfill" (which the plan declines as a separate
operation) with "rerun the canonical writer" — same code path, same
guarantees, no new logic.  After this runs, **every** SN has its
columns populated regardless of when it was originally written.  The
Phase 1 acceptance gate (§11 A1) then checks that no active-stage SN
has `grammar_physical_base IS NULL` after parse succeeded; if it does,
the deploy fails loudly instead of silently degrading the new search
features.

---

## 5. Search redesign (read side)

### 5.1 Query parsing (deterministic ISN call, no LLM)

Two helpers in `imas_codex/standard_names/search.py`:

```python
def parse_query_segments(query: str) -> dict[str, str]:
    """Best-effort grammar decomposition of *query*.

    Calls ``imas_standard_names.grammar.parse_standard_name`` when the
    query *looks like* an SN candidate
    (``re.fullmatch(r"[a-z][a-z0-9_]+", query.strip())``).  Returns the
    populated segments only (drops None).

    Pure / deterministic.  Never raises — parse failures yield {}.
    Callers may force a parse with ``as_candidate=True``.
    """

def tokenise_query(query: str) -> list[str]:
    """Lowercase token list for the free-text grammar fallback (F1).

    ``re.findall(r"[a-z][a-z0-9]+", query.lower())``.  Stopwords are
    NOT filtered — the grammar columns themselves are the filter; a
    spurious token like "of" simply finds no column match.
    """
```

**Free-text grammar fallback (F1).** When `parse_query_segments(query)
== {}` the grammar stream does **not** go dark.  It falls back to a
*token-against-any-column* match using `tokenise_query`:

```cypher
// Free-text fallback — query = "plasma stored thermal energy"
// tokens = ["plasma", "stored", "thermal", "energy"]
MATCH (sn:StandardName)
WHERE sn.grammar_physical_base   IN $tokens
   OR sn.grammar_subject         IN $tokens
   OR sn.grammar_component        IN $tokens
   OR sn.grammar_coordinate       IN $tokens
   OR sn.grammar_geometric_base   IN $tokens
   OR sn.grammar_secondary_base   IN $tokens
   OR sn.grammar_process          IN $tokens
   OR sn.grammar_position         IN $tokens
   OR sn.grammar_region           IN $tokens
   OR sn.grammar_device           IN $tokens
   OR sn.grammar_transformation   IN $tokens
   OR sn.grammar_object           IN $tokens
   OR sn.grammar_geometry         IN $tokens
   OR sn.grammar_binary_operator  IN $tokens
RETURN sn
LIMIT $k
```

This is the meaningful difference between "keyword grep on description"
(which the keyword stream already does) and **"keyword match against
parsed grammar tokens"** which is the user's headline ask.  Multi-word
free-text queries now exercise the grammar stream; SN-shaped queries
exercise the per-segment streams (see §5.2).

### 5.2 Streams (per-segment grammar streams + vector + keyword)

Each stream is an independently-bounded helper that returns
`list[StreamHit]`:

```python
class StreamHit(BaseModel):
    sn_id:   str
    rank:    int           # 1-based within this stream
    raw:     float | None  # native score (vector cosine, count, …)
```

**Per-segment streams replace weighted-count fusion (F2).**  Instead of
one grammar stream that orders by a weighted hit-count (where the
weights collapse to tie-breakers under RRF and are lost to alphabetical
tie-breaks in a small corpus), we submit **one stream per parsed
segment** to the fusion layer.  RRF then naturally weighs SNs that
match multiple segments higher (more streams ⇒ more RRF mass), and an
SN that matches the high-signal `physical_base` gets the same RRF mass
per stream as any other — but matching `physical_base` typically
co-occurs with matching `subject` for siblings, so siblings naturally
rise to the top via accumulated mass across multiple streams.  This is
the standard RRF idiom.

| Stream family | Helper | Source | k_default |
|---|---|---|---|
| **Vector** | `_vector_stream(gc, query, k)` | `db.index.vector.queryNodes('standard_name_desc_embedding', …)` (existing) | 20 |
| **Per-segment grammar** (one stream per non-null parsed segment) | `_segment_stream(gc, segment, value, k)` | `MATCH (sn:StandardName) WHERE sn.grammar_<segment> = $value RETURN sn` | 20 |
| **Free-text grammar** (only when parsed = {}) | `_freetext_grammar_stream(gc, tokens, k)` | the IN-any-column query in §5.1 | 20 |
| **Keyword** | `_keyword_stream(gc, query, k)` | `WHERE toLower(sn.id) CONTAINS $q OR toLower(sn.description) CONTAINS $q` (existing logic) | 20 |

Stream selection rule:

```
parsed = parse_query_segments(query)         # may be {}
streams = {"vector": _vector_stream(...), "keyword": _keyword_stream(...)}
if parsed:
    for segment, value in parsed.items():
        streams[f"grammar:{segment}"] = _segment_stream(gc, segment, value, k)
else:
    tokens = tokenise_query(query)
    if tokens:
        streams["grammar:freetext"] = _freetext_grammar_stream(gc, tokens, k)
```

**Per-segment Cypher** (`_segment_stream`):

```cypher
MATCH (sn:StandardName)
WHERE sn.grammar_<segment> = $value
RETURN sn.id AS sn_id
ORDER BY sn.id ASC
LIMIT $k
```

Cheap, deterministic, single-property predicate (indexable).  The
`<segment>` interpolation is built from a hard-coded `Literal[...]`
allow-list — never user-supplied.

**Why this delivers the user's headline goal.**  For a query
`"electron temperature"` (free-text):

1. `parse_query_segments` → `{}`.
2. `tokenise_query` → `["electron", "temperature"]`.
3. `grammar:freetext` stream finds every SN whose
   `grammar_subject == "electron"` OR `grammar_physical_base == "temperature"`
   etc.  All `electron_*_temperature` siblings appear with rank 1..N.
4. Vector stream surfaces semantically-similar names (e.g. `electron_pressure`).
5. Keyword stream catches description-text matches.
6. RRF fuses all streams — siblings sharing both tokens accumulate mass
   from both the freetext grammar match *and* the vector match,
   landing them above non-grammar-matched semantic neighbours.

For an SN-shaped query `"parallel_current_density_weight"`:

1. `parse_query_segments` → e.g. `{"physical_base": "current_density",
   "component": "parallel", "secondary_base": "weight"}`.
2. Three separate per-segment streams submitted: `grammar:physical_base`,
   `grammar:component`, `grammar:secondary_base`.
3. An SN sharing all three (a true sibling) accumulates RRF mass from
   all three streams plus vector and keyword — landing it at the top
   without any weight tuning.

**Bound on empty-segments case (F7).**  When `parsed == {}` AND
`tokenise_query(query) == []` (e.g. caller passes `""` or pure
punctuation), the grammar stream returns `[]` immediately without a
graph round-trip.  Search degrades to vector + keyword as before; this
is the documented fallback.

**Concurrent execution (F7).**  All streams are dispatched concurrently
via `asyncio.gather` against an async Cypher wrapper (`gc.aquery`) when
available, otherwise via `concurrent.futures.ThreadPoolExecutor` against
the sync `gc.query`.  Per-stream timeout is 2 s; on timeout the stream
contributes `[]` and search continues with the streams that returned.
The 22-SN dev graph completes all streams in well under 50 ms; this
matters at corpus scale.

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

**Worked unit-test fixture (used in §9).**  Query
`"parallel_current_density_weight"` parses to three segments, so the
streams submitted are: `vector`, `grammar:physical_base`,
`grammar:component`, `grammar:secondary_base`, `keyword` (5 streams).

| sn_id   | vector | g:phys_base | g:component | g:secondary_base | keyword | streams matched | RRF score (sum 1/(60+rank)) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|---|
| `parallel_current_density_weight` (true sibling) | 1 | 1 | 1 | 1 | 1 | 5 | 5/61 ≈ 0.0820 |
| `current_density` (shares phys_base only)        | 5 | 2 | — | — | 3 | 3 | 1/65+1/62+1/63 ≈ 0.0476 |
| `electron_pressure` (semantic neighbour)         | 2 | — | — | — | — | 1 | 1/62 ≈ 0.0161 |
| `parallel_velocity` (shares component only)      | — | — | 2 | — | — | 1 | 1/62 ≈ 0.0161 |

Order: true sibling > phys_base sharer > semantic/component-only.  The
test asserts the order and the per-hit `stream_ranks` provenance.

The §5.3 RRF math is unchanged; what changes is that **`physical_base`
no longer needs an explicit weight** — it earns its prominence by
being a distinct stream, plus correlating with other streams (true
siblings hit multiple per-segment streams).

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

The MCP-tool string formatter renders `stream_ranks` as
`[V:1 Gphys:1 Gcomp:1 Gsec:1 K:1]` inline next to each hit (or short
forms like `[V:- Gphys:2 K:3]` when streams miss) — short, scannable,
and makes the per-segment fusion behaviour visible to the operator.

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
    include_fallback: bool = False,
) -> list[BaseGroup]:
    """Return SN groups keyed by (physical_base, subject).

    Used by catalog audits to surface families and to spot orphans
    (groups of size 1 are likely candidates for renaming or for a
    new sibling).

    *include_fallback* (F6): when False (default), rows with
    ``sn.grammar_parse_fallback = true`` are returned in a separate
    ``unparseable`` bucket (not as singleton "orphans") to avoid
    parser-fallback strings masquerading as data.  See §4.2.2 for
    how the flag is set on write.
    """

class BaseGroup(BaseModel):
    base: str
    subject: str
    members: list[str]
    n: int
    is_fallback_bucket: bool = False  # True for the unparseable bucket
```

Cypher (two-bucket form):

```cypher
// Bucket 1 — parsed groups
MATCH (sn:StandardName)
WHERE ($pd IS NULL OR sn.physics_domain = $pd)
  AND sn.grammar_physical_base IS NOT NULL
  AND coalesce(sn.grammar_parse_fallback, false) = false
WITH sn.grammar_physical_base AS base,
     coalesce(sn.grammar_subject, '') AS subject,
     collect(sn.id) AS members
WHERE size(members) >= $min_group_size
RETURN base, subject, members, size(members) AS n, false AS is_fallback_bucket
ORDER BY n DESC, base ASC, subject ASC

// Bucket 2 — fallback rows (returned only when include_fallback OR as
// a separate pass invoked by the caller)
MATCH (sn:StandardName)
WHERE coalesce(sn.grammar_parse_fallback, false) = true
RETURN '<unparseable>' AS base, '' AS subject, collect(sn.id) AS members,
       count(sn) AS n, true AS is_fallback_bucket
```

### 5.7 Worked end-to-end example

Two query shapes — both must work.

**Shape 1: SN-shaped query** `parallel_current_density_weight`:

1. **Parse**: `physical_base='current_density'`, `component='parallel'`,
   `secondary_base='weight'` (assuming parser supports the suffix).
2. **Streams submitted**: `vector`, `grammar:physical_base`,
   `grammar:component`, `grammar:secondary_base`, `keyword` (5 streams).
3. Each per-segment stream is a single indexed `=` predicate on a
   column property.
4. **RRF fuse** — see the worked fixture in §5.3.

A pre-existing SN
`equilibrium_reconstruction_parallel_current_density_weight` that did
not embed well (under-described in `description`) but shares
`physical_base`, `component`, *and* `secondary_base` accumulates RRF
mass across three grammar streams plus keyword — landing it at the top
without any weight tuning.

**Shape 2: Free-text query** `"plasma stored thermal energy"` (the §11
A4 acceptance test):

1. **Parse**: `parse_query_segments(query) == {}` (multi-word).
2. **Tokenise** (F1): `["plasma", "stored", "thermal", "energy"]`.
3. **Streams submitted**: `vector`, `grammar:freetext`, `keyword`.
4. The `grammar:freetext` stream finds every SN whose
   `grammar_physical_base ∈ tokens` OR `grammar_subject ∈ tokens` …
   — matches on `physical_base='thermal_energy'` /
   `physical_base='stored_energy'` / etc., plus any SN whose
   `subject='plasma'`.
5. **RRF fuse** — multi-word free-text queries now exercise grammar
   matching, not just description-substring grep.

Pre-plan-40 v2, the F1 critique held: the parser-only path gated on
`re.fullmatch(r"[a-z][a-z0-9_]+", …)` would have bypassed grammar
entirely for shape 2.  Post-v2, both shapes reach the grammar layer.

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
| **1. Writer fix + self-backfill** | §4 — rename + rewrite `_write_segment_edges` → `_write_grammar_decomposition` (with `grammar_parse_fallback` flag, F6); replace inline column SET in `catalog_import.py`; remove `grammar_fields=` propagation from `compose_worker`; add the deploy hook that calls the writer over all existing SN ids (F3). | Unit: parser → column round-trip, idempotency, parser-narrowing clears columns (F4), fallback flag set; integration: deploy hook leaves zero active-stage SNs with NULL `grammar_physical_base` after parse succeeded. |
| **2. Read-side** | §5 — `parse_query_segments`, `tokenise_query`, `_segment_stream`, `_freetext_grammar_stream` (F1), `_vector_stream`, `_keyword_stream`, concurrent dispatch (F7), `rrf_fuse`, refactor `_search_standard_names`; add `siblings_by_segment` and `group_by_base` (with fallback bucket, F6) Python helpers; add `mode={"hybrid","vector"}` kwarg to `search_similar_names` (F5). | Unit: RRF math, per-segment-stream natural ranking (F2), tokenise edges, mode kwarg behaviour; integration: free-text and SN-shaped queries both reach grammar layer; siblings surface across vector misses. |
| **3. MCP exposure** | §7 — verify registration truth-table; add `siblings_by_segment` MCP tool; `with_grammar` / `with_neighbours` kwargs on `fetch_standard_names`. | Tool-availability test (asserts the four SN tools register under non-dd-only mode); end-to-end manual smoke against the dev `imas-codex` MCP server. |

Each phase commits and pushes independently.  Phase 2 hard-depends on
Phase 1 (the grammar stream is column-based; columns must be populated
first).  Phase 3 has only soft dependencies.

---

## 9. Test strategy (per AGENTS.md "Schema-Driven Testing")

**Discipline reminder:** AGENTS.md forbids adding LinkML schema
declarations *just to make tests green*.  The column slots used here
already exist in the schema (§4.1).  The one schema addition —
`grammar_parse_fallback` (boolean) — stores legitimate provenance
(F6); it is **not** added to make a test pass.  The fallback test in
§9.1 asserts the boolean's behaviour; it does not depend on the slot
declaration.

### 9.1 Phase 1 — writer

| Test | File | Kind | Asserts |
|------|------|------|---------|
| `test_write_grammar_columns_populated` | `tests/standard_names/test_graph_ops.py` | unit (mocked `gc.query`) | every parser-emitted segment lands on the SET payload |
| `test_open_vocab_physical_base_kept` | same | unit | `physical_base="major_radius"` (no GrammarToken) survives to `sn.grammar_physical_base` |
| `test_writer_idempotent` | same | unit | running twice with same input produces identical SET payloads |
| `test_writer_clears_segments_when_parser_narrows` (F4) | same | unit (live test graph or mocked) | seed an SN with `grammar_subject='electron'`; re-run with a parser fixture that yields no `subject`; assert `sn.grammar_subject IS NULL` after the second write — proves `SET prop = null` removes the property |
| `test_grammar_parse_fallback_set` (F6) | same | unit | seed `plasma_geometric_axis_vertical_centroid_position` (whole-string fallback); assert `sn.grammar_parse_fallback = true`; seed `electron_temperature` (clean parse); assert `sn.grammar_parse_fallback = false` |
| `test_phase1_self_backfill` (F3) | `tests/integration/test_compose_persistence.py` | integration | seed graph with 5 SNs that have no grammar columns (simulate legacy state); call the deploy hook; assert all 5 now have non-null `grammar_physical_base` |
| `test_grammar_columns_e2e_compose` | `tests/integration/test_compose_persistence.py` | integration (live test graph) | after `compose_batch` of 5 fixtures, `MATCH (sn) WHERE sn.grammar_physical_base IS NULL RETURN count(sn) → 0` |
| `test_no_grammar_fields_blob_on_new_writes` | same | integration | `sn.grammar_fields IS NULL` for all newly-persisted SNs |

### 9.2 Phase 2 — search

| Test | File | Kind | Asserts |
|------|------|------|---------|
| `test_parse_query_segments` | `tests/standard_names/test_search.py` | unit | natural-language queries → `{}`; SN-shaped queries → parsed dict |
| `test_tokenise_query` (F1) | same | unit | `"plasma stored thermal energy"` → `["plasma","stored","thermal","energy"]`; punctuation-only input → `[]` |
| `test_freetext_grammar_stream` (F1) | same | unit (mocked or fixture graph) | seed SNs with `grammar_subject='plasma'` and `grammar_physical_base='thermal_energy'`; multi-word query returns both via `grammar:freetext` |
| `test_rrf_fuse_worked_example` | same | unit | the §5.3 fixture produces exact ordering and scores within 1e-6 |
| `test_per_segment_streams_naturally_rank_siblings` (F2) | same | unit (mocked) | true sibling matching 3 segments outranks an SN matching 1 segment via accumulated RRF mass — no explicit weights |
| `test_siblings_by_segment` | same | unit | column-MATCH Cypher template renders for each allowed segment |
| `test_streams_run_concurrently` (F7) | same | unit (timed) | 3-stream dispatch returns within wall-time < sum of per-stream sleeps (proves concurrent execution); empty-tokens path makes 0 grammar round-trips |
| `test_search_similar_names_mode_kwarg` (F5) | same | unit | `mode="vector"` returns vector-only results; `mode="hybrid"` returns RRF-fused results; default is `"hybrid"` |
| `test_hybrid_search_surfaces_siblings` | `tests/integration/test_search.py` | integration | seeded graph: 5 SNs share `physical_base='temperature'`, only 1 has a strong vector hit; hybrid returns all 5 |
| `test_hybrid_search_freetext_query` (F1, A4-companion) | same | integration | seeded graph; query `"plasma stored thermal energy"` returns ≥ 1 SN via the `grammar:freetext` stream that vector + keyword would not have surfaced |

### 9.3 Phase 3 — MCP

| Test | File | Kind | Asserts |
|------|------|------|---------|
| `test_sn_tools_registered_when_include_standard_names` | `tests/llm/test_server_registration.py` | unit | `MCPServer(include_standard_names=True, dd_only=False)` exposes `search_standard_names`, `fetch_standard_names`, `list_standard_names`, `siblings_by_segment` |
| `test_sn_tools_hidden_under_dd_only` | same | unit | `dd_only=True` (or `include_standard_names=False`) ⇒ none registered |
| `test_fetch_with_neighbours` | `tests/llm/test_sn_tools.py` | integration | `_fetch_standard_names("electron_temperature", with_neighbours=True)` returns sibling counts derived from columns |

### 9.4 Schema-compliance

`tests/graph/test_schema_compliance.py` already asserts that every
property declared in the LinkML schema for `StandardName` is either
`None` or a string on persisted nodes.  After Phase 1, an additional
assertion is added (in the existing test module, not as a schema
declaration):

> For each persisted SN with `name_stage IN ('drafted', 'accepted',
> 'reviewed', 'superseded')`, the test re-parses `sn.id` with
> `imas_standard_names.grammar.parse_standard_name` *inside the test
> body*, then asserts that for every segment where the parser yields a
> non-None value, the corresponding `sn.grammar_<segment>` column is
> equal to that value.

**F8 clarification:** the predicate "parser emits a value" is
established by **calling the parser inside the assertion**, not by
reading a schema declaration.  No schema lookup is used as oracle —
this respects the AGENTS.md "Schema-Driven Testing" ban on schema-as-
test-crutch.  The parser is the source of truth; the schema only
declares storage shape.

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

1. **A1.** `MATCH (sn:StandardName) WHERE sn.grammar_physical_base IS NULL RETURN count(sn)` → `0` for SNs whose parser emitted a `physical_base`.  **F3 deploy gate:** the Phase 1 deploy script (§4.4) self-backfills via `_write_grammar_decomposition(gc, all_sn_ids)`; if any active-stage SN remains NULL after that, the deploy fails loudly.
2. **A2.** `MATCH (sn:StandardName) WHERE sn.grammar_fields IS NOT NULL RETURN count(sn)` → `0` (new writes do not emit the legacy blob).
3. **A3.** Calling `siblings_by_segment(value="temperature", segment="physical_base")` returns ≥ 2 rows (assuming the seed has them).
4. **A4 (F1).** **Both** SN-shaped and free-text queries reach the grammar layer:
    - SN-shaped: `_search_standard_names("parallel_current_density_weight")` produces hits whose `stream_ranks` include at least one `grammar:<segment>` key.
    - Free-text: `_search_standard_names("plasma stored thermal energy")` produces hits whose `stream_ranks` include `grammar:freetext`, and at least one returned SN was matched **only** via the grammar stream (i.e. absent from a vector-only top-20 over the same query).
5. **A5.** `fetch_standard_names("electron_temperature", with_neighbours=True)` returns a sibling count derived from the column lookup, not from the typed-edge fallback.
6. **A6.** `tests/graph/test_schema_compliance.py` and the new tests in §9 all pass.
7. **A7.** Running `_write_grammar_decomposition` twice on the same `name_ids` produces no graph diff; running it on an SN whose ISN-rotated parse narrows a segment **clears** the stale column (F4).
8. **A8.** Server-registration test asserts the four SN tools register under `dd_only=False` and none under `dd_only=True`.
9. **A9 (F6).** `group_by_base()` segregates parse-fallback rows into the `unparseable` bucket; a singleton `physical_base='plasma_geometric_axis_vertical_centroid_position'` does **not** appear as an "orphan" group.

---

## 12. Migration / clear policy

Per the user's policy (AGENTS.md "Reset and Clear Semantics" + the
direct guidance: *"we have a non-backward compatible policy to reduce
code clutter"*):

- **Self-backfill on deploy (F3).**  No bespoke backfill script.
  Instead, the Phase 1 deploy hook calls
  `_write_grammar_decomposition(gc, all_sn_ids)` once.  Because the
  writer is idempotent and lossless, this transforms "backfill" into
  "rerun the canonical writer" — same code path, same guarantees.
  See §4.4 for the snippet.  This removes the foot-gun where a
  forgotten `sn clear` would silently degrade the new search features
  on the existing 22 SNs.
- **`sn clear` continues to work** (it deletes all SNs anyway; no
  change required).  Operators may still run `sn clear` if they
  prefer; the deploy hook is a no-op against an empty graph.
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

- `search_similar_names(query, k, *, gc=None, mode="hybrid")` gains an
  explicit **mode kwarg (F5)**:
  - `mode="hybrid"` (default) — RRF-fused vector + per-segment grammar +
    keyword.  This is what plan 39's `search_existing_names` runner
    receives.
  - `mode="vector"` — strict vector nearest-neighbour (current
    behaviour).  Reserved for **collision-avoidance call sites** in
    `compose_worker` that pre-write a candidate against the embedding
    index and need stable, unfused semantics.  No grammar bias, no
    description-substring noise.
  Default `"hybrid"` gives plan 39 the win for free; `"vector"` is the
  escape hatch for any caller whose mental model is "this is the
  vector helper".
- Plan-39 `fn_id`-to-mode mapping (documented here for reference;
  plan 39's catalog table is the authoritative spec):
  - `search_existing_names` → `mode="hybrid"` (the runner the proposer
    LLM aims its query at — wants grammar awareness).
  - Compose-worker pre-write collision check (not in plan 39's
    catalog; lives in `compose_worker`) → `mode="vector"`.
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

1. **Per-segment weighting under RRF**: v2 dropped the explicit 3/2/1
   weights (F2) in favour of submitting one stream per parsed segment,
   so RRF mass accumulates naturally for SNs matching multiple
   segments.  If telemetry shows that `physical_base`-only matches are
   still under-ranked relative to vector neighbours, a future
   refinement may re-introduce per-segment **stream multipliers**
   (e.g. count `physical_base` mass twice in RRF) — but only with
   measurement to back the choice.
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
5. **Stopword filtering in `tokenise_query` (F1)**: v2 deliberately
   does **not** filter stopwords — the grammar columns themselves are
   the filter (a token like `"of"` finds no column match).  If
   telemetry shows pathological IN-column scans on common stopwords,
   add a tiny stopword set (`{"of", "the", "and", "a", "in", "on"}`)
   in a follow-up; defer until measured.
6. **Nice-to-have items adopted in v2 vs deferred**: F4, F6, F7, F8,
   F9 were all adopted (no nice-to-haves skipped).  No deferrals to
   note here.

---

*End of plan 40 v2.*
