# Plan 40 — Standard-Name Search & Fetch Facility (v3.1)

> **Status:** v3.1 — incorporates `include_standard_names=True` default flip from main commit `33514f2a`. No structural rewrite vs. v3.
> **Supersedes:** v3 (initial draft, SHA `9668afeb`, 765 lines), v2 (SHA `3d412c77`, 1280 lines), v1 (SHA `3048413e`, 1030 lines).
> **Sibling plan:** [39 — Structured-Fanout Standard-Name Discovery](39-structured-fanout.md).
> **Owner boundary:** plan 39 owns the **dispatcher / discovery loop**; plan 40 owns the **search & fetch internals** (writer fix, retrieval, MCP palette, naming, backing-function unification).
>
> ### What changed v3 → v3.1
>
> - §1, §3.4, §7.1, §8.3, §9.7, §10, §11 (A10 demoted to verified precondition; A14 added), §12, §14, §16, §17 updated to match landed `33514f2a` defaults.
> - **No CLI flag rename.** v3.0 proposed `--standard-names`; v3.1 keeps the existing `--include-standard-names/--no-include-standard-names` (default `True`) which already landed.
> - **Auto-detect deferred.** v3.0 made `_detect_has_standard_names()` a Phase 3 deliverable; v3.1 demotes it to optional polish (§16 Q1). Phase 3's primary deliverable becomes the 3 new tools + renames + DD-only suppression gate.
> - **A14 added** for the new DD-only suppression criterion.
>
> ### What changed v2 → v3
>
> 1. **MCP tool palette mirrors the DD family.** v2 only spec'd `search` + `fetch`. v3 specifies a 7-tool palette (`search_standard_names`, `fetch_standard_names`, `find_related_standard_names`, `list_standard_names`, `list_grammar_vocabulary`, `check_standard_names`, `get_standard_name_summary`) modelled 1:1 on `imas-codex-search_dd_paths`, `fetch_dd_paths`, `find_related_dd_paths`, etc.
> 2. **Tiered grammar-segment policy.** v2's three-stream RRF treated all 12 segments equally, which lets `component=x` flood results for queries like `x_component_of_magnetic_field_at_outboard_midplane`. v3 introduces Tier 1 / Tier 2 / Tier 3 weighting (see §5.4).
> 3. **Naming alignment.** All `_sn` / `_sns` suffixes in public APIs become `_standard_names` (full audit table in §17). Backing helper renames, MCP tool names, and CLI subcommand stems all converge.
> 4. **Backing-function unification.** v3 mirrors `imas_codex/graph/dd_search.py`: pure functions in `imas_codex/standard_names/search.py` are consumed by **both** the discovery pipeline workers (plan 39) **and** the MCP server thin wrappers. v2 left the pipeline calling its own private helpers in parallel; v3 eliminates that drift.
> 5. **`include_standard_names` default flipped to `True` (landed in main, commit `33514f2a`).** v2 §7.3 wrongly attributed the missing palette to `dd_only`. The real cause was `include_standard_names: bool = False` at `imas_codex/llm/server.py:1768` and `--include-standard-names` (default false) in `imas_codex/cli/serve.py:57`. Both have been flipped to **default `True`**, with a `--no-include-standard-names` opt-out. Plan 40 v3 takes that flip as a **verified precondition** (§7.1) and adds the missing `not self.dd_only` suppression (FACILITY_TOOLS-style) to keep DD-only deployments lean.
> 6. **Phase 4 added** — pipeline call-site migration to the unified backing functions.
> 7. **Acceptance criteria** extended with **A10–A13** covering the four new concerns.

---

## 1. Executive Summary

The standard-name (SN) search-and-fetch surface today is a thin, partially wired layer in front of a writer that throws grammar information away. v2 of this plan diagnosed that and proposed a writer fix plus a three-stream RRF search. v3 keeps the v2 plumbing (writer, free-text grammar stream, parser-narrowing test, mode kwarg, RRF concurrency) and **expands the deliverable** to a full MCP palette + naming/unification cleanup so the SN tool family looks and behaves like the DD tool family that downstream agents already know how to drive.

**Concretely v3 commits to:**

- Fix `_write_segment_edges` so per-segment columns are populated from the ISN parser regardless of GrammarToken existence (Phase 1, unchanged from v2).
- Refactor the search engine into pure backing functions in `imas_codex/standard_names/search.py`, modelled on `imas_codex/graph/dd_search.py` (Phase 2).
- Expose **seven** MCP tools (Phase 3), all reading from those backing functions, with auto-detected registration so the palette appears whenever the graph holds StandardNames.
- Migrate the discovery pipeline (plan 39) call sites onto the same backing functions (Phase 4).

**Non-goals** (kept in plan 39 or out of scope):

- ISN vocabulary changes; parser changes; promotion-candidate workflow surfaces beyond the existing `list_promotion_candidates` tool.
- Rewriting `enrich_workers.py` or the broader fanout dispatcher (plan 39 owns that).
- New embedding models or graph schema for non-grammar SN concerns (units, COCOS, physics domains stay as-is).

---

## 2. Goals & Non-Goals

### 2.1 Goals

1. **Grammar-aware retrieval.** Queries that name physical concepts (`magnetic_field`, `electron_temperature`) must surface those SNs even when only structural segments (`component=x`) match the query string.
2. **Tier-respecting fusion.** Tier 1 segments (physical_base, subject, geometric_base) dominate results; Tier 3 segments (coordinate, geometry, region, device, object) never carry a result alone.
3. **MCP palette parity with DD.** Anyone who can drive the DD family (`search_dd_paths`, `fetch_dd_paths`, `find_related_dd_paths`, `check_dd_paths`, `list_dd_paths`, `get_ids_summary`, plus `search_dd_clusters`) can drive the SN family by analogy.
4. **One backing implementation.** Pipeline workers and MCP wrappers share the same code path so plan 39's discovery loop and MCP-driven inspection always agree on what "search" means.
5. **Public-API naming consistency.** Drop `_sn` / `_sns` from public symbols; spell out `_standard_names`. Internal helpers may keep short names where private.
6. **Default-on tooling.** When the graph holds StandardNames, the SN MCP tools register without an extra flag.

### 2.2 Non-Goals

- **No schema redesign.** Existing typed grammar edges (HAS_PHYSICAL_BASE, HAS_SUBJECT, HAS_COMPONENT, etc.) and per-segment columns stay; v3 only fixes population and adds one optional `grammar_parse_fallback` flag (carried over from v2 F6).
- **No new embedding model.** SN embeddings continue to use the same encoder as DD paths.
- **No vocabulary churn.** ISN's closed vocabulary is owned upstream; Codex remains a consumer.
- **No rewrite of `_register_tools`.** v3 only adds three new tools and toggles registration gating.

---

## 3. Empirical Findings (verified at HEAD `5e4d3bbe`)

> **Methodology:** counts/edges below were captured live via `imas-codex-repl` against the running graph. Code-line references were read from disk at the same SHA. Carried verbatim from v2; treat as the ground truth that motivates the rest of the plan.

### 3.1 Graph census

```
StandardName nodes:                 22
  with grammar_fields JSON blob:    10 (legacy, written by older catalog import)
  with any per-segment column set:   0   ← THE BUG
  with HAS_SEGMENT edges:            7  (11 edges total)

Typed grammar edges (closed-vocab):
  HAS_COORDINATE:        3
  HAS_GEOMETRIC_BASE:    2
  HAS_COMPONENT:         1
  HAS_PHYSICAL_BASE:     0   ← never populated; physical_base is open-vocab
  HAS_SUBJECT:           0
  HAS_TRANSFORMATION:    0
  (others):              0
Total typed segment edges:           6
```

So of the three storage mechanisms the writer is trying to feed (per-segment columns, typed edges, generic HAS_SEGMENT), the per-segment column path is **0%** populated and the typed-edge path covers **27%** of names. The legacy `grammar_fields` JSON blob still exists on 10/22 nodes but is not read by any production query.

### 3.2 Root cause

`_write_segment_edges` at `imas_codex/standard_names/graph_ops.py:1976-2110` does roughly:

```python
for segment, value in parsed.segments.items():
    OPTIONAL MATCH (gt:GrammarToken {segment: segment, token: value})
    WHERE gt IS NOT NULL
    MERGE (sn)-[:HAS_<SEGMENT>]->(gt)
```

Two distinct dropouts result:

1. **Open-vocab segments (`physical_base`).** The ISN closed vocabulary does not contain physical-base tokens (they are intentionally open-ended SI-style identifiers). The OPTIONAL MATCH never finds a GrammarToken, so the edge is never created. This explains zero HAS_PHYSICAL_BASE edges despite every parsed name having one.
2. **Per-segment columns are never written.** The writer creates edges (sometimes) but never sets `sn.physical_base`, `sn.subject`, `sn.component`, etc. The schema declares those slots but the code never emits the `SET` clauses. `imas_codex/standard_names/catalog_import.py:424-443` has an inline `SET` block that runs only for the ad-hoc `catalog import` flow and is bypassed by the main `write_standard_names` builder.

### 3.3 Parser is reliable

`from imas_standard_names.grammar import parse_standard_name` is deterministic, has a stable public API in ISN ≥ 0.7.0, and returns a `ParsedStandardName` with a fixed `.segments` dict. There is no reason to gate population on GrammarToken existence — we have authoritative segment values in-process at write time. Latest ISN at HEAD is `0.7.0rc35`.

### 3.4 Existing surfaces

- `imas_codex/standard_names/search.py` — 154 lines; pure helpers `search_similar_names`, `search_similar_sns_with_full_docs`. Vector-only today; no grammar awareness.
- `imas_codex/llm/sn_tools.py` — 603 lines; MCP entry points `_search_standard_names`, `_fetch_standard_names`, `_list_standard_names`, `_list_grammar_vocabulary`. Mutually exclusive search branches at `sn_tools.py:107-115`:

  ```python
  if segment_filters:
      rows = _segment_filter_search_sn(gc, query, k, segment_filters)
  elif embedding is not None:
      rows = _vector_search_sn(gc, embedding, k)
  else:
      rows = _keyword_search_sn(gc, query, k)
  ```

  Branches do not blend; grammar-aware retrieval only fires when the caller already knows which segment to filter on, which the LLM caller almost never does.

- `imas_codex/llm/server.py:3084-3260` — five SN tools registered behind `if self.include_standard_names:` gate (search, fetch, list_standard_names, list_grammar_vocabulary, list_promotion_candidates). v2 incorrectly stated four; the fifth is `list_promotion_candidates` at line 3247.
- `imas_codex/llm/server.py:1768` — `include_standard_names: bool = True` (flipped from `False` in commit `33514f2a`). Stock deployments now register the palette by default. v2 §7.3 wrongly blamed `dd_only`; the real gate has always been this field.
- `imas_codex/cli/serve.py:57-60` — `--include-standard-names/--no-include-standard-names` flag, **default true** (since `33514f2a`), help text `"Expose standard-name MCP tools (on by default; pass --no-include-standard-names to suppress)"`. Plan 40 keeps this surface unchanged.

### 3.5 Schema (LinkML) state

`imas_codex/schemas/standard_name.yaml` lines 803-872 already declare the typed grammar segment edge slots. Per-segment columns (`physical_base`, `subject`, `component`, etc.) are also declared as scalar slots on `StandardName`. v3 adds **one** new slot `grammar_parse_fallback: boolean` recording whether the parser succeeded at write time.

### 3.6 DD palette baseline

For the MCP-parity goal, v3 mirrors these tools (verbatim names from the DD family):

| DD tool                 | SN counterpart (v3)              | Backing function (target)                                |
|-------------------------|----------------------------------|----------------------------------------------------------|
| `search_dd_paths`       | `search_standard_names`          | `imas_codex.standard_names.search.search_standard_names` |
| `fetch_dd_paths`        | `fetch_standard_names`           | `imas_codex.standard_names.search.fetch_standard_names`  |
| `find_related_dd_paths` | `find_related_standard_names`    | `imas_codex.standard_names.search.find_related`          |
| `check_dd_paths`        | `check_standard_names`           | `imas_codex.standard_names.search.check_names`           |
| `list_dd_paths`         | `list_standard_names` *(exists)* | (existing `_list_standard_names`)                        |
| `get_ids_summary`       | `get_standard_name_summary`      | `imas_codex.standard_names.search.summarise_family`      |
| `search_dd_clusters`    | *(no SN equivalent in v3)*       | —                                                        |

The seventh SN tool, `list_grammar_vocabulary`, is SN-specific (no DD analogue) and stays as-is.

`find_related_dd_paths` returns a **bucketed** report, with sections per relationship type (Cluster Siblings, Unit Companions, Coordinate Companions, …). `find_related_standard_names` mirrors that bucketing structure (see §7.2.3).

### 3.7 Live-verified DD tool shapes (excerpt)

Captured via `imas-codex-find_related_dd_paths` against `core_profiles/profiles_1d/electrons/temperature` to lock the response shape v3 SN tools must mirror:

```
## Path Context: core_profiles/profiles_1d/electrons/temperature
Total cross-IDS connections: 6

### Cluster Siblings (3 paths)
**electron temperature**
  - `core_instant_changes/.../electrons/temperature` [quantity] — …
  - `edge_profiles/profiles_1d/electrons/temperature` [quantity] — …
  - `edge_profiles/ggd/electrons/temperature` [quantity] — …

### Unit Companions (3 paths)
**eV**
  - `camera_x_rays/t_e_magnetic_axis` [quantity · radiation_measurement_diagnostics] — …
```

`find_related_standard_names` will use the same `## ` / `### ` / bullet layout for buckets {Grammar Family, Unit Companions, COCOS Companions, Cluster Siblings, Predecessors, Successors, Refined-From, Source Paths/Signals}.

### 3.8 Naming inventory (audit ground-truth)

Exact occurrences of `_sn` / `_sns` in the surfaces v3 touches:

```
imas_codex/llm/sn_tools.py:110     _segment_filter_search_sn
imas_codex/llm/sn_tools.py:112     _vector_search_sn
imas_codex/llm/sn_tools.py:114     _keyword_search_sn
imas_codex/llm/sn_tools.py:137     def _segment_filter_search_sn
imas_codex/llm/sn_tools.py:159     docstring ref to _keyword_search_sn
imas_codex/llm/sn_tools.py:205     def _vector_search_sn
imas_codex/llm/sn_tools.py:224     def _keyword_search_sn

imas_codex/standard_names/search.py: search_similar_names, search_similar_sns_with_full_docs
imas_codex/standard_names/orphan_sweep.py:59,103     stale_token_sn  (private dict key — keep)
imas_codex/standard_names/enrich_workers.py:499      _fetch_nearby_sns  (private — rename to _fetch_nearby_standard_names)
imas_codex/standard_names/enrich_workers.py:646      max_per_sn  (private kwarg — keep)
imas_codex/standard_names/graph_ops.py:838           fetch_docs_review_feedback_for_sns  (public — rename)
```

§17 has the full mapping with deprecation aliases.

---

## 4. Phase 1 — Writer Fix (mostly unchanged from v2)

> **Owner:** plan 40. **Touches:** `imas_codex/standard_names/graph_ops.py`, `catalog_import.py`, `schemas/standard_name.yaml`, models regeneration.

### 4.1 Behaviour change

Replace the OPTIONAL-MATCH-driven `_write_segment_edges` with `_write_grammar_decomposition` that:

1. Calls `parse_standard_name(name)` (ISN parser, deterministic).
2. **Always** populates per-segment columns on the `StandardName` node (open vocab) — `SET sn.physical_base = $physical_base, sn.subject = $subject, …`. Missing segments → `null`, which clears any stale value (Phase 1 backfill story below).
3. **Conditionally** writes typed edges only for segments whose value resolves to a GrammarToken (closed vocab). No drop on miss; columns already captured the value.
4. Sets `sn.grammar_parse_fallback = true` when the parser raised, otherwise `false`. This is the new schema slot (F6 carried from v2).

### 4.2 Schema (LinkML) addition

```yaml
# imas_codex/schemas/standard_name.yaml
StandardName:
  attributes:
    grammar_parse_fallback:
      range: boolean
      required: false
      description: |
        True iff the ISN grammar parser raised when writing this name.
        When true, per-segment columns and typed grammar edges are absent.
```

Re-run `uv run build-models --force` so the regenerated `models.py` carries the new field. Do **not** stage `models.py`.

### 4.3 Idempotence + Phase 1 self-backfill (F3 from v2)

The new writer is idempotent: re-running it on existing nodes overwrites columns and refreshes edges. On Phase 1 deploy we run a one-shot driver that re-enters the writer for every existing StandardName; the writer's `SET sn.physical_base = $value` semantics (with `value = null` for missing segments) will both populate the 22 nodes that lack columns and clear stale columns on any name whose grammar has narrowed since last write. No separate backfill script is needed.

### 4.4 Removed inline writer

Delete the inline `SET` block at `imas_codex/standard_names/catalog_import.py:424-443`; it is subsumed by `_write_grammar_decomposition`. Catalog import now calls the same writer everything else does.

### 4.5 Tests (writer-scope)

- `tests/standard_names/test_grammar_writer.py::test_columns_always_set_when_parser_succeeds`
- `…::test_columns_cleared_when_parser_narrowed` — write a name with `component=x`, then re-write the same name with a parsed form lacking `component`, assert column is `null`. (F4 from v2.)
- `…::test_grammar_parse_fallback_recorded_on_parser_error`
- `…::test_open_vocab_physical_base_populates_column_with_no_edge`
- `tests/graph/test_schema_compliance.py` — re-parses the LinkML schema inside the test body so it picks up the new slot without manual fixture edits (F8 from v2).

---

## 5. Phase 2 — Search Redesign

> **Owner:** plan 40. **Touches:** `imas_codex/standard_names/search.py`, new module `imas_codex/standard_names/grammar_query.py`, `imas_codex/llm/sn_tools.py`. Pure functions only — no MCP changes here.

### 5.1 Backing-function layout (mirrors `dd_search.py`)

```text
imas_codex/standard_names/search.py
├── search_standard_names(query, *, k, mode, include_grammar, segment_filters, gc) -> SearchResult
├── fetch_standard_names(names, *, include_grammar, include_neighbours, include_documentation, gc) -> FetchResult
├── find_related(name, *, relationship_types, max_results, gc) -> RelatedResult
├── check_names(names, *, gc) -> CheckResult
└── summarise_family(physical_base, *, gc) -> FamilySummary
```

These are the **single source of truth** for SN retrieval. MCP wrappers in `sn_tools.py` and pipeline workers in `standard_names/enrich_workers.py` both import from here. No private duplicates.

### 5.2 Tokenisation (carried from v2 F1)

```python
# imas_codex/standard_names/grammar_query.py
def tokenise_query(query: str) -> list[str]:
    """Lower-case, snake-split, drop ISN connector stopwords."""
```

Stopwords: `{"of", "at", "from", "in", "to", "for", "the", "a", "an"}`. These connectors carry no segment information and only inflate keyword recall.

### 5.3 Three-stream RRF (carried from v2 F2, refined)

For free-text queries the engine fans out three concurrent streams (asyncio.gather over a ThreadPoolExecutor — F7 from v2):

1. **Vector stream.** Cosine similarity against `sn.embedding`. RRF rank input: similarity-sorted list, top `k_candidates = 5*k`.
2. **Keyword stream.** SQLite-style FTS over `description` + `documentation` + `name`. RRF rank input: BM25-sorted top `k_candidates`.
3. **Grammar stream.** New in v2; **tier-aware** in v3. Per §5.4.

Final ranking: standard RRF with `k_rrf = 60`, summed across streams, ties broken by vector score.

### 5.4 Tiered grammar policy (NEW in v3)

The grammar stream is itself a fan-out over per-segment column matches. Token `t` from `tokenise_query(query)` matches a SN if `t IN [physical_base, subject, component, …]` for that SN. v2 treated all 12 segments equally; that lets a query like `x_component_of_magnetic_field_at_outboard_midplane` produce per-token streams in which `x → component=x` matches every x-component SN in the catalog (often hundreds) and dominates RRF aggregation.

v3 partitions segments into three tiers and applies tier-dependent RRF weights:

| Tier | Segments                                        | RRF weight | Eligibility                                                        |
|------|-------------------------------------------------|------------|--------------------------------------------------------------------|
| 1    | `physical_base`, `subject`, `geometric_base`    | 1.0        | Always contributes to fusion. May solely surface a result.         |
| 2    | `transformation`, `component`, `position`, `process` | 0.5   | Contributes only when at least one Tier-1 segment also hit for the same SN, OR when the candidate also appears in the vector or keyword stream. |
| 3    | `coordinate`, `geometry`, `region`, `device`, `object` | 0.25 | Modifier only. Never solely surfaces a result; adds a tie-break boost when fused with Tier-1 or vector/keyword evidence. |

**Mechanism in code:**

```python
def grammar_stream(tokens: list[str], gc) -> list[ScoredCandidate]:
    per_segment = {}
    for seg in TIER1 | TIER2 | TIER3:
        per_segment[seg] = gc.query(
            "MATCH (sn:StandardName) WHERE sn[$seg] IN $tokens RETURN sn.id AS id",
            seg=seg, tokens=tokens,
        )
    candidates = {}
    for seg, rows in per_segment.items():
        weight = TIER_WEIGHT[tier_of(seg)]
        for rank, row in enumerate(rows):
            candidates.setdefault(row["id"], []).append((seg, rank, weight))

    # Eligibility: drop Tier-2/3-only candidates whose SN does not also appear
    # in the vector OR keyword stream, AND has no Tier-1 segment hit.
    return _filter_by_tier_policy(candidates, vector_hits, keyword_hits)
```

Eligibility filter lives in `_filter_by_tier_policy`; it is the only place tiers couple to the other streams. The result is fused into the global RRF as a single grammar-stream rank list.

### 5.4.1 Worked example

Query: `x_component_of_magnetic_field_at_outboard_midplane`

`tokenise_query` → `["x", "component", "of", "magnetic", "field", "at", "outboard", "midplane"]` → after stopword drop → `["x", "component", "magnetic", "field", "outboard", "midplane"]`.

Per-segment matches against the 22-name graph:

- `physical_base ∋ {magnetic_field}` → 1 candidate (Tier 1, weight 1.0).
- `component ∋ {x}` → ~200 candidates in a fully-populated catalog (Tier 2, weight 0.5).
- `position ∋ {outboard_midplane}` → 3 candidates (Tier 2, weight 0.5).
- All other tokens → no segment matches.

**v2 behaviour (no tiers):** `component=x` floods the grammar stream with ~200 ranks, RRF aggregation pushes the right physical-base hit out of the top-`k`.

**v3 behaviour (tiered):** the magnetic_field hit gets full Tier-1 weight; the 200 `component=x` hits are kept but filtered to those that *also* appear in the vector or keyword stream OR also match a Tier-1 segment for this query. The `outboard_midplane` Tier-2 hits survive (3 total, manageable). The right SN — `x_component_of_magnetic_field_at_outboard_midplane` — wins because it scores in **all** of: vector, keyword, Tier-1 (physical_base=magnetic_field), Tier-2 (component=x, position=outboard_midplane).

### 5.5 `mode` kwarg (carried from v2 F5)

`search_standard_names(..., mode: Literal["hybrid", "vector"] = "hybrid")`. `vector` skips the keyword and grammar streams (useful for plan-39 worker code paths that already know they want pure semantic similarity). `hybrid` is default.

### 5.6 Siblings + group-by-base modes

`group_by_base: bool = False` returns results bucketed by `physical_base`, with two buckets: **typed** (rows where the writer captured a physical_base) and **fallback** (rows where `grammar_parse_fallback = true`). Carried from v2 F6.

`include_siblings: bool = False` — for each result row, also fetch every other SN sharing the same `physical_base` value. Useful when an LLM is exploring a family.

### 5.7 RRF concurrency (F7 from v2)

```python
async def _run_streams(query, embedding, gc):
    loop = asyncio.get_running_loop()
    return await asyncio.gather(
        loop.run_in_executor(POOL, vector_stream, embedding, gc),
        loop.run_in_executor(POOL, keyword_stream, query, gc),
        loop.run_in_executor(POOL, grammar_stream, tokenise_query(query), gc),
    )
```

Single shared `ThreadPoolExecutor(max_workers=4)` at module level. No new infrastructure.

---

## 6. Phase 2 (cont.) — Fetch + Scoping Kwargs

`fetch_standard_names(names, *, include_grammar, include_neighbours, include_documentation, return_fields, gc)`:

| kwarg                  | type        | default | effect                                                                  |
|------------------------|-------------|---------|-------------------------------------------------------------------------|
| `include_grammar`      | bool        | `True`  | Project per-segment columns + typed-edge tokens into the result rows.   |
| `include_neighbours`   | bool        | `False` | One-hop expand: predecessors, successors, refined-from, error-of links. |
| `include_documentation`| bool        | `True`  | Project `description` + `documentation` (large text). Off → light fetch.|
| `return_fields`        | list[str] \| None | None  | Whitelist of node properties to return. None → schema default set.    |

These are the same shape as the DD `fetch_dd_paths` `include_*` kwargs, mapped onto SN concerns. The MCP wrapper (§7.2.2) exposes them verbatim.

---

## 7. Phase 3 — MCP Exposure

> **Owner:** plan 40. **Touches:** `imas_codex/llm/server.py`, `imas_codex/llm/sn_tools.py`, `imas_codex/cli/serve.py`.

### 7.1 SN palette default state (verified precondition)

**Verified at HEAD.** Commit `33514f2a` flipped both defaults:

- `imas_codex/llm/server.py:1768`: `include_standard_names: bool = True` (was `False`).
- `imas_codex/cli/serve.py:57-60`: `--include-standard-names/--no-include-standard-names`, `default=True`, help text `"Expose standard-name MCP tools (on by default; pass --no-include-standard-names to suppress)"`.
- `tests/llm/test_tool_schemas.py::TestIncludeStandardNames` and `tests/core/test_cli.py::test_serve_default_options`/`test_serve_read_only` assert the new default.

So a stock `imas-codex serve` deployment **already registers all five existing SN tools** (search, fetch, list, list_grammar_vocabulary, list_promotion_candidates). Plan 40 inherits this and does **not** rename the flag.

**Remaining gap (this plan owns):** the SN block at `server.py:3084` gates only on `if self.include_standard_names:` — it does **not** check `self.dd_only`. DD-only deployments (graph with no facilities, see `_detect_dd_only` at `server.py:1869-1907`) therefore still register SN tools, which is wasteful when the operator clearly wants a stripped-down DD surface.

**v3 design (minimal):**

1. **Add `dd_only` suppression.** Change the Phase 3 gate to `if self.include_standard_names and not self.dd_only:`. Mirrors the FACILITY_TOOLS pattern at `server.py:1774-1783` and the `if not self.dd_only:` guards at `server.py:1979`, `2200`, `2321`, `3055`.
2. **Background re-evaluation.** When `_background_detect_dd_only` flips `self.dd_only` to `True` after startup, log a warning that the SN tools are already registered (re-registration is not supported by FastMCP); operators should restart with the corrected flag. This matches existing FACILITY_TOOLS behaviour.
3. **No flag rename.** `--include-standard-names/--no-include-standard-names` is the canonical name; the alternate `--standard-names` rename proposed in v3.0 is dropped.
4. **Optional polish (deferred).** A future change can add `_detect_has_standard_names()` mirroring `_detect_dd_only()` so the flag becomes truly auto. Out of scope for plan 40 — the default-on flag already covers the intended UX.

### 7.2 Seven-tool palette specification

All seven tools live under the existing SN registration block in `server.py`. Tools 1, 2, 4, 5 already exist (rename/extend); tools 3, 6, 7 are new.

#### 7.2.1 `search_standard_names`

```python
def search_standard_names(
    query: str,
    k: int = 20,
    mode: Literal["hybrid", "vector"] = "hybrid",
    segment_filters: dict[str, str] | None = None,
    include_grammar: bool = True,
    group_by_base: bool = False,
    include_siblings: bool = False,
    kind: str | None = None,
    pipeline_status: str | None = None,
) -> str:
    """Hybrid (vector + tiered grammar + keyword + RRF) search over StandardNames."""
```

Wraps `standard_names.search.search_standard_names`. Returns markdown via `_format_search_report`. Tier policy from §5.4 is applied internally; callers do not pick tiers.

#### 7.2.2 `fetch_standard_names`

```python
def fetch_standard_names(
    names: str,                           # space- or comma-separated
    include_grammar: bool = True,
    include_neighbours: bool = False,
    include_documentation: bool = True,
    return_fields: str | None = None,     # comma-separated whitelist
) -> str:
    """Fetch full entries for known standard names."""
```

Same call-shape as `fetch_dd_paths`. Existing `_fetch_standard_names` is extended with the new include kwargs.

#### 7.2.3 `find_related_standard_names` *(NEW)*

```python
def find_related_standard_names(
    name: str,
    relationship_types: str = "all",
    max_results: int = 20,
) -> str:
    """Find StandardNames related to a given name across multiple relationship signals.

    Buckets:
      - Grammar Family       (same physical_base)
      - Subject Companions   (same subject)
      - Unit Companions      (HAS_UNIT)
      - COCOS Companions     (cocos_transformation_type)
      - Cluster Siblings     (IN_CLUSTER)
      - Predecessors         (HAS_PREDECESSOR)
      - Successors           (HAS_SUCCESSOR)
      - Refined-From         (REFINED_FROM)
      - Source Paths         (FROM_DD_PATH via StandardNameSource)
      - Source Signals       (FROM_SIGNAL via StandardNameSource)
    """
```

Output mirrors `find_related_dd_paths`'s `## … / ### Bucket / bullets` shape (§3.7). `relationship_types="all"|"grammar"|"unit"|"cocos"|"cluster"|"lineage"|"source"` selects which buckets render.

#### 7.2.4 `list_standard_names` *(rename + keep)*

Existing tool; rename verifies. No behaviour change beyond the naming-alignment audit.

#### 7.2.5 `list_grammar_vocabulary` *(keep + document)*

Existing tool. v3 adds clearer docstring covering the segment-filter dimension. No signature change.

#### 7.2.6 `check_standard_names` *(NEW)*

```python
def check_standard_names(names: str) -> str:
    """Validate that names exist; for unknown names, suggest closest matches.

    Mirrors check_dd_paths. Uses Levenshtein distance over the StandardName.id
    space, plus grammar-aware suggestion: if the input parses, prefer suggestions
    sharing physical_base.
    """
```

Returns markdown table with columns `name | exists | suggestion | reason`.

#### 7.2.7 `get_standard_name_summary` *(NEW)*

```python
def get_standard_name_summary(physical_base: str) -> str:
    """Family overview keyed on physical_base.

    Returns:
      - count of StandardNames sharing this physical_base
      - distinct values of each Tier-1/Tier-2 segment within the family
      - representative names (sample of 5)
      - distinct units, COCOS types, physics domains
    """
```

Mirrors `get_ids_summary` shape but for SN families.

### 7.3 Per-tool registration sketches

Each new tool registers under the tightened guard `if self.include_standard_names and not self.dd_only:` (see §8.3 step 1). Pattern follows existing SN block:

```python
if self.include_standard_names and not self.dd_only:
    @self.mcp.tool()
    def find_related_standard_names(name: str, relationship_types: str = "all",
                                    max_results: int = 20) -> str:
        from imas_codex.llm.sn_tools import _find_related_standard_names
        return _find_related_standard_names(
            name=name,
            relationship_types=relationship_types,
            max_results=max_results,
        )

    @self.mcp.tool()
    def check_standard_names(names: str) -> str:
        from imas_codex.llm.sn_tools import _check_standard_names
        return _check_standard_names(names=names)

    @self.mcp.tool()
    def get_standard_name_summary(physical_base: str) -> str:
        from imas_codex.llm.sn_tools import _get_standard_name_summary
        return _get_standard_name_summary(physical_base=physical_base)
```

### 7.4 Backing-function unification mapping

| MCP tool                        | Thin wrapper in `sn_tools.py`     | Pure backing function in `standard_names/search.py` | Pipeline call sites (plan 39)                        |
|---------------------------------|-----------------------------------|-----------------------------------------------------|------------------------------------------------------|
| `search_standard_names`         | `_search_standard_names`          | `search_standard_names`                             | `enrich_workers._dispatch_similarity_lookup`         |
| `fetch_standard_names`          | `_fetch_standard_names`           | `fetch_standard_names`                              | `enrich_workers._fetch_target_for_review`            |
| `find_related_standard_names`   | `_find_related_standard_names`*   | `find_related`*                                     | `enrich_workers._fetch_nearby_standard_names` (renamed) |
| `list_standard_names`           | `_list_standard_names`            | (CRUD; bypasses search.py)                          | `vocab.list_existing`                                |
| `list_grammar_vocabulary`       | `_list_grammar_vocabulary`        | (CRUD; bypasses search.py)                          | `vocab.list_grammar_tokens`                          |
| `check_standard_names`*         | `_check_standard_names`*          | `check_names`*                                      | `enrich_workers._validate_predecessor_links`         |
| `get_standard_name_summary`*    | `_get_standard_name_summary`*     | `summarise_family`*                                 | (none today; available for plan-39 audits)           |

`*` = new in v3.

The discipline: **every cell in the "Pure backing function" column is the only place graph queries live.** Wrappers format. Pipeline imports the same functions. No private duplicates.

### 7.5 Naming-alignment audit table

See §17 for the full table. Highlights:

| Old (private/public)                 | New                                       | Deprecation                  |
|--------------------------------------|-------------------------------------------|------------------------------|
| `search_similar_names`               | `search_standard_names_vector`            | Module-private alias kept 1 release |
| `search_similar_sns_with_full_docs`  | `search_standard_names_with_documentation`| Module-private alias kept 1 release |
| `_segment_filter_search_sn`          | `_grammar_filter_search_standard_names`   | Internal — rename only       |
| `_vector_search_sn`                  | `_vector_search_standard_names`           | Internal — rename only       |
| `_keyword_search_sn`                 | `_keyword_search_standard_names`          | Internal — rename only       |
| `_fetch_nearby_sns`                  | `_fetch_nearby_standard_names`            | Internal — rename only       |
| `fetch_docs_review_feedback_for_sns` | `fetch_docs_review_feedback_for_standard_names` | Public — keep alias 1 release |

Private dict-key strings (`"stale_token_sn"`, `max_per_sn` kwarg) are retained: they are not API surfaces.

---

## 8. Phased Rollout

> **Phase 1 (writer fix)** unchanged from v2.

### 8.1 Phase 1 — Writer fix + self-backfill (1 PR)

1. Edit `imas_codex/schemas/standard_name.yaml`: add `grammar_parse_fallback`.
2. `uv run build-models --force`. Do not stage `models.py`.
3. Implement `_write_grammar_decomposition` in `graph_ops.py`; remove old `_write_segment_edges` body.
4. Delete inline `SET` block in `catalog_import.py:424-443`.
5. Add Phase-1 self-backfill driver `scripts/backfill_grammar_decomposition.py` calling `write_standard_names` for every existing SN id.
6. Tests per §4.5 + `tests/graph/test_schema_compliance.py` re-run.
7. Land + run backfill in deployed envs.

**Exit criterion:** all 22 (then N) StandardNames have non-null Tier-1 segment columns where the parser succeeds.

### 8.2 Phase 2 — Search backing functions (1 PR)

1. Create `imas_codex/standard_names/grammar_query.py` with `tokenise_query`, tier constants.
2. Refactor `imas_codex/standard_names/search.py` into the §5.1 layout (no MCP, no pipeline changes yet).
3. Implement vector / keyword / grammar streams + tier policy + RRF aggregation + `mode` kwarg.
4. Implement `find_related`, `check_names`, `summarise_family`.
5. Tests: §9 unit + integration.
6. **Wrappers in `sn_tools.py` switch to the new backing functions** but keep the same MCP-tool names; no new MCP tools yet.

**Exit criterion:** existing SN MCP tools still pass behavioural smoke, with grammar awareness now active.

### 8.3 Phase 3 — MCP palette (1 PR)

1. Tighten the registration gate at `server.py:3084` from `if self.include_standard_names:` to `if self.include_standard_names and not self.dd_only:` — adds DD-only suppression (FACILITY_TOOLS-style).
2. Register the three new tools (`find_related_standard_names`, `check_standard_names`, `get_standard_name_summary`).
3. Naming alignment renames per §7.5 + §17.
4. Tests: §9.7 (registration & dd_only suppression), §9.8 (find_related buckets), §9.9 (check suggestions), §9.10 (summary).
5. **No CLI flag rename.** `--include-standard-names/--no-include-standard-names` (default `True`) is already the canonical surface as of commit `33514f2a`.
6. *(Optional polish, may slip to a follow-up.)* Add `_detect_has_standard_names` mirroring `_detect_dd_only` so the default can become truly auto-detected. Not required for plan 40 acceptance.

**Exit criterion:** A10–A13 satisfied (§11).

### 8.4 Phase 4 — Pipeline call-site migration (1 PR) *(NEW in v3)*

1. Audit `imas_codex/standard_names/enrich_workers.py` and any other caller of the old private helpers (`_segment_filter_search_sn`, `_vector_search_sn`, `_keyword_search_sn`, `_fetch_nearby_sns`).
2. Replace with imports from `imas_codex.standard_names.search`.
3. Delete the old private helpers entirely (no aliases — they were never importable from outside `sn_tools.py`).
4. Re-run plan-39 dispatcher tests; ensure `mode="vector"` is wired through where pure semantic similarity is required.
5. Update plan 39 docs to point at the unified surface.

**Exit criterion:** `grep "_segment_filter_search_sn\|_vector_search_sn\|_keyword_search_sn\|_fetch_nearby_sns" imas_codex/` returns zero hits outside the deletion commits.

---

## 9. Tests

### 9.1 Writer tests — see §4.5.

### 9.2 Tokeniser

`tests/standard_names/test_grammar_query.py::test_tokenise_drops_stopwords` — `tokenise_query("x_component_of_magnetic_field")` returns `["x", "component", "magnetic", "field"]`.

### 9.3 Per-stream

- `…::test_vector_stream_returns_top_k_by_cosine`
- `…::test_keyword_stream_uses_bm25_ranking`
- `…::test_grammar_stream_returns_segment_hits` (no tier filter applied — pure stream).

### 9.4 Tiered policy *(NEW in v3)*

- `…::test_tier1_only_match_surfaces_alone` — query parses to physical_base=X, no other tokens; expect that SN in top-k.
- `…::test_tier3_only_match_does_not_surface_alone` — query parses to coordinate=z only; expect tier-3-only candidate filtered out unless vector or keyword stream also hit.
- `…::test_x_component_query_does_not_flood` — exact §5.4.1 example; expect `x_component_of_magnetic_field_at_outboard_midplane` ranked first, with all other `component=x` SNs not displacing the physical-base hit.

### 9.5 RRF

- `…::test_rrf_aggregation_summed_across_streams`
- `…::test_rrf_concurrency_does_not_change_results` — run streams sequentially and concurrently; assert identical ranks.

### 9.6 Mode kwarg

- `…::test_mode_vector_skips_keyword_and_grammar_streams`

### 9.7 Registration & DD-only suppression

**Already on main (verified at HEAD, do not duplicate):** `tests/llm/test_tool_schemas.py::TestIncludeStandardNames`:
- `test_sn_tools_present_by_default` — default `True` registers the SN block.
- `test_sn_tools_present_in_rw_server` — read-write server includes SN tools.
- `test_sn_tools_absent_when_opted_out` — `include_standard_names=False` suppresses.
- `test_sn_tools_present_when_flag_set` — explicit `True` registers.

`tests/core/test_cli.py::test_serve_default_options` and `test_serve_read_only` already assert the CLI default propagates as `include_standard_names=True`.

**New tests this plan adds** (`tests/llm/test_sn_tool_registration.py`):
- `test_sn_tools_absent_under_dd_only` — `dd_only=True, include_standard_names=True` → 0 SN tools registered (covers the gate tightening in §8.3 step 1).
- `test_sn_tools_present_when_dd_only_false_and_include_true` — explicit positive baseline mirroring the FACILITY_TOOLS pattern.
- *(Optional, only if §8.3 step 6 lands in this plan.)* `test_palette_auto_detected_against_sn_graph` — `_detect_has_standard_names()` returns True against a graph populated with `:StandardName`.

### 9.8 `find_related_standard_names` buckets *(NEW in v3)*

- `tests/llm/test_sn_find_related.py::test_returns_at_least_three_buckets_for_dense_name`
- `…::test_relationship_types_filter_drops_other_buckets`
- `…::test_grammar_family_bucket_uses_physical_base`

### 9.9 `check_standard_names` *(NEW in v3)*

- `tests/llm/test_sn_check.py::test_known_name_marked_exists`
- `…::test_unknown_name_levenshtein_suggestion`
- `…::test_grammar_aware_suggestion_prefers_same_physical_base`

### 9.10 `get_standard_name_summary` *(NEW in v3)*

- `tests/llm/test_sn_summary.py::test_returns_distinct_segment_values`
- `…::test_returns_unit_and_cocos_distincts`

### 9.11 Naming alignment *(NEW in v3)*

- `tests/standard_names/test_public_api_naming.py::test_no_sn_or_sns_suffix_in_public_symbols` — walk `imas_codex.standard_names`, `imas_codex.llm.sn_tools`; assert no exported name (i.e., no name without leading `_`) ends in `_sn` / `_sns`.

### 9.12 Schema compliance

Re-uses `tests/graph/test_schema_compliance.py` per §4.5; the test re-parses LinkML in-process so it picks up `grammar_parse_fallback`.

### 9.13 Pydantic attachment

`tests/standard_names/test_attachment_grammar_fields.py::test_grammar_fields_serialise_round_trip` — confirms `StandardNameAttachment.grammar_fields` accepts the new column dict (F9 from v2).

---

## 10. Configuration

No new YAML keys. Existing `imas_codex/schemas/standard_name.yaml` extended (§4.2). No CLI flag rename — `--include-standard-names/--no-include-standard-names` (default `True`) is already the canonical surface as of `33514f2a`.

---

## 11. Acceptance Criteria

- **A1.** All StandardName nodes in the graph have non-null Tier-1 segment columns iff the parser succeeds.
- **A2.** `_write_grammar_decomposition` is the **only** writer for grammar columns + edges; `catalog_import.py` no longer carries an inline SET block.
- **A3.** Free-text query `electron_temperature` returns the matching SN in top-3 even when the GrammarToken vocabulary lacks `electron_temperature` (open-vocab segment population).
- **A4.** `mode="vector"` skips keyword and grammar streams.
- **A5.** Concurrent and sequential stream execution yield identical ranks.
- **A6.** Re-writing a name whose parsed grammar narrowed clears stale columns (parser-narrowing test passes).
- **A7.** `grammar_parse_fallback = true` set on parser-error names; columns/edges absent.
- **A8.** Schema-compliance test passes against regenerated models without manual fixture edits.
- **A9.** `StandardNameAttachment.grammar_fields` round-trips the new column dict.
- **A10.** *(VERIFIED PRECONDITION, credit `33514f2a`.)* All 5 existing SN tools register by default in a stock `imas-codex serve` deployment; `tests/llm/test_tool_schemas.py::TestIncludeStandardNames` already enforces this on main. Plan 40 inherits and does not regress.
- **A11.** *(NEW)* Query `x_component_of_magnetic_field_at_outboard_midplane` ranks the matching SN first; no `component=x` flood pushes it out of the top-k (tested in a fixture catalog with ≥ 50 `component=x` decoy SNs).
- **A12.** *(NEW)* `find_related_standard_names("electron_temperature")` returns ≥ 3 distinct buckets in the markdown output.
- **A13.** *(NEW)* `grep -r "_sn\|_sns" imas_codex/standard_names/ imas_codex/llm/sn_tools.py` matches no public symbols (private dict-keys/kwargs may remain).
- **A14.** *(NEW)* Under `dd_only=True`, no SN tools are registered even with `include_standard_names=True` — covered by the new `test_sn_tools_absent_under_dd_only` test (§9.7).

---

## 12. Risks

- **Open-vocab explosion.** Populating `physical_base` columns from arbitrary parsed tokens means the column carries an open value space. Acceptable: it is exactly what `physical_base` is in ISN. Downstream queries that join on `physical_base` already use `=` (no need for GrammarToken existence).
- **Tier policy mis-tuned.** Tier weights {1.0, 0.5, 0.25} are seeded; observability over real query traffic may show mis-classification (e.g., `position` should perhaps be Tier 1). v3 lands the policy as a constant in `grammar_query.py`; revisiting weights is a follow-up plan, not a blocker.
- **DD-only deployments still see SN tools today.** Mitigated by the §8.3 step 1 gate tightening (`and not self.dd_only`); covered by A14.
- **Pipeline migration churn (Phase 4).** Plan 39 may have unmerged work that touches the same workers. Phase 4 lands last for that reason.

---

## 13. Telemetry

Existing `IMASCodexServer.tool_call_logger` covers MCP-side observability. Add per-stream timing log inside `search_standard_names`:

```python
logger.info("sn-search streams: vector=%.0fms keyword=%.0fms grammar=%.0fms n_total=%d",
            t_vec, t_kw, t_gr, len(results))
```

No new metrics infra.

---

## 14. Documentation

- `AGENTS.md` § "Standard Names" — update the MCP-tool subsection to list seven tools and link to §7.2 of this plan; note that registration is now default-on (since `33514f2a`) and document the `--no-include-standard-names` opt-out.
- README MCP tool table — extend with the three new tools; mention the default-on flag.
- `imas-codex serve --help` — already updated by `33514f2a` to `"on by default; pass --no-include-standard-names to suppress"`. No further change unless the optional auto-detect (§8.3 step 6) lands.

---

## 15. Plan 39 Boundary

Plan 39 owns the **structured-fanout dispatcher**: queue, claims, worker pool, rate limiting, ISN-version awareness. Plan 40 owns **what each fanout call actually does** when it hits the graph: tokenisation, tier policy, RRF, find-related buckets, fetch include kwargs.

Phase 4 (§8.4) is the explicit hand-off: plan-39 workers stop importing `_segment_filter_search_sn`/`_vector_search_sn`/`_keyword_search_sn` and start importing `search_standard_names` (mode="vector") plus `fetch_standard_names`. The dispatcher logic in plan 39 is unchanged.

---

## 16. Open Questions

1. **Auto-detect for SN registration (§8.3 step 6).** Should plan 40 also add `_detect_has_standard_names()` mirroring `_detect_dd_only()`, or defer? Default-on flag already covers UX; auto-detect is purely "DD-only graph shouldn't even need `--no-include-standard-names`". Recommend defer.
2. **Tier weights {1.0, 0.5, 0.25}.** Magic numbers seeded from the worked example; should we instead tune from a held-out query set? Defer — a separate plan once we have query traffic to learn from.
4. **`group_by_base` semantics across mode="vector"`.** Group-by-base is meaningful only when grammar is populated; for vector-only queries, `group_by_base=True` should error (or silently no-op). Suggest error with explicit message.
5. **`fetch` `return_fields` vs `include_*` precedence.** When both `return_fields` and `include_documentation` are set, which wins? Suggest `return_fields` is the final whitelist; `include_*` are convenience macros that expand into the whitelist.

---

## 17. Naming-Alignment Audit (full table)

| Path                                                | Old symbol                              | New symbol                                       | Visibility | Action                            |
|-----------------------------------------------------|-----------------------------------------|--------------------------------------------------|------------|-----------------------------------|
| `imas_codex/standard_names/search.py`               | `search_similar_names`                  | `search_standard_names_vector`                   | public     | rename + 1-release alias          |
| `imas_codex/standard_names/search.py`               | `search_similar_sns_with_full_docs`     | `search_standard_names_with_documentation`       | public     | rename + 1-release alias          |
| `imas_codex/standard_names/search.py`               | *(new)*                                 | `search_standard_names`                          | public     | new                               |
| `imas_codex/standard_names/search.py`               | *(new)*                                 | `fetch_standard_names`                           | public     | new                               |
| `imas_codex/standard_names/search.py`               | *(new)*                                 | `find_related`                                   | public     | new                               |
| `imas_codex/standard_names/search.py`               | *(new)*                                 | `check_names`                                    | public     | new                               |
| `imas_codex/standard_names/search.py`               | *(new)*                                 | `summarise_family`                               | public     | new                               |
| `imas_codex/standard_names/grammar_query.py`        | *(new module)*                          | `tokenise_query`, tier constants                 | public     | new                               |
| `imas_codex/llm/sn_tools.py`                        | `_segment_filter_search_sn`             | `_grammar_filter_search_standard_names`          | private    | rename                            |
| `imas_codex/llm/sn_tools.py`                        | `_vector_search_sn`                     | `_vector_search_standard_names`                  | private    | rename                            |
| `imas_codex/llm/sn_tools.py`                        | `_keyword_search_sn`                    | `_keyword_search_standard_names`                 | private    | rename                            |
| `imas_codex/llm/sn_tools.py`                        | *(new)*                                 | `_find_related_standard_names`                   | private    | new                               |
| `imas_codex/llm/sn_tools.py`                        | *(new)*                                 | `_check_standard_names`                          | private    | new                               |
| `imas_codex/llm/sn_tools.py`                        | *(new)*                                 | `_get_standard_name_summary`                     | private    | new                               |
| `imas_codex/standard_names/enrich_workers.py`       | `_fetch_nearby_sns`                     | `_fetch_nearby_standard_names`                   | private    | rename                            |
| `imas_codex/standard_names/enrich_workers.py`       | `max_per_sn` (kwarg)                    | `max_per_sn`                                     | private    | **keep** (private kwarg)          |
| `imas_codex/standard_names/orphan_sweep.py`         | `"stale_token_sn"` (dict key)           | `"stale_token_sn"`                               | internal   | **keep** (string literal)         |
| `imas_codex/standard_names/graph_ops.py`            | `fetch_docs_review_feedback_for_sns`    | `fetch_docs_review_feedback_for_standard_names`  | public     | rename + 1-release alias          |
| `imas_codex/standard_names/graph_ops.py`            | `_write_segment_edges`                  | `_write_grammar_decomposition`                   | private    | rename + behaviour change         |
| `imas_codex/cli/serve.py`                           | `--include-standard-names`              | `--include-standard-names`                       | public     | **keep** (default flipped to True in `33514f2a`) |
| `imas_codex/llm/server.py`                          | `include_standard_names: bool = True`   | `include_standard_names: bool = True`            | public     | **keep**; gate becomes `and not self.dd_only`    |

---

*End of plan 40 v3.1 — incorporates `33514f2a` `include_standard_names=True` default flip. Updated sections: §1 (changes header), §7.1, §8.3, §9.7, §10, §11 (A10/A14), §12, §14, §16, §17.*
