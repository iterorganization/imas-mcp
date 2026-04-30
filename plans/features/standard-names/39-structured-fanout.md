# Plan 39 — Structured Fan-Out for SN Compose Pipeline

> **Status**: DRAFT v3 — Opus 4.7 RD-revised.  Plan-only; no code in this task.
> **Branch**: `main`
> **Version history**:
> - v1 — initial draft, superseded.
> - v2 — gpt-5.4 RD pass: trimmed catalog, discriminated union, parent-lease
>   cost ownership, deterministic generate-site dup guard, granular failure
>   modes, dropped `max_depth` knob.
> - **v3 (this revision)** — Opus 4.7 RD pass: adds Phase 0 helper-extraction
>   prerequisite (B1); fixes async/sync mismatch with `asyncio.to_thread`
>   wrappers (B2); tiers cost estimate by escalation and rewrites budget
>   invariant (I1); within-cohort A/B for Phase-2 gate (I2); pins reviewer
>   dim allow-list (I3); hashes fully rendered prompt (I4); explicit
>   GraphClient lifecycle (I5); `Fanout` telemetry node + `batch_id`
>   linkage (I6); plus six small refinements (S1–S6).
> **Parent / context**: Plan 32 Phase 2 prototyped agentic tool-calling.
> **Related**: Plan 31 (compose retry chain B12), Plan 43 (compose prompt
> reduction), Plan 26 (review pipeline).

---

## 1. Executive summary

Structured (non-agentic) fan-out for the SN compose pipeline.  The LLM
emits a **bounded, schema-validated** list of search queries from a closed
catalog of backing functions; we execute them in parallel; results feed
the next LLM call.  Bounds are enforced at Pydantic parse time — there is
no agentic loop.

| Property                | Free tool-calling     | Structured fan-out (this plan)            |
|-------------------------|-----------------------|-------------------------------------------|
| Function set            | Open                  | Closed catalog, discriminated union       |
| Fan degree              | Unbounded             | `≤ max_fan_degree` (default 3)            |
| Chain depth             | Unbounded             | Architecturally fixed at 1                |
| Round-trips             | 1 + N×tool turns      | Exactly **2** LLM calls                   |
| Cost ownership          | Implicit              | Parent call-site lease, sub-event tagged  |
| Failure mode            | Mid-loop stall        | Granular modes, true no-op                |

Pattern: **(A) Query Proposer LLM → (B) Pure-Python parallel executor →
(C) Synthesizer LLM (the call-site's existing call)**.

**Phased rollout, gated on telemetry**:

- **Phase 0** — backing-helper extractions (prerequisites for Phase 1).
- **Phase 1** — refine_name fan-out (only true fan-out pilot).
- **Phase 1.5** — deterministic name-key dup guard (no fan-out, no LLM).
- **Phase 2/3** — deferred; gated on Phase 1 within-cohort A/B telemetry.

---

## 2. Goals & non-goals

### Goals

- Add `imas_codex/standard_names/fanout/` implementing
  (Proposer → Executor → Synthesizer).
- Reuse existing graph helpers via a closed, MVP-sized catalog — but
  **only after Phase 0 reshapes those helpers into the right
  signatures** (sync, `gc=` kwarg, public symbol).
- Plug into `refine_name` only; default off.
- Cost charged to the **parent call-site's `BudgetLease`** as sub-events.
- Granular failure modes; on any failure, fan-out is a true no-op.
- All-helper-sync runners explicitly wrapped in `asyncio.to_thread`.

### Non-goals

- Free agentic tool-calling.
- Depth-2+ chaining.
- Runtime-generated functions.
- Multi-model debate (orthogonal — RD-quorum review).
- Replacing B12 grammar-retry compose path.
- Pre-persist generate fan-out — replaced by deterministic name-key
  lookup (§5.2).
- New graph schema fields on existing nodes; the lone schema addition
  is a thin telemetry-only `Fanout` node (§8).

---

## 3. Function catalog (MVP)

A **closed** set of functions.  The proposer LLM emits a JSON list parsed
via a **Pydantic discriminated union on `fn_id`** so each call gets typed
args at parse time — no runtime arg validation, no `dict[str, Any]`, no
out-of-bounds `k` / `max_results`.

### 3.1 MVP catalog — final symbols (post-Phase-0)

All four entries point at sync helpers in `imas_codex/graph/dd_search.py`,
all four take `gc` as a parameter, none open their own `GraphClient`.

| ID                       | Backing symbol (post Phase 0)                            | Signature shape                                            | Cost class       |
|--------------------------|----------------------------------------------------------|------------------------------------------------------------|------------------|
| `search_existing_names`  | `imas_codex.graph.dd_search.search_similar_names`        | `(query, k, *, gc=None, include_superseded=False)` (sync)  | vector           |
| `search_dd_paths`        | `imas_codex.graph.dd_search.hybrid_dd_search`            | `(gc, query, ..., k=...)` (already conforms)               | vector + graph   |
| `related_dd_search`      | `imas_codex.graph.dd_search.related_dd_search`           | `(gc, path, *, max_results=...)` (sync, public)            | graph            |
| `cluster_search`         | `imas_codex.graph.dd_search.cluster_search`              | `(gc, query, *, scope=None, k=...)` (sync, public)         | vector           |

The proposer-visible `fn_id` is the human-friendly form
(`search_existing_names`, `search_dd_paths`, `find_related_dd_paths`,
`search_dd_clusters`); each maps 1:1 to a backing symbol via the catalog
registry.  Display names are not the same as backing symbol names so the
catalog can outlive helper renames.

### 3.2 Discriminated-union schemas (parse-time bounds)

```python
# imas_codex/standard_names/fanout/schemas.py
from typing import Annotated, Literal
from pydantic import BaseModel, Field

# ALL bounds enforced at parse time.  Out-of-bounds → whole FanoutPlan rejects.

class _SearchExistingNames(BaseModel):
    fn_id: Literal["search_existing_names"]
    query: str = Field(..., min_length=2, max_length=200)
    k: int = Field(default=5, ge=1, le=10)

class _SearchDDPaths(BaseModel):
    fn_id: Literal["search_dd_paths"]
    query: str = Field(..., min_length=2, max_length=200)
    k: int = Field(default=8, ge=1, le=15)
    # ids_filter / physics_domain are NOT LLM-supplied (see §3.3).

class _FindRelatedDDPaths(BaseModel):
    fn_id: Literal["find_related_dd_paths"]
    path: str = Field(..., min_length=3, max_length=300)
    max_results: int = Field(default=12, ge=1, le=20)

class _SearchDDClusters(BaseModel):
    fn_id: Literal["search_dd_clusters"]
    query: str = Field(..., min_length=2, max_length=200)
    k: int = Field(default=8, ge=1, le=15)

FanoutCall = Annotated[
    _SearchExistingNames | _SearchDDPaths
    | _FindRelatedDDPaths | _SearchDDClusters,
    Field(discriminator="fn_id"),
]

class FanoutPlan(BaseModel):
    queries: list[FanoutCall] = Field(default_factory=list, max_length=3)
    notes: str = Field(default="", max_length=200)   # logged only
```

### 3.3 Caller-injected scope

Proposer LLM picks **only** `fn_id` + intent arg (`query` / `path`) +
small `k` / `max_results`.  Known scope is injected at runtime from
caller context, not asked of the model:

| Injected at runtime               | Source                                |
|-----------------------------------|---------------------------------------|
| `physics_domain`                  | refine context (already known)        |
| `ids_filter`                      | derived from candidate's source path  |
| `dd_version`                      | settings default                      |

Runner signature:

```python
async def run_search_dd_paths(
    args: _SearchDDPaths,
    *,
    gc: GraphClient,
    scope: FanoutScope,
) -> FanoutResult: ...

class FanoutScope(BaseModel):
    physics_domain: str | None = None
    ids_filter: str | None = None
    dd_version: int | None = None
```

### 3.4 Grammar segments — sourced from ISN

If/when a vocab-segment-aware function is ever added, the segment
`Literal` MUST be built at module load from
`imas_standard_names.grammar.get_grammar_context()` (existing pattern in
`imas_codex/standard_names/audits.py`).  No hard-coded list.  Not in MVP.

### 3.5 Output shapes

```python
class FanoutHit(BaseModel):
    kind: Literal["standard_name", "dd_path", "cluster"]
    id: str
    label: str
    score: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

class FanoutResult(BaseModel):
    fn_id: str
    args: dict[str, Any]
    ok: bool
    hits: list[FanoutHit] = Field(default_factory=list)
    error: str | None = None
    elapsed_ms: float = 0.0
```

### 3.6 Phase 0 — helper extractions (PREREQUISITE for Phase 1)

**Three of four MVP runners cannot be implemented as thin wrappers
against the helpers as they exist today** (verified against `main`):

| Helper (current state)                                | Problem                                                                                              |
|-------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `standard_names.search.search_similar_names`          | Sync, opens its own `with GraphClient() as gc:`.  No `gc` parameter — under 6-pool × `max_fan_degree=3` worst case = up to 18 simultaneous fresh `GraphClient` instantiations. |
| `graph.dd_search.hybrid_dd_search`                    | ✓ Already conforms: sync, takes `gc`.                                                                |
| `standard_names.workers._related_path_neighbours`     | Leading-underscore private; would create cyclic import (`workers.py` ↔ `fanout.runners`).             |
| Cluster search                                        | **Does not exist as a callable helper.**  Available only as `tools.graph_search.ClusterSearchTool.search_dd_clusters` async method (decorated `@cache_results / @mcp_tool / @handle_errors`).  Calling it re-introduces the MCP boundary the plan disclaims. |

Phase 0 is the gating prerequisite.  Phase 1 cannot start until Phase 0
lands.  Phase 0 changes are **independently shippable** (they improve the
helpers regardless of fan-out).

#### Phase 0 deliverables

(a) **`search_similar_names` overload**: add `gc: GraphClient | None = None`
    keyword.  When `None`, open a session as today (back-compat).  When
    provided, reuse it.  Add `include_superseded: bool = False` keyword
    (S2): when `True`, drop the `pipeline_status='superseded'` exclusion
    (cycle-2 refine wants the just-superseded cycle-1 name as comparator).
    Move callable to `imas_codex/graph/dd_search.py`; keep an import-shim
    in `imas_codex/standard_names/search.py` until in-tree callers
    migrate (search call sites grep is small — ≤5 hits).

(b) **`related_dd_search` extraction**: move `_related_path_neighbours`
    body from `workers.py` to `imas_codex/graph/dd_search.py`, rename to
    `related_dd_search` (drop leading underscore, drop `_path_` infix to
    match `hybrid_dd_search` siblings).  Keep
    `from imas_codex.graph.dd_search import related_dd_search as
    _related_path_neighbours` shim in `workers.py` until callers migrate.

(c) **`cluster_search` extraction**: extract the Cypher + helper logic
    out of `ClusterSearchTool.search_dd_clusters` body in
    `imas_codex/tools/graph_search.py` into a sync
    `cluster_search(gc, query, *, scope=None, k=8) -> list[ClusterHit]`
    in `imas_codex/graph/dd_search.py`.  Rewrite
    `ClusterSearchTool.search_dd_clusters` to delegate to it (`async`
    wrapper preserves the MCP-tool decorator semantics).  Tests on the
    MCP side must continue to pass unchanged.

#### Phase 0 acceptance criteria

1. All four catalog backing symbols live in `imas_codex/graph/dd_search.py`;
   each is sync; each takes `gc` as the **first positional** parameter
   (or `gc=` kwarg for `search_similar_names` to preserve its existing
   signature shape).
2. No new graph queries, no schema changes — Phase 0 is pure refactor.
3. Existing MCP tool tests for `search_dd_clusters` (`tests/tools/...`)
   pass unchanged.
4. `tests/standard_names/test_search_similar_names.py::test_gc_reuse`:
   pass a `gc` kwarg; assert no new `GraphClient` is instantiated
   (mock the constructor).
5. `tests/graph/test_related_dd_search.py`: smoke test for the renamed
   function plus a back-compat test for the `_related_path_neighbours`
   shim.
6. Lint + format clean; full pytest green.
7. **No fan-out code lands in Phase 0.**  Phase 0 PR is pure helper
   refactor; Phase 1 PR follows.

---

## 4. Two-stage call pattern

### 4.1 Stage A — Query Proposer LLM

- **Input**: candidate name + reviewer-comment slice (per §5.1 trigger
  predicate, only `clarity` and `disambiguation` dims) + bounded
  `chain_history` excerpt (S3: last cycle's `reviewer_comments_per_dim`
  values truncated to **800 chars total**, no chain walk).
- **System prompt**: `sn/fanout_propose.md`.  First line is a literal
  `catalog_version=<sha256>` placeholder; see §6.1 for hash
  specification (I4: hash the **fully rendered system prompt**, not the
  schema dict).
- **User prompt**: dynamic context only.
- **Output schema**: `FanoutPlan`.
- **Validation**: `acall_llm_structured(response_model=FanoutPlan)`.
  Parse failure → `planner_schema_fail` (§7.2), no agentic retry.
- **Model**: `proposer-model` (default cheap-tier Haiku 4.5),
  `proposer-temperature` ≤ 0.2.
- **Post-parse query-side dedup (S1)**: in `dispatcher.propose()`,
  collapse calls with identical `(fn_id, normalized_query_or_path)`
  to the first occurrence; bump `duplicate_query_collapsed` counter.
  Normalization: lowercase + collapse whitespace.

### 4.2 Stage B — Function Executor (sync helpers, async wrapper)

**All four MVP backing helpers are synchronous** (Neo4j blocking I/O +
numpy embeddings).  A naive `async def runner(...): return helper(...)`
runs the helper **on the event loop** — `asyncio.gather` does not
parallelise it, and `asyncio.wait_for` cannot cancel sync code
mid-execution.

Mandatory pattern: every runner wraps its helper with
`asyncio.to_thread`:

```python
async def run_search_dd_paths(args, *, gc, scope) -> FanoutResult:
    t0 = time.monotonic()
    try:
        hits = await asyncio.wait_for(
            asyncio.to_thread(
                hybrid_dd_search,
                gc, args.query,
                physics_domain=scope.physics_domain,
                ids_filter=scope.ids_filter,
                k=args.k,
            ),
            timeout=settings.function_timeout_s,
        )
        return FanoutResult(fn_id=args.fn_id, args=args.model_dump(),
                            ok=True, hits=_to_hits(hits),
                            elapsed_ms=(time.monotonic() - t0) * 1000)
    except (asyncio.TimeoutError, Exception) as e:
        return FanoutResult(fn_id=args.fn_id, args=args.model_dump(),
                            ok=False, error=str(e),
                            elapsed_ms=(time.monotonic() - t0) * 1000)
```

`asyncio.to_thread` schedules onto the default thread pool
(`min(32, cpus + 4)` workers).  This pool is **shared** with existing
`asyncio.to_thread(persist_refined_name, ...)` calls in workers and the
rest of the 6-pool loop.  Under heavy fan-out load, refine persistence
may briefly contend with fan-out runners on the same thread pool.
Acceptable for MVP; revisit if measured contention shows up in
telemetry.  *No* dedicated thread pool in MVP — keep machinery minimal.

`total_timeout_s` wraps the outer `gather`; even if a sync helper
ignores its individual `function_timeout_s`, the gather-level
`wait_for` cancels Python-side waiters at the gate (the helper itself
keeps running on the worker thread until it completes — it cannot be
preempted, but its result is discarded).  Document this caveat.

### 4.3 Stage C — Synthesizer

Stage C is the call-site's existing LLM call (refine worker's existing
`acall_llm_structured`).  `run_fanout` returns the rendered evidence
string; the call-site injects it into its existing prompt template.

```
                ┌────────── Stage A ──────────┐
candidate +     │ Proposer LLM (cheap, low-T) │── FanoutPlan
 minimal ctx ──▶│  catalog_version embedded    │   (parse-time bounded)
                └─────────────────────────────┘
                              │ S1 query dedup
                              ▼
                ┌────────── Stage B ──────────┐
                │ asyncio.to_thread runners   │
                │ + per-call wait_for          │── list[FanoutResult]
                │ + total_timeout gather       │
                └─────────────────────────────┘
                              │ render
                              ▼  (returned to caller as evidence string)
                ┌────────── Stage C ──────────┐
candidate +     │ EXISTING refine LLM call    │── refined candidate
evidence ─────▶ │ (charges parent lease)      │
                └─────────────────────────────┘
```

---

## 5. Plug-in sites

### 5.1 Phase 1 fan-out site: `refine_name`

- **Trigger gate (orthogonality, RD point 9 v2)**: fan-out fires ONLY
  when *all* of:
  - deterministic enrichment (B12) and `chain_history` injection have
    already been applied for the current cycle, AND
  - `chain_length > 0`, AND
  - prior reviewer comments contain at least one trigger keyword
    (default: `{unclear, ambiguous, duplicate, consider, compare}`,
    configurable via `refine-trigger-keywords`).
- **Comment source (I3)**: trigger predicate flattens *only* the values
  of `reviewer_comments_per_dim` keyed by dims in the explicit allow-list
  `{"clarity", "disambiguation"}`.  Other dims (e.g. `convention`,
  `grammar`) are ignored — those failure modes are not what fan-out is
  designed to address.  The legacy free-form `reviewer_comments_name`
  is *not* used.  Allow-list is configurable via
  `refine-trigger-comment-dims`.
- **Excerpt size (S3)**: concatenated allow-listed values, total
  truncated to 800 chars (no multi-cycle walk).
- **Stage A inputs**: candidate name + truncated comment slice +
  injected `FanoutScope`.
- **Output**: rendered evidence string injected into the existing
  refine user prompt.

This is the only site that pays for a proposer LLM call in this plan.

### 5.2 Phase 1.5 — Deterministic name-key dup guard (NO fan-out)

Independently shippable (no fan-out infra required).

- **When**: AFTER B12's final candidate, BEFORE persisting the new
  `StandardName`.
- **What**: name-key lookup against existing `StandardName.id`:
  1. Exact case-folded match.
  2. Lexical-variant normalisation (case-fold, collapsed underscores,
     canonical-segment-order, known synonym swaps from
     `imas_codex/standard_names/canonical.py`).  Exact set enumerated
     in the Phase 1.5 PR — not inferred from ad-hoc tests.
- **No vector search** — `search_similar_names` is description-vector,
  not a name-level dup check.  Intentionally NOT used here.
- **Action on hit**: drop candidate, mark source `duplicate_of=<id>`,
  emit `dup_prevented` metric.
- **Implementation**: ~30 lines in `canonical.py`; no LLM call.

### 5.3 Phase 2 / 3 — DEFERRED

Per §8 telemetry gate.  Until met:

- vocab-gap suggestion remains a deterministic reviewer-side helper (or
  a future plan with its own structured-output wrapper around
  `_list_grammar_vocabulary`).
- docs style consistency stays out of scope.

---

## 6. Prompt design

### 6.1 `sn/fanout_propose.md` — system prompt + version hash (I4)

The proposer system prompt is computed at module load **as a single
string**:

```
catalog_version=<SHA256_HEX>          ← LITERAL first line, see below

You help an SN refine pipeline pull targeted DD context.

Available functions (pick AT MOST 3, OR ZERO if none would help):

- search_existing_names(query: str, k: int 1..10)
    Find existing StandardName nodes similar to a description string.

- search_dd_paths(query: str, k: int 1..15)
    Hybrid search over DD paths.  (Scope is supplied by the caller —
    do NOT include physics_domain or ids_filter in your output.)

- find_related_dd_paths(path: str, max_results: int 1..20)
    Cluster / coordinate / unit siblings of a known DD path.

- search_dd_clusters(query: str, k: int 1..15)
    Concept-level cluster discovery.

Output JSON conforming to the schema you have been given.  Returning
{"queries": []} is a valid answer when no query would help.  Do NOT
invent function names.  Do NOT add fields beyond the schema.
```

#### Hash specification (corrected per RD I4)

The hash covers the **fully rendered system prompt body** — *not* the
catalog schema dict.  Reasons:

- `Pydantic.model_json_schema()` output is **not stable across pydantic
  minor versions** (e.g. `definitions` → `$defs`).  A dependency bump
  silently invalidates every cached prefix on the wrong axis.
- Help text in the prompt is not in the schema dict — a help-text edit
  would change the actual prompt prefix without changing a schema-only
  hash, so the version line would *lie* about identity.
- Hashing the rendered string folds in any future
  `[tool.imas-codex.sn.fanout]` knobs that are rendered into the prompt
  (e.g. if `max-fan-degree` ever surfaces verbatim).

Procedure:

```python
# fanout/config.py
def _compute_catalog_version() -> str:
    body = render_template_without_version_line()      # everything below
                                                       # the version line
    return hashlib.sha256(body.encode("utf-8")).hexdigest()

CATALOG_VERSION = _compute_catalog_version()

# At prompt-render time, the version line is literally
#   f"catalog_version={CATALOG_VERSION}\n"
# prepended to the static body.  Hash is over the body only,
# but the line *contains* the hash, so any change to the body
# changes the line's literal content and busts the cache prefix.
```

Note in §6 explicitly: `proposer-temperature` is *not* part of the
hash (sampling parameters do not change the prompt; cache hits across
temperature changes are correct — cache covers the prompt, not the
sampling) (S6).

#### Caching path note

Prompt caching only takes effect on the direct OpenRouter path
(`supports_cache(proposer_model)` true AND
`OPENROUTER_API_KEY_IMAS_CODEX` present).  On the proxy path,
`cache_control` markers are stripped silently — fan-out still functions
but pays full prompt cost each call.  The runner emits a one-time
`fanout_cache_unavailable` info-log at startup if it detects the
proxy path.

### 6.2 `sn/fanout_evidence_block.md` — renderer template

Compact markdown block injected into the call-site's existing user
prompt via a `{{ fanout_evidence }}` placeholder:

```
## Fan-out evidence (queries=N, errors=M)

### search_existing_names("electron temperature core", k=5)
- electron_temperature_at_magnetic_axis  (score=0.91)  unit=eV  kind=scalar
- electron_temperature                    (score=0.87)  unit=eV  kind=scalar
...

### find_related_dd_paths("core_profiles/profiles_1d/electrons/temperature")
- core_profiles/profiles_1d/electrons/temperature_validity   (cluster)
- equilibrium/time_slice/profiles_1d/electrons/temperature   (unit-match)
```

Length-bounded: `result_hit_cap` (default 8 per result),
`evidence_token_cap` (default 2000 total; default **800 when the parent
call-site is on the escalator model**, see I1).

### 6.3 No new synthesizer prompt

The existing `sn/refine_name_user.md` gains a single
`{{ fanout_evidence }}` placeholder.  When fan-out is disabled / no-op,
the placeholder renders as `""` and the prompt is byte-identical to
baseline (asserted by the disabled-noop golden test).

---

## 7. Bounds, cost, failure modes

### 7.1 Bounds

| Bound                     | Default                              | Where enforced                                  |
|---------------------------|--------------------------------------|-------------------------------------------------|
| `max_fan_degree`          | 3                                    | `FanoutPlan.queries` `max_length` (parse time)  |
| Per-call `k`/`max_results`| varies (see §3.2)                    | Per-variant `Field(le=...)` (parse time)        |
| `function_timeout_s`      | 5.0                                  | `asyncio.wait_for` per runner call              |
| `total_timeout_s`         | 12.0                                 | `asyncio.wait_for` wrapping `gather`            |
| `result_hit_cap`          | 8                                    | Renderer truncates each `FanoutResult.hits`     |
| `evidence_token_cap`      | 2000 baseline / **800 escalation**   | Renderer truncates total evidence (I1)          |

`max_depth` remains intentionally absent.

### 7.2 Failure modes (granular, true no-op semantics)

| Mode                     | Trigger                                                              | Action                                                                                                                                       |
|--------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `planner_schema_fail`    | `FanoutPlan` parse fails after the SDK's structured-output retry     | Log loudly, increment metric, **true no-op** (caller proceeds with baseline prompt, `evidence == ""`).                                       |
| `planner_all_invalid`    | Plan parses but `len(queries) == 0` (incl. after S1 dedup)            | Log info, increment metric, **true no-op**.                                                                                                   |
| `executor_partial_fail`  | Some runners failed/timed out, others returned hits                  | Continue with what worked; render evidence from successful results only.                                                                     |
| `executor_all_empty`     | All runners `ok=True` but empty `hits`                               | **True no-op** (no stub text).                                                                                                               |
| `no_budget`              | Cumulative fan-out sub-event spend would breach `fanout-max-charge-per-cycle` (see §7.3) | **True no-op**, log `fanout_no_budget`.                                                                                                    |
| `feature_disabled`       | `enabled=false` or per-site flag false                               | Bypass entirely; no metrics emitted, no `Fanout` graph node written (S5).                                                                    |

"True no-op" means: existing user prompt rendered with
`fanout_evidence=""`; existing call proceeds exactly as if fan-out
were disabled.  Baseline behavior preserved bit-for-bit.

### 7.3 Cost ownership (corrected per RD I1)

`BudgetLease.charge_event` calls `_extend_reservation` and never
raises `BudgetExceeded` (verified `budget.py:169-192`).  Therefore:

- **The cap is enforceable only *before* an LLM call**, not as an
  invariant on `_reserved`.
- The Stage C synthesiser is the existing refine call **with a larger
  user prompt** (up to `evidence_token_cap` extra tokens).  At Opus
  escalation rates (~$15/Mtok input) 2000 tokens ≈ $0.03 above
  baseline — *much* more than v2's flat `fanout-cost-estimate=0.005`.
  Reservation pad must be **tiered** by escalation.

Corrected design:

```toml
[tool.imas-codex.sn.fanout]
# Cost added to the parent lease at refine-cycle start.
# Tiered by whether this cycle is Opus-escalation (chain_length >=
# rotation_cap - 1).  Baseline cycles are Sonnet/Haiku-class.
fanout-cost-estimate-baseline = 0.005
fanout-cost-estimate-escalation = 0.05

# Per-cycle hard cap on cumulative fan-out sub-event spend.
# Above this, run_fanout returns "" (no_budget mode).  Independent
# of the parent reservation; this is the cap that actually fires.
fanout-max-charge-per-cycle-baseline = 0.02
fanout-max-charge-per-cycle-escalation = 0.10
```

The refine worker, at cycle start, computes
`escalate = (chain_length >= rotation_cap - 1)` and adds the
escalation-tier estimate to its parent reservation when `escalate`,
otherwise the baseline.  `run_fanout` reads the same escalation flag
to:

- choose the cap (`fanout-max-charge-per-cycle-*`),
- shrink `evidence_token_cap` to 800 on escalation,
- pre-flight gate against the cap before the proposer call.

Charges:

- Stage A proposer: charged to parent lease via
  `lease.charge_event(cost, LLMCostEvent(phase="sn_fanout_refine_proposer", batch_id=fanout_run_id, ...))`.
- Stage C synthesizer: charges as today, with phase suffix
  `+fanout` and `batch_id=fanout_run_id` (see I6) so the analytics
  query can isolate fan-out-using rotations.

**Test invariant (rewritten per RD I1)**:

> After a refine cycle that uses fan-out,
> `parent_lease.charged ≤ original_reservation + cumulative_fanout_charges`,
> where `original_reservation` is the value snapshotted **before** the
> first `_extend_reservation` call.  Equivalently: extensions are
> attributable line-by-line to specific `LLMCostEvent`s; no charge is
> silently leaked or double-counted.

The naive v2 "`charged ≤ reserved`" invariant is tautological because
`_reserved` auto-grows; it is replaced by this stronger statement.

### 7.4 Idempotency

- Stage A: low-T + cached static system prompt → near-deterministic
  plans for identical inputs.
- Stage B: read-only graph queries.
- Stage C: existing call-site characteristics preserved.

---

## 8. Telemetry — durable Fanout node + `batch_id` join (I6)

### 8.1 Why a graph node, not just logs

v2 said "counters flow through structured-logging — no new infra."
Opus 4.7 RD: counters in logs do not roll up to a queryable store
across pod/process boundaries.  Phase-2 gate evaluation would degrade
to grepping log files, and the join from a fan-out invocation to its
proposer + synthesizer `LLMCostEvent`s relies on `phase` substring
matching with no explicit run id — fragile.

### 8.2 `Fanout` node (telemetry-only, runtime-written)

A thin runtime-telemetry node written **once per `run_fanout`
invocation** (skipped when `feature_disabled` per S5):

```cypher
(:Fanout {
  id: <fanout_run_id>,        // uuid4
  sn_id: <sn_id>,
  site: "refine_name",
  outcome: "ok" | "planner_schema_fail" | "planner_all_invalid"
         | "executor_partial_fail" | "executor_all_empty" | "no_budget",
  plan_size: <int>,           // queries in plan post-dedup
  hits_count: <int>,          // total hits across results
  evidence_tokens: <int>,
  arm: "on" | "off",          // for §8.4 within-cohort A/B
  escalate: <bool>,
  created_at: datetime()
})
```

- **NOT** a schema-managed graph node in the LinkML sense — it is a
  *runtime telemetry* node, in the same spirit as `LLMCost`.  It is
  *not* added to `agents/schema-reference.md`.  The plan declares the
  exemption explicitly so a future schema-compliance check does not
  flag it as drift.
- Acceptance: `tests/standard_names/fanout/test_telemetry_node.py`
  asserts the node is written for a representative invocation and
  matches the `Fanout` shape above.

### 8.3 `LLMCostEvent.batch_id` linkage

`LLMCostEvent` already has an unused `batch_id: str | None` field
(verified `budget.py:68`).  Use it to stamp `fanout_run_id` on **both**
the proposer event and the synthesizer event when fan-out fires.
Cypher join:

```cypher
MATCH (f:Fanout {id: $run_id})
MATCH (c:LLMCost {batch_id: f.id})
RETURN f, collect(c) AS charges
```

No new edges, no new schema fields — just a re-use of the existing
column.

### 8.4 Within-cohort A/B for Phase-2 gate (corrected per RD I2)

v2's gate was selection-biased: comparing fanout-using vs non-using
items conflates fan-out efficacy with cohort difficulty (the trigger
predicate selects already-failing, ambiguity-flagged items).  The fix:

When the trigger predicate matches:

1. Compute `arm = "on" if hash((sn_id, chain_length)) % 2 == 0 else "off"`.
2. If `arm == "on"`: run fan-out as planned.
3. If `arm == "off"`: skip fan-out (true no-op), but **still write a
   `Fanout` node with `outcome="off_arm"` and `arm="off"`** so the
   denominator is queryable.
4. Stamp `arm` onto the synthesizer `LLMCostEvent` via `batch_id`
   (the on-arm `Fanout.id`) — the off arm uses a UUID derived from
   `(sn_id, chain_length, "off")` so the join still works.

Gate metrics computed strictly *within* the trigger-eligible cohort:

- `accept_rate(arm=on) - accept_rate(arm=off)` (target > 0)
- `mean_refine_rotations(arm=on) - mean_refine_rotations(arm=off)` (target < 0)

Compounding caveat to acknowledge: with `DEFAULT_REFINE_ROTATIONS=3`
and trigger requiring `chain_length > 0`, fan-out fires only on
cycles 2 and 3, so the structural ceiling on rotation savings is
"save one cycle on already-bad items."  The plan documents this
ceiling so the gate isn't held to an unrealistic effect size.

### 8.5 Other counters

Logs-only counters (kept as today, not promoted to nodes):

- `duplicate_query_collapsed` (S1)
- `fn_call_count` per `fn_id`
- `fn_failure_count` per `fn_id, kind=timeout|exception`
- `fanout_cache_unavailable` (one-shot info)

These are useful for in-flight ops but not for Phase-2 gate evaluation,
which operates on `Fanout` + `LLMCost`.

### 8.6 Gate enforcement (S4)

The Phase-2 gate is "binding" only if mechanically enforced.  Add a
**CI lint** in `.github/workflows/lint.yml` (or equivalent existing
hook):

```bash
# Any future fan-out-expansion plan must cite the Phase-1 telemetry
# appendix.  Failure = lint failure.
for plan in plans/features/standard-names/4*-fanout*.md; do
  [[ -f "$plan" ]] && \
    grep -q '39-fanout-phase1-telemetry' "$plan" || {
      echo "ERROR: $plan must cite 39-fanout-phase1-telemetry.md"
      exit 1
    }
done
```

Cheap; makes the gate enforceable rather than aspirational.

---

## 9. Configuration

```toml
[tool.imas-codex.sn.fanout]
# Master switch.  Default off until rolled out.
enabled = false
# Hard cap on Stage A query count.
max-fan-degree = 3
# Per-function and total Stage B timeouts.
function-timeout-s = 5.0
total-timeout-s = 12.0
# Per-result hit cap (renderer).
result-hit-cap = 8
# Total evidence token caps (renderer).
evidence-token-cap-baseline = 2000
evidence-token-cap-escalation = 800
# Stage A model.
proposer-model = "openrouter/anthropic/claude-haiku-4.5"
proposer-temperature = 0.1
# Cost padding added to PARENT lease at refine-cycle start (tiered, I1).
fanout-cost-estimate-baseline = 0.005
fanout-cost-estimate-escalation = 0.05
# Per-cycle hard cap on fan-out sub-event spend (tiered, I1).
fanout-max-charge-per-cycle-baseline = 0.02
fanout-max-charge-per-cycle-escalation = 0.10
# Refine trigger.
refine-trigger-keywords = ["unclear", "ambiguous", "duplicate", "consider", "compare"]
# Reviewer-comment dim allow-list for the trigger predicate (I3).
refine-trigger-comment-dims = ["clarity", "disambiguation"]
# Comment excerpt total length cap (S3).
refine-trigger-comment-chars = 800

# Per-site enable flags.
[tool.imas-codex.sn.fanout.sites]
refine_name = false       # Phase 1 flips to true after Phase 0 lands.
# generate_name dup guard (Phase 1.5) is NOT a fan-out site — it has its
# own deterministic config under [tool.imas-codex.sn].
```

Loaded by `imas_codex/standard_names/fanout/config.py` into a
`FanoutSettings` Pydantic model.

---

## 10. Module layout & GraphClient lifecycle (I5)

```
imas_codex/standard_names/fanout/
  __init__.py
  catalog.py          # CatalogEntry registry (fn_id → runner)
  schemas.py          # Discriminated-union FanoutCall + FanoutPlan + result types
  runners.py          # async wrappers (asyncio.to_thread) around graph helpers
  dispatcher.py       # propose() / execute() / run_fanout() public API
  render.py           # format_results() — markdown evidence block
  config.py           # FanoutSettings + _compute_catalog_version()
  telemetry.py        # FanoutTelemetry emitter + Fanout-node writer
  README.md           # how to add a function to the catalog
```

### 10.1 GraphClient lifecycle (rule, I5)

The refine worker (`workers.py`) opens **one** `GraphClient` for the
entire refine cycle and passes it into:

- the existing `_hybrid_search_neighbours(gc, ...)` call (already
  takes `gc`), AND
- the new `run_fanout(..., gc=gc, ...)` call.

`run_fanout` passes that same `gc` into every runner via the kwarg.
Every catalog runner accepts `gc` as a keyword argument and reuses it.
**No runner instantiates `GraphClient`.**  This rule is what makes
Phase 0 deliverable (a) — the `gc=None` overload on
`search_similar_names` — load-bearing for the catalog.

### 10.2 Public API

```python
async def run_fanout(
    *,
    site: Literal["refine_name"],   # MVP: only one site
    candidate: CandidateContext,    # name + path + description + chain_length
    reviewer_comments_per_dim: dict[str, str] | None,
    scope: FanoutScope,             # caller-injected, not LLM-supplied
    gc: GraphClient,                # one client per refine cycle, see §10.1
    parent_lease: BudgetLease,
    settings: FanoutSettings,
    arm: Literal["on", "off"] = "on",   # §8.4 within-cohort A/B
    fanout_run_id: str | None = None,    # auto-uuid4 if None
) -> str:
    """Returns the rendered evidence block (possibly empty string).

    Empty string ⇒ true no-op (any failure mode in §7.2 OR arm=='off').
    Caller injects the (possibly empty) string into its existing
    {{ fanout_evidence }} placeholder.
    """
```

---

## 11. Phased rollout

### Phase 0 — Helper extractions (PREREQUISITE, Wave 1)

See §3.6.  Three sub-PRs (a/b/c), each independently shippable, each
land before Phase 1 starts.

### Phase 1 — Refine fan-out (Wave 2, after Phase 0)

**Goal**: Land framework + refine plug-in, default-off, with
`Fanout`-node telemetry.

**Files added**:

- `imas_codex/standard_names/fanout/{__init__,catalog,schemas,runners,dispatcher,render,config,telemetry}.py`
- `imas_codex/standard_names/fanout/README.md`
- `imas_codex/llm/prompts/sn/fanout_propose.md`
- `imas_codex/llm/prompts/sn/fanout_evidence_block.md`
- `tests/standard_names/fanout/test_schemas.py`
- `tests/standard_names/fanout/test_dispatcher.py`
- `tests/standard_names/fanout/test_runners.py`
- `tests/standard_names/fanout/test_render.py`
- `tests/standard_names/fanout/test_disabled_is_noop.py`
- `tests/standard_names/fanout/test_telemetry_node.py`
- `tests/standard_names/fanout/test_dispatcher_integration.py`
- `tests/standard_names/fanout/fixtures/refine_baseline.txt`

**Files modified**:

- `imas_codex/standard_names/workers.py` — refine path: trigger gate,
  arm hashing, parent-reservation tier, `run_fanout` call (passes the
  cycle's `gc`), evidence injection.
- `imas_codex/llm/prompts/sn/refine_name_user.md` — add
  `{{ fanout_evidence }}` placeholder (renders as `""` when disabled).
- `pyproject.toml` — add `[tool.imas-codex.sn.fanout]`.
- `imas_codex/standard_names/config/__init__.py` — export loader.
- `AGENTS.md` — short § on the structured fan-out pattern + Phase-2
  telemetry gate + GraphClient-lifecycle rule.
- `.github/workflows/lint.yml` (or existing pre-commit hook) — add the
  S4 CI lint for future fan-out-expansion plans.

**Acceptance criteria (Phase 1)** — additive to Phase 0:

1. `enabled=false`: refine prompts byte-identical to baseline (golden
   `test_disabled_is_noop.py`).
2. `enabled=true` + stub Stage A returning a fixed `FanoutPlan`:
   refine prompt receives a deterministic evidence block.
3. `FanoutPlan` parse-time validation rejects: unknown `fn_id`,
   out-of-bounds `k`, missing required `query`/`path`,
   `len(queries) > max_fan_degree`.  Each is a unit test.
4. `planner_schema_fail` and `planner_all_invalid` modes produce
   `evidence == ""` and emit metric + `Fanout` node with appropriate
   `outcome`.
5. `executor_partial_fail` observable with one stub runner raising —
   others' results render normally.
6. `executor_all_empty` produces `evidence == ""` (no stub text).
7. **Sync-runner timeout test (B2-mandated)**:
   `test_dispatcher.py::test_function_timeout_with_sync_helper` uses a
   sync `time.sleep(10)`-based runner via `asyncio.to_thread`; verifies
   the *Python-side* `wait_for` returns at `function_timeout_s` even
   though the helper continues running on the worker thread (the result
   is discarded).  An additional assertion verifies the result was
   `ok=False, error="timeout"`.
8. Total-timeout test parallel to (7) wraps the gather.
9. `no_budget` mode triggered when fan-out cumulative spend would
   breach `fanout-max-charge-per-cycle-*`; `evidence == ""`,
   `Fanout.outcome="no_budget"`.
10. **Cost invariant (rewritten per I1)**:
    `test_dispatcher.py::test_cost_attribution`: snapshot `original_reservation`
    before any `_extend_reservation`; after the cycle, assert
    `parent_lease.charged ≤ original_reservation + sum(fanout_charges)`,
    and assert that every `LLMCost` event with `batch_id == fanout_run_id`
    has phase prefix `sn_fanout_` or phase suffix `+fanout`.
11. `catalog_version` line at top of system prompt matches
    `sha256(rendered_body)`; mutating any character of the body
    (template, help text) flips the hash
    (`test_catalog_version_hash_covers_body`).
12. Trigger predicate uses only `clarity` / `disambiguation` dims;
    other dim values present in `reviewer_comments_per_dim` do not
    fire the trigger
    (`test_trigger_predicate.py::test_dim_allowlist`).
13. Comment excerpt is truncated to `refine-trigger-comment-chars`
    (`test_trigger_predicate.py::test_excerpt_truncated`).
14. `Fanout` graph node is written for `arm=on` AND `arm=off`
    invocations; not written when `feature_disabled`.
15. CI lint (S4) blocks a synthetic test plan that doesn't cite
    `39-fanout-phase1-telemetry.md`.
16. `uv run pytest tests/standard_names/fanout -v` green.
17. `uv run ruff check . && uv run ruff format --check .` clean.
18. `uv run pytest tests/graph/test_schema_compliance.py -v` green
    (`Fanout` node is runtime-only and exempt — exemption documented in
    the schema-compliance test if it would otherwise flag the label).

### Phase 1.5 — Deterministic name-key dup guard (Wave 1, parallel)

Independent of fan-out infra.

**Acceptance criteria (Phase 1.5)**:

1. Exact-match dup detected against existing `StandardName.id`.
2. Lexical-variant dup detected against the documented variant set
   (set enumerated explicitly in the PR).
3. Candidate dropped + source marked `duplicate_of=<id>`.
4. `dup_prevented_count` metric emitted.
5. Zero LLM cost.

### Phase 2 / 3 — DEFERRED, telemetry-gated

Per §8.4, gate is mechanically enforced via S4 CI lint and within-cohort
A/B metrics.

---

## 12. Test strategy

### 12.1 Phase 0

| File / case                                      | Coverage                                                  |
|--------------------------------------------------|-----------------------------------------------------------|
| `tests/standard_names/test_search_similar_names.py::test_gc_reuse` | `gc=` kwarg → no new `GraphClient` (mock the constructor).|
| `tests/standard_names/test_search_similar_names.py::test_include_superseded` | S2 path: superseded names included when flag set.    |
| `tests/graph/test_related_dd_search.py`           | Renamed function smoke + back-compat shim.                |
| `tests/tools/test_cluster_search.py` (existing)   | Pass unchanged after `cluster_search` extraction.         |

### 12.2 Phase 1 — unit

| File / case                                                       | Coverage                                                              |
|-------------------------------------------------------------------|-----------------------------------------------------------------------|
| `test_schemas.py::test_discriminator_routes`                       | Each `fn_id` parses to its typed model.                               |
| `test_schemas.py::test_unknown_fn_id_rejects`                      | Whole plan rejects.                                                   |
| `test_schemas.py::test_out_of_bounds_k_rejects`                    | `k=99` → ValidationError; bounds at parse time.                       |
| `test_schemas.py::test_missing_required_field_rejects`             | `_SearchDDPaths` without `query` → ValidationError.                   |
| `test_schemas.py::test_max_fan_degree`                             | `len(queries) > 3` → ValidationError.                                  |
| `test_dispatcher.py::test_query_dedup` (S1)                         | Two `(fn_id, normalized_query)`-identical calls collapse to one;      |
|                                                                    | `duplicate_query_collapsed` counter emitted.                          |
| `test_dispatcher.py::test_planner_schema_fail_noop`                 | Stub LLM returns invalid JSON → `evidence == ""`, `Fanout.outcome`    |
|                                                                    | matches.                                                               |
| `test_dispatcher.py::test_planner_all_invalid_noop`                 | Plan with `queries=[]` → `evidence == ""`.                            |
| `test_dispatcher.py::test_executor_partial_fail`                   | One stub runner raises; others render.                                 |
| `test_dispatcher.py::test_executor_all_empty_noop`                 | All runners return empty hits → `evidence == ""`.                     |
| `test_dispatcher.py::test_function_timeout_with_sync_helper` (B2)   | Sync `time.sleep(10)` runner; `wait_for` returns at 5 s with `ok=False`. |
| `test_dispatcher.py::test_total_timeout_with_sync_helpers` (B2)     | Multiple slow sync runners; gather aborts at 12 s.                    |
| `test_dispatcher.py::test_no_budget_noop`                          | Fan-out spend cap reached → `evidence == ""`, `Fanout.outcome`.       |
| `test_dispatcher.py::test_cost_attribution` (I1)                   | Snapshot `original_reservation`; assert per-charge attribution.       |
| `test_dispatcher.py::test_arm_assignment` (I2)                     | Hash routing yields ~50/50 across N synthetic items.                  |
| `test_dispatcher.py::test_arm_off_writes_fanout_node` (I2)         | `arm=off` still writes `Fanout` with `outcome=off_arm`.               |
| `test_dispatcher.py::test_scope_injection`                         | LLM-supplied `args` never include scope; runner sees scope from caller.|
| `test_dispatcher.py::test_catalog_version_hash_covers_body` (I4)    | Mutate a help-text char → hash flips. Mutate a comment in code that   |
|                                                                    | does not affect the rendered body → hash unchanged.                   |
| `test_runners.py`                                                  | Each runner happy-path + at least one failure path; uses real         |
|                                                                    | `asyncio.to_thread` + a sync stub helper.                             |
| `test_render.py::test_per_hit_cap`                                 | Renderer truncates each result.                                       |
| `test_render.py::test_total_token_cap_baseline_vs_escalation` (I1)  | 2000 (baseline) and 800 (escalation) caps both honoured.              |
| `test_render.py::test_empty_inputs_return_empty_string`             | No results / all empty → `""`.                                        |
| `test_disabled_is_noop.py`                                         | `enabled=false` → byte-identical to baseline fixture.                 |
| `test_disabled_is_noop.py::test_no_fanout_node_when_disabled` (S5)  | Disabled invocation does not write a `Fanout` node.                   |
| `test_telemetry_node.py`                                           | Node properties match the §8.2 shape; `batch_id` linkage on           |
|                                                                    | `LLMCost` events present.                                              |
| `test_trigger_predicate.py::test_dim_allowlist` (I3)               | Comment in disallowed dim does not trigger.                           |
| `test_trigger_predicate.py::test_excerpt_truncated` (S3)            | Total excerpt ≤ `refine-trigger-comment-chars`.                       |

### 12.3 Phase 1 — integration

| File                                  | Coverage                                                                       |
|---------------------------------------|--------------------------------------------------------------------------------|
| `test_dispatcher_integration.py`      | Real graph fixture; stubbed Stage A returns canonical plan; Stage B hits live |
|                                       | graph; well-known seed nodes returned; `Fanout` node persisted; LLMCost rows  |
|                                       | linked via `batch_id`.                                                         |

### 12.4 Phase 1.5 tests

| File                                       | Coverage                                                       |
|--------------------------------------------|----------------------------------------------------------------|
| `tests/standard_names/test_dup_guard.py`   | Exact + lexical-variant detection; persist skipped; metric emitted; zero LLM cost. |

### 12.5 CI lint (S4)

| File                                                     | Coverage                                                         |
|----------------------------------------------------------|------------------------------------------------------------------|
| `tests/lint/test_phase2_gate_lint.py`                    | Synthetic `plans/.../40-fanout-vocab.md` without the citation     |
|                                                          | exits non-zero; with the citation exits zero.                    |

---

## 13. Acceptance criteria (overall)

The plan is fully delivered when:

1. Phase 0 acceptance criteria met; helper extractions landed.
2. Phase 1 + Phase 1.5 acceptance criteria met.
3. Phase 1 enabled in production for ≥1 week with telemetry collected
   per §8 (within-cohort A/B).
4. The Phase-2 gate evaluation written up in
   `plans/features/standard-names/39-fanout-phase1-telemetry.md`,
   regardless of whether the gate passes.
5. Fan-out spend per refine cycle visible via the `Fanout` ↔ `LLMCost`
   join (`batch_id`).
6. No agentic-loop fallback in code (review checklist).
7. No hard-coded grammar segment lists in `fanout/`.
8. CI lint (S4) blocks future fan-out-expansion plans that do not
   cite the telemetry appendix.

---

## 14. Documentation updates required

| File                                                            | Change                                                       |
|-----------------------------------------------------------------|--------------------------------------------------------------|
| `AGENTS.md`                                                      | New § "Structured fan-out" + GraphClient-lifecycle rule +    |
|                                                                  | Phase-2 telemetry gate.                                      |
| `imas_codex/standard_names/fanout/README.md`                     | New — how to add a function to the catalog (discriminated-   |
|                                                                  | union pattern, scope-injection rule, version hash, `gc`      |
|                                                                  | reuse rule).                                                 |
| `imas_codex/standard_names/canonical.py`                         | Module docstring updated to describe the dup-guard helper.   |
| `imas_codex/graph/dd_search.py`                                  | Module docstring updated to list the four post-Phase-0       |
|                                                                  | catalog symbols.                                             |
| `plans/features/standard-names/39-...md`                         | This plan; tick boxes as work proceeds.                      |
| `plans/features/standard-names/39-fanout-phase1-telemetry.md`    | Created at end of Phase 1 with measured A/B values.          |

---

## 15. What this plan does NOT do

- ❌ No free agentic looping.
- ❌ No depth-2+ chaining (no knob, no future-proofing).
- ❌ No runtime function generation.
- ❌ No multi-model debate.
- ❌ No new graph schema additions on existing nodes; only the
  runtime-telemetry `Fanout` node (exempt from schema-compliance).
- ❌ No B12 grammar-retry compose-path changes.
- ❌ No pre-persist generate fan-out.
- ❌ No vector-similarity name dup check.
- ❌ No stub "fan-out failed" text on failure (true no-op only).
- ❌ No new thread pool.  Default `asyncio.to_thread` pool is shared
   with existing workers; contention acceptable for MVP.

---

## 16. Open questions for further RD review

1. **Trigger predicate refinement**: the keyword-based default is crude;
   may key on structured reviewer fields (`score < threshold AND
   category="ambiguity"`) once those are reliably populated.  Defer
   until Phase 1 telemetry shows the keyword variant is too noisy.
2. **Phase 1.5 lexical-variant set**: enumerated in the PR, not
   inferred — starter set: case-fold, collapsed underscores,
   canonical-segment-order, known synonym map from `canonical.py`.
3. **`Fanout` node retention**: telemetry-only nodes accumulate.
   Recommend a TTL sweep (delete `Fanout` older than 30 days) bundled
   into existing graph-housekeeping if there is one; otherwise note as
   a follow-up cleanup task.

---

*End of plan v3.*
