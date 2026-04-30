# Plan 39 — Structured Fan-Out for SN Compose Pipeline

> **Status**: DRAFT v2 — RD-revised.  Plan-only; no code in this task.
> **Branch**: `main`
> **Version history**:
> - v1 (initial draft) — superseded.
> - **v2 (this revision)** — addresses RD review: trimmed MVP catalog,
>   discriminated-union schemas, parent-lease cost ownership, deterministic
>   generate-site dup guard (no fan-out), granular failure modes, telemetry
>   gate before Phase 2, dropped `max_depth` knob.
> **Parent / context**: Plan 32 Phase 2 prototyped agentic tool-calling
> (`sn/generate_name_dd_tool_calling.md`).  This plan is the structured
> (non-agentic) successor.
> **Related**: Plan 31 (compose retry chain B12), Plan 43 (compose prompt
> reduction), Plan 26 (review pipeline).

---

## 1. Executive summary

The SN compose pipeline currently front-loads *all* DD context into the
compose prompt.  Plan 43 trimmed the static block; Plan 32 prototyped
free LLM tool-calling.  Free tool-calling is **rejected** for production:
unbounded depth/fan and high tail cost.

This plan specifies a **structured** fan-out pattern that preserves
information-on-demand with **hard bounds enforced at Pydantic parse time**:

| Property                | Free tool-calling     | Structured fan-out (this plan)            |
|-------------------------|-----------------------|-------------------------------------------|
| Function set            | Open                  | Closed catalog, discriminated union       |
| Fan degree              | Unbounded             | `≤ max_fan_degree` (default 3)            |
| Chain depth             | Unbounded             | Architecturally fixed at 1                |
| Round-trips             | 1 + N×tool turns      | Exactly **2** LLM calls                   |
| Cost ownership          | Implicit              | Parent call-site lease, sub-event tagged  |
| Failure mode            | Mid-loop stall        | Granular modes, true no-op on failures    |

Pattern: **(A) Query Proposer LLM → (B) Pure-Python parallel executor →
(C) Synthesizer LLM**.

**Phased rollout, gated on telemetry**:

- **Phase 1 (only true fan-out pilot)**: refine_name plug-in.
- **Phase 1.5 (deterministic, no fan-out)**: name-key dup guard before
  persist in generate.
- **Phase 2/3**: deferred — gated on Phase 1 telemetry showing positive
  lift after ≥1 week of production data.

---

## 2. Goals & non-goals

### Goals

- Add `imas_codex/standard_names/fanout/` module implementing the
  (Proposer → Executor → Synthesizer) pattern with **discriminated-union
  schema** for queries.
- Define a closed, MVP-sized function catalog wrapping existing graph
  helpers — no new graph queries.
- Plug into `refine_name` only, gated by config (default off).
- Cost charged to the **parent call-site's `BudgetLease`** as a sub-event,
  so caps already in force on the parent pool apply automatically.
- Granular failure modes; on any failure, fan-out is a true no-op (does
  not inject stub or "no useful evidence" text).

### Non-goals

- Free agentic tool-calling.
- Depth-2+ chaining.  No knob, no future-proofing — add a separate plan
  if a concrete need ever materialises.
- Runtime-generated functions.
- Multi-model debate (orthogonal — handled by RD-quorum review).
- Replacing B12 grammar-retry compose path.  B12 stays.
- Pre-persist generate fan-out — replaced by deterministic name-key
  lookup (§5.2).

---

## 3. Function catalog (MVP)

A **closed** set of functions.  The proposer LLM emits a JSON list
parsed via a **Pydantic discriminated union on `fn_id`** so each call
gets *typed* args at parse time — no runtime arg validation, no
`dict[str, Any]`, no out-of-bounds k/max_results.

All functions wrap **existing** Python helpers; direct in-process calls
(no MCP boundary).

### 3.1 MVP catalog

| ID                       | Backing function                                              | Cost class      | Why in MVP                                         |
|--------------------------|---------------------------------------------------------------|-----------------|----------------------------------------------------|
| `search_existing_names`  | `imas_codex.standard_names.search.search_similar_names`       | vector          | Description-similarity check (NOT a name dup check)|
| `search_dd_paths`        | `imas_codex.graph.dd_search.hybrid_dd_search`                 | vector + graph  | Cross-IDS / cross-domain DD context                |
| `find_related_dd_paths`  | `imas_codex.standard_names.workers._related_path_neighbours`  | graph           | Cluster/coordinate/unit siblings of a known DD path|
| `search_dd_clusters`     | `imas_codex.graph.dd_search` cluster path (existing)          | vector          | Concept-level grouping discovery                   |

**Dropped from MVP per RD review**:

- `fetch_existing_name` — useless at depth-1 without a prior search;
  if we ever need it, a depth-2 plan will reintroduce it.
- `list_grammar_vocabulary` — the existing helper
  (`imas_codex.llm.sn_tools._list_grammar_vocabulary`) returns *markdown*
  and only takes `segment`.  Building a thin pure-structured wrapper is
  out of scope for this plan; if vocab-gap surfacing is needed, it
  belongs in a deterministic reviewer-side helper (a separate plan), not
  in fan-out.
- `get_dd_identifiers` — same shape problem; defer.
- `search_signals` / `fetch_dd_paths` — irrelevant at compose-time refine.

### 3.2 Discriminated-union schemas (parse-time bound enforcement)

```python
# imas_codex/standard_names/fanout/schemas.py
from typing import Annotated, Literal
from pydantic import BaseModel, Field

# NOTE: All bounds (k, max_results, min/max length) are enforced by
# Pydantic at parse time.  No runtime post-validation, no silent
# clamping.  An out-of-bounds value rejects the *whole* FanoutCall.

class _SearchExistingNames(BaseModel):
    fn_id: Literal["search_existing_names"]
    query: str = Field(..., min_length=2, max_length=200)
    k: int = Field(default=5, ge=1, le=10)

class _SearchDDPaths(BaseModel):
    fn_id: Literal["search_dd_paths"]
    query: str = Field(..., min_length=2, max_length=200)
    k: int = Field(default=8, ge=1, le=15)
    # NOTE: ids_filter / physics_domain are NOT LLM-supplied (see §3.3);
    # they're injected by the caller.

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
    queries: list[FanoutCall] = Field(
        default_factory=list,
        max_length=3,            # mirror max_fan_degree config
    )
    # Optional rationale; logged only, not fed forward.
    notes: str = Field(default="", max_length=200)
```

A bad `fn_id`, missing required field, or out-of-bounds `k` causes
*parse* to fail at the call boundary — the entire `FanoutPlan` is
rejected.  Per RD point (5), this enters the `planner_schema_fail` mode
(true no-op, logged loudly).

### 3.3 Caller-injected scope (RD point 11)

The proposer LLM picks **only** `fn_id` and the *intent* arg (`query`
or `path`) plus a small `k`/`max_results`.  Known scope is injected by
the dispatcher from caller context, not asked of the model:

| Injected at runtime by dispatcher | Source                                      |
|-----------------------------------|---------------------------------------------|
| `physics_domain`                  | refine context (already known)              |
| `ids_filter`                      | derived from candidate's source DD path     |
| `facility`                        | not used at compose time; reserved          |
| `dd_version`                      | settings default                            |

The runner signature is therefore:

```python
async def run_search_dd_paths(
    args: _SearchDDPaths,
    *,
    gc: GraphClient,
    scope: FanoutScope,        # caller-injected
) -> FanoutResult: ...

class FanoutScope(BaseModel):
    physics_domain: str | None = None
    ids_filter: str | None = None
    dd_version: int | None = None
```

Shrinks the LLM-visible arg surface, the prompt size, and the error
surface (the model can't get scope wrong because it never sees it).

### 3.4 Grammar segments — sourced from ISN, not hard-coded (RD point 1)

If/when a vocab-segment-aware function is ever added, the segment
`Literal` MUST be built at module load time from
`imas_standard_names.grammar.get_grammar_context()` (see existing
pattern in `imas_codex/standard_names/audits.py`).  No hard-coded
segment list anywhere in `fanout/`.  This plan does not include such a
function in MVP; the rule is recorded here so it isn't reinvented when
one is added.

### 3.5 Output shape

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

---

## 4. Two-stage call pattern

### Stage A — Query Proposer LLM

- **Input**: candidate name + reviewer comment(s) + small `chain_history`
  excerpt (refine context).
- **System prompt**: `sn/fanout_propose.md` — short, lists the catalog
  and a verbatim **`catalog_version=<sha256>`** literal in the first line
  so any catalog change automatically invalidates prompt-cache prefixes
  (RD point 7).  Prompt caching only takes effect on the direct
  OpenRouter path (`supports_cache(model)` true and
  `OPENROUTER_API_KEY_IMAS_CODEX` set); proxy paths strip
  `cache_control` silently — the plan and the runner log a one-liner if
  caching is unavailable.
- **User prompt**: dynamic context only.
- **Output schema**: `FanoutPlan` (discriminated union).
- **Validation**: `acall_llm_structured` with `response_model=FanoutPlan`.
  Any parse failure → `planner_schema_fail` (see §7.2).  No agentic
  retry.
- **Model**: cheap-tier default (Haiku 4.5), low temperature (≤0.2);
  override via `[tool.imas-codex.sn.fanout].proposer-model`.

### Stage B — Function Executor (no LLM)

- For each `FanoutCall` (already typed by parse):
  - Look up runner by `fn_id`.
  - `asyncio.create_task(runner(args, gc=gc, scope=scope))` wrapped in
    `asyncio.wait_for(..., timeout=function_timeout_s)`.
- All tasks under `asyncio.gather(..., return_exceptions=True)`, the
  whole gather wrapped in `asyncio.wait_for(total_timeout_s)`.
- Per-call exceptions / timeouts → `FanoutResult(ok=False, error=...)`.
- Returns `list[FanoutResult]` in input order.
- Optional dedup across results (same `(kind, id)` collapses, max-score
  wins) before rendering.

### Stage C — Synthesizer

Stage C is **the call-site's existing LLM call** (e.g. the refine
worker's existing `acall_llm_structured` to refine a candidate).
`run_fanout` does **not** call Stage C itself — it returns the rendered
evidence string, which the call-site injects into its existing user
prompt.  This keeps the call-site's flow readable and avoids forcing a
new wrapper around every existing pool call.

```
                ┌────────── Stage A ──────────┐
candidate +     │ Proposer LLM (cheap, low-T) │
 minimal ctx ──▶│  catalog_version embedded    │── FanoutPlan
                └─────────────────────────────┘  (discriminated union,
                              │                   parse-time bounded)
                              ▼
                ┌────────── Stage B ──────────┐
                │ asyncio.gather + scope inj. │── list[FanoutResult]
                │ per-call + total timeout    │
                └─────────────────────────────┘
                              │
                              ▼  (returned to call-site as evidence string)
                ┌────────── Stage C ──────────┐
candidate +     │ EXISTING call-site LLM call │── refined / verified
evidence ─────▶ │ (e.g. refine_name worker)   │   candidate
                └─────────────────────────────┘
```

---

## 5. Plug-in sites

### 5.1 Phase 1 (only true fan-out): `refine_name`

- **Why first**: reviewer comment is a strong query signal; refine is
  already costly so one cheap proposer call is a small marginal.
- **Trigger gate (RD point 9)**: fan-out runs ONLY when *both*
  - the deterministic enrichment (B12) and `chain_history` injection
    have already been applied for the current cycle, AND
  - `chain_length > 0` AND the prior reviewer comment flagged ambiguity
    (configurable predicate; default: comment mentions any of
    `{"unclear", "ambiguous", "duplicate", "consider", "compare"}`).
- **Stage A inputs**: candidate name + reviewer comment + last
  `chain_history` slice + injected `FanoutScope` from caller.
- **Output**: rendered evidence string, injected into the existing
  refine user prompt.

This is the only site that pays for a proposer LLM call in this plan.

### 5.2 Phase 1.5 — Deterministic name-key dup guard for generate (NO fan-out)

Per RD point 3, the previously planned generate-site fan-out is
**replaced** by a deterministic, LLM-free dup guard:

- **When**: AFTER B12 produces its final candidate, BEFORE persisting
  the new `StandardName` node.
- **What**: a name-key lookup against existing StandardName ids.  Two
  forms:
  1. **Exact match** on the proposed `standard_name` string against
     existing `StandardName.id` (case-folded).
  2. **Lexical variants**: cheap normalization
     (e.g. underscore-join order canonicalisation, known synonym
     swaps from existing controlled-vocab maps) before comparison.
- **No vector search** — `search_similar_names` is description-vector,
  not a name-level dup check, and is intentionally NOT used here.
- **Action on hit**: drop the candidate, mark the source as
  `duplicate_of=<existing_id>`, increment a `dup_prevented` metric.
- **Implementation**: ~30 lines in
  `imas_codex/standard_names/canonical.py` (existing module — has the
  canonical-form helpers).  No fan-out infra used.

This phase is split out because it is independently valuable and
*does not* depend on the fan-out framework landing.  It can ship in
parallel with or before Phase 1.

### 5.3 Phase 2 / 3 — DEFERRED (telemetry-gated)

Per RD points 4 and 8, no further sites are added until Phase 1
telemetry has run **for at least one week of production traffic** AND
shows positive lift on at least one of:

- accept-rate delta on fanout-using vs non-using refine items, OR
- mean refine-rotation count delta (lower is better), OR
- `dup_prevented` rate (for the 1.5 deterministic guard).

Until then:

- vocab-gap suggestion remains in deterministic reviewer-side code (or
  is a separate later plan with its own structured wrapper around
  `_list_grammar_vocabulary`).
- docs style consistency stays out of scope.

---

## 6. Prompt design

Two prompt files under `imas_codex/llm/prompts/sn/`:

### 6.1 `sn/fanout_propose.md`

Structure:

```
catalog_version=<SHA256_HEX>           ← LITERAL first line, RD point 7

You help a Standard-Names refine pipeline pull targeted DD context.

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

The first-line `catalog_version=<sha256>` literal is generated at
module load by hashing the canonical catalog dict (sorted JSON of
`{fn_id: {schema_json}}`).  Any catalog change → new hash → cached
prompt prefix invalidates automatically.

Caching note (logged once per process at startup):

> Prompt caching active only when `supports_cache(proposer_model)` is
> true AND the runner is on the direct OpenRouter path (env
> `OPENROUTER_API_KEY_IMAS_CODEX` present).  On the proxy path, OpenAI/
> Anthropic `cache_control` markers are stripped silently — fan-out
> still functions but pays full prompt cost each call.

### 6.2 `sn/fanout_evidence_block.md` (renderer template)

The renderer (`fanout/render.py`) emits a compact markdown block to
inject into the call-site's existing user prompt:

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

Length-bounded: per-result hit cap (`result_hit_cap`, default 8),
total evidence cap (`evidence_token_cap`, default 2000).

### 6.3 No new synthesizer prompt

The existing refine prompt (`sn/refine_name_user.md`) is augmented at
render time with the evidence block via a single `{{ fanout_evidence }}`
placeholder.  When fan-out is disabled / no-op, the placeholder is the
empty string and the prompt is byte-identical to baseline (asserted by
the disabled-noop golden test).

---

## 7. Bounds, cost, and failure modes

### 7.1 Bounds

| Bound                     | Default | Where enforced                                   |
|---------------------------|---------|--------------------------------------------------|
| `max_fan_degree`          | 3       | `FanoutPlan.queries` `max_length` (parse time)    |
| Per-call k / max_results  | varies  | Per-variant `Field(le=...)` (parse time)         |
| `function_timeout_s`      | 5.0     | `asyncio.wait_for` per runner call               |
| `total_timeout_s`         | 12.0    | `asyncio.wait_for` wrapping Stage B `gather`     |
| `result_hit_cap`          | 8       | Renderer truncates each `FanoutResult.hits`      |
| `evidence_token_cap`      | 2000    | Renderer truncates total evidence                |

`max_depth` is **NOT a configurable knob** (RD point 10).  Depth is
architecturally fixed at 1 — there is no Stage D loop.

### 7.2 Failure modes (RD point 5 — explicit, granular)

| Mode                     | Trigger                                                         | Action                                                                           |
|--------------------------|-----------------------------------------------------------------|----------------------------------------------------------------------------------|
| `planner_schema_fail`    | `FanoutPlan` parse fails after the SDK's structured-output retry | Log loudly, increment metric, **true no-op** (no evidence injected — caller proceeds with baseline prompt). |
| `planner_all_invalid`    | Plan parses but `len(queries) == 0` after parse                  | Log info, increment metric, **true no-op**.                                      |
| `executor_partial_fail`  | Some runners failed/timed out, others returned hits              | Continue with what worked; render evidence from successful results only.         |
| `executor_all_empty`     | All runners returned `ok=True` but empty `hits`                  | **True no-op** (do NOT inject "no useful evidence" stub — that changes baseline).|
| `no_budget`              | `BudgetLease.charge_event` would overspend the parent reservation beyond a fan-out cap (see §7.3) | **True no-op**, log `fanout_no_budget`.                            |
| `feature_disabled`       | `enabled=false` or per-site flag false                           | Bypass entirely; no metrics emitted (silent).                                   |

"True no-op" means: the call-site's existing user prompt is rendered
with `fanout_evidence=""` and the existing call proceeds exactly as if
the feature were disabled.  No stub text, no "fan-out tried but
failed" comment.  Baseline behavior is preserved bit-for-bit.

### 7.3 Cost ownership (RD point 2 — corrected)

`BudgetLease` already supports auto-extension via `_extend_reservation`,
so a "fixed reservation" model would be misleading.  Corrected design:

- The **parent call-site** (refine pool worker) reserves once for the
  whole refine cycle, including a configured fan-out cost estimate
  (`fanout-cost-estimate`, default $0.005) added to the parent
  reservation.
- Fan-out's Stage A LLM call charges to the **parent lease** via
  `lease.charge_event(cost, LLMCostEvent(phase="sn_fanout_refine_proposer", ...))`.
- The synthesizer (Stage C) is the call-site's existing call and
  charges as it already does, but with a tag suffix when fan-out
  evidence was used (`phase="sn_refine_name+fanout"`) so the analytics
  query can isolate fan-out-using rotations.
- A soft cap `fanout-max-charge-per-cycle` (default $0.02) is
  enforced inside `run_fanout`: before issuing the proposer call, if
  `lease.remaining < fanout-max-charge-per-cycle - already_charged_to_fanout`
  → enter `no_budget` mode (true no-op).
- **Test invariant**: `lease.charged ≤ lease.reserved` after the
  parent cycle completes (note: `≤ final reserved`, since
  `BudgetLease._extend_reservation` may have grown the reservation —
  the invariant is that no charge is silently leaked or
  double-counted).

The plan does NOT introduce a separate `BudgetManager.reserve` call for
fan-out.  Single owner = parent.  Sub-events provide the visibility.

### 7.4 Idempotency

- Stage A: low-T + cached static system prompt → near-deterministic
  plans for identical inputs within a process lifetime.
- Stage B: read-only graph queries — deterministic given graph state.
- Stage C: existing call-site characteristics preserved.

---

## 8. Telemetry & rollout gate (RD point 8)

A `FanoutTelemetry` object emitted to logs (and graph-side `LLMCost`
where relevant) on every fan-out invocation, regardless of outcome:

| Metric                          | Type           | Notes                                          |
|---------------------------------|----------------|------------------------------------------------|
| `invocation_count`              | counter        | Tagged `(site, pool)`                          |
| `planner_schema_fail_count`     | counter        | Tagged `(site, model)`                         |
| `planner_all_invalid_count`     | counter        | Tagged `(site)`                                |
| `executor_partial_fail_count`   | counter        | Tagged `(site, fn_id)`                         |
| `executor_all_empty_count`      | counter        | Tagged `(site)`                                |
| `no_budget_count`               | counter        | Tagged `(site)`                                |
| `fn_call_count`                 | counter        | Tagged `(fn_id)`                               |
| `fn_failure_count`              | counter        | Tagged `(fn_id, kind=timeout|exception)`       |
| `evidence_tokens_rendered`      | histogram      | Median + p95                                   |
| `incremental_cost_per_fanout`   | histogram (USD)| Median + p95                                   |
| `accept_rate_delta`             | derived gauge  | Compare fanout-using vs non-using refines       |
| `refine_rotation_count_delta`   | derived gauge  | Likewise                                       |
| `dup_prevented_count`           | counter        | From the Phase 1.5 deterministic dup guard     |

These flow through the existing `LLMCostEvent` pipeline where
applicable (cost, tokens) and the existing structured-logging pipeline
otherwise.  No new infra.

**Phase 2 gate (binding)**: Phase 2/3 cannot start until Phase 1
(fan-out enabled in production) has run for **≥1 week** and emitted
telemetry showing positive lift on **at least one** of:

- `accept_rate_delta > 0` (fanout-using ≥ baseline, statistically),
- `refine_rotation_count_delta < 0` (fanout-using converges in fewer
  rotations), or
- `dup_prevented_count > 0` for the deterministic guard (Phase 1.5).

The gate is documented in `AGENTS.md` and re-asserted in any plan that
proposes new fan-out sites.

---

## 9. Configuration

```toml
[tool.imas-codex.sn.fanout]
# Master switch.  Default off until rolled out.
enabled = false
# Hard cap on Stage A query count (mirrored in FanoutPlan.queries.max_length).
max-fan-degree = 3
# Per-function timeout.
function-timeout-s = 5.0
# Total Stage B timeout.
total-timeout-s = 12.0
# Per-result hit cap fed to renderer.
result-hit-cap = 8
# Total evidence token budget rendered into the synthesizer prompt.
evidence-token-cap = 2000
# Stage A (proposer) model.
proposer-model = "openrouter/anthropic/claude-haiku-4.5"
proposer-temperature = 0.1
# Cost estimate added to the PARENT lease reservation when a refine cycle
# starts (covers Stage A + delta synth context tokens).
fanout-cost-estimate = 0.005
# Soft cap on cumulative fanout sub-event charges per refine cycle.
# Above this, fan-out enters no_budget mode (true no-op).
fanout-max-charge-per-cycle = 0.02
# Refine-side trigger predicate (substrings; comment must contain at
# least one to enable fan-out for that cycle).
refine-trigger-keywords = ["unclear", "ambiguous", "duplicate", "consider", "compare"]

# Per-site enable flags.  Master `enabled` must also be true.
[tool.imas-codex.sn.fanout.sites]
refine_name = false   # Phase 1 flips this when ready
# generate_name dup guard (Phase 1.5) is NOT a fan-out site — it has its
# own deterministic config under [tool.imas-codex.sn] (see canonical.py).
```

`max-depth` is intentionally absent (RD point 10).

Loaded by `imas_codex/standard_names/fanout/config.py` into a
`FanoutSettings` Pydantic model.

---

## 10. Module layout

```
imas_codex/standard_names/fanout/
  __init__.py
  catalog.py          # CatalogEntry registry (fn_id → runner)
  schemas.py          # Discriminated-union FanoutCall + FanoutPlan + result types
  runners.py          # async wrappers around existing helpers
  dispatcher.py       # propose() / execute() / run_fanout() public API
  render.py           # format_results() — markdown evidence block
  config.py           # FanoutSettings + version-hash computation
  telemetry.py        # FanoutTelemetry emitter
  README.md           # how to add a function to the catalog
```

Public API (called from `workers.py` refine path):

```python
async def run_fanout(
    *,
    site: Literal["refine_name"],   # MVP: only one site exists
    candidate: CandidateContext,    # name + path + description
    reviewer_comment: str | None,
    chain_history_excerpt: str | None,
    scope: FanoutScope,             # caller-injected, not LLM-supplied
    gc: GraphClient,
    parent_lease: BudgetLease,      # parent's lease, sub-event charged
    settings: FanoutSettings,
) -> str:
    """Returns the rendered evidence block (possibly empty string).

    Empty string ⇒ true no-op (any failure mode in §7.2).  Caller
    injects the (possibly empty) string into its existing prompt
    template.
    """
```

---

## 11. Phased rollout

### Phase 1 — Refine fan-out (Wave 1)

**Goal**: Land framework + refine plug-in, default-off, with full
telemetry.  Validate on `magnetic_field_diagnostics` probe loop.

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
- `tests/standard_names/fanout/test_dispatcher_integration.py`
- `tests/standard_names/fanout/fixtures/refine_baseline.txt`

**Files modified**:

- `imas_codex/standard_names/workers.py` — refine path: trigger gate +
  `run_fanout` + evidence injection into existing user prompt.
- `imas_codex/llm/prompts/sn/refine_name_user.md` — add
  `{{ fanout_evidence }}` placeholder (renders as `""` when disabled).
- `pyproject.toml` — add `[tool.imas-codex.sn.fanout]`.
- `imas_codex/standard_names/config/__init__.py` — export loader.
- `AGENTS.md` — short § on the structured fan-out pattern + the Phase 2
  telemetry gate.

**Acceptance criteria (Phase 1)**:

1. `enabled=false`: refine prompts are byte-identical to current
   baseline (golden test `test_disabled_is_noop.py`).
2. `enabled=true` + stub Stage A returning a fixed `FanoutPlan`:
   refine prompt receives a deterministic evidence block.
3. `FanoutPlan` parse-time validation rejects: unknown `fn_id`,
   out-of-bounds `k`, missing required `query`/`path`, plan with
   `len(queries) > max_fan_degree`.  Each is a unit test.
4. `planner_schema_fail` and `planner_all_invalid` modes produce
   `evidence == ""` and emit the corresponding metric.
5. `executor_partial_fail` is observable in tests with one stub runner
   raising — others' results render normally.
6. `executor_all_empty` produces `evidence == ""` (no stub text).
7. `function_timeout_s` and `total_timeout_s` enforced under simulated
   slow runners.
8. `no_budget` mode triggered when parent lease's remaining is below the
   per-cycle cap; `evidence == ""`.
9. **Cost invariant** (`test_dispatcher.py::test_no_leaked_budget`):
   after a refine cycle that uses fan-out, `parent_lease.charged ≤
   parent_lease.reserved` (`≤` allows for legitimate `_extend_reservation`),
   and the sum of `phase=sn_fanout_*` `LLMCost` events equals
   `parent_lease.charged - non_fanout_charges`.
10. `catalog_version` literal at top of system prompt matches
    `sha256(canonical_catalog_dict)` (asserted by
    `test_catalog_version_hash`).
11. `uv run pytest tests/standard_names/fanout -v` green.
12. `uv run ruff check . && uv run ruff format --check .` clean.
13. `uv run pytest tests/graph/test_schema_compliance.py -v` green
    (no schema changes, but verify).

### Phase 1.5 — Deterministic name-key dup guard (Wave 1, parallel to Phase 1)

**Independently shippable** — does not depend on fan-out infra.

**Files modified**:

- `imas_codex/standard_names/canonical.py` — add `find_existing_name_key()`
  helper + integration into the persist-time path in `workers.py`.
- `imas_codex/standard_names/workers.py` — call dup guard before persist.
- `pyproject.toml` — `[tool.imas-codex.sn].dup-guard-enabled = true`
  (default on; deterministic, low risk).

**Acceptance criteria (Phase 1.5)**:

1. Exact-match dup is detected against existing `StandardName.id` (case-folded).
2. Lexical-variant dup is detected against the documented variant set.
3. On hit, candidate is dropped + source marked `duplicate_of=<id>`.
4. `dup_prevented_count` metric emitted on every hit.
5. No LLM call introduced; cost invariant: zero LLM cost for the dup
   guard itself.

### Phase 2 / 3 — DEFERRED, telemetry-gated

Per §8, no further sites added until Phase 1 has emitted ≥1 week of
production telemetry showing positive lift.  When the gate is met, a
follow-up plan (40+) will:

- Specify the next site with its own discriminated-union additions.
- Justify the catalog expansion (e.g. add a structured wrapper around
  `_list_grammar_vocabulary` if vocab-gap auto-suggest is the chosen
  next site).
- Re-establish a Phase-2 telemetry gate for any subsequent expansion.

---

## 12. Test strategy

All Stage A LLM calls are stubbed in tests.  Integration tests use the
real graph fixture but stubbed LLM.

### 12.1 Unit (Phase 1)

| File / case                                                  | Coverage                                                             |
|--------------------------------------------------------------|----------------------------------------------------------------------|
| `test_schemas.py::test_discriminator_routes_to_correct_variant` | Each `fn_id` parses to its typed model.                          |
| `test_schemas.py::test_unknown_fn_id_rejects`                 | Whole plan rejects, no silent drop.                                  |
| `test_schemas.py::test_out_of_bounds_k_rejects`               | `k=99` → ValidationError; bounds at parse time.                      |
| `test_schemas.py::test_missing_required_field_rejects`        | `_SearchDDPaths` without `query` → ValidationError.                  |
| `test_schemas.py::test_max_fan_degree`                         | `len(queries) > 3` → ValidationError.                                 |
| `test_dispatcher.py::test_planner_schema_fail_noop`            | Stub LLM returns invalid JSON → `evidence == ""`, metric emitted.    |
| `test_dispatcher.py::test_planner_all_invalid_noop`            | Plan with `queries=[]` → `evidence == ""`.                           |
| `test_dispatcher.py::test_executor_partial_fail`               | One stub runner raises; rendered evidence omits the failure but      |
|                                                                | includes the rest.                                                   |
| `test_dispatcher.py::test_executor_all_empty_noop`             | All runners return empty `hits` → `evidence == ""` (no stub).        |
| `test_dispatcher.py::test_function_timeout`                     | Slow runner cancelled at `function_timeout_s`.                        |
| `test_dispatcher.py::test_total_timeout`                        | `gather` aborts at `total_timeout_s`; partial results returned.       |
| `test_dispatcher.py::test_no_budget_noop`                       | Parent lease near-exhausted → `evidence == ""`, metric emitted.      |
| `test_dispatcher.py::test_no_leaked_budget`                     | `parent_lease.charged ≤ parent_lease.reserved` post-cycle, and       |
|                                                                | sum of `phase=sn_fanout_*` events equals fan-out charged.            |
| `test_dispatcher.py::test_scope_injection`                     | LLM-supplied `args` never include scope fields; runner sees scope    |
|                                                                | from caller.                                                         |
| `test_dispatcher.py::test_catalog_version_hash_in_prompt`      | First line of rendered system prompt is                              |
|                                                                | `catalog_version=<sha256>` matching the runtime-computed hash.       |
| `test_runners.py`                                               | Each runner's happy-path + at least one failure path.                 |
| `test_render.py::test_per_hit_cap`                             | Renderer truncates each result's hits.                                |
| `test_render.py::test_total_token_cap`                         | Renderer truncates total evidence.                                    |
| `test_render.py::test_empty_inputs_return_empty_string`         | No results / all empty → `""` (not "no useful evidence").            |
| `test_disabled_is_noop.py`                                     | With `enabled=false`, refine call produces a prompt byte-identical   |
|                                                                | to the stored baseline fixture.                                       |

### 12.2 Integration (Phase 1)

| File                                  | Coverage                                                                  |
|---------------------------------------|---------------------------------------------------------------------------|
| `test_dispatcher_integration.py`      | Real graph fixture; stubbed Stage A returns a canonical plan; Stage B    |
|                                       | hits live graph (`search_existing_names`, `find_related_dd_paths`),      |
|                                       | asserts well-known seed nodes returned.                                  |

### 12.3 Phase 1.5 tests

| File                                       | Coverage                                                       |
|--------------------------------------------|----------------------------------------------------------------|
| `tests/standard_names/test_dup_guard.py`   | Exact + lexical-variant detection; persist path skipped; metric|
|                                            | emitted; zero LLM cost.                                        |

### 12.4 Telemetry tests

`test_telemetry.py` — every failure mode in §7.2 emits exactly the
expected metric tag set; happy path emits `invocation_count`,
`evidence_tokens_rendered`, `incremental_cost_per_fanout`.

---

## 13. Acceptance criteria (overall)

The plan is fully delivered when:

1. All Phase 1 + Phase 1.5 acceptance criteria met.
2. Phase 1 enabled in production for ≥1 week with telemetry collected
   per §8.
3. The Phase-2 gate evaluation is documented (a short
   `plans/features/standard-names/39-fanout-phase1-telemetry.md`
   appendix), regardless of whether the gate passes.
4. Fan-out spend per refine cycle visible in graph (filter
   `phase LIKE 'sn_fanout_%'`) and bounded by
   `fanout-max-charge-per-cycle`.
5. No agentic-loop fallback path exists in code (review checklist).
6. No hard-coded grammar segment lists in `fanout/` (RD point 1).

---

## 14. Documentation updates required

| File                                              | Change                                                       |
|---------------------------------------------------|--------------------------------------------------------------|
| `AGENTS.md`                                        | New § "Structured fan-out" linking to this plan + the Phase 2|
|                                                    | telemetry gate.                                              |
| `imas_codex/standard_names/fanout/README.md`       | New — how to add a function to the catalog (discriminated-   |
|                                                    | union pattern, scope-injection rule, version hash).          |
| `imas_codex/standard_names/canonical.py`           | Module docstring updated to describe the dup-guard helper.   |
| `plans/features/standard-names/39-...md`           | This plan; update Phase status checkboxes as work proceeds.  |
| `plans/features/standard-names/39-fanout-phase1-telemetry.md` | Created at end of Phase 1 with measured values    |
|                                                    | feeding the Phase-2 gate.                                    |

---

## 15. What this plan does NOT do (re-stated for the reviewer)

- ❌ No free agentic looping.
- ❌ No depth-2+ chaining (no knob, no future-proofing).
- ❌ No runtime function generation.
- ❌ No multi-model debate.
- ❌ No new graph queries / schema changes.
- ❌ No touching B12 grammar-retry compose path.
- ❌ No pre-persist generate fan-out (replaced by deterministic
  name-key dup guard).
- ❌ No vector-similarity name dup check (description-vector ≠ name dup).
- ❌ No stub "fan-out failed" text injected into prompts on failure
  (true no-op only).

---

## 16. Open questions for further RD review

1. **Refine trigger predicate**: keyword-based default
   (`refine-trigger-keywords`) is crude.  Should we instead key on the
   reviewer's structured output fields (e.g. `score < threshold` AND
   `category="ambiguity"`) once those are reliably populated?  Defer
   until Phase 1 telemetry shows the keyword variant is too noisy.
2. **Cache feasibility check at startup**: should the runner *probe*
   `supports_cache(proposer_model)` and emit a one-time warning if the
   proxy path is detected, or is the existing logging in
   `imas_codex/discovery/base/llm.py` sufficient?
3. **Phase 1.5 dup-guard variant set**: enumerate the lexical
   normalisations explicitly in the Phase 1.5 PR (do not infer from
   ad-hoc tests).  Suggest a starting set:
   case-fold, collapse repeated underscores, canonical-segment-order,
   known synonym map from `imas_codex/standard_names/canonical.py`.

---

*End of plan v2.*
