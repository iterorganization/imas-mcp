# Plan 39 — Structured Fan-Out for SN Compose Pipeline

> **Status**: DRAFT — pending RD review.  Plan-only; no code in this task.
> **Branch**: `main`
> **Parent / context**: Plan 32 Phase 2 explored agentic tool-calling
> (`sn/generate_name_dd_tool_calling.md`).  This plan defines the
> structured (non-agentic) successor.
> **Related**: Plan 31 (compose retry chain), Plan 43 (compose prompt
> reduction), Plan 26 (review pipeline).

---

## 1. Executive summary

The SN compose pipeline currently front-loads *all* DD context (cluster
siblings, related paths, vocab tokens) into a single ~39 K-token compose
prompt.  Plan 43 trimmed the static block; Plan 32 prototyped giving the
LLM free tool-calling access (`fetch_cluster_siblings`,
`fetch_reference_exemplar`, `fetch_version_history`).  Free tool-calling
is **rejected** for production for two reasons:

1. **Unbounded depth & fan**.  Tool-call chains depend on model whim and
   can exceed any fixed budget under tail conditions.
2. **High per-turn cost**.  Each tool turn is a full round-trip (system
   prompt + chain history); models tend to over-call.

This plan specifies a **structured** fan-out pattern that preserves the
information-on-demand benefit while imposing **hard bounds**:

| Property                | Free tool-calling     | Structured fan-out (this plan)     |
|-------------------------|-----------------------|------------------------------------|
| Function set            | Open (LLM picks tools)| Closed catalog (Pydantic-schema'd) |
| Fan degree              | Unbounded             | `≤ max_fan_degree` (default 3)     |
| Chain depth             | Unbounded             | `= max_depth` (default 1)          |
| Round-trips             | 1 + N×tool turns      | Exactly **2** LLM calls            |
| Cost predictability     | Tail-heavy            | Fixed reservation via `BudgetLease`|
| Failure mode            | Mid-loop stall        | Fall-through to existing chain     |

The pattern is: **(A) Query Proposer LLM** → **(B) Pure-Python parallel
executor** → **(C) Synthesizer LLM**.  Stages A and C are LLM calls;
Stage B is `asyncio.gather` over backing-function calls validated against
a closed catalog.

Phased rollout: start with the highest-value plug-in point (REFINE_NAME),
then GENERATE_NAME pre-persist consistency check, then DOCS.

---

## 2. Goals & non-goals

### Goals

- Add a reusable `imas_codex.standard_names.fanout` module that
  implements the (Proposer → Executor → Synthesizer) pattern.
- Define a closed function catalog wrapping existing graph helpers
  (`hybrid_dd_search`, `_related_path_neighbours`, `search_similar_names`,
  vocab/cluster/identifier helpers).
- Plug it into `refine_name` first, gated by config (default off).
- Cost-bounded by `BudgetLease`; falls through to existing chain on any
  hard failure (no silent agentic retry).

### Non-goals (explicitly out of scope)

- Free agentic tool-calling loops.
- LLM-driven function chaining beyond `max_depth` (default 1).
- Generating new functions at runtime (no LLM-defined tools).
- Multi-model debate (orthogonal — handled by RD-quorum review).
- Replacing the `_re_enrich_expanded` retry path in compose (B12).  That
  remains the cheap deterministic first-line retry; fan-out is the
  expensive targeted second line.

---

## 3. Function catalog

A **closed** set of backing functions exposed to Stage A (proposer) and
executed in Stage B.  The proposer LLM emits a JSON list of
`{function: <name>, args: {...}}` entries; each entry is validated
against a Pydantic input schema.  Anything else is rejected.

All functions wrap **existing** Python helpers — no new graph queries
are introduced for the catalog itself.  Direct in-process calls (no MCP
boundary).

### 3.1 Catalog (initial set)

| ID                          | Backing function                                              | Cost class      | Why in catalog                                         |
|-----------------------------|---------------------------------------------------------------|-----------------|--------------------------------------------------------|
| `search_existing_names`     | `imas_codex.standard_names.search.search_similar_names`       | vector          | Pre-persist consistency check / refine duplicate guard |
| `fetch_existing_name`       | new thin wrapper over `MATCH (sn:StandardName {id:$id})`      | graph (point)   | Resolve a candidate-similar name to full record        |
| `search_dd_paths`           | `imas_codex.graph.dd_search.hybrid_dd_search`                 | vector + graph  | Cross-IDS / cross-domain DD context                    |
| `find_related_dd_paths`     | `imas_codex.standard_names.workers._related_path_neighbours`  | graph           | Cluster/coordinate/unit siblings of a known DD path    |
| `list_grammar_vocabulary`   | `imas_codex.llm.sn_tools._list_grammar_vocabulary`            | graph (cheap)   | Resolve vocab gaps flagged by reviewer                 |
| `search_dd_clusters`        | `imas_codex.graph.dd_search` cluster path (existing)          | vector          | Concept-level grouping discovery                       |
| `get_dd_identifiers`        | `imas_codex.llm.sn_tools` (existing identifier helper)        | graph (cheap)   | Enum/identifier schema lookup                          |

`search_signals` and `fetch_dd_paths` (heavy) are **deferred**: signals
add facility scope the SN pipeline does not need at compose-time, and
`fetch_dd_paths` is point-lookup already covered by enrichment.  They
can be added in a later phase without touching the dispatcher.

### 3.2 Pydantic schemas

Each function has a `*Args` model.  All models live in
`imas_codex/standard_names/fanout/catalog.py`.

```python
class SearchExistingNamesArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    k: int = Field(default=5, ge=1, le=20)

class FetchExistingNameArgs(BaseModel):
    name_id: str = Field(..., min_length=1, max_length=200)

class SearchDDPathsArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=300)
    ids_filter: str | None = Field(default=None, max_length=80)
    physics_domain: str | None = Field(default=None, max_length=80)
    k: int = Field(default=8, ge=1, le=25)

class FindRelatedDDPathsArgs(BaseModel):
    path: str = Field(..., min_length=3, max_length=300)
    max_results: int = Field(default=12, ge=1, le=25)

class ListGrammarVocabularyArgs(BaseModel):
    segment: Literal[
        "physical_base", "subject", "transformation", "component",
        "coordinate", "process", "position", "region", "device",
        "geometric_base",
    ]
    prefix: str | None = Field(default=None, max_length=80)
    k: int = Field(default=30, ge=1, le=100)

class SearchDDClustersArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    scope: Literal["global", "domain", "ids"] | None = None
    k: int = Field(default=8, ge=1, le=20)

class GetDDIdentifiersArgs(BaseModel):
    query: str | None = Field(default=None, max_length=120)
```

The catalog itself is exposed as a `dict[FunctionId, CatalogEntry]`:

```python
@dataclass(frozen=True)
class CatalogEntry:
    fn_id: str
    args_model: type[BaseModel]
    runner: Callable[[BaseModel, GraphClient], Awaitable[FanoutResult]]
    cost_class: Literal["vector", "graph", "hybrid"]
    timeout_s: float        # per-call cap, defaulted from config
    description: str        # one-line, embedded into proposer prompt
```

### 3.3 Output shapes (`FanoutResult`)

Stage B converts each backing-function return into a uniform
`FanoutResult`:

```python
class FanoutHit(BaseModel):
    kind: Literal["standard_name", "dd_path", "cluster",
                  "vocab_token", "identifier"]
    id: str                         # path / sn-id / cluster-id / token
    label: str                      # short human-readable
    score: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
                                    # function-specific extras

class FanoutResult(BaseModel):
    fn_id: str
    args: dict[str, Any]
    ok: bool
    hits: list[FanoutHit] = Field(default_factory=list)
    error: str | None = None
    elapsed_ms: float = 0.0
```

Uniform shape lets the synthesizer prompt template iterate generically;
function-specific extras live in `payload`.

---

## 4. Two-stage call pattern

### Stage A — Query Proposer LLM

- **Input**: candidate name(s) + minimal context (path, description,
  physics_domain, optional `chain_history` slice from refine).
- **Prompt**: `sn/fanout_propose.md` — short, lists the catalog in
  `{fn_id}: {description}: {schema-summary}` form.  System prompt is
  **static** (cached aggressively); user prompt has the dynamic context.
- **Output schema** (`response_model`):

  ```python
  class FanoutPlan(BaseModel):
      queries: list[FanoutCall] = Field(..., max_length=MAX_FAN_DEGREE)

  class FanoutCall(BaseModel):
      fn_id: Literal[*CATALOG_KEYS]   # closed set
      args: dict[str, Any]            # validated post-receipt against
                                      # CATALOG[fn_id].args_model
      reason: str = Field(default="", max_length=160)
                                      # ≤1 sentence rationale (logged,
                                      # not fed back to synthesizer)
  ```

- **Validation**: `acall_llm_structured` enforces `response_model`.  Any
  `args` failing the per-function Pydantic validation is **dropped with
  a warning** (graceful degrade); if all queries are invalid, fan-out
  errors out and falls through to the existing chain.  No agentic retry.
- **Model**: low temperature (0.0–0.2); cheap tier by default
  (Haiku 4.5).  Overridable via `[tool.imas-codex.sn.fanout].proposer-model`.

### Stage B — Function Executor (pure Python, no LLM)

- For each validated `FanoutCall`:
  - Look up `CatalogEntry` by `fn_id`.
  - Spawn task via `asyncio.create_task(runner(args, gc))` wrapped in
    `asyncio.wait_for(..., timeout=entry.timeout_s)`.
- Run all tasks under a single `asyncio.gather(..., return_exceptions=True)`.
- Per-call failures (timeout, graph error) are captured into
  `FanoutResult(ok=False, error=...)` rather than aborting the batch.
- Returns `list[FanoutResult]` in the same order as input.
- Optional: deduplicate hits across results (same `(kind, id)` collapses
  into one entry, max-score wins) before passing to the synthesizer.

### Stage C — Synthesizer LLM

- **Input**: original candidate(s) + `list[FanoutResult]` rendered as a
  compact markdown block (one section per non-empty result).
- **Prompt**: `sn/fanout_synthesize.md`, parameterised by call-site
  (refine vs generate vs docs).  Synthesizer role is to *use* the
  fetched evidence; it does **not** re-call functions.
- **Output**: existing pool-specific schema (e.g. `RefineNameOutput`,
  `GenerateNameOutput`).  No new schema introduced — fan-out is a
  *context augmentation* of the same downstream call.
- **Model**: same model the call-site already uses (e.g. refine uses
  Sonnet 4.6); overridable via `[tool.imas-codex.sn.fanout].synthesizer-model`.

### Stage diagram

```
                ┌────────── Stage A ──────────┐
candidate +     │ Proposer LLM (cheap, low-T) │
 minimal ctx ──▶│  prompt: catalog summary    │── FanoutPlan(queries[≤N])
                └─────────────────────────────┘
                              │ validate against per-fn args_model
                              ▼
                ┌────────── Stage B ──────────┐
                │ asyncio.gather (pure Python)│
                │ per-call timeout, no LLM    │── list[FanoutResult]
                └─────────────────────────────┘
                              │
                              ▼
                ┌────────── Stage C ──────────┐
candidate +     │ Synthesizer LLM (same as    │
results ──────▶ │  call-site model)           │── refined / verified
                └─────────────────────────────┘    candidate
```

---

## 5. Where to plug it in

### 5.1 Plug-in matrix (ranked by expected value)

| Rank | Call-site                  | Trigger                           | Stage A inputs                       | Notes                                       |
|------|----------------------------|-----------------------------------|--------------------------------------|---------------------------------------------|
| 1    | `refine_name`              | review score below threshold      | candidate, reviewer comment(s),       | Highest value: reviewer comment is a strong |
|      |                            | (current refine entry condition)  | chain_history slice                  | query signal; refine is already costly so   |
|      |                            |                                   |                                      | one extra cheap call is a small marginal.   |
| 2    | `generate_name` pre-persist| candidate batch composed,         | candidate name + description         | Adds duplicate-guard before persisting.     |
|      |                            | before grammar round-trip         |                                      | Single fan with `search_existing_names`.    |
| 3    | VocabGap auto-suggest      | review fails with vocab-segment   | segment + offending token            | Single-function fan (just `list_grammar_   |
|      |                            | gap                               |                                      | vocabulary`); thin shim over fan-out.       |
| 4    | `generate_docs`            | doc generation start              | accepted name + cluster siblings     | Style consistency: pull related-path        |
|      |                            |                                   |                                      | docstrings.  Fan with `find_related_dd_     |
|      |                            |                                   |                                      | paths` + `search_existing_names`.           |

The first three can each be a single phase; (4) is a follow-up and may
be deferred indefinitely.

### 5.2 How fan-out composes with existing patterns

| Existing mechanism                                | Interaction                                                                |
|---------------------------------------------------|----------------------------------------------------------------------------|
| Plan 31 / B12 compose retry (`_re_enrich_expanded`) | **Orthogonal**.  B12 retries grammar-failed batches with expanded         |
|                                                    | hybrid-search context — deterministic, no extra LLM call.  Fan-out is the |
|                                                    | *next* line of defence and runs only at the refine pool, not during       |
|                                                    | initial compose.                                                          |
| `chain_history` (refine)                           | **Input**.  Stage A receives a slice of `chain_history` so the proposer   |
|                                                    | can target queries at the recurring failure mode.                          |
| RD-quorum review fan                               | **Different pattern**.  Review fans across *multiple judge models* on the |
|                                                    | same input.  Fan-out fans across *backing functions* on the same model.   |
|                                                    | They can co-exist: a refine cycle may run fan-out then feed the result    |
|                                                    | through the existing review chain.                                         |
| Existing tool-calling prompt (Plan 32 variant C)   | **Replaced** for production paths.  The legacy prompt remains in          |
|                                                    | `prompts/sn/generate_name_dd_tool_calling.md` for research only and is    |
|                                                    | not wired into pools.                                                     |

---

## 6. Prompt design

Two new prompt templates under `imas_codex/llm/prompts/sn/`:

### 6.1 `sn/fanout_propose.md`

- **System** (static, ≤500 tokens): role description + catalog block
  generated from `CATALOG` at module load time:

  ```
  Available functions (you MAY pick up to {MAX_FAN_DEGREE}; you MAY pick zero):
  - search_existing_names(query: str, k: int=5)
      Find existing StandardName nodes similar to a description.
  - fetch_existing_name(name_id: str)
      Full record for one StandardName by id.
  - search_dd_paths(query: str, ids_filter: str?, physics_domain: str?, k: int=8)
      Hybrid search over DD paths (cross-IDS).
  - find_related_dd_paths(path: str, max_results: int=12)
      Cluster/coordinate/unit siblings of a known DD path.
  - list_grammar_vocabulary(segment, prefix?, k=30)
      Allowed tokens for a grammar segment.
  - search_dd_clusters(query: str, scope?, k: int=8)
      Cluster discovery.
  - get_dd_identifiers(query: str?)
      Identifier-schema enums.

  Output JSON: {"queries": [{"fn_id": "...", "args": {...}, "reason": "..."}, ...]}.
  Pick at most {MAX_FAN_DEGREE} queries.  If no query would help, return
  {"queries": []} — that is a valid answer.
  ```

- **User** (dynamic, ≤300 tokens): candidate + context + (refine only)
  reviewer comment + chain_history excerpt.

- **Caching**: the system prompt is identical for all SN fan-out calls
  of a given catalog version — flagged with `cache_control` for
  Anthropic prompt caching where supported.  Catalog version is a
  hash of the catalog dict to invalidate cache on schema change.

### 6.2 `sn/fanout_synthesize.md`

- Two variants per call-site (`refine`, `generate`, `docs`) selected by
  the caller via `render_prompt(f"sn/fanout_synthesize_{site}", ...)`.
  All variants share a common header that explains the fan-out result
  block format.
- Inputs: original candidate + rendered fan-out evidence + (refine only)
  reviewer comment.
- Output schema: existing per-pool output model — no new schema.

### 6.3 Rendered evidence block (Stage B → Stage C)

Compact markdown built by `fanout.render.format_results()`:

```
## Fan-out evidence (3 queries, 0 errors)

### search_existing_names("electron temperature core", k=5)
- electron_temperature_at_magnetic_axis  (score=0.91)  unit=eV  kind=scalar
- electron_temperature                    (score=0.87)  unit=eV  kind=scalar
- electron_temperature_pedestal_top       (score=0.74)  unit=eV  kind=scalar

### find_related_dd_paths("core_profiles/profiles_1d/electrons/temperature")
- core_profiles/profiles_1d/electrons/temperature_validity   (cluster)
- equilibrium/time_slice/profiles_1d/electrons/temperature   (unit-match)
- ...

### list_grammar_vocabulary(segment="position", prefix="magnetic")
- magnetic_axis, separatrix, plasma_boundary, ...
```

Length-bounded: each section truncates at K hits (configurable; default
8) so total evidence stays ≲ 1500 tokens.

---

## 7. Bounds and safety

| Bound                          | Default | Enforced where                                         |
|--------------------------------|---------|--------------------------------------------------------|
| `max_fan_degree`               | 3       | `FanoutPlan.queries` `max_length` + post-validation     |
| `max_depth`                    | 1       | Architecturally — there is no Stage D loop             |
| `function_timeout_s`           | 5.0     | `asyncio.wait_for` per call in Stage B                 |
| `total_timeout_s`              | 12.0    | `asyncio.wait_for` wrapping Stage B `gather`           |
| Fan-out budget reservation     | dynamic | `BudgetLease.reserve(amount=fanout_cost_estimate)`     |
| Per-result hit cap             | 8       | Renderer truncates each `FanoutResult.hits`            |
| Total evidence token cap       | 2 000   | Renderer enforces; truncates with note if exceeded     |

### 7.1 Cost reservation

Fan-out reserves a single `BudgetLease` for the *sum* of expected costs:

```
estimate = (proposer_cost_per_call
            + N * graph_query_amortised_cost
            + synthesizer_extra_cost)
```

The synthesizer is the *replacement* for an LLM call the call-site
would have made anyway, so only the *delta* (extra context tokens) is
charged to fan-out — not the full synthesizer cost.  Keep the
accounting visible: every charge records `phase="sn_fanout_<site>"`
on the `LLMCostEvent` so audits can isolate fan-out spend.

### 7.2 Failure modes

| Failure                                       | Behaviour                                              |
|-----------------------------------------------|--------------------------------------------------------|
| Stage A model rejects schema / returns empty  | Skip fan-out; call Stage C directly with no evidence    |
|                                               | block — equivalent to existing call.  Log `fanout_skip`.|
| Stage A returns invalid `args`                | Drop bad calls, run remaining; if all dropped, skip.    |
| Stage A returns `fn_id` not in catalog        | Drop bad call (the `Literal[*CATALOG_KEYS]` should      |
|                                               | reject this at parse time; defence-in-depth check too). |
| Stage B function raises                       | Recorded as `FanoutResult(ok=False, error=...)`; other  |
|                                               | results pass through.                                   |
| Stage B `total_timeout_s` hit                 | Pending tasks cancelled; partial results pass through.  |
| Stage B all results failed/empty              | Synthesizer runs anyway — no evidence block, just a     |
|                                               | comment "fan-out produced no useful evidence".          |
| Budget reservation insufficient               | Skip fan-out (return None to caller); caller falls      |
|                                               | through to existing chain.  Log `fanout_no_budget`.     |

### 7.3 Idempotency / determinism

- Stage A is low-temperature (≤0.2) and the system prompt is static, so
  query plans are stable across re-runs of the same input within a
  short window.
- Stage B is read-only — function results are deterministic given graph
  state.
- Stage C's existing determinism characteristics are preserved; fan-out
  only mutates the *prompt context*.

---

## 8. Configuration

New section in `pyproject.toml`:

```toml
[tool.imas-codex.sn.fanout]
# Master switch.  Default off until rolled out per call-site.
enabled = false
# Hard cap on Stage A query count.
max-fan-degree = 3
# Hard cap on chain depth.  >1 is opt-in per call-site and not
# implemented in Phase 1.
max-depth = 1
# Per-function timeout.
function-timeout-s = 5.0
# Total Stage B timeout (wraps the gather).
total-timeout-s = 12.0
# Per-result hit cap fed to renderer.
result-hit-cap = 8
# Total evidence token budget rendered into the synthesizer prompt.
evidence-token-cap = 2000
# Stage A (proposer) model.  Cheap tier by default.
proposer-model = "openrouter/anthropic/claude-haiku-4.5"
# Stage A temperature.
proposer-temperature = 0.1
# Stage C (synthesizer) model.  When unset, defaults to the call-site
# model (e.g. refine_name uses [tool.imas-codex.sn-run].model).
# synthesizer-model = "openrouter/anthropic/claude-sonnet-4.6"

# Per-call-site enable flags.  Master `enabled` must also be true.
[tool.imas-codex.sn.fanout.sites]
refine_name      = false   # Phase 1 enables this
generate_name    = false   # Phase 2
vocab_gap        = false   # Phase 2
generate_docs    = false   # Phase 3 (deferred)
```

Loaded in `imas_codex/standard_names/config/` mirroring the existing
`sn-compose` / `sn-run` loaders.  All fields surface as `FanoutSettings`
(Pydantic) consumed by `fanout.dispatcher`.

---

## 9. Module layout

New module tree under `imas_codex/standard_names/fanout/`:

```
fanout/
  __init__.py
  catalog.py          # CATALOG dict + CatalogEntry dataclass + arg models
  runners.py          # async wrappers around existing helpers
  dispatcher.py       # propose() / execute() / run() public API
  render.py           # format_results() — markdown evidence block
  schemas.py          # FanoutPlan, FanoutCall, FanoutResult, FanoutHit
  config.py           # FanoutSettings (Pydantic) + loader
```

Public API (called from `workers.py` refine path in Phase 1):

```python
from imas_codex.standard_names.fanout import (
    FanoutSettings, FanoutResult, run_fanout,
)

async def run_fanout(
    *,
    site: Literal["refine_name", "generate_name", "vocab_gap", "generate_docs"],
    proposer_user_prompt: str,
    synthesizer_renderer: Callable[[list[FanoutResult]], str],
    gc: GraphClient,
    lease: BudgetLease,
    settings: FanoutSettings,
) -> str | None:
    """Returns the rendered evidence block to inject into the
    synthesizer's user prompt, or ``None`` if fan-out was skipped /
    failed (caller falls through)."""
```

Note: `run_fanout` does **not** call the synthesizer itself — it
returns the evidence string and the call-site's existing code performs
the synthesizer call with the augmented prompt.  This keeps the
call-site flow readable and avoids forcing every call-site through the
same synthesizer wrapper.

---

## 10. Phased rollout

### Phase 1 — Refine plug-in (Wave 1)

**Goal**: Land the framework + the refine_name plug-in behind
`fanout.sites.refine_name = false` (default off).  Validate on
`magnetic_field_diagnostics` probe loop (Plan 43 Phase E).

**Files added**:
- `imas_codex/standard_names/fanout/{__init__,catalog,runners,dispatcher,render,schemas,config}.py`
- `imas_codex/llm/prompts/sn/fanout_propose.md`
- `imas_codex/llm/prompts/sn/fanout_synthesize_refine.md`
- `tests/standard_names/fanout/test_dispatcher.py`
- `tests/standard_names/fanout/test_catalog_validation.py`
- `tests/standard_names/fanout/test_render.py`
- `tests/standard_names/fanout/test_dispatcher_integration.py`

**Files modified**:
- `imas_codex/standard_names/workers.py` — refine path picks up
  `run_fanout` when `settings.sites.refine_name=true`.
- `pyproject.toml` — add `[tool.imas-codex.sn.fanout]` section.
- `AGENTS.md` — short paragraph on the structured fan-out pattern.
- `imas_codex/standard_names/config/__init__.py` — export
  `FanoutSettings` loader.

**Acceptance criteria**:
1. With `enabled=false`, refine path is byte-identical to current
   behaviour (golden test).
2. With `enabled=true` + stubbed Stage A returning a fixed plan,
   refine prompt receives a deterministic evidence block.
3. Malformed Stage A output (bad `fn_id`, bad args) is dropped and
   logged; refine still completes.
4. Stage B per-call timeout fires under simulated slow runner.
5. `BudgetLease` reservation accounts for both Stage A and the
   synthesizer-context delta; total spend ≤ reservation in unit
   test.
6. `uv run pytest tests/standard_names/fanout -v` green.
7. `uv run pytest tests/graph/test_schema_compliance.py -v` green
   (no schema changes expected, but verify).
8. `uv run ruff check . && uv run ruff format --check .` clean.

### Phase 2 — Generate-name pre-persist + vocab-gap auto-suggest (Wave 2)

**Goal**: Add the second and third plug-in points.

**Files added**:
- `imas_codex/llm/prompts/sn/fanout_synthesize_generate.md`
- `tests/standard_names/fanout/test_generate_site.py`
- `tests/standard_names/fanout/test_vocab_gap_site.py`

**Files modified**:
- `imas_codex/standard_names/workers.py` — wire generate_name pre-
  persist hook.
- `imas_codex/standard_names/review/...` — vocab-gap shim.
- `pyproject.toml` — flip `sites.generate_name` and `sites.vocab_gap`
  to `true` after probe-loop validation.

**Acceptance criteria** (additional to Phase 1):
1. Generate-name pre-persist run flags duplicate names with
   ≥0.85 cosine similarity in a stub-graph fixture.
2. Vocab-gap shim produces a single-function fan and a deterministic
   suggestion.
3. Probe-loop run on one domain shows refine→accept rate improvement
   ≥10 % vs Phase 1 baseline (numerical target tunable post-RD).

### Phase 3 — Docs fan-out (deferred / opt-in)

**Goal**: Use fan-out to keep doc style consistent across cluster
siblings.  Lowest priority — likely landed only if Phase 1–2 results
warrant it.

**Files added**:
- `imas_codex/llm/prompts/sn/fanout_synthesize_docs.md`
- `tests/standard_names/fanout/test_docs_site.py`

**Files modified**:
- Docs worker (TBD location, currently in `workers.py`).
- `pyproject.toml` — `sites.generate_docs = true`.

---

## 11. Test strategy (per phase)

All test code stubs the Stage A LLM call and (where possible) backing
functions to keep tests hermetic.  Integration tests use the real graph
fixture with stubbed LLM — never the real LLM.

### 11.1 Unit (Phase 1)

| File                                              | Coverage                                                       |
|---------------------------------------------------|----------------------------------------------------------------|
| `test_catalog_validation.py`                      | Per-function `args_model` accept/reject cases; closed-set     |
|                                                   | `fn_id` literal rejects unknown ids.                          |
| `test_dispatcher.py::test_stage_b_parallel`       | Stage B fans 3 stub runners in parallel; verifies              |
|                                                   | `asyncio.gather` ordering and per-call timeout.                |
| `test_dispatcher.py::test_partial_failure`        | One runner raises; others pass through with `ok=True`.         |
| `test_dispatcher.py::test_total_timeout`          | Stage B aborts when `total_timeout_s` hits; partial results    |
|                                                   | returned.                                                      |
| `test_dispatcher.py::test_malformed_plan_drops`   | Stage A returns `fn_id` not in catalog → dropped + logged;     |
|                                                   | does not raise.                                                |
| `test_dispatcher.py::test_all_invalid_skips`      | All Stage A queries invalid → `run_fanout` returns `None`.     |
| `test_dispatcher.py::test_no_budget_skips`        | Insufficient lease → returns `None`; logs `fanout_no_budget`.  |
| `test_render.py::test_truncates_per_hit_cap`      | Renderer truncates `hits` to `result_hit_cap`.                 |
| `test_render.py::test_token_budget`               | Renderer truncates total evidence to `evidence_token_cap`.     |

### 11.2 Integration (Phase 1)

| File                                              | Coverage                                                       |
|---------------------------------------------------|----------------------------------------------------------------|
| `test_dispatcher_integration.py`                  | Real graph fixture (`tests/conftest.py` graph), stubbed Stage  |
|                                                   | A returns a canonical plan; Stage B hits the live graph for    |
|                                                   | `search_existing_names` + `find_related_dd_paths` and asserts  |
|                                                   | the well-known seed nodes are returned.                        |

### 11.3 Cost (Phase 1)

| File                                              | Coverage                                                       |
|---------------------------------------------------|----------------------------------------------------------------|
| `test_dispatcher.py::test_budget_charged`         | After fan-out, `lease.charged` ≤ `lease.reserved`; Stage A     |
|                                                   | charge present with `phase="sn_fanout_refine_name"`.           |

### 11.4 Phase 2/3 incremental tests

- `test_generate_site.py` — duplicate detection in pre-persist site.
- `test_vocab_gap_site.py` — single-function shim returns a vocab
  suggestion.
- `test_docs_site.py` — style-consistency evidence block.

### 11.5 Regression guard

A `tests/standard_names/fanout/test_disabled_is_noop.py` ensures that
with `enabled=false`, the refine path produces the **same** prompts and
calls as the pre-Phase-1 baseline (golden text comparison stored under
`tests/standard_names/fanout/fixtures/refine_baseline.txt`).

---

## 12. Acceptance criteria (overall)

The plan is considered fully delivered when:

1. All Phase 1 acceptance criteria met.
2. Phase 2 enabled in production for at least one full
   `magnetic_field_diagnostics` probe loop with measurable refine
   accept-rate improvement (target ≥10 %).
3. Fan-out spend per accepted name is bounded and visible in the
   `LLMCost` graph view (filter `phase LIKE 'sn_fanout_%'`).
4. Documentation updates landed (see §13).
5. No agentic-loop fallback path exists in the code (review checklist
   item).

---

## 13. Documentation updates required

| File                                         | Change                                                       |
|----------------------------------------------|--------------------------------------------------------------|
| `AGENTS.md`                                  | New § "Structured fan-out" under SN pipeline section,        |
|                                              | linking to this plan.                                        |
| `docs/architecture/sn-pipeline.md` (if exists) | Append a sub-section showing the (A→B→C) diagram.           |
| `imas_codex/standard_names/fanout/README.md` | New — explains how to add a new function to the catalog and  |
|                                              | how to add a new plug-in site.                               |
| `plans/features/standard-names/39-...md`     | This plan; mark Phase 1 status checkboxes as work proceeds.  |

---

## 14. What this plan does NOT do (re-stated for the reviewer)

- ❌ No free agentic looping.
- ❌ No LLM-driven function chaining beyond depth 1 (depth 2 is an
  optional future opt-in for refine and is **not** implemented in any
  phase of this plan).
- ❌ No runtime function generation — catalog is fixed at module load.
- ❌ No multi-model debate (orthogonal to RD-quorum review).
- ❌ No new graph queries or schema changes — fan-out wraps existing
  helpers only.
- ❌ Does not touch the B12 grammar-retry path in compose.

---

## 15. Open questions for RD review

1. **Catalog scope**: should `search_signals` and `fetch_dd_paths` be
   in the initial catalog despite being deferred?  Including them now
   costs only a few lines but expands the proposer prompt.
2. **Proposer model tier**: Haiku 4.5 default — should we A/B against
   `gpt-5-mini` for cross-vendor structured-output reliability?
3. **`max_depth = 2` opt-in**: refine is the obvious candidate (fan
   once → if reviewer comment changed → fan again).  Defer entirely or
   include as Phase 2.5?
4. **Cache invalidation**: catalog version hashed into the system-
   prompt cache key — confirm Anthropic/OpenAI cache APIs accept this.
5. **Cost estimation for reservation**: how to size `estimate` before
   knowing the answer?  Proposal: percentile from a 200-call calibration
   run, hard-cap at $0.05 per fan-out by default.

---

*End of plan.*
