# Structured fan-out catalog

This package implements plan 39 — Structured Fan-Out for the SN Compose Pipeline.

## Adding a function to the catalog

The catalog is **closed**: only entries declared in `catalog.py` are
recognised.  Adding a function is a four-step plan-revision step:

1. Add a Pydantic variant to `schemas.py`.  All bounds (`k`,
   `max_results`, string lengths) MUST be enforced at parse time via
   `Field(...)`.  Do not use `dict[str, Any]`; the discriminated union
   on `fn_id` is the contract.

2. Add an async runner to `runners.py`.  Wrap the sync helper with
   `asyncio.to_thread` and `asyncio.wait_for(timeout_s)`.  Accept
   `gc: GraphClient` and `scope: FanoutScope` as keyword arguments;
   **never** instantiate `GraphClient` inside a runner (plan 39 §10.1).

3. Register the entry in `catalog.py`'s `CATALOG` dict.

4. Add a section to the proposer prompt body
   (`imas_codex/llm/prompts/sn/fanout_propose.md`).  Editing the
   prompt body flips `CATALOG_VERSION`; the literal first line of the
   rendered system prompt becomes the new hash.

## Scope-injection rule

The proposer LLM picks **only** `fn_id` + intent arg (`query` /
`path`) + small `k` / `max_results`.  Known scope (`physics_domain`,
`ids_filter`, `dd_version`) is injected at runtime from caller context
via `FanoutScope` — the LLM never sees those fields.

## GraphClient lifecycle

The refine worker opens **one** `GraphClient` per refine cycle and
passes it to `run_fanout(..., gc=gc, ...)`.  `run_fanout` propagates
that same `gc` to every runner.  Runners are forbidden from opening
their own `GraphClient`.

## Default-off semantics

When `enabled=False` (the default), `run_fanout` is a true no-op:

- returns `""` immediately
- writes no `Fanout` graph node (S5)
- emits no metrics
- does not call the proposer LLM

The `{{ fanout_evidence }}` placeholder in the call-site's prompt
collapses to an empty line so the prompt is byte-identical to baseline
(plan 39 §6.3).
