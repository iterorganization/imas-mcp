# Extraction batching & prompt architecture — research report

> **Scope:** Three-workstream investigation of SN generation pipeline
> architectural weaknesses, conducted against the current `main`
> (post plan 30 / plan 31 WS-A).
> **Status:** research-only. Workstream 2(a) is the recommended
> immediate implementation. Workstreams 1 and 3 are deferred to
> plan 32 (`plans/features/standard-names/32-extraction-prompt-overhaul.md`).

---

## 1. Executive summary

The SN generation pipeline is **throughput-bound, not filter-bound**.
Empirical measurement against the live graph (682 valid SNs, 25.2%
coverage, 6 385 eligible DD paths) shows:

| Metric                                           | Value                    |
| ------------------------------------------------ | ------------------------ |
| Pre-split concept groups (cluster × unit)        | 1 523                    |
| Current `ExtractionBatch` count (after chunking) | **1 566**                |
| Single-item batches (payload size 1)             | **543 (34.7 %)**         |
| Mean batch size                                  | **4.08 items**           |
| Batches if grouped (physics_domain × unit) @ 50  | **475**                  |
| Mean batch size (name-only)                      | **13.4 items**           |
| LLM-call reduction vs. current                   | **≈ 69.7 %**             |
| Unclustered eligible paths                       | 528 (8.3 %)              |

At a cached-prompt cost of ~10 k system tokens per call, collapsing
the long tail of singleton batches alone saves roughly **one dollar
per hundred paths** even before prompt-caching hits. For a fresh
bootstrap of the remaining ≈ 4 780 uncovered paths, this is roughly
**1 091 avoided LLM calls**.

Recommendation: **implement Workstream 2(a)** — a "name-only" batch
mode that groups by `(physics_domain × unit)` in bins of 50, wired
into `sn generate --name-only`. It is fully additive (no changes to
the default pipeline), measurable in isolation, and unlocks the
transport desert (1 489 extracted-but-not-composed paths) in hours
rather than days. Workstreams 1 (rejection audit) and 3 (static vs.
tool-calling prompt A/B) are lower ROI and should proceed as plan 32
after 2(a) lands.

---

## 2. Workstream 1 — rejection audit (filter-bound hypothesis)

### 2.1 Correction to the original brief

The brief framed WS-1 as "audit `classifier.py` rejection rules".
Post plan 30, the classifier is **minimal**: only three deterministic
skips remain (`STR_*` names, `core_instant_changes/*` IDS, `*_error_*`
children). All substantive rejection has moved into
`IMASNode.node_category`, assigned during DD enrichment.

The eligibility filter is now:

```python
# imas_codex/core/node_categories.py
SN_SOURCE_CATEGORIES = frozenset({"quantity", "geometry"})
```

Five categories are currently **rejected** for SN extraction:

| category         | count  | intent                                   |
| ---------------- | ------ | ---------------------------------------- |
| `error`          | 27 577 | `*_error_upper/lower/index` companions   |
| `metadata`       | 8 824  | `ids_properties/*`, `code/*`, timestamps |
| `structural`     | 3 243  | container/AoS nodes, not leaf data       |
| `representation` | 1 817  | FEM/GGD interpolation coefficients       |
| `coordinate`     | 821    | axis arrays (`time`, `rho_tor_norm`, …)  |
| `fit_artifact`   | 239    | `*_fit/chi_squared`, `*_fit/time_*`      |

Sampled 20 random transport-domain rejections (see §2.2). All were
correctly excluded.

### 2.2 Spot-audit of 20 random transport rejections

All 20 fall into two classes:

- **`representation`** — GGD interpolation coefficient slots such as
  `plasma_transport/model/ggd/neutral/momentum/flux/radial_coefficients`.
  These describe a numerical representation of a quantity, not the
  quantity itself, and are already modelled by the associated
  `values` sibling node.
- **`fit_artifact`** — `*_fit/chi_squared`, `*_fit/time_measurement_width`.
  These are diagnostics of a fit procedure, not physics observables.
- **`coordinate`** — `time`, `rho_tor_norm` axis arrays. These have
  dedicated coordinate handling elsewhere.

**Conclusion:** no evidence of systematic over-rejection. The
remaining transport-coverage gap (1 489 paths `status='extracted'`
without an SN) is **budget-bound**, not filter-bound.

### 2.3 What a proper WS-1 audit looks like

If pursued in plan 32 Phase 1, it should:

1. Sample 100 random paths across all rejected categories (not just
   transport) and hand-classify each as correct or incorrect.
2. Report false-negative rate per category.
3. Produce a CSV of suspected-misclassified paths for DD review.
4. Recommend category schema adjustments if > 5 % false-negative.

This is **not** blocking for rc14 generation.

---

## 3. Workstream 2 — batching strategy (RECOMMENDED)

### 3.1 Empirical distribution (live graph, commit HEAD)

```text
Eligible DD paths (quantity|geometry, non-STR)        6 385
Pre-split groups (cluster × unit)                     1 523
Post-split ExtractionBatch count                      1 566
Singleton batches (size 1)                              543  (34.7 %)
Mean batch size                                        4.08
Median batch size                                         3
Max batch size (capped)                                  25
```

The `max_batch_size=25` ceiling in
`enrichment.group_by_concept_and_unit()` is essentially never binding:
even the largest (cluster × unit) groups are dominated by 2–6 items.
The singleton tail is driven by narrow clusters where a single unit
occurs once (e.g., `rho_tor_norm` in a scalar context, or a solitary
`T` toroidal-field variant).

### 3.2 Proposed "name-only" grouping

Group by `(physics_domain, unit)` instead of `(grouping_cluster, unit)`.
Bin at `name_only_batch_size` (default **50**).

Simulation against the same 6 385 paths:

| grouping key                        | batches | mean | singletons | median |
| ----------------------------------- | ------- | ---- | ---------- | ------ |
| `(cluster, unit)` @25 (current)     | 1 566   | 4.08 | 543        | 3      |
| `(physics_domain, unit)` @50 (new)  | **475** | 13.4 | **121**    | 22     |
| `(physics_domain, unit)` @100       |   312   | 20.5 |   108      | 38     |
| `(ids_name, physics_domain, unit)`  |   731   |  8.7 |   234      | 6      |

At 50-bin, LLM calls drop **69.7 %** with only a modest loss in
per-item cluster context (the prompt still receives the full
per-item structural/unit/COCOS context; it just loses the
sibling-path listing that was primarily useful for cross-IDS
synonymy).

### 3.3 Why (physics_domain × unit) is the right key

- **Coherence across IDSs.** Domain is coarser than cluster; a batch
  now contains, e.g., *all transport particle-flux quantities in
  `m^-2.s^-1` across `plasma_profiles`, `plasma_transport`,
  `core_transport`*. The LLM naturally identifies the sub-grouping
  (electron vs. ion vs. neutral, parallel vs. radial) and composes
  consistent names, which is the primary review complaint in
  `plans/research/standard-names/12-full-graph-review.md`.
- **Unit safety.** Keying on unit preserves the invariant that every
  batch candidate has the same SI unit, so the LLM never has to
  reconcile e.g., `T` vs. `V.s/Wb`.
- **Cache friendliness.** Domain+unit is a low-cardinality key
  (~ 150 bins across the whole DD). Anthropic prompt-cache hit
  rates improve because system+domain-context prefixes are stable
  across sequential batches in the same bin.
- **Avoids the long tail.** No (cluster × unit) group is ever
  singleton-only anymore unless the *entire domain-unit* has one
  member, which in practice applies to fewer than 30 domain-unit
  pairs (mostly alpha-lifecycle paths).

### 3.4 Implementation sketch (WS-2a scope)

1. **Config:** add `name-only-batch-size = 50` under
   `[tool.imas-codex.sn-generate]` in `pyproject.toml`.
2. **State:** `StandardNameBuildState.name_only: bool = False`.
3. **Grouping:** new `group_for_name_only()` in
   `imas_codex/standard_names/enrichment.py` keyed on
   `(physics_domain, unit)`, producing `ExtractionBatch(mode="name_only", …)`.
4. **Source plumbing:** `extract_dd_candidates()` in
   `imas_codex/standard_names/sources/dd.py` branches on `name_only`.
5. **Prompt:** new
   `imas_codex/llm/prompts/sn/compose_dd_name_only.md` — lean user
   prompt that omits per-item cluster siblings, COCOS block, and
   reviewer-feedback themes; adds an explicit
   "identify natural sub-groups before naming" instruction.
6. **Worker branch:** in `compose_worker` (line ≈ 1010), when
   `state.name_only`, render the new user prompt and skip the L2
   IDS-prefetch + L4 reviewer-feedback enrichment. Keep L6 grammar
   retry and L7 opus revision — they're per-candidate and cheap.
7. **CLI:** `sn generate --name-only [--name-only-batch-size N]`.
8. **Telemetry:** aggregate `(batch_size, items, cost, tokens_in,
   tokens_out)` per batch, print a summary at rotation end.

**Non-goal:** the default (non-`--name-only`) path is unchanged.
Existing review/enrichment flows still apply; `--name-only` just
gets a candidate named-and-classified faster, deferring the deep
per-item enrichment to a subsequent review pass.

### 3.5 Risks and mitigations

| risk                                              | mitigation                                                    |
| ------------------------------------------------- | ------------------------------------------------------------- |
| LLM truncates at N > ~30 items                    | Start at N = 50; fallback `chunk()` still caps the list.      |
| Lost cluster-sibling context degrades names       | L7 opus revision still sees cluster context per candidate.    |
| Name-only names pass review with weak description | Flag `source_kind='dd_name_only'`; review gate checks it.     |
| Prompt-cache thrash from new user-prompt variant  | New prompt is static; cache still warm after first batch.     |

### 3.6 Measurement plan for post-merge validation

After implementation, run two back-to-back 15-minute rotations:
one with `--name-only --budget 3.0`, one default `--budget 3.0`.
Report:

- `names_composed` per dollar.
- `review_pass_rate` (quarantine %) for each cohort.
- `median_batch_size`, `median_batch_latency_s`.
- `cache_read_tokens / cache_write_tokens`.

Success gate: name-only achieves ≥ 3× throughput at
≤ 1.2× quarantine rate. If quarantine regresses further, raise
`L6`/`L7` thresholds or fall back to default.

---

## 4. Workstream 3 — prompt architecture (static vs. tool-calling)

### 4.1 Current architecture (static, cache-optimised)

- **System prompt** `sn/compose_system.md` (≈ 7 800 words ≈ 10 k
  tokens) is sent verbatim on every call. Anthropic prompt caching
  hits it after the first call in a rotation, dropping marginal
  cost ~ 90 %.
- **User prompt** `sn/compose_dd.md` (≈ 1 900 words base + ≈ 100
  tokens per item) embeds batch-specific context: candidates,
  cluster siblings, cross-IDS paths, reviewer feedback, rate hints,
  COCOS guidance.
- **Partials** under `imas_codex/llm/prompts/shared/sn/_*.md`
  (~ 4 700 words total) encapsulate grammar, vocabulary, exemplars,
  COCOS rules — re-rendered into every prompt via Jinja.

### 4.2 Tool-calling alternative (speculative)

The adjacent ISN package (`~/Code/imas-standard-names`) exposes
MCP tools: `list_vocabulary`, `grammar_help`, `fetch_schema`,
`search_standard_names`, `validate_name`, etc. An alternative
compose architecture would:

1. Send a **short** system prompt describing the task and
   available tools (~1–2 k tokens).
2. Let the LLM **pull** grammar, vocabulary, and exemplars on
   demand via tool calls.
3. Finalize with a structured `compose_standard_name` tool call.

### 4.3 Why tool-calling is NOT obviously a win here

- **Cache defeat.** Tool-calling inserts model-generated content
  between system and final response, which breaks the prefix-cache
  optimisation; marginal cost per batch goes **up** unless the
  round-trip saves > 10 k tokens on average (unlikely).
- **Latency.** Each tool call adds a round trip; batch latency
  doubles or triples.
- **Non-determinism.** The LLM may pull different vocabulary slices
  for semantically identical batches, reducing run-to-run
  reproducibility — a regression for the current review workflow
  which depends on stable regenerations.
- **Dev cost.** Requires an MCP client inside `compose_worker`,
  auth plumbing, and retry logic; meaningful engineering investment.

### 4.4 Hybrid variant

A middle ground that warrants evaluation: keep the static system
prompt but trim by ~50 % (remove sections the LLM rarely cites),
and add **only one** tool — `fetch_exemplar_by_physics_domain` —
to pull the 3–5 most relevant exemplars on demand instead of
statically embedding 15 per prompt. This preserves caching while
reducing context bloat.

### 4.5 Proposed A/B harness (plan 32 Phase 2)

- **Sample:** 20 representative DD paths — 5 equilibrium, 5
  magnetics, 5 transport, 5 core_profiles. Include 5 already-in-graph
  SNs as regressions.
- **Arms:**
  1. `static` — current architecture.
  2. `static-lean` — trimmed system prompt, single exemplar tool.
  3. `tool-calling` — full MCP-driven composition.
- **Scoring:** reviewer (opus) scores each candidate 0–1 on
  correctness, grammar-compliance, and description quality. Track
  wall-clock, cost, and cache-read ratio.
- **Success gate:** `static-lean` or `tool-calling` wins on quality
  **and** cost. If neither wins on both, keep `static`.

### 4.6 Recommendation

Ship Workstream 2(a) first. Only invest in prompt architecture
experimentation (WS-3) if the name-only pass reveals a residual
quality plateau that cannot be closed by L7 opus revision. In that
case, start with the `static-lean` variant (low-risk trim) before
considering tool-calling.

---

## 5. ROI summary

| workstream              | estimated effort | estimated quality gain | estimated cost savings |
| ----------------------- | ---------------- | ---------------------- | ---------------------- |
| **WS-2a name-only**     | 1–2 days         | neutral (flagged)      | ≈ 70 % of bootstrap    |
| WS-1 category audit     | 1 day            | marginal               | zero                   |
| WS-3 static-lean prompt | 2–3 days         | unknown                | 10–30 %                |
| WS-3 tool-calling       | 1–2 weeks        | unknown                | likely negative        |

**Go:** WS-2a now. WS-1 + WS-3 defer to plan 32 pending results.
