# Standard Names — Design Decisions

Key architectural choices for the DD-enriched standard name pipeline
(Plan 20, schema v0.5.0).

## Decision Log

| # | Decision | Choice | Alternatives Considered | Rationale |
|---|----------|--------|------------------------|-----------|
| 1 | Unit source | DD (`HAS_UNIT` relationship), never LLM | LLM-generated units | LLMs hallucinate units (e.g. outputting `keV` when DD says `eV`). The DD is the authoritative source; injecting after LLM output eliminates a class of errors. |
| 2 | Grouping scope | Global `(cluster × unit)` | Per-IDS `(IDS × cluster × unit)` | Same physics concept across IDSs (e.g. `electron_temperature` in `core_profiles`, `core_sources`, `equilibrium`) must receive the same standard name. Global batching lets the LLM see all instances together. |
| 3 | Path classifier | 11 deterministic rules | LLM-based classification | Rules are fast (µs), free ($0), auditable, and reproducible. LLM classification would add latency, cost, and non-determinism for a task that is fully pattern-matchable. |
| 4 | Primary cluster selection | IDS-scope > domain-scope > global-scope priority | Random selection, majority vote | More specific scopes carry more informative cluster labels. IDS-scope clusters like `"equilibrium/boundary_geometry"` are more descriptive than global `"spatial_coordinates"`. Deterministic priority avoids instability. |
| 5 | Conflict handling | Filter conflicting entries, continue pipeline | Fail-fast on first conflict | Partial progress is better than atomic failure. A 500-path extraction shouldn't abort because 3 entries have a unit mismatch. Conflicts are logged and reported in `ConsolidationResult`. |
| 6 | Consolidation timing | After VALIDATE, before PERSIST | After PERSIST (reconciliation), during COMPOSE | Validating individual names first catches grammar errors cheaply. Consolidating before PERSIST avoids writing conflicting data to the graph and having to reconcile later. |
| 7 | Prompt caching | Static-first system message (~6k tokens) | Interleaved static/dynamic, single combined prompt | OpenRouter caches the common prefix of system messages across calls. Putting all static content first (grammar, vocabulary, examples) maximizes cache hits. Measured 32% cache rate across benchmark runs. |
| 8 | Review phase | Optional (`--skip-review`) | Mandatory, removed entirely | Review adds quality but doubles LLM cost. Making it optional lets users choose cost vs quality. Default: enabled for production runs. |
| 9 | Coalesce writes | `coalesce(new, existing)` for build path | Direct SET overwrite | Coalesce preserves catalog-imported data when `sn generate` re-runs. A subsequent mint never overwrites an accepted name's fields. The import path uses direct SET because the catalog is authoritative. |
| 10 | Dedup strategy | Keep longest documentation, union paths/tags | Keep first, keep highest confidence | Documentation length correlates with quality — the LLM that produced more context usually had richer batch input. Union of paths ensures no source mapping is lost. |
| 11 | Vocabulary gaps | Track as `vocab_gap` status + detail JSON | Silently skip, fail the candidate | Tracking gaps provides signal for grammar vocabulary extension. The candidate is still persisted with a `vocab_gap` status so it can be manually reviewed and the vocabulary updated. |
| 12 | Batch size limit | 25 paths per batch | Unlimited, smaller batches | 25 balances LLM context utilization (~4k tokens per batch) against coherence. Larger batches risk context overflow; smaller batches lose cross-path context and increase per-call overhead. |
| 13 | Concurrency model | `asyncio.Semaphore(5)` per COMPOSE phase | Sequential, unlimited parallel | 5 concurrent LLM calls balances throughput against rate limits and cost. Sequential is too slow for 100+ batches; unlimited risks rate-limit errors and uncontrolled spending. |
| 14 | Loop domain rotation | Stale-first one-domain-per-turn | Fair-share budget split across all domains | Fair-share created starvation when one domain's paths were expensive. Stale-first ensures every domain gets a turn before any domain gets two turns; combined with the $0.75 `MIN_VIABLE_TURN` floor it prevents wasting the tail of a budget on a partial turn. |
| 15 | RD-quorum review | Sequential blind pair + optional escalator | Parallel twin (simultaneous independent); single reviewer | See "RD-quorum (p39)" section below. |
| 16 | Budget leasing | Reserve-then-charge via `BudgetLease` | Optimistic accounting (charge post-hoc) | Optimistic accounting allowed multi-cycle review and large compose batches to overshoot the cost limit by up to $2. Reserving upfront guarantees worst-case cost is covered before a cycle starts. |

## Anti-Patterns (from prompt)

These are explicitly warned against in `compose_dd.md`:

1. **Inventing units** — Unit comes from DD, not the LLM
2. **Duplicating existing names** — Must check existing names list first
3. **Using camelCase** — Names are strictly `snake_case`
4. **Mixing `physical_base` and `geometric_base`** — Mutually exclusive
5. **Omitting `subject` for species-specific paths** — `electron_temperature` not `temperature`
6. **Over-qualifying** — Don't add segments that don't narrow meaning
7. **Coordinate names** — Skip pure coordinate grids (`rho_tor_norm` is not a standard name)
8. **Metadata names** — Skip timestamps, validity flags, indices

## Schema Version History

| Version | Changes |
|---------|---------|
| v0.4.0 | Initial StandardName node with core fields |
| v0.5.0 | Added review provenance (`reviewer_model`, `reviewer_score`, `reviewer_scores`, `reviewer_comments`, `reviewed_at`, `review_tier`), `vocab_gap_detail`, `catalog_commit_sha`. Extended `StandardNameReviewStatus` with `reviewed`, `validation_failed`, `vocab_gap`, `skipped`. |
| rc22 C3 | Axis-split review storage: axis-specific columns (`reviewer_score_name`, `reviewer_score_docs`, etc.) added. |
| p39-2 | **BREAKING** — removed shared reviewer slots (`reviewer_score`, `reviewer_scores`, `reviewer_comments`, `reviewer_comments_per_dim`, `reviewer_verdict`, `reviewer_model`, `reviewed_at`, `review_mode`). Only axis-specific columns remain. `StandardNameReviewMode.full` removed; values are `names` and `docs`. |
| p39-3 | Per-axis model chains in config (`[sn.review.names].models`, `[sn.review.docs].models`). Removed single-model `[sn.review]` primary/secondary config. |
| p39-4 | RD-quorum `Review` node fields: `review_axis`, `cycle_index`, `review_group_id`, `resolution_role`, `resolution_method`. `ReviewResolutionRole` and `ReviewResolutionMethod` enums added. |

## RD-quorum (p39)

**Why we moved from parallel-twin secondary to sequential RD-quorum:**

The pre-p39 "parallel twin" ran two reviewers simultaneously on the same batch, then compared
their scores post-hoc. This had a fundamental flaw: **the secondary score could not influence
the primary** — they ran independently and disagreements were only visible after the fact.
The primary's verdict was always used as the canonical result; the secondary was effectively
advisory noise.

Plan 39 preserves the independence signal for disagreement detection (cycles 0 and 1 are still
BLIND — no cross-contamination) while adding a genuine arbitration path:

- **Why sequential blindness?** Cycles 0 and 1 must not see each other's output so that their
  disagreement is a real signal, not a coordination artefact. The `{% if prior_reviews %}` prompt
  block is guarded — cycles 0 and 1 never receive it.
- **Why an escalator that sees both critiques?** The escalator (cycle 2) can only add value if it
  has access to both prior critiques. A blind third reviewer would just be more noise. By making
  it context-aware we enable substantive arbitration: the escalator knows *why* cycles 0 and 1
  disagreed and can resolve the specific dimension in dispute.
- **Why per-dimension tolerance (0.15) rather than aggregate?** Aggregate disagreement can mask a
  single bad dimension being swamped by agreement on others. Per-dimension tolerance gives finer
  resolution — a single egregiously mis-scored dimension triggers escalation even if overall scores
  are close.
- **Why hybrid batching (cycles 0+1 full-batch, cycle 2 per-item mini-batch)?** Cycles 0 and 1
  process all items together — this is the common case and costs 2× the single-reviewer price
  (same as the old parallel twin). Cycle 2 only runs on the ~15% of items that are disputed, using
  per-item mini-batches so the escalator gets focused context. The typical 3-cycle run costs ~15%
  more than a 2-cycle run, not 50%.
- **Why budget leasing?** The old optimistic accounting let multi-cycle review overshoot the cost
  limit because the secondary model's cost was not reserved before cycle 1 started. Reserving
  `batch_cost × num_models × 1.3` upfront guarantees worst-case cost is funded before any cycle
  begins. `BudgetLease.charge()` raises `BudgetExceeded` on overshoot — the invariant
  `pool + sum(reserved) + spent == total` holds at all times.
