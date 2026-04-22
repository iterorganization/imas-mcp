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

## rc22 Design Decisions

These decisions were made during the rc22 infrastructure pivot (Plan 39) to address
bootstrap failures found after Wave 5 of Plan 38.

### Decision rc22-A: Split `reviewer_score` into name + docs axes

**Choice:** Introduce `reviewer_score_name` / `reviewed_name_at` and `reviewer_score_docs` /
`reviewed_docs_at` as independent review axes alongside the existing canonical `reviewer_score`.
Docs review is gated on `reviewed_name_at IS NOT NULL`.

**Alternatives considered:** Single `reviewer_score` updated per phase; unified score with
a `review_mode` tag.

**Rationale:** The rc21 bootstrap produced 134 names but zero reviewed names — the review
phase was silently claiming zero eligible names because the status filter matched no names.
Rather than just fix the bug, the review model was rearchitected: name quality (grammar,
convention, semantic correctness) and documentation quality (description prose, physics
accuracy) are orthogonal concerns with different rubrics and different cost profiles.
Separating them into independent axes enables cheap name-only sweeps (`--target names`,
4 dimensions / 80 points) that gate the more expensive docs pass. The `reviewed_name_at IS
NULL` gate ensures a reviewer LLM never grades documentation for a name that has not yet
passed name review — preventing docs scores from polluting quality signals on bad names.
The canonical `reviewer_score` (full 6-dim pass) remains the export gate; the split axes
are additive provenance, not replacements.

### Decision rc22-B: Error siblings minted deterministically via uncertainty grammar

**Choice:** `_error_upper`, `_error_lower`, and `_error_index` companions are constructed
entirely in code using the ISN uncertainty modifier grammar rule (`model='deterministic:dd_error_modifier'`).
No LLM call is made. All review fields are pre-populated at mint time (`reviewer_score_name=1.0`,
`reviewed_name_at=now`, `reviewer_score_docs=1.0`, `reviewed_docs_at=now`).

**Alternatives considered:** LLM compose with explicit error-field prompt, manual curation,
skip error fields entirely.

**Rationale:** Error companion names are fully determined by their parent: `electron_temperature`
→ `electron_temperature_error_upper`. The uncertainty modifier grammar in ISN v0.7.0rc23
provides a closed, deterministic construction rule. An LLM would add latency, cost, and
non-determinism for a transformation that has no creative degrees of freedom. Pre-populating
review fields bypasses the review queue (there is nothing to review) and prevents error
siblings from cluttering the per-domain eligible-name count used by the review gate invariant.

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
| rc22 | Added review-split fields: `reviewer_score_name`, `reviewed_name_at`, `reviewer_score_docs`, `reviewed_docs_at`. Renamed relationship `SOURCE_DD_PATH` → `FROM_DD_PATH`. Added `SN_SOURCE_CATEGORIES` `coordinate` member. Error siblings minted deterministically (`model='deterministic:dd_error_modifier'`) via ISN v0.7.0rc23 uncertainty grammar. |
