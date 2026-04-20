# Feature 33 — CLI option harmonization

**Status:** In progress (Phase 1 + 2 landed, Phase 3/4 deferred)
**Owner:** imas-codex maintainers
**Research:** `plans/research/cli-option-matrix.md` (777-line audit — source of truth)

## Motivation

`imas-codex`'s CLI has grown organically across six command groups
(`discover`, `sn`, `embed`, `llm`, `graph`, `hpc`) with 50+ subcommands.
The research audit in `plans/research/cli-option-matrix.md` catalogues
the drift: the same concept is spelled `--ids` / `--domain` /
`--physics-domain`, `--limit` / `-n` / `--path-limit` / `--signal-limit` /
`--max-pages`, `-f` is overloaded for `--force`, `--focus`, `--foreground`,
and `--follow` etc. This makes muscle-memory transfer between commands
impossible and `--help` noisy.

The goal of this feature is a **single canonical vocabulary** (defined
in the research doc Part 6) with old spellings kept as aliases so
nothing breaks.

## Scope

Four phases, landed incrementally. Each phase has a dedicated commit
message style — `feat(sn):`, `refactor(sn):`, `refactor(cli):`, etc.

### Phase 1 — Research + enum foundations (landed)

- `plans/research/cli-option-matrix.md` — full audit + canonical
  vocabulary + `sn` consolidation proposal (10 visible subcommands)
  + per-commit sequencing plan.
- `imas_codex/schemas/standard_name.yaml` — added
  `StandardNameReviewMode` enum + `review_mode` attribute.

### Phase 2 — `sn review --name-only` (landed)

Pairs with `sn generate --name-only` (landed in parallel by the
extraction-batching work):

- `--name-only` flag on `sn review` selects a 4-dimension rubric
  (grammar / semantic / convention / completeness, /80) instead of
  the full 6-dimension review (/120).
- New `sn/review_name_only.md` prompt (sibling of `review.md`).
- `StandardNameQualityScoreNameOnly` +
  `StandardNameQualityReviewNameOnly[Batch]` Pydantic models.
- `StandardNameReviewState.name_only` + pipeline wiring through
  `_review_single_batch` → `_match_reviews_to_entries`.
- Downgrade guard in `extract_review_worker`: a `--name-only` run
  without `--force` will not overwrite a prior `review_mode == "full"`
  entry. `review_mode` is stamped onto each scored entry so the guard
  has state to work with on the next run.
- Unit + prompt-rendering tests in
  `tests/standard_names/test_review_name_only.py`.

### Phase 3 — Option rename on non-generate commands (expanded: landed)

`refactor(sn): canonicalise --physics-domain with --domain alias`
added `--physics-domain` (canonical) + `--domain` (alias) on:

- `sn benchmark`
- `sn publish`
- `sn review`
- `sn enrich` (multiple=True)
- `sn rotate` (required=True)

`--ids` kept distinct — it filters by IDS name, not physics domain
(see research doc §6.2).

**Cross-group expansion (landed in this PR):**

`refactor(cli): canonicalise --physics-domain and --ids aliases across
imas/sn` — same alias pattern (canonical first, old name preserved
through the click decorator) applied outside the `sn` group:

- `imas map run` — `--domain/-d` → `--physics-domain` (alias `--domain`,
  short `-d` preserved). Semantic match: domains filter physics
  domains, same as `sn` canonical.
- `imas dd build` — `--ids-filter` → `--ids` (alias `--ids-filter`).
  Semantic match: `--ids` is already the canonical IDS-name filter
  used by `imas dd search`, `sn review/enrich/benchmark/publish`.
- `sn gaps` — `--export {table,yaml}` → `--format {table,yaml}`
  (alias `--export`). Canonical for output formatting per research
  doc §3.
- `sn reconcile` — `--source-type {dd,signals}` → `--source`
  (alias `--source-type`). Canonical `--source` matches `sn generate`,
  `sn clear`, `sn seed`, `sn benchmark` which already use `--source`.

**Audit result for remaining top-level groups** (see research doc §1.3
— no code changes required):

- `graph` — `--facility`, `--force`, `--dry-run` already canonical.
  `-v` = `--version` inside `graph fetch/pull/tags` is a local tag
  selector, no `--verbose` competition. Keep as-is.
- `embed`, `llm` — no facility concept, so `-f` = `--foreground` /
  `--follow` is locally unambiguous. `--host/--port/--log-level`
  are infra-standard. Keep as-is.
- `tunnel` — `--neo4j/--embed/--llm/--timeout/--all` are all tunnel-
  specific. No collisions with canonical vocab. Keep as-is.
- `config`, `release`, `tools`, `serve`, `credentials`, `host`,
  `facilities`, `hpc` — already use `--force`, `--dry-run`, `--json`,
  `--verbose` canonically. No renames needed.

**Deferred (not yet implemented):**

- Freeing `-f` on facility-aware commands (`sn clear`, `discover
  paths/signals/code/wiki/documents`) so `-f` can consistently mean
  `--facility`. Requires coordinated touch of 6+ files + per-test
  verification — tracked separately.
- Harmonising `--limit` / `-n` vs `--path-limit` / `--signal-limit` /
  `--max-pages` on `discover` subcommands. Requires adding aliases
  on every subcommand — tracked separately.
- Harmonising `sn generate`'s option names to the canonical vocabulary.
  **Explicitly deferred until after the extraction-batching PR merges**
  — that agent owns `sn generate` right now and simultaneous edits
  would conflict.

### Phase 4 — Subcommand consolidation (deferred)

Per research doc §7, reduce `sn` from 15 visible subcommands to 10:

- Fold `sn seed` into `sn import --source {isn|west|all}` (hide `seed`).
- Hide `sn resolve-links` and `sn reconcile` as internal debug commands
  (still callable, no longer in `sn --help`).
- Keep `sn rotate`, `sn benchmark`, and the other eight as visible
  top-level commands.

This is deferred because it's user-visible and requires a migration
note + deprecation period. It should be bundled with a broader
CLI cleanup after Phase 3's alias additions settle.

## Test strategy

- `tests/cli/` — existing test suite (~138 passing) covers flag
  parsing for every subcommand. Any rename keeps tests green because
  the destination variable name is preserved and the old spelling
  stays as a click alias.
- `tests/standard_names/test_review_name_only.py` — new coverage for
  `--name-only` mode (score arithmetic, `review_mode` stamping,
  prompt content).
- No integration test needed for the downgrade guard — covered by a
  unit test on `_match_reviews_to_entries` + a documented invariant in
  the extract worker.

## Non-goals

- Click group restructuring (e.g. merging `discover` and `sn`) —
  out of scope, would break every user's scripts.
- Changing the default values of any option — backward-compat only.
- Rewriting prompts — `review.md` is untouched; `review_name_only.md`
  is a new sibling.

## References

- Research artifact: `plans/research/cli-option-matrix.md`
- Parallel work: `plans/research/extraction-batching-prompt-ab.md`
  (owns `sn generate` right now — do not touch its signature until
  that PR lands).
