# Plan 29 — SN Architectural Pivot: Generate/Enrich Split + ISN Grammar Upgrade

> **Archival note: this plan supersedes [plan 28](./28-sn-greenfield-pipeline.md).**
> Plan 28's "greenfield" framing is absorbed here; we adopt the clean-slate
> rotation (PHASE D) but re-scope the pipeline along two orthogonal axes
> identified in R1 (grammar), R2 (corpus audit), R3 (architecture).

Research sources (all citations below reference these):

- **R1** — `R1-grammar-findings.md` — ISN grammar fitness & F1–F8 proposals.
- **R2** — `R2-semantic-audit-raw.md` — Corpus audit of 141 valid SNs.
- **R3** — `R3-enrich-architecture.md` — Split-pipeline architecture.

---

## 1. Executive summary

- **Pivot A — split generate/enrich.** `sn generate` becomes names-only; a
  new standalone `sn enrich` (SELECT → CONTEXTUALISE → DOCUMENT → VALIDATE →
  PERSIST) writes descriptions/documentation/tags/links in a second pass
  with graph-aware context (R3 §B).
- **Pivot B — ISN grammar upgrade.** Fix the `Transformation` enum
  desync (R1 F1, zero-rename), add Fourier/mode-number decomposition
  segment (R1 F3), and introduce `over_` for region-scoped integrals
  (R1 F4). Released from `~/Code/iter-standard-names` as an rc bump.
- **Blocker — embedding backfill.** `n.embedding` is NULL on all 141
  StandardName nodes; `standard_name_desc_embedding` index is empty
  (R2 Blocker #1). PERSIST phase MUST embed descriptions; generate's
  embed pass must embed names. Without this, CONTEXTUALISE's
  nearby-SN surface degrades to Jaccard only.
- **Schema evolution.** New review-status values `named → enriched →
  reviewable` (R3 §C.1) + new enrichment provenance fields; new
  `REFERENCES` relationship between StandardNames (R3 §B.3).
- **Corpus cleanup.** Collapse the 23-way `wave_absorbed_power_*`
  family (R2 A.1) via qualifier axes (`species`, `population`,
  `toroidal_mode`, `flux_surface_average`) on StandardName. Fix ≥10
  misdomained SNs (R2 F.4) and 7 DD-leaked name bugs (R2 D.1/D.2).

## 2. Non-goals

- Not re-architecting `sn review` / `sn publish` — post-enrichment review
  flow is unchanged (R3 §C.2).
- Not introducing a tensor (rank ≥2) grammar segment (R1 D4 deferred).
- Not implementing the `basis_frame` segment (R1 F5 deferred — nice-to-have).
- Not splitting `Component` into axis vs physics-projection (R1 F6 deferred).
- Not introducing new physics_domain values — we move misdomained SNs to
  existing domains only (R2 F.4). A `kinetic_profiles` domain is future work.
- Not migrating live downstream consumers of `review_status='drafted'`; a
  one-release alias handles back-compat (R3 §C.1).
- Not touching `sn_generate`'s existing extract/compose/validate/consolidate
  worker topology — only the contract of what compose writes changes.

## 3. Architecture decisions (ADRs)

### ADR-1 — Split `sn generate` into names-only + standalone `sn enrich`
- **Context.** Compose currently asks the LLM for docs + name in one call
  but `--name-only` discards docs afterwards (R3 §A.2) — tokens wasted.
  Enrich exists only as an inline asyncio prototype in `cli/sn.py:1878–2110`
  with no claim discipline, no pipeline harness, no graph-aware context
  (R3 §A.3).
- **Decision.** `sn generate` always produces names-only (removes
  `--name-only` flag); a new `sn enrich` pipeline runs as a separate
  CLI verb with its own `WorkerSpec` DAG and `StandardNameEnrichState`
  (R3 §B.1).
- **Consequences.** Two distinct LLM calls, two distinct prompts, two
  cost budgets. Graph-primary coordination works for both. Prompt cache
  hit rate improves because the static system prompt is invariant per
  pipeline.

### ADR-2 — New lifecycle states `named → enriched → reviewable`
- **Context.** `drafted` currently covers "grammar-valid, no docs" AND
  "grammar-valid, has docs from compose" — conflated (R3 §C.1).
- **Decision.** Add `named` (output of generate), `enriched` (DOCUMENT
  finished), `reviewable` (VALIDATE passed). Keep `drafted` as a
  deprecated alias for one release; migration script maps
  `drafted → named` + backfills `named_at` (R3 §C.1, §F task 8).
- **Consequences.** SELECT phase filter `review_status='named'` is
  trivial. `sn review` unchanged (still reads `reviewable`/`accepted`).

### ADR-3 — New ISN grammar segments: `decomposition`, region `over_` template
- **Context.** Fourier / mode-number axes produce 12+ quarantined names
  (R1 A2 table) and regions are awkwardly squeezed into `positions.yml`
  (R1 A4, F4). Zero impact on 141 existing SNs.
- **Decision.** Land R1 F3 (new `decomposition` segment with Fourier +
  mode tokens) and R1 F4 (split `regions.yml`; new `over_` template)
  in ISN `0.7.0rc9` (or later). Defer F5 (basis_frame), F6
  (component split), F7 (canonical pattern emitter), F8 (length
  ceiling) as out-of-scope for this plan (§8).
- **Consequences.** ISN minor-bump consumed via the existing release
  CLI + `uv sync`. Unlocks MHD/wave Fourier naming and resolves the
  `ohmic_energy_at_halo_region_*` quarantine class.

### ADR-4 — Fix `Transformation` enum drift (R1 F1)
- **Context.** `transformations.yml:37-40` declares
  `real_part_of|imaginary_part_of|magnitude_of|phase_of` but the
  generated enum `model_types.py:208-226` and `TRANSFORMATION_TOKENS`
  (`constants.py:202-221`) omit them (R1 B1). `compose_standard_name`
  with `magnitude_of` raises `ValueError`. Parse of
  `magnitude_of_magnetic_field` silently falls through to open
  physical_base — a future collision hazard.
- **Decision.** Regenerate the enum + constants; add a CI `--check`
  step to pin codegen forever (R1 Validation strategy 1). Zero
  renames to existing SNs (tokens currently unparseable).
- **Consequences.** Unblocks all complex-valued / magnitude pathways.
  Closes the existing R1 B1 quarantine class.

### ADR-5 — Qualifier axes on StandardName (collapse wave-power 23-way)
- **Context.** R2 A.1 documents a 23-SN family conflating 5 orthogonal
  axes (species × fast/thermal × state × integration × toroidal_mode).
  Jaccard ≥0.9 pairs are rampant. This is a schema gap, not a grammar
  gap — grammar cannot carry per-instance qualifiers.
- **Decision.** Add StandardName attributes `species` (enum:
  `electrons|ions|total|neutrals`), `population`
  (`thermal|fast|runaway|null`), `toroidal_mode_qualifier`
  (`per_n_tor|summed|null`), `flux_surface_averaged` (bool),
  `integration_kind` (`global|density|inside_flux_surface|null`).
  Collapse the 23-SN family into 3 canonical SNs per R2 A.1
  recommendation (`wave_absorbed_power`, `wave_absorbed_power_density`,
  `wave_absorbed_power_inside_flux_surface`). Assign qualifiers per
  instance.
- **Consequences.** Enrichment must surface qualifier values to the
  LLM as structured context, not name tokens. Review tools need a
  qualifier-aware display. Dedup query density drops sharply.

### ADR-6 — Inter-name links as `(SN)-[:REFERENCES]->(SN)` + `sn.links` property
- **Context.** R3 §A.3 notes enrich currently writes `sn.links` but
  no edge; R2 §E identifies 8 high-value clusters and ≥28 SNs sharing
  `*_component_of_*` template suitable for edges.
- **Decision.** DOCUMENT writes `sn.links` (list of `name:xxx`).
  PERSIST materialises a `REFERENCES` edge when the target exists
  (R3 §B.3). `link_status` state machine unchanged
  (`graph_ops.py:33`).
- **Consequences.** Cheap neighbour traversal for future MCP tools.
  `link_retry_count` + `failed` state handle the
  `links → REFERENCES` inconsistency window (R3 §H.4).

### ADR-8 — Materialise ISN grammar as graph (ISN owns schema; codex imports)
- **Context.** Users want to filter/search SNs by grammar structure
  (e.g. "all SNs with substance=electron", "co-occurrence of transformations").
  Current state: SNs carry flat string slots (`substance`, `transformation`, …)
  that support `WHERE` but not traversal; vocabulary tokens have no graph
  identity; ISN YAML is the only place grammar structure exists.
- **Decision.** Add a read-only graph mirror of the ISN grammar, versioned
  per ISN release and refreshed only by a dedicated sync CLI. The LinkML
  schema describing Grammar\* nodes **lives in the ISN package** next to
  `entry_schema.json` (ISN already owns `schemas/generate.py`,
  `schemas/validate.py`, and `grammar_codegen/`). imas-codex `imports:`
  that schema via package-resource URI and extends `StandardName` with
  a `HAS_SEGMENT` relationship to the imported `GrammarToken` class.
  - ISN adds `imas_standard_names/schemas/grammar_graph.yaml` +
    `imas_standard_names/graph/sync.py` (driver-agnostic). Optional
    extra `imas-standard-names[graph]` pulls the `neo4j` driver.
  - imas-codex `standard_name.yaml` uses LinkML `imports:` to pull in
    Grammar\* classes; `uv run build-models --force` resolves both.
  - CLI: `imas-codex graph sync-isn-grammar` wraps
    `isn.graph.sync.sync_grammar(graph_client)`. Auto-invoked in the
    release-CLI dep-bump step whenever `imas-standard-names` changes.
  - `persist_worker` uses `isn.graph.sync.segment_edge_specs(parsed)`
    to MERGE `HAS_SEGMENT` edges after name embedding.
  - Drift is eliminated structurally: the schema ships with the YAML
    it describes. ISN CI validates schema ↔ YAML at release time;
    imas-codex CI validates installed ISN version vs active
    `ISNGrammarVersion` in graph.
- **Consequences.**
  - Single source of truth by construction — no mirrored vocab lists.
  - CONTEXTUALISE worker (PHASE C.2) replaces regex-based sibling
    lookup with graph traversal `(sn)-[:HAS_SEGMENT]->(gt)<-[:HAS_SEGMENT]-(sibling)`;
    higher-signal LLM prompts.
  - Analytics (vocab coverage, co-occurrence, orphan tokens) become
    2-line Cypher aggregations.
  - Releases become coupled: ISN grammar changes that alter the graph
    schema force an imas-codex rebuild — correct behaviour (breaking
    schema change = breaking dep).
  - Additive only; zero impact on existing 141 SNs. PHASE E gated to
    land after PHASE B (needs schema 0.6.0 plus stable ISN rc from
    PHASE A).

### ADR-7 — Embedding backfill policy
- **Context.** R2 Blocker #1: all 141 SNs have NULL embedding;
  `standard_name_desc_embedding` index is empty. Any enrichment
  pipeline that calls `search_similar_sns_with_full_docs`
  (`search.py:64`) returns nothing — silent degradation to Jaccard.
- **Decision.**
  - `sn generate` PERSIST phase MUST embed the SN *name* (used to
    find nearby SNs by name-similarity even before docs exist).
  - `sn enrich` PERSIST phase MUST embed the SN *description* after
    DOCUMENT writes it — inline `embed_descriptions_batch` call
    (R3 §B.2 PERSIST, §H.2).
  - Embedding is a precondition of the `reviewable` state: VALIDATE
    fails the SN if embedding still NULL after PERSIST retry.
- **Consequences.** Semantic search over SNs works immediately after
  each pipeline; CONTEXTUALISE nearby-SN surface is deterministic.

---

## 4. Phased work breakdown

Phases are dependency-ordered. `model` column is the owning model for
primary execution; review uses opus-4.7 unless noted.

### PHASE A — ISN grammar fixes (repo: `~/Code/iter-standard-names`, model: `opus-4.7`)

Exit criterion: ISN releases an rc tag; imas-codex `uv sync` pulls
it; `parse/compose` round-trip tests pass on 141 existing SNs.

| # | Scope | File refs | Impact |
|---|---|---|---|
| A.1 | Regenerate `Transformation` enum (R1 F1). Add CI `--check` step. | `grammar_codegen/generate`, `grammar/model_types.py:208`, `grammar/constants.py:202` | 0 renames; unlocks `magnitude_of_*` |
| A.2 | Add `decomposition` segment + vocab (R1 F3). | new `grammar/vocabularies/decomposition.yml`; `specification.yml`; `model.py` exclusivity | 0 renames |
| A.3 | Split `regions.yml` from `positions.yml`; add `over_` template (R1 F4). | `positions.yml`, new `regions.yml`, `specification.yml:209-226`, `support.py` | ~5–15 renames (grep for `_at_halo_region_*`) |
| A.4 | Release new ISN rc via release CLI; tail CI; bump `imas-standard-names` version in `pyproject.toml`; `uv sync`. | imas-codex `pyproject.toml`, `uv.lock` | — |

### PHASE B — imas-codex schema & generate refactor (model: `opus-4.6`)

Exit criterion: `sn generate` produces names-only; all 141 existing SNs
migrated `drafted → named`; `test_schema_compliance` green; name-embeddings
populated.

| # | Scope | File refs |
|---|---|---|
| B.1 | Schema bump to `0.6.0`: new review_status enum values (`named`, `enriched`, `reviewable`); new qualifier fields (ADR-5); enrichment provenance fields (`enrich_model`, `enrich_cost`, `enrich_tokens`, `enrich_validated_at`, `enrich_validation_issues`, `named_at`); `REFERENCES` relationship. Run `uv run build-models --force`. | `imas_codex/schemas/standard_name.yaml`, `imas_codex/graph/models.py` (generated) |
| B.2 | Remove `--name-only` flag + all branches on `state.name_only`. Always names-only in compose. Update `compose_system.md` + `compose_user.md` to drop doc-field asks. `compose_worker` sets `review_status='named'` + `named_at`. | `cli/sn.py:108-116, 140, 309-310, 344`; `standard_names/state.py:48`; `standard_names/workers.py:1145-1160, 1326-1412, 1547`; `llm/prompts/sn/compose_system.md` |
| B.3 | Embedding backfill: extend `persist_worker` to embed the *name* via `embed_names_batch` in addition to description; add VALIDATE precondition that enrich requires non-NULL name embedding. | `standard_names/workers.py:1788-1923`; `embeddings/sn.py` |
| B.4 | Migration script `scripts/migrate_review_status_0_6.py` — `drafted → named` + backfill `named_at=coalesce(generated_at, datetime())`. Run once against live graph; idempotent. | new file |
| B.5 | Data cleanup: (a) fix 7 DD-leaked extractor bugs (R2 D.1 table); (b) repair 10 DD paths sourcing ≥2 SNs (R2 D.2 table); (c) move ≥10 misdomained SNs (R2 F.4 table); (d) rename 2 tautological names (R2 §B). Scripted via Cypher. | new `scripts/sn_corpus_repair_v1.py` |

### PHASE C — imas-codex standalone `sn enrich` (model: `opus-4.6`)

Exit criterion: `sn enrich --dry-run` succeeds on a seed of 10 SNs;
integration test `test_generate_enrich_roundtrip` green; names traverse
`named → enriched → reviewable`; ≥3 REFERENCES edges per SN on average
(success criterion 7.b).

| # | Scope | File refs |
|---|---|---|
| C.1 | New `imas_codex/standard_names/enrich_pipeline.py::run_sn_enrich_engine` + `enrich_workers.py` with SELECT / CONTEXTUALISE / DOCUMENT / VALIDATE / PERSIST; `StandardNameEnrichState(DiscoveryStateBase)` + per-phase `WorkerStats`. | new files, pattern from `pipeline.py:27-106` |
| C.2 | CONTEXTUALISE projects: (i) full DD path docs + IDS description (no 200-char truncation) via `_DD_CONTEXT_QUERY` / `_CROSS_IDS_QUERY` reuse, (ii) top-5 nearby SNs via `search_similar_sns_with_full_docs` (`search.py:64`), (iii) up-to-10 physics_domain siblings at `review_status IN ['enriched','accepted']`, (iv) COCOS + unit via `HAS_UNIT`. `asyncio.Semaphore(8)` bounded concurrency. | `standard_names/enrich_workers.py`, `graph_ops.py` |
| C.3 | PERSIST merges `(sn)-[:REFERENCES]->(target:StandardName {id:x})` per `name:x` link when target exists; recomputes `link_status` via `_compute_link_status`. | `graph_ops.py:33, 326, 1518-1564`; `enrich_workers.py` |
| C.4 | Rewrite `llm/prompts/sn/enrich_system.md` (static, cacheable, role + style-guide include + output schema + hard constraints) and `enrich_user.md` (per-batch context from §D.2 of R3). New include `llm/prompts/sn/_enrich_style_guide.md` with LaTeX + US-spelling rules. | existing prompt files + new include |
| C.5 | Extend `StandardNameEnrichItem` (`models.py:255-280`) with `cross_reference_rationale`, `documentation_excerpt` fields + `field_validator` rejecting `dd:` / URL prefixes in `links`. Wrapper `StandardNameEnrichBatch` unchanged. ISN `create_standard_name_entry(data, name_only=False)` used in VALIDATE, not as LLM schema. | `standard_names/models.py` |
| C.6 | Replace old enrich prototype entirely. Delete inline asyncio loop in `sn_enrich`; new handler uses `run_sn_enrich_engine` + progress display + `run_discovery_engine` harness. | `cli/sn.py:1878-2110` |
| C.7 | CLI flags per R3 §E.1: `--ids`, `--physics-domain`, `--status`, `--name` (repeatable), `--from-model`, `--force`, `-c/--cost-limit`, `--batch-size`, `--limit`, `--dry-run`, `--enrich-model`. Add `sn-enrich` model key to `[tool.imas-codex.models]` in `pyproject.toml` (defaults to same as `sn-generate`). | `cli/sn.py`, `pyproject.toml` |
| C.8 | Tests per R3 §G: unit (`test_enrich_workers_unit.py`, `test_enrich_prompt.py`, `test_enrich_models.py`), graph integration (`test_enrich_integration.py`), roundtrip (`test_generate_enrich_roundtrip.py`), claim-safety (`test_enrich_claims.py`). | new test files |

### PHASE D — Clean slate rotation + validation (execution: `sonnet-4.6`, review: `opus-4.7`)

Exit criterion: full 7-domain coverage; ≥3 REFERENCES/SN avg; zero
Transformation-enum quarantines; reviewer sign-off A/B vs prior 141.

| # | Scope | Notes |
|---|---|---|
| D.1 | Clear all StandardNames + StandardNameSources. Cypher: `MATCH (n:StandardName) DETACH DELETE n; MATCH (s:StandardNameSource) DETACH DELETE s`. Verify embed index reports empty. | Destructive; requires confirmation prompt |
| D.2 | `sn generate` rotation across all 7 physics_domains (`equilibrium`, `auxiliary_heating`, `waves`, `magnetohydrodynamics`, `edge_plasma_physics`, `turbulence`, `core_plasma_physics`); `--cost-limit 2.0` each. Sonnet-4.6 execution. | ~$14 budget |
| D.3 | **Senior review (opus-4.7)** of generated names: A/B quality against the prior 141. Metrics: Jaccard-cluster density (success 7.a), grammar-valid rate, domain-balance. | Gate before PHASE D.4 |
| D.4 | `sn enrich` rotation across all 7 physics_domains; `--cost-limit 2.0` each. Sonnet-4.6 execution. | ~$14 budget |
| D.5 | **Final review (opus-4.7)**: quality assessment, link density (≥3 REFERENCES/SN avg), cross-name coherence, US-spelling compliance, LaTeX rendering sanity. | Sign-off gate |
| D.6 | Documentation updates per §9. Move plan 28 to `plans/features/standard-names/completed/` with an archival header noting supersession by plan 29. | — |

### PHASE E — ISN grammar as graph (ISN owns schema; imas-codex imports)

Exit criterion: `imas-codex graph sync-isn-grammar` idempotent; full
grammar graph present; every `named|enriched|reviewable` SN has
`HAS_SEGMENT` edges matching `parse_standard_name(sn.id)`; drift CI
green in BOTH repos.

**Ownership boundary** (revised per follow-up): ISN owns the LinkML
schema for Grammar\* node types, the sync logic, and the parser→edge
mapping. imas-codex's schema `imports:` the ISN schema and wires
`StandardName.HAS_SEGMENT` to the imported `GrammarToken` class. This
eliminates scope drift structurally — the schema ships with the
grammar YAML it describes.

| # | Scope | Repo |
|---|---|---|
| E.1 | Design spike (opus-4.7): finalise cross-repo schema import mechanics, sync CLI contract, drift semantics, failure modes when a parsed SN token is missing from graph. Write to `files/E-grammar-graph-design.md`. | session |
| E.2 | ISN: add `imas_standard_names/schemas/grammar_graph.yaml` (LinkML) declaring `ISNGrammarVersion`, `GrammarSegment`, `GrammarToken`, `GrammarTemplate` classes + `DEFINES`, `HAS_TOKEN`, `NEXT`, `USES_TEMPLATE` relationships. Register as `package_data`. Extend `schemas/generate.py` + `schemas/validate.py` to emit+check this schema alongside `entry_schema.json`. CI test asserts YAML vocab tokens ↔ schema-derived Pydantic models stay in sync. | ISN |
| E.3 | ISN: add `imas_standard_names/graph/sync.py` exposing `sync_grammar(graph_client, *, active_version)` and `segment_edge_specs(parsed)→list[(segment, token_id, position)]`. Driver-agnostic (duck-typed `query(cypher, **params)` interface). Unit tests. Ship behind optional extra `imas-standard-names[graph]` with `neo4j` driver pin. | ISN |
| E.4 | ISN: release rc (release CLI) bundling E.2 + E.3. | ISN |
| E.5 | imas-codex: bump `imas-standard-names[graph]` dep. Extend `imas_codex/schemas/standard_name.yaml` with `imports:` of the ISN schema via package-resource URI; add `HAS_SEGMENT` relationship on `StandardName` with `range: GrammarToken`. Run `uv run build-models --force`; verify cross-schema resolution works. | imas-codex |
| E.6 | imas-codex: CLI verb `imas-codex graph sync-isn-grammar` — thin wrapper around `isn.graph.sync.sync_grammar`. Release-CLI hook auto-runs on any `imas-standard-names` version change. | imas-codex |
| E.7 | imas-codex: extend PHASE B `persist_worker` (B.3) to call `isn.graph.sync.segment_edge_specs` and MERGE `HAS_SEGMENT {position}` edges after name embed. Fail fast with "run graph sync-isn-grammar" error if any target token missing (drift check at write time). | imas-codex |
| E.8 | imas-codex: `tests/graph/test_grammar_sync.py` (drift: installed ISN version == `ISNGrammarVersion{active:true}`), `test_grammar_segment_edges.py` (every `named\|enriched\|reviewable` SN has edges matching `parse_standard_name`). | imas-codex |

---

## 5. Todos (SQL-ready)

### Phase A — ISN grammar (iter-standard-names repo)

| id | title | model | deps |
|---|---|---|---|
| `a1-regen-transformation-enum` | Regenerate Transformation enum from YAML; add CI --check gate | opus-4.7 | — |
| `a2-add-decomposition-segment` | Add `decomposition` segment + vocab for Fourier/mode numbers | opus-4.7 | a1-regen-transformation-enum |
| `a3-split-regions-over-template` | Split `regions.yml`; add `over_` segment template | opus-4.7 | a1-regen-transformation-enum |
| `a4-release-isn-rc` | Cut ISN rc via release CLI; bump imas-codex dep; uv sync | opus-4.7 | a2-add-decomposition-segment, a3-split-regions-over-template |

### Phase B — schema & generate refactor (imas-codex)

| id | title | model | deps |
|---|---|---|---|
| `b1-schema-bump-0-6` | standard_name.yaml 0.6.0: new statuses, qualifier fields, enrich provenance, REFERENCES rel; build-models --force | opus-4.6 | a4-release-isn-rc |
| `b2-remove-name-only-flag` | Remove `--name-only` flag + all branches; compose writes review_status='named' | opus-4.6 | b1-schema-bump-0-6 |
| `b3-embed-names-in-persist` | persist_worker embeds SN name; VALIDATE precondition on non-NULL name embedding | opus-4.6 | b2-remove-name-only-flag |
| `b4-migrate-drafted-to-named` | scripts/migrate_review_status_0_6.py; idempotent; run once | opus-4.6 | b1-schema-bump-0-6 |
| `b5-corpus-repair-v1` | scripts/sn_corpus_repair_v1.py: extractor bugs, DD-path collisions, misdomained SNs, tautological renames | opus-4.6 | b1-schema-bump-0-6 |

### Phase C — standalone sn enrich (imas-codex)

| id | title | model | deps |
|---|---|---|---|
| `c1-enrich-pipeline-skeleton` | enrich_pipeline.py + enrich_workers.py skeleton with 5 WorkerSpecs; StandardNameEnrichState | opus-4.6 | b1-schema-bump-0-6, b3-embed-names-in-persist |
| `c2-contextualise-worker` | CONTEXTUALISE: DD docs, nearby SNs (vector), domain siblings, COCOS/unit | opus-4.6 | c1-enrich-pipeline-skeleton |
| `c3-document-worker-llm` | DOCUMENT: batched acall_llm_structured with sn-enrich model, budget tracking | opus-4.6 | c2-contextualise-worker, c5-enrich-prompts |
| `c4-validate-and-persist-worker` | VALIDATE (ISN parse + US-spelling + link-format) + PERSIST (REFERENCES edges + description embed) | opus-4.6 | c3-document-worker-llm |
| `c5-enrich-prompts` | Rewrite enrich_system.md + enrich_user.md; new _enrich_style_guide.md; extend StandardNameEnrichItem model | opus-4.6 | b1-schema-bump-0-6 |
| `c6-replace-enrich-prototype` | Delete inline asyncio enrich in cli/sn.py; wire run_sn_enrich_engine | opus-4.6 | c4-validate-and-persist-worker |
| `c7-enrich-cli-flags` | CLI options per R3 §E.1; add sn-enrich model key in pyproject.toml | opus-4.6 | c6-replace-enrich-prototype |
| `c8-enrich-tests` | Unit, integration, roundtrip, claim-safety tests per R3 §G | opus-4.6 | c7-enrich-cli-flags |

### Phase D — clean slate rotation

| id | title | model | deps |
|---|---|---|---|
| `d1-clear-sn-graph` | DETACH DELETE StandardName + StandardNameSource; verify embed index empty | sonnet-4.6 | c8-enrich-tests, b4-migrate-drafted-to-named, b5-corpus-repair-v1 |
| `d2-generate-rotation-7-domains` | `sn generate` across 7 physics_domains at $2 cap each | sonnet-4.6 | d1-clear-sn-graph |
| `d3-senior-review-names` | opus-4.7 A/B review of generated names vs prior 141; gate | opus-4.7 | d2-generate-rotation-7-domains |
| `d4-enrich-rotation-7-domains` | `sn enrich` across 7 physics_domains at $2 cap each | sonnet-4.6 | d3-senior-review-names |
| `d5-final-quality-review` | opus-4.7 final review: link density, coherence, US-spelling, LaTeX | opus-4.7 | d4-enrich-rotation-7-domains |
| `d6-docs-update-archive-plan-28` | AGENTS.md SN section, plans/README, move plan 28 to completed/ | opus-4.6 | d5-final-quality-review |

### Phase E — ISN grammar as graph (split: ISN + imas-codex)

| id | title | model | repo | deps |
|---|---|---|---|---|
| `e1-design-spike` | Opus-4.7 design spike: cross-repo schema import mechanics, sync contract, drift semantics; output `files/E-grammar-graph-design.md` | opus-4.7 | session | — |
| `e2-isn-grammar-graph-schema` | ISN: add `schemas/grammar_graph.yaml` (LinkML); extend `schemas/generate.py`+`validate.py`; CI test YAML↔schema | opus-4.6 | ISN | e1-design-spike |
| `e3-isn-graph-sync-module` | ISN: add `graph/sync.py` (`sync_grammar`, `segment_edge_specs`); optional extra `[graph]`; unit tests | opus-4.6 | ISN | e2-isn-grammar-graph-schema |
| `e4-isn-release-rc` | ISN: release rc bundling E.2+E.3 via ISN release CLI; tail CI | sonnet-4.6 | ISN | e3-isn-graph-sync-module |
| `e5-codex-schema-import` | imas-codex: bump ISN dep, `imports:` ISN schema in `standard_name.yaml`, add `HAS_SEGMENT` rel, rebuild models | opus-4.6 | imas-codex | e4-isn-release-rc, b1-schema-bump-0-6 |
| `e6-codex-sync-cli` | imas-codex: `graph sync-isn-grammar` CLI + release-CLI hook | opus-4.6 | imas-codex | e5-codex-schema-import |
| `e7-codex-persist-segment-edges` | imas-codex: extend `persist_worker` to MERGE `HAS_SEGMENT` via ISN helper; fail-fast on drift | opus-4.6 | imas-codex | e6-codex-sync-cli, b3-persist-worker-status-machine |
| `e8-codex-grammar-tests` | imas-codex: drift test + segment-edge integration test | opus-4.6 | imas-codex | e7-codex-persist-segment-edges |

---

## 6. Risks & mitigations

| # | Risk | Source | Mitigation |
|---|---|---|---|
| R1 | Embedding-NULL blocker breaks CONTEXTUALISE nearby-SN surface | R2 Blocker #1 | ADR-7: PERSIST embeds descriptions inline; b3 task embeds names; VALIDATE precondition enforces non-NULL |
| R2 | ISN grammar changes force rename of existing 141 SNs | R1 F4 (~5–15 renames for `at_*_region`) | PHASE D deletes all existing SNs anyway; grammar changes land *before* D.2 rotation so new corpus starts clean |
| R3 | Live-graph schema migration on 0.5 → 0.6 | R3 §H.1 | `scripts/migrate_review_status_0_6.py` idempotent; run once; `drafted` alias kept for one release |
| R4 | Prompt cache hit rate degrades from per-batch siblings/nearby context | R3 §H.3 | Static-first: siblings/nearby go in **user** prompt; system prompt (style guide + schema + role) is invariant and fully cached |
| R5 | Inter-repo release cycle (ISN rc) adds ~15–20min latency | PHASE A.4 | A.1 can ship independently (zero-rename) and unblock B/C while A.2/A.3 bake; sequence task deps so B.1 depends on a4 but c1 only needs b1 |
| R6 | Concurrent enrich + generate corrupt claims | R3 §A.3 | Reuse `claim_token` across phases; SELECT claims only `review_status='named'`, DOCUMENT only `'enriched'` — disjoint claim pools |
| R7 | 23-SN wave-power collapse loses information | R2 A.1 | Qualifier fields (ADR-5) carry every axis structurally; corpus-repair script preserves source_paths |

---

## 7. Success criteria

| # | Metric | Target | Measurement |
|---|---|---|---|
| 7.a | Jaccard duplicate cluster density | ≥30% reduction vs prior 141 corpus | Offline Jaccard scan pre/post; count J≥0.75 pairs per 100 SNs |
| 7.b | REFERENCES edges per enriched SN | ≥3 average | `MATCH (sn:StandardName)-[r:REFERENCES]->() RETURN avg(count(r))` |
| 7.c | Transformation-enum quarantines | zero | Post-A.1 scan of quarantine log for `magnitude_of_*`, `real_part_of_*`, `imaginary_part_of_*`, `phase_of_*` |
| 7.d | physics_domain coverage | all 7 domains populated | `MATCH (sn:StandardName {review_status:'reviewable'}) RETURN sn.physics_domain, count(*) ORDER BY 1` |
| 7.e | SN name embeddings populated | 100% of `named` SNs | `MATCH (sn:StandardName) WHERE sn.review_status IN ['named','enriched','reviewable'] AND sn.embedding IS NULL RETURN count(sn)` = 0 |
| 7.f | ISN grammar round-trip on new corpus | 100% parse+compose round-trip | `tests/test_grammar_round_trip.py` green on full post-D.4 corpus |
| 7.g | Cost | ≤ $30 end-to-end for PHASE D | Sum of `state.total_cost` across 14 runs (7 gen + 7 enrich) |
| 7.h | COCOS coverage | ≥35 SNs with non-null `cocos_transformation_type` | R2 F.3 target; enrich prompt instructs to populate |

## 8. Out-of-scope / future work

- R1 F5 `basis_frame` segment (lab/plasma/guiding-centre) — defer.
- R1 F6 split `Component` into axis vs physics_projection — defer.
- R1 F7 canonical-pattern emitter for `transformation` + `binary_operator` — defer.
- R1 F8 length ceiling / tautology soft warnings — defer to audit layer.
- New physics_domain `kinetic_profiles` (R2 F.4) — for now reuse
  `core_plasma_physics`; create a dedicated domain in a later plan.
- Review UI surfacing qualifier axes (ADR-5) — assumes text display;
  dedicated review UI work tracked separately.
- Formal operator calculus with argument slots for `derivative_of`,
  `normalized`, `maximum_of` (R1 C2, E3) — defer.
- Tensor (rank ≥2) grammar support (R1 C4).
- Replacing `name:` links with typed references (`related_to:`,
  `computed_from:`) — future enrichment.

## 9. Documentation updates (per AGENTS.md checklist)

- `AGENTS.md` — SN pipeline section: update to describe split
  generate/enrich topology; list new review_status values; link to
  this plan.
- `plans/features/standard-names/README.md` (if present) — add plan
  29 entry; mark plan 28 as superseded.
- `plans/features/standard-names/28-sn-greenfield-pipeline.md` —
  prepend archival banner: "**Superseded by [plan 29](./29-architectural-pivot.md)**"
  and move to `completed/` folder in task `d6-*`.
- `imas_codex/schemas/standard_name.yaml` — attribute `description`
  fields must document the new enum values + qualifier fields +
  enrich provenance fields.
- `agents/schema-reference.md` is auto-generated; no manual edit.
  Will be regenerated via `uv sync` after `build-models --force`.
- Inline CLI help strings for `sn generate` (remove `--name-only`
  docs) and `sn enrich` (new flags).
- Prompt catalogue: `llm/prompts/sn/*.md` headers mention the
  new split.
