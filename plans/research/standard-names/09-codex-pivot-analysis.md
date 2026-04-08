# Strategic Pivot Analysis: Generation in Codex, Governance in Standard-Names

**Status:** Approved  
**Date:** 2025-07-18 (revised 2026-04-08)  
**Researched by:** 5 parallel Opus 4.6 agents + direct investigation  

---

## Executive Summary

**The decision:** All pipeline development (generation, review, minting) moves to **imas-codex**. This project (imas-standard-names) retains grammar, validation, catalog management, MCP tools, and website.

**Why codex:** It already provides ~8,000+ lines of proven pipeline infrastructure (graph-as-ledger, supervised workers, LLM structured output, cost tracking, claim-based concurrency). Building this independently here would duplicate battle-tested code.

**Critical scope correction:** Standard names are not just about the IMAS Data Dictionary. The pipeline mints from **multiple sources**:
- IMAS DD paths (`IMASNode` → standard names for physics quantities in the DD)
- Facility signals (`FacilitySignal` → standard names for experimentally measured quantities)
- Potentially other graph entities in the future

Therefore the codex pipeline is a **top-level CLI group** (`imas-codex sn`), not nested under `imas`. It sits alongside `discover` and `imas` as a peer command group.

---

## 1. Evidence Summary

### 1.1 What codex already provides

| Capability | Codex Module | Lines | Standard-Names Equivalent |
|---|---|---|---|
| Supervised parallel workers | `discovery/base/engine.py` | 224 | Would need to build from scratch |
| Claim-based concurrency | `discovery/base/claims.py` | ~200 | Would need to build from scratch |
| Pipeline state machine | `discovery/base/supervision.py` | 1,143 | Would need to build from scratch |
| LLM structured output + retry | `discovery/base/llm.py` | 938 | Would need to build from scratch |
| Cost tracking + budget limits | Built into `WorkerStats` | — | Would need to build from scratch |
| Worker restart with backoff | `supervised_worker()` | ~200 | Would need to build from scratch |
| DD path enrichment pipeline | `graph/dd_workers.py` | ~500 | Would need to build from scratch |
| DD path context gathering | `graph/dd_enrichment.py` | ~700 | Would need to build from scratch |
| Graph-as-work-queue | `graph/client.py` + `claims.py` | ~1,200 | Would need to build from scratch |
| Embed server (GPU) | `embeddings/server.py` | ~800 | Not applicable |
| Jinja2 prompt templates | `llm/prompt_loader.py` | 1,176 | Would need to build from scratch |
| Schema-driven ontology | `graph/schema.py` (LinkML) | 826 | Not applicable |
| StandardName graph node | `schemas/facility.yaml` L2088 | — | Already defined, unpopulated |
| MEASURES relationship | `schemas/facility.yaml` L2138 | — | Links signals → standard names |
| CLI discover framework | `cli/discover/common.py` | ~350 | Would need to build from scratch |
| Service health monitoring | `discovery/base/services.py` | ~900 | Not applicable |

**Total infrastructure codex provides:** ~8,000+ lines of proven, battle-tested code.  
**Total we would need to rewrite in standard-names:** ~6,000-8,000 lines.

### 1.2 What standard-names uniquely provides

| Capability | Module | Lines | Portable to codex? |
|---|---|---|---|
| Grammar specification | `grammar/specification.yml` | ~800 | Import as dependency |
| Parse/compose functions | `grammar/support.py` | 264 | Import as dependency |
| Grammar codegen | `grammar_codegen/` | ~1,443 | Stays in standard-names |
| Semantic validation (8 rules) | `validation/semantic.py` | 244 | Import as dependency |
| Quality validation (NLP) | `validation/quality.py` | 388 | Import as dependency |
| Catalog management | `catalog/`, `database/` | ~1,400 | Stays in standard-names |
| MCP tools (14 tools) | `tools/` | ~3,500 | Stays in standard-names |
| Entry models (Pydantic) | `models.py` | 681 | Import as dependency |

**Key finding:** The grammar system has zero external dependencies. It exports clean public APIs via `__all__` and can be imported by codex with `pip install imas-standard-names`.

### 1.3 Codex StandardName graph node already exists

The LinkML schema at `schemas/facility.yaml:2088-2110` already defines:

```yaml
StandardName:
  id: string (snake_case, required, identifier)
  description: string
  canonical_units: range Unit
  
# Relationships:
#   FacilitySignal -[:MEASURES]-> StandardName (cross-facility semantic linking)
#   StandardName -[:HAS_UNIT]-> Unit
#   StandardName -[:IN_DOMAIN]-> PhysicsDomainNode
```

No code yet populates these nodes. This is the natural extension point.

---

## 2. Proposed Architecture

### 2.1 Repository responsibilities

```
┌─────────────────────────────────────────────┐
│            imas-standard-names               │
│                                              │
│  Grammar     ← specification.yml + codegen   │
│  Validation  ← semantic + quality + structural│
│  Catalog     ← YAML store + SQLite + FTS     │
│  MCP Tools   ← interactive chat interface     │
│  Website     ← mkdocs-material + mike        │
│  Models      ← Pydantic entry definitions     │
│                                              │
│  Exports:                                    │
│    compose_standard_name()                   │
│    parse_standard_name()                     │
│    validate_models()                         │
│    StandardNameEntry (Pydantic)              │
│    Grammar enums (Component, Position, etc.) │
└────────────────────┬────────────────────────┘
                     │ pip install (one-way dep)
                     ▼
┌─────────────────────────────────────────────┐
│                imas-codex                     │
│                                              │
│  NEW: sn/ module (top-level, NOT under imas)│
│    pipeline.py  ← orchestrator               │
│    workers.py   ← compose, review, validate  │
│    state.py     ← SNBuildState                  │
│    graph_ops.py ← claim/mark/release          │
│    sources/                                  │
│      dd.py      ← mint from IMASNode paths   │
│      signals.py ← mint from FacilitySignals  │
│    prompts/     ← Jinja2 generation templates │
│                                              │
│  CLI (top-level peer of discover, imas):     │
│    imas-codex sn build --source dd          │
│    imas-codex sn build --source signals     │
│    imas-codex sn build --ids equilibrium    │
│    imas-codex sn status                     │
│    imas-codex sn benchmark                  │
│                                              │
│  Graph nodes:                                │
│    StandardName (existing, now populated)     │
│    StandardName -[:DERIVED_FROM]-> IMASNode   │
│    FacilitySignal -[:MEASURES]-> StandardName │
│                                              │
│  Output:                                     │
│    Minted YAML files → PR to catalog repo     │
└─────────────────────────────────────────────┘
```

**Why `mint` as a top-level group:** Standard names are minted from multiple data sources — IMAS DD paths, facility signals, and potentially others. This is not an IMAS-specific operation (`imas sn build` is wrong) nor a facility discovery operation (`discover standard-names` is wrong). Minting is its own domain that consumes output from both `imas` and `discover` pipelines.

### 2.2 Pipeline phases — multi-source architecture

The pipeline is source-agnostic at its core. Different sources feed candidates into the same compose → review → validate → mint flow:

```
Sources:                          Core Pipeline:
                                  
  DD paths ──┐                    COMPOSE ──→ REVIEW ──→ VALIDATE ──→ MINT
              ├──→ EXTRACT ──→    
  Signals ───┘                    

Source plugins:
  dd.py:      Walk IMASNode paths (IDS-filtered, domain-filtered)
              Group by semantic cluster, gather tree context
              Output: candidate extraction batches

  signals.py: Walk FacilitySignal nodes (facility-filtered)
              Group by physics domain, gather signal context
              Output: candidate extraction batches
```

Using the codex `WorkerSpec` pattern with `PipelinePhase`:

```
SNBuildState(DiscoveryStateBase):
  
  extract_phase:  Source-specific path/signal walking
  compose_phase:  LLM generates candidate names + descriptions
                  Uses grammar rules + source context as prompt
                  Grammar validation (compose → parse round-trip)
  review_phase:   Cross-model "rubber duck" review
                  Generate with Model A, review with Model B
  validate_phase: Full validation pipeline
                  Import validate_models() from standard-names
  mint_phase:     Convert validated candidates to catalog YAML
                  Create PR to imas-standard-names-catalog
```

### 2.3 Graph node extensions

New node for pipeline tracking:

```yaml
StandardNameCandidate:
  id: string (candidate identifier)
  name: string (proposed standard name)
  description: string
  unit: string
  kind: enum (scalar/vector/metadata)
  source_ids: string (IDS that triggered generation)
  source_cluster: string (semantic cluster label)
  status: enum (extracted → composed → reviewed → validated → minted → rejected)
  review_model: string (model used for review)
  review_score: float
  review_notes: string
  generation_model: string
  generation_cost: float
  
# Relationships:
#   StandardNameCandidate -[:DERIVED_FROM]-> IMASNode
#   StandardNameCandidate -[:IN_CLUSTER]-> IMASSemanticCluster
#   StandardNameCandidate -[:PROMOTED_TO]-> StandardName (after minting)
```

### 2.4 Cross-model review design

Instead of human review at scale, use a different LLM family:

| Phase | Model Family | Role |
|---|---|---|
| **Compose** | Claude (Sonnet/Haiku) | Generate candidate names + descriptions |
| **Review** | Gemini (Flash/Pro) | Independent review: grammar correctness, physics accuracy, naming quality |
| **Arbitrate** (on disagreement) | GPT (4o/4o-mini) | Tie-breaking when compose + review disagree |

The review prompt includes:
- Grammar rules (injected from standard-names)
- The DD path context (from graph)
- The candidate name + description
- Existing catalog names (for consistency checking)

---

## 3. Addressing User Concerns

### 3.1 Over-reliance on clustering

**Solution:** The extract phase walks paths directly, not clusters:

```python
# Primary input: IDS-filtered path enumeration
imas-codex discover standard-names --ids equilibrium --cost-limit 5

# Secondary input: domain-filtered
imas-codex discover standard-names --domain magnetics

# Tertiary: cluster-assisted grouping (for deduplication, not input)
# Clusters inform WHICH paths map to the SAME standard name
# But the pipeline discovers paths independently
```

The `DDBuildState` pattern already supports `ids_filter: set[str] | None` — we inherit this directly.

### 3.2 No human review at scale

**Solution:** Cross-model rubber duck review (Section 2.4). The review phase uses a different model family than composition. Both models have access to the same grammar rules and DD context, but bring independent reasoning.

### 3.3 Benchmarking phase needed

**Solution:** Add a benchmarking command before production runs:

```bash
# Benchmark: run 20 paths through all model combos, compare quality
imas-codex discover standard-names benchmark \
  --ids equilibrium \
  --sample-size 20 \
  --models "claude-sonnet-4-20250514,gemini-2.5-flash,gpt-4o-mini"
```

This generates a comparison table: cost per name, grammar pass rate, review agreement rate, description quality score.

### 3.4 Exploiting existing codex infrastructure

**Fully exploited.** The generation pipeline:
- Uses `run_discovery_engine()` (no new orchestration code)
- Uses `SupervisedWorkerGroup` (parallel workers with restart)
- Uses `PipelinePhase` (dependency tracking)
- Uses `acall_llm_structured()` (LLM with retry + cost)
- Uses `GraphClient.create_nodes()` (batch graph writes)
- Uses claim-based concurrency (no new locking code)
- Uses the embed server (for candidate embedding)
- Uses Jinja2 prompt templates (loaded via `prompt_loader.py`)

---

## 4. Risk Analysis

### 4.1 Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **Compile-time coupling** | Medium | One-way only: codex → standard-names. Standard-names never imports codex. Version pin in codex's deps. |
| **Deployment complexity** | Low | Generation pipeline only runs in codex dev environment (Neo4j + GPU). Standard-names remains lightweight for users. |
| **Release cycle mismatch** | Low | Grammar changes in SN trigger codex dep bump via CI. Catalog output is just YAML files — format-stable. |
| **Integration testing** | Medium | Test suite in codex mocks standard-names grammar + validation. E2E tests run in CI with both repos. |
| **Grammar evolution** | Low | Grammar API is stable (`compose_standard_name`, `parse_standard_name`). Breaking changes are rare and versioned. |
| **Codex Neo4j requirement** | Low | Only needed for generation. Standard-names catalog consumers never need Neo4j. |

### 4.2 What we avoid

By not building the pipeline in standard-names:
- **~6,000-8,000 lines** of pipeline/worker/state/LLM code not duplicated
- **No new infrastructure** (no graph DB, no embed server, no worker supervision)
- **No parallel worker framework** to maintain separately
- **No LLM client** to build and maintain (litellm integration, retry, cost tracking)
- **No prompt template system** to build (Jinja2 loader + schema providers)

---

## 5. Implementation Phasing

### Phase 0: Foundation (in imas-codex)
- Add `imas-standard-names` as dependency
- Create `discovery/standard_names/` module skeleton
- Extend LinkML schema with `StandardNameCandidate` node
- Create initial Jinja2 prompt templates

### Phase 1: Extract + Compose (in imas-codex)
- Implement path walking (IDS-filtered, domain-filtered)
- Implement grammar-aware name composition via LLM
- Grammar validation round-trip (compose → parse)
- CLI: `imas-codex discover standard-names --ids <ids>`

### Phase 2: Review + Validate (in imas-codex)
- Cross-model review pipeline
- Import `validate_models()` from standard-names
- Review scoring and accept/reject logic

### Phase 3: Mint + Publish (in both repos)
- YAML generation from validated candidates
- GitHub API integration for catalog PRs
- Benchmarking command

### Phase 4: Feedback Loop
- DD path linking back to standard-names catalog
- Populate `ids_paths` field in catalog entries
- Graph relationship: `StandardName -[:DERIVED_FROM]-> IMASNode`

---

## 6. What Stays in imas-standard-names

The standard-names project retains its core mission:

1. **Grammar authority**: specification.yml, codegen, enums — the authoritative source
2. **Validation authority**: semantic, structural, quality checks — the gatekeeper
3. **Catalog management**: YAML store, SQLite, FTS — the persistence layer
4. **MCP tools**: interactive chat interface — the user-facing API
5. **Website/docs**: mkdocs-material — the publication platform
6. **Catalog repo**: imas-standard-names-catalog — the data store

What it **no longer needs** from the previous plans:
- ~~Feature 01: Prompt System~~ → codex has Jinja2 + prompt_loader
- ~~Feature 02: LLM Pipeline Infrastructure~~ → codex has llm.py + cost tracking
- ~~Feature 03: Batch Generation Pipeline~~ → codex has discovery engine
- ~~Feature 04: CLI Dispatch Commands~~ → codex has CLI discover framework
- ~~Feature 05: DD Integration (generation parts)~~ → codex has DD graph access

What it **keeps** from the previous plans:
- Feature 05 (Phase 1 only): DD path linking in existing catalog entries
- Feature 06: Grammar Extensions (Maarten's feedback) — grammar must evolve here

---

## 7. Comparison: Build in SN vs Build in Codex

| Dimension | Build in Standard-Names | Build in Codex |
|---|---|---|
| **New code required** | ~6,000-8,000 lines | ~1,500-2,500 lines |
| **Infrastructure deps** | Would need Neo4j adapter, LLM client, worker framework | All exist already |
| **State management** | Would need to build from scratch | Graph-as-ledger proven across 5 domains |
| **Parallel execution** | Would need worker supervision | `SupervisedWorkerGroup` battle-tested |
| **Cost control** | Would need cost tracking | `WorkerStats.cost` already tracks USD |
| **DD access** | Via adapter importing codex functions | Direct graph queries |
| **Prompt system** | Would need template engine | Jinja2 + schema providers exist |
| **Standard Name node** | No graph | Already defined in LinkML schema |
| **Cross-facility linking** | Not possible | `FacilitySignal -[:MEASURES]-> StandardName` |
| **Benchmarking** | Would need LLM comparison framework | `acall_llm_structured` supports any model |
| **Resume/retry** | Would need checkpoint system | Claim-based: crash-safe by design |

**Verdict:** Building in codex requires ~70% less new code, inherits proven patterns, and provides capabilities (graph relationships, cross-facility linking, crash recovery) that would be extremely difficult to replicate in standard-names.

---

## 8. Critique Integration

An independent rubber-duck review (Opus 4.6, 23 tool calls deep investigation) identified 5 findings:

### 8.1 Finding 1: Build pipeline, not discovery domain (adopted, revised)

Standard name generation is not "discovering what exists at a facility" — it is "synthesizing canonical vocabulary from multiple graph sources." Every codex discovery domain takes `facility` as a required argument, and standard names are facility-agnostic.

**Original resolution:** Place at `graph/sn_workers.py`, CLI as `imas-codex imas sn build`.

**Revised resolution:** The pipeline mints from **multiple sources** (DD paths, facility signals, potentially others), so it is not IMAS-specific. Create a new **top-level `sn` module** in codex (`imas_codex/sn/`) with source plugins. CLI becomes `imas-codex sn build --source dd` or `imas-codex sn build --source signals`. This sits alongside `discover` and `imas` as a peer command group.

### 8.2 Finding 2: Transient candidates, not permanent nodes (adopted)

Creating permanent `StandardNameCandidate` graph nodes for every LLM proposal — including rejected ones — would pollute the graph with thousands of noise nodes. The DD build pipeline only creates `IMASNode` for paths that actually exist in the DD XML.

**Resolution:** Candidates are tracked in-memory during pipeline runs. Only validated, minted names become `StandardName` graph nodes. Run metadata (rejected candidates, review scores) stored as structured JSON logs, not graph nodes. The `DERIVED_FROM` relationship links the final `StandardName` → `IMASNode` without intermediate candidate nodes.

### 8.3 Finding 3: Data contract at the boundary, optional library import (partially adopted)

A compile-time dependency creates version-lock risk: grammar evolution in standard-names could cause codex's installed version to reject valid new names.

**Resolution:** Hybrid approach:
- **Hard requirement:** The mint boundary uses a JSON schema contract (matching `StandardNameEntry` Pydantic schema). Codex outputs YAML files that standard-names validates independently.
- **Soft dependency:** Codex can optionally `pip install imas-standard-names` for local grammar validation during composition (faster iteration). If not installed, codex uses schema-only validation and defers full validation to the mint phase.
- **Version strategy:** Codex pins to `>=X.Y.0,<X+1.0.0` and CI tests against standard-names main branch.

### 8.4 Finding 4: Batched PRs with confidence tiers (adopted)

A single monolithic PR with hundreds of names creates a review avalanche.

**Resolution:** The mint phase produces batched PRs by IDS or physics domain. Each PR includes:
- **Tier 1 (high confidence):** Grammar-valid, strong DD lineage, cross-model agreement. Auto-merge candidates.
- **Tier 2 (needs review):** Ambiguous physical_base choices, multiple candidate compositions for one DD path. Require human judgment.
- **Tier 3 (rejected):** Grammar-invalid or review-rejected. Listed in PR description for transparency but not included.

### 8.5 Finding 5: Catalog-aware deduplication (adopted)

The compose phase must know what already exists in the catalog. Without this, the pipeline would propose names that collide with existing entries or duplicate work across runs.

**Resolution:** The compose phase:
1. Queries the existing catalog before proposing names
2. Checks for name collisions (same physical concept → same canonical name from different DD paths)
3. Sets appropriate `deprecates`/`superseded_by` fields when re-runs produce superior compositions
4. Tracks provenance linking to specific DD version and pipeline run

---

## 9. Decision: Approved

**All pipeline development moves to imas-codex. This repo retains governance.**

The split is:

| Concern | Where | Why |
|---------|-------|-----|
| Grammar (specification, codegen, enums) | imas-standard-names | Authoritative source of naming rules |
| Validation (semantic, structural, quality) | imas-standard-names | Gatekeeper for catalog quality |
| Catalog management (YAML, SQLite, FTS) | imas-standard-names | Persistence and query layer |
| MCP tools (interactive chat) | imas-standard-names | User-facing API for agents |
| Website/documentation | imas-standard-names | Publication platform |
| Generation pipeline (extract, compose) | **imas-codex** | Has graph, workers, LLM infra |
| Review pipeline (cross-model) | **imas-codex** | Has LLM client + cost tracking |
| Minting (YAML output, PR creation) | **imas-codex** | Produces artifacts for catalog |
| Benchmarking (model comparison) | **imas-codex** | Has multi-model LLM support |
| StandardName graph population | **imas-codex** | Owns the graph |

**Dependency direction:** One-way. Codex imports standard-names for grammar + validation. Standard-names never imports codex.

**Data sources for minting:** DD paths, facility signals, and potentially others. The pipeline is source-agnostic with pluggable extractors.

**CLI placement in codex:** `imas-codex sn` — a top-level command group, not nested under `imas` or `discover`.

**What changes in imas-standard-names:**
1. Grammar API exports: ensure clean importability (Feature 01)
2. DD path linking for existing 309 entries (Feature 02)
3. Grammar extensions per Maarten's feedback (Feature 03)
4. JSON schema contract for cross-project validation (Feature 04)

**What happens in imas-codex:**
1. `imas_codex/sn/` — new top-level module with source plugins
2. `imas_codex/sn/sources/dd.py` — extract from IMASNode paths
3. `imas_codex/sn/sources/signals.py` — extract from FacilitySignal nodes
4. `imas_codex/cli/sn.py` — top-level `sn` CLI group
5. `llm/prompts/sn/` — Jinja2 generation/review templates
6. `schemas/facility.yaml` — extend StandardName with DERIVED_FROM, source metadata
7. Benchmarking command for model comparison
