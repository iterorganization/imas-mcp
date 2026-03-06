# Discovery Infrastructure Unification

Status: **Planning**
Priority: High — prerequisite for documents CLI, signals expansion, and IMAS mapping tool

## Problem Statement

The discovery subsystem has grown organically across five domains (paths, wiki, signals, code, documents) to 62,000+ lines. Each domain independently implements the same structural patterns for CLI setup, worker orchestration, progress display, service access, and LLM interaction. This duplication creates three strategic risks:

1. **Scaling cost** — Every new discovery tool (documents CLI, IMAS mapping) requires re-implementing ~500 lines of CLI boilerplate, ~300 lines of state management, and ~800 lines of progress display infrastructure that already exists in four other places.

2. **Consistency drift** — Domains diverge in how they handle service health, shutdown, error recovery, and display formatting. The documents CLI already shows this: it has no service monitor, no rich display, and no graceful shutdown. Bug fixes and improvements made to one domain's orchestration don't propagate.

3. **Cognitive load** — A developer modifying worker supervision, service monitoring, or LLM calling patterns must understand and update five parallel implementations.

## Scope

This plan covers the unification of **shared infrastructure** across discovery domains. It explicitly excludes domain-specific logic (scanner implementations, scoring algorithms, graph operations) which should remain in their respective modules.

Areas of concern, by duplication severity:

| Area | Instances | Lines/Instance | Total Waste |
|------|-----------|---------------|-------------|
| CLI async boilerplate | 5 commands × ~60 lines | 60 | ~240 |
| DiscoveryState dataclasses | 5 variants | 50-150 | ~400 |
| Progress display subclasses | 4 full implementations | 700-1650 | ~2000 |
| Parallel engine skeletons | 5 `run_parallel_*()` fns | 200-400 (skeleton) | ~1200 |
| LLM response model field sets | 3 families × N models | 30-60 | ~200 |
| `log_print()` closures | 6 copies | 5 | trivial but symptomatic |

## Guiding Principles

1. **Extract, don't abstract** — Move repeated code into shared functions with clear signatures. Avoid framework-level abstractions that obscure control flow.

2. **Domain code stays domain code** — Scanners, scorers, graph_ops, and enrichment logic remain in their domain directories. Only orchestration and presentation infrastructure moves.

3. **New tools must compose** — The documents CLI, signals expansion, and IMAS mapping tool should be able to assemble a fully-featured discovery command from shared parts without copying boilerplate.

4. **Incremental migration** — Each unification step must leave all existing commands functional. No big-bang rewrites.

5. **Progress display as data** — Domain-specific display should be declarative configuration (stage names, worker groups, resource metrics) fed into a generic rendering engine, not imperative code.

---

## Unification Areas

### 1. CLI Command Harness

**Current state:** Each CLI command re-implements identical setup logic: `should_use_rich()` detection, `configure_cli_logging()`, `Console()` or `logging.basicConfig()`, a local `log_print()` closure, `create_discovery_monitor()`, `safe_asyncio_run()`, `install_shutdown_handlers()`, async refresh/ticker task creation, and task cancellation in a finally block. This pattern appears verbatim across paths, wiki, signals, code, and documents (minus the pieces documents doesn't have yet).

Additionally, `cli/discover/common.py` defines composable option decorators (`cost_options`, `phase_options`, `worker_options`, `focus_option`) and utility functions (`format_cost`, `format_duration`, `print_summary_table`) that **no CLI command actually uses** — they all define their own options and formatting inline.

**Direction:** A single harness function that accepts a domain-specific configuration object and an async entry point, handling all cross-cutting concerns. CLI commands become thin wrappers that declare their options, build a config, and delegate. The existing common.py decorators should either be adopted or removed — dead code erodes trust.

**Future impact:** The documents CLI needs rich display, service monitoring, and graceful shutdown. The IMAS mapping tool will need the same. Without a harness, each will re-implement the same 60-line async setup block.

### 2. Discovery State

**Current state:** Each parallel engine defines its own `*DiscoveryState` dataclass containing facility, cost_limit, focus, deadline, per-phase `WorkerStats`, `PipelinePhase` instances, a `stop_requested` flag, and a `should_stop()` property that checks budget, deadline, and phase completion. The structural skeleton is identical; only the phase names and domain-specific limits differ.

**Direction:** A generic `DiscoveryState` base in `discovery/base/` that manages phases, budget, deadline, and stop logic. Domain engines extend it only with domain-specific fields (e.g., `reference_shot` for signals, `min_score` for code). Phase registration becomes declarative — a list of phase names with their dependencies — rather than hand-wired `PipelinePhase` fields.

**Future impact:** The IMAS mapping tool will have its own phases (e.g., candidate → proposed → validated). Declarative phase management means it configures phases, not reimplements stop logic.

### 3. Parallel Engine Skeleton

**Current state:** Five `run_parallel_*()` functions follow the same 12-step skeleton — preflight, state init, phase wiring, stop event watching, worker group creation, supervised worker registration, orphan recovery, supervision loop, cleanup, and stats return. The skeleton is ~200-400 lines in each engine; the differentiation is in worker function references and phase names.

**Direction:** A shared `run_discovery_engine()` that receives a pipeline definition (phases, worker factories, orphan specs) and executes the standard skeleton. Domain-specific engines become declarative: "here are my phases, here are the worker functions for each phase, here's how many of each to run."

The key insight is that worker functions themselves are already domain-specific and self-contained. The orchestration around them — starting them, supervising them, handling signals, recovering orphans — is entirely generic.

**Future impact:** Adding the IMAS mapping engine becomes a configuration exercise rather than a 400-line reimplementation. Testing orchestration logic (shutdown, orphan recovery, budget tracking) can be done once against the shared engine.

### 4. Progress Display

**Current state:** The base class (`BaseProgressDisplay`, 1454 lines) already provides composable utilities: `StreamQueue`, `WorkerStats`, `PipelineRowConfig`, `ResourceConfig`, `build_pipeline_section()`, `build_resource_section()`. Domain subclasses extend it with `_build_pipeline_section()` and `_build_resources_section()` implementations. These implementations are 700-1650 lines each and follow the same structure: compute progress data, count workers, format activity text, construct row configs, call builders.

Each domain also defines its own `ProgressState` dataclass (separate from `DiscoveryState` in the parallel engine) containing display-specific counters, streaming queues, and rate trackers. The display state and engine state are linked via progress callbacks (`on_scan_progress`, `on_triage_progress`, etc.) defined in the CLI command.

**Direction:** The pipeline section can be driven from a declarative pipeline definition — the same one used by the engine skeleton. Each stage has a name, a worker group name, count fields, and an activity formatter. The display class receives this definition and renders it without domain-specific code.

The ProgressState/DiscoveryState duality should be examined. Currently, the CLI creates both a `ProgressState` (for display) and a `DiscoveryState` (in the engine), then bridges them via callbacks. This double bookkeeping is a source of complexity. Consider whether the display can observe the engine state directly, or whether a shared state object can serve both roles.

**Future impact:** The documents CLI currently has no progress display. A declarative display definition would let it add one in ~20 lines of configuration rather than ~900 lines of implementation.

### 5. Service Access

**Current state:** `GraphClient()` is instantiated ~233 times across 32 files, always as a short-lived context manager. This is consistent but high-ceremony. The embedding system is well-centralized through `embed_description_worker`. LLM access is mostly centralized through `base/llm.py`, with two exceptions where `wiki/scoring.py` and `base/image.py` call `litellm.acompletion()` directly with duplicated retry/cost logic.

Service health monitoring via `ServiceMonitor` is well-designed but only connected to CLI commands that manually wire it up. The documents CLI lacks it entirely.

**Direction:**

- **LLM access:** The two direct `litellm.acompletion()` call sites should route through `acall_llm_structured()`. The multimodal content case (images) may need the structured call API to accept message content parts, but the retry, cost tracking, and error handling should not be duplicated.

- **Graph access:** The per-operation `with GraphClient() as gc:` pattern works adequately thanks to Neo4j driver pooling. No fundamental change needed, but the one non-context-manager usage in `tdi.py` should be fixed.

- **Service monitoring:** When the CLI harness manages service monitoring, it automatically becomes available to all domains including new ones. No domain-specific wiring needed.

**Future impact:** The IMAS mapping tool will use LLM calls extensively for proposing mappings. It must go through the centralized cost-tracked path from day one. Having service monitoring automatic via the harness means mapping workers get health-gated execution for free.

### 6. LLM Response Models

**Current state:** Three families of Pydantic `*Batch/*Result` models share field sets:

- **Paths/Code family:** 9-11 identical `score_*: float` fields (data_access, imas_integration, convention_compliance, etc.) with identical descriptions and the same composite score formula
- **Wiki/Document/Image family:** 6 identical `score_*: float` fields (data_documentation, physics_content, code_documentation, etc.)
- **Signals family:** Distinct fields, no overlap

Common fields like `physics_domain: PhysicsDomain`, `keywords: list[str]`, and `description: str` appear across most models with identical types and descriptions.

**Direction:** Define shared field sets (mixins or base models) for the two scoring families and for common metadata fields. Domain models compose from these rather than copy-pasting field definitions. The composite score formula should be a single implementation.

This requires care — LLM structured output models must remain flat Pydantic classes since LiteLLM serialises the schema for the model provider. Inheritance or mixins at the Python level are fine as long as the resulting JSON schema from `model_json_schema()` remains flat and explicit.

**Future impact:** The IMAS mapping tool will need its own scoring model (confidence, evidence quality, transformation complexity). Shared base fields ensure consistent conventions. Any future scoring dimension changes propagate automatically.

---

## Considerations for Future Tools

### Documents CLI

The documents CLI is the nearest customer for this unification. It currently operates without rich display, service monitoring, or graceful shutdown. After unification, it should gain all three by composing from the shared harness and declaring its pipeline stages.

### Signals Expansion

The signals pipeline needs to support additional facility data access patterns (PPF at JET, EDAS at JT-60SA) alongside the existing TDI/MDSplus scanners. The scanner plugin architecture is already well-isolated. The main unification benefit is that adding new scanner types doesn't require touching orchestration code — new scanners register themselves, and the shared engine runs them.

### IMAS Mapping Tool

This tool represents a qualitatively different discovery domain — it consumes the output of other domains (FacilitySignals, TreeNodes, IMASPaths) rather than discovering raw data. Its pipeline will be:

1. **Candidate generation** — graph queries joining signals to potential IMAS paths
2. **LLM mapping proposal** — structured output with transformation details
3. **Validation** — optional live data comparison
4. **Persistence** — IMASMapping nodes with evidence chains

The IMAS mapping tool benefits from every unification area: the CLI harness provides setup and display, the engine skeleton provides worker management, the shared state provides budget/deadline tracking, the LLM access provides cost-tracked structured output, and the progress display provides live visibility.

Without unification, building this tool means copying and adapting ~2000 lines from an existing domain. With unification, it means writing the domain-specific worker functions and a declarative pipeline definition.

---

## Migration Strategy

Unification should proceed bottom-up, from lowest-risk shared utilities to highest-impact structural changes:

**Phase 1 — Consolidate what's already supposed to be shared.** Adopt or remove the unused common.py decorators and utilities. Fix the two direct LiteLLM call sites. Fix the non-context-manager GraphClient usage. These are isolated correctness improvements with no architectural risk.

**Phase 2 — Extract the CLI harness.** Move the repeated async setup/teardown/display pattern into a shared function. Migrate one domain at a time, validating that behavior is preserved. The documents CLI becomes the test case for harness adoption.

**Phase 3 — Unify state and engine skeleton.** Extract the generic DiscoveryState base and `run_discovery_engine()`. This is the highest-value change but also the highest-risk, as it touches the core control flow of every domain. Migrate the simplest engine first (static), then expand.

**Phase 4 — Declarative progress display.** Once the engine skeleton provides a standard pipeline definition, the progress display can render from it. This decouples display implementation from domain logic entirely.

**Phase 5 — Shared scoring models.** Extract common field sets into base models. This is low-risk but touches prompt templates and schema providers, so it requires validation of LLM output quality.

---

## Success Criteria

- A new discovery domain (e.g., IMAS mappings) can implement a fully-featured CLI command with rich display, service monitoring, graceful shutdown, and budget tracking by writing: one click command (~30 lines), one pipeline definition (~20 lines), and domain-specific worker functions.

- Bug fixes to shutdown handling, service monitoring, or budget tracking apply to all domains automatically.

- The documents CLI has feature parity with other domains (rich display, service monitoring) without domain-specific display infrastructure.

- Total discovery subsystem line count decreases despite adding new functionality.
