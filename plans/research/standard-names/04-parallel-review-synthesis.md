# Parallel Review Synthesis

## Executive Summary

Four parallel review agents analyzed the feature plans, codebase, MCP tools, and stakeholder feedback for the standard names batch generation system. The reviews converge on a central finding: **the project's domain logic is mature, but the plans overestimate the infrastructure gap and under-leverage existing code**.

The grammar engine, validation pipeline, catalog management, and MCP tool backing functions already provide ~80% of the schema provider work envisioned in the feature plans. The real gap is prompt orchestration and parallel dispatch — not data assembly. Meanwhile, Maarten's feedback reveals grammar extensions (binary operators, missing transformations) and naming policy gaps that are absent from all feature plans.

Key recommendations:
1. **Merge Features 01 and 05** — they produce identical template files
2. **Reduce waves from 5 to 4** — move pipeline data models to Wave 1 and add a Wave 0 setup step
3. **Bypass MCP for static providers** — import directly from `grammar.constants` and `grammar.field_schemas`
4. **Add a Feature 06** for grammar extensions identified by Maarten's feedback
5. **Resolve pyproject.toml as a Wave 0 coordination step** — prevent merge conflicts

## Cross-Cutting Themes

### Theme 1: The codebase is more capable than the plans assume

The research agent (Agent 2) found that MCP tools already function as schema providers delivering rich context objects. `get_grammar()`, `get_schema()`, and `get_vocabulary()` build structured output from auto-generated constants (`SEGMENT_RULES`, `SEGMENT_ORDER`, `SEGMENT_TEMPLATES`, `SEGMENT_TOKEN_MAP`). The `field_schemas.py` module (~39.7KB) contains `FIELD_GUIDANCE` and `NAMING_GUIDANCE` dictionaries ready for prompt injection.

The MCP tools agent (Agent 3) confirmed: grammar and vocabulary backing functions are pure, deterministic, and fully cacheable. Total static context budget is ~13KB — small enough to inline in every prompt.

**Implication**: Schema providers (Feature 01 Phase 2) should extract and wrap existing functions, not build from scratch. The feature plan should acknowledge this and reduce the estimated effort.

### Theme 2: Feature 01 and 05 are the same feature

The feature plan agent (Agent 1) identified that both features create template files in `prompts/generation/` and `prompts/review/`. Feature 01 Phase 3 and Feature 05 Phases 2-4 produce the same deliverables. The content audit (05 Phase 1) is prerequisite analysis for template authoring (01 Phase 3).

**Implication**: Merge into a single "Prompt System" feature with four phases: (1) loader core, (2) schema providers, (3) all template content, (4) MCP integration + deprecation.

### Theme 3: Pipeline data models have no real dependency on Wave 1

Feature 03 Phase 1 defines Pydantic models (`GenerationRequest`, `CandidateName`, `ReviewResult`, `BatchResult`, `PipelineConfig`). These are pure data definitions with no dependency on the prompt loader or LLM client. The feature plan agent identified this and recommended moving to Wave 1.

**Implication**: Pipeline models can be defined in parallel with prompt loader and LLM client work, providing stable interfaces earlier for downstream features.

### Theme 4: Missing foundational infrastructure

Multiple agents identified gaps in the plans' prerequisites:
- **Dependencies**: Jinja2 and anyio not in `pyproject.toml`
- **Shared exceptions**: No `exceptions.py` for cross-feature error types
- **Configuration**: No strategy for model/budget defaults (env vars, pyproject.toml, config file)
- **Logging**: No logging strategy for batch pipelines
- **Directory scaffolding**: `prompts/`, `pipeline/`, `llm/` directories need creation

**Implication**: Add a Wave 0 setup step for dependency management, directory scaffolding, shared exceptions, and configuration strategy.

### Theme 5: Stakeholder feedback reveals grammar gaps

Maarten's feedback (Agent 4) identified transformations and patterns absent from the grammar:
- `square_of_X`, `change_over_time_in_X` (unary transformations)
- `product_of_X_and_Y`, `ratio_of_X_to_Y` (binary operators)
- Units: None vs dimensionless distinction not enforced
- Missing standard name entries: `flux_loop_name`, `coil_current`, `passive_current`

None of the five feature plans address grammar extensions. These are prerequisites for generating correct names in affected domains.

**Implication**: Add Feature 06 for grammar extensions. Some items (binary operators) are substantial grammar changes that should be planned carefully.

### Theme 6: MCP tool layer should be bypassed for batch pipelines

The MCP tools agent (Agent 3) recommended that static providers import directly from `grammar.constants` and `grammar.field_schemas` rather than going through the MCP tool layer. MCP adds serialization overhead and request/response cycles that are unnecessary when the consumer is in-process Python code.

**Implication**: Schema providers should use direct imports for static data (grammar, vocabulary, field schemas) and reserve MCP calls for dynamic operations (search, catalog queries).

## Consolidated Gap List

Deduplicated across all four agents:

### Infrastructure Gaps

| ID | Gap | Source Agent(s) | Priority |
|----|-----|-----------------|----------|
| G01 | Jinja2, anyio missing from pyproject.toml | 1 | Critical (blocks all work) |
| G02 | No shared `exceptions.py` | 1 | High |
| G03 | No configuration strategy (model, budget defaults) | 1 | High |
| G04 | No logging strategy for batch pipelines | 1 | Medium |
| G05 | pyproject.toml merge conflict risk | 1 | High (Wave 0 coordination) |
| G06 | CLI namespace collision with existing entry points | 1 | Medium |
| G07 | No deprecation plan for `agents/` directory | 1, 2 | Low (post-MVP) |

### Design Gaps

| ID | Gap | Source Agent(s) | Priority |
|----|-----|-----------------|----------|
| G08 | Feature 01+05 overlap (identical template outputs) | 1 | Critical (blocks planning) |
| G09 | Feature 03 Phase 1 over-serialized (no real dep on Wave 1) | 1 | High |
| G10 | Feature 02 sync wrappers unnecessary (async-only) | 1 | Medium |
| G11 | Cross-review output model undefined | 1 | Medium |
| G12 | Path scanning strategy undefined in orchestrator | 1 | Medium |
| G13 | Schema providers underspecified — existing functions not mapped | 2, 3 | High |

### Domain Gaps (Maarten Feedback)

| ID | Gap | Source Agent(s) | Priority |
|----|-----|-----------------|----------|
| G14 | Missing transformations: square_of, change_over_time_in | 4 | High |
| G15 | Missing binary operators: product_of_X_and_Y, ratio_of_X_to_Y | 4 | High (grammar extension) |
| G16 | Units None vs dimensionless not enforced | 4 | Medium |
| G17 | Missing entries: flux_loop_name, coil_current, passive_current | 4 | Medium |
| G18 | Sign convention documentation partial | 4 | Low |
| G19 | Tag vocabulary divergence from Maarten's IDS-aligned proposal | 4 | Low (track, don't block) |

### Technical Debt

| ID | Gap | Source Agent(s) | Priority |
|----|-----|-----------------|----------|
| G20 | `Path(__file__).resolve()` fragility in agent workflows | 2 | Medium |
| G21 | Hardcoded `.vscode/mcp.json` path | 2 | Medium |
| G22 | `nest_asyncio.apply()` hack | 2 | Low |
| G23 | Empty `WorkflowBase` class | 2 | Low |
| G24 | Duplicate agent definition | 2 | Low |

## Updated Phasing Recommendation

### Wave 0: Foundation Setup (1 day, single agent)

- Add Jinja2, anyio to `pyproject.toml` dependencies
- Create directory scaffolding: `prompts/`, `pipeline/`, `llm/`
- Create shared `exceptions.py` with base exception hierarchy
- Define configuration strategy (environment variables with defaults)
- Resolve CLI entry point namespace

### Wave 1: Core Infrastructure (parallel, 2 agents)

| Agent | Work |
|-------|------|
| Agent A | Prompt System Phase 1 (loader core) + Phase 2 (schema providers) |
| Agent B | LLM Infrastructure Phase 1 (async client) + Phase 3 (model config) |
| Agent C | Pipeline data models (Feature 03 Phase 1) — moved from Wave 2 |

### Wave 2: Content and Workers (parallel, 3 agents)

| Agent | Work |
|-------|------|
| Agent A | Prompt System Phase 3 (all templates — generation, review, shared) |
| Agent B | Pipeline generation workers (Feature 03 Phase 2) |
| Agent C | Pipeline review workers (Feature 03 Phase 3) |
| Agent D | LLM cost tracking + caching (Feature 02 Phases 2, 4) |

### Wave 3: Integration (parallel, 2 agents)

| Agent | Work |
|-------|------|
| Agent A | Pipeline orchestrator (Feature 03 Phase 4) |
| Agent B | CLI framework + generate command (Feature 04 Phases 1-2) |

### Wave 4: Polish and Extensions (parallel, 3 agents)

| Agent | Work |
|-------|------|
| Agent A | CLI review + utility commands (Feature 04 Phases 3-4) |
| Agent B | Prompt System Phase 4 (MCP integration + agent deprecation) |
| Agent C | Grammar extensions (Feature 06 — Maarten gaps) |

## Feature Consolidation Recommendations

1. **Merge Features 01 + 05** → new "Prompt System" feature (01-prompt-system-v2)
2. **Feature 02** → keep, drop sync wrappers, add configuration strategy
3. **Feature 03** → keep, move Phase 1 to Wave 1
4. **Feature 04** → keep, add namespace collision resolution
5. **Feature 05** → absorbed into merged Feature 01
6. **Add Feature 06** → Grammar Extensions (binary operators, missing transformations, unit validation)
7. **Add Wave 0** → Foundation setup (dependencies, scaffolding, exceptions, configuration)

## Risk Assessment Update

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| pyproject.toml merge conflicts | High | Medium | Wave 0 consolidation step |
| Schema providers duplicate MCP backing logic | Medium | Low | Direct imports, not MCP calls |
| Binary operator grammar is a large change | Medium | High | Design review before implementation |
| Agent deprecation breaks existing workflows | Low | High | Gradual migration with both systems active |
| Template content quality varies by author | Medium | Medium | Cross-review step in Wave 2 |
