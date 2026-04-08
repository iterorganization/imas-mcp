# Comparative Analysis: Standard Names vs Codex Prompt Systems

## Executive Summary

The standard names project operates at a fundamentally different maturity level than codex's prompt infrastructure. Codex treats prompts as **engineered artifacts** with versioning, caching, schema injection, and cost-bounded parallel execution. Standard names treats prompts as **prose instructions** consumed by a human-operated chat interface. The gap is structural, not incremental — closing it requires adopting codex's architectural patterns rather than improving the current approach in place.

## Dimension-by-Dimension Comparison

### 1. Prompt Format and Templating

| Aspect | Standard Names | Codex |
|--------|---------------|-------|
| Format | Static markdown files | Markdown + YAML frontmatter + Jinja2 |
| Templating | None | Full Jinja2 (includes, loops, conditionals) |
| Variables | None | `{{ facility }}`, `{{ signal_source_detail }}` |
| Includes | None | `{% include "schema/physics-domains.md" %}` |
| Loops | None | `{% for dim in score_dimensions %}` |
| Conditional | None | `{% if score_calibration %}` |
| Rendering modes | Single (static read) | Two (static parse + dynamic render) |

**Impact**: Standard names prompts cannot adapt to different contexts. The same prompt is used whether generating 1 name or 50, whether the domain is magnetics or transport. Codex prompts self-configure based on the task.

### 2. Schema and Context Injection

| Aspect | Standard Names | Codex |
|--------|---------------|-------|
| Grammar rules | LLM calls `get_grammar()` at runtime | Pre-injected via schema provider |
| Vocabulary | LLM calls `get_vocabulary()` at runtime | Enum values injected into template |
| Output schema | Natural language description | Pydantic JSON example + field descriptions |
| Domain context | LLM discovers via multiple tool calls | Pre-assembled per-call context block |
| Calibration examples | None | Real scored examples from knowledge graph |

**Impact**: Standard names wastes 3-5 tool calls per session to discover static context that could be pre-injected. Each tool call adds latency and token cost.

**Quantitative estimate**: A typical standard name generation session makes ~8-12 MCP tool calls. Of these, 3-5 are pure context discovery (grammar, vocabulary, schema, existing names). If this context were pre-injected into the prompt, those calls would be eliminated, saving ~2000-4000 tokens and ~30-60 seconds per session.

### 3. Batch Processing and Parallelism

| Aspect | Standard Names | Codex |
|--------|---------------|-------|
| Processing model | Sequential, one-at-a-time | Parallel workers with supervisor |
| Batch size | 1-5 names per session | 20-50 items per batch |
| Concurrency | None | 2-8 workers per pipeline |
| Pipeline stages | Manual (human drives each step) | Automated (scan → triage → enrich → score) |
| Phase dependencies | None | Declarative (`depends_on=["extract_phase"]`) |
| Orphan recovery | None | Automatic reset for stuck items |

**Impact**: Codex can process 500+ items in a single pipeline run. Standard names requires a human session per small batch.

### 4. Cost and Budget Management

| Aspect | Standard Names | Codex |
|--------|---------------|-------|
| Cost tracking | None | Per-call extraction from LiteLLM response |
| Budget limits | None | `--cost-limit 5.0` per CLI invocation |
| Time limits | None | `--time 30` (minutes) |
| Budget exhaustion | Silent failure | `ProviderBudgetExhausted` exception → halt all |
| Cost reporting | None | Per-worker stats in progress output |

**Impact**: Standard names has no visibility into LLM spend. A poorly constructed prompt or retry loop could burn significant budget with no alerting.

### 5. Error Handling and Retry Logic

| Aspect | Standard Names | Codex |
|--------|---------------|-------|
| Retry on API error | Basic try/except | Exponential backoff (5s base, 5 retries) |
| Retryable detection | None | Pattern matching (429, 503, timeout, JSON) |
| Parse error recovery | Crash | Retry the entire LLM call |
| Validation error | Return error dict | Retry with fresh attempt |
| Budget exhaustion | None | Detect 402, halt all workers immediately |

### 6. Prompt Organization

| Aspect | Standard Names | Codex |
|--------|---------------|-------|
| Prompt files | 4 files in `.github/prompts/` | 30+ files in `llm/prompts/` |
| Shared fragments | None | 14 files in `shared/schema/` |
| Naming convention | Flat, descriptive filenames | Domain-prefixed (e.g., `paths/scorer`) |
| Traceability | None | `used_by` frontmatter links to consumer code |
| Version hashing | None | `get_prompt_content_hash()` for cache invalidation |

### 7. LLM Provider Integration

| Aspect | Standard Names | Codex |
|--------|---------------|-------|
| Provider | OpenRouter via pydantic_ai | OpenRouter via LiteLLM |
| Model selection | Hardcoded per agent | Configurable via CLI `--model` |
| Token limits | None | Model-family-aware (Gemini: 16K, Claude: 32K) |
| Prompt caching | None | `inject_cache_control()` + static-first layout |
| Noise suppression | None | `suppress_litellm_noise()` |
| Offline support | None | Air-gap detection, local model cost maps |

### 8. Output Structure and Validation

| Aspect | Standard Names | Codex |
|--------|---------------|-------|
| Output format | Pydantic via pydantic_ai | Pydantic via `call_llm_structured()` |
| JSON sanitization | None | Strip markdown fences, fix truncation |
| Batch output | List of entries | Batch model (e.g., `ScoreBatch`, `SignalMappingBatch`) |
| Confidence scores | Pass/fail threshold (0.7) | Multi-dimensional (0.0-1.0 per dimension) |
| Evidence tracking | None | `primary_evidence`, `evidence_summary`, `scoring_reason` |
| Unmapped handling | None | Explicit `unmapped` array with disposition codes |

## Architectural Gap Analysis

### What Standard Names Has That Codex Doesn't

1. **MCP tool ecosystem**: 15+ well-designed tools for grammar, vocabulary, search, validation
2. **Grammar validation engine**: Formal grammar parsing with Pydantic models
3. **Catalog management**: YAML-backed catalog with edit history and rollback
4. **Interactive flexibility**: Can handle novel edge cases through conversation

### What Codex Has That Standard Names Needs

1. **Prompt loader with frontmatter and schema injection**
2. **Cached schema providers (grammar, vocabulary, scoring criteria)**
3. **Two-mode rendering (static + dynamic)**
4. **Parallel worker engine with supervision**
5. **Cost-bounded execution with retry logic**
6. **CLI dispatch for pipeline orchestration**
7. **Prompt caching optimization (static-first layout)**
8. **Calibration example injection**
9. **System/user prompt separation for caching**
10. **Batch output models with structured errors**

### Synergy Opportunities

The standard names MCP tools are a natural **context source** for a codex-style prompt system:

```
┌─────────────────────┐     ┌─────────────────────┐
│  MCP Tools           │     │  Prompt Loader       │
│  (grammar, vocab,    │────▶│  (Jinja2 + schema    │
│   search, validate)  │     │   providers)          │
└─────────────────────┘     └──────────┬────────────┘
                                       │
                            ┌──────────▼────────────┐
                            │  Rendered Prompt       │
                            │  (grammar rules +      │
                            │   vocabulary tokens +   │
                            │   existing names +      │
                            │   DD context)           │
                            └──────────┬────────────┘
                                       │
                            ┌──────────▼────────────┐
                            │  LLM Pipeline          │
                            │  (batch dispatch,      │
                            │   structured output,    │
                            │   cost tracking)        │
                            └───────────────────────┘
```

The MCP tools would shift from being called interactively by the LLM to being called programmatically by the prompt loader to **pre-assemble context**. The LLM still uses some tools for dynamic research (e.g., searching for name collisions) but receives a much richer starting context.

## Conclusion

The standard names project has excellent domain logic (grammar engine, catalog management, MCP tools) but lacks production prompt infrastructure. The codex project has excellent prompt infrastructure but operates in a different domain. The implementation plan should **port codex's prompt architecture** into standard names, adapting it to the standard name generation domain while preserving the existing MCP tool ecosystem.

The key insight is that the MCP tools and the prompt pipeline are complementary, not competing. MCP tools provide the data layer; the prompt pipeline provides the orchestration and context assembly layer.
