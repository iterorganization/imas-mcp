# IMAS Codex Prompt Engineering Patterns

## Overview

The `imas-codex` project implements a mature, production-grade prompt engineering system built around **Jinja2 templating**, **schema-derived dynamic content injection**, **cached schema providers**, and **parallel worker dispatch with cost-bounded execution**. This document catalogues the patterns that have been proven across discovery, scoring, enrichment, mapping, and clustering pipelines.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     imas-codex Prompt System                        │
│                                                                     │
│  ┌──────────────────┐   ┌──────────────────┐   ┌────────────────┐  │
│  │  prompts/*.md    │   │  prompt_loader.py │   │ Schema         │  │
│  │  (Jinja2 +       │──▶│  parse/render/    │◀──│ Providers      │  │
│  │   frontmatter)   │   │  schema injection │   │ (@lru_cache)   │  │
│  └──────────────────┘   └────────┬─────────┘   └────────────────┘  │
│                                  │                                  │
│                    ┌─────────────▼──────────────┐                   │
│                    │   Rendered Prompt (string)  │                   │
│                    └─────────────┬──────────────┘                   │
│                                  │                                  │
│  ┌──────────────────┐   ┌───────▼──────────┐   ┌────────────────┐  │
│  │  discovery/       │   │  base/llm.py     │   │  LiteLLM +     │  │
│  │  base/engine.py   │──▶│  call_structured  │──▶│  OpenRouter    │  │
│  │  (workers,        │   │  (retry, cost,   │   │  (caching,     │  │
│  │   supervision)    │   │   Pydantic parse) │   │   routing)     │  │
│  └──────────────────┘   └──────────────────┘   └────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 1. Prompt File Format

Every prompt is a markdown file with YAML frontmatter [^1]:

```markdown
---
name: signals/enrichment
description: Batch signal enrichment for physics domain classification
used_by: imas_codex.discovery.signals.parallel.enrich_worker
task: enrichment
dynamic: true
schema_needs:
  - physics_domains
  - signal_enrichment_schema
  - diagnostic_categories
---

You are a tokamak physics expert classifying fusion facility data signals.

## Task
...

{% include "schema/physics-domains.md" %}

## Output Format
{{ signal_enrichment_schema_example }}

### Field Requirements
{{ signal_enrichment_schema_fields }}
```

### Frontmatter Fields

| Field | Purpose |
|-------|---------|
| `name` | Prompt identifier with path prefix (e.g., `signals/enrichment`) |
| `description` | Human-readable purpose |
| `used_by` | Python module that consumes this prompt |
| `task` | Task classification for model selection |
| `dynamic` | Whether the prompt requires Jinja2 rendering |
| `schema_needs` | List of schema providers to inject |

### Key Design Principle: Static-First Ordering

Prompts are structured with **static content first, dynamic content last**. This enables API-level prompt caching (Anthropic/Gemini cache breakpoints):

1. Role definition and task description (static)
2. Schema references via `{% include %}` (static, shared)
3. Scoring dimensions via `{% for dim in ... %}` (semi-static, cached per process)
4. Per-call context via `{{ variable }}` (dynamic, changes every call)

**Source**: `imas_codex/llm/prompt_loader.py:1-53` [^1]

## 2. Include System

Shared prompt fragments live in `prompts/shared/` and are included via Jinja2:

```
prompts/shared/
├── safety.md             # Read-only policy, resource limits
├── tools.md              # Tool usage guidelines (rg, fd, tokei)
├── completion.md         # Completion criteria (MVE, full exploration)
└── schema/               # Schema-derived template fragments
    ├── path-purposes.md
    ├── physics-domains.md
    ├── score-dimensions.md
    ├── scoring-output.md
    ├── score-output.md
    ├── discovery-categories.md
    ├── diagnostic-categories.md
    ├── dimension-calibration.md
    └── ... (14 files total)
```

Include resolution happens at two levels [^2]:
1. **Static includes** (`parse_prompt_file`): Regex-based `{% include "file.md" %}` for MCP registration
2. **Dynamic includes** (`render_prompt`): Full Jinja2 environment with template loader

This separation means prompts can be registered as MCP resources (with includes resolved) without requiring the full Jinja2 rendering pipeline.

**Source**: `imas_codex/llm/prompt_loader.py:87-106` [^2]

## 3. Schema Provider System

The most powerful pattern in codex is **cached schema providers** — functions decorated with `@lru_cache` that load domain schemas once per process and inject them into prompts:

```python
@lru_cache(maxsize=1)
def _provide_score_dimensions() -> dict[str, Any]:
    """Provide score_* field definitions from FacilityPath schema."""
    schema = _get_linkml_schema()
    facility_path_slots = schema.get_all_slots("FacilityPath")
    score_dimensions = [
        {"field": name, "label": ..., "description": desc}
        for name, slot_info in facility_path_slots.items()
        if name.startswith("score_") and slot_info.get("type") == "float"
    ]
    return {"score_dimensions": score_dimensions}
```

### Provider Registry

A central registry maps `schema_needs` names to provider functions [^3]:

```python
_SCHEMA_PROVIDERS = {
    "path_purposes": _provide_path_purposes,
    "discovery_categories": _provide_discovery_categories,
    "score_dimensions": _provide_score_dimensions,
    "scoring_schema": _provide_scoring_schema,
    "physics_domains": _provide_physics_domains,
    "signal_enrichment_schema": _provide_signal_enrichment_schema,
    "diagnostic_categories": _provide_diagnostic_categories,
    "imas_enrichment_schema": _provide_imas_enrichment_schema,
    "cluster_vocabularies": _provide_cluster_vocabularies,
    # ... 20+ providers total
}
```

### Pydantic Schema Injection

Two helper functions generate prompt-embeddable schema representations [^4]:

1. **`get_pydantic_schema_json()`** — Generates JSON example instances from Pydantic models
2. **`get_pydantic_schema_description()`** — Generates markdown field descriptions with types

These are used in prompts as:
```markdown
{{ signal_enrichment_schema_example }}   <!-- JSON structure example -->
{{ signal_enrichment_schema_fields }}    <!-- Markdown field descriptions -->
```

**Source**: `imas_codex/llm/prompt_loader.py:211-357` [^4]

## 4. Two-Mode Rendering

The prompt system supports two rendering modes:

### Static Mode (`parse_prompt_file`)
- Resolves `{% include %}` directives only
- Used for MCP tool registration (prompts as resources)
- No Jinja2 variables or loops evaluated

### Dynamic Mode (`render_prompt`)
- Full Jinja2 rendering with schema context
- Loads only the schema providers declared in `schema_needs`
- Merges additional per-call context (facility name, calibration data)

```python
def render_prompt(name, context=None, prompts_dir=None):
    prompt_def = prompts[name]
    schema_needs = prompt_def.metadata.get("schema_needs")
    full_context = get_schema_for_prompt(name, schema_needs)
    if context:
        full_context.update(context)
    template = env.from_string(prompt_def.content)
    return template.render(full_context)
```

**Source**: `imas_codex/llm/prompt_loader.py:1116-1163` [^5]

## 5. System/User Prompt Separation

For mapping pipelines, codex uses a **system + user prompt** pattern [^6]:

| File | Role | Content |
|------|------|---------|
| `signal_mapping_system.md` | System prompt (static, cacheable) | Task rules, scoring criteria, COCOS handling, output format |
| `signal_mapping.md` | User prompt (dynamic, per-call) | Facility, IDS, signal source, IMAS fields, code references |

The system prompt stays constant across all calls in a pipeline run, enabling API-level caching. The user prompt changes per signal source with Jinja2 variables like `{{ facility }}`, `{{ signal_source_detail }}`, `{{ imas_fields }}`.

This pattern repeats across all mapping stages:
- `target_assignment_system.md` + `target_assignment.md`
- `assembly_system.md` + `assembly.md`
- `metadata_population_system.md` + `metadata_population.md`
- `validation.md` (combined)

## 6. Calibration Example Injection

Scoring prompts include real examples from the knowledge graph for score calibration [^7]:

```markdown
{% if score_calibration %}
## Score Calibration Examples

{% for category, examples in score_calibration.items() %}
{% for ex in examples %}
- `{{ ex.path }}` ({{ ex.facility }}) — score={{ ex.score }}, {{ ex.total_lines }} LOC
{% endfor %}
{% endfor %}
{% endif %}
```

This pattern provides **grounded scoring** — the LLM sees what previous scores look like for similar items, reducing drift and improving consistency across batches.

## 7. LLM Infrastructure (`discovery/base/llm.py`)

The codex LLM layer provides [^8]:

### Structured Output with Retry
```python
batch, cost, tokens = call_llm_structured(
    model="google/gemini-3-flash-preview",
    messages=[{"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt}],
    response_model=ScoreBatch,  # Pydantic model
)
```

- Retry wraps **both** API call and Pydantic parsing
- Exponential backoff: 5s base, doubles each retry
- Retryable patterns: 429, 503, timeout, JSON parse errors, validation errors
- Budget exhaustion detection (402 Payment Required → halt all workers)

### Cost Tracking
- Per-call cost extraction from LiteLLM response
- Accumulated cost across workers
- CLI-level budget limits (`--cost-limit 5.0`)
- Time limits (`--time 10` minutes)

### Model-Aware Token Limits
```python
MODEL_TOKEN_LIMITS = {
    "gemini": {"max_tokens": 16000, "timeout": 120},
    "claude": {"max_tokens": 32000, "timeout": 120},
}
```

### Prompt Caching
- `inject_cache_control()` adds cache breakpoints to messages
- Static system prompts designed for 5-minute Anthropic / Gemini cache windows
- `test_prompt_caching.py` script validates cache hit rates across providers

## 8. Parallel Worker Engine

The discovery engine (`discovery/base/engine.py`) provides [^9]:

```python
result = await run_discovery_engine(
    state=my_state,
    workers=[
        WorkerSpec("scan", "scan_phase", scan_worker, count=2),
        WorkerSpec("triage", "triage_phase", triage_worker, count=3),
        WorkerSpec("enrich", "enrich_phase", enrich_worker, depends_on=["scan_phase"]),
        WorkerSpec("score", "score_phase", score_worker, depends_on=["triage_phase"]),
    ],
    orphan_specs=[OrphanRecoverySpec(...)],
    stop_event=stop_event,
)
```

**Features**:
- **Phase dependencies**: Workers wait for upstream phases
- **Supervision**: Orphan recovery for stuck items
- **Stop events**: Graceful shutdown on budget/time exhaustion
- **Progress tracking**: Per-worker stats (items processed, errors, cost)
- **Cost-based termination**: Scorer halts when budget depleted

## 9. CLI Dispatch Pattern

Every discovery/mapping pipeline is exposed as a CLI command [^10]:

```bash
# Discovery pipelines
imas-codex discover paths tcv --cost-limit 5.0 --time 30
imas-codex discover signals jet --enrich-workers 4
imas-codex discover wiki tcv --scan-only

# Mapping pipeline
imas-codex imas map jet -d magnetic_field_systems --cost-limit 25 --model sonnet
```

Each CLI command:
1. Configures logging and services
2. Loads facility configuration
3. Renders prompts with facility-specific context
4. Dispatches parallel workers with budget/time limits
5. Reports progress with Rich progress bars
6. Persists results to Neo4j graph

## 10. Prompt Organization Summary

```
imas_codex/llm/prompts/
├── shared/                    # Reusable fragments (safety, tools, schema)
│   ├── safety.md
│   ├── tools.md
│   ├── completion.md
│   └── schema/               # 14 schema-derived template fragments
├── paths/                    # Directory discovery/scoring
│   ├── triage.md             # First-pass classification
│   └── scorer.md             # Evidence-based rescoring
├── code/                     # Source file scoring
│   ├── triage.md
│   └── scorer.md
├── signals/                  # Signal enrichment
│   ├── enrichment.md         # Batch physics classification
│   ├── individualization.md
│   └── source_unwind.md
├── discovery/                # Frontier discovery
│   ├── roots.md
│   ├── enricher.md
│   ├── data_access.md
│   └── static-enricher.md
├── exploration/              # Interactive exploration
│   └── facility.md
├── wiki/                     # Wiki scoring
│   ├── scout.md
│   ├── scorer.md
│   ├── document-scorer.md
│   └── image-captioner.md
├── mapping/                  # IMAS signal mapping
│   ├── target_assignment_system.md
│   ├── target_assignment.md
│   ├── signal_mapping_system.md
│   ├── signal_mapping.md
│   ├── assembly_system.md
│   ├── assembly.md
│   ├── metadata_population_system.md
│   ├── metadata_population.md
│   └── validation.md
├── imas/                     # IMAS DD enrichment
│   ├── enrichment.md
│   ├── identifier_enrichment.md
│   └── ids_enrichment.md
└── clusters/                 # Semantic cluster labeling
    └── labeler.md
```

**Total**: 30+ prompt files across 9 domains, with 14 shared schema fragments.

---

## Footnotes

[^1]: `imas_codex/llm/prompt_loader.py:1-53` — Module docstring and prompt format specification
[^2]: `imas_codex/llm/prompt_loader.py:87-106` — `_resolve_includes()` function
[^3]: `imas_codex/llm/prompt_loader.py:878-905` — `_SCHEMA_PROVIDERS` registry
[^4]: `imas_codex/llm/prompt_loader.py:211-357` — `get_pydantic_schema_json()` and `get_pydantic_schema_description()`
[^5]: `imas_codex/llm/prompt_loader.py:1116-1163` — `render_prompt()` function
[^6]: `imas_codex/llm/prompts/mapping/signal_mapping_system.md` and `signal_mapping.md`
[^7]: `imas_codex/llm/prompts/paths/scorer.md:174-192` — Score calibration template
[^8]: `imas_codex/discovery/base/llm.py:1-120` — LLM infrastructure module
[^9]: `imas_codex/discovery/base/engine.py:1-100` — Discovery engine skeleton
[^10]: `imas_codex/cli/discover/signals.py:1-80` — CLI dispatch pattern
