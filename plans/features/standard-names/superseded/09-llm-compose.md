# 09: Schema Providers + LLM Compose

**Status:** Pending
**Priority:** Critical — this is THE blocking gap
**Depends on:** Nothing (all infrastructure exists)
**Effort:** 2-3 days

## Problem

Two coupled problems must be solved together:

### A. Impoverished prompt context

The compose prompt (`sn/compose_dd.md`) receives bare enum lists from
`build_grammar_context()`:
```python
{"subjects": ["electron", "ion", ...], "positions": ["magnetic_axis", ...]}
```

Meanwhile, the imas-standard-names library has rich prompting infrastructure
that produces ~800 lines of structured context:
- 42 real examples grouped by composition pattern
- Segment descriptions with critical distinctions (component vs coordinate)
- Template application rules, exclusivity constraints
- Usage guidance per segment
- Tokamak parameters for grounding documentation examples
- Field schema guidance with common mistakes

These are available as **direct Python imports** — no MCP needed:
```python
from imas_standard_names.grammar.constants import (
    SEGMENT_RULES, SEGMENT_ORDER, SEGMENT_TEMPLATES,
    SEGMENT_TOKEN_MAP, EXCLUSIVE_SEGMENT_PAIRS,
    APPLICABILITY_INCLUDE, GENERIC_PHYSICAL_BASES,
)
from imas_standard_names.grammar.field_schemas import (
    FIELD_GUIDANCE, TYPE_SPECIFIC_REQUIREMENTS,
)
from imas_standard_names.tools.grammar import (
    _build_canonical_pattern, _build_segment_order_constraint,
    _get_segment_descriptions, _build_template_application_rule,
    _build_segment_usage_guidance,
)
```

### B. Heuristic compose worker

The `compose_worker` in `workers.py` uses a keyword-matching heuristic that
recognizes only 13 hardcoded words. The benchmark module already has a working
LLM compose pattern using `acall_llm_structured()`.

## Approach

### Phase 1a: Schema Provider System

Create `imas_codex/standard_names/schema_providers.py` implementing the 3-tier caching
design from `plans/research/standard-names/06-schema-provider-design.md`.

**Tier 1: Process-lifetime (static, ~15KB)**

These import directly from `imas_standard_names` and are cached with
`@lru_cache(maxsize=1)`. They never change during a pipeline run.

| Provider | Source | Content |
|----------|--------|---------|
| `grammar_context` | `grammar.constants.*`, `tools.grammar._build_*` | Canonical pattern, segment order, template rules, exclusivity pairs |
| `segment_descriptions` | `tools.grammar._get_segment_descriptions()` | Per-segment rich descriptions with critical distinctions |
| `segment_usage_guidance` | `tools.grammar._build_segment_usage_guidance()` | Usage patterns, example constructions per segment |
| `vocabulary_tokens` | `SEGMENT_TOKEN_MAP` | Token lists per segment with counts |
| `field_schema_guidance` | `FIELD_GUIDANCE`, `TYPE_SPECIFIC_REQUIREMENTS` | Per-field validation rules, common mistakes |

**Tier 2: Catalog-lifetime (~4KB)**

Loaded from the existing StandardName graph nodes or YAML catalog.
Invalidated on write. Cached per session.

| Provider | Source | Content |
|----------|--------|---------|
| `existing_names_summary` | Graph query | Count, by-kind breakdown, sample names |
| `example_entries` | `resources/standard_name_examples/` (42 YAML files) | Curated high-quality entries for few-shot prompting |

**Tier 3: Per-call dynamic (~3KB)**

Assembled fresh for each LLM batch call.

| Provider | Source | Content |
|----------|--------|---------|
| `dd_paths_context` | Extract worker output | Projected path info for the batch's IDS |
| `tokamak_parameters` | `resources/tokamak_parameters/*.yml` | Real machine dimensions for grounding doc examples |

**Entry point:**
```python
async def get_schema_for_prompt(
    schema_needs: list[str],
    *,
    dynamic_context: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Load requested providers and merge their output as prompt variables."""
```

Each prompt template declares its `schema_needs` in frontmatter — only the
requested providers are loaded. This keeps prompt size predictable (~20KB).

### Phase 1b: Rewrite compose prompt

Rewrite `llm/prompts/sn/compose_dd.md` to consume the rich schema provider
output instead of bare enum lists. The prompt should reference:
- `{{ grammar_context }}` — canonical pattern, segment order, template rules
- `{{ segment_descriptions }}` — critical distinctions per segment
- `{{ vocabulary_tokens }}` — valid tokens per segment
- `{{ example_entries }}` — few-shot examples from the curated set
- `{{ dd_paths_context }}` — DD path info for the current batch
- `{{ tokamak_parameters }}` — machine parameters for grounding examples

### Phase 1c: LLM compose worker

Replace `_compose_single()` with batch LLM composition mirroring the
benchmark's `_run_model()` pattern:

```python
async def compose_worker(state: SNBuildState, **_kwargs) -> None:
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.schema_providers import get_schema_for_prompt
    from imas_codex.standard_names.models import SNComposeBatch
    from imas_codex.settings import get_model

    model = get_model("language")

    # Load static + catalog context once
    static_context = await get_schema_for_prompt([
        "grammar_context", "segment_descriptions", "vocabulary_tokens",
        "field_schema_guidance", "example_entries", "existing_names_summary",
    ])

    # Group candidates by IDS for coherent batches
    batches = _group_candidates_by_ids(state.candidates)

    composed = []
    for batch in batches:
        # Load per-call context for this batch
        dynamic_context = await get_schema_for_prompt(
            ["dd_paths_context", "tokamak_parameters"],
            dynamic_context={"ids_name": batch["group_key"], "items": batch["items"]},
        )
        prompt_context = {**static_context, **dynamic_context, **batch}
        prompt_text = render_prompt("sn/compose_dd", prompt_context)
        messages = [{"role": "user", "content": prompt_text}]

        result, cost, tokens = await acall_llm_structured(
            model=model, messages=messages, response_model=SNComposeBatch,
        )
        state.compose_stats.cost += cost
        for c in result.candidates:
            composed.append(c.model_dump())

    state.composed = composed
    state.compose_phase.mark_done()
```

## Files to Create/Modify

### New: `imas_codex/standard_names/schema_providers.py`

The schema provider system with 3-tier caching.

### Modify: `imas_codex/llm/prompts/sn/compose_dd.md`

Rewrite to consume rich schema provider variables instead of bare enums.
Add `schema_needs` frontmatter declaring required providers.

### Modify: `imas_codex/standard_names/workers.py`

- Delete `_compose_single()` and `_extract_physical_base()` (lines 172-247)
- Rewrite `compose_worker()` to use batch LLM calls with schema providers
- Rename `state.validated` → `state.composed` (see plan 10)

### Modify: `imas_codex/standard_names/state.py`

- Add `composed: list[dict]` field (rename from `validated`)

### Modify: `imas_codex/standard_names/benchmark.py`

- Replace `build_grammar_context()` with `get_schema_for_prompt()` call
- This ensures benchmark and pipeline use identical prompt context

## Acceptance Criteria

- Schema providers return ~15KB of static context (grammar, segments, vocabulary)
- `sn build --source dd --ids equilibrium` calls the LLM and produces real names
- Grammar round-trip validation passes for >80% of composed names
- Cost tracked in `state.compose_stats.cost`
- `--dry-run` still works (skips LLM, reports candidate count)
- Benchmark continues to work with the same schema providers

## Testing

- Unit test: schema providers return expected content structure and size
- Unit test: `@lru_cache` actually caches Tier 1 providers
- Integration: `sn build --source dd --ids equilibrium --dry-run`
- Integration: `sn build --source dd --ids equilibrium` (end-to-end)
- Integration: `sn benchmark --ids equilibrium --max-candidates 10`
- Quality: composed names parse via grammar round-trip

## Tokamak Parameters for Documentation

The 42 standard name example files in `resources/standard_name_examples/`
include `documentation` fields with LaTeX-formatted content referencing
real physical dimensions (e.g., "ITER major radius R₀ = 6.2 m"). The
tokamak parameters database (12 YAML files with sourced values) prevents
hallucinations in these documentation strings.

For Phase 1, tokamak parameters are loaded as a per-call provider and
injected into the prompt alongside DD path context. The LLM uses real
machine parameters (major radius, minor radius, B_T, I_p, etc.) when
generating documentation examples instead of inventing plausible but
incorrect values.

## Notes

- The compose prompt already handles skipping metadata/index paths
- Existing names list prevents duplicates in the prompt
- `SNComposeBatch` model includes both `candidates` and `skipped` lists
- **Import backing functions directly** from `imas_standard_names` — never
  call MCP tools from the pipeline
- The `_build_*` functions in `tools/grammar.py` are private but stable —
  consider making them public or extracting their logic into `constants.py`
