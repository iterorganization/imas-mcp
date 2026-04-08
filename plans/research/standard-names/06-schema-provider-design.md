# Schema Provider Design

## Overview

Schema providers are functions that assemble domain context (grammar rules, vocabulary, field schemas, catalog data) for injection into prompt templates. Rather than having the LLM discover this context via MCP tool calls at runtime, providers pre-assemble it at prompt render time.

The design uses a 3-tier caching strategy matched to the volatility of each data source. Static data (grammar, vocabulary) is cached for process lifetime. Catalog-derived data (existing names, examples) is cached per catalog load. Dynamic data (DD queries) is assembled per call.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Schema Provider System                     │
│                                                              │
│  Tier 1: Process-Lifetime (static, ~13KB total)              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐  │
│  │ grammar_     │ │ vocabulary_  │ │ field_schema_        │  │
│  │ overview     │ │ tokens       │ │ guidance             │  │
│  │ (~4KB)       │ │ (~2KB)       │ │ (~4KB)               │  │
│  └──────┬───────┘ └──────┬───────┘ └──────────┬───────────┘  │
│         │                │                     │              │
│  ┌──────▼────────────────▼─────────────────────▼───────────┐  │
│  │ Direct imports from grammar.constants, field_schemas     │  │
│  │ No MCP round-trips for static data                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                              │
│  Tier 2: Catalog-Lifetime (~3KB per domain)                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐  │
│  │ existing_    │ │ tag_         │ │ example_             │  │
│  │ names_       │ │ taxonomy     │ │ entries              │  │
│  │ inventory    │ │ (~1KB)       │ │ (~2KB)               │  │
│  └──────┬───────┘ └──────┬───────┘ └──────────┬───────────┘  │
│         │                │                     │              │
│  ┌──────▼────────────────▼─────────────────────▼───────────┐  │
│  │ Loaded from catalog Repository at startup, invalidated   │  │
│  │ on write_standard_names                                  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                              │
│  Tier 3: Per-Call (dynamic, ~3KB per query)                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐  │
│  │ dd_cluster   │ │ dd_path_     │ │ dd_cocos_            │  │
│  │ context      │ │ docs         │ │ context              │  │
│  │ (~1.5KB)     │ │ (~1.5KB)     │ │ (~0.5KB)             │  │
│  └──────┬───────┘ └──────┬───────┘ └──────────┬───────────┘  │
│         │                │                     │              │
│  ┌──────▼────────────────▼─────────────────────▼───────────┐  │
│  │ Assembled per call via DDAdapter (direct imports from    │  │
│  │ imas-codex backing functions — no MCP round-trips)       │  │
│  │ Projection applied to reduce size                        │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Tier 1: Process-Lifetime Providers (Static)

These providers import directly from the grammar module's auto-generated constants. They never touch MCP or the network. Cache with `@lru_cache(maxsize=1)`.

### `grammar_overview` (~4KB)

**Backing source**: `grammar.constants.SEGMENT_RULES`, `SEGMENT_ORDER`, `SEGMENT_TEMPLATES`

**Content**: Grammar composition rules, segment ordering, template patterns. Equivalent to the output of `get_grammar()` MCP tool but without MCP overhead.

```python
from imas_standard_names.grammar.constants import (
    SEGMENT_ORDER,
    SEGMENT_RULES,
    SEGMENT_TEMPLATES,
)

@lru_cache(maxsize=1)
def _provide_grammar_overview() -> dict[str, str]:
    """Assemble grammar rules for prompt injection."""
    rules_text = format_segment_rules(SEGMENT_RULES, SEGMENT_ORDER)
    templates_text = format_templates(SEGMENT_TEMPLATES)
    return {
        "grammar_overview": rules_text,
        "grammar_templates": templates_text,
    }
```

### `vocabulary_tokens` (~2KB)

**Backing source**: `grammar.constants.SEGMENT_TOKEN_MAP`

**Content**: Token lists for each vocabulary segment (component, subject, device, object, geometry, position, process, coordinate). Static list of ~200 tokens across all segments.

```python
from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP

@lru_cache(maxsize=1)
def _provide_vocabulary_tokens() -> dict[str, str]:
    """Assemble vocabulary tokens for prompt injection."""
    sections = []
    for segment, tokens in SEGMENT_TOKEN_MAP.items():
        sections.append(f"### {segment}\n{', '.join(sorted(tokens))}")
    return {"vocabulary_tokens": "\n\n".join(sections)}
```

**Note on usage statistics**: The MCP `get_vocabulary()` tool includes usage frequency counts, which require iterating all 309 catalog entries (~expensive). For prompt injection, raw token lists suffice. Usage stats are only relevant for interactive exploration.

### `field_schema_guidance` (~4KB)

**Backing source**: `grammar.field_schemas.FIELD_GUIDANCE`, `TYPE_SPECIFIC_REQUIREMENTS`

**Content**: Per-field documentation for standard name entries (name, kind, unit, tags, description, status). Includes naming guidance, prohibited patterns, and examples.

```python
from imas_standard_names.grammar.field_schemas import (
    FIELD_GUIDANCE,
    TYPE_SPECIFIC_REQUIREMENTS,
)

@lru_cache(maxsize=1)
def _provide_field_schema_guidance() -> dict[str, str]:
    """Assemble field schema guidance for prompt injection."""
    guidance_text = format_field_guidance(FIELD_GUIDANCE)
    type_reqs_text = format_type_requirements(TYPE_SPECIFIC_REQUIREMENTS)
    return {
        "field_guidance": guidance_text,
        "type_requirements": type_reqs_text,
    }
```

### `entry_schema` (~3KB)

**Backing source**: `models.StandardNameEntry` Pydantic model

**Content**: JSON schema example + field descriptions generated from the Pydantic model. Shows the LLM the exact structure it should produce.

```python
@lru_cache(maxsize=1)
def _provide_entry_schema() -> dict[str, str]:
    """Generate schema example and field docs from Pydantic model."""
    return {
        "entry_schema_example": get_pydantic_schema_json(StandardNameEntry),
        "entry_schema_fields": get_pydantic_schema_description(StandardNameEntry),
    }
```

## Tier 2: Catalog-Lifetime Providers

These providers read from the loaded catalog. They are invalidated when the catalog changes (e.g., after `write_standard_names`). Use a cache that can be explicitly cleared.

### `existing_names_summary` (~1KB)

**Content**: Count of existing names, breakdown by kind, sample of recent entries. Provides catalog awareness without listing every name.

```python
def _provide_existing_names_summary(repository: Repository) -> dict[str, str]:
    """Summarize existing catalog for context."""
    names = repository.list_all()
    by_kind = Counter(n.kind for n in names)
    samples = names[:10]  # Most recent
    return {
        "existing_names_count": str(len(names)),
        "existing_names_by_kind": format_counter(by_kind),
        "existing_names_samples": format_name_list(samples),
    }
```

### `tag_taxonomy` (~1KB)

**Content**: Available tags with their usage counts. Helps the LLM choose appropriate tags.

### `example_entries` (~2KB)

**Content**: Curated high-quality entries as YAML examples. Used in few-shot prompting.

## Tier 3: Per-Call Providers (Dynamic)

These assemble context specific to the current generation/review request. Not cached — assembled fresh each call.

### `dd_paths_context` (~1.5KB projected)

**Backing source**: IMAS DD MCP tools (`list_imas_paths`, `fetch_imas_paths`)

**Content**: Projected path information for the target IDS/domain. The full `search_imas()` output is ~79KB — far too large. Apply projection to extract only path, type, units, and description (~3KB for a typical IDS).

```python
async def _provide_dd_paths_context(
    ids_name: str,
    paths: list[str],
) -> dict[str, str]:
    """Assemble DD path context for specific generation request."""
    path_docs = await fetch_imas_paths(paths, ids=ids_name)
    projected = project_path_docs(path_docs, fields=["path", "type", "units", "doc"])
    return {"dd_paths_context": format_path_table(projected)}
```

### `domain_existing_names` (~1.5KB)

**Content**: Names already in the catalog for the target domain. Prevents duplicates and ensures naming consistency within the domain.

## Provider Registry

```python
_STATIC_PROVIDERS: dict[str, Callable] = {
    "grammar_overview": _provide_grammar_overview,
    "vocabulary_tokens": _provide_vocabulary_tokens,
    "field_schema_guidance": _provide_field_schema_guidance,
    "entry_schema": _provide_entry_schema,
}

_CATALOG_PROVIDERS: dict[str, Callable] = {
    "existing_names_summary": _provide_existing_names_summary,
    "tag_taxonomy": _provide_tag_taxonomy,
    "example_entries": _provide_example_entries,
}

_DYNAMIC_PROVIDERS: dict[str, Callable] = {
    "dd_paths_context": _provide_dd_paths_context,
    "domain_existing_names": _provide_domain_existing_names,
}

async def get_schema_for_prompt(
    schema_needs: list[str],
    *,
    repository: Repository | None = None,
    dynamic_context: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Load only the requested providers and merge their output."""
    context: dict[str, str] = {}
    for need in schema_needs:
        if need in _STATIC_PROVIDERS:
            context.update(_STATIC_PROVIDERS[need]())
        elif need in _CATALOG_PROVIDERS:
            if repository is None:
                raise ValueError(f"Catalog provider '{need}' requires a repository")
            context.update(_CATALOG_PROVIDERS[need](repository))
        elif need in _DYNAMIC_PROVIDERS:
            if dynamic_context is None:
                raise ValueError(f"Dynamic provider '{need}' requires context")
            context.update(await _DYNAMIC_PROVIDERS[need](**dynamic_context))
    return context
```

## Size Budget Analysis

| Provider | Tier | Size | Cache Duration |
|----------|------|------|---------------|
| grammar_overview | 1 | ~4KB | Process lifetime |
| vocabulary_tokens | 1 | ~2KB | Process lifetime |
| field_schema_guidance | 1 | ~4KB | Process lifetime |
| entry_schema | 1 | ~3KB | Process lifetime |
| **Tier 1 subtotal** | | **~13KB** | |
| existing_names_summary | 2 | ~1KB | Catalog lifetime |
| tag_taxonomy | 2 | ~1KB | Catalog lifetime |
| example_entries | 2 | ~2KB | Catalog lifetime |
| **Tier 2 subtotal** | | **~4KB** | |
| dd_paths_context | 3 | ~1.5KB | Per call |
| domain_existing_names | 3 | ~1.5KB | Per call |
| **Tier 3 subtotal** | | **~3KB** | |
| **Total per prompt** | | **~20KB** | |

At ~20KB per rendered prompt, a generation system prompt fits well within model context windows (128K+ for modern models). The static 13KB prefix is ideal for API-level prompt caching.

## Caching Strategy

### Tier 1: `@lru_cache(maxsize=1)`

Grammar constants and Pydantic schemas never change during a process lifecycle. Simple `functools.lru_cache` is sufficient.

### Tier 2: Explicit cache with invalidation

Catalog data changes when `write_standard_names` is called. Use a simple cache dict with a `clear()` method that the repository calls on write:

```python
_catalog_cache: dict[str, Any] = {}

def clear_catalog_cache() -> None:
    """Invalidate all catalog-derived caches."""
    _catalog_cache.clear()
```

### Tier 3: No caching

Per-call providers assemble fresh context each time. The data is small (~3KB) and changes with every request.

## Mapping: Existing Functions → Providers

| Existing Function/Module | Provider | Adaptation Needed |
|--------------------------|----------|-------------------|
| `grammar.constants.SEGMENT_RULES` | `grammar_overview` | Format as markdown |
| `grammar.constants.SEGMENT_ORDER` | `grammar_overview` | Include in rules formatting |
| `grammar.constants.SEGMENT_TEMPLATES` | `grammar_overview` | Format template examples |
| `grammar.constants.SEGMENT_TOKEN_MAP` | `vocabulary_tokens` | Format as token lists |
| `grammar.field_schemas.FIELD_GUIDANCE` | `field_schema_guidance` | Format as markdown sections |
| `grammar.field_schemas.TYPE_SPECIFIC_REQUIREMENTS` | `field_schema_guidance` | Format per-kind requirements |
| `models.StandardNameEntry` | `entry_schema` | Pydantic schema helpers (adapt from codex) |
| `Repository.list_all()` | `existing_names_summary` | Aggregate + sample |
| `tools.grammar.GrammarTool.get_grammar()` | (bypassed) | Direct import instead |
| `tools.vocabulary.VocabularyTool.get_vocabulary()` | (bypassed) | Direct import instead |
| `tools.schema.SchemaTool.get_schema()` | (bypassed) | Direct import instead |

## Implementation Path

### Phase 1: Static providers (Wave 1, with loader core)

1. Create `imas_standard_names/prompts/providers.py`
2. Implement `_provide_grammar_overview()` — direct import from constants
3. Implement `_provide_vocabulary_tokens()` — direct import from constants
4. Implement `_provide_field_schema_guidance()` — direct import from field_schemas
5. Implement `_provide_entry_schema()` — Pydantic schema helpers
6. Create `_STATIC_PROVIDERS` registry
7. Write formatting helpers (`format_segment_rules`, `format_templates`, etc.)
8. Tests: verify output size, content correctness, caching behavior

### Phase 2: Catalog providers (Wave 1, after repository integration)

1. Implement `_provide_existing_names_summary()`
2. Implement `_provide_tag_taxonomy()`
3. Implement `_provide_example_entries()`
4. Add cache invalidation hook to repository write path
5. Tests: verify invalidation, content correctness

### Phase 3: Dynamic providers (Wave 2, with pipeline workers)

1. Implement `_provide_dd_paths_context()` — async, uses DD tools
2. Implement `_provide_domain_existing_names()` — async, uses catalog
3. Implement DD response projection (reduce ~79KB → ~3KB)
4. Implement `get_schema_for_prompt()` unified entry point
5. Tests: verify projection, async behavior, error handling

## Design Decision: Direct Import vs MCP

For static providers, we bypass the MCP tool layer and import directly from the grammar module. Rationale:

1. **Performance**: No serialization/deserialization overhead
2. **Simplicity**: No async MCP round-trip for deterministic data
3. **Reliability**: No MCP server availability dependency
4. **Type safety**: Direct Python objects, not JSON strings

MCP tools remain the interface for interactive use (human-operated chat sessions). The provider system is the interface for batch pipelines.
