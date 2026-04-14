# 14: Standard Name MCP Tools & Benchmark Enhancement

**Status:** Ready to implement
**Depends on:** 11 (rich schema + embedding wiring)
**Enables:** MCP-assisted SN queries, model quality evaluation
**Agent:** engineer (MCP tools follow established patterns in llm/server.py)

## Problem A: No SN search/fetch in imas-codex MCP

The imas-codex MCP server (`imas_codex/llm/server.py`) has no tools for querying
standard names from the graph. Users and agents can't discover, search, or fetch
standard name entries. The graph is the richest data source (embeddings, DD links,
facility signal links, grammar decomposition).

**Prerequisite:** Plan 11 Phase 4d wires embedding generation into the persist
worker. Without embeddings, vector search returns zero results. If deploying
before embeddings are wired, fall back to keyword-only search.

## Problem B: Benchmark is grammar-only

The `sn benchmark` command only measures grammar validity and reference overlap.
It doesn't evaluate documentation quality, naming conventions, semantic accuracy,
or entry completeness. No quality tiers. No reviewer model.

## Part A: SN MCP Tools

### Phase 1: Search standard names

**Files:**
- `imas_codex/llm/sn_tools.py` — new module (follows search_tools.py pattern)
- `imas_codex/llm/server.py` — register tools

Add `search_standard_names` tool:

```python
@tool
def search_standard_names(
    query: str,
    kind: str | None = None,
    tags: list[str] | None = None,
    review_status: str | None = None,
    k: int = 20,
) -> str:
    """Search standard names by physics concept.

    Hybrid search (vector + keyword) over StandardName descriptions
    and documentation. Enriched with DD path links, unit info, and
    grammar decomposition.
    """
```

Uses `standard_name_desc_embedding` vector index for semantic search,
combined with keyword matching on name/tags. Falls back to keyword-only
if no embeddings present.

### Phase 2: Fetch standard names

Add `fetch_standard_names` tool:

```python
@tool
def fetch_standard_names(names: str) -> str:
    """Fetch full entries for known standard names.

    Returns complete metadata: description, documentation, unit, kind,
    tags, links, ids_paths, grammar fields, provenance, review status.
    """
```

### Phase 3: List standard names

Add `list_standard_names` tool:

```python
@tool
def list_standard_names(
    tag: str | None = None,
    kind: str | None = None,
    review_status: str | None = None,
) -> str:
    """List standard names with optional filters.

    Returns name, description, kind, unit, status for each entry.
    """
```

**Acceptance:**
- MCP server exposes 3 SN tools
- `search_standard_names("electron temperature")` returns relevant names
- `fetch_standard_names("electron_temperature")` returns full entry
- `list_standard_names(tag="equilibrium")` returns filtered names

## Part B: Benchmark Enhancement

### Phase 4: Quality tier labels

**Files:** `imas_codex/standard_names/benchmark_labels.yaml` — new file

Curate quality labels for ~20 benchmark reference entries:

```yaml
outstanding:  # Rich docs, correct grammar, cross-linked, LaTeX
  - electron_temperature
  - plasma_current
  - safety_factor
  - position_of_magnetic_axis
  - bootstrap_current
good:  # Correct grammar, adequate docs
  - toroidal_component_of_magnetic_field_at_magnetic_axis
  - centroid_of_plasma_boundary
  - bolometer_radiated_power
  - collisionality
adequate:  # Correct grammar, thin docs
  - area_of_poloidal_magnetic_field_probe
  - tokamak_scenario
  - time
poor:  # Grammar valid but naming debatable
  - banana_orbits
  - h_mode
```

### Phase 5: Reviewer model

**Files:** `imas_codex/standard_names/benchmark.py`, `imas_codex/llm/prompts/sn/review_benchmark.md`

Add `--reviewer-model` option that uses a frontier model to evaluate outputs:

```bash
imas-codex sn benchmark \
  --models anthropic/claude-sonnet-4-6,google/gemini-2.5-flash \
  --reviewer-model anthropic/claude-opus-4-6 \
  --ids equilibrium --limit 50
```

The reviewer receives:
1. Grammar rules (same as compose)
2. Labeled examples at each quality tier
3. Generated entry (name + all fields)
4. Rubric: grammar correctness, semantic accuracy, documentation quality,
   naming conventions, unit consistency

Returns per-entry: quality tier, score (0-100), reasoning.

New metrics in benchmark report:
- **Quality distribution**: % outstanding / good / adequate / poor per model
- **Documentation richness**: avg doc length, equation count, cross-ref count
- **Grammar pattern coverage**: does model use subject, component, position, etc.
- **Entry completeness**: % of fields populated per entry

**Acceptance:**
- `sn benchmark --reviewer-model <model>` produces quality-scored report
- Report includes quality distribution table per model
- Labeled examples used as scoring anchors

## Test Plan

- Unit tests for MCP tool registration and response format
- Unit test: search with and without embeddings (keyword fallback)
- Unit test: benchmark loads quality tier labels
- Unit test: quality tier classification against labeled examples
- Integration test: search returns graph-resident standard names
