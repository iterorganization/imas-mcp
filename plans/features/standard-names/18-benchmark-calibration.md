# 18: Benchmark Calibration, Model Selection & Reviewer Enhancement

**Status:** Ready to implement (after Plan 16)
**Depends on:** 16 (benchmark parity — results meaningless until prompts match build)
**Agent:** architect (requires research + design decisions)

## Problem

The benchmark needs improvements in three areas before we can make grounded model
selection decisions for production minting:

1. **Gold reference set is too small** — 30 entries from 4 IDSs
2. **No calibration dataset** — reviewer has no full-entry examples to anchor scores
3. **Reviewer prompt is ad-hoc** — inline string, no grammar context, no system prompt
4. **No structured approach to model selection** — we need cost/speed/quality tradeoffs

## Design Principle: Separate Gold Set from Calibration Set

The rubber-duck critique caught this: the reference set and calibration set serve
different purposes and should not be mixed.

| Dataset | Purpose | Contents | Location |
|---------|---------|----------|----------|
| **Gold reference** | Exact match: did model produce the right name? | `source_path → expected_name + fields` | `benchmark_reference.py` |
| **Calibration set** | Score anchoring: what does "outstanding" look like? | Full entries with expected tier + score | `benchmark_calibration.yaml` |

## Phase 1: Expand gold reference set

**Files:** `imas_codex/sn/benchmark_reference.py`

Expand from 30 to ~50 entries covering more IDSs:

| IDS | Current | Target |
|-----|---------|--------|
| equilibrium | 18 | 20 |
| core_profiles | 6 | 10 |
| magnetics | 4 | 6 |
| summary | 2 | 4 |
| core_transport | 0 | 4 |
| mhd_linear | 0 | 2 |
| nbi | 0 | 2 |
| edge_profiles | 0 | 2 |

Fix the questionable rogowski_coil reference entry (line 116-121 in
benchmark_reference.py — maps major_radius position to rogowski_coil object,
which is not physically meaningful).

## Phase 2: Create calibration dataset

**Files:** `imas_codex/sn/benchmark_calibration.yaml` (new)

Create ~15 hand-crafted full entries spanning quality tiers. Source material from
the `imas-standard-names` package's `resources/standard_name_examples/` directory
(40+ curated examples available).

### Structure

```yaml
# Each entry is a complete standard name with expected quality assessment
entries:
  - name: electron_temperature
    tier: outstanding
    expected_score: 95
    description: "Temperature of the electron population."
    documentation: >
      Electron temperature $T_e$ is a fundamental plasma parameter...
      Typical range: 0.1–30 keV in tokamak core...
    unit: eV
    kind: scalar
    tags: [core_profiles, equilibrium]
    fields:
      physical_base: temperature
      subject: electron
    reason: >
      Canonical physics quantity. Rich documentation with LaTeX, typical
      values, cross-references. Perfect grammar decomposition.

  - name: banana_orbits
    tier: poor
    expected_score: 15
    description: "Banana orbits"
    documentation: ""
    unit: null
    kind: metadata
    tags: []
    fields:
      physical_base: banana_orbits
    reason: >
      Not a measurable quantity. No documentation. "banana_orbits" is
      not a valid physical_base token. No unit possible.
```

### Tier distribution (15 entries)

| Tier | Count | Score Range | Examples |
|------|-------|-------------|----------|
| outstanding | 4 | 85-100 | electron_temperature, plasma_current, safety_factor, poloidal_magnetic_flux |
| good | 4 | 60-79 | toroidal_component_of_magnetic_field, electron_pressure, loop_voltage, stored_energy |
| adequate | 4 | 40-59 | minor_radius, aspect_ratio, time, elongation |
| poor | 3 | 0-39 | banana_orbits, h_mode, (invalid grammar example) |

Source the "outstanding" and "good" entries from `imas-standard-names` examples
where available. Hand-craft the "poor" examples to test error detection.

## Phase 3: Enhance reviewer prompt

**Files:**
- `imas_codex/llm/prompts/sn/review_benchmark.md` (new template)
- `imas_codex/sn/benchmark.py` — update `score_with_reviewer()`

Replace the inline rubric string with a proper Jinja2 template:

```markdown
---
name: sn/review_benchmark
description: Quality scoring for benchmark entries
used_by: imas_codex.sn.benchmark.score_with_reviewer
schema_needs: []
---

You are a physics nomenclature expert evaluating standard name entries.

## Grammar Rules
{{ canonical_pattern }}
{{ segment_order }}

## Scoring Rubric
...

## Calibration Examples

{% for entry in calibration_entries %}
### {{ entry.name }} — {{ entry.tier }} ({{ entry.expected_score }}/100)
{{ entry.reason }}
{% endfor %}

## Entries to Review
```

Key improvements:
- System/user message split (enables caching when reviewer runs multiple batches)
- Full calibration entries as anchors (not just names)
- Grammar rules included so reviewer can validate grammar correctness
- Structured rubric with specific scoring dimensions

### Scoring dimensions (each 0-20, sum = 0-100)

1. **Grammar correctness** (0-20): Valid parse, correct segment usage
2. **Semantic accuracy** (0-20): Name correctly describes the physics quantity
3. **Documentation quality** (0-20): LaTeX, typical values, cross-references
4. **Naming conventions** (0-20): Follows established patterns, consistent with peers
5. **Entry completeness** (0-20): Unit, kind, tags, links, ids_paths populated

Update `QualityReview` model:

```python
class QualityReview(BaseModel):
    name: str
    quality_tier: str
    score: int = Field(ge=0, le=100)
    grammar_score: int = Field(ge=0, le=20)
    semantic_score: int = Field(ge=0, le=20)
    documentation_score: int = Field(ge=0, le=20)
    convention_score: int = Field(ge=0, le=20)
    completeness_score: int = Field(ge=0, le=20)
    reasoning: str
```

## Phase 4: Model selection framework

### Candidate models for production minting

Based on the SN pipeline requirements (structured output, grammar adherence,
physics knowledge, documentation quality), evaluate:

**Compose phase (language model):**
| Model | Strengths | Concerns |
|-------|-----------|----------|
| `claude-sonnet-4` | Strong structured output, good physics | Cost |
| `claude-sonnet-4-5` | Previous gen, well-tested | Superseded |
| `gpt-4o` | Fast, good structured output | Physics depth |
| `gemini-2.5-flash` | Very fast, cheap | Grammar adherence unknown |
| `gpt-5.1` | Newest, potentially strongest | Cost, untested |

**Review phase (reasoning model):**
| Model | Strengths | Concerns |
|-------|-----------|----------|
| `claude-opus-4-6` | Deepest reasoning | Expensive |
| `o4-mini` | Good reasoning, cheaper | Structured output reliability |
| `claude-sonnet-4` | Good balance | May agree with itself if same family |

**Reviewer (benchmark scoring):**
| Model | Recommendation |
|-------|----------------|
| `claude-opus-4-6` | Best for calibration alignment |

### Benchmark execution plan

After Plans 16 is implemented:

```bash
# Phase A: Validate prompt caching works
imas-codex sn benchmark \
  --models openrouter/anthropic/claude-sonnet-4 \
  --ids equilibrium --max-candidates 20 -v
# Check logs for cache_read_input_tokens > 0 on second batch

# Phase B: Multi-model comparison (small)
imas-codex sn benchmark \
  --models openrouter/anthropic/claude-sonnet-4,openrouter/google/gemini-2.5-flash-preview,openrouter/openai/gpt-4o \
  --ids equilibrium --max-candidates 30 \
  --reviewer-model openrouter/anthropic/claude-opus-4-6

# Phase C: Winner deep-dive (larger)
imas-codex sn benchmark \
  --models <winner-from-B>,<runner-up> \
  --max-candidates 80 \
  --reviewer-model openrouter/anthropic/claude-opus-4-6 \
  --runs 2  # consistency check
```

### Decision criteria

| Metric | Weight | Threshold |
|--------|--------|-----------|
| Grammar valid % | Critical | ≥95% or disqualify |
| Fields consistent % | High | ≥85% |
| Reference recall | High | ≥60% |
| Avg quality score | High | ≥65 |
| Cost per name | Medium | <$0.01 preferred |
| Names/min | Medium | >30 preferred |
| Cache hit rate | Low | >50% confirms caching works |

## Phase 5: Token caching verification

Prompt caching is critical for cost efficiency at scale. After Plan 16 lands:

1. Run benchmark with a single model and 2+ batches
2. Check DEBUG logs for `cache_read_input_tokens` and `cache_creation_input_tokens`
3. First batch should show `cache_creation_input_tokens > 0`
4. Second batch should show `cache_read_input_tokens > 0`
5. If no cache fields in response, check:
   - Model string has `openrouter/` prefix (required for cache_control passthrough)
   - System message has `cache_control: {"type": "ephemeral"}` breakpoint
   - LiteLLM proxy is forwarding cache_control blocks

## Acceptance criteria

1. Gold reference expanded to 50+ entries across 8+ IDSs
2. Calibration dataset with 15 full entries across 4 quality tiers
3. Reviewer prompt uses Jinja2 template with system/user split
4. Reviewer produces 5-dimensional scores (grammar, semantic, docs, convention, completeness)
5. Benchmark results inform model selection with clear winner for compose + review
6. Token caching verified working end-to-end

## On renaming `build` → `mint`

**Decision: Not yet.** The rubber-duck critique agrees — the pipeline needs to
stabilize first. Revisit after the benchmark shows consistent, high-quality output
across model changes. When ready, add `mint` as the primary command name and keep
`build` as a hidden alias.

## Documentation updates

| Target | Update |
|--------|--------|
| `AGENTS.md` | Update SN benchmark section with new calibration workflow |
| `plans/features/standard-names/00-implementation-order.md` | Add plans 16-18 |

## Test plan

- Unit test: calibration YAML loads and validates
- Unit test: reviewer prompt renders with calibration entries
- Unit test: QualityReview model accepts 5-dimensional scores
- Unit test: expanded reference set passes grammar round-trip
- Integration: benchmark runs with reviewer and produces scored report
