# Plan 21: Architecture Boundary and Prompt Infrastructure

> Establish the imas-codex ↔ imas-standard-names boundary.
> Own the generation pipeline. Import grammar as a library. Clean prompt infrastructure.

## Problem Statement

The standard name generation pipeline in imas-codex works end-to-end but has structural
debt from its rapid development:

1. **Private API coupling:** 5 imports from ISN's `tools/grammar.py` use `_` prefixed
   functions that can break on any ISN internal refactor.
2. **Hardcoded prompt guidelines:** 96 lines of composition rules are inline in
   `compose_system.md` instead of managed as structured YAML configuration.
3. **No shared prompt fragments:** Grammar reference duplicated between compose and review
   prompts — a change to one must be manually mirrored to the other.
4. **ISN validation not imported:** codex's `validate_worker` uses grammar round-trip
   (parse→compose) but skips ISN's 8 semantic validation checks (unit consistency,
   tag physics domain alignment, description quality, etc).
5. **Import path fragility:** `catalog_import.py` imports `StandardNameEntry` from
   `imas_standard_names.catalog.edit` which is a re-export — should import from
   `imas_standard_names.models` directly.
6. **No formal boundary documentation:** Neither project documents what it owns vs delegates.

## Approach

Six phases, each independently testable and committable. Phases 1-3 are internal
refactors with no behavioral change. Phases 4-5 add new capability. Phase 6 documents.

## Evidence: Current State

### Private API imports (imas_codex/sn/context.py)

```python
from imas_standard_names.tools.grammar import (
    _build_canonical_pattern,         # Returns composition pattern string
    _build_segment_order_constraint,  # Returns ordering constraint text
    _build_template_application_rule, # Returns template usage rules
    _get_segment_descriptions,        # Returns per-segment descriptions dict
    _get_vocabulary_description,      # Returns formatted vocab section list
)
```

These will be replaced by ISN's new `get_grammar_context()` public API (ISN Plan 05, Phase 0).

### Hardcoded guidelines in compose_system.md (lines 146-228)

```markdown
## Composition Rules
1. Each standard name must describe exactly one physical quantity...
2. Use only tokens from the vocabulary above...
...
## Output Schema
Return a JSON array matching this schema...
## Kind Classification
| Kind | Use when... |
...
## Links Guidance
- `ids_paths`: ...
```

These 83 lines of rules should live in a YAML config file so they can be:
- Versioned independently of template structure
- Shared across prompts (compose and review both need rules)
- Edited without touching Jinja2 syntax

### Duplicate grammar context

Both `compose_system.md` and `review.md` render grammar vocabulary and rules.
They do this independently — if grammar formatting changes, both must be updated.
A shared `{% include "sn/_grammar_reference.md" %}` fragment eliminates this.

### ISN semantic validation (not yet imported)

ISN's `validation/semantic.py` has 8 checks codex doesn't use:
1. Unit consistency (SI prefixes, compound units)
2. Tag physics domain alignment
3. Coordinate system validity
4. Duplicate name detection
5. Link format validation
6. Vocabulary token usage check
7. Description metadata leakage
8. Kind-specific field requirements

Codex's `validate_worker` only does parse→compose round-trip.

## Phases

### Phase 1: Replace Private ISN Imports

**Goal:** Use ISN's public `get_grammar_context()` API instead of 5 private functions.

**Prerequisite:** ISN Plan 05 Phase 0 is merged and released.

**Files to modify:**

- `imas_codex/sn/context.py`:
  - Replace the 5 private imports with single import:
    ```python
    from imas_standard_names.grammar.context import get_grammar_context
    ```
  - In `build_compose_context()`, call `ctx = get_grammar_context()` once
  - Map context keys to existing template variables:
    - `ctx["canonical_pattern"]` → `canonical_pattern`
    - `ctx["segment_order"]` → `segment_order`
    - `ctx["template_rules"]` → `template_rules`
    - `ctx["vocabulary_sections"]` → `vocabulary_sections`
    - `ctx["segment_descriptions"]` → `segment_descriptions`

- `imas_codex/sn/catalog_import.py`:
  - Change import from `imas_standard_names.catalog.edit` to `imas_standard_names.models`
    (ISN plan removes catalog.edit — prepare now)

**Tests:**
- Update `tests/sn/test_context.py` (if exists) or add tests verifying `build_compose_context()`
  output is unchanged after the import switch
- Run `uv run pytest tests/sn/` — all 437 tests must pass

**Verification:**
- Diff the output of `build_compose_context()` before and after — keys and values identical

### Phase 2: Extract Guidelines to YAML Config

**Goal:** Move hardcoded prose guidelines from prompts into structured YAML config files.

**Files to create:**

- `imas_codex/llm/config/sn_guidelines.yaml` (NEW ~150 lines):
  ```yaml
  composition_rules:
    - rule: "Each standard name describes exactly one physical quantity"
      category: identity
    - rule: "Use only tokens from the provided vocabulary"
      category: vocabulary
    ...

  output_schema:
    description: "Return JSON matching the SNComposeBatch schema"
    notes:
      - "Each candidate has: name, kind, description, documentation, tags, links"

  kind_classification:
    scalar: "Single value per time point (Te, Ip, q95)"
    vector: "Array of values (profiles, spectra)"
    metadata: "Non-physical (diagnostic names, code versions)"

  links_guidance:
    ids_paths: "Full IMAS path from root IDS"
    dd_paths: "Relative path within IDS"

  anti_patterns:
    - pattern: "Embedding units in the name"
      example: "electron_temperature_ev"
      correction: "electron_temperature (unit comes from DD)"
    - pattern: "Overly specific names"
      example: "core_profiles_1d_electron_temperature_fit"
      correction: "electron_temperature_fit"
    ...

  unit_policy:
    source: "DD HAS_UNIT relationship"
    rule: "Never invent units. Use DD unit exactly as provided."
    when_missing: "Leave unit field empty, flag as vocab_gap"
  ```

- `imas_codex/llm/config/sn_review_criteria.yaml` (NEW ~60 lines):
  ```yaml
  dimensions:
    grammar:
      weight: 20
      description: "Parse/compose round-trip, segment ordering, vocabulary compliance"
    semantic:
      weight: 20
      description: "Name accurately describes the physical quantity"
    documentation:
      weight: 20
      description: "Description and documentation are clear, complete, non-redundant"
    convention:
      weight: 20
      description: "Follows naming conventions (no units in name, no path fragments)"
    completeness:
      weight: 20
      description: "All required fields filled, links populated, tags relevant"
    compliance:
      weight: 20
      description: "Follows compose prompt instructions exactly"

  verdict_rules:
    accept: "total >= 72 AND no dimension is 0"
    reject: "total < 48 OR any dimension is 0"
    revise: "all other cases"

  tiers:
    outstanding: { min: 102, label: "Outstanding" }
    good: { min: 72, label: "Good" }
    adequate: { min: 48, label: "Adequate" }
    poor: { min: 0, label: "Poor" }
  ```

**Files to modify:**

- `imas_codex/llm/prompts/sn/compose_system.md`:
  - Replace hardcoded rules (lines ~146-228) with:
    ```jinja
    {% include "sn/_guidelines.md" %}
    ```
  - The include fragment reads from YAML config at render time

- `imas_codex/llm/prompts/sn/review.md`:
  - Replace hardcoded rubric with YAML-driven rendering
  - Scoring dimensions loaded from `sn_review_criteria.yaml`

- `imas_codex/llm/prompt_loader.py`:
  - Add helper to load YAML config files from `llm/config/`:
    ```python
    def load_prompt_config(name: str) -> dict:
        """Load YAML config from imas_codex/llm/config/{name}.yaml"""
    ```
  - Or integrate with existing `render_prompt()` to auto-load config YAML as template context

**Tests:**
- `tests/llm/test_prompt_config.py` — verify YAML configs load and validate
- Render compose_system.md and review.md, verify output matches pre-refactor output
- Full SN test suite passes

### Phase 3: Create Shared Prompt Fragments

**Goal:** Eliminate duplication between compose and review prompts via `{% include %}`.

**Files to create:**

- `imas_codex/llm/prompts/sn/_grammar_reference.md` (NEW ~40 lines):
  ```jinja
  ## Grammar Reference

  **Canonical Pattern:** {{ canonical_pattern }}

  **Segment Order:** {{ segment_order }}

  **Vocabulary:**
  {% for section in vocabulary_sections %}
  ### {{ section.segment }}
  {% for token in section.tokens %}
  - `{{ token.name }}`: {{ token.description }}
  {% endfor %}
  {% endfor %}
  ```

- `imas_codex/llm/prompts/sn/_guidelines.md` (NEW ~50 lines):
  ```jinja
  {# Renders composition rules, anti-patterns, unit policy from YAML config #}
  ## Composition Rules
  {% for rule in guidelines.composition_rules %}
  {{ loop.index }}. {{ rule.rule }}
  {% endfor %}

  ## Anti-Patterns
  {% for ap in guidelines.anti_patterns %}
  - ❌ `{{ ap.example }}` → ✅ `{{ ap.correction }}` ({{ ap.pattern }})
  {% endfor %}

  ## Unit Policy
  {{ guidelines.unit_policy.rule }}
  ```

- `imas_codex/llm/prompts/sn/_scoring_rubric.md` (NEW ~30 lines):
  ```jinja
  {# Renders review scoring dimensions from YAML config #}
  ## Scoring Rubric ({{ criteria.dimensions | length }} dimensions, 0-{{ total_max }} total)
  {% for name, dim in criteria.dimensions.items() %}
  ### {{ name | title }} (0-{{ dim.weight }})
  {{ dim.description }}
  {% endfor %}
  ```

**Files to modify:**

- `imas_codex/llm/prompts/sn/compose_system.md`:
  - Add `{% include "sn/_grammar_reference.md" %}`
  - Add `{% include "sn/_guidelines.md" %}`
  - Remove inline grammar and guidelines sections

- `imas_codex/llm/prompts/sn/review.md`:
  - Add `{% include "sn/_grammar_reference.md" %}`
  - Add `{% include "sn/_scoring_rubric.md" %}`
  - Remove inline grammar and rubric sections

**Tests:**
- Render both prompts, verify grammar section is character-identical
- Full SN test suite passes

### Phase 4: Import ISN Semantic Validation

**Goal:** Use ISN's 8 semantic checks in codex's `validate_worker`.

**Files to modify:**

- `imas_codex/sn/workers.py` (`validate_worker`):
  - Add import: `from imas_standard_names.validation.semantic import validate_semantic`
  - After parse→compose round-trip, run ISN semantic checks:
    ```python
    from imas_standard_names.validation.semantic import validate_semantic
    from imas_standard_names.validation.description import validate_description

    semantic_issues = validate_semantic(entry_dict)
    description_issues = validate_description(entry_dict)
    ```
  - Merge ISN validation issues into the existing soft-validation framework
  - Map ISN issue severity to codex's warning/error classification

**New validation checks gained:**
1. Unit consistency (SI prefix + compound unit validation)
2. Tag-physics domain alignment
3. Coordinate system validity
4. Description metadata leakage detection
5. Kind-specific field requirements

**Tests:**
- `tests/sn/test_validate_integration.py` — verify ISN checks run within validate_worker
- Test that a name with a bad unit gets flagged by ISN checks
- Test that a name with metadata leakage in description gets flagged
- Full SN test suite passes

### Phase 5: Centralize Calibration Loading

**Goal:** Single calibration loading path for both mint review and benchmark.

**Files to create:**

- `imas_codex/sn/calibration.py` (NEW ~40 lines):
  ```python
  """Calibration dataset loader for quality review."""

  import importlib.resources
  import yaml

  _CACHE: list[dict] | None = None

  def load_calibration() -> list[dict]:
      """Load benchmark calibration entries (cached)."""
      global _CACHE
      if _CACHE is None:
          ref = importlib.resources.files("imas_codex.sn") / "benchmark_calibration.yaml"
          _CACHE = yaml.safe_load(ref.read_text())
      return _CACHE

  def get_calibration_for_prompt() -> list[dict]:
      """Return calibration entries formatted for prompt rendering."""
      return [
          {
              "name": e["name"],
              "tier": e["tier"],
              "total": e["total"],
              "reasoning": e["reasoning"],
          }
          for e in load_calibration()
      ]
  ```

**Files to modify:**

- `imas_codex/sn/workers.py` (`review_worker`):
  - Replace inline calibration loading with `from imas_codex.sn.calibration import load_calibration`

- `imas_codex/sn/benchmark.py` (`score_with_reviewer`):
  - Replace inline calibration loading with `from imas_codex.sn.calibration import load_calibration`

**Tests:**
- `tests/sn/test_calibration.py` — verify loading, caching, prompt formatting
- Verify both mint and benchmark use identical calibration data

### Phase 6: Documentation and Boundary Definition

**Goal:** Document the project boundary and updated architecture.

**Files to create:**

- `docs/architecture/boundary.md` (NEW):
  ```markdown
  # Project Boundary: imas-codex ↔ imas-standard-names

  ## imas-codex owns
  - Standard name **generation** (LLM pipeline: extract → compose → review → validate → consolidate → persist)
  - DD enrichment (clusters, siblings, coordinates, parent structures)
  - Quality review with calibrated 6-dimensional scoring
  - Graph storage and lifecycle management (drafted → published → accepted)
  - Benchmark framework for model evaluation
  - Catalog publish (graph → YAML) and import (YAML → graph) cycle

  ## imas-standard-names owns
  - Grammar specification (specification.yml → code-gen)
  - Vocabulary definition (YAML → StrEnums)
  - Parse/compose functions (string ↔ structured)
  - 4-layer validation (structural, semantic, description, quality)
  - Read-only catalog server (MCP tools for lookup/search)
  - Curated resources (examples, tokamak parameters, tag descriptions)

  ## Data flow
  1. codex `sn mint` → generates candidates from DD paths using LLM
  2. codex `sn publish` → exports validated names to YAML catalog files
  3. Human review → accepts/rejects/revises in YAML
  4. codex `sn import` → imports reviewed catalog back to graph
  5. ISN `standard-names build` → builds .db from YAML catalog
  6. ISN MCP tools → serve .db to LLM agents for lookup

  ## API contract
  - codex imports from ISN: `get_grammar_context()`, `parse_standard_name()`,
    `compose_standard_name()`, `validate_semantic()`, `validate_description()`,
    vocabulary constants, tag constants, curated resources
  - ISN exposes NO write operations via MCP or Python API
  - Changes to ISN grammar specification are coordinated releases
  ```

- Update `docs/architecture/standard-names.md`:
  - Add prompt infrastructure section (YAML configs, shared fragments)
  - Add ISN integration section (what we import, why)
  - Update pipeline diagram to show validation integration

- Update `AGENTS.md`:
  - Add project boundary reference: `docs/architecture/boundary.md`
  - Document YAML config files and their purpose
  - Document `{% include %}` fragments and when to create new ones

## Dependency Order

```
ISN Phase 0 (public API)
    ↓
Codex Phase 1 (replace private imports)
    ↓
Codex Phase 2 (YAML configs) ←── independent of Phase 1
    ↓
Codex Phase 3 (shared fragments) ←── depends on Phase 2
    ↓
Codex Phase 4 (ISN validation) ←── depends on Phase 1
    ↓
Codex Phase 5 (centralize calibration) ←── independent
    ↓
Codex Phase 6 (docs) ←── depends on all above
```

Phases 2 and 5 can run in parallel with Phase 1 (no cross-dependency).
Phase 4 needs Phase 1 (ISN public API available).
Phase 3 needs Phase 2 (YAML configs exist to include).

## Implementation Notes

- **Each phase is one commit.** Run `uv run pytest tests/sn/` after each.
- **Phase 1 is blocked on ISN Plan 05 Phase 0.** Start with Phase 2 while waiting.
- **Prompt output must be identical** after Phases 2-3. Diff rendered prompts to verify.
- **No behavioral change** in Phases 1-3, 5. Phase 4 adds new validations (may cause
  previously-passing names to get warnings — this is correct behavior).
- **YAML config files are NOT prompts.** They live in `imas_codex/llm/config/`, not
  in `imas_codex/llm/prompts/`. Prompts `{% include %}` fragments that read from configs.

## Documentation Updates

| Target | Phase |
|--------|-------|
| `docs/architecture/boundary.md` | Phase 6 — NEW |
| `docs/architecture/standard-names.md` | Phase 6 — update pipeline section |
| `AGENTS.md` | Phase 6 — boundary reference, YAML config docs |
| `plans/README.md` | Phase 6 — mark plan 21 done |
