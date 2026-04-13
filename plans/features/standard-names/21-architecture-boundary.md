# Plan 21: Architecture Boundary and Prompt Infrastructure

> Establish the imas-codex ↔ imas-standard-names boundary.
> Own the generation pipeline. Import grammar as a library. Clean prompt infrastructure.

## Problem Statement

The standard name generation pipeline in imas-codex works end-to-end but has structural
debt from its rapid development:

1. **Private API coupling:** 5 imports from ISN's `tools/grammar.py` use `_` prefixed
   functions that can break on any ISN internal refactor.
2. **Hardcoded prompt content that belongs to ISN:** Composition rules, kind
   classification, anti-patterns in `compose_system.md` duplicate ISN's grammar
   authority. If ISN updates a naming convention, codex's inline rules won't know.
3. **No shared prompt fragments:** Grammar reference duplicated between compose and review
   prompts — a change to one must be manually mirrored to the other.
4. **ISN validation not imported:** codex's `validate_worker` uses grammar round-trip
   (parse→compose) but skips ISN's 8 semantic validation checks (geometric qualifier
   requirements, component/coordinate base type checks, orientation completeness, etc).
5. **Import path fragility:** `catalog_import.py` imports `StandardNameEntry` from
   `imas_standard_names.catalog.edit` which is a re-export — should import from
   `imas_standard_names.models` directly. ISN Plan 05 deletes `catalog/edit.py`.
6. **No formal boundary documentation:** Neither project documents what it owns vs delegates.

## Boundary Principle

**ISN owns ALL naming knowledge.** If ISN updates the grammar spec and a rule should
change, it belongs in ISN. Codex owns how to instruct an LLM to produce output and
how to evaluate that output.

**ISN-owned (via `get_grammar_context()`):**
- Grammar rules (canonical pattern, segment order, vocabulary)
- Naming conventions (composition rules, anti-patterns, reuse guidance)
- Kind definitions (scalar/vector/metadata classification)
- Documentation guidance, tag descriptions, applicability rules
- Field guidance, type-specific requirements

**Codex-owned (inline in prompts or codex config):**
- LLM output format (JSON schema instructions, candidate field list)
- DD-specific pipeline rules ("skip array indices", "no unit field", confidence thresholds)
- Review scoring criteria (6-dim rubric, tiers, verdicts)
- Calibration dataset

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

### Broken import in catalog_import.py

```python
# Lines 264, 370 — imports from module ISN Plan 05 deletes
from imas_standard_names.catalog.edit import StandardNameEntry
# Should be:
from imas_standard_names.models import StandardNameEntry
```

### Hardcoded guidelines in compose_system.md (lines 145-228)

Content analysis — what belongs where:

| Lines | Content | Owner | Action |
|-------|---------|-------|--------|
| 147-151 | Grammar rules (physical_base/geometric_base, pattern, tokens) | ISN | Move to ISN `get_grammar_context()` |
| 152 | "Reuse existing standard names" | ISN | Naming convention → ISN API |
| 153 | "Skip array indices, metadata" | Codex | DD classifier rule, keep in prompt |
| 154 | "Set confidence < 0.5" | Codex | Pipeline rule, keep in prompt |
| 155 | "Do NOT output unit field" | Codex | DD enrichment rule, keep in prompt |
| 157-179 | Output format / candidate schema | Codex | LLM output instructions |
| 180-194 | Documentation template | Mixed | ISN provides DOCUMENTATION_GUIDANCE, codex formats for LLM |
| 195-213 | Tags controlled vocabulary | ISN | Already imported from ISN tag_types |
| 215-219 | Kind classification | ISN | ISN defines Kind enum |
| 221-228 | Links guidance | Codex | Pipeline instructions |

### ISN semantic validation (not yet imported)

ISN's `validation/semantic.py` — `run_semantic_checks()`:
- **Signature:** `run_semantic_checks(entries: dict[str, StandardNameEntry]) -> list[str]`
- **Requires:** Pydantic `StandardNameEntry` models (not raw dicts)
- **Checks:** geometric qualifier requirements, component/coordinate base type,
  orientation vector completeness, trajectory/path qualification, extent dimensionality,
  physical_base+object, dimensionless quantities, provenance/operator gradient units

Codex's `validate_worker` operates on `list[dict]` from LLM output — impedance mismatch.
Phase 4 must include adapter code to construct `StandardNameEntry` from raw dicts.

### Duplicate grammar context

Both `compose_system.md` and `review.md` render grammar vocabulary and rules.
They do this independently — if grammar formatting changes, both must be updated.
A shared `{% include "sn/_grammar_reference.md" %}` fragment eliminates this.

The Jinja2 `PromptsLoader` searches `prompts/shared/` then `prompts/` root, so
`{% include "sn/_grammar_reference.md" %}` resolves to `prompts/sn/_grammar_reference.md`.
Verified in `prompt_loader.py` lines 1095-1113.

## Phases

### Phase 1: Replace Private ISN Imports and Fix Broken Paths

**Goal:** Use ISN's public `get_grammar_context()` API instead of 5 private functions.
Fix the `StandardNameEntry` import that will break when ISN removes `catalog/edit.py`.

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
  - ISN-owned naming guidelines now also come from `get_grammar_context()`:
    - `ctx["naming_guidance"]` → composition rules for prompt
    - `ctx["kind_definitions"]` → kind classification for prompt
    - `ctx["anti_patterns"]` → anti-pattern examples for prompt

- `imas_codex/sn/catalog_import.py` (lines 264, 370):
  - Change `from imas_standard_names.catalog.edit import StandardNameEntry`
    to `from imas_standard_names.models import StandardNameEntry`

**Tests:**
- `tests/sn/test_context.py` — verify `build_compose_context()` output unchanged
- `tests/sn/test_grammar_contract.py` (NEW) — API contract test:
  ```python
  def test_grammar_context_contract():
      from imas_standard_names.grammar.context import get_grammar_context
      ctx = get_grammar_context()
      required = {"canonical_pattern", "segment_order", "template_rules",
                  "vocabulary_sections", "segment_descriptions", "naming_guidance"}
      assert required <= set(ctx.keys())
  ```
- Run `uv run pytest tests/sn/` — all tests must pass

### Phase 2: Prompt Restructure — ISN Content via API, Codex Content via Includes

**Goal:** Remove hardcoded naming rules from prompts. ISN-owned content comes from
`get_grammar_context()` at render time. Codex-owned pipeline instructions use
`{% include %}` fragments for shared rendering.

**Key insight:** No YAML config files needed for naming guidelines. ISN's
`get_grammar_context()` is the single source of truth. Codex only needs a YAML
config for its review scoring criteria (used by both mint and benchmark).

**Files to create:**

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

- `imas_codex/llm/prompts/sn/_grammar_reference.md` (NEW ~40 lines):
  Shared grammar rendering (vocabulary, pattern, segment order) used by both
  compose and review prompts.

- `imas_codex/llm/prompts/sn/_scoring_rubric.md` (NEW ~30 lines):
  Review scoring dimensions rendered from `sn_review_criteria.yaml`.

**Files to modify:**

- `imas_codex/llm/prompts/sn/compose_system.md`:
  - Replace hardcoded grammar rules (lines 147-151) with ISN-provided
    `naming_guidance` from context (rendered via `{% include "sn/_grammar_reference.md" %}`)
  - Keep codex-specific pipeline instructions (lines 153-155) inline
  - Replace hardcoded kind classification (lines 215-219) with ISN-provided
    `kind_definitions` from context
  - Keep output format section (lines 157-179) — codex LLM instructions

- `imas_codex/llm/prompts/sn/review.md`:
  - Replace inline grammar section with `{% include "sn/_grammar_reference.md" %}`
  - Replace inline rubric with `{% include "sn/_scoring_rubric.md" %}`

- `imas_codex/llm/prompt_loader.py`:
  - Add helper to load YAML config files from `llm/config/`:
    ```python
    def load_prompt_config(name: str) -> dict:
        """Load YAML config from imas_codex/llm/config/{name}.yaml"""
    ```

**Tests:**
- Render compose_system.md and review.md, verify grammar section is character-identical
  between the two
- Verify ISN-owned content no longer hardcoded — comes from `get_grammar_context()`
- Full SN test suite passes

### Phase 3: Import ISN Semantic Validation

**Goal:** Use ISN's semantic checks in codex's `validate_worker`.

**Prerequisite:** ISN Plan 05 Phase 0 merged (stable imports).

**Files to modify:**

- `imas_codex/sn/workers.py` (`validate_worker`):
  - Add ISN semantic validation after parse→compose round-trip
  - **Adapter required:** ISN's `run_semantic_checks()` expects
    `dict[str, StandardNameEntry]` (Pydantic models). Codex has raw dicts.
    Build adapter:
    ```python
    from imas_standard_names.models import create_standard_name_entry
    from imas_standard_names.validation.semantic import run_semantic_checks

    def _run_isn_validation(candidates: list[dict]) -> dict[str, list[str]]:
        """Adapt ISN validation to work with raw LLM output dicts."""
        entries = {}
        for c in candidates:
            try:
                entry = create_standard_name_entry(**c)
                entries[c["standard_name"]] = entry
            except (ValidationError, KeyError):
                continue  # Skip malformed candidates
        issues = run_semantic_checks(entries)
        return _group_issues_by_name(issues)
    ```
  - Merge ISN validation issues into the existing soft-validation framework
  - ISN issues become warnings (not hard failures) — LLM output may not perfectly
    match ISN's entry schema

**New validation checks gained:**
1. Geometric qualifier requirements (orientation/path bases need object)
2. Component/coordinate with base type checks
3. Orientation vector completeness
4. Trajectory/path qualification
5. Extent dimensionality
6. Physical base + object checks
7. Dimensionless quantity detection
8. Provenance/operator gradient unit heuristics

**Tests:**
- `tests/sn/test_validate_integration.py` — verify ISN checks run within validate_worker
- Test that adapter handles malformed dicts gracefully (no crash)
- Test that a name with missing geometric qualifier gets flagged
- Full SN test suite passes

### Phase 4: Centralize Calibration Loading

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

### Phase 5: Documentation and Boundary Definition

**Goal:** Document the project boundary and updated architecture.

**Files to create:**

- `docs/architecture/boundary.md` (NEW):
  ```markdown
  # Project Boundary: imas-codex ↔ imas-standard-names

  ## imas-codex owns
  - Standard name **generation** (LLM pipeline: extract → compose → review →
    validate → consolidate → persist)
  - DD enrichment (clusters, siblings, coordinates, parent structures)
  - Quality review with calibrated 6-dimensional scoring
  - Graph storage and lifecycle management (drafted → published → accepted)
  - Benchmark framework for model evaluation
  - Catalog publish (graph → YAML) and import (YAML → graph) cycle

  ## imas-standard-names owns
  - Grammar specification (specification.yml → code-gen)
  - Vocabulary definition (YAML → StrEnums)
  - Parse/compose functions (string ↔ structured)
  - Naming conventions (composition rules, anti-patterns, kind definitions)
  - 4-layer validation (structural, semantic, description, quality)
  - Read-only catalog server (MCP tools for lookup/search)
  - Curated resources (examples, tokamak parameters, tag descriptions)

  ## Principle
  ISN defines what a valid standard name IS. Codex decides what names to CREATE.
  If a rule would change when ISN updates the grammar spec, it belongs in ISN.
  If a rule is about how to instruct an LLM, it belongs in codex.

  ## API contract
  - codex imports from ISN: `get_grammar_context()`, `parse_standard_name()`,
    `compose_standard_name()`, `run_semantic_checks()`,
    vocabulary constants, tag constants, curated resources
  - ISN exposes NO write operations via MCP or Python API
  - Changes to ISN grammar specification are coordinated releases
  ```

- Update `docs/architecture/standard-names.md`:
  - Add ISN integration section (what we import, why)
  - Update pipeline diagram to show validation integration

- Update `AGENTS.md`:
  - Add project boundary reference: `docs/architecture/boundary.md`
  - Document review criteria YAML config and its purpose
  - Document `{% include %}` fragments and when to create new ones

## Dependency Order

```
ISN Phase 0 (public API — expanded with naming guidelines)
    ↓
Codex Phase 1 (replace private imports + fix StandardNameEntry path)
    ↓
Codex Phase 2 (prompt restructure + shared fragments + review criteria YAML)
    ↓
Codex Phase 3 (ISN validation with adapter) ←── depends on Phase 1
    ↓
Codex Phase 4 (centralize calibration) ←── independent of Phase 3
    ↓
Codex Phase 5 (docs) ←── depends on all above
```

Phases 2 and 4 can run in parallel (no cross-dependency).
Phase 3 needs Phase 1 (ISN public API available + StandardNameEntry import fixed).

## Implementation Notes

- **Each phase is one commit.** Run `uv run pytest tests/sn/` after each.
- **Phase 1 is blocked on ISN Plan 05 Phase 0.** Start with Phase 4 while waiting.
- **Prompt output must be semantically identical** after Phase 2 — ISN API returns
  the same rules that were previously hardcoded. Diff rendered prompts to verify.
- **No behavioral change** in Phases 1-2, 4. Phase 3 adds new validations (may cause
  previously-passing names to get warnings — this is correct behavior).
- **No `sn_guidelines.yaml` in codex.** Naming knowledge comes from ISN. Only
  `sn_review_criteria.yaml` lives in codex (scoring is codex's evaluation framework).
- **Include path verified:** `PromptsLoader` searches `prompts/shared/` then `prompts/`
  root, so `{% include "sn/_grammar_reference.md" %}` resolves correctly.

## Documentation Updates

| Target | Phase |
|--------|-------|
| `docs/architecture/boundary.md` | Phase 5 — NEW |
| `docs/architecture/standard-names.md` | Phase 5 — update pipeline section |
| `AGENTS.md` | Phase 5 — boundary reference, YAML config docs |
| `plans/README.md` | Phase 5 — mark plan 21 done |
