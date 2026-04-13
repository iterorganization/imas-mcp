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

### Phase 3: Full ISN Validation via Pydantic Model Construction

**Goal:** Construct `StandardNameEntry` Pydantic models in validate_worker to fire ALL
27 ISN validators. This is the single highest-value change — it closes 15 validation
gaps in one step.

**Prerequisite:** ISN Plan 05 Phase 0 merged (stable imports).

**Critical insight:** Codex currently NEVER constructs `StandardNameEntry` Pydantic
models from LLM output. This means ALL 18 field validators that fire during model
construction are bypassed, plus 9 post-construction semantic checks. The current
validate_worker only does grammar round-trip (parse → compose) and soft metadata checks.

**Architecture: Three-layer validation**

```
Layer 1: Pydantic model construction (create_standard_name_entry)
         → 18 field validators fire automatically
         → Catches: name pattern, grammar vocabulary consistency, link format,
            physics domain vocabulary, secondary tag vocabulary, sign convention
            format, unit canonicalization + pint validation, description max_length,
            provenance rules

Layer 2: Post-construction semantic checks (run_semantic_checks)
         → 9 grammar-semantic checks on constructed models
         → Catches: geometric qualifier requirements, component/coordinate base
            type, orientation completeness, trajectory qualification, extent
            dimensionality, physical base + object, dimensionless detection

Layer 3: Description quality checks (validate_description)
         → Metadata leakage detection (tag-redundant phrases, structural phrases)
```

**Files to modify:**

- `imas_codex/sn/workers.py` (`validate_worker`):

  Replace the current soft-validation section (lines 695-753) with ISN model
  construction + semantic validation:

  ```python
  from imas_standard_names.models import create_standard_name_entry
  from imas_standard_names.validation.semantic import run_semantic_checks
  from imas_standard_names.validation.description import validate_description
  from pydantic import ValidationError

  def _validate_via_isn(entry: dict) -> tuple[StandardNameEntry | None, list[str]]:
      """Construct ISN Pydantic model and collect all validation issues.

      Returns (model_or_None, list_of_issue_strings).
      If model construction fails, returns (None, [error_messages]).
      """
      issues = []

      # Map codex dict keys to ISN model fields
      isn_dict = {
          "name": entry.get("id", ""),
          "kind": entry.get("kind", "scalar"),
          "description": entry.get("description", ""),
          "documentation": entry.get("documentation", ""),
          "unit": entry.get("unit", ""),
          "tags": entry.get("tags", []),
          "links": entry.get("links", []),
          "physics_domain": entry.get("physics_domain", ""),
          "ids_paths": entry.get("imas_paths", []),
          "validity_domain": entry.get("validity_domain"),
          "constraints": entry.get("constraints", []),
      }

      # Layer 1: Pydantic model construction (fires 18 validators)
      try:
          model = create_standard_name_entry(isn_dict)
      except ValidationError as e:
          for err in e.errors():
              field = ".".join(str(loc) for loc in err["loc"])
              issues.append(f"[pydantic:{field}] {err['msg']}")
          return None, issues

      # Layer 2: Semantic checks (9 grammar-semantic checks)
      semantic_issues = run_semantic_checks({isn_dict["name"]: model})
      issues.extend(f"[semantic] {i}" for i in semantic_issues)

      # Layer 3: Description quality (metadata leakage)
      desc_issues = validate_description(isn_dict)
      issues.extend(f"[description] {i}" for i in desc_issues)

      return model, issues
  ```

  **Classification of ISN issues:**
  - **Hard fail** (entry rejected): Pydantic `ValidationError` on name, kind, or grammar
    fields (name won't round-trip or is structurally invalid)
  - **Soft fail** (entry passes with warnings): Pydantic errors on description length,
    sign convention format, link format. Semantic and description issues.
  - **Auto-fix** (entry corrected silently): Unit canonicalization (ISN reorders tokens),
    tag normalization (ISN strips whitespace, removes empties)

  Separate hard vs soft based on which validator field the error comes from:
  ```python
  HARD_FAIL_FIELDS = {"name", "kind"}  # structural validity
  # Everything else is a warning — the name is valid but entry quality is low
  ```

  **Benefit of model construction:**
  - Unit canonicalization happens automatically (ISN sorts unit tokens
    lexicographically, rejects `/` and `*` syntax)
  - If pint is available, dimensional analysis validates the unit string
  - The constructed `StandardNameEntry` can be cached for downstream use
    (e.g., direct persistence to ISN catalog format)

- `imas_codex/sn/workers.py` — Update stats tracking:
  - Add counters: `isn_model_ok`, `isn_model_fail`, `isn_semantic_issues`,
    `isn_description_issues`
  - Log aggregate stats at end of validate phase

**Validation checks gained (27 total from 3 layers):**

Layer 1 — Pydantic (18 checks):
1. Name pattern: no `__`, lowercase, `^[a-z][a-z0-9_]*$`
2. Grammar vocabulary consistency (component_of → Component enum, etc.)
3. Link format validation (http:// or name:xxx pattern)
4. Physics domain controlled vocabulary (32 valid values)
5. Secondary tag controlled vocabulary (50+ values)
6. Documentation sign convention format
7. Unit canonicalization (reject `/`, `*`, whitespace; sort tokens)
8. Unit pint dimensional analysis
9. Provenance operator/reduction naming rules
10. Description max_length=180
11. Deprecated governance (deprecated → must have superseded_by)
12. Tags/constraints list normalization
13. Extra fields rejection (ConfigDict extra="forbid")
14-18. Provenance sub-model validators (operator token pattern, base pattern, etc.)

Layer 2 — Semantic (9 checks):
19. Geometric qualifier requirements
20. Component with base type check
21. Coordinate with base type check
22. Orientation vector completeness
23. Trajectory/path qualification
24. Extent dimensionality
25. Physical base + object check
26. Dimensionless quantity detection
27. Gradient operator unit heuristic

Layer 3 — Description (2+ checks):
28. Tag-redundant phrase detection
29. Structural phrase detection

**Tests:**
- `tests/sn/test_validate_isn.py` (NEW):
  - Test model construction succeeds for well-formed LLM output
  - Test model construction catches double-underscore names
  - Test unit canonicalization auto-fixes token order
  - Test pint validation rejects nonsense units
  - Test link format validation catches malformed links
  - Test physics domain validation catches invalid values
  - Test semantic checks flag missing geometric qualifiers
  - Test description checks flag metadata leakage
  - Test hard vs soft failure classification
  - Test graceful handling of malformed dicts (no crash)
- Run `uv run pytest tests/sn/` — all tests must pass

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

### Phase 5: Close Context Enrichment Gaps

**Goal:** Import the ISN context areas where ISN provides richer data than codex,
closing the 4 gaps identified in the A/B comparison.

**Context gaps from A/B analysis (ISN wins):**

| Gap | ISN provides | Codex status |
|-----|-------------|-------------|
| Quick-start / common patterns | 5-step guide, 11 patterns, critical distinctions | Not provided to LLM |
| Vocabulary usage statistics | Token frequencies, most_common, unused tokens | Not provided |
| Existing name search | FTS+fuzzy search, 20 results per concept | Flat list from graph |
| Base JSON schema | Full Pydantic JSON schema via `get_schema()` | `get_pydantic_schema_json()` for compose only |

**Files to modify:**

- **ISN Plan 05 Phase 0 dependency**: `get_grammar_context()` must expose:
  - `quick_start`: 5-step generation guide
  - `common_patterns`: 11 frequent patterns with examples
  - `critical_distinctions`: Base vs modifier, orientation vs direction, etc.
  - `vocabulary_usage_stats`: Per-segment frequency data from catalog

  If ISN doesn't add these in Phase 0, request a Phase 0.1 addition.

- `imas_codex/sn/context.py` (`build_compose_context()`):
  - Add `quick_start`, `common_patterns`, `critical_distinctions` from
    `get_grammar_context()` to compose context dict
  - Add `vocabulary_usage_stats` for LLM to understand token frequency

- `imas_codex/llm/prompts/sn/compose_system.md`:
  - Add quick-start section before grammar reference (helps LLM orientation)
  - Add common patterns section with "do this, not that" examples
  - Add token frequency note to vocabulary section ("commonly used:", "rarely used:")

- `imas_codex/sn/workers.py` (`review_worker`):
  - Add existing-name search to review context: query graph for names with
    similar tokens to each candidate, provide as "nearby names" context
  - This replaces ISN's `search_standard_names()` capability
  - Use simple prefix/token matching on graph, not semantic search (fast)

**Tests:**
- Verify `build_compose_context()` includes all new keys
- Verify compose_system.md renders quick-start and common-patterns sections
- Verify review context includes nearby existing names
- Full SN test suite passes

### Phase 6: Documentation and Boundary Definition

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
    `compose_standard_name()`, `create_standard_name_entry()`,
    `run_semantic_checks()`, `validate_description()`,
    vocabulary constants, tag constants, curated resources
  - ISN exposes NO write operations via MCP or Python API
  - Changes to ISN grammar specification are coordinated releases
  ```

- Update `docs/architecture/standard-names.md`:
  - Add ISN integration section (what we import, why)
  - Update pipeline diagram to show 3-layer validation
  - Add A/B comparison summary table

- Update `AGENTS.md`:
  - Add project boundary reference: `docs/architecture/boundary.md`
  - Document review criteria YAML config and its purpose
  - Document `{% include %}` fragments and when to create new ones
  - Document 3-layer validation architecture (Pydantic → semantic → description)

## Dependency Order

```
ISN Phase 0 (public API — expanded with naming guidelines, common patterns, usage stats)
    ↓
Codex Phase 1 (replace private imports + fix StandardNameEntry path)
    ↓
Codex Phase 2 (prompt restructure + shared fragments + review criteria YAML)
    ↓
Codex Phase 3 (full ISN Pydantic validation) ←── depends on Phase 1
    ↓
Codex Phase 4 (centralize calibration) ←── independent of Phase 3
    ↓
Codex Phase 5 (context enrichment gaps) ←── depends on Phase 1 + ISN Phase 0.1
    ↓
Codex Phase 6 (docs) ←── depends on all above
```

Phases 2, 4, and 5 can run in parallel (no cross-dependency).
Phase 3 needs Phase 1 (ISN public API available + StandardNameEntry import fixed).
Phase 5 needs ISN Phase 0 to expose common_patterns and vocabulary_usage_stats.

## Implementation Notes

- **Each phase is one commit.** Run `uv run pytest tests/sn/` after each.
- **Phase 1 is blocked on ISN Plan 05 Phase 0.** Start with Phase 4 while waiting.
- **Prompt output must be semantically identical** after Phase 2 — ISN API returns
  the same rules that were previously hardcoded. Diff rendered prompts to verify.
- **No behavioral change** in Phases 1-2, 4. Phase 3 adds ISN Pydantic validation
  (27 checks) — may cause previously-passing names to get warnings or fail. This
  is correct behavior: codex now enforces the same standards as ISN.
- **Phase 3 is the highest-value phase.** A single `create_standard_name_entry()` call
  closes 15 validation gaps simultaneously. Prioritize this over context enrichment.
- **No `sn_guidelines.yaml` in codex.** Naming knowledge comes from ISN. Only
  `sn_review_criteria.yaml` lives in codex (scoring is codex's evaluation framework).
- **Include path verified:** `PromptsLoader` searches `prompts/shared/` then `prompts/`
  root, so `{% include "sn/_grammar_reference.md" %}` resolves correctly.

## A/B Comparison Summary

Codex pipeline vs ISN agent workflow — winners by area:

| Area | ISN leads | Codex leads | Tie |
|------|-----------|-------------|-----|
| Context provided to generator | 4 | 2 | 8 |
| Validation checks | 15 | 3 | 1 |
| Review quality | 1 | 5 | 0 |
| **Total** | **20** | **10** | **9** |

**After Phase 3:** ISN validation checks: 15 → 0 (all closed by Pydantic model construction).
**After Phase 5:** ISN context lead: 4 → 0 (all gaps closed by enriching compose context).
**Remaining ISN lead:** Iterative retry loop (5 rounds). Assessed as lower priority — the
unified review + Pydantic validation catches issues upfront rather than iterating.

## Documentation Updates

| Target | Phase |
|--------|-------|
| `docs/architecture/boundary.md` | Phase 6 — NEW |
| `docs/architecture/standard-names.md` | Phase 6 — update pipeline + A/B comparison |
| `AGENTS.md` | Phase 6 — boundary reference, YAML config, 3-layer validation |
| `plans/README.md` | Phase 6 — mark plan 21 done |
