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
   (parse→compose) but skips ISN's 27 validators (18 Pydantic field + 9 semantic).
   The A/B comparison showed ISN leads 15:3 on validation surface.
5. **Import path fragility:** `catalog_import.py` imports `StandardNameEntry` from
   `imas_standard_names.catalog.edit` which is a re-export — should import from
   `imas_standard_names.models` directly. ISN Plan 05 deletes `catalog/edit.py`.
6. **No formal boundary documentation:** Neither project documents what it owns vs delegates.
7. **Validation issues are lost:** `validate_worker` logs soft-validation warnings but
   never persists them. `SNQualityReview.issues` is returned by the LLM but never
   extracted or stored. The schema has no `validation_issues` property. This means
   each name is validated but the results are discarded — forcing re-validation if
   anyone wants to review issues later.
8. **0-120 scoring is opaque:** The quality score is stored as an integer 0-120 (6
   dimensions × 0-20). This is an implementation artifact of the LLM prompt design.
   The schema says "0-100" (already inconsistent). External consumers need a universal
   0-1 normalized score. The per-dimension 0-20 integer scoring is correct for LLM
   consistency (avoids float clustering) but the aggregate should be normalized.
9. **No existing-name collision avoidance in compose context:** The graph has a full
   semantic search over StandardName nodes (`search_standard_names` MCP tool using
   `standard_name_desc_embedding` vector index), but this is never used to supply
   "nearby existing names" to the compose or review prompts.

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
  # Review scoring criteria — codex-owned evaluation framework.
  # Per-dimension scoring uses integers 0-20 in the LLM prompt.
  # Aggregate score is normalized to 0-1 (sum / 120).
  dimensions:
    grammar:
      max_score: 20
      description: "Parse/compose round-trip, segment ordering, vocabulary compliance"
    semantic:
      max_score: 20
      description: "Name accurately describes the physical quantity"
    documentation:
      max_score: 20
      description: "Description and documentation are clear, complete, non-redundant"
    convention:
      max_score: 20
      description: "Follows naming conventions (no units in name, no path fragments)"
    completeness:
      max_score: 20
      description: "All required fields filled, links populated, tags relevant"
    compliance:
      max_score: 20
      description: "Follows compose prompt instructions exactly"

  verdict_rules:
    accept: "score >= 0.60 AND no dimension is 0"
    reject: "score < 0.40 OR any dimension is 0"
    revise: "all other cases"

  tiers:
    outstanding: { min: 0.85 }
    good: { min: 0.60 }
    adequate: { min: 0.40 }
    poor: { min: 0.0 }
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

### Phase 3: ISN Validation + Persistent Issues + Normalized Scoring

**Goal:** Three connected changes that close the major quality gaps:
1. Construct ISN Pydantic models for full validation (closes 15 validation gaps)
2. Persist ALL validation issues to the graph (validate once, review with context)
3. Normalize quality scores to 0-1 (universal scale)

**Prerequisite:** ISN Plan 05 Phase 0 merged (stable imports).

#### 3a. Validate-Once Architecture

**Critical insight:** validate_worker currently acts as a **filter** (logs issues, discards
them). It should be an **annotator** — ALL validation issues are attached to the entry and
persisted to the graph. Only names that cannot be parsed at all are rejected.

```
validate_worker (annotator, not filter)
    ↓
entry["validation_issues"] = ["[pydantic:unit] ...", "[semantic] ...", ...]
    ↓
review_worker receives entries WITH their validation issues
    ↓
LLM reviewer sees issues as scoring context → more accurate review
    ↓
persist_worker writes validation_issues to StandardName node
    ↓
Human reviewer sees issues in graph → no need to re-validate
```

**Hard fail boundary (entry truly rejected — cannot proceed):**
- Name is empty, None, or unparseable (grammar round-trip fails completely)
- Name string cannot be parsed by `parse_standard_name()` at all

**Everything else passes through with issues attached:**
- Pydantic validation errors (name pattern, vocabulary, unit, links, etc.)
- Semantic check warnings
- Description quality warnings
- These are valuable review context, not rejection criteria

**Why this boundary:** Even a name with ISN validation issues may be the best available
name for a DD path. The review LLM should see the issues and factor them into its score.
A human reviewer should see them too. Rejecting names that have fixable issues wastes
the LLM generation cost. Instead: annotate, review with context, let humans decide.

#### 3b. Schema + Graph Changes

**Files to modify:**

- `imas_codex/schemas/standard_name.yaml`:
  ```yaml
  validation_issues:
    description: >-
      JSON-encoded list of validation issues from ISN Pydantic model
      construction, semantic checks, and description quality checks.
      Each issue is a tagged string like "[pydantic:unit] Invalid unit".
      Persisted for review context — validate once, review with full context.
    multivalued: true
  validation_layer_summary:
    description: >-
      JSON-encoded summary of validation by layer:
      {pydantic: {passed: bool, error_count: int},
       semantic: {issue_count: int},
       description: {issue_count: int}}
  ```

- `imas_codex/sn/graph_ops.py` (`write_standard_names`):
  - Add `validation_issues` and `validation_layer_summary` to the MERGE query
  - Use `coalesce(b.validation_issues, sn.validation_issues)` to preserve existing

#### 3c. Three-Layer Validation Implementation

```
Layer 1: Pydantic model construction (create_standard_name_entry)
         → 18 field validators fire automatically
         → Issues tagged: [pydantic:field_name] message

Layer 2: Post-construction semantic checks (run_semantic_checks)
         → 9 grammar-semantic checks on constructed models
         → Issues tagged: [semantic] message

Layer 3: Description quality checks (validate_description)
         → Metadata leakage detection
         → Issues tagged: [description] message
```

**Files to modify:**

- `imas_codex/sn/workers.py` (`validate_worker`):

  Replace the current soft-validation section with ISN model construction +
  issue annotation:

  ```python
  from imas_standard_names.models import create_standard_name_entry
  from imas_standard_names.validation.semantic import run_semantic_checks
  from imas_standard_names.validation.description import validate_description
  from pydantic import ValidationError

  def _validate_via_isn(entry: dict) -> tuple[list[str], dict]:
      """Construct ISN Pydantic model and collect ALL validation issues.

      Returns:
          (issues: list[str], layer_summary: dict)

      This function is purely an annotator — it never rejects entries.
      Parseability is checked upstream by the grammar round-trip in
      validate_worker. This function attaches quality annotations.
      """
      issues = []
      summary = {"pydantic": {"passed": True, "error_count": 0},
                 "semantic": {"issue_count": 0, "skipped": False},
                 "description": {"issue_count": 0}}

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
      }

      # Layer 1: Pydantic model construction (fires 18 validators)
      model = None
      try:
          model = create_standard_name_entry(isn_dict)
      except ValidationError as e:
          summary["pydantic"]["passed"] = False
          summary["pydantic"]["error_count"] = len(e.errors())
          for err in e.errors():
              field = ".".join(str(loc) for loc in err["loc"])
              issues.append(f"[pydantic:{field}] {err['msg']}")

      # Layer 2: Semantic checks (only if model constructed)
      if model is not None:
          sem_issues = run_semantic_checks({isn_dict["name"]: model})
          summary["semantic"]["issue_count"] = len(sem_issues)
          issues.extend(f"[semantic] {i}" for i in sem_issues)
      else:
          summary["semantic"]["skipped"] = True

      # Layer 3: Description quality
      desc_issues = validate_description(isn_dict)
      summary["description"]["issue_count"] = len(desc_issues)
      issues.extend(f"[description] {i['message']}" for i in desc_issues)

      return issues, summary
  ```

  After calling `_validate_via_isn()`, attach issues to the entry:
  ```python
  issues, layer_summary = _validate_via_isn(entry)
  entry["validation_issues"] = issues
  entry["validation_layer_summary"] = json.dumps(layer_summary)
  # Entry always passes through to review (unless grammar round-trip failed)
  ```

#### 3d. Review Context: Validation Issues

- `imas_codex/sn/workers.py` (`review_worker` / `_review_batch`):
  - Pass each entry's `validation_issues` to the review prompt context
  - The LLM reviewer sees the issues and factors them into scoring

- `imas_codex/llm/prompts/sn/review.md`:
  - Add section: "## Validation Issues" with per-entry ISN validation results
  - Instruct reviewer: "If validation issues are present, assess whether they are
    genuine quality problems or false positives. Factor genuine issues into your
    grammar and convention scores."

#### 3e. Normalize Scoring to 0-1

**Why:** The current 0-120 scale is an implementation artifact. Per-dimension 0-20
integer scoring is correct for LLM prompt consistency (avoids "everything is 0.73"
float clustering — documented in LLM-as-judge literature). But the aggregate total
should be normalized to 0-1 for:
- Universal comparability (industry standard)
- Schema consistency (current schema says "0-100" — already wrong)
- Human readability (0.85 vs 102/120)
- Threshold clarity (accept ≥ 0.60 vs accept ≥ 72)

**What stays the same:**
- LLM prompt still asks for 0-20 integer per dimension (6 dimensions)
- Calibration entries still have per-dimension 0-20 scores
- `SNQualityScore` fields remain `int` in range `[0, 20]`
- The LLM never sees or computes the total — we do

**What changes:**

- `imas_codex/sn/models.py` (`SNQualityScore`):
  ```python
  @property
  def score(self) -> float:
      """Normalized quality score (0-1). Sum of 6 dimensions / 120."""
      return self.total / 120.0

  @property
  def tier(self) -> str:
      s = self.score
      if s >= 0.85:   return "outstanding"
      elif s >= 0.60: return "good"
      elif s >= 0.40: return "adequate"
      return "poor"
  ```
  Keep `total` property for internal use but add `score` as the public API.

- `imas_codex/schemas/standard_name.yaml`:
  - Change `reviewer_score` description: "Normalized quality score (0-1)"
  - Change `reviewer_score` range: `float` (was `integer`)

- `imas_codex/sn/workers.py` (`review_worker`):
  - Change: `original["reviewer_score"] = review.scores.score`  (was `.total`)

- `imas_codex/sn/benchmark.py`:
  - Change score output to use `.score` (0-1) instead of `.total` (0-120)
  - Update table format: `f"{score:.2f}"` instead of `f"{total}/120"`

- `imas_codex/sn/benchmark_calibration.yaml`:
  - Add `expected_score` as normalized 0-1 alongside existing `expected_score` integers
  - Or: compute normalized at load time: `entry["score"] = entry["expected_score"] / 120.0`

- `imas_codex/llm/prompts/sn/review.md`:
  - Tier descriptions use 0-1: "Outstanding (≥0.85), Good (≥0.60), Adequate (≥0.40), Poor (<0.40)"
  - Verdict rules: "accept if score ≥ 0.60 AND no dimension is 0"
  - Calibration examples show normalized scores
  - Per-dimension 0-20 integer scoring instruction unchanged

- `imas_codex/llm/config/sn_review_criteria.yaml` (Phase 2):
  - Already created with 0-1 thresholds (forward-compatible from Phase 2)

- **Graph migration** (run once after deploying Phase 3):
  ```cypher
  // Normalize existing integer reviewer_score values (0-120) to float (0-1)
  MATCH (sn:StandardName)
  WHERE sn.reviewer_score IS NOT NULL AND sn.reviewer_score > 1.0
  SET sn.reviewer_score = sn.reviewer_score / 120.0
  RETURN count(sn) AS migrated
  ```
  Also regenerate auto-generated models: `uv run build-models --force`
  (updates `models.py`, `dd_models.py`, `schema_context_data.py` to reflect
  the integer→float type change on `reviewer_score`)

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
  - Test issues are attached to entry dict (not just logged)
  - Test graceful handling of malformed dicts (no crash)
- `tests/sn/test_scoring.py` (NEW):
  - Test SNQualityScore.score returns float 0-1
  - Test tier thresholds at 0.85/0.60/0.40 boundaries
  - Test verdict derivation with normalized thresholds
  - Test calibration entries round-trip (expected_score / 120 matches)
- `tests/sn/test_validate_persistence.py` (NEW):
  - Test validation_issues are persisted to graph
  - Test validation_layer_summary is persisted
  - Test review prompt includes validation issues as context
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
      """Return calibration entries formatted for prompt rendering.
      Includes normalized 0-1 score alongside per-dimension 0-20 integers."""
      return [
          {
              "name": e["name"],
              "tier": e["tier"],
              "score": round(e["expected_score"] / 120.0, 2),
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

**Files to create:**

- `imas_codex/sn/search.py` (NEW ~30 lines):
  ```python
  """Standard name search helpers for pipeline use.

  Provides structured dict results (not formatted strings) for programmatic
  use in compose and review context. Wraps the same embedding + vector index
  infrastructure as the `search_standard_names` MCP tool.
  """
  from imas_codex.embeddings import Encoder
  from imas_codex.graph.client import GraphClient

  def search_similar_names(query: str, k: int = 5) -> list[dict]:
      """Find existing StandardName nodes similar to query text.

      Uses Encoder to embed query, then vector search on
      standard_name_desc_embedding index. Returns list of dicts
      with keys: id, description, kind, unit, score.
      """
  ```

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
  - Add existing-name search: for each DD path batch, call
    `search_similar_names(batch_description)` to find top-5 existing names
    as collision avoidance context ("names already covering this area")

- `imas_codex/llm/prompts/sn/compose_system.md`:
  - Add quick-start section before grammar reference (helps LLM orientation)
  - Add common patterns section with "do this, not that" examples
  - Add token frequency note to vocabulary section ("commonly used:", "rarely used:")

- `imas_codex/sn/workers.py` (`review_worker`):
  - Add existing-name collision avoidance to review context using
    `search_similar_names()` — embed each candidate's description and search
    for nearby existing names. Provide as "nearby names" in review prompt.

**Tests:**
- `tests/sn/test_search.py` — verify `search_similar_names()` returns structured dicts
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
  - codex imports from ISN (exhaustive — ISN must not rename/remove without coordination):
    - Grammar: `get_grammar_context()`, `parse_standard_name()`, `compose_standard_name()`
    - Models: `StandardNameEntry`, `create_standard_name_entry()`
    - Validation: `run_semantic_checks()`, `validate_description()`, `run_structural_checks()`
    - Constants: grammar enums (vocabulary StrEnums), tag constants, `PhysicsDomain`
  - ISN exposes NO write operations via MCP or Python API
  - Changes to any contract function require a coordinated release
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
Codex Phase 3 (ISN validation + persistent issues + 0-1 scoring)
    ←── depends on Phase 1 (ISN public API) + Phase 2 (YAML config exists)
    ↓
Codex Phase 4 (centralize calibration) ←── independent, can parallel Phase 2
    ↓
Codex Phase 5 (context enrichment gaps) ←── depends on Phase 1 + ISN Phase 0
    ↓
Codex Phase 6 (docs) ←── depends on all above
```

Phase 4 can run in parallel with Phases 2-3 (no cross-dependency).
Phase 5 can run in parallel with Phase 4 (no cross-dependency).
Phase 3 needs Phase 1 AND Phase 2 (ISN API + YAML config file must exist).
Phase 5 needs ISN Phase 0 to expose common_patterns and vocabulary_usage_stats.

## Implementation Notes

- **Each phase is one commit.** Run `uv run pytest tests/sn/` after each.
- **Phase 1 is blocked on ISN Plan 05 Phase 0.** Start with Phase 4 while waiting.
- **Prompt output must be semantically identical** after Phase 2 — ISN API returns
  the same rules that were previously hardcoded. Diff rendered prompts to verify.
- **No behavioral change** in Phases 1-2, 4. Phase 3 adds ISN Pydantic validation
  (27 checks) and changes the scoring scale from 0-120 to 0-1. Names that previously
  passed may now have validation_issues attached — this is correct and desired.
- **Phase 3 is the highest-value phase.** It delivers three connected improvements:
  (1) 27 ISN validators via Pydantic model construction, (2) persistent validation
  issues as review context, (3) normalized 0-1 scoring.
- **Validate once, review with context.** validate_worker annotates entries with ALL
  issues. review_worker sees those issues. persist_worker stores them. No re-validation.
- **Hard fail is narrow:** Only names that cannot be parsed AT ALL are rejected.
  Everything else passes through with issues attached. Even a name with Pydantic
  warnings may be the best available name for a DD path — let review and human
  reviewers decide.
- **0-1 scoring rationale:** Per-dimension integer 0-20 is kept in the LLM prompt
  (avoids float clustering). The aggregate is normalized to 0-1 (universal scale).
  The LLM never sees or computes the total — we compute `sum / 120.0`.
- **Existing semantic search:** The `standard_name_desc_embedding` vector index and
  `_vector_search_sn()` function in `sn_tools.py` already provide full semantic +
  keyword search over StandardName nodes. Phase 5 uses this for collision avoidance
  context — no new search infrastructure needed.
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

**After Phase 3:** ISN validation leads 15 → 0 (all closed by Pydantic model construction).
Validation issues persisted to graph. Scoring normalized to 0-1.
**After Phase 5:** ISN context lead 4 → 0 (collision avoidance via existing vector index,
enriched context from expanded `get_grammar_context()`).
**Remaining ISN lead:** Iterative retry loop (5 rounds). Assessed as lower priority — the
validate-once + persistent-issues architecture catches issues upfront rather than iterating.

## Documentation Updates

| Target | Phase |
|--------|-------|
| `docs/architecture/boundary.md` | Phase 6 — NEW |
| `docs/architecture/standard-names.md` | Phase 6 — update pipeline + A/B comparison |
| `AGENTS.md` | Phase 6 — boundary reference, YAML config, 3-layer validation, 0-1 scoring |
| `plans/README.md` | Phase 6 — mark plan 21 done |
