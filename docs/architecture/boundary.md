# Architecture Boundary: imas-codex ↔ imas-standard-names

## Principle

**imas-standard-names** (ISN) owns grammar, vocabulary, and validation.
**imas-codex** owns the generation pipeline, evaluation, and graph persistence.

## What ISN Provides (Import Boundary)

| API | Module | Purpose |
|-----|--------|---------|
| `get_grammar_context()` | `imas_standard_names.grammar` | All grammar context: vocabulary, patterns, naming guidance (19 keys) |
| `parse_standard_name()` | `imas_standard_names.grammar` | Grammar round-trip validation |
| `compose_standard_name()` | `imas_standard_names.grammar` | Canonical name composition from parsed segments |
| `create_standard_name_entry()` | `imas_standard_names.models` | Full Pydantic model construction (18 validators) |
| `StandardNameEntry` | `imas_standard_names.models` | Discriminated union model for catalog entries |
| `run_semantic_checks()` | `imas_standard_names.validation.semantic` | 9 semantic grammar checks |
| `validate_description()` | `imas_standard_names.validation.description` | Description quality checks |

`get_grammar_context()` is the single entry point for all grammar data. It
returns 19 keys including vocabulary sections, segment order, exclusive pairs,
naming guidance, quick start, common patterns, and critical distinctions.
`context.py` calls it once (module-level cache) and builds the compose context
from the returned dict — no private ISN imports.

**ISN minimum version**: >=0.7.0rc3 (specified in `pyproject.toml`).

## What Codex Owns

- **Pipeline orchestration**: 6-phase DAG (extract → compose → review → validate → consolidate → persist)
- **LLM evaluation**: 6-dimensional scoring (grammar, semantic, documentation, convention, completeness, compliance)
- **Review criteria**: `imas_codex/llm/config/sn_review_criteria.yaml`
- **Graph persistence**: StandardName nodes, relationships, validation issues
- **Calibration**: `benchmark_calibration.yaml` for reviewer consistency
- **Collision avoidance**: Vector search for similar existing names (`imas_codex/standard_names/search.py`)
- **Prompt infrastructure**: Shared fragments via `{% include %}`, prompt configs via `load_prompt_config()`

## Three-Layer Validation

```
Layer 1: Pydantic model construction (create_standard_name_entry)
         → 18 field validators fire automatically
         → Issues tagged: [pydantic:field] message

Layer 2: Post-construction semantic checks (run_semantic_checks)
         → 9 grammar-semantic checks on constructed models
         → Issues tagged: [semantic] message

Layer 3: Description quality checks (validate_description)
         → Metadata leakage detection, quality checks
         → Issues tagged: [description] message
```

Validation is **annotation-only** — entries are never rejected by ISN validation.
The only hard rejection is an unparseable name (grammar round-trip fails in
`parse_standard_name()`). Issues are persisted to the graph as
`validation_issues` (list) and `validation_layer_summary` (JSON), and are
shown to the LLM reviewer as context.

**Implementation**: `_validate_via_isn()` in `imas_codex/standard_names/workers.py` runs
all three layers in sequence and returns `(issues, layer_summary)`.

## Scoring Architecture

- LLM scores 6 dimensions × 0–20 integers (avoids float clustering in LLM output)
- Aggregate normalized to 0–1: `sum / 120.0` (via `SNQualityScore.score` property)
- Tiers: outstanding ≥0.85, good ≥0.60, adequate ≥0.40, poor <0.40
- The `reviewer_score` stored in the graph is a float 0–1
- Scoring criteria defined in `imas_codex/llm/config/sn_review_criteria.yaml`
- Verdict rules: accept (≥0.60, no zero dimensions), reject (<0.40 or any zero), revise (otherwise)

## Prompt Infrastructure

| File | Role |
|------|------|
| `llm/prompts/sn/compose_system.md` | Static system prompt — uses `{% include "sn/_grammar_reference.md" %}` |
| `llm/prompts/sn/compose_dd.md` | Dynamic user prompt with batch context |
| `llm/prompts/sn/review.md` | Review prompt with 6-dimension scoring rubric |
| `llm/prompts/shared/sn/_grammar_reference.md` | Shared grammar fragment (Jinja template) |
| `llm/prompts/shared/sn/_scoring_rubric.md` | Shared scoring rubric (reference fragment) |
| `llm/config/sn_review_criteria.yaml` | Scoring dimensions, tiers, verdict rules |

`load_prompt_config("sn_review_criteria")` loads the YAML config with caching.
`render_prompt()` handles `{% include %}` resolution from the `shared/` directory.

## Rules

1. **Never import from ISN private modules** — only `grammar`, `models`, `validation.*`
2. **Never hardcode grammar rules** — get them from `get_grammar_context()`
3. **Review criteria live in codex** — scoring is codex's evaluation framework
4. **ISN minimum version**: >=0.7.0rc3 (specified in pyproject.toml)
5. **Validation is annotation-only** — never reject based on ISN validation; persist issues to graph
