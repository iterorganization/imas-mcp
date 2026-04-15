# 26: Standard Name Pipeline Quality Iteration

**Status:** New
**Depends on:** Plans 20, 23, 25 (pending — partially overlapping)
**Agent type:** Architect + Engineer

## Investigation Findings

### Provenance Groups in Graph

| Group | Source | Count | Avg Score | Quality |
|-------|--------|-------|-----------|---------|
| **A: ISN-imported** | `model=None, accepted` | 25 | N/A (unreviewed) | Reference quality — human-curated, terse descriptions, good grammar |
| **B: Old CLI** | `model=None, drafted` | 39 | 0.623 | Mixed — some creative names, inconsistent grammar, adequate docs |
| **C: Gemini Flash Lite** | `google/gemini-3.1-flash-lite-preview` | 31 | 0.660 | Weakest LLM group — more grammar violations, anti-patterns like `_profile` suffix |
| **D: Claude Sonnet** | `anthropic/claude-sonnet-4.6` | 247 | 0.931 | Best automated — 159 outstanding, rich docs, occasional placeholder names |

**Provenance is fully distinguishable** via `model` + `review_status` fields. `generated_at` timestamps provide additional temporal ordering.

### Review Data Extraction (Confirmed Working)

All review fields extractable from graph:
- `reviewer_score` (float 0-1): normalized 6-dim aggregate
- `reviewer_scores` (JSON): per-dimension breakdown (grammar/semantic/documentation/convention/completeness/compliance, each 0-20)
- `reviewer_comments` (string): detailed text critique
- `review_tier` (string): outstanding/good/adequate/poor
- `reviewer_model` (string): which LLM scored it
- `reviewed_at` (datetime): timestamp

### Grammar Dimension Analysis

| Grammar Score | Count | Pattern |
|---------------|-------|---------|
| 20/20 | 178 | Clean parse, valid segments |
| 10/20 | 26 | Parses but unusual combinations |
| 0-5/20 | 26 | Grammar violations, invalid tokens, placeholder names |

**19 names score grammar=0/20** — these are structural failures:
- Placeholder names: `primary_transport_quantity`, `primary_quantity`
- Invalid constructs: `second_derivative_of...with_respect_to...`
- Missing vocab: `non_inductive_current` (need process segment `non_inductive` or reframe)

### Vocab Gap Analysis

**Zero `vocab_gap_detail` entries in graph** — the compose→graph wiring is disconnected.
Vocab gaps are collected in `state.stats["vocab_gaps"]` during compose but never persisted.

**Observed grammar gaps from reviewer comments:**
1. `_profile` suffix anti-pattern (Gemini): `safety_factor_profile`, `elongation_profile`
2. `_from_` preposition: `current_from_passive_loop`
3. Missing transformation tokens: `second_derivative_of` (only `square_of`, `change_over_time_in`, `logarithm_of`, `inverse_of` exist)
4. Missing process tokens: `non_inductive` (only specific drive types exist)
5. `_coordinate` suffix: `toroidal_flux_coordinate` (not a valid segment)
6. `_due_to_charge_exchange` misuse: process segment applied to wrong physical_base

### The "Base-Only Escape Hatch" Problem

**72 names** use ≤3 segments without structural qualifiers (`_of_`, `_at_`, `_component_`).
Many are **correctly simple** (e.g., `electron_temperature`, `plasma_current` — legitimate bare-quantity patterns).
But some are **underspecified**: `current_density`, `magnetic_field`, `magnetic_flux` — these lack qualification that would make them more specific.

**Key insight:** `physical_base` is **open vocabulary** in ISN. The LLM can always compose a name using just a physical_base segment. This is both:
- **Correct behavior** for fundamental quantities (temperature, pressure, etc.)
- **An escape hatch** when the LLM can't figure out the proper multi-segment decomposition

**Current system has no mechanism to distinguish these cases.**

### Naming vs Documentation Iteration

A critical observation: the **name** (grammar decomposition, segment choice) and the **documentation** (physics description, equations, typical values) are **separately iterable**:
- Name iteration: change grammar segments, vocab tokens, segment ordering
- Documentation iteration: improve equations, add sign conventions, fix variable definitions
- Both share the same `StandardName` node — can be updated independently

The review pipeline scores them on separate dimensions (grammar vs documentation) but treats remediation as atomic (regenerate the whole entry). We should support **targeted fixes**.

---

## Phase 1: Vocab Gap Wiring (Complete the Designed Pipeline)

### 1A: Persist vocab gaps from compose worker

**File:** `imas_codex/standard_names/workers.py` (~line 706)

Currently:
```python
if result.vocab_gaps:
    for vg in result.vocab_gaps:
        state.stats.setdefault("vocab_gaps", []).append({...})
```

Add: persist each gap as a StandardName stub node with `review_status='vocab_gap'`.

**Graph ops:** Add `write_vocab_gap_stubs()` to `graph_ops.py` that creates StandardName nodes:
```cypher
MERGE (sn:StandardName {id: $gap_id})
SET sn.review_status = 'vocab_gap',
    sn.vocab_gap_detail = $detail_json,
    sn.source = 'dd',
    sn.generated_at = datetime()
WITH sn
MATCH (src:IMASNode {id: $source_path})
MERGE (src)-[:HAS_STANDARD_NAME]->(sn)
```

Gap ID format: `vocab_gap:{segment}:{needed_token}` (dedups across batches).

### 1B: Base-only detection in compose post-processing

After compose, before persist, detect names where the LLM used only `physical_base` (no other segments) for a path that has cluster siblings with richer names.

**Heuristic:** If `grammar_fields` has only `physical_base` key AND the cluster contains names with 3+ segments → flag as `needs_grammar_review`.

Store as a property on the StandardName: `grammar_complexity = 'base_only' | 'simple' | 'compound'`.

### 1C: Review prompt: grammar richness assessment

Add to the review prompt (section 1, Grammar Correctness):
```
- **[I1.7]** Is the name using the simplest possible decomposition when a richer one exists?
  Score lower if the name is a bare physical_base but the DD context suggests
  component, subject, position, or process qualifiers would be more specific.
  Compare against cluster siblings for evidence.
```

### 1D: Tests

- Test vocab gap stub creation and dedup
- Test base-only detection heuristic
- Test grammar_complexity property assignment
- Mock compose output with vocab_gaps field populated

---

## Phase 2: Targeted Remediation (Name vs Docs Separately)

### 2A: `sn fix` subcommand for targeted fixes

New CLI command that fixes **only** the specified dimension:

```bash
# Fix grammar only (re-compose the name, keep documentation)
sn fix --grammar --tier poor

# Fix documentation only (keep name, regenerate docs)
sn fix --documentation --tier adequate

# Fix both for specific names
sn fix --grammar --documentation --names "safety_factor_profile,elongation_profile"
```

**Implementation:** New pipeline mode that:
1. Reads existing StandardName from graph
2. Sends to LLM with specific instruction: "Fix ONLY the grammar/documentation"
3. Merges result back, preserving the unchanged dimension

### 2B: Review-guided regeneration

Use `reviewer_scores` JSON to automatically identify what needs fixing:
- `grammar < 10` → regenerate name
- `documentation < 10` → regenerate docs
- `convention < 10` → check for anti-patterns, regenerate name
- `semantic < 10` → full regeneration (fundamental misunderstanding)

```bash
sn fix --auto  # Uses reviewer_scores to determine what to fix
```

### 2C: Name normalization pass

Post-generation normalization for known anti-patterns:
- Strip `_profile` suffix
- Replace `_from_` with proper device/object prefix
- Replace `_in_eV` / `_in_meters` with unit field
- Normalize `volume_averaged_` → `flux_surface_averaged_` where appropriate

This should be a **deterministic post-processor**, not LLM-based.

---

## Phase 3: Compose Prompt Improvements

### 3A: Inject existing high-quality names as few-shot examples

The compose prompt already has `existing_names` and `nearby_existing_names` sections.
Enhance by including full entries (not just names) for the top 3-5 outstanding-tier names
from the same IDS.

This gives the LLM concrete quality targets: "Your output should look like THIS."

### 3B: Physical base guidance

`physical_base` is open vocabulary with no examples in the vocab sections (tokens list is empty).
The `generic_physical_bases.yml` file (12 tokens that require qualification) is NOT injected into the prompt.

**Fix:** Include in compose system prompt:
1. List of generic physical bases that MUST be qualified
2. Examples of common specific physical bases (from existing outstanding names)
3. Clear instruction: "If you can't decompose beyond physical_base, add to vocab_gaps"

### 3C: Transformation segment guidance

Only 4 transformations exist: `square_of`, `change_over_time_in`, `logarithm_of`, `inverse_of`.
The LLM invents others (`second_derivative_of`, `radial_derivative_of`).

**Fix:** Explicit negative examples in prompt:
- ❌ `second_derivative_of_X` → ✅ Report as vocab_gap or use `square_of` + clarify in docs
- ❌ `radial_derivative_of_X` → ✅ `radial_gradient_of_X` if gradient is a valid base, else vocab_gap

### 3D: Binary operator guidance

Only 3 operators: `product_of`, `ratio_of`, `difference_of`.
The LLM invents compound structures (`_with_respect_to_`, `_to_electron_density_ratio`).

**Fix:** Add to prompt with clear examples of when to use operators vs separate names.

---

## Phase 4: ISN Feedback Loop

### 4A: Gap collection and export

New CLI command:
```bash
sn gaps                          # List all vocab gaps from graph
sn gaps --segment subject        # Filter by segment
sn gaps --export yaml > gaps.yml # Export for ISN issue
```

### 4B: Gap analysis report

Aggregate gaps by segment and frequency:
```
Subject gaps:    fast_ion (12 paths), alpha_particle (3 paths)
Process gaps:    non_inductive (5 paths), bootstrap_driven (2 paths)
Transform gaps:  radial_derivative (8 paths), second_derivative (4 paths)
Position gaps:   wall_surface (3 paths), vacuum_vessel (2 paths)
```

### 4C: ISN coordination format

Generate a structured proposal document suitable for filing as a GitHub issue
on `imas-standard-names`:

```yaml
vocab_extension_proposal:
  version: "0.8.0"  # Target ISN version
  segments:
    subject:
      - token: alpha_particle
        rationale: "Needed for fast particle physics in core_profiles/profiles_1d/ion"
        example_names:
          - alpha_particle_density
          - alpha_particle_temperature
        dd_paths: [list of source paths]
    transformation:
      - token: radial_derivative_of
        rationale: "DD has extensive radial derivative quantities"
        example_names:
          - radial_derivative_of_electron_temperature
```

### 4D: Post-ISN-release regeneration

After new ISN version with expanded vocab:
```bash
pip install --upgrade imas-standard-names
sn fix --vocab-gaps  # Regenerate only vocab_gap entries using new vocab
```

---

## Phase 5: Cross-Provenance Quality Leveling

### 5A: Review ISN-imported names

The 25 accepted ISN-imported names have NEVER been reviewed (all `review_tier=None`).
Run review to establish baseline:
```bash
sn review --status accepted --cost-limit 1.0
```

### 5B: Regenerate Gemini Flash Lite names

Group C (Gemini) has avg score 0.660 vs Sonnet's 0.931. These 31 names should be
regenerated with Sonnet for quality parity:
```bash
sn generate --source dd --ids equilibrium --ids core_profiles --force --cost-limit 3.0
```

### 5C: Regenerate Old CLI names

Group B (39 names, avg 0.623) generated without current prompt infrastructure.
Target the `adequate` and `poor` tier names for regeneration.

### 5D: Name deduplication audit

Check for duplicate concepts across provenance groups:
- ISN `electron_temperature_at_magnetic_axis` vs Sonnet `electron_temperature` + position
- ISN `elongation_of_plasma_boundary` vs Gemini `elongation_profile`
- Multiple names for same physical concept across IDS

---

## Implementation Priority

| Phase | Effort | Impact | Priority |
|-------|--------|--------|----------|
| 1A (vocab gap wiring) | Small | High — closes designed-but-disconnected pipeline | **P0** |
| 1B (base-only detection) | Small | Medium — surfaces hidden quality issues | P1 |
| 3B (physical_base guidance) | Small | High — addresses root cause of grammar=0 names | **P0** |
| 3C-3D (transform/operator guidance) | Small | Medium — prevents known anti-patterns | P1 |
| 2A (targeted fix CLI) | Medium | High — enables efficient quality iteration | P1 |
| 1C (review grammar richness) | Small | Medium — catches base-only escape hatch in review | P1 |
| 3A (few-shot examples) | Small | Medium — improves output consistency | P2 |
| 4A-4B (gap CLI) | Small | Medium — enables ISN feedback loop | P2 |
| 2B (auto-fix from scores) | Medium | High — automates quality improvement | P2 |
| 5A-5D (cross-provenance) | Medium | Medium — levels quality across groups | P3 |
| 4C-4D (ISN coordination) | Small | Long-term — systematic vocab extension | P3 |
| 2C (normalization pass) | Medium | Medium — deterministic anti-pattern fix | P3 |

---

## Documentation Updates

| Target | Update Needed |
|--------|---------------|
| `AGENTS.md` | Update SN pipeline section with grammar_complexity, vocab gap persistence, `sn fix` and `sn gaps` commands |
| `docs/architecture/standard-names.md` | Add quality iteration workflow, provenance tracking, targeted remediation |
| Prompt templates | Phases 3A-3D changes |
| Schema | `grammar_complexity` property on StandardName |

## Relationship to Existing Plans

- **Plan 20** (DD-Enriched Generation): Phase 3A overlaps. Plan 20 is broader (full DD context injection). This plan focuses on specific prompt gaps found in production data.
- **Plan 23** (Quality Parity): Phase 5 overlaps. Plan 23 has a benchmark harness (Phase 0) that should be implemented first. This plan provides the remediation mechanisms that Plan 23 would use.
- **Plan 25** (Standalone Review): Already implemented. Review pipeline is operational.
- **Session plan Track 2** (Vocab Gap): Phase 1A and 4A-4D are the implementation of that design.
