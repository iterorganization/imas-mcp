# 26: Standard Name Pipeline Quality Iteration

**Status:** In Progress — infrastructure DONE, prompt/gap/CLI work TODO
**Depends on:** Plan 25 (standalone review — DONE)
**Related:** Plans 20, 23 (pending, lower priority); ISN plan 06 (blocked on this)
**Agent type:** Engineer

## What's Done

### Infrastructure (100%)

| Component | Evidence |
|-----------|----------|
| Generate pipeline stop_fn | `pipeline.py:56` — `_downstream_should_stop()` |
| Review pipeline stop_fn + persist worker | `review/pipeline.py:156` — same pattern |
| Graph state machine fields | `claimed_at`, `claim_token`, `validated_at`, `consolidated_at` on StandardName |
| Claim/mark/release graph_ops | `graph_ops.py:537-762` — `@retry_on_deadlock`, claim tokens |
| Standalone review pipeline | `review/` — EXTRACT, ENRICH, REVIEW, PERSIST (6 modules) |
| Review CLI | `cli/sn.py:1289` — `--unreviewed`, `--re-review`, `--cost-limit` |
| Vocab gap instruction in user prompt | `compose_dd.md:148-156` — tells LLM to report gaps |

### Bootstrap (342 names, target achieved)

Outstanding: 167, Good: 36, Adequate: 32, Poor: 9. 203 good/outstanding (target: 100).

### Investigation (complete)

Grammar analysis of all 342 names: 222 well-structured, 41 legitimate base-only,
41 compound-base stuffing, 20 transform-needed, 9 process-needed, 9 parse failures.
Provenance groups distinguishable via `model` + `review_status`. Full evidence in
ISN plan 06.

---

## Phase 1: Compose Prompt Improvements (P0)

**File:** `imas_codex/llm/prompts/shared/sn/_grammar_reference.md`

The compose LLM generates grammar errors because the prompt has blind spots. These
are all additions to the shared grammar reference template.

### 1A: Process segment `_due_to_` usage

The grammar supports process attribution via `_due_to_` but the prompt never
explains this pattern. All 26 process tokens are listed but with no usage guidance.

Add after "Preposition Usage":

- ✅ `plasma_current_due_to_bootstrap` (process=bootstrap, base=plasma_current)
- ✅ `heating_power_due_to_ohmic` (process=ohmic, base=heating_power)
- ❌ `bootstrap_current` (stuffs process into physical_base)
- ❌ `ohmic_heating_power` (stuffs process into physical_base)

### 1B: Generic physical base qualification

ISN defines 12 generic physical bases that MUST be qualified. This list is not in
the prompt — the LLM cannot know `velocity` alone is invalid.

Add table: area, current, energy, flux, frequency, number_density, power, pressure,
temperature, velocity, voltage, volume — each with invalid-alone and qualified examples.

### 1C: Transformation boundaries with gap reporting

Only 4 transformation tokens exist. The LLM invents others (`time_derivative_of`,
`second_derivative_of`, `volume_averaged`, etc.). These are valid physics operations
that ISN grammar does not yet support.

**Do NOT force compromised workarounds.** Instead:
- List the 4 valid tokens explicitly
- Instruct the LLM: if you need a transformation not listed, report it as a `vocab_gap`
  with segment=transformation and the needed token
- The LLM should still generate the best name it can (may be base-only) alongside
  the gap report

This feeds the ISN feedback loop — gaps become extension proposals.

### 1D: Known grammar ambiguity warnings

`diamagnetic` appears in both component and coordinate vocabularies (they share
all 12 tokens). `radial_component_of_diamagnetic_velocity` triggers a parse error.

This is a **grammar ambiguity**, not a vocabulary gap. The prompt should warn about
known component/coordinate overlap tokens and instruct the LLM to flag affected
names for ISN grammar review rather than producing broken names.

---

## Phase 2: Gap Infrastructure (P0)

### 2A: VocabGap node type in schema

Gaps are NOT standard names — they must be a separate node type to avoid contaminating
StandardName lifecycle, extract skip logic, and review pipelines.

**File:** `imas_codex/schemas/standard_name.yaml`

```yaml
VocabGap:
  description: >-
    Records a missing grammar token identified during SN composition.
    Linked to source DD paths via HAS_VOCAB_GAP relationships.
  class_uri: sn:VocabGap
  attributes:
    id:
      identifier: true
      description: "Format: vocab_gap:{segment}:{needed_token}"
    segment:
      description: Grammar segment missing the token
      required: true
    needed_token:
      description: The token that should exist
      required: true
    example_count:
      description: Number of source paths that need this token
      range: integer
    first_seen_at:
      range: datetime
    last_seen_at:
      range: datetime
```

Relationships:
- `(IMASNode)-[:HAS_VOCAB_GAP {reason, observed_at}]->(VocabGap)`
- `(FacilitySignal)-[:HAS_VOCAB_GAP {reason, observed_at}]->(VocabGap)`

Per-source `reason` lives on the relationship (source-specific), not the node
(gap-global). Node stores the normalized gap concept; relationships store evidence.

### 2B: `write_vocab_gaps()` in graph_ops

**File:** `imas_codex/standard_names/graph_ops.py`

New function that:
1. MERGEs VocabGap nodes by `id` (dedup across batches)
2. Creates HAS_VOCAB_GAP relationships from source paths with per-source reason
3. Increments `example_count`, updates `last_seen_at`

### 2C: Wire compose worker to graph persistence

**File:** `imas_codex/standard_names/workers.py` (~line 706)

Currently gaps collect in `state.stats["vocab_gaps"]` and are lost on pipeline exit.
After collecting, immediately call `write_vocab_gaps()` to persist. This ensures
gaps survive cost-limit interruption.

### 2D: Deterministic ambiguity classification in validate worker

**File:** `imas_codex/standard_names/workers.py` (validate worker)

The validate worker already runs `parse_standard_name()`. When it catches errors
like "component and coordinate cannot both be set", tag `validation_issues` with
`[grammar:ambiguity:component_coordinate_overlap]`. This is deterministic, not
LLM-dependent, and generalizes to any future grammar ambiguity patterns.

### 2E: Review prompt grammar richness [I1.7]

**File:** `imas_codex/llm/prompts/sn/review.md`

Add to section 1 (Grammar Correctness):
```
- **[I1.7]** Is the name using the simplest decomposition when a richer one exists?
  Score lower if the name is a bare physical_base but the DD context suggests
  component, subject, position, or process qualifiers would be more specific.
```

### 2F: Tests

- VocabGap node creation and dedup across batches
- HAS_VOCAB_GAP relationship with per-source reason
- Mock compose output with vocab_gaps field populated
- Validate worker ambiguity detection for known overlaps
- Gaps survive pipeline cost-limit interruption

---

## Phase 3: CLI Improvements (P1)

### 3A: `--force` replacing `--re-review` on review CLI

**File:** `imas_codex/cli/sn.py`

Rename `--re-review` to `--force` for consistency with `sn generate --force`.
Both mean "process even if already processed." Keep `--re-review` as a hidden
deprecated alias.

### 3B: `--from-model` filter on generate CLI

**File:** `imas_codex/cli/sn.py`

New option for provenance-based regeneration:
```bash
sn generate --from-model gemini --force
```

Selects StandardNames whose `model` field CONTAINS the given substring. Model
strings are long (`google/gemini-3.1-flash-lite-preview`) so substring matching
is intentional — document in help text.

`--from-model` implies `--force` (selecting by model only makes sense for
regeneration). Error if `--from-model` used without `--force` being either
explicit or implied.

### 3C: `--name-only` flag on generate CLI

**File:** `imas_codex/cli/sn.py`, compose prompt variants

New flag: compose focuses on naming/grammar only. Documentation fields are
preserved from existing graph data via coalesce (never write empty-string
placeholders — omit fields so graph coalescing works).

**Review invalidation rule:** `--name-only` must clear `review_input_hash`
so stale reviews don't appear current.

### 3D: `sn enrich` command (name-centric docs enrichment)

**File:** `imas_codex/cli/sn.py`, new pipeline module

Separate command for documentation iteration:
```bash
sn enrich --ids equilibrium --cost-limit 3.0
sn enrich --status drafted
```

Key design:
- Operates on **existing StandardName nodes**, not source paths
- Takes the name as fixed input, generates rich documentation
- Builds docs from the **name plus ALL linked DD paths** (handles shared names)
- Clears `review_input_hash` to invalidate stale reviews
- Does NOT change the name, grammar fields, kind, or unit

### 3E: `sn gaps` command

```bash
sn gaps                          # List all VocabGap nodes
sn gaps --segment transformation # Filter by segment
sn gaps --export yaml            # Export for ISN issue filing
```

Queries VocabGap nodes and their HAS_VOCAB_GAP relationships to show evidence.

---

## Phase 4: Cross-Provenance Leveling (P3)

### 4A: Regenerate Gemini Flash Lite names (31 names, avg 0.660)

```bash
sn generate --from-model gemini --force --cost-limit 3.0
```

### 4B: Regenerate poor Old CLI names (39 names, avg 0.623)

Target `adequate` and `poor` tier only.

### 4C: Review ISN-imported names (25, never reviewed)

```bash
sn review --status accepted --cost-limit 1.0
```

### 4D: Name deduplication audit

Check for duplicate concepts across provenance groups.

---

## Implementation Priority

| Phase | Effort | Impact | Priority |
|-------|--------|--------|----------|
| 1A-1D (compose prompt fixes) | Small | High — prevents errors at source | P0 |
| 2A-2C (VocabGap infra) | Small | High — closes gap pipeline | P0 |
| 2D (ambiguity detection) | Small | Medium — deterministic classification | P0 |
| 2E-2F (review prompt + tests) | Small | Medium | P1 |
| 3A (--force rename) | Small | Small — consistency | P1 |
| 3B (--from-model) | Small | Medium — provenance targeting | P1 |
| 3C-3D (--name-only + sn enrich) | Medium | High — separate iteration | P1 |
| 3E (sn gaps) | Small | Medium — ISN feedback | P2 |
| 4A-4D (cross-provenance) | Medium | Medium | P3 |

## ISN Plan Relationship

ISN plan 06 proposes vocab extensions (7 transformations, 3 processes, 2 subjects).

**Do NOT implement ISN plan 06 yet.** Sequence:
1. Codex Phase 1 prompt fixes → reduces false gaps (LLM not using existing grammar)
2. Codex Phase 2 gap infra → collects real gaps with evidence
3. Run generation + review cycle → produces clean gap data
4. ISN vocab extensions → based on evidence from step 3
5. Codex regeneration → after ISN release with new tokens

## Review Invalidation Rule

Any operation that changes name, description, documentation, kind, unit, or tags
must invalidate review freshness by clearing `review_input_hash`. This applies to:
- `sn generate --force` (full regeneration)
- `sn generate --name-only` (name change)
- `sn enrich` (docs change)

## Documentation Updates

| Target | When |
|--------|------|
| `AGENTS.md` | After Phase 3: `sn enrich`, `sn gaps`, `--from-model`, `--force` on review |
| Prompt templates | Phase 1 |
| Schema reference | Auto-rebuilt after Phase 2A (VocabGap node type) |
