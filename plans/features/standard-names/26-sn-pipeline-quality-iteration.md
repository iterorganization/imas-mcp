# 26: Standard Name Pipeline Quality Iteration

**Status:** In Progress — prerequisite infrastructure DONE, prompt improvements and gap wiring TODO
**Depends on:** Plan 25 (standalone review — **DONE**, moved to completed)
**Related:** Plans 20, 23 (pending — partially overlapping, lower priority)
**ISN plan:** `~/Code/imas-standard-names/plans/features/06-vocabulary-extensions-from-codex-bootstrap.md`
**Agent type:** Engineer

## What's Done

### Infrastructure (100% complete)

| Component | Status | Commit/Evidence |
|-----------|--------|-----------------|
| Generate pipeline stop_fn | ✅ DONE | `a663ebd8` — `_downstream_should_stop()` in `pipeline.py:56` |
| Review pipeline stop_fn | ✅ DONE | `0299b594` — same pattern in `review/pipeline.py:156` |
| Review persist worker should_stop_fn | ✅ DONE | `88f0a806` — `should_stop_fn=_downstream_should_stop` |
| Review source_id matching fallback | ✅ DONE | `88f0a806` — fallback by `standard_name` field |
| Graph state machine fields | ✅ DONE | `e0db97ed` — `claimed_at`, `claim_token`, `validated_at`, `consolidated_at` on StandardName schema |
| Claim/mark/release graph_ops | ✅ DONE | `graph_ops.py:537-762` — full claim pattern with `@retry_on_deadlock` |
| Standalone review pipeline | ✅ DONE | `review/pipeline.py` — EXTRACT→ENRICH→REVIEW→PERSIST, 6 modules |
| Review CLI | ✅ DONE | `cli/sn.py:1289` — `--unreviewed`, `--re-review`, `--cost-limit`, etc. |
| Negative transform examples in prompt | ✅ DONE | `_grammar_reference.md` — ❌/✅ pairs for anti-patterns |
| Curated examples in compose prompt | ✅ PARTIAL | `compose_system.md:67` — template present, runtime-injected |
| Vocab gap instruction in user prompt | ✅ DONE | `compose_dd.md:148-156` — tells LLM to report gaps |

### Bootstrap Results (operational)

342 StandardName nodes across 8 IDS. Tier distribution:
- Outstanding: 167, Good: 36, Adequate: 32, Poor: 9
- Target achieved: 203 names with good/outstanding (target was ≥100)

### Investigation Complete

Provenance groups distinguishable via `model` + `review_status`. Grammar analysis
of all 342 names: 222 well-structured, 41 legitimate base-only, 41 compound-base
stuffing, 20 transform-needed, 9 process-needed, 9 parse failures. See ISN plan 06
for full evidence tables and proposed vocab extensions.

---

## What's NOT Done

### Phase 1: Compose Prompt Improvements ← START HERE

The compose LLM generates names with grammar errors because the prompt lacks critical
guidance. The grammar reference (`shared/sn/_grammar_reference.md`) and compose system
prompt (`compose_system.md`) have three blind spots identified from production data:

#### 1A: Process segment usage (9+ names affected)

**Problem:** LLM writes `bootstrap_current` instead of `plasma_current_due_to_bootstrap`.
The grammar supports process segments via `_due_to_` connector, but the prompt never
explains this. All 26 process tokens are listed in the vocabulary section but with no
usage pattern.

**File:** `imas_codex/llm/prompts/shared/sn/_grammar_reference.md`

**Fix:** Add after the "Preposition Usage" section a "Process Segments" section explaining
the `_due_to_` connector pattern with positive/negative examples:
- ✅ `plasma_current_due_to_bootstrap` (process=bootstrap, base=plasma_current)
- ❌ `bootstrap_current` (stuffs process into physical_base)

#### 1B: Generic physical base qualification (12 tokens)

**Problem:** ISN defines 12 generic physical bases (area, current, energy, flux,
frequency, number_density, power, pressure, temperature, velocity, voltage, volume)
that MUST be qualified with a subject, device, or object. This list is NOT in the
prompt — the LLM has no way to know `velocity` alone is invalid.

**File:** `imas_codex/llm/prompts/shared/sn/_grammar_reference.md`

**Fix:** Add "Generic Physical Bases" section with table showing invalid-alone vs
qualified examples for each of the 12 tokens.

#### 1C: Transformation token boundary (20 names affected)

**Problem:** LLM invents transformation tokens that don't exist (`time_derivative_of`,
`second_derivative_of`, `radial_derivative_of`). Only 4 tokens exist: `square_of`,
`change_over_time_in`, `logarithm_of`, `inverse_of`.

**File:** `imas_codex/llm/prompts/shared/sn/_grammar_reference.md`

**Fix:** Add "Transformation Boundaries" section listing ONLY the 4 valid tokens and
explicit common-mistake corrections:
- ❌ `time_derivative_of_X` → ✅ `change_over_time_in_X`
- ❌ `second_derivative_of_X` → ✅ Report as vocab_gap
- ❌ `volume_averaged_X` → ✅ Report as vocab_gap

#### 1D: `diamagnetic` disambiguation (6 parse failures)

**Problem:** `diamagnetic` appears in both component and coordinate vocabularies.
`radial_component_of_diamagnetic_velocity` triggers "component and coordinate cannot
both be set" because parser sees `diamagnetic` as coordinate conflicting with `radial`
as component.

**File:** `imas_codex/llm/prompts/shared/sn/_grammar_reference.md`

**Fix:** Add to negative examples:
- ❌ `radial_component_of_diamagnetic_velocity` → parser conflict
- ✅ `radial_component_of_diamagnetic_drift_velocity` — use `diamagnetic_drift_velocity`
  as physical_base to avoid ambiguity with the `diamagnetic` coordinate token

### Phase 2: Vocab Gap Persistence

#### 2A: `write_vocab_gap_stubs()` in graph_ops.py

The compose worker collects vocab gaps in `state.stats["vocab_gaps"]` (workers.py:706)
but they're lost when the pipeline exits. Add a graph persistence function.

**File:** `imas_codex/standard_names/graph_ops.py`

Gap ID format: `vocab_gap:{segment}:{needed_token}` (dedups across batches).

#### 2B: Wire compose worker to call `write_vocab_gap_stubs()`

**File:** `imas_codex/standard_names/workers.py` (~line 706)

After collecting gaps in `state.stats["vocab_gaps"]`, immediately persist them.
This ensures gaps survive pipeline interruption (cost limit hit, etc.).

#### 2C: Review prompt grammar richness [I1.7]

**File:** `imas_codex/llm/prompts/sn/review.md`

Add to section 1 (Grammar Correctness):
```
- **[I1.7]** Is the name using the simplest possible decomposition when a richer
  one exists? Score lower if the name is a bare physical_base but the DD context
  suggests component, subject, position, or process qualifiers would be more specific.
```

#### 2D: Tests

- Test vocab gap stub creation and dedup
- Mock compose output with vocab_gaps field populated
- Test that gaps survive pipeline interruption

### Phase 3: Gap CLI and ISN Feedback

#### 3A: `sn gaps` CLI command

```bash
sn gaps                          # List all vocab gaps from graph
sn gaps --segment transformation # Filter by segment
sn gaps --export yaml            # Export for ISN issue filing
```

#### 3B: Post-ISN-release regeneration workflow

After ISN releases vocab extensions:
```bash
uv pip install --upgrade imas-standard-names
sn reset --status vocab_gap --to drafted
sn generate --source dd --force  # on affected IDS
sn review --re-review
```

### Phase 4: Cross-Provenance Leveling (Lower Priority)

#### 4A: Regenerate Gemini Flash Lite names (31 names, avg 0.660)

```bash
sn generate --source dd --ids equilibrium --ids core_profiles --force --cost-limit 3.0
```

#### 4B: Regenerate poor Old CLI names (39 names, avg 0.623)

Target `adequate` and `poor` tier only.

#### 4C: Review ISN-imported names (25, never reviewed)

```bash
sn review --status accepted --cost-limit 1.0
```

#### 4D: Name deduplication audit

Check for duplicate concepts across provenance groups.

### Phase 5: Targeted Remediation (Future)

#### 5A: `sn fix` subcommand

Separate grammar fixes from documentation fixes. Uses `reviewer_scores` to determine
what to fix automatically. Not needed until Phases 1-3 are complete.

---

## Implementation Priority

| Phase | Effort | Impact | Priority |
|-------|--------|--------|----------|
| **1A-1D (compose prompt fixes)** | **Small** | **High — prevents grammar errors at source** | **P0** |
| 2A-2B (vocab gap persistence) | Small | High — closes designed pipeline | P0 |
| 2C (review grammar richness) | Small | Medium — catches base-only in review | P1 |
| 2D (tests) | Small | Medium — validates Phase 2 | P1 |
| 3A (sn gaps CLI) | Small | Medium — enables ISN feedback | P2 |
| 4A-4C (cross-provenance) | Medium | Medium — levels quality | P3 |
| 5A (sn fix) | Medium | Medium — targeted remediation | P3 |

## ISN Plan Relationship

ISN plan 06 (`~/Code/imas-standard-names/plans/features/06-vocabulary-extensions-from-codex-bootstrap.md`)
proposes vocab extensions (7 transformations, 3 processes, 2 subjects) based on this investigation.

**Sequence:** Codex Phase 1 prompt fixes FIRST → reduces grammar errors at source.
Then Phase 2 gap wiring → collects remaining gaps for ISN. ISN vocab extensions come
AFTER codex-side improvements, when we have clean gap data from improved prompts.

Do NOT implement ISN plan 06 yet. The codex-side prompt improvements (Phase 1) will
reduce the gap count significantly. Many "gaps" are actually the LLM not using grammar
that already exists (e.g., `bootstrap_current` vs `plasma_current_due_to_bootstrap`).

## Documentation Updates

| Target | Update Needed |
|--------|---------------|
| `AGENTS.md` | Update SN section with vocab gap persistence, `sn gaps` command when implemented |
| Prompt templates | Phase 1 changes to `_grammar_reference.md` |
