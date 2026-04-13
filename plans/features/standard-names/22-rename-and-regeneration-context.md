# 22: Rename sn→standard_names, mint→generate, and Regeneration Context

**Status:** Ready to implement
**Depends on:** Plan 21 (architecture boundary) — complete
**Effort:** 5 phases, partially parallelizable

## Problem Statement

Three related improvements need to happen together:

1. **Navigational clarity** — `imas_codex/standard_names/` is cryptic; `imas_codex/standard_names/`
   is self-documenting. The CLI subcommand `sn` stays (it's short for terminal use),
   but the Python package, test directory, and documentation should use the full name.

2. **Command naming** — `sn mint` implies a one-shot minting action. `sn generate`
   better conveys the iterative, improvable nature of LLM-backed name creation.
   The `minted` status does NOT exist (confirmed: schema uses `drafted`), so this
   is a CLI-only rename.

3. **Regeneration context** — When `--force` re-generates names for paths that already
   have a `HAS_STANDARD_NAME` relationship, the LLM currently sees a flat list of ALL
   existing names for collision avoidance, but does NOT know which specific name was
   previously assigned to each path in the batch. This is a missed signal: the LLM
   should see `previous_name: electron_temperature` on the path item so it can either
   reuse, improve, or consciously replace it. This context must be framed as a
   **suggestion, not an anchor** — the LLM must feel free to diverge.

## Design Decisions

### D1: `sn/` directory → `standard_names/` (Python package only)

The CLI subcommand stays `sn` — it's a terminal command and brevity matters.
The rename is:
- `imas_codex/standard_names/` → `imas_codex/standard_names/`
- `tests/sn/` → `tests/standard_names/`
- All `from imas_codex.standard_names.X` → `from imas_codex.standard_names.X`

**NOT renamed:**
- CLI group name (`sn`) — stays short for terminal use
- Prompt subdirectory (`imas_codex/llm/prompts/sn/`) — prompt names like `sn/compose_dd`
  are internal identifiers, not navigational paths. Renaming would break prompt caching
  (names are keys in the prompt registry) and adds no clarity.
- Shared includes (`imas_codex/llm/prompts/shared/sn/`) — same reasoning
- Config section (`[tool.imas-codex.sn.benchmark]`) — pyproject.toml config key
- Config file (`imas_codex/llm/config/sn_review_criteria.yaml`) — data file name
- Worker log names (`sn_extract_worker`, etc.) — log prefixes are short by convention
- Schema prefix (`sn: https://imas.iter.org/schemas/standard_name/`) — XML namespace

### D2: `mint` → `generate` (CLI command + pipeline function)

Rename scope:
- `sn.command("mint")` → `sn.command("generate")`
- `sn_mint()` function → `sn_generate()`
- `run_sn_mint_engine()` → `run_sn_generate_engine()`
- Help text: "Mint" → "Generate" everywhere
- AGENTS.md: `sn mint` → `sn generate` in all tables, code blocks, prose
- Docs: `standard-names.md`, `standard-names-decisions.md`
- Skills: `service-ops/SKILL.md`

**NOT renamed:**
- `minted` status — does not exist. Schema uses `drafted` for LLM-generated names.
- `--reset-to` choices remain `extracted` and `drafted` (no `minted` value to change)

### D3: Previous Name Injection (Regeneration Context)

**Current flow (--force):**
```
extract_worker:
  get_named_source_ids()    → skipped (force=True)
  get_existing_standard_names() → Set[all SN IDs]  ← flat collision list
  
compose_worker:
  existing_names = sorted(batch.existing_names)[:200]  ← ALL names, not per-path
  nearby_existing_names = search_similar_names(...)     ← semantic search
```

**New flow (--force):**
```
extract_worker:
  get_named_source_ids()    → skipped (force=True)
  get_existing_standard_names() → Set[all SN IDs]  ← flat collision list (keep)
  get_source_name_mapping()  → Dict[source_id → {name, description, kind}]  ← NEW
  
  For each item in batch:
    if item.path in source_name_mapping:
      item["previous_name"] = source_name_mapping[item.path]  ← INJECT

compose_worker:
  existing_names = sorted(batch.existing_names)[:200]  ← unchanged
  nearby_existing_names = ...                           ← unchanged
  # previous_name already on each item, rendered in template
```

**Prompt rendering (compose_dd.md):**
```markdown
### equilibrium/time_slice/profiles_1d/psi
- **Description:** Poloidal magnetic flux
- **Unit:** Wb
- **Previous name:** `poloidal_magnetic_flux` *(from prior generation — reuse if
  still appropriate, or improve. Do NOT feel anchored to this name.)*
```

The framing is critical: "reuse if still appropriate, or improve" — not "keep this name".
The LLM should treat it as one signal among many, not as a constraint.

### D4: `--fresh` flag instead of `--exclude-existing`

**Decision: Not needed.** This is overthinking. The existing mechanisms cover all cases:

| Goal | Mechanism |
|------|-----------|
| Generate names for uncovered paths only | `sn generate` (default — skips named paths) |
| Regenerate names, with previous context | `sn generate --force` (includes named paths, shows previous_name) |
| Regenerate from scratch, no prior context | `sn reset --ids equilibrium && sn generate` (clear first, then generate fresh) |

Adding `--fresh` / `--exclude-existing` would create a confusing middle ground between
`--force` (improve) and reset+generate (start over). The two-step `reset && generate`
workflow is explicit and avoids feature creep.

### D5: `--force` Shows Previous Name by Default

When `--force` is used, previous names are ALWAYS shown to the LLM. There is no flag
to suppress this — if you want a clean slate, use `sn reset` first. This keeps the
`--force` contract simple: "re-generate with full context including what was there before."

## Scope Inventory

### Python Package Rename (`imas_codex/standard_names/` → `imas_codex/standard_names/`)

**Source files to move (22 files):**
```
imas_codex/standard_names/__init__.py          → imas_codex/standard_names/__init__.py
imas_codex/standard_names/benchmark.py         → imas_codex/standard_names/benchmark.py
imas_codex/standard_names/benchmark_calibration.yaml → imas_codex/standard_names/benchmark_calibration.yaml
imas_codex/standard_names/benchmark_reference.py → imas_codex/standard_names/benchmark_reference.py
imas_codex/standard_names/calibration.py       → imas_codex/standard_names/calibration.py
imas_codex/standard_names/catalog_import.py    → imas_codex/standard_names/catalog_import.py
imas_codex/standard_names/classifier.py        → imas_codex/standard_names/classifier.py
imas_codex/standard_names/consolidation.py     → imas_codex/standard_names/consolidation.py
imas_codex/standard_names/context.py           → imas_codex/standard_names/context.py
imas_codex/standard_names/enrichment.py        → imas_codex/standard_names/enrichment.py
imas_codex/standard_names/graph_ops.py         → imas_codex/standard_names/graph_ops.py
imas_codex/standard_names/models.py            → imas_codex/standard_names/models.py
imas_codex/standard_names/pipeline.py          → imas_codex/standard_names/pipeline.py
imas_codex/standard_names/progress.py          → imas_codex/standard_names/progress.py
imas_codex/standard_names/publish.py           → imas_codex/standard_names/publish.py
imas_codex/standard_names/search.py            → imas_codex/standard_names/search.py
imas_codex/standard_names/seed.py              → imas_codex/standard_names/seed.py
imas_codex/standard_names/state.py             → imas_codex/standard_names/state.py
imas_codex/standard_names/workers.py           → imas_codex/standard_names/workers.py
imas_codex/standard_names/sources/__init__.py  → imas_codex/standard_names/sources/__init__.py
imas_codex/standard_names/sources/base.py      → imas_codex/standard_names/sources/base.py
imas_codex/standard_names/sources/dd.py        → imas_codex/standard_names/sources/dd.py
imas_codex/standard_names/sources/signals.py   → imas_codex/standard_names/sources/signals.py
```

**Test files to move (19 files):**
```
tests/sn/conftest.py               → tests/standard_names/conftest.py
tests/sn/__init__.py               → tests/standard_names/__init__.py
tests/sn/test_benchmark.py         → tests/standard_names/test_benchmark.py
tests/sn/test_calibration.py       → tests/standard_names/test_calibration.py
tests/sn/test_catalog_import.py    → tests/standard_names/test_catalog_import.py
tests/sn/test_classifier.py        → tests/standard_names/test_classifier.py
tests/sn/test_consolidation.py     → tests/standard_names/test_consolidation.py
tests/sn/test_enrichment.py        → tests/standard_names/test_enrichment.py
tests/sn/test_grammar_contract.py  → tests/standard_names/test_grammar_contract.py
tests/sn/test_graph_ops.py         → tests/standard_names/test_graph_ops.py
tests/sn/test_integration.py       → tests/standard_names/test_integration.py
tests/sn/test_publish.py           → tests/standard_names/test_publish.py
tests/sn/test_review.py            → tests/standard_names/test_review.py
tests/sn/test_scoring.py           → tests/standard_names/test_scoring.py
tests/sn/test_search.py            → tests/standard_names/test_search.py
tests/sn/test_seed.py              → tests/standard_names/test_seed.py
tests/sn/test_sn_tools.py          → tests/standard_names/test_sn_tools.py
tests/sn/test_validate_isn.py      → tests/standard_names/test_validate_isn.py
tests/sn/test_validate_persistence.py → tests/standard_names/test_validate_persistence.py
```

**Files with imports to update (~30 files):**
- `imas_codex/cli/sn.py` — ~15 lazy imports from `imas_codex.standard_names.*`
- All 22 source files in `imas_codex/standard_names/` that cross-import siblings
- `imas_codex/cli/__init__.py` — line 73 `from imas_codex.cli.sn import sn`
- All 19 test files that import from `imas_codex.standard_names.*`

**Documentation to update:**
- `AGENTS.md` — ~17 lines referencing `sn/`, `imas_codex.sn`, `sn mint`
- `docs/architecture/standard-names.md` — references to `sn mint`, `imas_codex/standard_names/`
- `docs/architecture/standard-names-decisions.md` — `sn mint` references
- `.github/skills/service-ops/SKILL.md` — `sn mint` reference
- `plans/features/standard-names/09-sn-generate.md` — historical plan, update references
- `plans/README.md` — if it references `sn/`

### `mint` → `generate` Rename

**Code changes:**
- `imas_codex/cli/sn.py`:
  - `@sn.command("mint")` → `@sn.command("generate")`
  - `def sn_mint(` → `def sn_generate(`
  - Help text: "Mint" → "Generate" in docstrings and usage examples
- `imas_codex/standard_names/pipeline.py`:
  - `async def run_sn_mint_engine(` → `async def run_sn_generate_engine(`
- All callers of `run_sn_mint_engine`

**Documentation:**
- `AGENTS.md`: `sn mint` → `sn generate` (~8 occurrences)
- `docs/architecture/standard-names.md`: `sn mint` → `sn generate`
- `docs/architecture/standard-names-decisions.md`: `sn mint` → `sn generate`
- `.github/skills/service-ops/SKILL.md`: `sn mint` → `sn generate`

### Previous Name Injection

**New graph query in `graph_ops.py`:**
```python
def get_source_name_mapping() -> dict[str, dict]:
    """Return mapping of source_id → previous standard name info.
    
    Used by extract_worker in --force mode to inject previous_name
    context into batch items so the LLM can improve on prior names.
    """
    with GraphClient() as gc:
        results = gc.query("""
            MATCH (src)-[:HAS_STANDARD_NAME]->(sn:StandardName)
            RETURN src.id AS source_id,
                   sn.id AS name,
                   sn.description AS description,
                   sn.kind AS kind,
                   sn.review_status AS review_status
        """)
        mapping = {}
        for r in results:
            mapping[r["source_id"]] = {
                "name": r["name"],
                "description": r.get("description"),
                "kind": r.get("kind"),
                "review_status": r.get("review_status"),
            }
        return mapping
```

**Extract worker changes (`workers.py`):**
```python
# In extract_worker, after batches are built:
if state.force:
    source_names = get_source_name_mapping()
    for batch in batches:
        for item in batch.items:
            path = item.get("path", item.get("signal_id"))
            if path and path in source_names:
                item["previous_name"] = source_names[path]
```

**Prompt template changes (`compose_dd.md`):**
```markdown
### {{ item.path }}
- **Description:** {{ item.description }}
{% if item.previous_name %}
- **Previous name:** `{{ item.previous_name.name }}` *(from prior generation —
  reuse if still appropriate, or improve. This is a suggestion, not a constraint.)*
{% if item.previous_name.description %}  - Prior description: {{ item.previous_name.description[:200] }}{% endif %}
{% if item.previous_name.review_status == 'accepted' %}  - ⚠️ This name was human-accepted — only replace with a clearly better alternative.{% endif %}
{% endif %}
```

Note the escalation: if the previous name was `accepted`, the prompt warns the LLM
to only replace it if clearly better. For `drafted` names, the LLM has full freedom.

**Test requirements:**
- Unit test: extract_worker with force=True injects previous_name into items
- Unit test: compose prompt renders previous_name section correctly
- Unit test: accepted names get the "only replace if clearly better" warning
- Integration test: end-to-end force regeneration includes previous_name context

## Phases

### Phase 1: Package Rename (`imas_codex/standard_names/` → `imas_codex/standard_names/`)

**Parallelism:** Must complete BEFORE all other phases (they need correct import paths)
**Agent:** Single agent with `git mv` + sed-based import rewriting
**Estimated files:** ~70 files

**Steps:**
1. Create `imas_codex/standard_names/` and `imas_codex/standard_names/sources/`
2. `git mv` all files from `imas_codex/standard_names/` to `imas_codex/standard_names/`
3. `git mv` all files from `tests/sn/` to `tests/standard_names/`
4. Find-and-replace ALL `from imas_codex.standard_names.` → `from imas_codex.standard_names.` in:
   - All `.py` files under `imas_codex/`
   - All `.py` files under `tests/`
5. Update `imas_codex/cli/__init__.py` line 73 (if it imports from the old path)
6. Clean up empty `imas_codex/standard_names/` and `tests/sn/` directories
7. Run `uv run ruff check --fix . && uv run ruff format .`
8. Run `uv run pytest tests/standard_names/ -x -q` to verify all 480+ tests pass
9. Do NOT commit yet — Phase 2 will commit together

**Critical constraints:**
- The CLI file stays at `imas_codex/cli/sn.py` — it's the `sn` subcommand, not a package file
- Prompt paths (`sn/compose_dd`, etc.) are NOT renamed — they're prompt registry keys
- The `render_prompt("sn/compose_dd", ...)` calls are NOT changed
- `[tool.imas-codex.sn.benchmark]` config key is NOT changed
- Worker log names (`sn_extract_worker`) are NOT changed

### Phase 2: Command Rename (`mint` → `generate`)

**Parallelism:** Can run in parallel with Phase 1 IF working on separate files,
  but safer to run after Phase 1 to avoid merge conflicts in `cli/sn.py`
**Agent:** Single agent

**Steps:**
1. In `imas_codex/cli/sn.py`:
   - `@sn.command("mint")` → `@sn.command("generate")`
   - `def sn_mint(` → `def sn_generate(`
   - Update all help text / docstrings: "mint" → "generate", "Mint" → "Generate"
   - Update usage examples in docstrings
2. In `imas_codex/standard_names/pipeline.py` (note: new path from Phase 1):
   - `async def run_sn_mint_engine(` → `async def run_sn_generate_engine(`
   - Update all docstrings
3. Update all callers of `run_sn_mint_engine` → `run_sn_generate_engine`
   (only in `imas_codex/cli/sn.py`)
4. Run `uv run ruff check --fix . && uv run ruff format .`
5. Run `uv run pytest tests/standard_names/ -x -q` to verify
6. Do NOT commit yet — will commit with Phase 1

### Phase 3: Documentation Update

**Parallelism:** Can run in parallel with Phase 2 (touches different files)
**Agent:** Single agent

**Steps:**
1. `AGENTS.md` — Update all references:
   - `sn mint` → `sn generate` (~8 occurrences in tables, code blocks, prose)
   - `imas_codex/standard_names/` → `imas_codex/standard_names/` where path is referenced
   - `imas_codex.standard_names.` → `imas_codex.standard_names.` where import is referenced
   - Update the CLI commands table (line ~749)
   - Update the key modules table
   - Update the lifecycle documentation
   - Update `sn mint --reset-to` → `sn generate --reset-to`
2. `docs/architecture/standard-names.md`:
   - `sn mint` → `sn generate`
   - Module path references
3. `docs/architecture/standard-names-decisions.md`:
   - `sn mint` → `sn generate`
4. `.github/skills/service-ops/SKILL.md`:
   - `sn mint` → `sn generate`
5. Do NOT update plans/ — they are historical records

### Phase 4: Previous Name Injection (Regeneration Context)

**Parallelism:** Can run in parallel with Phase 3 (touches worker/graph code, not docs)
**Depends on:** Phase 1 (needs correct import paths in `imas_codex/standard_names/`)
**Agent:** Opus 4.6 — this requires careful prompt engineering

**Steps:**

1. **New graph query** — Add `get_source_name_mapping()` to
   `imas_codex/standard_names/graph_ops.py`:
   ```python
   def get_source_name_mapping() -> dict[str, dict]:
       """Return mapping of source_id → previous standard name details.
       
       Used by extract_worker in --force mode to inject per-path
       previous_name context so the LLM can improve on prior names.
       Returns dict mapping source_id to {name, description, kind, review_status}.
       """
       with GraphClient() as gc:
           results = gc.query("""
               MATCH (src)-[:HAS_STANDARD_NAME]->(sn:StandardName)
               RETURN src.id AS source_id,
                      sn.id AS name,
                      sn.description AS description,
                      sn.kind AS kind,
                      sn.review_status AS review_status
           """)
           mapping = {}
           for r in results:
               sid = r["source_id"]
               # If multiple names exist for same source, prefer accepted
               if sid not in mapping or r.get("review_status") == "accepted":
                   mapping[sid] = {
                       "name": r["name"],
                       "description": r.get("description"),
                       "kind": r.get("kind"),
                       "review_status": r.get("review_status"),
                   }
           return mapping
   ```

2. **Extract worker update** — In `imas_codex/standard_names/workers.py`,
   inside `extract_worker()`, after batches are built and ONLY when `state.force`:
   ```python
   # Inject previous name context for --force regeneration
   if state.force:
       from imas_codex.standard_names.graph_ops import get_source_name_mapping
       source_names = get_source_name_mapping()
       injected = 0
       for batch in batches:
           for item in batch.items:
               path = item.get("path", item.get("signal_id"))
               if path and path in source_names:
                   item["previous_name"] = source_names[path]
                   injected += 1
       if injected:
           wlog.info("Injected previous_name context for %d items", injected)
   ```

3. **Prompt template update** — In `imas_codex/llm/prompts/sn/compose_dd.md`,
   add to the per-item rendering block (after the existing fields, before
   cluster_siblings):
   ```markdown
   {% if item.previous_name %}
   - **Previous name:** `{{ item.previous_name.name }}` *(from prior generation —
     reuse if still appropriate, or improve. This is a suggestion, not a constraint.)*
   {% if item.previous_name.description %}  - Prior docs: {{ item.previous_name.description[:200] }}{% endif %}
   {% if item.previous_name.review_status == 'accepted' %}  - ⚠️ This name was human-accepted — prefer keeping it unless you have a clearly better alternative.{% endif %}
   {% endif %}
   ```

4. **System prompt guidance** — In `imas_codex/llm/prompts/sn/compose_system.md`,
   add a section under "Composition Rules" (after rule 9):
   ```markdown
   10. When a **Previous name** is shown for a path, treat it as context:
       - If the previous name is good, reuse it (stability matters)
       - If you can clearly improve it, do so and explain in the documentation
       - If the previous name was marked as human-accepted (⚠️), strongly prefer keeping it
       - Never feel anchored — a bad previous name should be replaced without hesitation
   ```

5. **Tests** — Add to `tests/standard_names/test_integration.py` or a new
   `tests/standard_names/test_regeneration_context.py`:
   - `test_get_source_name_mapping` — mock GraphClient, verify mapping structure
   - `test_extract_worker_injects_previous_name_on_force` — mock graph ops,
     verify items have `previous_name` when force=True
   - `test_extract_worker_no_previous_name_without_force` — verify items do NOT
     have `previous_name` when force=False
   - `test_compose_prompt_renders_previous_name` — render template with
     `previous_name` in item, verify markdown output contains the section
   - `test_compose_prompt_accepted_warning` — render template with
     `previous_name.review_status = "accepted"`, verify ⚠️ warning appears
   - `test_compose_prompt_no_previous_name` — render template without
     `previous_name`, verify section is absent

### Phase 5: Commit, Verify, and Push

**Parallelism:** Sequential — must run after all other phases
**Agent:** Single agent or manual

**Steps:**
1. Run full test suite: `uv run pytest tests/standard_names/ -x -q`
2. Run lint: `uv run ruff check --fix . && uv run ruff format .`
3. Verify no broken imports: `uv run python -c "from imas_codex.standard_names.pipeline import run_sn_generate_engine; print('OK')"`
4. Verify CLI: `uv run imas-codex sn generate --help` (should show generate, not mint)
5. Verify old paths are gone: `test ! -d imas_codex/sn` (should succeed)
6. Stage only modified files (never `git add -A`):
   ```bash
   git add imas_codex/standard_names/ tests/standard_names/
   git add imas_codex/cli/sn.py imas_codex/cli/__init__.py
   git add AGENTS.md docs/ .github/skills/
   git add imas_codex/llm/prompts/sn/  # prompt template changes
   ```
7. Commit:
   ```bash
   uv run git commit -m "refactor: rename sn→standard_names, mint→generate, add regeneration context

   - Move imas_codex/standard_names/ to imas_codex/standard_names/ for navigational clarity
   - Move tests/sn/ to tests/standard_names/
   - Rename CLI command 'sn mint' to 'sn generate'
   - Rename run_sn_mint_engine to run_sn_generate_engine
   - Add previous_name injection in --force mode for regeneration context
   - Add get_source_name_mapping() graph query
   - Update compose_dd.md and compose_system.md for previous name handling
   - Update AGENTS.md, architecture docs, and skill files"
   ```
8. Pull and push: `git pull --no-rebase origin main && git push origin main`

## Fleet Execution Strategy

### Wave 1 (single agent, blocking)
- **Phase 1 + Phase 2**: Package rename + command rename
  - Must be ONE agent to avoid merge conflicts in shared files (`cli/sn.py`)
  - This agent does the `git mv`, import rewriting, and function renames
  - Validates with pytest before completing

### Wave 2 (parallel, after Wave 1)
- **Phase 3**: Documentation update agent (AGENTS.md, docs/, skills/)
- **Phase 4**: Regeneration context agent (graph_ops, workers, prompts, tests)

### Wave 3 (sequential, after Wave 2)
- **Phase 5**: Verify + commit + push

## Verification Checklist

After all phases complete:

- [ ] `imas_codex/standard_names/` directory does not exist
- [ ] `tests/sn/` directory does not exist
- [ ] `imas_codex/standard_names/__init__.py` exists
- [ ] `tests/standard_names/__init__.py` exists
- [ ] `uv run imas-codex sn generate --help` works
- [ ] `uv run imas-codex sn mint --help` fails (command not found)
- [ ] `uv run pytest tests/standard_names/ -x -q` passes all ~480 tests
- [ ] `grep -r "from imas_codex.sn\." --include="*.py" imas_codex/ tests/` returns nothing
- [ ] `grep -r "import imas_codex.sn" --include="*.py" imas_codex/ tests/` returns nothing
- [ ] `uv run python -c "from imas_codex.standard_names.pipeline import run_sn_generate_engine"` succeeds
- [ ] `uv run python -c "from imas_codex.standard_names.graph_ops import get_source_name_mapping"` succeeds
- [ ] AGENTS.md contains no `sn mint` references (only `sn generate`)
- [ ] AGENTS.md `sn/` → `standard_names/` where referencing Python paths

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Missed import during rename | Phase 1 uses `grep -r` verification; Phase 5 runs full test suite |
| Prompt cache invalidation | Prompt names (`sn/compose_dd`) are NOT changed — cache preserved |
| Parallel agent conflicts | Phase 1+2 are single-agent; Wave 2 agents touch disjoint files |
| Previous name anchoring | Prompt explicitly says "suggestion, not constraint"; accepted names get extra protection |
| `pyproject.toml` config key | `[tool.imas-codex.sn.benchmark]` stays — `settings.py` reads this key, no rename needed |
| Historical plans referencing `sn mint` | Plans are historical records — NOT updated (decision D2) |
