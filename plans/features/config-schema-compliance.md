# Config Schema Compliance: Parameter Utilization & Cleanup

**Goal**: Ensure all facility config schema fields are (a) named generically, (b) consumed by discovery pipelines, and (c) not hardcoded in scanner code. Remove or relocate documentary-only fields that clutter the typed schema. Fix scanner code that ignores config it should read.

**Prerequisite**: Completed facility-config-schema-restructure plan (committed `a8cc4a1`).

**Scope**: `facility_config.yaml` schema, 4 public YAML configs, 5 scanner plugins, 3 remote scripts, wiki CLI dispatch, graph facility schema.

---

## Problem Summary

The previous assessment identified 4 categories of issues across ~120 schema fields:

| Category | Count | Issue |
|----------|-------|-------|
| Facility-specific naming | 1 | `jpf_subsystems` embeds JET acronym in schema |
| Documentary fields masquerading as schema | 2 | `tdi_access_functions`, `missing_libraries` — never consumed |
| Hardcoded paths ignoring config | 4 | EDAS scanner hardcodes `lib_path`, `api_path`; `limiter_dir` ignored |
| Unconsumed forward-looking fields | 5 | `languages`, `sal_endpoint`, `greens`, `device_ppfs`, `n_points` (LimiterVersion) |

Additionally, the **EDAS scanner** builds `DataAccess.connection_template` with hardcoded paths (`/analysis/src/eddb`, `/analysis/lib/libeddb.so`) instead of reading from `config.api_path` and `config.lib_path`.

---

## Phase 1: Schema Cleanup — Rename, Remove, Relocate

### 1.1 Rename `jpf_subsystems` → `subsystem_codes`

**Schema**: `facility_config.yaml` → `MDSplusConfig`

The field holds two-character diagnostic subsystem identifiers. While JET calls them "JPF subsystems," the concept — a list of subsystem codes for TDI function enumeration — is generic. Any MDSplus thin-client with subsystem-based organization could use it.

**Changes**:
- Schema: rename slot `jpf_subsystems` → `subsystem_codes`, update description to be generic
- Config: `jet.yaml` rename key `jpf_subsystems:` → `subsystem_codes:`
- Scanner: `mdsplus.py` change `config.get("jpf_subsystems")` → `config.get("subsystem_codes")`
- Remote script: `enumerate_mdsplus_tdi.py` change JSON input key `jpf_subsystems` → `subsystem_codes`
- Documentation: scanner docstring update

**Graph migration**: None — `jpf_subsystems` is never persisted to the graph. It's only used as scanner input.

### 1.2 Remove `tdi_access_functions` from schema

**Schema**: `facility_config.yaml` → `MDSplusConfig`

This field is explicitly marked "Not consumed programmatically — for reference only" in the schema description. It's a documentary mapping of TDI function signatures (e.g., `dpf("{subsystem}/{signal}", {shot})`). It required a `build_models.py` type fix override (`Optional[str]` → `Optional[dict]`) because LinkML's `default_range: string` can't handle dicts.

**Problem**: Documentary data in a typed schema creates maintenance burden (type fix in build pipeline) and schema noise. The function signatures belong in `data_access_patterns.notes` — exactly where other documentary data lives.

**Changes**:
- Schema: remove `tdi_access_functions` slot from `MDSplusConfig`
- Config: `jet.yaml` move `tdi_access_functions` block → `data_access_patterns.notes` entries:
  - `"TDI functions: dpf('{subsystem}/{signal}', {shot}) for JPF, ppf('{dda}/{dtype}', {shot}) for PPF"`
  - `"jpfsubsystems() lists all subsystem codes, jpfincludedsubsystems({shot}) for active"`
- Build: remove `tdi_access_functions` type fix from `build_models.py`

**Graph migration**: None — the field was never persisted.

### 1.3 Remove `missing_libraries` from schema

**Schema**: `facility_config.yaml` → `DataAccessPatternsConfig`

Same category as `tdi_access_functions` — a documentary field tracking known shared library issues. Description says "Documentary field for tracking known blockers." Required a `build_models.py` type fix override. No code reads it.

**Changes**:
- Schema: remove `missing_libraries` slot from `DataAccessPatternsConfig`
- Config: `tcv.yaml` move `missing_libraries` block → `data_access_patterns.notes` entries:
  - `"Missing libjmmshr_gsl.so blocks GREENS computation in static tree (greenem.fun)"`
  - `"Missing libblas.so symlink (libblas.so.3 exists) blocks _matmul.fun used by staticgreen"`
- Build: remove `missing_libraries` type fix from `build_models.py`

**Graph migration**: None.

### 1.4 Resolve `limiter_dir` redundancy

**Schema**: `facility_config.yaml` → `DeviceXMLConfig`

The `DeviceXMLConfig.limiter_dir` field is redundant with `LimiterVersion.source_dir`. The scanner (`device_xml.py:1957`) reads `source_dir` from each `LimiterVersion` entry, never `limiter_dir` from the parent. However, `limiter_dir` was added by the restructure plan as a "default filesystem fallback."

**Resolution**: Make `limiter_dir` the default source for `LimiterVersion` entries that lack their own `source_dir`. This completes the plan's intent.

**Changes**:
- Scanner: `device_xml.py` — when building `limiter_files` list, if a `LimiterVersion` lacks `source_dir`, fall back to parent `config.get("limiter_dir")`:
  ```python
  default_limiter_dir = config.get("limiter_dir")
  for lv in limiter_versions:
      if lv.get("file"):
          entry = {"name": lv["name"], "file": lv["file"]}
          source_dir = lv.get("source_dir") or default_limiter_dir
          if source_dir:
              entry["source_dir"] = source_dir
          limiter_files.append(entry)
  ```
- Config: `jet.yaml` — `Mk2ILW` limiter entry currently has no `source_dir` because the file is in the git repo. The `limiter_dir` value (`/home/chain1/input/efit/Limiters`) serves as fallback for `Mk2A`, `Mk2GB`, `Mk2GB-NS`, `Mk2HD` entries. Those currently have explicit `source_dir` so no YAML change needed, but the code now handles the fallback correctly.

**Graph migration**: None — `limiter_dir` is config-only, not graphed.

---

## Phase 2: Parameterize Hardcoded Paths

### 2.1 EDAS scanner: read `lib_path` and `api_path` from config

**Critical issue**: The EDAS scanner and both remote scripts (`enumerate_edas.py`, `check_edas.py`) hardcode three JT-60SA-specific paths:

| Path | Hardcoded in | Config field |
|------|-------------|-------------|
| `/analysis/lib/libeddb.so` | `edas.py:126`, `enumerate_edas.py:90`, `check_edas.py:86` | `lib_path` |
| `/analysis/src/eddb` | `edas.py:124`, `enumerate_edas.py:56` (multiple), `check_edas.py:73` | `api_path` |
| `/analysis/lib` | `enumerate_edas.py:60`, `check_edas.py:77` | (fallback for api_path) |

**Architecture**: The remote scripts run via `async_run_python_script()` which passes JSON on stdin. Config values must flow through the JSON input.

**Changes**:

#### 2.1.1 Scanner (`edas.py`)

- Read `lib_path` and `api_path` from config dict
- Pass them to remote script as JSON input fields
- Use them in `DataAccess.connection_template` instead of hardcoded paths:

```python
api_path = config.get("api_path", "/analysis/src/eddb")
lib_path = config.get("lib_path", "/analysis/lib/libeddb.so")

# Pass to remote script
output = await async_run_python_script(
    "enumerate_edas.py",
    {
        "ref_shot": shot_str,
        "api_path": api_path,
        "lib_path": lib_path,
    },
    ...
)

# DataAccess node uses config values
connection_template=(
    f"import sys\n"
    f"sys.path.insert(0, '{api_path}')\n"
    f"from eddb_pwrapper import eddbWrapper\n"
    f"db = eddbWrapper('{lib_path}')\n"
    f"db.eddbOpen()"
),
```

#### 2.1.2 Remote scripts (`enumerate_edas.py`, `check_edas.py`)

- Read `api_path` and `lib_path` from JSON input
- Use them instead of hardcoded paths
- Keep hardcoded values as defaults for backward compatibility:

```python
api_path = config.get("api_path", "/analysis/src/eddb")
lib_path = config.get("lib_path", "/analysis/lib/libeddb.so")

sys.path.insert(0, api_path)
# ... import eddbWrapper ...
db = eddbWrapper(lib_path)
```

#### 2.1.3 Check script (`check_edas.py`)

Same pattern as enumerate — read paths from JSON input with defaults.

**Graph migration**: Existing `DataAccess` node `jt-60sa:edas:eddb` has the hardcoded path in `connection_template`. After running the scanner with the fix, the node will be updated on next `discover signals jt-60sa -s edas` run via `MERGE` on the node ID. **No manual migration needed** — the scanner creates/updates the DataAccess node on each run.

### 2.2 Validate `api_path` and `header_path` are consumed

After 2.1, `api_path` and `lib_path` will be consumed. `header_path` remains documentary — no scanner reads C headers. However, `header_path` is a legitimate forward-looking field for a future C-header-based signal discovery approach. **No action needed** — leave as documentary with clear description.

---

## Phase 3: Consume Forward-Looking Config Fields

### 3.1 `languages` → WikiPage graph property + scoring hint

**Current state**: `WikiSiteConfig.languages` exists in schema and JT-60SA config (`["ja", "en"]`) but no code reads it. JT-60SA wiki content is bilingual Japanese/English — LLM scoring of Japanese pages is suboptimal without language context.

**Changes**:

#### 3.1.1 Graph schema: add `content_language` to WikiPage

Add a `content_language` slot to `WikiPage` in `imas_codex/schemas/facility.yaml`:

```yaml
content_language:
  description: >-
    Primary natural language of page content (ISO 639-1 code).
    Used to select appropriate LLM prompts for non-English content.
    E.g., "ja" for Japanese, "en" for English.
```

#### 3.1.2 Wiki CLI dispatch: pass `languages` to bulk discovery

In `wiki.py` CLI, read `languages` from site config and pass to `bulk_discover_pages()`:

```python
elif site_type == "twiki_raw":
    discover_kwargs["data_path"] = site.get("data_path", base_url)
    discover_kwargs["web_name"] = site.get("web_name", "Main")
    discover_kwargs["exclude_patterns"] = site.get("exclude_patterns")
    discover_kwargs["languages"] = site.get("languages")  # NEW
```

#### 3.1.3 Bulk discovery: store `content_language` on WikiPage nodes

When creating WikiPage nodes in `parallel.py`, if `languages` is provided, set `content_language` on each page. For bilingual sites, detection requires content analysis — but storing the site-level hints is a first step:

- If site has exactly 1 language: set `content_language` directly
- If site has 2+ languages: set `content_language = languages[0]` (primary), add `available_languages` for reference

#### 3.1.4 Scoring: pass language hint to LLM prompt

In `scoring.py`, when generating scoring prompts, include language context so the LLM handles non-English content appropriately:

```python
if content_language and content_language != "en":
    context += f"\nNote: Content may be in {content_language}. Score based on technical content regardless of language."
```

**Graph migration**: Add `content_language` property to existing WikiPage nodes for JT-60SA. This is a Cypher update:

```cypher
MATCH (wp:WikiPage)
WHERE wp.facility_id = 'jt-60sa'
SET wp.content_language = 'ja'
```

This sets a reasonable default. Future scoring runs will refine per-page.

### 3.2 `sal_endpoint` → PPF scanner SAL fallback

**Current state**: `PPFConfig.sal_endpoint` defined in schema and `jet.yaml` (`https://sal.jet.uk`) but exploration notes say "SAL REST at sal.jet.uk — not responding as of exploration (2026)."

**Resolution**: SAL is currently unreachable. The PPF scanner uses ppf library via SSH instead. **No code change now** — leave as documentary. When SAL becomes available, the scanner can add a REST-based fallback using this config value.

**Action**: Add a comment to the PPF scanner docstring noting `sal_endpoint` is reserved for future SAL REST access. No code changes.

### 3.3 `greens` and `device_ppfs` → device_xml scanner parameters

**Current state**: Both are in `DeviceXMLVersion` and populated in `jet.yaml` but the scanner only reads `device_xml`, `snap_file`, and `limiter`. The `greens` path maps each epoch to its Green's function directory. The `device_ppfs` path maps to PPF routing namelists.

#### 3.3.1 `greens` → pass to remote parse script

The `greens` directory path is needed for full machine-description ingestion. The `parse_greens_table.py` script already handles Green's table parsing, and the `_persist_greens_table_nodes()` function exists in `device_xml.py:1669`.

**However**: There's a separate `greens_table` static source that handles Green's table version-to-shot mapping (read from a Fortran namelist, not from `DeviceXMLVersion.greens`). The `DeviceXMLVersion.greens` field simply records which Green's directory is active for each epoch — this is cross-referencing metadata, not input to a scanner.

**Resolution**: Leave `greens` as documentary metadata for now. It provides provenance — "epoch p79854 uses Green's from `DMSS_105_T_200C/`". The GreensTable ingestion pipeline reads the mapping file independently. If a future scanner needs to read the actual Green's function data (not just the mapping), it should use this path at that point.

**Action**: Update schema description to clarify its provenance role. No code changes.

#### 3.3.2 `device_ppfs` → document as provenance metadata

The PPF routing namelist (`Devices/MAGN/device_ppfs`) maps magnetic probes to PPF/JPF signal sources. It's the same file for all 14 epochs at JET. This is EFIT-specific provenance — useful for cross-referencing but not consumed by a scanner.

**Resolution**: Keep as documentary metadata. If a future PPF→MDSplus signal cross-referencing feature is built, it would read this file. Currently orthogonal to any scanner pipeline.

**Action**: Update schema description to clarify. No code changes.

### 3.4 `n_points` in LimiterVersion → validation in scanner

**Current state**: `LimiterVersion.n_points` is populated for Mk2A (108) and Mk2ILW (251) in `jet.yaml`. The remote parse script (`parse_device_xml.py`) already returns `n_points` per parsed limiter. But the scanner code (`device_xml.py`) doesn't cross-check config `n_points` against parsed `n_points`.

**Changes**:

In `device_xml.py`, after parsing limiter files, validate against config:

```python
for lv in limiter_versions:
    name = lv["name"]
    expected_n = lv.get("n_points")
    if expected_n and name in parsed_limiters:
        actual_n = parsed_limiters[name].get("n_points", 0)
        if actual_n != expected_n:
            logger.warning(
                "Limiter '%s': expected %d points, got %d",
                name, expected_n, actual_n,
            )
```

**Graph migration**: None — `n_points` is in config only.

---

## Phase 4: Graph Migration

### 4.1 Scope

Only two graph changes are needed:

| Change | Type | Nodes affected |
|--------|------|---------------|
| Add `content_language` property to WikiPage | Schema additive | JT-60SA WikiPages (~5800+) |
| Update `DataAccess` connection_template for EDAS | Data fix | 1 node (`jt-60sa:edas:eddb`) |

### 4.2 WikiPage content_language

**Schema change**: Add `content_language` slot to `WikiPage` in `imas_codex/schemas/facility.yaml`. This is additive — no breaking change.

**Data migration** (run after `uv run build-models --force`):

```cypher
// Set default content_language for JT-60SA wiki pages
MATCH (wp:WikiPage)
WHERE wp.facility_id = 'jt-60sa'
SET wp.content_language = 'ja'
RETURN count(wp) AS updated
```

Other facilities don't need this — their wiki content is English.

### 4.3 DataAccess EDAS template

**No manual migration** — running `discover signals jt-60sa -s edas` after the scanner fix will update the `DataAccess` node's `connection_template` via MERGE on the existing node ID `jt-60sa:edas:eddb`.

### 4.4 No destructive migrations

All changes are additive:
- New property `content_language` on WikiPage
- Updated string value in DataAccess.connection_template

No node deletions, no relationship changes, no schema breaking changes.

---

## Phase 5: Test Updates

### 5.1 Config compliance tests

After schema changes (removing `tdi_access_functions` and `missing_libraries`, renaming `jpf_subsystems`), the existing `tests/config/test_facility_config_compliance.py` tests will automatically catch any remaining drift — they compare YAML keys against model fields.

### 5.2 Scanner unit tests

Add or update tests for:

| Scanner | Test | What to verify |
|---------|------|----------------|
| EDAS | `test_edas_scanner_reads_lib_path` | Scanner reads `lib_path` from config, passes to remote script |
| EDAS | `test_edas_scanner_reads_api_path` | Scanner reads `api_path` from config, passes to remote script |
| MDSplus | `test_mdsplus_scanner_reads_subsystem_codes` | Scanner reads `subsystem_codes` (not `jpf_subsystems`) |
| device_xml | `test_device_xml_limiter_dir_fallback` | Scanner falls back to `limiter_dir` when `source_dir` absent |
| device_xml | `test_device_xml_n_points_validation` | Scanner warns when parsed n_points != config n_points |

### 5.3 Build pipeline test

Verify `build_models.py` type fix list no longer includes removed fields:
- `tdi_access_functions` type fix removed
- `missing_libraries` type fix removed

---

## Implementation Order

| Step | Phase | Files | Risk |
|------|-------|-------|------|
| 1 | 1.1 | `facility_config.yaml`, `jet.yaml`, `mdsplus.py`, `enumerate_mdsplus_tdi.py` | Low — field rename only |
| 2 | 1.2 | `facility_config.yaml`, `jet.yaml`, `build_models.py` | Low — remove unused field |
| 3 | 1.3 | `facility_config.yaml`, `tcv.yaml`, `build_models.py` | Low — remove unused field |
| 4 | 1.4 | `device_xml.py` | Low — add fallback logic |
| 5 | 2.1 | `edas.py`, `enumerate_edas.py`, `check_edas.py` | Medium — changes remote script interface |
| 6 | 3.1 | `facility.yaml`, `wiki.py`, `parallel.py`, `scoring.py` | Medium — new graph property |
| 7 | 3.4 | `device_xml.py` | Low — add validation logging |
| 8 | 4.2 | Cypher migration | Low — additive property |
| 9 | 5.* | Test files | Low |
| 10 | — | Regenerate models, run tests, commit | — |

**Estimated scope**: ~15 files modified, ~200 lines changed. No breaking changes to graph structure or CLI interface.

---

## Out of Scope

- **`sal_endpoint`**: SAL is unreachable (2026). Leave as documentary until SAL comes back online.
- **`header_path`**: EDAS C header parsing is a future feature. Leave as documentary.
- **`greens`/`device_ppfs`**: Provenance metadata, not scanner input. Descriptions updated but no code consumption.
- **Per-page language detection**: Phase 3.1 sets site-level default. True per-page detection (e.g., using `langdetect`) is a separate feature.

---

## Migration Checklist

- [ ] Rename `jpf_subsystems` → `subsystem_codes` in schema
- [ ] Rename `jpf_subsystems` → `subsystem_codes` in jet.yaml
- [ ] Update mdsplus.py scanner to read `subsystem_codes`
- [ ] Update enumerate_mdsplus_tdi.py to use `subsystem_codes` key
- [ ] Remove `tdi_access_functions` from schema
- [ ] Move tdi_access_functions content to jet.yaml notes
- [ ] Remove `tdi_access_functions` type fix from build_models.py
- [ ] Remove `missing_libraries` from schema
- [ ] Move missing_libraries content to tcv.yaml notes
- [ ] Remove `missing_libraries` type fix from build_models.py
- [ ] Implement `limiter_dir` fallback in device_xml.py
- [ ] Parameterize EDAS scanner to read lib_path, api_path from config
- [ ] Update enumerate_edas.py to accept lib_path, api_path in JSON
- [ ] Update check_edas.py to accept lib_path, api_path in JSON
- [ ] Add content_language to WikiPage in facility.yaml
- [ ] Pass languages config to wiki CLI dispatch
- [ ] Store content_language on WikiPage nodes
- [ ] Add language hint to scoring prompts
- [ ] Add n_points validation to device_xml scanner
- [ ] Update greens/device_ppfs schema descriptions
- [ ] `uv run build-models --force`
- [ ] Run graph migration (content_language for jt-60sa)
- [ ] Update/add scanner tests
- [ ] Run full test suite
- [ ] Commit and push
