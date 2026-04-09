# Feature 02: DD Path Linking (Existing Entries)

**Repository:** imas-standard-names  
**Wave:** 1 (parallel with 01, 03)  
**Depends on:** None  
**Enables:** Codex generation pipeline (provides baseline mappings to avoid duplicates)

---

## Problem

Standard names have an `ids_paths` field (`list[str]`) but no entries populate it. The 309 existing catalog entries need DD path mappings to:
1. Connect standard names to the data they describe
2. Provide baseline data for codex's generation pipeline (deduplication)
3. Enable DD version migration tracking

## Scope

**In scope:** Phase 1 (data population) and Phase 4 (version tracking) only.  
**Out of scope:** Phases 2-3 (DD context providers and full-DD generation) — these move to imas-codex as part of the build pipeline pivot.

## Deliverables

### Phase 1: Populate `ids_paths` for Existing Entries

**Owner:** Agent (data population)  
**Wave:** 1 (no code dependencies — starts immediately)  
**Dependencies:** None  
**Estimated effort:** 2–3 days

Populate `ids_paths` for the 309 existing catalog entries using cluster-based discovery via direct imports from `imas-codex` backing functions.

#### Workflow

1. For each existing standard name, call `search_dd_clusters()` backing function to find matching global clusters
2. Extract cluster member paths as `ids_paths` candidates
3. Supplement with `find_related_dd_paths()` for paths outside clusters
4. Validate all candidate paths with `check_dd_paths()` against DD v4.1.0
5. Apply path classification rules (exact equivalence only — exclude aggregations, fits, errors)
6. Persist via `edit_standard_names` + `write_standard_names`

#### Path Classification Rules

| Pattern | Action | Example |
|---------|--------|---------|
| Same leaf name, different IDS | Include | `core_profiles/.../electrons/temperature` ✓ |
| Aggregated/reduced variant | Exclude (separate name) | `summary/volume_average/t_e` ✗ |
| Fit/reconstruction metadata | Exclude | `*_fit`, `*_fit/measured` ✗ |
| Error/uncertainty companions | Exclude | `*_error_upper`, `*_error_lower` ✗ |
| Grid/coordinate paths | Evaluate case-by-case | `*/grid/psi` → may need separate name |

#### Acceptance Criteria

- [ ] All 309 existing entries have `ids_paths` populated (where applicable — some names may have no DD equivalent)
- [ ] All paths validated against DD v4.1.0 via `check_dd_paths`
- [ ] `fetch_standard_names` MCP tool returns `ids_paths` in output (fix if missing)
- [ ] Public documentation renderers continue to exclude `ids_paths`
- [ ] Test coverage for path validation and classification logic

---

### Phase 2-3: Superseded — Moved to imas-codex

DD context providers (Phase 2) and full-DD generation (Phase 3) are now part of the imas-codex SN Build Pipeline. See `plans/research/09-codex-pivot-analysis.md` for details.

---

### Phase 4: DD Version Tracking and Migration

**Owner:** Agent (infrastructure)  
**Wave:** 1+ (parallel, no pipeline dependency)  
**Dependencies:** None (Phase 1 data needed for meaningful testing)  
**Estimated effort:** 3–4 days

Automated DD version tracking and path migration when the DD is updated.

#### Components

##### Mapping Manifest

Repository-level file tracking DD version and coverage:

```yaml
# catalog-root/dd_mapping_manifest.yml
dd_version: "4.1.0"
validated_at: "2025-07-18"
validation_tool: "check_dd_paths"
entry_count: 309
mapped_count: 0  # entries with non-empty ids_paths
coverage_stats:
  total_dd_leaf_paths: 14700
  mapped_paths: 0
  unmapped_physics_paths: 9500
```

##### Path Revalidation Pipeline

```
1. Detect DD version change (compare manifest vs current)
2. Batch-validate all ids_paths via check_dd_paths() backing function
3. Identify broken paths (renamed, removed)
4. For renamed paths:
   a. Follow rename chains via get_dd_version_context(follow_rename_chains=True)
   b. Auto-update ids_paths where rename chain provides clear target
5. For removed paths:
   a. Flag for human review
   b. Search for replacement via find_related_dd_paths()
6. For new paths (added in new version):
   a. Discover via export_imas_domain() diff
   b. Assign to existing standard names via cluster membership
   c. Flag unmatched as candidates for new standard names
7. Update manifest with new version and coverage stats
```

##### Migration CLI Command

```bash
# Validate current mappings against latest DD
uv run standard-names dd-validate

# Migrate paths to new DD version
uv run standard-names dd-migrate --from 4.0.0 --to 4.1.0

# Coverage report
uv run standard-names dd-coverage
```

#### Acceptance Criteria

- [ ] Mapping manifest created and maintained in catalog root
- [ ] Revalidation pipeline detects broken paths on DD version change
- [ ] Rename chain following auto-updates >90% of renamed paths
- [ ] CLI commands for validation, migration, and coverage reporting
- [ ] Migration produces a diff report for human review
- [ ] Test coverage for rename chain following and path revalidation
- [ ] Integration test: simulate v3→v4 migration on a subset of paths

---

## Privacy and Visibility

`ids_paths` data follows the established privacy model:

| Layer | `ids_paths` Visible? |
|-------|---------------------|
| YAML files (source of truth) | Yes |
| MCP tools (`fetch_standard_names`) | Yes |
| Documentation site (mkdocs) | No |
| Graph database (generated edges) | Yes |

See [DD Linking Design](../research/08-dd-linking-design.md) for detailed privacy architecture.

---

## Dependencies on Other Features

```
Feature 02: DD Path Linking
├── Phase 1: No dependencies (data population via DD MCP tools)
└── Phase 4: No dependencies (parallel infrastructure)

Phases 2-3 → moved to imas-codex SN Build Pipeline
```

## Definition of Done

This feature is complete when:

1. All existing 309 entries have `ids_paths` populated where applicable
2. DD version migration is automated with >90% auto-update rate for renames
3. Coverage reporting shows current mapping state
4. All code has 100% test coverage
5. Documentation updated for DD integration workflow
