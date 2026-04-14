# 13: Publish Pipeline & Catalog Export

**Status:** Ready to implement
**Depends on:** 11 (rich compose — all phases)
**Enables:** Human review workflow via catalog PRs → Plan 12 feedback import
**Agent:** engineer (well-defined: extend existing publish.py)

## Problem

The existing `sn publish` command generates YAML files but:
1. Exports only bare fields (name, kind, unit, tags, status, description)
2. Drops grammar fields, documentation, links, ids_paths, constraints
3. `kind` is hardcoded to "physical" instead of using catalog-standard values
4. Tags are only derived from IDS name, not from the full tag vocabulary
5. No round-trip guarantee — publish → import loses data

## Design

`sn publish` exports graph StandardName nodes (with `review_status: drafted`)
to YAML files matching the `imas-standard-names-catalog` directory structure.
The export must be **lossless** — `import-catalog` on the exported files must
reconstruct the same graph state (minus graph-only fields).

## Phase 1: Fix lossy export

**Files:** `imas_codex/standard_names/publish.py`

Rewrite `generate_yaml_entry()` to include all catalog fields:

```python
def generate_yaml_entry(sn: dict) -> dict:
    """Convert a graph StandardName to catalog YAML format."""
    entry = {
        "name": sn["name"],
        "description": sn["description"],
        "status": "draft",
        "kind": sn.get("kind", "scalar"),
        "unit": sn.get("unit"),
        "tags": sn.get("tags", []),
    }
    if sn.get("documentation"):
        entry["documentation"] = sn["documentation"]
    if sn.get("links"):
        entry["links"] = [{"name": link} for link in sn["links"]]
    if sn.get("ids_paths"):
        entry["ids_paths"] = sn["ids_paths"]
    if sn.get("constraints"):
        entry["constraints"] = sn["constraints"]
    if sn.get("validity_domain"):
        entry["validity_domain"] = sn["validity_domain"]
    if sn.get("model") or sn.get("source"):
        entry["provenance"] = {
            "mode": "generated",
            "tool": "imas-codex",
            "model": sn.get("model"),
            "source": sn.get("source"),
            "confidence": sn.get("confidence"),
        }
    return entry
```

### Output directory structure

Match catalog layout — group by primary tag:

```
output/
  core-physics/
    electron_temperature.yml
    ion_temperature.yml
  equilibrium/
    safety_factor.yml
    magnetic_axis_position.yml
```

**Acceptance:**
- Exported YAML contains all catalog fields
- Round-trip: publish → import-catalog → publish = same output
- `kind` reflects actual value, not hardcoded "physical"

## Phase 2: Update graph query

**Files:** `imas_codex/standard_names/graph_ops.py`

Update `get_validated_standard_names()` to return all rich fields:

```cypher
MATCH (sn:StandardName)
WHERE sn.review_status = 'drafted'
AND coalesce(sn.confidence, 1.0) >= $confidence_min
OPTIONAL MATCH (src)-[:HAS_STANDARD_NAME]->(sn)
OPTIONAL MATCH (src)-[:IN_IDS]->(ids:IDS)
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
RETURN sn.id AS name,
       sn.description AS description,
       sn.documentation AS documentation,
       sn.kind AS kind,
       u.id AS unit,
       sn.tags AS tags,
       sn.links AS links,
       sn.ids_paths AS ids_paths,
       sn.constraints AS constraints,
       sn.validity_domain AS validity_domain,
       sn.confidence AS confidence,
       sn.model AS model,
       sn.source AS source,
       sn.physical_base AS physical_base,
       sn.subject AS subject,
       sn.component AS component,
       sn.coordinate AS coordinate,
       sn.position AS position,
       sn.process AS process,
       collect(DISTINCT ids.id) AS source_ids_names
ORDER BY sn.id
```

**Acceptance:** Query returns all rich fields for export.

## Phase 3: Batched PR creation

**Files:** `imas_codex/cli/sn.py` (sn_publish), `imas_codex/standard_names/publish.py`

Enhance `--create-pr` workflow:
1. Group entries by primary tag (directory)
2. Create one PR per tag group (manageable review units)
3. PR title: `feat(sn): add {count} standard names for {tag}`
4. PR body: summary table of names with descriptions and confidence
5. Update graph: set `review_status = 'published'` for exported names

**Acceptance:**
- `sn publish --create-pr --catalog-repo org/repo` creates PRs
- Each PR contains YAML files for one tag group
- Graph nodes updated to `review_status: published`

## Phase 4: Duplicate detection

**Files:** `imas_codex/standard_names/publish.py`

Before export, check for existing entries in catalog:
1. Load existing catalog entries from `--catalog-dir`
2. Compare by name — skip exact matches
3. Detect near-duplicates (same name, different fields) — warn and show diff
4. Report: new, updated, unchanged, conflicts

**Acceptance:**
- Publish with existing catalog dir skips already-present names
- Near-duplicate warnings shown in output

## Test Plan

- Unit test: `generate_yaml_entry()` round-trip with all fields
- Unit test: directory structure matches catalog layout
- Unit test: review_status transitions (drafted → published on export)
- Integration test: publish → import-catalog → publish = same output
- Unit test: duplicate detection against existing catalog
