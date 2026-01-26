# Discovery Pipeline Architecture

Remote facility exploration and content discovery for fusion code indexing.

## Overview

The discovery pipeline identifies and prioritizes files of interest across remote facilities. It operates in phases: **scan** → **score** → **ingest**.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Discovery Pipeline                         │
├─────────────┬─────────────┬─────────────┬──────────────────────┤
│    Scan     │    Score    │   Ingest    │       Link           │
├─────────────┼─────────────┼─────────────┼──────────────────────┤
│ List paths  │ LLM scoring │ Parse code  │ FacilityUser→Person  │
│ User lookup │ Evidence    │ Embeddings  │ Evidence dedup       │
│ Metadata    │ Weights     │ CodeChunk   │ Cross-facility       │
└─────────────┴─────────────┴─────────────┴──────────────────────┘
```

## Data Model

See [facility.yaml](../../imas_codex/schemas/facility.yaml) for complete schema definitions.

**Core nodes:**
- `FacilityPath` — Discovered directory/file with scores
- `Evidence` — LLM-generated indicators for scoring decisions
- `FacilityUser` — User account at a specific facility
- `Person` — Cross-facility identity (Schema.org semantics)
- `SourceFile` — Ingested code file
- `CodeChunk` — Embedded code segment

**Key relationships:**
- `FacilityPath -[:HAS_EVIDENCE]-> Evidence`
- `FacilityUser -[:IS_PERSON]-> Person`
- `FacilityPath -[:OWNED_BY]-> FacilityUser`

## Scoring

LLM-based scoring assigns interest scores (0.0-1.0) across dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| `code`    | 1.0    | Source code presence |
| `data`    | 0.8    | Data files (HDF5, NetCDF) |
| `docs`    | 0.6    | Documentation/readmes |
| `imas`    | 1.2    | IMAS-related content |

**Final score formula:**
```
score = Σ(dimension_score × weight) / Σ(weights)
```

> **Note:** This uses weighted average across dimensions. Directories with high relevance in only one dimension (e.g., pure data with no code) may score lower than expected. An alternative max-based approach would preserve single-dimension excellence.

**Thresholds:**
- `CONTAINER_THRESHOLD = 0.1` — Minimum score to expand directories
- Paths below threshold are marked `skipped`

See [scorer.py](../../imas_codex/discovery/scorer.py) for implementation.

## User Enrichment

User information is extracted during the scan phase via GECOS parsing.

### GECOS Configuration

Each facility's `user_info` block defines parsing behavior:

```yaml
# Example from iter.yaml
user_info:
  name_format: last_first      # ITER uses "Last First [EXT]"
  gecos_suffix_pattern: "\\s+EXT$"  # Pattern to strip
  lookup_tools:               # Ordered fallback list
    - getent                  # POSIX standard (includes LDAP/NIS)
    - passwd                  # /etc/passwd fallback
    - id                      # Existence check only
```

**GECOS** = General Comprehensive Operating System — the 5th field in `/etc/passwd` containing the user's full name.

Facilities use different name formats:
- **ITER**: `"Last First [EXT]"` → `name_format: last_first`
- **EPFL/JET**: `"First Last"` → `name_format: first_last`

### Adding New Facilities

When adding a new facility:
1. Create `imas_codex/config/facilities/<facility>.yaml`
2. Add `user_info` block with appropriate `name_format`
3. Test with `getent passwd <username>` on the facility to determine format

See [user_enrichment.py](../../imas_codex/discovery/user_enrichment.py) for implementation.

## Cross-Facility Linking

Users are deduplicated continuously via Person nodes:

1. **ORCID** — Unique identifier if available
2. **Normalized name** — `given_name|family_name` (ASCII-folded, lowercase)
3. **Email** — If discoverable from git configs or profiles

The `Person` class uses [Schema.org](https://schema.org/Person) semantics (`givenName`, `familyName`, `name`, `sameAs` for ORCID).

## Evidence Nodes

LLM scoring produces evidence for transparency:

```yaml
Evidence:
  code_indicators: ["__init__.py present", "setup.py found"]
  data_indicators: ["*.h5 files detected"]
  imas_indicators: ["imas.imasdef import"]
```

Evidence nodes are content-addressed (SHA256 of indicators) for automatic deduplication.

## CLI Commands

```bash
# Scan and score in one run (recommended)
uv run imas-codex discover paths iter --limit 500

# Scan only (no LLM cost)
uv run imas-codex discover paths iter --scan-only

# Score only (rescore already-scanned paths)
uv run imas-codex discover paths iter --score-only

# Check discovery status
uv run imas-codex discover status iter

# Clear and restart
uv run imas-codex discover clear iter
```

### SSH Connection Errors

If discovery fails with exit code 255:
```
Scan failed for iter: Command 'scan_directories.py' returned non-zero exit status 255.
```

This indicates SSH connection failure. Verify connectivity:
```bash
ssh iter  # Should connect successfully
```

Common causes:
- VPN not connected
- SSH key not loaded (`ssh-add`)
- Host not configured in `~/.ssh/config`

## Related Documentation

- [agents/explore.md](../../agents/explore.md) — Exploration agent workflow
- [agents/ingest.md](../../agents/ingest.md) — Ingestion pipeline
- [facility.yaml](../../imas_codex/schemas/facility.yaml) — Schema definitions
