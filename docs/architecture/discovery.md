# Discovery Pipeline Architecture

Remote facility exploration and content discovery for fusion code indexing.

## Overview

The discovery pipeline identifies and prioritizes files of interest across remote facilities. It operates in phases: **scan** → **score** → **ingest**.

```
┌─────────────────────────────────────────────────────────┐
│                   Discovery Pipeline                     │
├───────────────────┬─────────────────┬───────────────────┤
│       Scan        │      Score      │      Ingest       │
├───────────────────┼─────────────────┼───────────────────┤
│ List paths        │ LLM scoring     │ Parse code        │
│ User lookup       │ Evidence nodes  │ Embeddings        │
│ GECOS→FacilityUser│ Expand/skip     │ CodeChunk         │
│ Person linking    │                 │                   │
└───────────────────┴─────────────────┴───────────────────┘
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

LLM-based scoring assigns interest scores (0.0-1.0) across per-purpose dimensions aligned with the DiscoveryRootCategory taxonomy:

### Code Dimensions
| Dimension | Description |
|-----------|-------------|
| `score_modeling_code` | Forward modeling/simulation code (CHEASE, ASTRA, JOREK) |
| `score_analysis_code` | Experimental analysis code (diagnostics, reconstruction) |
| `score_operations_code` | Real-time operations code (control systems, DAQ) |

### Data Dimensions
| Dimension | Description |
|-----------|-------------|
| `score_modeling_data` | Modeling outputs (HDF5 runs, parameter scans) |
| `score_experimental_data` | Experimental shot data (MDSplus, pulse files) |

### Infrastructure Dimensions
| Dimension | Description |
|-----------|-------------|
| `score_data_access` | Data access tools (IMAS wrappers, MDSplus readers) |
| `score_workflow` | Workflow and orchestration tools |
| `score_visualization` | Plotting and rendering tools |

### Support Dimensions
| Dimension | Description |
|-----------|-------------|
| `score_documentation` | Documentation, readmes, tutorials |
| `score_imas` | Cross-cutting IMAS relevance |

**Final score formula:**
```
score = max(all per-purpose scores) + quality_boosts
```

Using MAX (not weighted average) ensures paths excelling in a single dimension rank appropriately. A pure data directory with `score_modeling_data=0.9` should rank high, even if all code scores are 0.

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
