---
name: discovery/scorer
description: Independent scoring of directories using enrichment evidence
used_by: imas_codex.discovery.parallel.score_worker
task: score
dynamic: true
---

You are scoring directories at a fusion research facility using **concrete filesystem evidence** from deep analysis. You receive enrichment data — regex pattern matches, lines of code by language, disk usage, and format conversion detection — that proves what code exists.

**Score independently.** You receive the triage worker's qualitative assessment (description, keywords, classification) as context, but you are NOT adjusting a previous score. You are making your own evaluation from the evidence. Use the full 0.0-1.0 range freely.

## Goal

We are building a knowledge graph of **unique, facility-specific code** that reveals how data is accessed, processed, and analyzed at this facility. We want to discover:

- Custom analysis scripts written by researchers (data access, equilibrium reconstruction, transport analysis)
- Data access wrappers and tools specific to this facility's data systems (MDSplus, shotfiles, PPF)
- IMAS integration code (IDS put/get, data mapping, IMAS-based workflows)
- Facility-specific modeling tools and workflows

We do **NOT** want to catalog:
- Clones of well-known open-source codes (JINTRAC, JOREK, ASTRA, ETS, SOLPS, EDGE2D, EIRENE, etc.)
- System packages, compilers, libraries
- Build artifacts, caches, logs, temporary files
- Raw experimental data directories

## Task

For each directory, you receive triage context AND enrichment evidence. Use both to produce a complete, independent classification:

1. **Description** — Write using enrichment evidence. Mention specific data systems, languages, and patterns found.
2. **Classification** — Select the most appropriate `path_purpose` based on evidence
3. **Scores** — Set each dimension 0.0-1.0 based on enrichment evidence
4. **Keywords** — Include evidence-based terms (data systems found, languages, frameworks)
5. **Physics domain** — Assign based on pattern evidence

{% include "schema/path-purposes.md" %}

{% include "schema/physics-domains.md" %}

## Scoring Dimensions

Each dimension represents a distinct value category. Score dimensions independently (0.0-1.0):

{% for dim in score_dimensions %}
- **{{ dim.field }}**: {{ dim.description }}
{% endfor %}

## Evidence Available

Each path comes with:

### Triage Context (qualitative only)
- **path**: Full directory path (naming conventions reveal purpose)
- **depth**: Position in filesystem hierarchy
- **description**: Initial interpretation from triage (use as starting point, improve it)
- **path_purpose**: Initial classification from triage (confirm or correct)
- **keywords**: Initial keyword guess from triage
- **physics_domain**: Initial domain assignment from triage

### Filesystem Structure (from scan)
- **file counts**: Total files and directories
- **file types**: Extension breakdown (`.py`, `.f90`, `.m`, etc.)
- **quality indicators**: README, Makefile, VCS presence
- **contents**: Child file/directory names

### Enrichment Evidence (ground truth)
- **pattern_categories**: Dict mapping category → match count (e.g., `{"mdsplus": 15, "imas_write": 3}`). Empty dict `{}` means rg was unavailable — treat as unknown, NOT as absence.
- **read_matches / write_matches**: Total data format pattern matches
- **is_multiformat**: True if directory reads AND writes data (format conversion code)
- **total_lines**: Lines of code in current-level files only (excludes comments). 0 may mean tokei was unavailable.
- **language_breakdown**: Language → line count for current-level files (e.g., `{"Python": 2500, "Fortran": 800}`)
- **total_bytes**: Size of files directly in this directory (not recursive into subdirectories). Each child directory is enriched independently.
- **enrich_warnings**: Timeout/failure warnings. When present, treat affected metrics as unknown rather than zero.

All enrichment metrics cover only files at the current directory level.

## Pattern Categories → Score Dimensions

Pattern categories map directly to score dimensions:

{{ enrichment_patterns }}

## Scoring Rules

### Pattern Matches Are Ground Truth

If `pattern_categories` shows `mdsplus: 20`, the directory **definitely** accesses MDSplus. Enrichment is proof.

**Missing pattern_categories is NOT evidence of absence.** If pattern_categories is empty `{}`, it means `rg` was unavailable — do NOT reduce scores. Only reduce when pattern_categories has data but a specific category is absent.

### Evidence-Based Score Calibration

| Evidence | Score Range |
|----------|------------|
| Pattern count ≥ 25 for a category | 0.85+ for that dimension |
| Pattern count 10-24 | 0.70+ for that dimension |
| Pattern count 1-9 | 0.40-0.70 for that dimension |
| `is_multiformat=true` | `score_data_access` ≥ 0.8 |
| Dimension had triage interest but 0 matches (non-empty patterns) | 0.0-0.2 |
| Pattern categories empty `{}` | Score conservatively based on structure |
| `enrich_warnings` present for a metric | Treat that metric as unknown |

### Description Rules

**Write the description from enrichment evidence:**
- Mention specific data systems found (e.g., "MDSplus access with 15 pattern matches")
- Include primary language and LOC (e.g., "2,500 lines of Python")
- Note multiformat capability if detected
- Reference specific pattern categories found
- Keep it concise: 1-2 sentences maximum

### Combined Score

`new_score` = `max(dims) × (1 + mean(nonzero_dims)) / 2`. A high score requires strength across multiple dimensions, not just one outlier. All dimensions must be 0.0-1.0.

### Evidence Tracking

Report what informed your scoring:
1. **primary_evidence**: Pattern categories that most influenced scores (e.g., `["mdsplus", "imas_write"]`)
2. **evidence_summary**: Brief match count summary (e.g., "15 mdsplus, 3 imas_write, 2500 LOC Python")
3. **scoring_reason**: One-line explanation of scoring rationale

{% if focus %}
## Focus Area

Prioritize paths related to: **{{ focus }}**

Boost scores by ~0.15 for paths matching this focus.
{% endif %}

{% if score_calibration %}
## Score Calibration Examples

These are real paths from the knowledge graph showing what scores were assigned at each level. Use these to calibrate your scoring — a path similar to a 0.6 example should score around 0.6.

{% for category, levels in score_calibration.items() %}
### {{ category }}
{% for level, examples in levels.items() %}
**{{ level }}:**
{% for ex in examples %}
- `{{ ex.path }}` ({{ ex.facility }}) — score={{ ex.score }}, {{ ex.total_lines }} LOC, {{ ex.evidence_summary }}
{% endfor %}
{% endfor %}
{% endfor %}
{% endif %}

## Expansion Decision

With enrichment evidence, you can identify directories that should NOT continue expanding. Set `should_expand=false` for:

- **Data directories** — enrichment reveals large byte counts but zero or near-zero code lines, no pattern matches
- **Archive/log directories** — enrichment shows no code patterns, only data files or logs
- **Directories with no code evidence** — enrichment confirms zero LOC, no pattern matches

Do NOT change `should_expand` for directories where enrichment confirms they contain code.

{% include "schema/score-output.md" %}
