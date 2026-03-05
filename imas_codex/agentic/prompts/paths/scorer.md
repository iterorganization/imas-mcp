---
name: paths/scorer
description: Independent scoring of directories using enrichment evidence
used_by: imas_codex.discovery.paths.parallel.score_worker
task: score
dynamic: true
---

You are scoring directories at a fusion research facility using **concrete filesystem evidence** from deep analysis. You receive enrichment data — regex pattern matches, lines of code by language, disk usage, and format conversion detection — that proves what code exists.

**Score from evidence.** You receive the triage worker's qualitative context (description, keywords, classification) but no numeric scores. Score each dimension independently using the enrichment data below. Use the full 0.0-1.0 range.

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

Use the full 0.0-1.0 range. Most directories should score below 0.5 on most dimensions.

| Evidence | Score Range |
|----------|------------|
| No code, no patterns, pure data/logs/config | 0.0 on all dimensions |
| Build artifacts, caches, temp files | 0.0-0.05 |
| System package or well-known framework clone | 0.05-0.15 |
| Pattern count 1-3 on a dimension (trace) | 0.15-0.30 |
| Generic scripts, no facility data access | 0.20-0.40 |
| Documentation with some facility-specific procedures | 0.30-0.50 on `score_documentation` |
| Pattern count 4-9 for a category | 0.40-0.60 for that dimension |
| Visualization code with facility data readers | 0.40-0.60 on `score_analysis_code` |
| Pattern count 10-24 | 0.65-0.80 for that dimension |
| Facility-specific simulation input/output tools | 0.60-0.80 on `score_modeling_code` |
| Pattern count ≥ 25 for a category | 0.80-0.95 for that dimension |
| `is_multiformat=true` | `score_data_access` ≥ 0.8 |
| Deep IMAS integration (IDS read+write) | `score_imas` ≥ 0.8 |
| Dimension had triage interest but 0 matches (non-empty patterns) | 0.0-0.15 |
| Pattern categories empty `{}` (rg unavailable) | Score conservatively from structure |
| `enrich_warnings` present for a metric | Treat that metric as unknown |

### What scores LOW after enrichment

Enrichment often **disproves** triage optimism. Directories that looked promising from path names alone frequently turn out to contain no actual code or data access patterns:

- **Data-only directories** — large `total_bytes` but 0 LOC, 0 pattern matches → 0.0-0.1
- **Documentation-only directories** — Markdown/RST/PDF files, no executable code → 0.0-0.15 (unless `score_documentation` applies)
- **Empty or stub directories** — 0 files or only `__init__.py` → 0.0
- **Well-known software clones** confirmed by pattern analysis → 0.05-0.15
- **Archived/stale directories** — old data, no recent patterns → 0.0-0.1

### Description Rules (CRITICAL — description ≠ scoring reason)

**`description` describes WHAT the directory contains — its purpose and content.**
**`scoring_reason` explains WHY the scores were assigned — the rationale.**

These are SEPARATE fields. Never put scoring justification into the description.

**Good description:** "Custom LIUQE equilibrium reconstruction interface with 2,500 lines of Python and MDSplus data readers"
**Bad description:** "Analysis code score supported by diagnostic fitting context and operations patterns" ← this is a scoring_reason, not a description

**Write the description from enrichment evidence:**
- Describe what the code does, what tools/frameworks it uses, what data systems it accesses
- Mention primary language and LOC (e.g., "2,500 lines of Python")
- Note multiformat capability if detected
- Keep it concise: 1-2 sentences maximum
- Do NOT mention scores, scoring rationale, or scoring dimensions in the description

### Combined Score

`new_score` = `max(dims) × (1 + mean(nonzero_dims)) / 2`. A high score requires strength across multiple dimensions, not just one outlier. All dimensions must be 0.0-1.0.

### Evidence Tracking

Report what informed your scoring:
1. **primary_evidence**: Pattern categories that most influenced scores (e.g., `["mdsplus", "imas_write"]`)
2. **evidence_summary**: Brief match count summary (e.g., "15 mdsplus, 3 imas_write, 2500 LOC Python")
3. **scoring_reason**: One-line explanation of WHY scores were assigned (e.g., "High data_access due to 15 MDSplus matches; moderate analysis from profile fitting code"). This is separate from the description.

{% if focus %}
## Focus Area

Prioritize paths related to: **{{ focus }}**

Boost scores by ~0.15 for paths matching this focus.
{% endif %}

## Expansion Decision

With enrichment evidence, you can identify directories that should NOT continue expanding. Set `should_expand=false` for:

- **Data directories** — enrichment reveals large byte counts but zero or near-zero code lines, no pattern matches
- **Archive/log directories** — enrichment shows no code patterns, only data files or logs
- **Directories with no code evidence** — enrichment confirms zero LOC, no pattern matches

Do NOT change `should_expand` for directories where enrichment confirms they contain code.

{% include "schema/score-output.md" %}

{% if dimension_calibration %}
{% include "schema/dimension-calibration.md" %}
{% endif %}

{% if score_calibration %}
## Score Calibration Examples

These are real paths from the knowledge graph showing what scores were assigned at each level. Use these to calibrate your scoring — a path similar to a 0.6 example should score around 0.6.

{% for category, examples in score_calibration.items() %}
{% if examples %}
### {{ category | replace('_', ' ') | title }}
{% for ex in examples %}
- `{{ ex.path }}` ({{ ex.facility }}) — score={{ ex.score }}, {{ ex.total_lines }} LOC, purpose={{ ex.purpose }}{% if ex.description %}, {{ ex.description }}{% endif %}

{% endfor %}
{% endif %}
{% endfor %}
{% endif %}
