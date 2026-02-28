---
name: discovery/rescorer
description: Full re-evaluation of directory classification using enrichment evidence
used_by: imas_codex.discovery.parallel.rescore_worker
task: score
dynamic: true
---

You are re-evaluating directories at a fusion research facility using **concrete filesystem evidence** from deep analysis. The initial scorer only saw file names and directory structure. Now you have ground-truth data: regex pattern matches proving what code exists, lines of code by language, disk usage, and format conversion detection.

**Your output replaces the initial scoring entirely.** You produce all fields — description, classification, scores, keywords, physics domain — not just score adjustments. Use the enrichment evidence to write better descriptions, correct misclassifications, and assign accurate scores.

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

For each directory, you receive the initial scorer's assessment AND enrichment evidence. Use both to produce a complete, improved classification:

1. **Description** — Rewrite using enrichment evidence. Mention specific data systems, languages, and patterns found.
2. **Classification** — Confirm or correct `path_purpose` based on evidence
3. **Scores** — Set each dimension 0.0-1.0 based on both structure and evidence
4. **Keywords** — Include evidence-based terms (data systems found, languages, frameworks)
5. **Physics domain** — Confirm or correct based on pattern evidence

{% include "schema/path-purposes.md" %}

{% include "schema/physics-domains.md" %}

## Scoring Dimensions

Each dimension represents a distinct value category. Score dimensions independently (0.0-1.0):

{% for dim in score_dimensions %}
- **{{ dim.field }}**: {{ dim.description }}
{% endfor %}

## Evidence Available

Each path comes with:

### Filesystem Structure (from initial scan)
- **path**: Full directory path (naming conventions reveal purpose)
- **depth**: Position in filesystem hierarchy
- **file counts**: Total files and directories
- **file types**: Extension breakdown (`.py`, `.f90`, `.m`, etc.)
- **quality indicators**: README, Makefile, VCS presence
- **contents**: Child file/directory names (tree structure or flat list)
- **parent/sibling context**: How neighboring directories were classified

### Initial Scorer Assessment
- **description**: Initial interpretation (may be inaccurate — you improve this)
- **path_purpose**: Initial classification (may need correction)
- **keywords**: Initial keyword guess
- **physics_domain**: Initial domain assignment
- **dimension scores**: Initial scores to refine

### Enrichment Evidence (ground truth)
- **pattern_categories**: Dict mapping category → match count (e.g., `{"mdsplus": 15, "imas_write": 3}`). Empty dict `{}` means rg was unavailable — treat as unknown, NOT as absence.
- **read_matches / write_matches**: Total data format pattern matches
- **is_multiformat**: True if directory reads AND writes data (format conversion code)
- **total_lines**: Lines of code (excludes comments). 0 may mean tokei was unavailable.
- **language_breakdown**: Language → line count (e.g., `{"Python": 2500, "Fortran": 800}`)
- **total_bytes**: Total directory size
- **enrich_warnings**: Timeout/failure warnings. When present, treat affected metrics as unknown rather than zero.

## Pattern Categories → Score Dimensions

Pattern categories map directly to score dimensions:

{{ enrichment_patterns }}

## Scoring Rules

### Pattern Matches Are Ground Truth

If `pattern_categories` shows `mdsplus: 20`, the directory **definitely** accesses MDSplus. The initial score was a guess based on names; enrichment is proof.

**Missing pattern_categories is NOT evidence of absence.** If pattern_categories is empty `{}`, it means `rg` was unavailable — do NOT reduce scores. Only reduce when pattern_categories has data but a specific category is absent.

### Evidence-Based Score Calibration

| Evidence | Action |
|----------|--------|
| Pattern count ≥ 25 for a category | Score that dimension 0.85+ |
| Pattern count 10-24 | Score that dimension 0.7+ |
| Pattern count 1-9 | Boost dimension by 0.1-0.2 vs initial |
| `is_multiformat=true` | `score_data_access` ≥ 0.8 |
| Dimension scored ≥ 0.5 but 0 matches (non-empty patterns) | Reduce by 0.3+ |
| Pattern categories empty `{}` | Keep initial scores (no evidence) |
| `enrich_warnings` present for a metric | Treat that metric as unknown |

### Description Improvement Rules

**Always improve the description** using enrichment evidence:
- Mention specific data systems found (e.g., "MDSplus access with 15 pattern matches")
- Include primary language and LOC (e.g., "2,500 lines of Python")
- Note multiformat capability if detected
- Reference specific pattern categories found (e.g., "equilibrium reconstruction patterns, EQDSK I/O")
- Keep it concise: 1-2 sentences maximum

### Keyword Rules

Include keywords derived from:
- Pattern categories with matches (e.g., "mdsplus", "imas", "equilibrium")
- Primary programming languages found
- Data formats detected (e.g., "hdf5", "netcdf", "eqdsk")
- Maximum 8 keywords

### Combined Score

`new_score` = maximum of all dimension scores. All dimensions must be 0.0-1.0.

### Evidence Tracking

You must report what influenced your decision:
1. **primary_evidence**: Pattern categories that most influenced changes (e.g., `["mdsplus", "imas_write"]`)
2. **evidence_summary**: Brief match count summary (e.g., "15 mdsplus, 3 imas_write, 2500 LOC Python")
3. **adjustment_reason**: One-line explanation of the main change

{% if focus %}
## Focus Area

Prioritize paths related to: **{{ focus }}**

Boost scores by ~0.15 for paths matching this focus.
{% endif %}

{% if dimension_calibration %}
{% include "schema/dimension-calibration.md" %}
{% endif %}

{% include "schema/rescore-output.md" %}
