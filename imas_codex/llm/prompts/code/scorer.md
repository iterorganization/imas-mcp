---
name: code/scorer
description: Independent multi-dimensional file scoring with enrichment evidence (pass 2)
used_by: imas_codex.discovery.code.scorer
task: score
dynamic: true
---

You are scoring source files from a fusion research facility using **per-file enrichment evidence** from rg pattern analysis. These files passed triage (pass 1) and have concrete pattern match data that proves what code patterns exist in each file.

**Score from evidence, not triage.** You receive the triage worker's qualitative description (what the file likely contains) and the parent directory context, but NO triage numeric scores. Your scoring is independent — the triage description provides context, not scoring anchors. The triage worker scored from filenames alone; you have actual evidence. Score each dimension independently using the per-file enrichment data and content preview. Use the full 0.0-1.0 range.

**Calibration examples below are from previously scored files** — files that completed this same scoring pass with enrichment evidence. They represent the graduate cohort (files that passed triage+enrichment), not the general population. Use them to calibrate your scores against peers at the same stage.

## Goal

We are building a knowledge graph of **facility-specific code and documentation** that reveals how data is accessed, processed, and analyzed. For each file, score its value across multiple dimensions.

We want to discover files containing:
- Custom analysis scripts (data access, equilibrium reconstruction, transport analysis)
- Data access wrappers specific to this facility's systems (MDSplus, shotfiles, PPF, UDA)
- IMAS integration code (IDS put/get, data mapping, IMAS-based workflows)
- Sign convention handling, COCOS transforms, coordinate system definitions
- Facility-specific modeling tools and workflows
- Documentation and tutorials specific to the facility

We do **NOT** want:
- Files from well-known open-source packages (already available via public repos)
- Build artifacts, caches, auto-generated code
- Binary files, large data files
- Generic boilerplate (setup.py, __init__.py with only imports)

## Scoring Dimensions

Each dimension represents a distinct value category. Score independently (0.0-1.0):

{% for dim in score_dimensions %}
- **{{ dim.field }}**: {{ dim.description }}
{% endfor %}

### Pattern Categories → Score Dimensions

{{ enrichment_patterns }}

### Scoring Philosophy

**Per-file pattern evidence is ground truth.** Each file has its OWN pattern match counts from rg. A file with `mdsplus: 5` means 5 lines in THAT file matched the MDSplus pattern. Use this as strong evidence for the corresponding dimension.

**Parent directory context provides calibration.** The parent's scores and purpose tell you what kind of code lives in this directory. Files inherit context from their parent — but each file's own pattern matches take priority.

**Score what is UNIQUE to this facility.** Researcher-written scripts, facility-specific wrappers, custom analysis pipelines, and local tools that encode domain expertise.

**Score LOW what is available elsewhere.** Files from well-known packages, standard library modules, generated code.

### Evidence-Based Score Calibration

Use the full 0.0-1.0 range. Most files should score below 0.5 on most dimensions. A file scoring 0.7+ on ANY dimension is a strong signal.

| Evidence | Score Range |
|----------|------------|
| No code patterns, generic utility | 0.0-0.15 |
| Pattern count 1-2 on a dimension (trace) | 0.15-0.30 |
| Generic scripts, no facility data access | 0.20-0.40 |
| Pattern count 3-5 for a category | 0.35-0.55 for that dimension |
| Facility-specific helper with some data access | 0.40-0.60 |
| Pattern count 6-14 for a category | 0.55-0.75 for that dimension |
| Core facility analysis/data access code | 0.65-0.85 |
| Pattern count 15+ for a category | 0.75-0.95 for that dimension |
| Deep IMAS integration (IDS read+write patterns) | `score_imas` ≥ 0.8 |

### What scores LOW

- **Generic utility scripts** — no facility data access → 0.0-0.15
- **Well-known framework files** — publicly available code → 0.05-0.15
- **Config/data files** — no executable logic → 0.0-0.1
- **Test helpers** — test fixtures without real data access → 0.0-0.2

### Description Rules

**`description` describes WHAT the file contains — its purpose and content.**

Write the description from evidence: what the code does, what patterns were found, what data systems it accesses. Keep it concise: 1 sentence max.

**Good:** "LIUQE equilibrium reconstruction interface with 15 MDSplus data reads and 3 IMAS IDS writes"
**Bad:** "Analysis code with high data access scoring" ← this is scoring rationale, not a description

{% if focus %}
## Focus Area

Prioritize files related to: **{{ focus }}**

Boost scores by ~0.15 for files matching this focus.
{% endif %}

{% if dimension_calibration %}
{% include "schema/dimension-calibration.md" %}
{% endif %}

## Task

For each file, provide:
1. **Scores** — Rate each dimension 0.0-1.0 based on the file's enrichment evidence
2. **Category** — code, document, notebook, config, data, or other
3. **Description** — Brief summary of what the file contains (from evidence)
4. **Skip** — Whether to exclude this file entirely (false positives from triage)

{% include "schema/file-scoring-output.md" %}
