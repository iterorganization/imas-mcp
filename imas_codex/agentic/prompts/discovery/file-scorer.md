---
name: discovery/file-scorer
description: Multi-dimensional file scoring with per-file enrichment evidence (pass 2)
used_by: imas_codex.discovery.code.scorer
task: score
dynamic: true
---

You are scoring source files from a fusion research facility to assess their relevance for knowledge graph ingestion. These files have already passed triage (pass 1) and have per-file enrichment evidence from rg pattern matching.

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

### Pattern Categories (what was searched via rg)

{{ enrichment_patterns }}

### Scoring Philosophy

**Per-file pattern evidence is ground truth.** Each file now has its OWN pattern match counts from rg. A file with `mdsplus: 5` means 5 lines in THAT file matched the MDSplus pattern. Use this as strong evidence.

**Parent directory context provides calibration.** The parent's scores and purpose tell you what kind of code lives in this directory. Files inherit context from their parent.

**Score what is UNIQUE to this facility.** Researcher-written scripts, facility-specific wrappers, custom analysis pipelines, and local tools that encode domain expertise.

**Score LOW what is available elsewhere.** Files from well-known packages, standard library modules, generated code.

**Use the full score range:**
- **0.0**: No relevance to this dimension
- **0.1-0.3**: Minimal — generic utility, support file
- **0.3-0.5**: Moderate — some relevant content
- **0.5-0.7**: Significant — facility-specific code for this dimension
- **0.7-0.85**: High — core implementation for this dimension
- **0.85-1.0**: Exceptional — primary file for this exact dimension

**Most files should score below 0.5 on most dimensions.** A file scoring 0.7+ on ANY dimension is a strong signal.

{% if focus %}
## Focus Area

Prioritize files related to: **{{ focus }}**

Boost scores by ~0.15 for files matching this focus.
{% endif %}

## Task

For each file, provide:
1. **Scores** — Rate each dimension 0.0-1.0 based on the file's content evidence
2. **Category** — code, document, notebook, config, data, or other
3. **Description** — Brief summary of what the file likely contains
4. **Skip** — Whether to exclude this file entirely (false positives from triage)

{% include "schema/file-scoring-output.md" %}
