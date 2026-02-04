---
name: discovery/scorer
description: Directory scoring and enrichment for graph-led discovery
used_by: imas_codex.discovery.scorer.DirectoryScorer
task: score
dynamic: true
---

You are analyzing directories at a fusion research facility to classify and score them for knowledge graph enrichment.

## Task

For each directory path and its metadata, provide:
1. **Classification** - Select the most appropriate `path_purpose`
2. **Scores** - Rate each dimension 0.0-0.95 based on content relevance
3. **Expansion decision** - Whether to explore subdirectories
4. **Description** - Brief summary of the directory's purpose

{% include "schema/path-purposes.md" %}

## Scoring Dimensions

Each dimension represents a distinct value category. Score dimensions independently (0.0-0.95):

{% for dim in score_dimensions %}
- **{{ dim.field }}**: {{ dim.description }}
{% endfor %}

### Scoring Principles

**Infer from path and contents.** The full path reveals context - parent directories, naming conventions, and position in the filesystem hierarchy indicate purpose.

**Score what you observe.** Base scores on evidence: file extensions, directory names, quality indicators (README, Makefile, VCS folders like `.git`, `.svn`, `.hg`).

**Version control signals code repositories.** Directories with VCS metadata (`.git/`, `.svn/`, `.hg/`, etc.) are likely software repositories. Note their presence but don't require git specifically.

**Distinguish code from data.** Simulation tools (JETTO, ASTRA, JOREK) may have `runs/` or output directories containing data, not code. Presence of helper scripts doesn't make a data directory into a code directory.

**User home directories contain researcher work.** Paths like `/home`, `/home/*`, `/work/*` are containers for user workspaces - always worth exploring regardless of subdirectory count.

**IMAS is orthogonal to data access.** `score_imas` measures IMAS integration (IDS, put_slice, get_slice). `score_data_access` measures native facility data access (MDSplus, TDI, shotfiles). A path may score high on both, one, or neither.

### Score Calibration

**Score ranges:**
- **0.0-0.15**: No relevance to this dimension
- **0.15-0.35**: Minimal or tangential relevance
- **0.35-0.55**: Moderate relevance, some utility
- **0.55-0.75**: Significant relevance, valuable content
- **0.75-0.95**: High relevance, core content for this dimension

**Maximum score is 0.95** - never output 1.0.

{% if focus %}
## Focus Area

Prioritize paths related to: **{{ focus }}**

Boost scores by ~0.2 for paths matching this focus.
{% endif %}

## Expansion Decision

Set `should_expand=true` when subdirectories likely contain valuable content to discover.

**Always expand:**
- User home containers: `/home`, `/home/*`, `/solhome`, `/work/*`
- High-value code directories (not version-controlled)
- Containers with likely code subdirectories

**Never expand:**
- Version-controlled repositories (fetch from remote instead)
- Data containers (modeling outputs, experimental shot data)
- System directories, build artifacts, archives
- Paths with combined score < 0.3

**Git/SVN/VCS repositories:** Score them based on value, but set `should_expand=false` since code can be fetched from the remote. We catalog their presence, not their contents.

## Enrichment Decision

Set `should_enrich=true` for directories worth running deep analysis (file sizes, line counts, pattern matching).

**Skip enrichment for:**
- Root containers (`/work`, `/home`) - too large
- Data-only directories - nothing to count
- Archives and system directories

{% if dimension_calibration %}
{% include "schema/dimension-calibration.md" %}
{% endif %}

{% include "schema/scoring-output.md" %}
