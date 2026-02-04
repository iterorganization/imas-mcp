---
name: discovery/rescorer
description: Score refinement using enrichment data from deep filesystem analysis
used_by: imas_codex.discovery.parallel.rescore_worker
task: score
dynamic: true
---

You are refining directory scores using enrichment data from deep filesystem analysis. The initial scoring was based on path names and directory listings. Now you have concrete metrics from running code analysis tools.

## Task

For each directory, use the enrichment data to refine the per-dimension scores. You have:
- **Initial scores** from the first-pass scorer (based on path and contents)
- **LLM description** of the directory from initial scoring
- **Enrichment metrics** from deep analysis (lines of code, languages, file sizes)

Decide whether the enrichment evidence justifies adjusting scores up or down.

## Enrichment Metrics Available

Each path comes with these metrics from deep analysis:

- **total_lines**: Lines of code counted by tokei (excludes comments/blanks)
- **total_bytes**: Disk usage from dust (indicates data volume)
- **language_breakdown**: Dict mapping language names to line counts
  - Example: `{"Python": 1500, "Fortran": 3200, "C": 800}`
- **is_multiformat**: True if directory contains format conversion code
  - HIGH VALUE signal - indicates data interface/mapping work

## Scoring Principles

**Adjust based on evidence.** Only change scores when enrichment provides new information that wasn't visible from the directory listing.

**Language signals purpose.** Fortran-heavy codebases often indicate physics simulation. Python-heavy codebases often indicate analysis or workflow code. Mixed language (Python + Fortran) often indicates interfaces.

**Lines of code indicate significance.** Directories with > 5,000 LOC are substantial codebases deserving higher scores. Directories with < 100 LOC that initially scored high may be overrated.

**Multiformat is a strong signal.** If `is_multiformat=true`, this indicates data conversion code - highly valuable for understanding facility data flows. Boost `score_data_access` and potentially `score_imas`.

**Large byte sizes indicate data.** If `total_bytes` is large (> 1GB) but `total_lines` is low, this is data storage, not code.

**Respect initial judgments.** The initial scorer saw the directory structure and contents. If enrichment doesn't provide strong counter-evidence, keep scores similar.

### Score Adjustments

When adjusting scores, consider:
- **Boost (+0.1 to +0.3)** when enrichment confirms or strengthens initial classification
- **Reduce (-0.1 to -0.3)** when enrichment contradicts initial classification
- **No change** when enrichment is neutral or confirms initial score

**Combined score can exceed 1.0.** If a directory is exceptional across multiple dimensions (large codebase, multiformat, physics domain), the combined score can go above 1.0. Do not artificially cap scores.

{% if dimension_calibration %}
{% include "schema/dimension-calibration.md" %}
{% endif %}

## Required Output

For each directory, provide:
1. **Adjusted dimension scores** (or null to keep original)
2. **New combined score** (maximum of dimension scores, can exceed 1.0)
3. **Adjustment reason** - brief explanation of why scores changed

{% include "schema/rescore-output.md" %}
