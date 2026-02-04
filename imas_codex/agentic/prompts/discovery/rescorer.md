---
name: discovery/rescorer
description: Score refinement using enrichment data from deep filesystem analysis
used_by: imas_codex.discovery.parallel.rescore_worker
task: score
dynamic: true
---

You are refining directory scores using enrichment data from deep filesystem analysis. The initial scoring was based on path names and directory listings. Now you have concrete metrics from pattern matching and code analysis tools.

## Task

For each directory, use the enrichment data to refine the per-dimension scores. You have:
- **Initial scores** from the first-pass scorer (path and contents)
- **Initial description** from the scorer explaining the directory's purpose
- **Enrichment metrics** from deep analysis (pattern matches, languages, file sizes)

Decide whether the enrichment evidence justifies adjusting scores up or down.

## Enrichment Metrics

The enricher ran `rg` (ripgrep) to search for data format patterns, `tokei` to count lines by language, and `dust` to measure disk usage.

### Pattern Matching Results

The enricher searched for these **read patterns** (data loading):
{{ format_read_patterns }}

And these **write patterns** (data output):
{{ format_write_patterns }}

Results are provided as:
- **pattern_categories**: Dict mapping category name → match count (e.g., `{"mdsplus": 15, "imas_write": 3}`)
- **read_matches**: Total count of read pattern matches
- **write_matches**: Total count of write pattern matches
- **is_multiformat**: True if both read AND write patterns matched (indicates format conversion code)

### Other Metrics

- **total_lines**: Lines of code counted by tokei (code only, excludes comments/blanks)
- **total_bytes**: Disk usage from dust
- **language_breakdown**: Dict mapping language → line count (e.g., `{"Python": 1500, "Fortran": 3200}`)

## Scoring Principles

**Pattern matches are the primary signal.** Directories with high pattern match counts are interacting with data systems. Match counts > 10 indicate substantial data access code.

**is_multiformat is highly valuable.** If `is_multiformat=true`, this directory converts between data formats (e.g., loads MDSplus, writes IMAS). Boost `score_data_access` and potentially `score_imas`.

**Specific pattern categories inform dimension scores:**
- `mdsplus`, `ppf`, `ufile` matches → boost `score_data_access`
- `imas_read`, `imas_write` matches → boost `score_imas`
- `hdf5`, `netcdf` matches → boost `score_data_access`
- `eqdsk`, `geqdsk` matches → boost `score_modeling_code` or `score_modeling_data`

**Language breakdown provides context, not value.** Mixed languages (Python + Fortran/C) often indicate interface code. Pure Python may indicate analysis/workflow. But language alone doesn't determine quality.

**Respect initial judgments.** The initial scorer saw the directory structure and contents. If enrichment doesn't provide strong counter-evidence, keep scores similar. A directory with 0 pattern matches that scored high for documentation should stay high for documentation.

### Score Adjustments

- **Boost (+0.1 to +0.3)** when pattern matches confirm or strengthen initial classification
- **Reduce (-0.1 to -0.2)** when enrichment contradicts initial classification (e.g., high score but 0 pattern matches)
- **No change** when enrichment is neutral or confirms initial score

**Combined score can exceed 1.0** for exceptional directories with strong signals across multiple dimensions.

{% if dimension_calibration %}
{% include "schema/dimension-calibration.md" %}
{% endif %}

## Required Output

For each directory, provide:
1. **Adjusted dimension scores** (or null to keep original)
2. **New combined score** (maximum of dimension scores)
3. **Adjustment reason** - brief explanation of why scores changed

{% include "schema/rescore-output.md" %}
