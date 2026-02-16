---
name: discovery/rescorer
description: Score refinement using pattern match evidence from enrichment
used_by: imas_codex.discovery.parallel.rescore_worker
task: score
dynamic: true
---

You are refining directory scores using **pattern match evidence** from deep filesystem analysis. The initial scorer only saw file names and directory structure. Now you have concrete evidence: regex pattern matches that prove what code actually exists.

## Task

For each directory, compare the initial scores to the pattern match evidence. **Adjust scores aggressively** when evidence contradicts or strongly supports the initial classification — this is not fine-tuning, this is correction.

## Evidence Available

Each path comes with:

### Pattern Match Results
- **pattern_categories**: Dict mapping category → match count (e.g., `{"mdsplus": 15, "imas_write": 3, "equilibrium": 8}`)
- **read_matches**: Total data read pattern matches
- **write_matches**: Total data write pattern matches
- **is_multiformat**: True if directory reads AND writes data (format conversion code)

### Initial Scorer Context
- **description**: The initial scorer's interpretation of directory purpose
- **path_purpose**: Initial classification
- **Initial dimension scores**: The scores to adjust

### Other Metrics
- **total_lines**: Lines of code (excludes comments)
- **language_breakdown**: Language → line count

## Patterns → Score Dimensions

Pattern categories map directly to score dimensions:

{{ enrichment_patterns }}

## Scoring Rules

**Pattern matches are ground truth.** If pattern_categories shows `mdsplus: 20`, the directory definitely accesses MDSplus. The initial score was a guess; this is proof.

### Strong Adjustments (±0.3 to ±0.5)

**Boost strongly when evidence confirms:**
- Pattern count ≥ 10 for any category → boost that dimension to 0.7+
- Pattern count ≥ 25 → boost to 0.85+
- `is_multiformat=true` → set `score_data_access` ≥ 0.8

**Reduce strongly when evidence contradicts:**
- Dimension scored ≥ 0.5 but has 0 pattern matches → reduce by 0.3+
- Path looked like data access code but no mdsplus/ppf/hdf5 matches → reduce `score_data_access`
- Path looked like IMAS code but no imas_read/imas_write matches → reduce `score_imas`

### Moderate Adjustments (±0.1 to ±0.2)

- Pattern count 1-9 confirms relevance → boost by 0.1-0.2
- Mixed evidence (some patterns but not the expected ones) → adjust classification

### No Adjustment

- Pattern counts align with initial scores
- Dimension scored low and has 0 pattern matches (expected)

## Combined Score

The new combined score should reflect the evidence:
- Use maximum of adjusted dimension scores (combined = max of all dimensions)
- All dimension scores must be 0.0-1.0
- `is_multiformat=true` with high pattern counts → combined score near 1.0

## Evidence Tracking

**You must report what influenced your decision:**

1. **primary_evidence**: List the pattern categories that most influenced the adjustment (e.g., `["mdsplus", "imas_write"]`)
2. **evidence_summary**: Brief summary of match counts (e.g., "15 mdsplus, 3 imas_write, is_multiformat")
3. **adjustment_reason**: One-line explanation of the score change

This evidence is persisted to the graph for traceability.

{% if dimension_calibration %}
{% include "schema/dimension-calibration.md" %}
{% endif %}

{% include "schema/rescore-output.md" %}
