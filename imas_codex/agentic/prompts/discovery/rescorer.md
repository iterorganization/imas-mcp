---
name: discovery/rescorer
description: Per-dimension score refinement using enrichment data
used_by: imas_codex.discovery.parallel.rescore_worker
task: score
dynamic: true
---

You are refining directory scores using enrichment data from deep filesystem analysis. Score each dimension INDEPENDENTLY based on the evidence available.

## Task

For each directory, analyze the enrichment metrics to make informed adjustments to EACH relevant score dimension. The enrichment data provides concrete quantitative evidence that was not available during first-pass scoring.

**Key principle**: Each dimension should be adjusted independently based on evidence relevant to that dimension. Do not apply blanket adjustments - be specific about what evidence supports each dimension's score.

## Per-Dimension Scoring Rules

Evaluate each dimension independently. Only adjust dimensions where enrichment provides new evidence.

### Code Dimensions

**score_modeling_code (0.0-1.0)**
- **Boost if**: Fortran presence (common in physics codes), high LOC in compiled languages, physics domain keywords
- **Evidence**: language_breakdown showing Fortran, C, C++; total_lines > 5000
- **Example**: `{Fortran: 38000, Python: 7000}` → boost +0.15

**score_analysis_code (0.0-1.0)**
- **Boost if**: Python-heavy with scientific libraries, diagnostic processing indicators
- **Evidence**: language_breakdown showing Python dominance; total_lines > 1000
- **Example**: `{Python: 12000}` for diagnostic processing → boost +0.10

**score_operations_code (0.0-1.0)**
- **Boost if**: Real-time indicators (C, Python with timing code)
- **Evidence**: Small tight codebase with specific patterns
- **Rarely adjusted** - enrichment data less informative for this

**score_data_access (0.0-1.0)**
- **Boost significantly if**: is_multiformat=true (reads one format, writes another)
- **Evidence**: Multiformat detection is strongest signal
- **Example**: multiformat conversion code → boost +0.20

**score_imas (0.0-1.0)**
- **Boost if**: Path contains IMAS patterns, multiformat includes IMAS formats
- **Evidence**: is_multiformat + path naming
- **Example**: IMAS format converter → boost +0.15

### Data Dimensions

**score_modeling_data, score_experimental_data (0.0-1.0)**
- **Boost if**: Large total_bytes (> 1GB indicates substantial data)
- **Note**: These are less often adjusted since enrichment focuses on code

### Support Dimensions

**score_documentation (0.0-1.0)**
- **Boost if**: Markdown or RST detected in language_breakdown
- **Evidence**: `{Markdown: 500}` → boost +0.05

**score_workflow, score_visualization (0.0-1.0)**
- **Rarely adjusted** - enrichment data less informative

## Enrichment Metrics Available

Each path comes with these metrics from deep analysis:

- **total_lines**: Lines of code counted by tokei (0 = no code, or tokei unsupported files)
- **total_bytes**: Disk usage from dust (indicates data volume)
- **language_breakdown**: Dict mapping language names to line counts
  - Example: `{"Python": 1500, "Fortran": 3200, "C": 800}`
- **is_multiformat**: True if directory contains format conversion code
  - This is HIGH VALUE - indicates data interface work
- **path_purpose**: Classification from initial scoring
- **Initial scores**: All 10 dimension scores from first pass

## Evidence-Based Adjustment Guidelines

### Lines of Code Signals

| LOC Range | Primary Signal | Adjustment |
|-----------|----------------|------------|
| > 20,000 | Major codebase | +0.15 to primary dimension |
| 5,000-20,000 | Substantial code | +0.10 to primary dimension |
| 1,000-5,000 | Medium project | +0.05 to primary dimension |
| 100-1,000 | Small utility | No change |
| < 100 | Minimal code | Consider -0.05 if overscored |

### Language Signals

| Pattern | Dimension to Boost | Adjustment |
|---------|-------------------|------------|
| Fortran dominant | score_modeling_code | +0.10 |
| Python + Fortran | Both modeling + data_access | +0.10 each |
| Python only (large) | score_analysis_code | +0.08 |
| C/C++ heavy | score_operations_code or modeling | +0.08 |
| Julia present | score_modeling_code | +0.10 |
| Markdown present | score_documentation | +0.05 |

### Multiformat Signal (CRITICAL)

`is_multiformat=true` is the strongest enrichment signal:
- **Primary boost**: score_data_access +0.15 to +0.25
- **Secondary boost**: score_imas +0.10 (if IMAS format likely)
- **Reason**: Format conversion code is exactly what IMAS integration needs

## Combined Score Calculation

After adjusting individual dimensions, compute `new_score` as the maximum across all dimensions, capped at 1.5:

```
new_score = min(1.5, max(score_modeling_code, score_analysis_code, ..., score_imas))
```

This allows exceptional directories to exceed 1.0 when enrichment provides strong evidence.

{% if enriched_examples %}
## Cross-Facility Enrichment Examples

Learn from these previously enriched directories how metrics correlate with scores:

{% if enriched_examples.high_loc %}
**High Lines of Code (5000+):**
{% for p in enriched_examples.high_loc %}
- `{{ p.path }}` [{{ p.facility }}] - {{ p.total_lines }} LOC, score={{ p.score }}, {{ p.purpose }}
{% endfor %}
{% endif %}

{% if enriched_examples.fortran_heavy %}
**Fortran-Dominant Codebases:**
{% for p in enriched_examples.fortran_heavy %}
- `{{ p.path }}` [{{ p.facility }}] - {{ p.language_breakdown }}, score={{ p.score }}
{% endfor %}
{% endif %}

{% if enriched_examples.multiformat %}
**Multi-Format Conversion Code (HIGH VALUE):**
{% for p in enriched_examples.multiformat %}
- `{{ p.path }}` [{{ p.facility }}] - multiformat=true, score={{ p.score }}, {{ p.purpose }}
{% endfor %}
{% endif %}

{% if enriched_examples.python_heavy %}
**Python-Dominant Analysis:**
{% for p in enriched_examples.python_heavy %}
- `{{ p.path }}` [{{ p.facility }}] - {{ p.language_breakdown }}, score={{ p.score }}
{% endfor %}
{% endif %}

## Score Distribution Context

These examples show how scores distribute across the graph. Use this to calibrate your adjustments and understand what score levels mean:

{% if enriched_examples.score_high %}
**High Scores (0.75+) - Exceptional value:**
{% for p in enriched_examples.score_high %}
- `{{ p.path }}` [{{ p.facility }}] - score={{ p.score }}, {{ p.total_lines }} LOC, {{ p.purpose }}
{% endfor %}
{% endif %}

{% if enriched_examples.score_medium %}
**Medium Scores (0.5-0.75) - Solid value:**
{% for p in enriched_examples.score_medium %}
- `{{ p.path }}` [{{ p.facility }}] - score={{ p.score }}, {{ p.total_lines }} LOC, {{ p.purpose }}
{% endfor %}
{% endif %}

{% if enriched_examples.score_low %}
**Lower Scores (0.25-0.5) - Limited value:**
{% for p in enriched_examples.score_low %}
- `{{ p.path }}` [{{ p.facility }}] - score={{ p.score }}, {{ p.total_lines }} LOC, {{ p.purpose }}
{% endfor %}
{% endif %}

{% if enriched_examples.small_code %}
**Small Codebases (under 500 LOC):**
{% for p in enriched_examples.small_code %}
- `{{ p.path }}` [{{ p.facility }}] - {{ p.total_lines }} LOC, score={{ p.score }}
{% endfor %}
{% endif %}
{% endif %}

{% include "schema/rescore-output.md" %}
