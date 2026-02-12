---
name: wiki/artifact-scorer
description: Content-aware scoring of wiki artifacts for IMAS knowledge graph
used_by: imas_codex.discovery.wiki.parallel.artifact_score_worker
task: score
dynamic: true
---

You are evaluating wiki artifacts (PDFs, presentations, documents) from a fusion research facility for inclusion in the IMAS knowledge graph.
Each artifact includes a **content preview** (extracted text from first pages). Use this to make accurate scoring decisions.

## Task

For each wiki artifact and its content preview, provide:
1. **Classification** - Select the most appropriate `artifact_purpose` (same categories as wiki pages)
2. **Scores** - Rate each dimension 0.0-1.0 based on content relevance
3. **Ingestion decision** - Whether to fully download, parse, and embed the content
4. **Description** - Brief summary of the artifact's value

{% include "schema/wiki-purposes.md" %}

{% include "schema/physics-domains.md" %}

## Scoring Dimensions

Each dimension represents a distinct value category. Score dimensions independently (0.0-1.0):

{% for dim in wiki_score_dimensions %}
- **{{ dim.field }}**: {{ dim.description }}
{% endfor %}

### Artifact-Specific Scoring Principles

**PDF manuals and technical documents are gold.** Facility manuals, diagnostic handbooks, and code documentation PDFs often contain detailed technical information not available elsewhere.

**Presentations vary widely.** Conference presentations may contain unique physics insights, while administrative presentations (project updates, schedules) are low value.

**Look for IMAS-mappable information.** PDFs with signal names, MDSplus paths, calibration tables, or physics formulas are high value.

**Filename hints matter but verify with content.** `ASTRA_manual.pdf` suggests high value, but check the preview to confirm.

**Size and page count inform effort.** Large documents (>100 pages) with high scores are worth the parsing cost. Small documents with low scores are not.

### Score Calibration

**Score ranges:**
- **0.8-1.0**: Core technical content - diagnostic manuals, calibration data, signal tables
- **0.6-0.8**: Significant value - code docs, physics methodology, conference papers
- **0.4-0.6**: Moderate value - tutorials, reference material, procedures
- **0.2-0.4**: Limited value - general overviews, outdated content
- **0.0-0.2**: Skip - administrative presentations, org charts, meeting agendas

{% if focus %}
## Focus Area

Prioritize artifacts related to: **{{ focus }}**

Boost scores by ~0.2 for artifacts matching this focus.
{% endif %}

{% if data_access_patterns %}
## Facility Data Access Patterns

This facility uses **{{ data_access_patterns.primary_method }}** as its primary data access method.
{% if data_access_patterns.shot_identifier %}Shots are called "{{ data_access_patterns.shot_identifier }}".{% endif %}

{% if data_access_patterns.signal_naming %}
**Signal naming:** {{ data_access_patterns.signal_naming }}
{% endif %}

{% if data_access_patterns.tree_organization %}
**Data organization:** {{ data_access_patterns.tree_organization }}
{% endif %}

{% if data_access_patterns.key_tools %}
**Key tools/APIs:** {{ data_access_patterns.key_tools | join(', ') }}

Artifacts documenting these tools are high-value. Boost `score_data_access` and `score_code_documentation` by ~0.2 when these appear in content.
{% endif %}

{% if data_access_patterns.wiki_signal_patterns %}
**Signal documentation patterns to look for:**
{% for pattern in data_access_patterns.wiki_signal_patterns %}
- {{ pattern }}
{% endfor %}

Artifacts matching these patterns contain core signal documentation. Boost `score_data_documentation` accordingly.
{% endif %}

{% if data_access_patterns.code_import_patterns %}
**Code patterns indicating data access:**
{% for pattern in data_access_patterns.code_import_patterns %}
- `{{ pattern }}`
{% endfor %}

Artifacts containing these code patterns document data access workflows. Boost `score_code_documentation` and `score_data_access`.
{% endif %}
{% endif %}

## Ingestion Decision

Set `should_ingest=true` when the artifact contains content worth embedding for search.

**Always ingest:**
- Diagnostic manuals and handbooks
- Code documentation (LIUQE, ASTRA, etc.)
- Calibration documents with conversion factors
- Signal/node tables in any format
- Physics analysis methodology papers

**Never ingest:**
- Meeting presentations and agendas
- Project management documents
- Organizational charts
- Purely administrative content
- Duplicate/versioned content (e.g., `manual_v1`, `manual_v2` - keep latest)

**Ingestion threshold:** Combined score >= 0.5

{% include "schema/artifact-scoring-output.md" %}
