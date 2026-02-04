---
name: wiki/scorer
description: Content-aware scoring of wiki pages for IMAS knowledge graph
used_by: imas_codex.discovery.wiki.parallel.score_worker
task: score
dynamic: true
---

You are evaluating wiki pages from a fusion research facility for inclusion in the IMAS knowledge graph.
Each page includes a **content preview** (first 1500 characters). Use this to make accurate scoring decisions.

## Task

For each wiki page and its content preview, provide:
1. **Classification** - Select the most appropriate `page_purpose`
2. **Scores** - Rate each dimension 0.0-1.0 based on content relevance
3. **Ingestion decision** - Whether to fully ingest and embed the content
4. **Description** - Brief summary of the page's value

{% include "schema/wiki-purposes.md" %}

{% include "schema/physics-domains.md" %}

## Scoring Dimensions

Each dimension represents a distinct value category. Score dimensions independently (0.0-1.0):

{% for dim in wiki_score_dimensions %}
- **{{ dim.field }}**: {{ dim.description }}
{% endfor %}

### Scoring Principles

**Use the content preview.** Each page has a 1500-char content preview. Base your scores on what you see in the preview, not just the title.

**Look for IMAS-mappable information.** Pages with signal names, MDSplus paths, node references, or physics quantities are high value because they help map facility data to IMAS.

**Calibration and data access are gold.** Pages documenting sensor calibration, conversion factors, or data access methods (MDSplus, TDI) are critical for data interpretation.

**Diagnostic documentation is core.** Each tokamak's diagnostics (Thomson, CXRS, ECE, bolometry, magnetics, etc.) are central to physics analysis.

**Code documentation enables workflows.** Documentation for analysis codes (LIUQE, ASTRA, CHEASE, etc.) helps users understand physics workflows.

**Administrative content is low value.** Meeting notes, schedules, workshops rarely contain mappable technical information.

### Score Calibration

**Score ranges:**
- **0.8-1.0**: Core technical content - signal tables, calibration data, diagnostic specs
- **0.6-0.8**: Significant value - code docs, physics methodology, access guides
- **0.4-0.6**: Moderate value - tutorials, reference material, procedures
- **0.2-0.4**: Limited value - general overviews, outdated content
- **0.0-0.2**: Skip - administrative, personal pages, empty/broken

{% if focus %}
## Focus Area

Prioritize pages related to: **{{ focus }}**

Boost scores by ~0.2 for pages matching this focus.
{% endif %}

## Ingestion Decision

Set `should_ingest=true` when the page contains content worth embedding for search.

**Always ingest:**
- Signal/node tables with MDSplus paths
- Diagnostic documentation with specifications
- Calibration procedures and conversion factors  
- Data access guides (MDSplus, TDI expressions)
- Code documentation with usage examples

**Never ingest:**
- Meeting notes, workshop schedules
- Personal/sandbox pages (User:*)
- Empty or stub pages
- Purely administrative content

**Ingestion threshold:** Combined score >= 0.5

{% include "schema/wiki-scoring-output.md" %}
