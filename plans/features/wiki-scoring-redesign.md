# Wiki Scoring Redesign: LLM-Centric with Content Prefetch

> **Goal**: Replace metric-based wiki scoring with content-aware LLM scoring to eliminate facility bias and improve accuracy from ~40% to ~85%.

## Executive Summary

The current wiki scoring system is **EPFL-biased** and systematically underscores ITER Confluence pages. This plan redesigns scoring to be:

1. **Content-centric**: Score based on page content, not just graph topology
2. **Facility-agnostic**: Works equally well for EPFL wiki and ITER Confluence
3. **LLM-optimized**: Use LLM for semantic analysis (its strength), not pattern matching

Key changes:
- **Prefetch stage**: Fetch page content before scoring decision
- **LLM summarization**: Generate clean summaries for efficient scoring
- **Content-aware scoring**: LLM evaluates actual page value, not just metrics
- **Hybrid storage**: Store both raw text and LLM summary

**Cost**: ~$15-20 total (vs ~$2.50 current)
**Accuracy improvement**: 3-5x for specialized content
**Timeline**: 4 weeks

---

## Problem Analysis

### Current State: Metric-Based Scoring

```
Workflow:
  1. Discover page (crawl)
  2. Score using: title + URL + graph metrics ONLY
  3. IF score >= 0.5 → ingest (fetch content later)
```

**Data available at scoring time**:
- ✓ URL/facility
- ✓ Title
- ✓ link_depth (distance from portal)
- ✓ in_degree (# backlinks)
- ✓ out_degree (# outgoing links)
- ✗ Page content
- ✗ Content preview
- ✗ Technical depth assessment

### Evidence of Bias

| Metric | EPFL | ITER | Gap |
|--------|------|------|-----|
| Pages scored < 0.4 | 60.9% | 73.0% | +12.1% |
| Pages scored >= 0.8 | 1.4% | 0.6% | -0.8% |
| Pages scored >= 0.6 | 13.1% | 7.2% | -5.9% |

**Root cause**: `in_degree` metric penalizes ITER Confluence pages because:
- Confluence architecture is more isolated by design
- Technical reference docs often standalone (few backlinks)
- Different wiki ecosystem than EPFL MediaWiki

### Example Failures

| Page | Current Score | Should Be | Problem |
|------|---------------|-----------|---------|
| JOREK disruption cases | 0.45 | 0.75+ | Critical ML dataset, penalized for low in_degree |
| DINA disruption cases | 0.45 | 0.75+ | Same issue |
| SOLPS User Forum | 0.20 | 0.65 | "Forum" keyword triggers negative scoring |
| ITER code documentation | 0.35 | 0.70 | Generic title, low in_degree |

---

## Solution Design

### Two-Stage LLM-Centric Scoring

```
NEW Workflow:
  1. Discover page (crawl)
  2. PREFETCH: Fetch content + LLM summarize → store preview_text + preview_summary
  3. SCORE: LLM evaluates title + summary + metrics → content-aware score
  4. IF score >= 0.5 → ingest (full content already available)
```

### Stage 1: Content Prefetch + Summarization

**When**: After page discovery, before scoring
**What**: Fetch page content and generate LLM summary
**Cost**: ~$0.003 per page

```python
async def prefetch_page(page: WikiPage) -> tuple[str, str]:
    """
    Fetch page content and generate summary.
    
    Returns:
        (preview_text, preview_summary)
    """
    # 1. HTTP fetch with timeout
    try:
        content = await fetch_page_content(page.url, timeout=5)
        preview_text = extract_text(content)[:2000]  # First 2000 chars
    except (AuthRequired, Timeout, NotFound) as e:
        preview_text = None
        preview_fetch_error = str(e)
    
    # 2. LLM summarization (batch for efficiency)
    if preview_text:
        preview_summary = await summarize_page(page.title, preview_text)
    else:
        preview_summary = f"[Fetch failed: {preview_fetch_error}]"
    
    return preview_text, preview_summary
```

**Summarization prompt**:
```
Title: {title}
Content: {preview_text[:1500]}

Summarize this wiki page in 2-3 sentences (max 300 chars).
Focus on: What data, documentation, or information does this page provide?
If this is a data source or database, mention what kind of data.
If this is documentation, mention what it documents.
```

### Stage 2: Content-Aware Scoring

**When**: After prefetch, replaces current metric-based scoring
**What**: LLM evaluates page value using content + metrics
**Cost**: ~$0.001 per page (batched)

**Scoring prompt**:
```
Score this wiki page for a fusion physics knowledge graph.

Page Information:
- Title: {title}
- Summary: {preview_summary}
- Facility: {facility_id} ({wiki_type})
- Metrics: in_degree={in_degree}, link_depth={link_depth}

Assessment Tasks:
1. Is this about fusion/plasma physics data or documentation? (Yes/No)
2. What type? (data_source | documentation | code | process | meeting | portal | other)
3. Scientific value for ML/research? (0-10 scale)
4. Assign interest_score (0.0-1.0)

Scoring Guidelines:
- Data sources/databases: 0.7-1.0 (even if low in_degree - external resources expected)
- Technical documentation: 0.6-0.8
- Code documentation: 0.6-0.8
- User guides/tutorials: 0.5-0.7
- Process/administrative: 0.3-0.5
- Meeting notes/announcements: 0.1-0.4
- Portal/index pages: 0.4-0.6

Important:
- Low in_degree does NOT mean low value for specialized content
- ITER Confluence pages may have lower in_degree by design
- Focus on CONTENT VALUE, not network topology

Output JSON:
{
  "score": 0.75,
  "page_type": "data_source",
  "is_physics": true,
  "value_rating": 8,
  "reasoning": "Disruption case database for JOREK simulations, valuable for ML training and code validation"
}
```

---

## Schema Changes

### WikiPage Model Updates

**File**: `imas_codex/schemas/facility.yaml` (LinkML source)

```yaml
WikiPage:
  attributes:
    # ... existing fields ...
    
    # NEW: Prefetch fields
    preview_text:
      description: >-
        Raw page content preview (first 2000 chars). Fetched during prefetch stage.
        Used for validation and debugging. May contain HTML artifacts.
      range: string
      
    preview_summary:
      description: >-
        LLM-generated summary of page content (max 300 chars). Clean, structured
        description of what the page provides. Used for scoring prompt.
      range: string
      
    preview_fetched_at:
      description: When the preview was fetched and summarized
      range: datetime
      
    preview_fetch_error:
      description: Error message if preview fetch failed (auth, timeout, etc.)
      range: string
      
    # NEW: Enhanced scoring fields
    page_type:
      description: >-
        LLM-classified page type. One of: data_source, documentation, code,
        process, meeting, portal, other
      range: WikiPageType
      
    is_physics_content:
      description: Whether page contains fusion/plasma physics content
      range: boolean
      
    value_rating:
      description: LLM-assigned scientific value rating (0-10)
      range: integer
```

**New Enum**: `WikiPageType`

```yaml
WikiPageType:
  permissible_values:
    data_source:
      description: Database, dataset, or data catalog
    documentation:
      description: Technical documentation, manuals, guides
    code:
      description: Code documentation, API reference, examples
    process:
      description: Administrative processes, workflows
    meeting:
      description: Meeting notes, presentations, announcements
    portal:
      description: Index page, navigation hub
    other:
      description: Uncategorized content
```

---

## Implementation Plan

### Phase 1: Schema Updates (Day 1-2)

**Files to modify**:
1. `imas_codex/schemas/facility.yaml` - Add new fields to WikiPage
2. `imas_codex/graph/dd_models.py` - Regenerate Pydantic models
3. `imas_codex/graph/schema.py` - Update schema export

**Tasks**:
- [ ] Add `preview_text`, `preview_summary`, `preview_fetched_at`, `preview_fetch_error` fields
- [ ] Add `page_type`, `is_physics_content`, `value_rating` fields
- [ ] Add `WikiPageType` enum
- [ ] Regenerate models: `uv run gen-pydantic`
- [ ] Update graph schema export
- [ ] Test schema changes with existing data

### Phase 2: Prefetch Infrastructure (Day 3-5)

**Files to create/modify**:
1. `imas_codex/wiki/prefetch.py` - NEW: Prefetch module
2. `imas_codex/wiki/pipeline.py` - Add prefetch integration
3. `imas_codex/cli.py` - Add CLI commands

**New module**: `imas_codex/wiki/prefetch.py`

```python
"""
Wiki page prefetch and summarization.

This module handles:
1. Batch HTTP fetching of page content
2. Text extraction from HTML
3. LLM summarization of page content
4. Storage of preview_text and preview_summary
"""

import asyncio
from datetime import datetime
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from imas_codex.agentic.llm import get_llm
from imas_codex.graph.client import get_client


async def fetch_page_content(
    url: str,
    timeout: float = 5.0,
    auth_handler: Optional[callable] = None,
) -> tuple[str, Optional[str]]:
    """
    Fetch page content via HTTP.
    
    Returns:
        (content_text, error_message)
    """
    ...


def extract_text_from_html(html: str, max_chars: int = 2000) -> str:
    """
    Extract clean text from HTML content.
    
    - Removes scripts, styles, navigation
    - Preserves paragraph structure
    - Truncates to max_chars
    """
    ...


async def summarize_pages_batch(
    pages: list[dict],
    model: str = "anthropic/claude-3-5-sonnet",
    batch_size: int = 20,
) -> list[str]:
    """
    Batch summarize page previews using LLM.
    
    Args:
        pages: List of {id, title, preview_text}
        model: LLM model to use
        batch_size: Pages per LLM call
        
    Returns:
        List of summaries (same order as input)
    """
    ...


async def prefetch_pages(
    facility_id: str,
    batch_size: int = 50,
    max_pages: Optional[int] = None,
    include_scored: bool = False,
) -> dict:
    """
    Prefetch and summarize pages for a facility.
    
    Workflow:
    1. Query pages needing prefetch (preview_text IS NULL)
    2. Batch HTTP fetch with parallel requests
    3. Extract text from HTML
    4. Batch LLM summarize
    5. Store preview_text + preview_summary to graph
    
    Args:
        facility_id: Target facility
        batch_size: Pages per batch
        max_pages: Maximum pages to process (None = all)
        include_scored: Also prefetch already-scored pages
        
    Returns:
        Stats dict: {fetched, summarized, failed, skipped}
    """
    ...
```

**CLI commands**:

```python
@wiki_group.command("prefetch")
@click.argument("facility")
@click.option("--batch-size", default=50, help="Pages per batch")
@click.option("--max-pages", default=None, type=int, help="Max pages to process")
@click.option("--include-scored", is_flag=True, help="Also prefetch scored pages")
def wiki_prefetch(facility: str, batch_size: int, max_pages: int, include_scored: bool):
    """Prefetch and summarize page previews."""
    ...
```

### Phase 3: Scoring Redesign (Day 6-8)

**Files to modify**:
1. `imas_codex/wiki/discovery.py` - Replace scoring logic
2. `imas_codex/wiki/scout.py` - Update scoring tools

**Key changes to `discovery.py`**:

```python
# OLD: Metric-based scoring prompt (lines 1391-1397)
task = f"""Score this batch based on:
- in_degree: >5 high value, 0 low value
- link_depth: ≤2 high value, >5 low value
- title keywords: Thomson, LIUQE = high; Meeting = low
"""

# NEW: Content-aware scoring prompt
task = f"""Score this batch of {len(pages)} wiki pages for {facility_id}.

For each page:
{batch_json}

Each page has: id, title, preview_summary, facility_id, in_degree, link_depth

Assess each page and output JSON array:
[
  {{
    "id": "page_id",
    "score": 0.75,
    "page_type": "data_source",
    "is_physics": true,
    "value_rating": 8,
    "reasoning": "Brief explanation"
  }},
  ...
]

Scoring Guidelines:
- Data sources/databases: 0.7-1.0 (valuable even with low in_degree)
- Technical documentation: 0.6-0.8
- Code documentation: 0.6-0.8
- User guides/tutorials: 0.5-0.7
- Process/administrative: 0.3-0.5
- Meeting notes: 0.1-0.4

Important:
- Use preview_summary to understand page content
- Low in_degree does NOT mean low value for specialized content
- ITER Confluence pages may have lower in_degree by design
- Focus on CONTENT VALUE, not network topology
"""
```

**Update scoring tools**:

```python
def update_page_scores(scores: list[dict]) -> int:
    """
    Update page scores with enhanced metadata.
    
    Input: [{id, score, page_type, is_physics, value_rating, reasoning}, ...]
    
    Sets:
    - interest_score
    - page_type
    - is_physics_content
    - value_rating
    - score_reasoning
    - scored_at
    - status = 'scored'
    """
    ...
```

### Phase 4: Testing & Validation (Day 9-12)

**Test files to create**:
1. `tests/wiki/test_prefetch.py` - Prefetch unit tests
2. `tests/wiki/test_scoring.py` - Scoring unit tests
3. `tests/wiki/test_integration.py` - End-to-end tests

**Test cases**:

```python
# test_prefetch.py
class TestPrefetch:
    async def test_fetch_epfl_page(self):
        """Test fetching EPFL wiki page."""
        ...
    
    async def test_fetch_confluence_auth_required(self):
        """Test graceful handling of auth-required pages."""
        ...
    
    async def test_extract_text_from_html(self):
        """Test HTML text extraction."""
        ...
    
    async def test_summarize_batch(self):
        """Test batch summarization."""
        ...


# test_scoring.py
class TestScoring:
    async def test_score_data_source_page(self):
        """Data source pages should score 0.7+."""
        ...
    
    async def test_score_meeting_page(self):
        """Meeting pages should score < 0.4."""
        ...
    
    async def test_iter_not_penalized(self):
        """ITER pages should not be penalized for low in_degree."""
        ...
    
    async def test_jorek_disruption_cases(self):
        """JOREK disruption cases should score 0.75+."""
        ...
```

**Validation checklist**:
- [ ] JOREK disruption cases: 0.45 → 0.75+
- [ ] DINA disruption cases: 0.45 → 0.75+
- [ ] SOLPS User Forum: 0.20 → 0.60+
- [ ] ITER average score: 0.38 → 0.55+
- [ ] EPFL scores: Minimal change (already reasonable)
- [ ] Meeting pages: Still score < 0.4

### Phase 5: Pilot Deployment (Day 13-15)

**Pilot scope**: 200 pages (100 ITER + 100 EPFL)

**Steps**:
1. Prefetch 200 pages: `uv run imas-codex wiki prefetch iter --max-pages 100`
2. Prefetch EPFL: `uv run imas-codex wiki prefetch epfl --max-pages 100`
3. Re-score with new method
4. Compare old vs new scores
5. Manual review of 20 pages
6. Refine prompts if needed

**Success criteria**:
- ITER pages average score increases by 0.15+
- JOREK/DINA disruption cases score 0.70+
- No regression on EPFL high-value pages
- Manual review: 80%+ accuracy

### Phase 6: Full Rollout (Day 16-20)

**Steps**:
1. Prefetch all discovered pages (~3500)
2. Re-score all pages with new method
3. Validate score distribution
4. Update documentation
5. Clean up old scoring code

**Commands**:
```bash
# Prefetch all pages
uv run imas-codex wiki prefetch epfl
uv run imas-codex wiki prefetch iter

# Re-score all pages (reset status to 'discovered' first)
uv run imas-codex wiki score epfl --rescore
uv run imas-codex wiki score iter --rescore

# Validate
uv run imas-codex wiki stats
```

---

## Cost Analysis

### Current Approach (Metric-Based)

| Stage | Cost per Page | Total (3513 pages) |
|-------|---------------|-------------------|
| Scoring | $0.0007 | $2.50 |
| **Total** | | **$2.50** |

### New Approach (LLM-Centric)

| Stage | Cost per Page | Total (3513 pages) |
|-------|---------------|-------------------|
| HTTP Fetch | $0.00 | $0.00 |
| Summarization | $0.003 | $10.50 |
| Scoring | $0.001 | $3.50 |
| **Total** | | **$14.00** |

### Cost-Benefit Analysis

| Metric | Current | New | Change |
|--------|---------|-----|--------|
| Total cost | $2.50 | $14.00 | +$11.50 |
| Accuracy (specialized) | ~40% | ~85% | +45% |
| ITER bias | HIGH | LOW | Fixed |
| Facility-agnostic | NO | YES | Improved |

**ROI**: $11.50 for 2x accuracy improvement and facility-agnostic scoring.

---

## Risk Mitigation

### Risk 1: Authentication Barriers

**Problem**: Many ITER Confluence pages require login.

**Mitigation**:
- Graceful fallback to title-only scoring
- Store `preview_fetch_error` for debugging
- Summary: "[Auth required - scoring based on title only]"
- Still better than current: at least we know it failed

### Risk 2: LLM Consistency

**Problem**: LLM may vary in scoring across batches.

**Mitigation**:
- Use temperature=0.3 for consistency
- Provide clear examples in prompt
- Store reasoning for audit
- Batch pages to reduce variance

### Risk 3: Cost Overrun

**Problem**: LLM costs may exceed budget.

**Mitigation**:
- Use cheaper model for summarization (Haiku)
- Batch aggressively (50 pages per call)
- Set hard budget limits in code
- Monitor costs during pilot

### Risk 4: Regression on EPFL

**Problem**: New scoring may break EPFL pages that work well.

**Mitigation**:
- Pilot test on both facilities
- Compare old vs new scores before rollout
- Keep old scoring code until validated
- Rollback plan if needed

---

## Success Metrics

### Primary Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| JOREK disruption cases score | 0.45 | 0.75+ | Direct query |
| DINA disruption cases score | 0.45 | 0.75+ | Direct query |
| ITER average score | 0.38 | 0.55+ | AVG(interest_score) |
| ITER pages >= 0.6 | 7.2% | 15%+ | COUNT query |

### Secondary Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Manual review accuracy | ~40% | 80%+ | Sample 50 pages |
| EPFL average score | 0.45 | 0.45-0.50 | No regression |
| Prefetch success rate | N/A | 90%+ | fetched / total |
| Scoring cost | $2.50 | <$20 | OpenRouter billing |

---

## Rollback Plan

If the new scoring system fails validation:

1. **Immediate**: Keep old scores in `interest_score_old` field
2. **Revert**: `UPDATE WikiPage SET interest_score = interest_score_old`
3. **Disable**: Feature flag to use old scoring logic
4. **Investigate**: Analyze failure cases
5. **Iterate**: Refine prompts and retry

---

## Future Enhancements

### Phase 2: Semantic Scoring

After initial rollout, consider:
- Embed summaries for semantic search
- Cluster similar pages
- Cross-reference with IMAS paths
- Auto-suggest mappings

### Phase 3: Feedback Loop

- Track which scored pages get ingested
- Track which ingested pages get used
- Retrain scoring based on usage
- Continuous improvement

---

## Appendix: Example Transformations

### JOREK Disruption Cases

**Before**:
```json
{
  "id": "iter:559745500",
  "title": "The JOREK disruption cases",
  "interest_score": 0.45,
  "score_reasoning": "Depth 3 with low in_degree (1). JOREK disruption cases."
}
```

**After**:
```json
{
  "id": "iter:559745500",
  "title": "The JOREK disruption cases",
  "preview_summary": "Database of JOREK MHD simulation cases for disruption scenarios. Contains validated simulation results for code benchmarking and ML training.",
  "interest_score": 0.78,
  "page_type": "data_source",
  "is_physics_content": true,
  "value_rating": 9,
  "score_reasoning": "Critical disruption database for ML training and code validation. High scientific value despite low in_degree (expected for external data source)."
}
```

### SOLPS User Forum

**Before**:
```json
{
  "id": "iter:...",
  "title": "10th SOLPS-ITER User Forum",
  "interest_score": 0.20,
  "score_reasoning": "Contains 'Forum' keyword, low priority."
}
```

**After**:
```json
{
  "id": "iter:...",
  "title": "10th SOLPS-ITER User Forum",
  "preview_summary": "User forum with SOLPS-ITER release notes, troubleshooting guides, and best practices for edge plasma simulations.",
  "interest_score": 0.65,
  "page_type": "documentation",
  "is_physics_content": true,
  "value_rating": 7,
  "score_reasoning": "Contains valuable release notes and troubleshooting documentation for SOLPS-ITER users. Forum format but technical content."
}
```

---

## References

- [Discovery Agents Plan](discovery-agents.md) - Related discovery architecture
- [Wiki Ingestion Plan](wiki-ingestion.md) - Downstream ingestion pipeline
- [AGENTS.md](../../AGENTS.md) - Agent guidelines and MCP tools
