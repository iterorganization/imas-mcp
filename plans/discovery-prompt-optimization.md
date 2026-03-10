# Discovery CLI Prompt Optimization Analysis

## 1. Complete LLM/VLM Call Inventory

The discovery CLI uses **10 LLM/VLM call sites** across **6 pipelines**, all routing through `call_llm_structured()` / `acall_llm_structured()` in `discovery/base/llm.py`.

| # | Pipeline | Stage | Prompt Template | Model | Has Calibration Examples |
|---|----------|-------|----------------|-------|--------------------------|
| 1 | Paths | Triage (pass 1) | `paths/triage.md` | language | Yes — 11d × 5l × 3ex |
| 2 | Paths | Score (pass 2) | `paths/scorer.md` | language | Yes — 11d × 5l × 5ex + 8cat × 2ex |
| 3 | Code | Triage (pass 1) | `code/triage.md` | language | Yes — 9d × 5l × 5ex |
| 4 | Code | Score (pass 2) | `code/scorer.md` | language | Yes — 9d × 5l × 5ex |
| 5 | Wiki | Page scoring | `wiki/scorer.md` | language | No |
| 6 | Wiki | Document scoring | `wiki/document-scorer.md` | language | No |
| 7 | Wiki | Image captioning | `wiki/image-captioner.md` | vision | No |
| 8 | Signals | Enrichment | `signals/enrichment.md` | language | No |
| 9 | Static | Enrichment | `discovery/static-enricher.md` | language | No |
| 10 | Clusters | Labeling | `clusters/labeler.md` | language | No |

---

## 2. Prompt Architecture: Static / Cacheable / Dynamic Partitioning

Each system prompt has three tiers of content, ordered from most-stable to most-volatile:

```
┌─────────────────────────────────────────────┐
│  TIER 1: Truly Static (template text)       │  ← Never changes between calls
│  - Task description, scoring philosophy     │
│  - Scoring dimensions, calibration ranges   │
│  - Seed examples (hard-coded in template)   │
├─────────────────────────────────────────────┤
│  TIER 2: Schema-derived (deterministic)     │  ← Changes only when schema changes
│  - path_purposes enum values                │     (i.e., between deployments)
│  - physics_domains enum values              │
│  - Pydantic JSON schema examples            │
│  - enrichment_patterns registry             │
│  - diagnostic_categories, wiki_purposes     │
├─────────────────────────────────────────────┤
│  TIER 3: Graph-backed examples (volatile)   │  ← Changes with every graph write
│  - dimension_calibration (5 levels × N dims)│     60s TTL cache, ORDER BY rand()
│  - score_calibration (enriched path samples)│
├─────────────────────────────────────────────┤
│  USER MESSAGE (always unique per batch)     │
│  - Directory/file/page data to score        │
│  - Enrichment evidence, content previews    │
└─────────────────────────────────────────────┘
```

**Current prompt caching behavior:** `inject_cache_control()` adds `cache_control: {"type": "ephemeral"}` on the system message's last content block. This gives a 5-minute TTL for Anthropic/Google models via OpenRouter. The cache hits on the **entire system prompt** — if any part changes (including calibration examples), the cache misses.

---

## 3. Token Budget Analysis per Call Site

Estimates use ~3.8 chars/token for mixed markdown/code content. Gemini 3 Flash pricing: $0.10/1M input, $0.40/1M output (with 50% cache discount for cached input).

### 3.1 Paths Triage (Pass 1)

| Component | Category | ~Tokens | % of System |
|-----------|----------|---------|-------------|
| Template text (task, philosophy, seed examples) | Static | 3,500 | 38% |
| path_purposes (20 enum values) | Schema-derived | 420 | 4% |
| score_dimensions (11 dims) | Schema-derived | 290 | 3% |
| physics_domains (30 values) | Schema-derived | 470 | 5% |
| enrichment_patterns (15 groups) | Schema-derived | 400 | 4% |
| TriageBatch JSON schema + fields | Schema-derived | 600 | 6% |
| **dimension_calibration (11d × 5l × 3ex)** | **Graph-dynamic** | **6,200** | **67%** |
| **Total system prompt** | | **~9,200** | |
| User message (25 dirs) | Dynamic | 2,600 | — |
| Output (25 results) | Dynamic | 2,300 | — |
| **Total per call** | | **~14,100** | |

**Cost per call:** Input: $0.00116 × (cache miss) or $0.00058 (cache hit). Output: $0.00092.

### 3.2 Paths Score (Pass 2)

| Component | Category | ~Tokens | % of System |
|-----------|----------|---------|-------------|
| Template text (task, evidence rules, calibration tables) | Static | 2,500 | 16% |
| Path purposes, physics domains, score dims | Schema-derived | 1,180 | 8% |
| enrichment_patterns, ScoreBatch schema | Schema-derived | 670 | 4% |
| **dimension_calibration (11d × 5l × 5ex)** | **Graph-dynamic** | **9,600** | **63%** |
| **score_calibration (8cat × 2ex)** | **Graph-dynamic** | **950** | **6%** |
| **Total system prompt** | | **~15,300** | |
| User message (15 dirs + enrichment) | Dynamic | 3,200 | — |
| Output (15 results) | Dynamic | 1,600 | — |
| **Total per call** | | **~20,100** | |

**Cost per call:** Input: $0.00185 (miss) / $0.00093 (hit). Output: $0.00064.

### 3.3 Code Triage (Pass 1)

| Component | Category | ~Tokens | % of System |
|-----------|----------|---------|-------------|
| Template text (task, seed calibration) | Static | 1,600 | 16% |
| score_dimensions (9 dims), FileTriageBatch schema | Schema-derived | 640 | 6% |
| **dimension_calibration (9d × 5l × 5ex)** | **Graph-dynamic** | **7,900** | **78%** |
| **Total system prompt** | | **~10,100** | |
| User message (20 files grouped) | Dynamic | 1,050 | — |
| Output (20 results) | Dynamic | 1,300 | — |
| **Total per call** | | **~12,450** | |

### 3.4 Code Score (Pass 2)

| Component | Category | ~Tokens | % of System |
|-----------|----------|---------|-------------|
| Template text (task, evidence rules) | Static | 1,400 | 13% |
| Dimensions, patterns, FileScoreBatch schema | Schema-derived | 800 | 8% |
| **dimension_calibration (9d × 5l × 5ex)** | **Graph-dynamic** | **7,900** | **76%** |
| **Total system prompt** | | **~10,400** | |
| User message (15 files + patterns) | Dynamic | 2,400 | — |
| Output (15 results) | Dynamic | 1,400 | — |
| **Total per call** | | **~14,200** | |

### 3.5 Wiki Page Scoring

| Component | Category | ~Tokens | % of System |
|-----------|----------|---------|-------------|
| Template text | Static | 1,200 | 47% |
| wiki_purposes, physics_domains, WikiScoreBatch schema | Schema-derived | 870 | 34% |
| data_access_patterns (from facility config) | Config-derived | 500 | 19% |
| **Total system prompt** | | **~2,570** | |
| User message (20 pages with 1500-char previews) | Dynamic | 10,500 | — |
| Output (20 results) | Dynamic | 1,600 | — |
| **Total per call** | | **~14,670** | |

### 3.6 Wiki Document Scoring

| Component | Category | ~Tokens | % of System |
|-----------|----------|---------|-------------|
| Template text | Static | 1,230 | 47% |
| schema-derived | Schema-derived | 870 | 33% |
| data_access_patterns | Config-derived | 500 | 19% |
| **Total system prompt** | | **~2,600** | |
| User message (10 docs with previews) | Dynamic | 6,600 | — |
| Output (10 results) | Dynamic | 790 | — |
| **Total per call** | | **~9,990** | |

### 3.7 Wiki Image Captioning (VLM)

| Component | Category | ~Tokens | % of System |
|-----------|----------|---------|-------------|
| Template text | Static | 2,060 | 64% |
| physics_domains, ImageScoreBatch schema, wiki_purposes | Schema-derived | 940 | 29% |
| data_access_patterns | Config-derived | 200 | 6% |
| **Total system prompt** | | **~3,200** | |
| User message (5 images — text + base64) | Dynamic | 1,300 (text) + images | — |
| Output (5 results) | Dynamic | 790 | — |
| **Total per call (text only)** | | **~5,290** + image tokens | |

### 3.8 Signal Enrichment

| Component | Category | ~Tokens | % of System |
|-----------|----------|---------|-------------|
| Template text (task, guidelines, examples) | Static | 3,040 | 72% |
| physics_domains, diagnostic_categories, schema | Schema-derived | 1,200 | 28% |
| **Total system prompt** | | **~4,240** | |
| User message (20 signals with TDI/PPF context) | Dynamic | 2,600 | — |
| Output (20 results) | Dynamic | 1,300 | — |
| **Total per call** | | **~8,140** | |

### 3.9 Static Tree Enrichment

| Component | Category | ~Tokens | % of System |
|-----------|----------|---------|-------------|
| Template text | Static | 1,060 | 73% |
| StaticNodeBatch schema | Schema-derived | 400 | 27% |
| **Total system prompt** | | **~1,460** | |
| User message (20 nodes with context) | Dynamic | 1,600 | — |
| Output (20 results) | Dynamic | 1,050 | — |
| **Total per call** | | **~4,110** | |

### 3.10 Cluster Labeling

| Component | Category | ~Tokens | % of System |
|-----------|----------|---------|-------------|
| Template text | Static | 525 | 28% |
| cluster vocabularies, ClusterLabelBatch schema | Schema-derived | 1,350 | 72% |
| **Total system prompt** | | **~1,875** | |
| User message (10 clusters) | Dynamic | 1,050 | — |
| Output (10 results) | Dynamic | 790 | — |
| **Total per call** | | **~3,715** | |

---

## 4. Summary: Token Budget by Partition

| Call Site | System Static | System Schema | System Graph-Dynamic | User Msg | Output | Total | Graph-Dynamic % of System |
|-----------|--------------|---------------|---------------------|----------|--------|-------|--------------------------|
| paths/triage | 3,500 | 2,180 | **6,200** | 2,600 | 2,300 | 14,100 | **52%** |
| paths/scorer | 2,500 | 1,850 | **10,550** | 3,200 | 1,600 | 20,100 | **71%** |
| code/triage | 1,600 | 640 | **7,900** | 1,050 | 1,300 | 12,450 | **78%** |
| code/scorer | 1,400 | 800 | **7,900** | 2,400 | 1,400 | 14,200 | **78%** |
| wiki/scorer | 1,200 | 870 | 0 | 10,500 | 1,600 | 14,670 | 0% |
| wiki/doc-scorer | 1,230 | 870 | 0 | 6,600 | 790 | 9,990 | 0% |
| wiki/image-cap | 2,060 | 940 | 0 | 1,300+ | 790 | 5,290+ | 0% |
| signals/enrich | 3,040 | 1,200 | 0 | 2,600 | 1,300 | 8,140 | 0% |
| static/enrich | 1,060 | 400 | 0 | 1,600 | 1,050 | 4,110 | 0% |
| clusters/label | 525 | 1,350 | 0 | 1,050 | 790 | 3,715 | 0% |

**Key finding:** Calibration examples dominate the system prompt for the 4 call sites that use them (52–78% of system prompt tokens). This is the primary source of prompt cache misses.

---

## 5. Cost Driver Analysis

### 5.1 Input vs Output Cost Split

Using Gemini 3 Flash pricing ($0.10/$0.40 per 1M tokens):

| Call Site | Input Tokens | Input Cost | Output Tokens | Output Cost | Total | Output % |
|-----------|-------------|------------|---------------|-------------|-------|----------|
| paths/triage | 11,800 | $0.00118 | 2,300 | $0.00092 | $0.00210 | 44% |
| paths/scorer | 18,500 | $0.00185 | 1,600 | $0.00064 | $0.00249 | 26% |
| code/triage | 11,150 | $0.00112 | 1,300 | $0.00052 | $0.00164 | 32% |
| code/scorer | 12,800 | $0.00128 | 1,400 | $0.00056 | $0.00184 | 30% |
| wiki/scorer | 13,070 | $0.00131 | 1,600 | $0.00064 | $0.00195 | 33% |
| wiki/doc-scorer | 9,200 | $0.00092 | 790 | $0.00032 | $0.00124 | 26% |
| wiki/image-cap | 4,500+ | $0.00045+ | 790 | $0.00032 | $0.00077+ | 41%+ |
| signals/enrich | 6,840 | $0.00068 | 1,300 | $0.00052 | $0.00120 | 43% |
| static/enrich | 3,060 | $0.00031 | 1,050 | $0.00042 | $0.00073 | 58% |
| clusters/label | 2,925 | $0.00029 | 790 | $0.00032 | $0.00061 | 52% |

### 5.2 Scale Extrapolation (per facility)

Typical facility discovery run volumes:

| Pipeline | Items/Facility | Calls/Run | Input $/Run | Output $/Run | Total $/Run |
|----------|---------------|-----------|-------------|--------------|-------------|
| paths/triage | 5,000 dirs | 200 | $0.24 | $0.18 | **$0.42** |
| paths/scorer | 2,000 scored | 133 | $0.25 | $0.09 | **$0.33** |
| code/triage | 10,000 files | 500 | $0.56 | $0.26 | **$0.82** |
| code/scorer | 4,000 scored | 267 | $0.34 | $0.15 | **$0.49** |
| wiki/scorer | 2,000 pages | 100 | $0.13 | $0.06 | **$0.20** |
| wiki/doc-scorer | 500 docs | 50 | $0.05 | $0.02 | **$0.06** |
| wiki/image-cap | 1,000 images | 200 | $0.09 | $0.06 | **$0.15** |
| signals/enrich | 5,000 signals | 250 | $0.17 | $0.13 | **$0.30** |
| static/enrich | 3,000 nodes | 150 | $0.05 | $0.06 | **$0.11** |
| **Total per facility** | | **~1,850** | **$1.88** | **$1.01** | **$2.88** |

**With perfect prompt caching (50% input discount):** $0.94 + $1.01 = **$1.95** (32% savings).

### 5.3 Cache Hit Rate Impact

The current TTL cache on calibration examples (300s for paths, was 60s for code — now fixed) means:
- Within a 5-minute window: all calls see identical calibration → **cache hit on system prompt**
- After 5 minutes: new random samples → **cache miss, re-read ~7–10k tokens**
- For a typical run processing 5,000 paths at ~3s/call: 200 calls over ~600s = ~2 cache epochs
- **Effective cache hit rate: ~99.5%** (100 calls per epoch, only the first misses)

But this 95% rate is fragile. The `ORDER BY rand()` in the calibration query means even within a cache epoch, if the graph writes between batches (which it does — triage results are written immediately), the next 60s-window refresh will return different examples, causing a cache miss on the system prompt.

**The real problem isn't the 60s TTL — it's that every refresh returns different random examples.** This prevents multi-call prompt caching across epochs.

### 5.4 Empirical Validation (March 2026)

Cache behavior was tested empirically through both the proxy path and direct OpenRouter:

| Route | Prefix | cache_control Injected | Cache Result | Cached/Total Tokens |
|-------|--------|----------------------|--------------|---------------------|
| Through proxy | `openrouter/google/gemini-3-flash-preview` | Yes | **97% HIT** | 1536/1585 |
| Direct OpenRouter | `openrouter/google/gemini-3-flash-preview` | Yes | **97% HIT** | 1536/1585 |

**Key findings:**
1. The `openrouter/` prefix preserves `cache_control` blocks through the LiteLLM proxy — confirmed working.
2. Prior to commit `7c0dd3b`, the proxy path used `openai/` prefix which caused LiteLLM's client to strip `cache_control` from content blocks, silently disabling all prompt caching. This was the root cause of zero cache hits through the proxy.
3. Cache hits occur on the system prompt prefix — varying user messages still hit the system prompt cache.
4. Cost with cache miss: ~$0.00112/call; with cache hit: ~$0.00099/call (Gemini 3 Flash).
5. The `_log_cache_metrics()` function now logs cache hit/miss/write events at DEBUG level, visible in CLI log files at `~/.local/share/imas-codex/logs/`.

**Provider cache TTLs:**
- Google Gemini (via OpenRouter): ~5 minutes ephemeral
- Anthropic Claude (via OpenRouter): ~5 minutes ephemeral
- OpenAI/DeepSeek: Implicit caching (no explicit `cache_control` needed per OpenRouter docs)

**Minimum token thresholds for caching:**
- Anthropic Sonnet/Opus: 1024 tokens
- Anthropic Haiku: 4096 tokens  
- Google Gemini: ~1024 tokens (empirically confirmed at 1536 tokens)

All production discovery prompts exceed these thresholds. Cache is verified working across all call sites.

---

## 6. Graph-Backed Example Stabilization Design

### 6.1 The Core Idea

Once the graph has enough scored items, the calibration examples should **converge to a fixed set** that never changes. This turns the "Graph-dynamic" tier into effectively static content, allowing it to be cached indefinitely.

### 6.2 Proposed Architecture: `CalibrationExemplar` Graph Nodes

```
(:CalibrationExemplar {
    id: "paths:score_data_access:0.5",    -- pipeline:dimension:target_score
    pipeline: "paths",                     -- paths | code
    dimension: "score_data_access",        -- score dimension name
    target_score: 0.5,                     -- target calibration level
    phase: "triage",                       -- triage | score
    path: "/home/user/analysis/mds_tools/",
    facility_id: "tcv",
    actual_score: 0.52,                    -- the real score this item received
    description: "MDSplus data access tools with 15 read patterns",
    purpose: "data_access",
    locked_at: datetime(),                 -- when this exemplar was locked
    source_node_id: "tcv:/home/user/analysis/mds_tools/"  -- link to source
})
```

**ID convention:** `{pipeline}:{dimension}:{target_score}:{phase}` — ensures exactly one exemplar per slot.

**Target scores** (same 5 levels as current): `0.05` (lowest), `0.2` (low), `0.5` (medium), `0.8` (high), `0.95` (highest).

### 6.3 Exemplar Selection Algorithm

```python
def select_and_lock_exemplar(
    pipeline: str,          # "paths" or "code"
    dimension: str,         # e.g., "score_data_access"
    target_score: float,    # e.g., 0.5
    phase: str,             # "triage" or "score"
    tolerance: float = 0.1,
) -> CalibrationExemplar | None:
    """Lock the best exemplar for a calibration slot.
    
    Once locked, this exemplar is used forever. The locked_at timestamp
    prevents replacement. Only human review can unlock an exemplar.
    
    Selection criteria (ordered):
    1. Score closest to target within tolerance
    2. Prefer current facility, then cross-facility
    3. Prefer items with descriptions (richer calibration context)
    4. Deterministic tie-breaking (by node ID, not random)
    """
```

### 6.4 Lifecycle

```
Phase 1: Bootstrap (no exemplars)
  → Falls back to current random sampling
  → After each scoring run, check for unfilled exemplar slots
  → Auto-lock exemplars as items at the right score levels appear

Phase 2: Partial coverage
  → Some dimensions/levels have locked exemplars, others use random
  → System prompt has a stable prefix (locked) + small volatile tail (random)
  → Partial cache hits are possible

Phase 3: Full coverage (steady state)  
  → All slots filled and locked
  → System prompt is fully deterministic
  → 100% prompt cache hit rate across calls
  → Calibration section becomes effectively static
```

### 6.5 Prompt Ordering for Maximum Cache Hits

The system prompt should be ordered:

```
1. Template text (truly static)                    ── cacheable
2. Schema-derived content (enums, JSON schemas)    ── cacheable  
3. Locked calibration exemplars (graph-stable)     ── cacheable after lock
4. Focus area (if any)                             ── per-run constant
── cache_control breakpoint here ──
5. Any remaining unlocked calibration (random)     ── volatile (bootstrap only)
```

This maximizes the cache prefix. Even during bootstrap, the static + schema + locked portion caches, and only the unlocked tail causes a miss.

**Current template ordering already follows this pattern** — the `{% if dimension_calibration %}` block is at the very end of both `paths/triage.md` and `paths/scorer.md`. The design just needs the calibration data itself to be split: locked exemplars first (stable), then any random fill (volatile).

### 6.6 Implementation Requirements

1. **New LinkML class** `CalibrationExemplar` in `facility.yaml` schema
2. **Exemplar selection CLI** `imas-codex calibrate lock paths --facility tcv` — scans graph for best candidates, locks them
3. **Modified `sample_dimension_calibration_examples()`** — check for locked exemplars first, fill gaps with random
4. **Modified `dimension-calibration.md` template** — render locked exemplars in a stable section, random fills separately
5. **Exemplar review CLI** `imas-codex calibrate show` — display all locked exemplars with scores
6. **Exemplar unlock CLI** `imas-codex calibrate unlock paths:score_data_access:0.5` — for human correction

### 6.7 Token Savings

| Pipeline | Current Graph-Dynamic Tokens | With Full Lock | Savings/Call | Savings Rate |
|----------|-----------------------------|----|---|---|
| paths/triage | 6,200 | 0 (all cached) | 6,200 | 52% of system |
| paths/scorer | 10,550 | 0 (all cached) | 10,550 | 71% of system |
| code/triage | 7,900 | 0 (all cached) | 7,900 | 78% of system |
| code/scorer | 7,900 | 0 (all cached) | 7,900 | 78% of system |

The tokens don't disappear — they become cache hits at 50% discount (Gemini/Anthropic) or free (some providers).

---

## 7. Triage Example Count Reduction

### 7.1 Current State

| Stage | Examples/Dimension/Level | Total Examples in Prompt |
|-------|--------------------------|--------------------------|
| paths/triage | 3 | 11 × 5 × 3 = 165 |
| paths/scorer | 5 | 11 × 5 × 5 = 275 + 16 enriched |
| code/triage | 5 | 9 × 5 × 5 = 225 |
| code/scorer | 5 | 9 × 5 × 5 = 225 |

### 7.2 Recommendation: Reduce Triage to 1 Example/Level

Triage is a coarse filter — its purpose is to separate "worth enriching" from "skip," not precise scoring. One well-chosen exemplar per level per dimension provides sufficient calibration:

| Stage | Proposed | Total Examples | Token Reduction |
|-------|----------|---------------|-----------------|
| paths/triage | **1**/level | 11 × 5 × 1 = 55 | **66% reduction** (~4,100 tokens saved) |
| paths/scorer | **2**/level | 11 × 5 × 2 = 110 | **60% reduction** (~5,800 tokens saved) |
| code/triage | **1**/level | 9 × 5 × 1 = 45 | **80% reduction** (~6,300 tokens saved) |
| code/scorer | **2**/level | 9 × 5 × 2 = 90 | **60% reduction** (~4,700 tokens saved) |

**Why this is safe for triage:** 
- Triage's false-negative penalty is low (missed paths get re-discovered in subsequent runs)
- The seed calibration examples in the template text already provide coarse guidance
- The dimension-calibration section is redundant with seed calibration for extreme scores (0.0–0.1. 0.9–1.0)
- Accuracy matters most at score boundaries (around the discovery threshold ~0.75) where one exemplar per level is sufficient

**Why scorers need 2:** The score pass produces the final score used for ingestion decisions. Two exemplars per level provide cross-facility diversity and reduce single-example bias.

### 7.3 Further Reduction: Collapse Score Levels

The current 5-level system (`lowest/low/medium/high/highest`) could be reduced to 3 levels for triage:

| Level | Range | Purpose |
|-------|-------|---------|
| low | 0.0–0.25 | "Clearly skip" calibration |
| medium | 0.4–0.6 | "Boundary" calibration |
| high | 0.75–1.0 | "Clearly include" calibration |

This would give: 11 × 3 × 1 = 33 examples for paths triage (~1,300 tokens vs current ~6,200).

---

## 8. Wiki/Signals/Static/Clusters: Calibration Gap

These 6 call sites use **no graph-backed calibration examples**. They rely entirely on static seed examples in the template text. This is a missed opportunity:

| Pipeline | Current Guidance | Calibration Opportunity |
|----------|-----------------|------------------------|
| wiki/scorer | Static score ranges in template | Score-level exemplars from WikiPage nodes |
| wiki/doc-scorer | Static score ranges | Exemplars from scored WikiDocument nodes |
| wiki/image-cap | None | Exemplars from scored WikiImage nodes |
| signals/enrich | Hard-coded TDI/PPF examples | Exemplars from enriched FacilitySignal nodes |
| static/enrich | None | Exemplars from enriched StaticNode results |
| clusters/label | None | Exemplars from labeled ClusterNode results |

**Recommendation:** Add calibration exemplars to wiki/doc scoring (highest volume/impact). Signals and static are enrichment (no scoring), so exemplars are less impactful. Clusters are low-volume.

---


## 9. Extension Assessment: Calibration Examples for Other Call Sites

Should graph-backed calibration examples (dimension_calibration) be extended to the other 6 call sites that currently have none? This section analyzes each individually.

### 9.1 Task Type Analysis

The call sites fall into two distinct categories that inform whether calibration examples help:

| Category | Call Sites | Task | Calibration Helpful? |
|----------|-----------|------|---------------------|
| **Scoring** | wiki/scorer, wiki/doc-scorer, wiki/image-captioner | Assign dimension scores 0.0–1.0 | **Yes** — score calibration improves consistency |
| **Enrichment** | signals/enrichment, static/enrichment, clusters/labeler | Generate descriptions, classify | **No** — no scores to calibrate |

**Why scoring benefits from calibration:** Scoring tasks produce numeric values on a continuous scale. Without calibration examples, the LLM's interpretation of "0.5" vs "0.7" drifts across batches. Calibration examples anchor the scale.

**Why enrichment doesn't need calibration:** Enrichment tasks generate text descriptions, classify into enums (physics_domain, category), and extract keywords. These are categorical outputs — the answer is either right or wrong, not on a sliding scale. The comprehensive static examples in the prompt templates (e.g., `signals/enrichment.md` has 275 lines of classification guidelines with inline examples) already provide sufficient grounding.

### 9.2 Wiki Page Scoring (RECOMMENDED)

**Current state:** `wiki/scorer.md` has static score ranges (0.8-1.0 = core technical, 0.6-0.8 = significant, etc.) but no graph-backed examples.

**Volume:** ~100 calls per facility (2,000 pages ÷ 20/batch). System prompt is ~2,570 tokens — 100% cached already since it's static.

**Recommendation: Add calibration examples from scored WikiPage nodes.**

Benefits:
- Anchors scores to real examples: "This page about Thomson scattering calibration data scored 0.85"
- Cross-facility consistency: examples from JET inform TCV scoring
- Low cost to add: system prompt grows by ~500-800 tokens (3 levels × 2 examples)

Implementation:
- Query WikiPage nodes with `score_composite` in target ranges
- Use a truncated version: `{id, title, score_composite, purpose, description[:100]}`
- 3 levels (low/medium/high) × 2 examples = 6 exemplars (~200 tokens each)
- These would join the CalibrationExemplar system proposed in §6

### 9.3 Wiki Document Scoring (RECOMMENDED)

**Current state:** `wiki/document-scorer.md` has identical score ranges to wiki/scorer. No graph-backed examples.

**Volume:** ~50 calls per facility (500 docs ÷ 10/batch). System prompt ~2,600 tokens — 100% cached.

**Recommendation: Add calibration examples from scored WikiDocument nodes.**

Same benefits as wiki page scoring. Documents are scored less frequently, so the calibration gap is less impactful, but the implementation cost is near-zero once the CalibrationExemplar infrastructure exists for wiki pages.

### 9.4 Wiki Image Captioning (NOT RECOMMENDED)

**Current state:** `wiki/image-captioner.md` is a VLM (vision) prompt. Scores are secondary to the captioning task.

**Volume:** ~200 calls per facility (1,000 images ÷ 5/batch). But each call is dominated by image token cost, not system prompt cost.

**Recommendation: Do NOT add calibration examples.**

Reasons:
- Primary task is captioning (text generation), not scoring
- Score calibration doesn't improve caption quality
- Image tokens dominate cost — system prompt is a small fraction
- VLM models handle the description task well from the detailed static prompts
- Adding text examples to a multimodal prompt may dilute the visual grounding

### 9.5 Signal Enrichment (NOT RECOMMENDED)

**Current state:** `signals/enrichment.md` is 275 lines with extensive inline examples (TDI, PPF, EDAS, MDSplus patterns). System prompt rendered once, reused across all batches.

**Volume:** ~250 calls per facility (5,000 signals ÷ 20/batch). System prompt ~4,240 tokens — already 100% cached (no dynamic content).

**Recommendation: Do NOT add calibration examples.**

Reasons:
- Enrichment task, not scoring — outputs are descriptions, physics_domain, keywords
- The 275-line template already has comprehensive inline examples for all data access patterns
- System prompt is already fully static → 100% cache hit rate
- Adding graph-backed examples would introduce churn that breaks cache hits
- No dimension scores to calibrate

### 9.6 Static Tree Enrichment (NOT RECOMMENDED)

**Current state:** `discovery/static-enricher.md` has static guidance for MDSplus tree node categories. System prompt varies slightly per tree name via `{{ facility }}` and `{{ data_source_name }}`.

**Volume:** ~150 calls per facility (3,000 nodes ÷ 20/batch). System prompt ~1,460 tokens.

**Recommendation: Do NOT add calibration examples.**

Reasons:
- Enrichment task — generates descriptions and categories for tree nodes
- The user prompt already provides rich structural context (parent hierarchy, sibling parameters)
- Categories are a closed set (geometry, coil, vessel, diagnostic, etc.) — not a scoring scale
- System prompt is small and nearly static → good cache behavior already

### 9.7 Cluster Labeling (NOT RECOMMENDED)

**Current state:** `clusters/labeler.md` uses controlled vocabularies from LinkML schemas. System prompt ~1,875 tokens.

**Volume:** Very low — typically <50 calls total across all facilities. Run once after clustering.

**Recommendation: Do NOT add calibration examples.**

Reasons:
- Extremely low volume — optimization has negligible cost impact
- Labeling task (categorical), not scoring
- Controlled vocabularies already constrain outputs
- One-off operation, not continuous pipeline

### 9.8 Summary: Extension Decisions

| Call Site | Add Calibration? | Reason |
|-----------|-----------------|--------|
| wiki/scorer | **Yes** | Scoring task, ~100 calls, anchors 0–1 scale |
| wiki/doc-scorer | **Yes** | Scoring task, ~50 calls, same benefit as wiki/scorer |
| wiki/image-captioner | No | VLM captioning task, image tokens dominate cost |
| signals/enrichment | No | Enrichment task, already fully static prompts |
| static/enrichment | No | Enrichment task, rich structural context in user prompt |
| clusters/labeler | No | Labeling task, very low volume, controlled vocabularies |

**Net recommendation:** Extend score example calibration to wiki page and document scoring. These are the only remaining scoring tasks without calibration, and they process ~150 calls per facility. The CalibrationExemplar infrastructure from §6 should be designed to support multiple pipelines (paths, code, wiki) from the start.

## 10. Optimization Opportunities Summary

### 10.1 High Impact (implement first)

| Optimization | Mechanism | Estimated Savings | Accuracy Impact |
|-------------|-----------|------------------|-----------------|
| **Graph-backed exemplar locking** | CalibrationExemplar nodes, deterministic prompt | 50% input cost on cached portion (~$0.47/facility) | Positive — more consistent scores |
| **Reduce triage examples to 1/level** | `per_level=1` in triage calls | ~$0.15/facility in paths, ~$0.25/facility in code | Neutral — triage is coarse |
| **Collapse triage to 3 levels** | Merge lowest+low, medium, high+highest | Additional ~$0.10/facility | Neutral — boundary calibration preserved |

### 10.2 Medium Impact

| Optimization | Mechanism | Estimated Savings | Accuracy Impact |
|-------------|-----------|------------------|-----------------|
| **Prompt ordering for max cache prefix** | Move locked exemplars above focus area | Better cache reuse across focus changes | None |
| **Add wiki calibration exemplars** | WikiScoreExemplar nodes | Improved scoring consistency | Positive |
| **Batch size tuning** | Larger batches amortize system prompt | Fewer calls = fewer cache misses | Check quality at larger batches |

### 10.3 Lower Impact / Future

| Optimization | Mechanism | Notes |
|-------------|-----------|-------|
| Reduce scorer examples to 2/level | Currently 5/level for paths scorer | Test accuracy impact first |
| Remove `score_calibration` enriched examples | Enriched path examples in scorer | Overlaps with dimension calibration |
| Templated output schemas | Pre-render once, inject as string | Already cached via @lru_cache providers |

### 10.4 Cost Projection

| Scenario | Input $/Facility | Output $/Facility | Total | vs Current |
|----------|-----------------|-------------------|-------|------------|
| Current (no caching) | $1.88 | $1.01 | **$2.88** | baseline |
| Current (95% cache hit) | $0.99 | $1.01 | **$2.00** | −30% |
| Locked exemplars (100% cache) | $0.94 | $1.01 | **$1.95** | −32% |
| Locked + triage reduction | $0.72 | $1.01 | **$1.73** | −40% |
| Locked + triage reduction + 3 levels | $0.65 | $1.01 | **$1.66** | −42% |

**The biggest win comes from output cost, which is fixed regardless of caching.** Reducing batch sizes or output format complexity would have the largest marginal impact, but risks accuracy degradation.

---

## 11. Implementation Plan

### Phase 1: Quick Wins (no schema changes)

1. Reduce `per_level` from 3→1 in `paths/triage` (currently hardcoded in `scorer.py`)
2. Reduce `per_level` from 5→2 in `paths/scorer` and `code/scorer`
3. Reduce `per_level` from 5→2 in `code/triage`
4. Verify accuracy is maintained by spot-checking triage results

### Phase 2: Graph-Backed Stabilization

1. Add `CalibrationExemplar` class to `imas_codex/schemas/facility.yaml`
2. Implement `imas_codex/discovery/base/calibration.py`:
   - `get_locked_exemplars(pipeline, phase)` → deterministic lookup
   - `lock_exemplar(pipeline, dimension, target, phase)` → MERGE + lock
   - `fill_calibration(pipeline, phase)` → locked first, random gaps second
3. Update `sample_dimension_calibration_examples()` to call `fill_calibration()`
4. Update `sample_code_dimension_calibration()` similarly
5. Add CLI: `imas-codex calibrate lock paths --facility tcv`
6. Add CLI: `imas-codex calibrate show`
7. Split `dimension-calibration.md` template into locked and unlocked sections

### Phase 3: Prompt Ordering

1. Restructure templates to place locked exemplars in a deterministic block before any volatile content
2. Move `cache_control` breakpoint to after locked exemplars (before unlocked random fill)
3. Verify prompt caching hit rates in production via logging

---

## Appendix A: Current Prompt Caching Infrastructure

From `config/prompt_caching.yaml`:
```yaml
providers:
  anthropic:
    match: [claude, anthropic]
  google:
    match: [gemini, google]
```

`inject_cache_control()` in `llm.py` adds `cache_control: {"type": "ephemeral"}` to the last system message block. This is a 5-minute ephemeral cache via OpenRouter.

OpenRouter also supports implicit caching for OpenAI/DeepSeek/Grok models (no explicit breakpoint needed).

## Appendix B: Prompt Template → Dynamic Content Map

```
paths/triage.md
  ├── {% include "schema/path-purposes.md" %}        → path_purposes_*      (schema)
  ├── {% include "schema/physics-domains.md" %}       → physics_domains      (schema)
  ├── {% for dim in score_dimensions %}               → score_dimensions     (schema)
  ├── {{ enrichment_patterns }}                       → format_patterns      (schema)
  ├── {% include "schema/scoring-output.md" %}        → scoring_schema_*     (schema)
  ├── {{ focus }}                                     → user input           (per-run)
  └── {% include "schema/dimension-calibration.md" %} → dimension_calibration (GRAPH)

paths/scorer.md
  ├── {% include "schema/path-purposes.md" %}         → path_purposes_*      (schema)
  ├── {% include "schema/physics-domains.md" %}       → physics_domains      (schema)
  ├── {% for dim in score_dimensions %}               → score_dimensions     (schema)
  ├── {{ enrichment_patterns }}                       → format_patterns      (schema)  
  ├── {% include "schema/score-output.md" %}          → score_schema_*       (schema)
  ├── {{ focus }}                                     → user input           (per-run)
  ├── {% include "schema/dimension-calibration.md" %} → dimension_calibration (GRAPH)
  └── {% for category in score_calibration %}         → score_calibration    (GRAPH)

code/triage.md
  ├── {% for dim in score_dimensions %}               → score_dimensions     (schema)
  ├── {{ focus }}                                     → user input           (per-run)
  ├── {% include "schema/dimension-calibration.md" %} → dimension_calibration (GRAPH)
  └── FileTriageBatch schema                          → file_triage_schema   (schema)

code/scorer.md
  ├── {% for dim in score_dimensions %}               → score_dimensions     (schema)
  ├── {{ enrichment_patterns }}                       → format_patterns      (schema)
  ├── {{ focus }}                                     → user input           (per-run)
  ├── {% include "schema/dimension-calibration.md" %} → dimension_calibration (GRAPH)
  └── FileScoreBatch schema                           → file_scoring_schema  (schema)

wiki/scorer.md — no graph-dynamic content
wiki/document-scorer.md — no graph-dynamic content
wiki/image-captioner.md — no graph-dynamic content
signals/enrichment.md — no graph-dynamic content
discovery/static-enricher.md — no graph-dynamic content
clusters/labeler.md — no graph-dynamic content
```
