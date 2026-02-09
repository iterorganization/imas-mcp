# Docs Discovery Refactoring Plan

**Status**: Planning  
**Scope**: Migrate wiki CLI to `discover docs`, unify state machines, composable pipelines

## 1. Multilingual Embedding Model

### Qwen3-Embedding Model Family (June 2025)

| Model | Size | Dims | Context | MTEB Score | Notes |
|-------|------|------|---------|------------|-------|
| `Qwen3-Embedding-0.6B` | 0.6B | 1024 | 32K | 65.68 | Best efficiency, WSL-friendly |
| `Qwen3-Embedding-4B` | 4B | 2560 | 32K | 68.82 | Good balance |
| `Qwen3-Embedding-8B` | 8B | 4096 | 32K | 70.58 | #1 MTEB multilingual |

### Benchmark Results (Feb 2026)

Tested on IMAS DD retrieval corpus with English and multilingual queries:

| Model | Dim | Load | Memory | EN Accuracy | Multilingual Accuracy |
|-------|-----|------|--------|-------------|----------------------|
| all-MiniLM-L6-v2 | 384 | 1.9s | 32 MB | 100% | 83% |
| Qwen3-Embedding-0.6B | 1024 | 4.7s | 2.3 GB | 100% | **100%** |

**Key Findings:**
- Qwen3 achieves **100% multilingual accuracy** (Japanese, French) vs 83% for MiniLM
- Japanese queries "電子温度プロファイル" → correctly retrieves `core_profiles/.../temperature`
- Memory footprint ~70x larger but acceptable for server deployment
- Encode time ~8x slower but still fast enough (1s for 20 texts)

**Recommendation**: `Qwen3-Embedding-0.6B` for initial deployment
- 0.6B parameters fits in 4GB VRAM (WSL compatible)
- 119 language support including Japanese, Chinese, French, German
- Instruction-aware: can customize embeddings for retrieval vs clustering
- MRL support: can reduce dimensions post-hoc if needed (1024 → 384)

**Migration path**:
```python
# pyproject.toml
[tool.imas-codex]
imas-embedding-model = "Qwen/Qwen3-Embedding-0.6B"

# Backward compatibility: cache includes model name hash
# All DD embeddings regenerated on first build after switch
```

**Benchmark before switching**:
```bash
uv run pytest tests/embeddings/test_model_comparison.py \
    --models "all-MiniLM-L6-v2,Qwen/Qwen3-Embedding-0.6B" \
    --test-cases "dd_retrieval,wiki_search,code_search"
```

---

## 2. State Machine Unification

### Current State Machines

**PathStatus** (discovery/path exploration):
```
discovered → listing → listed → scoring → scored
                                         ↘ skipped
                                         ↘ expanding
```

**WikiPageStatus** (wiki ingestion):
```
scanned → scored → ingested
                ↘ skipped
                ↘ failed
```

**SourceFileStatus** (code ingestion):
```
discovered → ingested
           ↘ failed
           ↘ stale
```

### Proposed Unified Model: `ResourceStatus`

All discovery pipelines share a common lifecycle with domain-specific completion terms:

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Unified Resource Lifecycle                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  DISCOVERY PHASE (graph population)                                 │
│  ──────────────────────────────────                                 │
│  discovered: Resource found, node created in graph                  │
│                                                                      │
│  PROCESSING PHASE (domain-specific)                                 │
│  ────────────────────────────────                                   │
│  processing: Worker claimed (transient, with fallback)              │
│  processed: Domain work complete                                    │
│                                                                      │
│  TERMINAL STATES                                                    │
│  ───────────────                                                    │
│  skipped: Intentionally not processed (low value, filtered)         │
│  failed: Error during processing                                    │
│  stale: Previously processed, may need refresh                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Mapping Existing to Unified

| Domain | Current | Proposed Unified | Notes |
|--------|---------|------------------|-------|
| **Paths** | discovered | discovered | ✓ Same |
| | listing | processing | Transient |
| | listed | enumerated | Domain-specific completion for scan |
| | scoring | processing | Transient |
| | scored | scored | Domain-specific completion for LLM eval |
| | skipped | skipped | ✓ Same |
| | excluded | filtered | Rename: pattern-matched exclusion |
| | stale | stale | ✓ Same |
| **Wiki** | scanned | discovered | Renamed: link found |
| | scored | scored | ✓ Same |
| | ingested | ingested | Domain-specific completion |
| | skipped | skipped | ✓ Same |
| | failed | failed | ✓ Same |
| | stale | stale | ✓ Same |
| **Files** | discovered | discovered | ✓ Same |
| | ingested | ingested | ✓ Same |
| | failed | failed | ✓ Same |
| | stale | stale | ✓ Same |

### Decision: Keep Domain-Specific Enums

After analysis, **keep separate enums** but with **shared terminology** where phases overlap:

**Rationale**:
1. Paths have multi-phase discovery (scan → score → expand) not in other domains
2. Wiki has "scanned" phase that paths don't have
3. Type safety: `WikiPage.status` should only accept wiki-valid states
4. Single enum would have 20+ values, confusing which apply where

**Action**: Align terminology, don't merge enums:
- All use `discovered` for initial state (wiki changes `scanned` → `discovered`)
- All use `skipped` (not `excluded`)
- All use `failed` for errors
- All use `stale` for refresh candidates

---

## 3. Enum Renames in Schema

### PathPurpose → ResourcePurpose

Current `PathPurpose` applies to directories. With wiki pages, we need the same taxonomy for different resource types.

**Before**: `PathPurpose` (facility.yaml)
**After**: `ResourcePurpose` (common.yaml)

Values remain unchanged, just namespace moved for reuse:
- `modeling_code`, `analysis_code`, `operations_code`
- `documentation`, `visualization`, `workflow`
- `container`, `archive`, `build_artifact`, `system`

**Usage**:
```python
# FacilityPath node
path.purpose = ResourcePurpose.modeling_code

# WikiPage node  
page.purpose = ResourcePurpose.documentation

# SourceFile node
file.purpose = ResourcePurpose.analysis_code
```

### DiscoveryStatus → PathPhase

Current `DiscoveryStatus` is poorly named - it only applies to path scanning.

**Before**: `DiscoveryStatus` with values: `pending`, `scanned`, `scored`, `skipped`
**After**: Merge into `PathStatus` or delete

This enum duplicates `PathStatus`. Recommend removal.

### WikiPageStatus Rename: scanned → discovered

Align with other domains:
```yaml
WikiPageStatus:
  permissible_values:
    discovered:  # was: crawled
      description: Page found via link extraction, awaiting scoring
    scored:
      description: LLM evaluated, interest_score set
    # ... rest unchanged
```

### Summary of Schema Changes

| Current | Proposed | Scope |
|---------|----------|-------|
| `PathPurpose` | `ResourcePurpose` | Move to common.yaml |
| `DiscoveryStatus` | DELETE | Merge into PathStatus |
| `WikiPageStatus.scanned` | `WikiPageStatus.discovered` | Rename value |
| `WikiSiteType` | DELETE | Already deprecated, use DocSourceType |

---

## 4. Composable Pipeline Architecture

### Current Structure

```
imas_codex/
├── discovery/           # Path discovery
│   ├── scanner.py       # SSH enumeration
│   ├── scorer.py        # LLM scoring
│   ├── frontier.py      # Graph state management
│   └── parallel.py      # Async coordination
│
├── wiki/               # Wiki discovery
│   ├── discovery.py    # Scan + score
│   ├── scraper.py      # HTML fetching
│   ├── pipeline.py     # Chunking + embedding
│   └── confluence.py   # REST API client
```

### Proposed Structure

```
imas_codex/
├── discovery/
│   ├── base.py              # NEW: Abstract pipeline classes
│   ├── paths/               # Renamed from root discovery/
│   │   ├── scanner.py
│   │   ├── scorer.py
│   │   └── frontier.py
│   │
│   ├── docs/                # NEW: Unified docs discovery
│   │   ├── scanner.py       # Abstract + MediaWiki/Confluence impls
│   │   ├── scorer.py        # Shared scoring logic
│   │   └── fetcher.py       # Content fetching
│   │
│   └── shared/              # NEW: Reusable components
│       ├── chunker.py       # Text chunking
│       ├── embedder.py      # Embedding generation
│       ├── extractor.py     # Entity extraction
│       └── artifacts.py     # PDF/image processing
│
├── wiki/                    # DEPRECATED → discovery/docs
│   └── (files moved)
```

### Abstract Base Classes

```python
# imas_codex/discovery/base.py

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Iterator

T = TypeVar("T")  # Resource type (FacilityPath, WikiPage, etc.)

class DiscoveryPipeline(ABC, Generic[T]):
    """Base class for all discovery pipelines."""
    
    facility_id: str
    
    @abstractmethod
    def scan(self) -> Iterator[T]:
        """Discover resources, yield nodes for graph."""
        ...
    
    @abstractmethod
    def score(self, resources: list[T]) -> list[T]:
        """LLM-score resources, return scored nodes."""
        ...
    
    @abstractmethod
    def ingest(self, resources: list[T]) -> IngestStats:
        """Process high-value resources."""
        ...


class ResourceScorer(ABC, Generic[T]):
    """LLM-based scoring with shared cost tracking."""
    
    def __init__(self, model: str, cost_limit: float):
        self.model = model
        self.cost_limit = cost_limit
        self.cost_spent = 0.0
    
    @abstractmethod
    def build_prompt(self, batch: list[T]) -> str:
        """Build scoring prompt for a batch of resources."""
        ...
    
    def score_batch(self, batch: list[T]) -> list[ScoredResource]:
        """Score a batch, track cost, return results."""
        # Shared LLM call + cost tracking logic
        ...


class ArtifactProcessor(ABC):
    """Process non-text artifacts (PDFs, images)."""
    
    @abstractmethod
    def can_process(self, artifact: Artifact) -> bool:
        """Return True if this processor handles the artifact."""
        ...
    
    @abstractmethod
    async def process(self, artifact: Artifact) -> ProcessedArtifact:
        """Extract text/captions, generate embeddings."""
        ...
```

---

## 5. CLI Migration: wiki → discover docs

### Current Wiki CLI (deprecated)

```bash
imas-codex wiki discover tcv        # Full pipeline
imas-codex wiki scan tcv           # Link extraction
imas-codex wiki score tcv           # LLM evaluation
imas-codex wiki ingest tcv          # Content processing
imas-codex wiki status tcv          # Progress
imas-codex wiki sites tcv           # List sources
imas-codex credentials set ... # Auth management
```

### New discover docs CLI

```bash
# Discovery
imas-codex discover docs tcv                 # Full pipeline
imas-codex discover docs tcv --scan-only    # Link extraction
imas-codex discover docs tcv --score-only    # LLM evaluation (no scan)
imas-codex discover docs tcv --source wiki   # Specific source type

# Source management (moved from wiki sites)
imas-codex discover sources list             # Existing
imas-codex discover sources add ...          # Existing

# Status integration
imas-codex discover status tcv --domain docs

# Credential management (stays under wiki for now)
imas-codex credentials set iter
```

### Implementation Plan

1. **Phase 1**: Move wiki/ → discovery/docs/, update imports
2. **Phase 2**: Wire up `discover docs` command with new structure
3. **Phase 3**: Add deprecation warnings to wiki CLI
4. **Phase 4**: Remove wiki CLI in next minor release

---

## 6. Implementation Order

| Phase | Task | Effort | Dependency |
|-------|------|--------|------------|
| 1a | Rename `PathPurpose` → `ResourcePurpose` in schema | S | None |
| 1b | Rename `WikiPageStatus.scanned` → `discovered` | S | None |
| 1c | Delete `DiscoveryStatus`, `WikiSiteType` enums | S | None |
| 2 | Create `discovery/base.py` with abstract classes | M | None |
| 3 | Move `wiki/` → `discovery/docs/` | M | 2 |
| 4 | Implement `discover docs` CLI command | M | 3 |
| 5 | Add Qwen3-Embedding support | M | None (parallel) |
| 6 | Deprecate `wiki` CLI group | S | 4 |
| 7 | Image captioning pipeline | L | 3 |

---

## 7. Breaking Changes

### Schema Changes (require regeneration)

```bash
uv run build-models --force
```

Affected generated files:
- `imas_codex/graph/models.py` - Pydantic models
- `imas_codex/graph/dd_models.py` - DD models

### Graph Migration

For existing graphs, add migration Cypher:

```cypher
// Rename WikiPage.status from 'crawled' to 'discovered'
MATCH (wp:WikiPage {status: 'scanned'})
SET wp.status = 'discovered'

// Rename FacilityPath.path_purpose field uses no change needed (values same)
```

### Import Path Changes

```python
# Before
from imas_codex.wiki.discovery import WikiDiscovery
from imas_codex.wiki.pipeline import WikiIngestionPipeline

# After
from imas_codex.discovery.docs import DocsDiscoveryPipeline
from imas_codex.discovery.docs.pipeline import DocsIngestionPipeline
```

---

## Open Questions

1. **Artifact storage**: Store images inline (base64) or just URLs + captions?
   - Recommendation: URLs + captions, download on demand

2. **Multi-wiki per facility**: Keep current list structure?
   - Yes, `wiki_sites` list in facility YAML works well

3. **Language detection**: Detect before embedding or let model handle?
   - Qwen3 handles mixed-language natively, no detection needed

4. **Credential migration**: ~~Keep under `wiki credentials` or move?~~ Moved to top-level `credentials` command.
   - Credentials are service-agnostic, not wiki-specific

