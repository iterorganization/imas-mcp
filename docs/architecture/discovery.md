# Discovery Pipeline Architecture

Multi-domain facility exploration and content discovery for the IMAS knowledge graph.

## Overview

The discovery system comprises **six domain pipelines** that populate the knowledge graph with facility-specific code, documentation, data signals, and metadata. Each domain has its own CLI command under `imas-codex discover <domain>`, plus supporting commands under `imas-codex enrich` and `imas-codex imas`.

The pipelines form a **dependency graph** — some domains produce graph nodes and embeddings that other domains consume as dynamic LLM prompt context. Running them in the optimal order maximizes context quality and minimizes LLM cost.

## Dependency Graph

```
                    ┌──────────────┐
                    │  IMAS DD     │  imas-codex imas build
                    │  (foundation)│  IMASPath, clusters, embeddings
                    └──────┬───────┘
                           │ imas_path_embedding
                           │ cluster_label_embedding
                           ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │    PATHS     │   │    WIKI      │   │   STATIC     │
    │  (parallel)  │   │  (parallel)  │   │  (parallel)  │
    │ scan→score→  │   │ scan→score→  │   │ extract→     │
    │ enrich→      │   │ ingest→      │   │ enrich       │
    │ refine       │   │ artifacts→   │   │              │
    │              │   │ images       │   │              │
    └──────┬───────┘   └──────┬───────┘   └──────────────┘
           │                  │
           │ FacilityPath     │ WikiChunk
           │ (scored ≥0.7)    │ wiki_chunk_embedding
           │                  │ mdsplus_paths_mentioned
           ▼                  ▼
    ┌──────────────┐   ┌──────────────┐
    │    CODE      │   │   SIGNALS    │
    │  (parallel)  │   │  (parallel)  │
    │ scan→triage→ │   │ scan→enrich→ │
    │ score→ingest │   │ check        │
    │              │   │              │
    └──────┬───────┘   └──────────────┘
           │                  ▲
           │ code_chunk_      │ wiki_context
           │ embedding        │ code_chunk_embedding
           │                  │ imas_path_embedding
           └──────────────────┘
    ┌──────────────┐
    │  ENRICH      │  imas-codex enrich nodes
    │  NODES       │  (TreeNode metadata)
    │  (agentic)   │  Uses code + graph context
    └──────────────┘
```

### Dependency Details

| Consumer | Depends On | What It Uses | Impact If Missing |
|----------|-----------|--------------|-------------------|
| `discover code` | `discover paths` | Scored FacilityPath nodes (≥0.7) | No paths to scan for files |
| `discover signals` (enrich) | `discover wiki` | WikiChunk nodes, `wiki_chunk_embedding` index | No wiki descriptions/units injected into signal enrichment prompts → more LLM hallucination |
| `discover signals` (enrich) | `discover code` | `code_chunk_embedding` index | No source code usage patterns in enrichment prompts |
| `discover documents` | `discover paths` | Scored FacilityPath nodes (≥0.5) | No paths to scan for documents/images |
| `enrich nodes` | `discover code` + graph | CodeChunk, TreeNode siblings | Less context for TreeNode physics descriptions |

## Optimal Discovery Sequence

### Phase 0: Foundation (no facility dependency)

```bash
imas-codex imas build        # IMAS DD: IMASPath nodes, embeddings, clusters
```

Populates IMASPath, DDVersion, Unit, IMASSemanticCluster nodes. Creates `imas_path_embedding` and `cluster_label_embedding` vector indexes used by signal enrichment.

### Phase 1: Independent Facility Pipelines (run in parallel)

These three pipelines have **no cross-dependencies** and can run simultaneously:

```bash
imas-codex discover paths tcv       # Directory structure → FacilityPath nodes
imas-codex discover wiki tcv        # Wiki pages → WikiPage, WikiChunk nodes
imas-codex discover static tcv      # Static MDSplus trees → TreeNode nodes
```

### Phase 2: Dependent Pipelines (requires Phase 1)

```bash
imas-codex discover code tcv        # Requires scored paths from Phase 1
imas-codex discover documents tcv   # Requires scored paths from Phase 1
```

### Phase 3: Fully Enriched (requires Phase 1 + 2)

```bash
imas-codex discover signals tcv     # Benefits from wiki + code + IMAS context
imas-codex enrich nodes             # Benefits from code chunks in graph
```

### Quick Reference

| Order | Command | Requires | Produces |
|-------|---------|----------|----------|
| 0 | `imas build` | Nothing | IMASPath, clusters, embeddings |
| 1a | `discover paths` | Facility config | FacilityPath (scored) |
| 1b | `discover wiki` | Wiki URLs in config | WikiPage, WikiChunk, WikiArtifact, Image |
| 1c | `discover static` | MDSplus static_trees config | TreeModelVersion, TreeNode |
| 2a | `discover code` | Scored FacilityPaths (≥0.7) | CodeFile, SourceFile, CodeChunk |
| 2b | `discover documents` | Scored FacilityPaths (≥0.5) | Document, Image |
| 3a | `discover signals` | Wiki + code + IMAS (optional but improves quality) | FacilitySignal, DataAccess |
| 3b | `enrich nodes` | TreeNode + code context | Enriched TreeNode descriptions |

---

## Domain Pipelines

### 1. Paths Discovery (`discover paths`)

**Purpose:** Walk remote filesystem to discover and classify directories.

**Internal pipeline:** `scan → score → expand → enrich → refine`

| Phase | Worker | Method | Description |
|-------|--------|--------|-------------|
| Scan | SSH | `scan_directories.py` remote script | Enumerate directory contents (files, subdirs, README) |
| Score | LLM | `discovery/scorer` prompt | Multi-dimensional scoring + expansion decision |
| Expand | SSH | `scan_directories.py` | Recursively scan children of expanded paths |
| Enrich | SSH | `rg` pattern matching | Deep analysis: pattern categories, LOC, language breakdown |
| Refine | LLM | `discovery/refiner` prompt | Refine scores using enrichment evidence |

**State machine:** `discovered → scanned → scored → (expand?) → (enrich?) → refined`

**Graph coordination:** Workers claim paths atomically via `claimed_at` timestamp. The graph acts as a thread-safe work queue — no two workers can claim the same path. Orphan recovery via 10-minute timeout.

**Prompt:** `discovery/scorer` is dynamic with `schema_needs`:
- `path_purposes` — PathPurpose enum values
- `score_dimensions` — 10 scoring dimensions from schema
- `scoring_schema` — ScoreBatch Pydantic schema for structured output
- `format_patterns` — Pattern category descriptions
- `physics_domains` — PhysicsDomain enum

**Dynamic context injected at runtime:**
- `dimension_calibration` — Sampled examples at 5 score levels per dimension from existing scored paths (cross-facility calibration)
- `focus` — Optional natural language focus from `--focus` flag

**Refine prompt:` `discovery/refiner` receives enrichment evidence (pattern match counts, LOC, language breakdown) to correct initial scores.

```bash
imas-codex discover paths tcv                    # Full pipeline
imas-codex discover paths tcv --scan-only        # SSH only, no LLM cost
imas-codex discover paths tcv --score-only       # Refine from graph
imas-codex discover paths tcv -f "equilibrium"   # Focus scoring
imas-codex discover paths tcv --enrich-threshold 0.75  # Auto-enrich high-value
```

### 2. Wiki Discovery (`discover wiki`)

**Purpose:** Scrape, score, and ingest facility wiki documentation.

**Internal pipeline:** `scan → score → ingest → artifacts → images`

| Phase | Worker | Method | Description |
|-------|--------|--------|-------------|
| Scan | HTTP | Confluence/MediaWiki API | Discover pages via link traversal from portals |
| Score | LLM | `wiki/scorer` prompt | Content-aware scoring with 1500-char preview |
| Ingest | Chunking | Tree-sitter + embedding | Chunk pages, extract MDSplus paths, embed |
| Artifacts | HTTP + LLM | `wiki/artifact-scorer` | Score PDFs/presentations, parse with VLM |
| Images | VLM | `wiki/image-captioner` | Caption images with physics-aware descriptions |

**Prompt:** `wiki/scorer` with `schema_needs`:
- `wiki_page_purposes` — page classification categories
- `wiki_score_dimensions` — wiki-specific scoring dimensions
- `wiki_scoring_schema` — WikiScoreBatch Pydantic schema
- `physics_domains` — PhysicsDomain enum

**Key outputs consumed by other pipelines:**
- `WikiChunk.mdsplus_paths_mentioned` — Pre-extracted MDSplus paths per chunk
- `WikiChunk.ppf_paths_mentioned` — Pre-extracted PPF paths (JET)
- `WikiChunk.imas_paths_mentioned` — Pre-extracted IMAS paths
- `WikiChunk.conventions_mentioned` — Sign conventions, coordinate systems
- `WikiChunk.units_mentioned` — Units found in text
- `wiki_chunk_embedding` vector index — Semantic search for signal enrichment

```bash
imas-codex discover wiki tcv                    # Full pipeline
imas-codex discover wiki tcv --scan-only        # Discover pages only
imas-codex discover wiki tcv --score-only       # Score discovered pages
imas-codex discover wiki tcv --store-images     # Keep image bytes in graph
```

### 3. Static Tree Discovery (`discover static`)

**Purpose:** Extract and enrich MDSplus static (machine-description) trees.

**Internal pipeline:** `extract → units → enrich`

| Phase | Worker | Method | Description |
|-------|--------|--------|-------------|
| Extract | SSH + MDSplus | `extract_static_tree.py` remote script | Walk version, enumerate nodes, ingest to graph |
| Units | SSH + MDSplus | `extract_units.py` remote script | Batch unit extraction for NUMERIC/SIGNAL nodes |
| Enrich | LLM | `discovery/static-enricher` prompt | Batch physics descriptions for enrichable nodes |

**Graph coordination:** TreeModelVersion nodes use `status` + `claimed_at` for worker claim coordination, matching the pattern used by all other discovery domains. Workers claim versions/nodes atomically, and orphan recovery handles stale claims after 300s.

**State machine:** `discovered → ingested | failed`

**Prompt:** `discovery/static-enricher` with `schema_needs`:
- `static_enrichment_schema` — StaticEnrichBatch Pydantic schema

Rendered with `facility` and `tree_name` context variables.

```bash
imas-codex discover static tcv                    # Full pipeline
imas-codex discover static tcv --tree tc_static   # Specific tree only
imas-codex discover static tcv --versions 1,2,3   # Specific versions
imas-codex discover static tcv --no-enrich         # Extract + units only
imas-codex discover static tcv --dry-run           # Preview without graph writes
imas-codex discover static tcv --force             # Re-extract already-ingested versions
imas-codex discover static tcv --cost-limit 5.0    # Max LLM spend
imas-codex discover static tcv --extract-workers 2 # Parallel extraction workers
imas-codex discover static tcv --enrich-workers 3  # Parallel enrichment workers
imas-codex discover status tcv -d static           # Static domain status
imas-codex discover clear tcv -d static            # Clear static data
```

### 4. Code Discovery (`discover code`)

**Purpose:** Discover, score, and ingest source code files from high-value paths.

**Depends on:** Scored FacilityPath nodes from `discover paths` (default `min_score ≥ 0.7`)

**Internal pipeline:** `scan → triage → score → ingest`

| Phase | Worker | Method | Description |
|-------|--------|--------|-------------|
| Scan | SSH | `discover_files.py` remote script | Enumerate files at depth=1 + per-file `rg` pattern matching |
| Triage | LLM | `discovery/file-triage` prompt | Fast keep/skip decision per file |
| Score | LLM | `discovery/file-scorer` prompt | Multi-dimensional scoring with per-file enrichment evidence |
| Ingest | SSH + Parse | tree-sitter chunking + embedding | Fetch file content, chunk, embed, extract IDS/MDSplus refs |

**Triage prompt:** `discovery/file-triage` — fast pass 1 to filter out boilerplate

**Score prompt:** `discovery/file-scorer` with `schema_needs`:
- `file_score_dimensions` — 9 code-relevant scoring dimensions
- `file_scoring_schema` — FileScoreBatch Pydantic schema
- `format_patterns` — Pattern category descriptions for rg evidence

**Path selection:** FacilityPaths are processed highest-value first using weighted dimension scores (`data_access`, `imas`, `convention`, `analysis`, `modeling`).

```bash
imas-codex discover code tcv                    # Full pipeline
imas-codex discover code tcv --min-score 0.8    # Only highest-value paths
imas-codex discover code tcv --scan-only        # SSH enumeration only
imas-codex discover code tcv -f equilibrium     # Focus on equilibrium code
```

### 5. Signal Discovery (`discover signals`)

**Purpose:** Discover and classify facility data signals across all data sources.

**Internal pipeline:** `scan → enrich → check`

| Phase | Worker | Method | Description |
|-------|--------|--------|-------------|
| Scan | Plugins | Scanner registry | Enumerate signals from configured data sources |
| Enrich | LLM | `discovery/signal-enrichment` prompt | Physics domain classification with multi-source context |
| Check | SSH | MDSplus/PPF/EDAS queries | Validate signal accessibility with test shot |

**Scanner plugin system:** Scanners are auto-detected from facility `data_sources` config:
- `tdi` — TDI function enumeration (TCV)
- `mdsplus` — MDSplus tree traversal
- `ppf` — PPF DDA/Dtype enumeration (JET)
- `edas` — EDAS category enumeration (JT-60SA)
- `wiki` — Extract signals from wiki content (always runs first)

**Wiki scanner runs first** so that `wiki_context` is populated before the `enrich_worker` starts processing signals from other scanners.

**Enrichment prompt:** `discovery/signal-enrichment` with `schema_needs`:
- `physics_domains` — PhysicsDomain enum
- `signal_enrichment_schema` — SignalEnrichmentBatch Pydantic schema
- `diagnostic_categories` — Diagnostic classification taxonomy

#### Dynamic Context Injection

The signal enrichment worker injects **five levels of context** into each LLM call:

| Level | Source | Vector Index | Description |
|-------|--------|-------------|-------------|
| **Facility wiki** | `wiki_chunk_embedding` | Semantic search | Sign conventions, coordinate systems, COCOS — cached per facility |
| **Group wiki** | `wiki_chunk_embedding` | Semantic search | Diagnostic/tree-specific wiki content — per signal group |
| **Per-signal wiki** | `state.wiki_context` | Direct path match | Exact MDSplus/PPF path → description/units from wiki chunks |
| **Code context** | `code_chunk_embedding` | Semantic search | Source code patterns, sign conventions, units — per signal group |
| **TDI source** | TDIFunction graph nodes | Direct fetch | Full TDI function source code for TCV signals |

**Dynamic context injection:**
- `_fetch_code_context()` — Queries `code_chunk_embedding` for source code patterns related to each signal group. Used to extract sign conventions, units, and implementation details.

IMAS mapping context is intentionally excluded from signal enrichment — signal description and IMAS mapping are separate concerns. Signals should be accurately described before being mapped to IMAS paths.

```bash
imas-codex discover signals tcv                            # Full pipeline
imas-codex discover signals tcv --scan-only                # Enumerate only
imas-codex discover signals tcv --enrich-only              # Enrich discovered signals
imas-codex discover signals tcv -s tdi,mdsplus             # Specific scanners
imas-codex discover signals tcv --reference-shot 84000     # Override test shot
```

### 6. Document Discovery (`discover documents`)

**Purpose:** Discover document and image files from scored paths.

**Depends on:** Scored FacilityPath nodes from `discover paths` (default `min_score ≥ 0.5`)

**Internal pipeline:** `scan → fetch → caption`

| Phase | Worker | Method | Description |
|-------|--------|--------|-------------|
| Scan | SSH | `discover_files.py` | Enumerate documents/images at scored paths |
| Fetch | SSH | SCP/cat | Retrieve image bytes |
| Caption | VLM | `wiki/image-captioner` | Physics-aware captioning + scoring |

```bash
imas-codex discover documents tcv                # Full pipeline
imas-codex discover documents tcv --scan-only    # Enumeration only
```

---

## Prompt Architecture

All prompts live in `imas_codex/agentic/prompts/` as markdown files with YAML frontmatter.

### Prompt Categories

| Directory | Prompts | Used By |
|-----------|---------|---------|
| `discovery/` | `scorer`, `refiner`, `enricher`, `roots`, `file-triage`, `file-scorer`, `static-enricher`, `signal-enrichment`, `data_access` | Discovery pipelines |
| `wiki/` | `scout`, `scorer`, `artifact-scorer`, `image-captioner` | Wiki pipeline |
| `exploration/` | `facility` | Interactive exploration agent |
| `clusters/` | `labeler` | IMAS DD cluster labeling |
| `shared/` | `safety`, `tools`, `completion` | Included by other prompts |
| `shared/schema/` | `path-purposes`, `score-dimensions`, `physics-domains`, `wiki-purposes`, `discovery-categories`, `diagnostic-categories`, `cluster-vocabularies`, scoring output schemas | Schema-injected includes |

### Rendering Modes

1. **Static** (`parse_prompt_file`) — Resolves `{% include %}` directives only. Used for MCP prompt registration.
2. **Dynamic** (`render_prompt`) — Full Jinja2 rendering with schema providers. Used by scoring/enrichment workers.

### Schema Providers

Each prompt declares `schema_needs` in YAML frontmatter (or uses defaults from `_DEFAULT_SCHEMA_NEEDS`). Providers are `@lru_cache`d and loaded on demand:

| Provider | Data Source | Prompts Using It |
|----------|-------------|-----------------|
| `path_purposes` | PathPurpose enum from LinkML | `discovery/scorer` |
| `score_dimensions` | `score_*` fields from FacilityPath schema | `discovery/scorer` |
| `scoring_schema` | ScoreBatch Pydantic model | `discovery/scorer` |
| `refine_schema` | RefineBatch Pydantic model | `discovery/refiner` |
| `physics_domains` | PhysicsDomain enum from LinkML | scorer, wiki, signals, images |
| `wiki_page_purposes` | WikiPagePurpose enum | `wiki/scorer`, `wiki/artifact-scorer` |
| `wiki_score_dimensions` | Wiki score fields | `wiki/scorer`, `wiki/artifact-scorer` |
| `wiki_scoring_schema` | WikiScoreBatch model | `wiki/scorer` |
| `artifact_scoring_schema` | ArtifactScoreBatch model | `wiki/artifact-scorer` |
| `image_caption_schema` | ImageCaptionBatch model | `wiki/image-captioner` |
| `signal_enrichment_schema` | SignalEnrichmentBatch model | `discovery/signal-enrichment` |
| `diagnostic_categories` | Diagnostic taxonomy | `discovery/signal-enrichment` |
| `static_enrichment_schema` | StaticEnrichBatch model | `discovery/static-enricher` |
| `cluster_vocabularies` | Physics concept vocabulary | `clusters/labeler` |
| `cluster_label_schema` | ClusterLabelBatch model | `clusters/labeler` |
| `file_score_dimensions` | 9 code scoring dimensions | `discovery/file-scorer` |
| `file_scoring_schema` | FileScoreBatch model | `discovery/file-scorer` |
| `file_triage_schema` | FileTriageBatch model | `discovery/file-triage` |
| `format_patterns` | Enrichment pattern categories | scorer, refiner, file-scorer |
| `discovery_categories` | DiscoveryRootCategory enum | `discovery/roots` |
| `data_access_fields` | DataAccess schema fields | `discovery/data_access` |

### Dynamic Context (Runtime)

Beyond schema providers, workers inject runtime context into prompts:

| Context | Injected By | Source | Description |
|---------|-------------|--------|-------------|
| `dimension_calibration` | `score_worker` | Graph query | Sampled scored paths at 5 levels per dimension |
| `enriched_examples` | `refine_worker` | Graph query | Example enriched paths, cross-facility |
| `enrichment_patterns` | `refine_worker` | `PATTERN_REGISTRY` | Pattern→dimension mapping text |
| `wiki_context` | `enrich_worker` (signals) | WikiChunk graph query | Path-matched descriptions/units from wiki |
| `facility_wiki_context` | `enrich_worker` (signals) | `wiki_chunk_embedding` search | Sign conventions, coordinate systems |
| `group_wiki_context` | `enrich_worker` (signals) | `wiki_chunk_embedding` search | Diagnostic-specific wiki documentation |
| `tdi_source` | `enrich_worker` (signals) | TDIFunction graph nodes | Full TDI function source code |
| `focus` | User CLI flag | `--focus` option | Natural language focus area |

---

## Data Model

See [facility.yaml](../../imas_codex/schemas/facility.yaml) for complete schema definitions.

### Core Nodes

| Node Type | Created By | Description |
|-----------|-----------|-------------|
| `FacilityPath` | `discover paths` | Directory with multi-dimensional scores |
| `WikiPage` | `discover wiki` | Wiki page metadata + scores |
| `WikiChunk` | `discover wiki` | Embedded page segment with extracted paths |
| `WikiArtifact` | `discover wiki` | PDF/presentation attached to wiki pages |
| `Image` | `discover wiki/documents` | Captioned image with VLM description |
| `CodeFile` | `discover code` | Discovered source file with per-file enrichment |
| `SourceFile` | `discover code` (ingest) | Fetched + parsed source file |
| `CodeChunk` | `discover code` (ingest) | Tree-sitter parsed code segment with embedding |
| `FacilitySignal` | `discover signals` | Classified data signal with physics domain |
| `DataAccess` | `discover signals` | Data access method template |
| `TreeModelVersion` | `discover static` | Static tree version metadata |
| `TreeNode` | `discover static` | MDSplus tree node with physics description |
| `Document` | `discover documents` | Document file (PDF, etc.) |
| `Evidence` | `discover paths` | LLM scoring rationale (content-addressed) |
| `FacilityUser` | `discover paths` | User account (GECOS-parsed) |
| `Person` | `discover paths` | Cross-facility identity |

### Key Relationships

```
FacilityPath -[:AT_FACILITY]-> Facility
FacilityPath -[:HAS_EVIDENCE]-> Evidence
FacilityPath -[:OWNED_BY]-> FacilityUser
FacilityUser -[:IS_PERSON]-> Person
WikiPage -[:AT_FACILITY]-> Facility
WikiPage -[:HAS_CHUNK]-> WikiChunk
WikiPage -[:HAS_ARTIFACT]-> WikiArtifact
WikiPage -[:HAS_IMAGE]-> Image
CodeFile -[:AT_PATH]-> FacilityPath
SourceFile -[:HAS_CHUNK]-> CodeChunk
FacilitySignal -[:AT_FACILITY]-> Facility
FacilitySignal -[:DATA_ACCESS]-> DataAccess
FacilitySignal -[:MAPS_TO_IMAS]-> IMASPath
TreeNode -[:HAS_NODE]-> TreeNode
TreeNode -[:IN_VERSION]-> TreeModelVersion
```

### Vector Indexes Used by Discovery

| Index | Node.Property | Used For |
|-------|--------------|----------|
| `wiki_chunk_embedding` | WikiChunk.embedding | Signal enrichment wiki context |
| `code_chunk_embedding` | CodeChunk.embedding | Signal enrichment code context (defined, not yet used) |
| `imas_path_embedding` | IMASPath.embedding | Signal enrichment IMAS suggestions (defined, not yet used) |
| `cluster_label_embedding` | IMASSemanticCluster.embedding | Signal enrichment cluster context (defined, not yet used) |
| `facility_signal_desc_embedding` | FacilitySignal.description_embedding | Semantic signal search |

---

## Scoring

LLM-based scoring assigns interest scores (0.0-1.0) across per-purpose dimensions aligned with the DiscoveryRootCategory taxonomy:

### Code Dimensions
| Dimension | Description |
|-----------|-------------|
| `score_modeling_code` | Forward modeling/simulation code (CHEASE, ASTRA, JOREK) |
| `score_analysis_code` | Experimental analysis code (diagnostics, reconstruction) |
| `score_operations_code` | Real-time operations code (control systems, DAQ) |

### Data Dimensions
| Dimension | Description |
|-----------|-------------|
| `score_modeling_data` | Modeling outputs (HDF5 runs, parameter scans) |
| `score_experimental_data` | Experimental shot data (MDSplus, pulse files) |

### Infrastructure Dimensions
| Dimension | Description |
|-----------|-------------|
| `score_data_access` | Data access tools (IMAS wrappers, MDSplus readers) |
| `score_workflow` | Workflow and orchestration tools |
| `score_visualization` | Plotting and rendering tools |

### Support Dimensions
| Dimension | Description |
|-----------|-------------|
| `score_documentation` | Documentation, readmes, tutorials |
| `score_imas` | Cross-cutting IMAS relevance |
| `score_convention` | Sign conventions, COCOS, coordinate systems |

**Final score formula:**
```
score = max(all per-purpose scores)
```

Scores are the LLM's final word — no post-processing boosts, multipliers, or caps.

## Worker Architecture

All parallel pipelines share the same `SupervisedWorkerGroup` architecture from `discovery/base/`:

- **Claim-based coordination:** Workers claim items by setting `claimed_at` timestamp (atomic graph operation)
- **Orphan recovery:** Items claimed >10 minutes ago without completion are reclaimed
- **Phase tracking:** Each pipeline phase (`PipelinePhase`) tracks idle count and pending work
- **Termination:** Pipeline stops when all phases are idle with no pending graph work, or budget/deadline reached
- **Cost tracking:** Per-worker cost accumulation with configurable `--cost-limit`

## User Enrichment

User information is extracted during the scan phase via GECOS parsing.

Each facility's `user_info` block defines parsing behavior:

```yaml
user_info:
  name_format: last_first      # "Last First [EXT]" (ITER)
  gecos_suffix_pattern: "\\s+EXT$"
  lookup_tools: [getent, passwd, id]
```

Users are deduplicated via Person nodes using ORCID, normalized name, or email.

## Management Commands

```bash
imas-codex discover status tcv              # All domain statistics
imas-codex discover status tcv -d wiki      # Wiki domain only
imas-codex discover clear tcv               # Clear all domains
imas-codex discover clear tcv -d paths      # Clear paths only
imas-codex discover seed tcv                # Seed root paths from config
imas-codex discover inspect tcv             # Debug view of scanned/scored paths
```

## Related Documentation

- [agents/explore.md](../../agents/explore.md) — Exploration agent workflow
- [agents/ingest.md](../../agents/ingest.md) — Ingestion pipeline
- [facility.yaml](../../imas_codex/schemas/facility.yaml) — Schema definitions
- [graph.md](graph.md) — Graph architecture
- [wiki.md](wiki.md) — Wiki scraping details
