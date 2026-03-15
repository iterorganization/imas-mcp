# Mapping Pipeline: Search Enrichment & Mapping Fidelity

**Status**: Plan  
**Created**: 2026-03-15  
**Updated**: 2026-03-15  
**Scope**: `imas_codex/ids/`, `imas_codex/llm/search_tools.py`, `imas_codex/tools/graph_search.py`, `imas_codex/schemas/facility.yaml`

## Executive Summary

The IMAS mapping pipeline (`imas map run`) generates signal-level mappings from
facility signal sources to IMAS IDS fields. This plan addresses two interrelated
areas: **search enrichment** (better context for mapping decisions) and **mapping
fidelity** (correct handling of no-match, many-to-one, and one-to-many mappings).

### Search Enrichment Gaps

1. **Scored wiki and code content** — 982 wiki pages and 569 code files in
   pf_active-relevant domains are never consulted during mapping
2. **Score dimension filtering** — 6 content dimensions and 11 code/path
   dimensions exist on graph nodes but are unused in search queries
3. **Cross-domain semantic bridging** — source embeddings can be cosine-compared
   directly to IMAS target embeddings for high-precision candidate ranking
4. **MCP tool discoverability** — agents calling `search_docs`, `search_code`,
   `search_signals` have no visibility into available physics domains or score
   dimensions at call time

### Mapping Fidelity Gaps

5. **Forced low-confidence matches** — The pipeline currently produces a binding
   for every assigned source, even when no credible IMAS target exists. Many
   facility signals (operational parameters, diagnostics metadata, facility-
   specific computed quantities) have no IMAS equivalent. These should be
   persisted as explicit "no mapping" decisions with evidence-based reasoning,
   not forced into low-confidence bindings.
6. **Many-to-one mapping legitimacy** — Multiple source signals mapping to the
   same IMAS target is a valid and expected pattern (e.g., machine description
   data from different epochs, signals at different processing stages: raw,
   subsampled, filtered). The pipeline and validation need to distinguish
   legitimate many-to-one mappings from erroneous duplicates.
7. **One-to-many mapping support** — A single source signal should map to
   multiple IMAS targets when IMAS stores the same physical parameter in
   multiple locations (e.g., plasma current in `core_profiles`, `equilibrium`,
   `summary`). The IMAS clustering system already groups these cross-IDS
   equivalent paths and should be exploited to discover them.

## Current State

### Pipeline Architecture

```
gather_context → assign_sections → map_signals → discover_assembly → validate → persist
     │                │                 │
     │                │                 ├─ per-source semantic search (IMAS only)
     │                │                 ├─ code references (SignalSource→CodeChunk)
     │                │                 └─ unit analysis + COCOS context
     │                │
     │                └─ ALL sources + full IDS subtree + semantic hits + clusters
     │
     ├─ physics domain scoping (target IDS domains → filter sources)
     ├─ signal source query (enriched sources with metadata)
     ├─ IDS subtree fetch
     ├─ per-source semantic search against imas_node_embedding
     ├─ cross-facility mapping precedent
     └─ section cluster listing
```

### What Works Well

- **Physics domain scoping** reduces candidate sources dramatically (e.g., JET
  pf_active: 2288 → 197 sources, 91% reduction)
- **Per-source semantic search** in `gather_context()` runs each source's
  description against `imas_node_embedding` to find candidate IMAS fields
- **Hybrid vector + text search** on signals, wiki, and code via the MCP tools
- **Rich enrichment metadata** — signals have descriptions, keywords, units,
  COCOS, diagnostics; IMAS nodes have enriched descriptions, keywords, physics
  domains

### Gap Analysis

#### Gap 1: Wiki and Code Content Not Used in Mapping

The mapping pipeline calls `fetch_source_code_refs()` which traverses
`SignalSource → representative FacilitySignal → SignalNode → CodeChunk`.
This narrow traversal only finds code that directly reads the specific signal.

**Missing content** (verified via live graph queries):
- **Wiki pages**: 982 pages in `magnetic_field_systems`/`plasma_control` domains
  with `score_composite > 0.3`, including pages like "Magnetostatic Ontology"
  (imas_relevance=0.80), "EFCC" (data_access=0.60), "PIW:RTPS:Handbook"
  (data_access=0.85) — all highly relevant to pf_active mapping
- **Code files**: 569 facility paths in relevant domains with scored dimensions,
  including CREATE-NL simulation suites (imas=0.95), magnetic field analysis
  tools (imas=0.92), ripple analysis environments
- **Code chunks**: Semantic search against `code_chunk_embedding` using source
  descriptions yields relevant code like pfcoil geometry parsers (score=0.877),
  EFIT XML handlers (0.868), coil geometry extractors (0.862)

**Impact**: The LLM makes mapping decisions without access to domain expertise
documented in wiki pages or data access patterns demonstrated in code files.

#### Gap 2: Score Dimension Filtering Unused

Every wiki page has 6 content scoring dimensions and every facility path has
11 code/path scoring dimensions. These are stored on graph nodes but **never
used in any search query**.

**Content scoring dimensions** (WikiPage, Document):
```
score_data_documentation  — Signal tables, node lists, shot databases
score_physics_content     — Physics explanations, methodology, theory
score_code_documentation  — Software docs, API references, usage guides
score_data_access         — MDSplus paths, TDI expressions, access methods
score_calibration         — Calibration info, conversion factors, sensor specs
score_imas_relevance      — IMAS integration, IDS references, mapping hints
```

**Code/path scoring dimensions** (FacilityPath, SourceFile):
```
score_modeling_code       — Forward modeling/simulation code
score_analysis_code       — Experimental analysis code
score_operations_code     — Real-time operations code
score_data_access         — Data access tools (shared with content)
score_workflow            — Workflow/orchestration
score_visualization       — Visualization tools
score_documentation       — Documentation
score_imas                — IMAS relevance
score_convention          — Convention handling (COCOS, units)
score_modeling_data        — (path-only) Modeling data
score_experimental_data    — (path-only) Experimental data
```

**Mapping-relevant dimensions**:
- `score_data_access` — Most critical. When building signal mappings, we need
  code and docs that show _how_ data is accessed. Filtering on this dimension
  surfaces TDI expressions, MDSplus patterns, and API examples.
- `score_imas_relevance` / `score_imas` — Directly relevant. Content scored
  high on IMAS relevance explicitly discusses IDS structure, mapping patterns,
  or IMAS integration.
- `score_convention` — For COCOS/sign convention handling. Code scored high on
  this dimension contains sign flip logic, coordinate transformations, and unit
  conversion patterns.
- `score_calibration` — For unit conversion context. Calibration docs contain
  conversion factors and sensor specifications.

#### Gap 3: No Cross-Domain Semantic Bridge

The pipeline currently performs unidirectional semantic search: source
description → IMAS node embeddings. A richer approach would compute
**bidirectional embedding matches** across all content types:

```
Source content (signals, wiki, code) ←→ Target content (IMAS nodes)
```

**Verified via live testing**: Embedding a signal source description
("turns per element for Poloidal Field coil 10") and searching against
`imas_node_embedding` yields `pf_active/coil/element/turns_with_sign`
at cosine=0.893. The same embedding against `code_chunk_embedding` yields
pfcoil geometry code at 0.877. Against `wiki_chunk_embedding` yields
disruption database entries at 0.865.

This means we can build a **match matrix** where rows are source signals
and columns are IMAS target paths, with each cell being a cosine similarity.
High-scoring cells are strong mapping candidates. Adding wiki and code chunks
to the source side provides additional bridging evidence.

#### Gap 4: MCP Tool Docstrings Lack Dynamic Content

Agents calling `search_signals`, `search_docs`, `search_code` have no way to
know what `physics_domain` values are valid or what score dimensions exist.
The `physics_domain` parameter on `search_signals` accepts a string but
doesn't list the 22 valid enum values. The `search_docs` and `search_code`
tools don't even accept a `physics_domain` parameter.

#### Gap 5: No Graceful Handling of Unmappable Signals

The current pipeline model (`SignalMappingBatch`) only supports positive
bindings and escalation flags. There is no structured way to say "this signal
has no IMAS target and here is why." The `unassigned_groups` field on
`SectionAssignmentBatch` captures sources that don't fit any section, but
doesn't persist them with a reason. Sources that are assigned to a section
but have no valid field target are forced into low-confidence bindings or
silently dropped.

**Expected categories of unmappable signals**:
- Facility-specific operational parameters (e.g., power supply voltages,
  cryogenics temperatures) with no IMAS IDS equivalent
- Diagnostic metadata (e.g., calibration timestamps, acquisition rates)
  that are properties of the measurement, not measured quantities
- Computed indicators (e.g., local plasma state indices) that are
  facility-specific derived quantities
- Signals from systems not yet covered by IMAS DD versions in use

**Impact**: Low-confidence bindings pollute the graph with incorrect
`MAPS_TO_IMAS` relationships, reducing overall mapping quality and
complicating downstream IDS assembly.

#### Gap 6: Duplicate Target Detection Conflates Valid Many-to-One Patterns

`validate_mappings()` detects duplicate targets and creates escalation flags,
but does not distinguish between:

1. **Legitimate many-to-one** — multiple sources validly map to the same IMAS
   target because they represent:
   - The same quantity across machine description epochs (e.g., coil
     geometry from different commissioning campaigns)
   - Different stages of a processing pipeline (raw measurement,
     subsampled, filtered, ELM-averaged)
   - Redundant diagnostics measuring the same quantity

2. **Erroneous duplicates** — two unrelated sources mapping to the same target
   due to LLM confusion

**Impact**: Legitimate patterns generate false escalation noise, masking
genuinely erroneous duplicates.

#### Gap 7: No Cluster-Aware One-to-Many Mapping

The IMAS clustering system groups paths that represent the same physical
parameter across multiple IDSs (e.g., plasma current appears in
`core_profiles/global_quantities/ip`, `equilibrium/time_slice/global_quantities/ip`,
`summary/global_quantities/ip/value`). The mapping pipeline does not use
this clustering to discover additional targets when a source maps to one
member of a cluster.

**Impact**: Each `imas map run` invocation only maps within a single IDS,
missing cross-IDS mappings that the cluster index could supply. Even within
a single IDS, related paths in the same cluster are not surfaced as
candidates.

## Phased Implementation

### Phase 1: Mapping Fidelity — Output Model, Prompt, and Schema (Gaps 5, 6, 7)

**Goal**: Enable the pipeline to produce correct outputs for the three
fundamental mapping cardinalities: no-match (0:1), many-to-one (N:1), and
one-to-many (1:N). The majority of this work is in prompt refinement, output
model development, and schema evolution — minimal changes to pipeline
orchestration.

**Targets**:
- `imas_codex/ids/models.py` — output model changes
- `imas_codex/llm/prompts/mapping/signal_mapping.md` — prompt refinement
- `imas_codex/llm/prompts/mapping/section_assignment.md` — prompt refinement
- `imas_codex/schemas/facility.yaml` — schema extensions
- `imas_codex/ids/validation.py` — validation refinement
- `imas_codex/ids/mapping.py` — persist unmapped decisions

**Changes**:

#### 1a. Explicit No-Match Output Model (Gap 5)

Add a structured model for signals that have no IMAS target. This replaces
the current pattern of forcing low-confidence bindings or silently dropping
unmatched signals.

```python
class MappingDisposition(StrEnum):
    """Why a signal was or was not mapped."""
    MAPPED = "mapped"                       # Positive binding exists
    NO_IMAS_EQUIVALENT = "no_imas_equivalent"  # No corresponding IDS field
    METADATA_ONLY = "metadata_only"         # Signal is diagnostic metadata, not a measurement
    FACILITY_SPECIFIC = "facility_specific" # Facility-specific quantity, no IDS coverage
    INSUFFICIENT_CONTEXT = "insufficient_context"  # Could map but evidence is inadequate
    DD_VERSION_GAP = "dd_version_gap"       # Target exists in newer DD but not current

class UnmappedSignal(BaseModel):
    """A signal that was evaluated and determined to have no valid IMAS target."""
    source_id: str = Field(description="SignalSource node id")
    disposition: MappingDisposition = Field(
        description="Why this signal has no mapping"
    )
    evidence: str = Field(
        description=(
            "Concise evidence-based explanation. Must reference concrete facts: "
            "searched IMAS paths, checked IDS sections, why no match exists. "
            "Example: 'No IDS field for PF supply interlock status. "
            "Searched pf_active/supply — only current/voltage/energy fields "
            "exist. Interlock is a facility-specific operational parameter.'"
        )
    )
    nearest_imas_path: str | None = Field(
        default=None,
        description="Closest IMAS path considered but rejected, if any"
    )
    nearest_similarity: float | None = Field(
        default=None, ge=0, le=1,
        description="Cosine similarity to nearest_imas_path, if computed"
    )
```

Extend `SignalMappingBatch` to include unmapped signals:

```python
class SignalMappingBatch(BaseModel):
    ids_name: str
    section_path: str
    mappings: list[SignalMappingEntry]
    unmapped: list[UnmappedSignal] = Field(default_factory=list)  # NEW
    escalations: list[EscalationFlag] = Field(default_factory=list)
```

Extend `ValidatedSignalMapping` to carry the disposition and evidence:

```python
class ValidatedSignalMapping(BaseModel):
    source_id: str
    source_property: str = "value"
    target_id: str
    transform_expression: str = "value"
    source_units: str | None = None
    target_units: str | None = None
    cocos_label: str | None = None
    confidence: float = Field(ge=0, le=1)
    disposition: MappingDisposition = MappingDisposition.MAPPED  # NEW
    evidence: str = ""  # NEW — reasoning for mapping or non-mapping
```

#### 1b. Prompt Refinement for No-Match Honesty (Gap 5)

Update `signal_mapping.md` to explicitly instruct the LLM to produce
`unmapped` entries when no credible target exists. The key prompt additions:

```markdown
## No-Match Handling

Not every signal has an IMAS equivalent. When no target field exists:

1. **Do not force a low-confidence mapping.** A confidence < 0.3 mapping is
   worse than an explicit "no mapping" decision.
2. **Add to `unmapped`** with a `disposition` explaining why:
   - `no_imas_equivalent` — The physical quantity has no IDS field
   - `metadata_only` — The signal is diagnostic metadata (acquisition rate,
     calibration timestamp) not a measured/computed quantity
   - `facility_specific` — Facility-specific operational parameter
   - `insufficient_context` — Could map but evidence is too weak to commit
   - `dd_version_gap` — Target exists in newer DD but not current version
3. **Provide evidence**: Reference the IMAS paths you searched, the section
   fields available, and why none match. Cite specific field names.
4. **Set `nearest_imas_path`** if you found a close-but-wrong candidate,
   and explain in `evidence` why it was rejected.

**Confidence threshold**: If your best candidate has confidence < 0.3, emit
an `unmapped` entry instead of a mapping.
```

Also update `section_assignment.md` to make `unassigned_groups` first-class
with per-source reasoning:

```python
class UnassignedSource(BaseModel):
    """A source that could not be assigned to any IDS section."""
    source_id: str
    disposition: MappingDisposition
    evidence: str

class SectionAssignmentBatch(BaseModel):
    ids_name: str
    assignments: list[SectionAssignment]
    unassigned: list[UnassignedSource] = Field(default_factory=list)  # replaces unassigned_groups
```

#### 1c. Schema Extension for Mapping Reasoning (Gap 5)

Add a `mapping_reason` property to the `MAPS_TO_IMAS` relationship and a
`mapping_disposition` + `mapping_evidence` on `SignalSource` for unmapped
signals:

```yaml
# In SignalSource attributes (facility.yaml):
mapping_disposition:
  description: >-
    Disposition of mapping evaluation. Set to the MappingDisposition
    value after the mapping pipeline processes this source.
    'mapped' if MAPS_TO_IMAS exists, otherwise the reason no mapping exists.
  range: MappingDispositionEnum

mapping_evidence:
  description: >-
    Concise evidence-based explanation for the mapping disposition.
    For mapped signals: why this target was chosen, with references.
    For unmapped signals: what was searched and why no target fits.
  range: string
```

The `MAPS_TO_IMAS` relationship already carries `confidence` and
`transform_expression` as properties. Add `evidence` as a relationship
property in `persist_mapping_result()`:

```python
# In persist_mapping_result(), step 4:
gc.query(
    """
    MATCH (sg:SignalSource {id: $sg_id})
    MATCH (ip:IMASNode {id: $target_id})
    MERGE (sg)-[r:MAPS_TO_IMAS]->(ip)
    SET r.source_property = $source_property,
        r.transform_expression = $transform_expression,
        r.confidence = $confidence,
        r.evidence = $evidence
        ...
    """,
    evidence=fm.evidence,
    ...
)
```

For unmapped signals, persist the disposition directly on the `SignalSource`
node (no `MAPS_TO_IMAS` relationship is created):

```python
# New step in persist_mapping_result():
for um in unmapped_signals:
    gc.query(
        """
        MATCH (sg:SignalSource {id: $sg_id})
        SET sg.mapping_disposition = $disposition,
            sg.mapping_evidence = $evidence
        """,
        sg_id=um.source_id,
        disposition=um.disposition.value,
        evidence=um.evidence,
    )
```

#### 1d. Many-to-One Validation Refinement (Gap 6)

The current `validate_mappings()` flags all duplicate targets as escalations.
Refine this to classify many-to-one patterns:

```python
class DuplicateTargetClassification(StrEnum):
    """Classification of why multiple sources map to the same target."""
    EPOCH_VARIANTS = "epoch_variants"         # Same quantity, different epochs
    PROCESSING_STAGES = "processing_stages"   # Raw/filtered/averaged variants
    REDUNDANT_DIAGNOSTICS = "redundant_diagnostics"  # Different instruments, same measurement
    LEGITIMATE_OTHER = "legitimate_other"     # Other valid many-to-one pattern
    ERRONEOUS = "erroneous"                   # Likely LLM error
```

Update the duplicate target detection in `validate_mappings()` to:

1. Group bindings by `target_id`
2. For each group with >1 binding, check source metadata:
   - If sources share a `group_key` prefix but differ by index → epoch variants
   - If sources have the same `physics_domain` and similar descriptions
     but different `representative_signal` tree paths → processing stages
   - If sources are from different diagnostics → redundant diagnostics
3. Only flag `ERRONEOUS` when sources have unrelated physics domains or
   descriptions with low mutual similarity

This is primarily a prompt + output model refinement. Add a `many_to_one_note`
field to `SignalMappingEntry`:

```python
class SignalMappingEntry(BaseModel):
    ...
    many_to_one_note: str | None = Field(
        default=None,
        description=(
            "When multiple sources map to this target, explain why. "
            "E.g., 'Epoch variant: same coil geometry from 2019 commissioning' "
            "or 'Processing stage: raw measurement before ELM filtering'"
        ),
    )
```

Update the signal_mapping prompt to instruct the LLM:

```markdown
## Many-to-One Mappings

Multiple source signals mapping to the same IMAS target is expected and valid.
Common patterns:
- **Epoch variants**: The same physical quantity measured/defined at different
  machine configuration epochs (e.g., coil geometry from different commissioning
  campaigns). All are correct mappings — which epoch to use is resolved at
  assembly time via the `source_epoch_field`.
- **Processing stages**: Raw, subsampled, filtered, or ELM-averaged variants
  of the same measurement. All map to the same IMAS field — which stage to use
  is a user/workflow choice.
- **Redundant diagnostics**: Different instruments measuring the same quantity
  (e.g., two independent Ip Rogowski coils).

When you map multiple sources to the same target, set `many_to_one_note` on
each mapping to explain the relationship between the sources.
```

#### 1e. Cluster-Aware One-to-Many Mapping (Gap 7)

Enable the pipeline to discover additional IMAS targets for a source signal
by consulting the IMAS semantic cluster index. This is a prompt + context
enrichment change — the `SignalMappingEntry` model already supports multiple
mappings per source via the `mappings` list, and the prompt already says
"Return ALL valid mappings for each source."

**Context enrichment** in `gather_context()`:

```python
# After semantic search, look up clusters for top-scoring IMAS matches
from imas_codex.clusters.search import ClusterSearcher

cluster_searcher = ClusterSearcher.load()
for source_id, candidates in source_candidates.items():
    for cand in candidates[:3]:  # Top 3 semantic hits
        cluster_hits = cluster_searcher.search_by_path(cand["id"])
        if cluster_hits:
            # Add cluster members as additional target candidates
            for hit in cluster_hits:
                for member_path in hit.paths:
                    if member_path != cand["id"]:
                        candidates.append({
                            "id": member_path,
                            "score": hit.similarity_score * cand["score"],
                            "via_cluster": hit.label,
                            "documentation": f"Cluster member: {hit.description}",
                        })
```

**Prompt addition** to `signal_mapping.md`:

```markdown
### IMAS Cluster Candidates

Some semantic candidates below are cluster members — IMAS paths that store
the same physical parameter in different IDSs. When a source maps to one
member of a cluster, evaluate whether it should also map to other members.

Cluster members from different IDSs (e.g., `core_profiles/.../ip` and
`equilibrium/.../ip`) are valid one-to-many mappings if the source signal
genuinely represents that quantity. Set appropriate confidence — the primary
IDS target (within the current `{{ ids_name }}`) should have higher confidence
than cross-IDS targets.
```

**Note**: Cross-IDS mappings from clusters will naturally be limited to paths
within the target IDS for any single `imas map run` invocation. The cluster
context primarily helps when the same parameter appears in multiple _sections_
within the same IDS, or when a future multi-IDS mapping mode is added.

**Validation**:
- Run `imas map run --no-persist jet pf_active` before and after
- Verify that unmapped signals appear in output with dispositions
- Confirm that legitimate many-to-one patterns no longer produce false
  escalations
- Check that cluster candidates appear in semantic context
- Measure prompt token impact (expect modest increase from cluster context)

### Phase 2: Score-Dimension Filtering in Search Tools (Gap 2)

**Goal**: Enable filtering search results by score dimensions across all three
content search tools.

**Targets**:
- `imas_codex/llm/search_tools.py` — `_search_docs()`, `_search_code()`
- `imas_codex/llm/search_tools.py` — `_vector_search_wiki_chunks()`,
  `_vector_search_code_chunks()`
- `imas_codex/llm/server.py` — MCP tool registration docstrings

**Changes**:

1. **Add `physics_domain` parameter to `_search_docs()` and `_search_code()`**

   Both functions currently lack physics domain filtering. Add a `physics_domain`
   parameter that filters WikiPage/FacilityPath nodes by their `physics_domain`
   property, mirroring the existing pattern in `_vector_search_signals()`.

   For wiki: filter via `WikiChunk → WikiPage` traversal where
   `WikiPage.physics_domain = $domain`.

   For code: filter via `CodeChunk → SourceFile → FacilityPath` traversal where
   `FacilityPath.physics_domain = $domain`, or directly on
   `CodeChunk.facility_id` combined with path-level domain.

2. **Add `min_score` parameter with optional `score_dimension` selector**

   ```python
   def _search_docs(
       query: str,
       facility: str,
       *,
       physics_domain: str | None = None,   # NEW
       min_score: float | None = None,       # NEW
       score_dimension: str | None = None,   # NEW — defaults to score_composite
       ...
   )
   ```

   When `min_score` is set, filter results to only those where
   `node.{score_dimension} >= min_score`. When `score_dimension` is None,
   use `score_composite`. Valid dimensions come from
   `CONTENT_SCORE_DIMENSIONS` (docs) or `CODE_SCORE_DIMENSIONS` (code).

3. **Add the same parameters to `_search_signals()`**

   Signals don't have content scoring dimensions but do have a
   `score_composite`-equivalent via the SignalSource `status` and enrichment.
   The `physics_domain` filter already exists. Consider adding a `min_score`
   based on the vector similarity threshold.

**Validation**:
- Unit tests: parametrize across all 22 physics domains confirming filter works
- Integration test: `search_docs("coil current", "jet", physics_domain="magnetic_field_systems")`
  should return fewer, more relevant results than unfiltered

### Phase 3: Wiki and Code Context in Mapping Pipeline (Gap 1)

**Goal**: Enrich the LLM prompt context in `gather_context()` and
`map_signals()` with relevant wiki and code content, filtered by physics
domain and score dimensions.

**Targets**:
- `imas_codex/ids/tools.py` — new functions
- `imas_codex/ids/mapping.py` — `gather_context()`, `map_signals()`
- `imas_codex/llm/prompts/mapping/signal_mapping.md` — prompt template

**Changes**:

1. **New tool function: `fetch_wiki_context()`**

   ```python
   def fetch_wiki_context(
       facility: str,
       physics_domains: list[str],
       *,
       query: str | None = None,
       min_imas_relevance: float = 0.5,
       k: int = 10,
       gc: GraphClient | None = None,
   ) -> list[dict[str, Any]]:
       """Fetch wiki chunks relevant to the mapping task.

       Uses physics_domain + score_imas_relevance filtering to find
       high-value documentation, then optionally narrows with semantic search.
       """
   ```

   Implementation: Query WikiChunks where parent WikiPage has
   `physics_domain IN $domains AND score_imas_relevance >= $min_score`,
   optionally run vector search on `wiki_chunk_embedding` to rank by
   query relevance. Return chunk text + page title + scores.

2. **New tool function: `fetch_code_context()`**

   ```python
   def fetch_code_context(
       facility: str,
       physics_domains: list[str],
       *,
       query: str | None = None,
       score_dimension: str = "score_data_access",
       min_score: float = 0.5,
       k: int = 10,
       gc: GraphClient | None = None,
   ) -> list[dict[str, Any]]:
       """Fetch code chunks demonstrating data access patterns.

       Filters FacilityPaths by physics_domain and score_data_access,
       then retrieves their CodeChunks. This surfaces code that shows
       HOW signals are read, complementing the narrow
       fetch_source_code_refs() which only finds code for a single signal.
       """
   ```

3. **Integrate into `gather_context()`**

   After querying signal sources and before returning, call both new functions
   using `target_domains` from the physics domain scoping step:

   ```python
   wiki_context = fetch_wiki_context(
       facility, target_domains,
       min_imas_relevance=0.5, k=15, gc=gc,
   )
   code_context = fetch_code_context(
       facility, target_domains,
       score_dimension="score_data_access",
       min_score=0.5, k=15, gc=gc,
   )
   ```

   Add to the returned context dict as `"wiki_context"` and `"code_context"`.

4. **Add to `map_signals()` prompt**

   Pass wiki and code context to the `signal_mapping` prompt template:
   ```python
   prompt = _render_prompt(
       "signal_mapping",
       ...
       wiki_context=_format_wiki_context(context.get("wiki_context", [])),
       code_data_access=_format_code_context(context.get("code_context", [])),
   )
   ```

   Add corresponding sections to `signal_mapping.md`:
   ```markdown
   ### Domain Documentation

   Wiki documentation relevant to this physics domain (filtered by
   IMAS relevance score):

   {{ wiki_context }}

   ### Data Access Code Patterns

   Code examples showing how similar signals are accessed at this facility
   (filtered by data_access score):

   {{ code_data_access }}
   ```

5. **Per-source wiki/code narrowing** (optional for Phase 3)

   In `map_signals()` per-section loop, optionally run semantic search on wiki
   and code using the source description for per-source context, similar to
   the existing `source_candidates` pattern for IMAS paths. This is lower
   priority as domain-level context may suffice initially.

**Validation**:
- Run `imas map run --no-persist jet pf_active` before and after, compare
  prompt lengths and binding quality
- Check cost increase is acceptable (wiki/code context adds tokens)
- Verify no regression on existing binding quality

### Phase 4: Semantic Match Matrix (Cross-Domain Bridge) (Gap 3)

**Goal**: Implement a semantic bridge that computes cosine similarity between
source content embeddings and target IMAS node embeddings, returning a ranked
match matrix that the LLM uses as strong candidate signals.

**Targets**:
- `imas_codex/ids/tools.py` — new function `compute_semantic_matches()`
- `imas_codex/ids/mapping.py` — `gather_context()`, `map_signals()`
- `imas_codex/llm/prompts/mapping/signal_mapping.md`

**Changes**:

1. **New function: `compute_semantic_matches()`**

   ```python
   def compute_semantic_matches(
       source_descriptions: list[tuple[str, str]],  # (source_id, text)
       target_ids_name: str,
       *,
       gc: GraphClient | None = None,
       k_per_source: int = 5,
       include_wiki: bool = True,
       include_code: bool = True,
   ) -> dict[str, list[dict[str, Any]]]:
       """Compute semantic match vectors between sources and targets.

       For each source, embeds its description and searches against:
       1. imas_node_embedding (primary: target IMAS fields)
       2. wiki_chunk_embedding (bridging: domain documentation)
       3. code_chunk_embedding (bridging: data access patterns)

       Returns a dict mapping source_id → ranked match list, where each
       match has {target_id, score, content_type, excerpt}.
       """
   ```

   This function extends the existing per-source semantic search in
   `gather_context()` which currently only searches `imas_node_embedding`.
   The new version searches all three indexes and returns a unified match
   set per source.

2. **Batch embedding optimization**

   Instead of encoding one source at a time, batch all source descriptions
   and compute all embeddings in a single encoder call:

   ```python
   texts = [desc for _, desc in source_descriptions]
   embeddings = encoder.embed_texts(texts)
   ```

   Then run each embedding against the three vector indexes. This is
   significantly faster than the current one-at-a-time approach.

3. **Match matrix formatting for prompt**

   Format the top-k matches per source as a structured section:

   ```markdown
   ### Semantic Match Matrix

   For each source, the top matching IMAS fields, wiki excerpts, and code
   patterns are listed by cosine similarity:

   **Source: jet:magnetic_field_systems/pf_coil_current**
   - IMAS: pf_active/coil/current (0.922) — "Time-dependent electrical current..."
   - Wiki: "Magnetostatic Ontology" (0.865) — "PF coil current mapping to IMAS..."
   - Code: pfcoil.py:chunk_2 (0.877) — "def read_pf_current(...)"

   **Source: jet:magnetic_field_diagnostics/pf_v5-ipla**
   - IMAS: pf_active/supply/current (0.912) — "Output current of a PF supply..."
   ```

4. **Replace existing per-source semantic search**

   The current `source_candidates` computation in `gather_context()` loops
   over sources one at a time and only searches IMAS nodes. Replace this with
   `compute_semantic_matches()` which is batched and searches all indexes.

**Validation**:
- Compare cosine scores from the match matrix against actual binding targets
  in existing validated mappings
- Measure latency increase (batch embedding should be faster than one-at-a-time)
- Verify wiki/code bridge matches add value vs noise in LLM output

### Phase 5: Dynamic MCP Tool Docstrings (Gap 4)

**Goal**: Update MCP tool docstrings to include available physics domains and
score dimensions at registration time, so calling agents know what values are
valid without consulting external documentation.

**Targets**:
- `imas_codex/llm/server.py` — tool registrations for `search_signals`,
  `search_docs`, `search_code`
- `imas_codex/tools/graph_search.py` — `@mcp_tool` description strings

**Changes**:

1. **Dynamic docstring generation from schema enums**

   At server startup / tool registration time, import the `PhysicsDomain` enum
   and score dimension constants, and inject them into the tool descriptions:

   ```python
   from imas_codex.core.physics_domain import PhysicsDomain
   from imas_codex.discovery.base.scoring import (
       CODE_SCORE_DIMENSIONS, CONTENT_SCORE_DIMENSIONS
   )

   _PHYSICS_DOMAINS_DOC = ", ".join(sorted(d.value for d in PhysicsDomain))
   _CONTENT_SCORES_DOC = ", ".join(CONTENT_SCORE_DIMENSIONS)
   _CODE_SCORES_DOC = ", ".join(CODE_SCORE_DIMENSIONS)
   ```

2. **Update `search_signals` docstring** (codex server — `server.py`)

   ```python
   @self.mcp.tool()
   def search_signals(
       query: str,
       facility: str,
       physics_domain: str | None = None,
       ...
   ) -> str:
       f"""Search facility signals with full graph enrichment.
       ...
       Args:
           physics_domain: Filter by physics domain. Valid values:
               {_PHYSICS_DOMAINS_DOC}
       ...
       """
   ```

3. **Update `search_docs` docstring** with new `physics_domain` and
   `score_dimension` parameters

   ```python
   @self.mcp.tool()
   def search_docs(
       query: str,
       facility: str,
       physics_domain: str | None = None,     # NEW
       min_score: float | None = None,         # NEW
       score_dimension: str | None = None,     # NEW
       ...
   ) -> str:
       f"""Search documentation with physics domain and score filtering.
       ...
       Args:
           physics_domain: Filter by physics domain. Valid values:
               {_PHYSICS_DOMAINS_DOC}
           min_score: Minimum score threshold (0.0-1.0)
           score_dimension: Score dimension to filter on. Valid values:
               {_CONTENT_SCORES_DOC}
               Defaults to score_composite.
       ...
       """
   ```

4. **Update `search_code` docstring** similarly

   Valid score dimensions for code are different from content:
   `{_CODE_SCORES_DOC}`.

5. **Update graph-native MCP `@mcp_tool` descriptions**

   In `imas_codex/tools/graph_search.py`, the `@mcp_tool` decorator takes a
   static description string. These should also enumerate valid physics domains.

**Validation**:
- Call `get_graph_schema()` or inspect tool descriptions to verify domains and
  dimensions are listed
- Verify docstrings update when new domains are added to the schema (rebuild
  models → reimport)

## Score Dimension Usage Recommendations

Based on analysis of live graph data, the following score dimensions are
recommended for specific mapping pipeline contexts:

### For `map_signals()` — Signal-Level Mapping

| Context Need | Primary Dimension | Threshold | Rationale |
|-------------|-------------------|-----------|-----------|
| Data access patterns | `score_data_access` | ≥0.5 | Surfaces MDSplus paths, TDI expressions, API patterns |
| IMAS integration | `score_imas` / `score_imas_relevance` | ≥0.5 | Content that explicitly discusses IMAS/IDS mapping |
| Convention handling | `score_convention` | ≥0.3 | COCOS, sign flips, unit conversions |
| Calibration context | `score_calibration` | ≥0.3 | Conversion factors, sensor specifications |

### For `assign_sections()` — Section Assignment

| Context Need | Primary Dimension | Threshold | Rationale |
|-------------|-------------------|-----------|-----------|
| Physics explanations | `score_physics_content` | ≥0.5 | Understanding what each section represents |
| Data documentation | `score_data_documentation` | ≥0.5 | Signal tables mapping signals to IDS |

### For `discover_assembly()` — Assembly Pattern Discovery

| Context Need | Primary Dimension | Threshold | Rationale |
|-------------|-------------------|-----------|-----------|
| Data access code | `score_data_access` | ≥0.6 | How data is structured in source systems |
| Modeling code | `score_modeling_code` | ≥0.5 | How simulations assemble coil/circuit data |

## PhysicsDomain Enum Values (for reference)

The `PhysicsDomain` enum (22 values) from `physics_domains.yaml`:

**Core Plasma Physics**: `equilibrium`, `transport`, `magnetohydrodynamics`, `turbulence`  
**Heating & Current Drive**: `auxiliary_heating`, `current_drive`  
**Plasma-Material Interactions**: `plasma_wall_interactions`, `divertor_physics`, `edge_plasma_physics`  
**Diagnostics**: `particle_measurement_diagnostics`, `electromagnetic_wave_diagnostics`, `radiation_measurement_diagnostics`, `magnetic_field_diagnostics`, `mechanical_measurement_diagnostics`  
**Control & Operations**: `plasma_control`, `machine_operations`  
**Engineering Systems**: `magnetic_field_systems`, `structural_components`, `plant_systems`  
**Data & Workflow**: `data_management`, `computational_workflow`  
**Fallback**: `general`

## Implementation Order & Dependencies

```
Phase 1 (mapping fidelity: models + prompts + schema)
    ↓
Phase 2 (search tool score filters)
    ↓
Phase 3 (pipeline wiki/code context)       ←— depends on Phase 2 tools
    ↓
Phase 4 (semantic match matrix)            ←— depends on Phase 3 context model
    ↓
Phase 5 (dynamic docstrings)               ←— can run in parallel with 2-4
```

Phase 1 is foundational — it defines the output model contract that all
subsequent phases build on. Phase 5 is independent and can be implemented
alongside any other phase.

## Risk Assessment

- **Token budget**: Adding wiki + code context to prompts increases token
  count. Mitigated by: physics domain filtering (reduces to ~5-10% of total),
  score dimension thresholds (further reduces), and k-limit parameters.
- **Latency**: Semantic bridge requires N × 3 vector index queries (one per
  source × 3 indexes). Mitigated by: batch embedding, parallelized queries.
- **False positives**: Low-quality wiki/code matches could mislead the LLM.
  Mitigated by: score thresholds, cosine similarity cutoffs, and clear prompt
  framing that marks these as "supporting evidence, not authoritative".
- **Over-rejection**: Instructing the LLM not to force mappings could lead to
  too many `unmapped` signals. Mitigated by: requiring evidence for rejection
  (not just low confidence), and tracking unmapped rates per facility/IDS to
  detect regression. Validate against known-good mappings.
- **Cluster noise in one-to-many**: Cluster members from unrelated IDSs could
  distract the LLM. Mitigated by: filtering cluster candidates to the target
  IDS (within-IDS clusters only for Phase 1), scoring cluster candidates by
  the product of source-to-primary similarity and cluster coherence, and
  capping cluster-derived candidates per source.

## Files Modified Per Phase

### Phase 1
- `imas_codex/ids/models.py` — `MappingDisposition`, `UnmappedSignal`, updated `SignalMappingBatch`, `UnassignedSource`, `ValidatedSignalMapping` with disposition/evidence
- `imas_codex/llm/prompts/mapping/signal_mapping.md` — no-match, many-to-one, cluster prompts
- `imas_codex/llm/prompts/mapping/section_assignment.md` — unassigned reasoning
- `imas_codex/schemas/facility.yaml` — `MappingDispositionEnum`, `mapping_disposition`, `mapping_evidence` on SignalSource, `evidence` on MAPS_TO_IMAS
- `imas_codex/ids/mapping.py` — persist unmapped signals, pass cluster context
- `imas_codex/ids/validation.py` — classify many-to-one patterns
- `imas_codex/ids/tools.py` — cluster candidate lookup in `gather_context()`
- `tests/ids/` — tests for new models, unmapped persistence, many-to-one classification

### Phase 2
- `imas_codex/llm/search_tools.py` — add physics_domain and score_dimension params
- `imas_codex/llm/server.py` — update tool registration signatures
- `tests/` — new parametrized tests for filtered search

### Phase 3
- `imas_codex/ids/tools.py` — new `fetch_wiki_context()`, `fetch_code_context()`
- `imas_codex/ids/mapping.py` — `gather_context()`, `map_signals()` integration
- `imas_codex/llm/prompts/mapping/signal_mapping.md` — new template sections
- `tests/ids/` — tests for new tool functions

### Phase 4
- `imas_codex/ids/tools.py` — new `compute_semantic_matches()`
- `imas_codex/ids/mapping.py` — replace `source_candidates` with match matrix
- `imas_codex/llm/prompts/mapping/signal_mapping.md` — match matrix section
- `tests/ids/` — tests for semantic matching

### Phase 5
- `imas_codex/tools/utils.py` — `physics_domain_doc()`, `score_dimensions_doc()`
- `imas_codex/llm/server.py` — dynamic docstrings in tool registration
- `imas_codex/tools/graph_search.py` — dynamic `@mcp_tool` descriptions
- `tests/tools/` — docstring content validation
