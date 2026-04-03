# Embedding Text Optimization for IMAS DD Search

## Problem Statement

The graph-backed IMAS DD search achieves MRR 0.489 on 50 benchmark queries
after scoring and fusion tuning. Parameter sweeps show the **ceiling is set
by embedding quality**, not scoring — boost parameters have ±0.02 MRR impact
while vector-only MRR is 0.384. The root cause is severely under-utilized
embeddings:

- Average embed text: **76 chars (~19 tokens)** — using **0.058% of the
  Qwen3-Embedding-0.6B model's 32K token capacity**
- Descriptions average 61 chars (44% under 50 chars) due to 150-char prompt limit
- **No physics abbreviations** (Ip, Te, ne, q, ψ) in any descriptions
- **Full IMAS path not included** — terminal segments like `ip`, `b0` missing
  from embed space
- **Keywords** (2.8 per node avg) stored but **never embedded**
- **Documentation** (often more detailed) **never embedded**
- **No instruction-aware prefix** despite Qwen3 native support for `prompt_name`

### Current Baselines

| Metric | Value |
|--------|-------|
| Overall MRR | 0.489 |
| Abbreviation MRR | 0.318 |
| Vector-only MRR | 0.384 |
| Avg embed text length | 76 chars |
| Model capacity utilization | 0.058% |

### Target

| Metric | Target |
|--------|--------|
| Overall MRR | ≥ 0.65 |
| Abbreviation MRR | ≥ 0.55 |
| Avg embed text length | ~250-400 chars |
| Model capacity utilization | ~0.3-0.5% |

---

## Architecture Overview

Five code-change phases execute in parallel, then a single DD rebuild pass
(re-enrich → re-embed) applies all changes at once. The rebuild is Phase 6.

```
Phase 1 ──┐
Phase 2 ──┤
Phase 3 ──┼── All code committed ──► Phase 6: DD Rebuild (serial)
Phase 4 ──┤                           ├── reset-to enriched
Phase 5 ──┘                           ├── re-enrich (--force)
                                      ├── re-embed (auto)
                                      └── benchmark evaluation
```

---

## Phase 1: Enrich Embed Text with Path, Documentation, and Keywords

**Goal:** Transform `generate_embedding_text()` from a 76-char minimal string
to a ~250-400 char rich text that includes the full IMAS path, description,
documentation, and keywords — still under 0.5% of model capacity.

**Agent:** engineer

### Files to Modify

**`imas_codex/graph/build_dd.py`** — `generate_embedding_text()` (lines 143-174)

Current:
```python
def generate_embedding_text(path, path_info, ids_info=None):
    desc = (path_info.get("description") or "").strip()
    text = desc if desc else (path_info.get("documentation") or "").strip()
    if not text:
        return text
    ids_name = path.split("/")[0] if "/" in path else ""
    if ids_name:
        readable_ids = ids_name.replace("_", " ")
        return f"{readable_ids}: {text}"
    return text
```

New:
```python
def generate_embedding_text(path, path_info, ids_info=None):
    """Generate rich embedding text for an IMAS DD node.

    Includes the full IMAS path (for terminal segment matching like
    'ip', 'b0', 'q'), the enriched description, documentation excerpt,
    and keywords. Targets ~250-400 chars to improve vector search recall
    while staying well under the Qwen3 model's 32K token limit.
    """
    desc = (path_info.get("description") or "").strip()
    doc = (path_info.get("documentation") or "").strip()
    keywords = path_info.get("keywords") or []

    # Primary text: prefer enriched description, fall back to documentation
    primary = desc if desc else doc
    if not primary:
        return ""

    parts: list[str] = []

    # 1. Full IMAS path — injects terminal segments (ip, b0, psi, t_e)
    #    and hierarchy into the embedding space
    parts.append(path)

    # 2. Primary description
    parts.append(primary)

    # 3. Documentation excerpt (if different from description and adds info)
    if doc and doc != desc and len(doc) > 20:
        # Cap at 300 chars to avoid noise from very long documentation
        parts.append(doc[:300])

    # 4. Keywords — searchable terms not in description or path
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")

    return ". ".join(parts)
```

### Hash Invalidation

The `compute_embedding_hash()` function hashes `f"{model_name}:{text}"`. Since
`generate_embedding_text()` now produces different text for every node, all
20K embedding hashes will be stale. The embed phase will automatically
regenerate all embeddings — no special migration needed.

### Tests to Add/Update

**`tests/graph/test_build_dd.py`** (or new file `tests/graph/test_embedding_text.py`):
- Test that `generate_embedding_text()` includes full path in output
- Test that documentation is included when different from description
- Test that keywords are appended
- Test that empty description/documentation returns empty string
- Test that very long documentation is capped at 300 chars

### Acceptance Criteria

- [ ] `generate_embedding_text()` returns text containing the full IMAS path
- [ ] Documentation included when available and different from description
- [ ] Keywords appended as "Keywords: k1, k2, ..."
- [ ] Empty inputs return empty string (no crash)
- [ ] All existing tests pass
- [ ] Unit tests for the new function behavior

---

## Phase 2: Add Instruction-Aware Embedding (query-side `prompt_name`)

**Goal:** Use Qwen3-Embedding's native instruction-aware encoding for
query embeddings. SentenceTransformer supports `prompt_name` parameter
which prepends a task instruction to queries. Documents don't need
instructions (asymmetric retrieval). Expected 1-5% MRR improvement.

**Agent:** engineer

### Files to Modify

**`imas_codex/embeddings/encoder.py`** — `embed_texts()` and `embed_texts_with_result()`

Add a `prompt_name` parameter that passes through to `model.encode()`:

```python
def embed_texts(self, texts: list[str], *, prompt_name: str | None = None, **kwargs) -> np.ndarray:
    # ... existing code ...
    # Local backend
    model = self.get_model()
    encode_kwargs = {
        "convert_to_numpy": True,
        "normalize_embeddings": self.config.normalize_embeddings,
        "batch_size": self.config.batch_size,
        "show_progress_bar": False,
        **kwargs,
    }
    if prompt_name:
        encode_kwargs["prompt_name"] = prompt_name
    return model.encode(texts, **encode_kwargs)
```

**`imas_codex/embeddings/config.py`** (or wherever EncoderConfig lives)

Add prompts configuration to the encoder initialization:

```python
# After model loading, register query/document prompts
model.prompts = {
    "query": "Instruct: Given a fusion physics search query, retrieve the most relevant IMAS Data Dictionary path.\nQuery: ",
}
# No document prompt — asymmetric retrieval (documents embedded raw)
```

**`imas_codex/tools/graph_search.py`** — Both `_embed_query()` methods

```python
def _embed_query(self, query: str) -> list[float]:
    """Embed query text with instruction prefix for retrieval."""
    encoder = _get_encoder()
    return encoder.embed_texts([query], prompt_name="query")[0].tolist()
```

**`imas_codex/embeddings/server.py`** — `EmbedRequest` and `_encode_texts_sync()`

Add `prompt_name` to the remote embedding API:

```python
class EmbedRequest(BaseModel):
    texts: list[str] = Field(...)
    normalize: bool = Field(True, ...)
    dimension: int | None = Field(None, ...)
    prompt_name: str | None = Field(
        None,
        description="Named prompt for instruction-aware models (e.g., 'query'). "
        "Applied only for local encoding.",
    )
```

Pass through to `_encode_texts_sync()` and the encoder.

**`imas_codex/embeddings/client.py`** — `RemoteEmbeddingClient.embed()`

Add `prompt_name` parameter to client that passes it in the HTTP request body.

### Important Design Decision

The instruction prefix is **query-side only**. Document embeddings (DD build)
do NOT use any prompt — this is asymmetric retrieval as recommended by Qwen3
documentation. The query prompt tells the model "this is a search query,
find matching documents" which guides the embedding direction.

### Fallback Behavior

If the model doesn't support `prompt_name` (e.g., older sentence-transformers),
the parameter is silently passed through and may be ignored by `encode()`.
Since `**kwargs` already forwards to `model.encode()`, this is backwards
compatible.

### Tests to Add/Update

- Test that `embed_texts(prompt_name="query")` produces different embeddings
  than `embed_texts()` (if model supports prompts)
- Test that the remote server accepts `prompt_name` in the request body
- Test that `_embed_query()` passes `prompt_name="query"` to the encoder

### Acceptance Criteria

- [ ] `Encoder.embed_texts()` accepts and forwards `prompt_name` parameter
- [ ] Query prompt registered on model during initialization
- [ ] Both `_embed_query()` instances in `graph_search.py` pass `prompt_name="query"`
- [ ] Remote server and client support `prompt_name` pass-through
- [ ] All existing tests pass (backwards compatible)

---

## Phase 3: Update Enrichment Prompt — Abbreviations and Longer Descriptions

**Goal:** Update the LLM enrichment prompt to require physics abbreviations
in descriptions and allow 200-400 characters instead of the current 150-char
limit. Also increase keyword count from 5 to 8 and require abbreviation
symbols in keywords.

**Agent:** engineer

### Files to Modify

**`imas_codex/llm/prompts/imas/enrichment.md`**

Key changes to the prompt:

1. **Lift character limit**: "under 150 characters" → "200–400 characters"
2. **Require abbreviations**: Add instruction to include standard physics
   abbreviations in parentheses after first mention
3. **Update examples**: Replace 46-69 char examples with 150-250 char examples
   that include abbreviations
4. **Increase keyword count**: "Up to 5" → "Up to 8" and require abbreviation
   symbols

New task section:
```markdown
## Task

Write a **physics-aware description** (2–4 sentences, **200–400 characters**)
for each IMAS Data Dictionary path. The description must name the physical
quantity with its standard abbreviation, and explain what distinguishes this
node from similar nodes elsewhere in the DD.

For each path in the batch, provide:

1. **description**: A physics-aware description (200–400 characters) that:
   - Names the physical quantity AND its standard abbreviation/symbol in
     parentheses (e.g., "electron temperature (Te)", "plasma current (Ip)")
   - Explains what makes this specific path unique (IDS context, dimensionality)
   - Mentions related physics concepts users might search for

2. **keywords**: Up to 8 searchable terms — including:
   - The standard abbreviation/symbol (e.g., "Ip", "Te", "q", "ψ")
   - Physics concepts, measurement types, diagnostic names
   - Common alternative names NOT already in the description or path

3. **physics_domain**: Primary physics domain ONLY if clearly different from
   the IDS-level domain. Use null to inherit.
```

Add new section after "Critical Guidelines":

```markdown
### Include Standard Physics Abbreviations

If the physical quantity has a standard abbreviation or symbol in fusion
physics, include it in parentheses after the first mention. These are
critical for search — users frequently search using abbreviations.

Common abbreviations (include when relevant):

| Full Name | Abbreviation/Symbol |
|-----------|-------------------|
| Plasma current | Ip |
| Electron temperature | Te |
| Ion temperature | Ti |
| Electron density | ne |
| Safety factor | q |
| Poloidal flux | ψ, psi |
| Toroidal magnetic field | Bt, B0 |
| Poloidal beta | βp, beta_pol |
| Effective charge | Zeff |
| Loop voltage | Vloop |
| Stored energy | Wmhd |
| Major radius | R, R0 |
| Minor radius | a |
| Elongation | κ, kappa |
| Triangularity | δ, delta |
| Internal inductance | li |
| Normalized beta | βN, beta_N |
| Bootstrap current | Ibs |
| Ohmic current | Ioh |
| Resistivity | η, eta |
| Collisionality | ν*, nu_star |
| Greenwald density | nGW |

Also include the **terminal path segment name** if it uses a known
abbreviation (e.g., for path ending in `/ip`, mention "Ip" explicitly).
```

Update examples:
```markdown
### Examples

**GOOD** (rich, includes abbreviations, searchable):

- `"Total plasma current (Ip), the toroidal component of current flowing
   through the plasma. Sign follows ip_like COCOS convention. Primary global
   parameter for plasma stability and confinement."` (195 chars)
- `"Electron temperature (Te) radial profile from core profile analysis.
   Thermal energy of the electron population as a function of the normalized
   toroidal flux coordinate rho_tor_norm."` (183 chars)
- `"Safety factor (q) profile from equilibrium reconstruction. Measures the
   field line helicity — ratio of toroidal to poloidal magnetic flux. Key
   indicator for MHD stability limits."` (178 chars)

**BAD** (too short, no abbreviations):

- `"Electron temperature radial profile from core profile analysis."` (63 chars) — no abbreviation, too short
- `"Total plasma current."` (22 chars) — no abbreviation, no context
- `"Safety factor profile from equilibrium."` (40 chars) — missing physics context
```

**`imas_codex/graph/dd_enrichment.py`** — `IMASPathEnrichmentResult` Pydantic model

Update the field description and constraint:
```python
class IMASPathEnrichmentResult(BaseModel):
    path_index: int = Field(...)
    description: str = Field(
        description=(
            "Physics-aware description (2-4 sentences, 200-400 characters) "
            "that names the physical quantity with its standard abbreviation "
            "and what distinguishes this node. "
            "Do NOT repeat units, data type, or coordinate information."
        )
    )
    keywords: list[str] = Field(
        default_factory=list,
        max_length=8,  # Was 5
        description=(
            "Searchable keywords (up to 8) — physics abbreviations/symbols, "
            "concepts, measurement types, diagnostic names, and related "
            "terms not already in the description or path name"
        ),
    )
```

### Hash Invalidation

The enrichment hash is computed from `f"{path_id}:{documentation}:{siblings}"` +
model name. Changing the LLM model name or using `--force` will invalidate all
hashes. The DD rebuild in Phase 6 will use `--force` to re-enrich everything.

### Tests to Add/Update

- Prompt rendering test: verify the new prompt renders correctly with schema includes
- Validate that `IMASPathEnrichmentResult` accepts 200-400 char descriptions
- Validate that `max_length=8` for keywords is enforced

### Acceptance Criteria

- [ ] Prompt updated with abbreviation table and 200-400 char target
- [ ] Examples in prompt are 150-250 chars with abbreviations
- [ ] Pydantic model accepts longer descriptions and 8 keywords
- [ ] Prompt renders correctly (test with `render_prompt()`)
- [ ] All existing tests pass

---

## Phase 4: Expand Physics Abbreviation Mappings

**Goal:** Expand the `PHYSICS_ABBREVIATIONS` dict in `query_analysis.py` from
19 entries to ~35+ entries covering more physics terms that appear as IMAS
path terminals. Also add reverse mappings (full name → abbreviation) for
use in the enrichment prompt's abbreviation table.

**Agent:** engineer

### Files to Modify

**`imas_codex/tools/query_analysis.py`** — `PHYSICS_ABBREVIATIONS` dict

Add missing abbreviations that appear as IMAS path terminals:

```python
PHYSICS_ABBREVIATIONS: dict[str, list[str]] = {
    # Existing entries (keep all)
    "ip": ["plasma current", "ip"],
    "te": ["electron temperature", "te"],
    "ti": ["ion temperature", "ti"],
    "ne": ["electron density", "ne"],
    "ni": ["ion density", "ni"],
    "bt": ["toroidal magnetic field", "b_field_tor", "bt", "b0"],
    "bp": ["poloidal magnetic field", "b_field_pol", "bp"],
    "q": ["safety factor", "q"],
    "psi": ["poloidal flux", "psi"],
    "beta": ["plasma beta", "beta_pol", "beta_tor", "beta_normal", "beta"],
    "li": ["internal inductance", "li"],
    "wmhd": ["stored energy", "w_mhd", "wmhd"],
    "zeff": ["effective charge", "z_eff", "zeff"],
    "vloop": ["loop voltage", "v_loop", "vloop"],
    "bpol": ["poloidal magnetic field", "b_field_pol", "bpol"],
    "btor": ["toroidal magnetic field", "b_field_tor", "btor"],
    "te0": ["central electron temperature", "te", "te0"],
    "ti0": ["central ion temperature", "ti", "ti0"],
    "ne0": ["central electron density", "ne", "ne0"],
    # --- New entries ---
    "b0": ["vacuum toroidal field", "toroidal magnetic field", "b0", "bt"],
    "r0": ["reference major radius", "magnetic axis radius", "r0"],
    "a": ["minor radius", "plasma minor radius", "a"],
    "kappa": ["elongation", "plasma elongation", "kappa"],
    "delta": ["triangularity", "plasma triangularity", "delta"],
    "rho": ["normalized radius", "rho_tor_norm", "rho"],
    "phi": ["toroidal flux", "toroidal angle", "phi"],
    "theta": ["poloidal angle", "geometric angle", "theta"],
    "j_tor": ["toroidal current density", "j_tor"],
    "j_parallel": ["parallel current density", "j_parallel"],
    "nbi": ["neutral beam injection", "beam heating", "nbi"],
    "ecrh": ["electron cyclotron heating", "ech", "ecrh"],
    "icrh": ["ion cyclotron heating", "icrf", "icrh"],
    "lh": ["lower hybrid", "lower hybrid heating", "lh"],
    "ece": ["electron cyclotron emission", "ece"],
    "mse": ["motional stark effect", "mse"],
    "bol": ["bolometry", "radiated power", "bol"],
    "sxr": ["soft x-ray", "sxr"],
    "p_ohm": ["ohmic power", "p_ohmic", "p_ohm"],
    "tau_e": ["energy confinement time", "tau_e"],
    "n_e_line": ["line-integrated density", "nel", "n_e_line"],
}
```

### Tests to Add/Update

**`tests/search/test_search_strategy.py`** or similar:
- Test that new abbreviations are recognized by `QueryAnalyzer`
- Test that `analyze("b0")` returns expanded terms including "vacuum toroidal field"
- Test that `analyze("nbi")` returns expanded terms including "neutral beam injection"

### Acceptance Criteria

- [ ] `PHYSICS_ABBREVIATIONS` expanded to ~35+ entries
- [ ] All new entries produce valid `QueryIntent` with `is_abbreviation=True`
- [ ] No duplicate keys (key collision check)
- [ ] All existing tests pass
- [ ] Unit tests for new abbreviation expansions

---

## Phase 5: Update Regression Thresholds and Benchmark Infrastructure

**Goal:** After the DD rebuild (Phase 6), MRR should significantly improve.
Prepare the benchmark infrastructure for measuring the improvement and update
the regression thresholds. Also add new benchmark queries targeting the
specific improvements (abbreviation-in-embed, path-terminal matching).

**Agent:** engineer

### Files to Modify

**`tests/search/benchmark_data.py`** — Add targeted benchmark queries

Add new queries that specifically test the improvements:

```python
# New queries that should improve with embed text optimization
BenchmarkQuery(
    query_text="ip",
    expected_paths=["equilibrium/time_slice/global_quantities/ip", "magnetics/ip"],
    category="abbreviation",
),
BenchmarkQuery(
    query_text="B0",
    expected_paths=["equilibrium/vacuum_toroidal_field/b0"],
    category="abbreviation",
),
# ... (verify existing abbreviation queries cover the improvements)
```

**`tests/search/test_search_benchmarks.py`** — Prepare threshold placeholders

The current thresholds (MRR ≥ 0.40, Abbreviation MRR ≥ 0.25) were empirically
set from the pre-optimization baseline. After the DD rebuild, run the full
benchmark suite and update:

```python
class TestSearchQualityGate:
    # These thresholds will be updated after Phase 6 DD rebuild
    # Pre-optimization: MRR=0.489, Abbr=0.318
    # Target post-optimization: MRR≥0.65, Abbr≥0.55
    MRR_THRESHOLD = 0.40        # Keep conservative until rebuild validates
    ABBREVIATION_MRR_THRESHOLD = 0.25  # Keep conservative until rebuild validates
```

**`tests/search/test_search_evaluation.py`** — Add embed text quality metrics

Add a new evaluation that measures embed text richness:

```python
class TestEmbedTextQuality:
    """Verify embed text includes path, abbreviations, and keywords."""

    async def test_embed_text_contains_full_path(self, graph_client):
        """Embed text should contain the full IMAS path."""
        result = graph_client.query("""
            MATCH (p:IMASNode {id: 'equilibrium/time_slice/global_quantities/ip'})
            RETURN p.embedding_text AS text
        """)
        assert "equilibrium/time_slice/global_quantities/ip" in result[0]["text"]

    async def test_embed_text_contains_keywords(self, graph_client):
        """Embed text should contain keywords."""
        result = graph_client.query("""
            MATCH (p:IMASNode)
            WHERE p.keywords IS NOT NULL AND size(p.keywords) > 0
            RETURN p.embedding_text AS text, p.keywords AS kw
            LIMIT 5
        """)
        for r in result:
            assert "Keywords:" in r["text"]
```

### Acceptance Criteria

- [ ] New benchmark queries added for path-terminal and abbreviation matching
- [ ] Embed text quality tests added (path inclusion, keyword inclusion)
- [ ] Regression thresholds documented as pending update after Phase 6
- [ ] All existing tests pass

---

## Phase 6: DD Rebuild — Re-Enrich and Re-Embed (Serial, Post-Merge)

**Goal:** Execute a single DD rebuild pass that re-enriches all 20K nodes
with the updated prompt and re-embeds with the new `generate_embedding_text()`
function. This is the only phase that touches the live graph.

**Agent:** human-supervised (run manually after code merge)

### Prerequisites

- All Phase 1-5 code changes merged to main
- Embedding server running (`imas-codex embed status`)
- Neo4j graph running (`imas-codex graph status`)
- LLM proxy configured (`get_model("language")` resolves)

### Execution Plan

```bash
# Step 1: Verify all code changes are merged
git pull --no-rebase origin main

# Step 2: Rebuild models (schema may have changed)
uv run build-models --force

# Step 3: Run the DD rebuild with force re-enrichment
# --reset-to enriched: clears embeddings but keeps extracted data
# --force: re-enrich all paths regardless of hash (new prompt)
uv run imas-codex imas dd build --force

# This will:
#   1. Re-enrich all 20K paths with the new prompt (abbreviations + longer descriptions)
#   2. Re-embed all paths with the new generate_embedding_text() function
#   3. Re-cluster with updated embeddings
#   4. Re-compute cluster centroids
```

### Cost Estimate

| Component | Estimate |
|-----------|----------|
| LLM re-enrichment (20K nodes × ~$0.001/node) | ~$20-40 |
| GPU re-embedding (20K nodes, local) | ~5 minutes |
| Cluster recomputation | ~2 minutes |
| Total wall time | ~60-90 minutes (LLM-dominated) |

### Post-Rebuild Validation

```bash
# Step 4: Verify embed text quality
uv run python -c "
from imas_codex.graph.client import GraphClient
gc = GraphClient()
r = list(gc.query('''
    MATCH (p:IMASNode {id: \"equilibrium/time_slice/global_quantities/ip\"})
    RETURN p.embedding_text AS text, p.description AS desc, p.keywords AS kw
'''))
print(f'Embed text: {r[0][\"text\"]}')
print(f'Description: {r[0][\"desc\"]}')
print(f'Keywords: {r[0][\"kw\"]}')
gc.close()
"

# Step 5: Run the benchmark suite
uv run pytest tests/search/test_search_benchmarks.py -v --tb=short

# Step 6: Run the full DoE evaluation
uv run pytest tests/search/test_search_evaluation.py -v -k "test_evaluate" --tb=short

# Step 7: If MRR improved, update regression thresholds
# Edit tests/search/test_search_benchmarks.py with new empirical thresholds
```

### Rollback Plan

If MRR drops (shouldn't, but just in case):
```bash
# Pull the previous graph export
uv run imas-codex graph pull --tag <previous-tag>
# Or restore from backup
uv run imas-codex graph restore
```

### Acceptance Criteria

- [ ] All 20K paths re-enriched with new descriptions containing abbreviations
- [ ] Embed text for `ip` node contains full path + "Ip" abbreviation
- [ ] Average embed text length > 200 chars (up from 76)
- [ ] Overall MRR ≥ 0.60 (up from 0.489)
- [ ] Abbreviation MRR ≥ 0.45 (up from 0.318)
- [ ] Regression thresholds updated to reflect new baseline
- [ ] All tests pass

---

## Parallelization Strategy

```
┌─────────────────────────────────────────────────────┐
│ Parallel Code Changes (no graph writes)             │
│                                                     │
│  Agent 1: Phase 1 (generate_embedding_text)         │
│  Agent 2: Phase 2 (instruction-aware encoding)      │
│  Agent 3: Phase 3 (enrichment prompt + Pydantic)    │
│  Agent 4: Phase 4 (abbreviation expansions)         │
│  Agent 5: Phase 5 (benchmark infrastructure)        │
│                                                     │
│  All agents commit to main independently            │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ Serial: Phase 6 — DD Rebuild                        │
│                                                     │
│  1. Pull all changes                                │
│  2. uv run imas-codex imas dd build --force         │
│  3. Run benchmarks                                  │
│  4. Update regression thresholds                    │
│  5. Commit threshold updates                        │
└─────────────────────────────────────────────────────┘
```

### File Conflict Avoidance

Each phase touches disjoint files:

| Phase | Primary Files | Conflict Risk |
|-------|---------------|---------------|
| 1 | `build_dd.py` (generate_embedding_text), new test file | None |
| 2 | `encoder.py`, `server.py`, `client.py`, `graph_search.py` (_embed_query only) | Low (graph_search.py) |
| 3 | `enrichment.md`, `dd_enrichment.py` (Pydantic model only) | None |
| 4 | `query_analysis.py` | None |
| 5 | `benchmark_data.py`, `test_search_benchmarks.py`, `test_search_evaluation.py` | None |

**Note on Phase 2 + graph_search.py**: Phase 2 modifies only the `_embed_query()`
methods (2 instances, ~4 lines each). No other phase touches these methods.
Phase 5 may modify the same test files but different classes/functions.

---

## Documentation Updates

After Phase 6 completes:

| Target | Update |
|--------|--------|
| `AGENTS.md` | Update embedding text format description if referenced |
| `plans/README.md` | Move this plan to completed/delete |
| Research report | Archive in session workspace |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| LLM produces worse descriptions | Low | Medium | Compare samples before full rebuild |
| Instruction prefix hurts instead of helps | Low | Low | Can disable with one-line change |
| Longer embed text slows encoding | Very Low | Low | 400 chars = ~100 tokens, negligible |
| DD rebuild fails mid-way | Low | Medium | Idempotent — resume with same command |
| MRR doesn't improve as expected | Medium | Low | Keep conservative regression thresholds |
