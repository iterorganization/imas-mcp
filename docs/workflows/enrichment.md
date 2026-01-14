# TreeNode LLM Enrichment Plan

> Using Gemini Flash to generate metadata and discover graph relationships for MDSplus TreeNodes.

## Overview

We have ~17,000 TreeNodes with missing descriptions and physics domain assignments. This plan describes a two-phase enrichment approach using LLM inference to:

1. **Generate metadata** - descriptions, units validation, physics domain classification
2. **Discover graph links** - IMAS path mappings, diagnostic relationships, analysis code connections

## Current State

| Metric | Value |
|--------|-------|
| Total epoch-aware TreeNodes | 17,458 |
| Missing description | 17,369 |
| Missing physics_domain | 17,386 |
| Nodes with code references | 663 |
| Nodes with path only | ~16,700 |

## Phase 1: Metadata Enrichment

### Approach

1. **Context-aware batching**: Group nodes by tree/subtree for coherent context
2. **Code example injection**: For nodes with `DataReference` links, include code snippets
3. **SSH introspection**: LLM can request MDSplus metadata via SSH when uncertain
4. **Confidence filtering**: Only store descriptions with medium+ confidence

### Node Categories

| Category | Count | Strategy |
|----------|-------|----------|
| **High context** | ~663 | Code examples + path analysis |
| **Self-documenting** | ~5,000 | Path name parsing (e.g., `I_PL` → "Plasma current") |
| **Metadata nodes** | ~3,000 | Skip (`:IGNORE`, `:FOO`, `:VERSION_NUM`, etc.) |
| **Uncertain** | ~8,000 | SSH query or mark as low confidence |

### Skip Patterns (metadata/internal nodes)

```
:IGNORE, :FOO, :BAR, :VERSION_NUM, :COMMENT, :ERROR_BAR, 
:UNITS, :CONFIDENCE, :TRIAL, :USER_NAME, :TIME_INDEX
```

### LLM Prompt Structure

```
You are enriching MDSplus TreeNode metadata for the TCV tokamak.

For each path, provide:
- description: 1-2 sentence physics description (or null if uncertain)
- physics_domain: one of [equilibrium, magnetics, heating, diagnostics, transport, mhd, control, machine]
- confidence: high|medium|low
- suggested_units: if units appear wrong (optional)

Context available:
- Tree: {tree_name}
- Subtree: {subtree}
- Code examples using this path (if any)
- MDSplus node type and current units

Paths to describe:
{batch of 50 paths with available context}
```

### SSH Introspection Capability

When the LLM is uncertain, it can request:
```python
# Get node metadata from MDSplus
ssh epfl "python3 -c \"
import MDSplus
tree = MDSplus.Tree('results', 80000)
node = tree.getNode('\\\\RESULTS::TOP.EQUIL_1.RESULTS:I_PL')
print('usage:', node.usage)
print('description:', node.description)
print('units:', node.units)
\""
```

### Output Schema

```json
{
  "path": "\\RESULTS::TOP.EQUIL_1.RESULTS:I_PL",
  "description": "Plasma current from LIUQE equilibrium reconstruction",
  "physics_domain": "equilibrium",
  "confidence": "high",
  "suggested_units": null,
  "source": "code_context"  // or "path_analysis" or "ssh_query"
}
```

## Phase 2: Graph Link Enhancement

After metadata enrichment, discover and create new graph relationships.

### 2.1 IMAS Path Mapping

Map TreeNodes to IMAS DD paths based on physics semantics:

| TreeNode | IMAS Path | Mapping Type |
|----------|-----------|--------------|
| `I_PL` | `equilibrium/time_slice/global_quantities/ip` | direct |
| `PSI` | `equilibrium/time_slice/profiles_2d/psi` | direct |
| `CXRS_006` | `charge_exchange/channel/*/...` | diagnostic |

### 2.2 Diagnostic Relationships

Link TreeNodes to Diagnostic entities:
- `CXRS_*` → Diagnostic(name="CXRS", type="spectroscopy")
- `THOMSON_*` → Diagnostic(name="Thomson Scattering", type="laser")
- `FIR_*` → Diagnostic(name="FIR Interferometer", type="interferometry")

### 2.3 Analysis Code Connections

Link TreeNodes to AnalysisCode entities:
- `LIUQE_*` → AnalysisCode(name="LIUQE", type="equilibrium")
- `ASTRA_*` → AnalysisCode(name="ASTRA", type="transport")
- `CXSFIT_*` → AnalysisCode(name="CXSFIT", type="spectroscopy")

## Implementation

### CLI: `imas-codex agent enrich`

```bash
# Discover and enrich nodes needing metadata
uv run imas-codex agent enrich --discover

# Dry run to preview
uv run imas-codex agent enrich --discover --dry-run

# Specific tree only
uv run imas-codex agent enrich --discover --tree results

# High-context nodes first (have code examples)
uv run imas-codex agent enrich --discover --with-context-only

# Limit number of nodes
uv run imas-codex agent enrich --discover --limit 100

# Explicit paths (for targeted enrichment)
uv run imas-codex agent enrich "\\RESULTS::IBS" "\\RESULTS::LIUQE"
```

### Cost Estimate

| Configuration | Requests | Input Tokens | Output Tokens | Cost |
|--------------|----------|--------------|---------------|------|
| Basic (path only) | 348 | 2.1M | 0.5M | $0.42 |
| With code context | 348 | 2.5M | 0.5M | $0.50 |
| With SSH queries | 500 | 3.0M | 0.6M | $0.70 |

**Total estimated cost: $0.50 - $0.70**

### Rate Limiting

- OpenRouter rate limit: ~100 requests/minute for Flash
- Batch size: 50 paths/request
- Expected duration: 5-10 minutes

## Success Criteria

| Metric | Target |
|--------|--------|
| Nodes with descriptions | >80% (14,000+) |
| High confidence descriptions | >50% (8,500+) |
| Physics domain assigned | >90% |
| IMAS mappings discovered | >200 |
| New Diagnostic links | >50 |

## Future Enhancements

1. **Incremental enrichment**: Re-run on new code examples to improve confidence
2. **User feedback loop**: Flag incorrect descriptions for correction
3. **Cross-facility transfer**: Apply TCV learnings to other facilities
4. **Embedding updates**: Re-embed TreeNodes with descriptions for better search

## Dependencies

- OpenRouter API key in `.env`
- Neo4j running with current graph
- SSH access to EPFL (for introspection)
