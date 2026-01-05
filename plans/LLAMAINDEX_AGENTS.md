# LlamaIndex Agents Architecture

> **Status**: Implemented (January 2026)
> **Module**: `imas_codex.agents`

## Overview

This document describes the LlamaIndex-based agent system for autonomous exploration, metadata enrichment, and IMAS mapping discovery.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     imas_codex.agents                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Enrichment  │  │   Mapping   │  │ Exploration │  Agents     │
│  │   Agent     │  │    Agent    │  │    Agent    │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                      │
│                    ┌─────▼─────┐                                │
│                    │   Tools   │                                │
│                    └─────┬─────┘                                │
│         ┌────────────────┼────────────────┐                     │
│         │                │                │                     │
│   ┌─────▼─────┐   ┌──────▼──────┐  ┌──────▼──────┐             │
│   │  Neo4j    │   │    SSH      │  │   Search    │             │
│   │  Graph    │   │  MDSplus    │  │  IMAS/Code  │             │
│   └───────────┘   └─────────────┘  └─────────────┘             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    OpenRouter API                               │
│              (google/gemini-3-flash-preview)                    │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Types

### Enrichment Agent

**Purpose**: Generate physics-accurate metadata for MDSplus TreeNodes

**Capabilities**:
- Cross-reference graph data with code examples
- Query MDSplus directly via SSH for node metadata
- Generate descriptions with proper physics terminology
- TCV-specific knowledge (LIUQE, ASTRA, CXRS, etc.)

**Example Task**:
```
Analyze the TreeNode path \RESULTS::IBS and provide:
1. Physics description
2. Units and data type
3. Related TreeNodes
```

### Mapping Agent

**Purpose**: Discover semantic mappings between MDSplus and IMAS

**Capabilities**:
- Understand physics semantics of TreeNodes
- Search IMAS DD for equivalent paths
- Analyze unit and coordinate compatibility
- Document transformation requirements

**Example Task**:
```
Find IMAS equivalents for:
- \RESULTS::THOMSON:NE → core_profiles/profiles_1d/electrons/density?
- \RESULTS::LIUQE:IP → equilibrium/time_slice/global_quantities/ip?
```

### Exploration Agent

**Purpose**: Systematically explore facility data systems

**Capabilities**:
- Navigate codebases via SSH (rg, fd, ls)
- Identify high-value physics domains
- Document findings in the knowledge graph
- Queue files for ingestion

**Example Task**:
```
Explore the CXRS diagnostic in TCV:
- Find related TreeNodes
- Identify source code locations
- Document data structure
```

## Tools

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `query_neo4j` | Cypher query | Query results | Explore graph structure |
| `ssh_mdsplus_query` | tree, path, shot | Node metadata | Get MDSplus descriptions |
| `ssh_command` | shell command | stdout/stderr | File exploration, custom scripts |
| `search_code_examples` | search term | Code snippets | Find usage patterns |
| `search_imas_paths` | natural language | IMAS paths | Semantic search DD |
| `get_tree_structure` | tree name | Hierarchy | View TreeNode structure |

## LLM Configuration

### Provider: OpenRouter

All LLM calls go through OpenRouter for unified access to multiple models.

```python
# Configuration in llm.py
DEFAULT_MODEL = "google/gemini-3-flash-preview"
API_BASE = "https://openrouter.ai/api/v1"
```

### Model Selection

| Model | Context | Cost | Best For |
|-------|---------|------|----------|
| `gemini-3-flash-preview` | 1M | $ | Default, cost-effective |
| `gemini-2.5-pro-preview` | 1M | $$ | Complex reasoning |
| `claude-sonnet-4` | 200K | $$$ | Nuanced analysis |

### Temperature Settings

- **Enrichment**: 0.3 (deterministic, physics accuracy)
- **Mapping**: 0.3 (consistent mappings)
- **Exploration**: 0.5 (creative discovery)

## Usage Patterns

### CLI

```bash
# Quick task
imas-codex agent run "Describe \RESULTS::ASTRA"

# With agent type
imas-codex agent run "Find IMAS paths for Te" --type mapping

# Batch enrichment
imas-codex agent enrich "\RESULTS::IBS" "\RESULTS::ASTRA"
```

### Python API

```python
from imas_codex.agents import quick_agent_task, get_enrichment_agent, run_agent

# Quick one-liner
result = await quick_agent_task("Describe \RESULTS::IBS")

# With control
agent = get_enrichment_agent(verbose=True)
result = await run_agent(agent, "Analyze this path...")
```

### Batch Processing

```python
from imas_codex.agents import batch_enrich_paths

paths = ["\\RESULTS::IBS", "\\RESULTS::ASTRA", "\\RESULTS::LIUQE"]
results = await batch_enrich_paths(paths)

for r in results:
    print(f"{r.path}: {r.description}")
```

## Future Work

### Phase 1: Enhanced Enrichment (Current)

- [x] Basic agent module structure
- [x] Neo4j, SSH, and search tools
- [x] CLI integration
- [ ] Structured output parsing (JSON mode)
- [ ] Graph persistence of enrichment results
- [ ] Confidence scoring

### Phase 2: IMAS Mapping Pipeline

- [ ] Mapping result schema (LinkML)
- [ ] Unit conversion detection
- [ ] Coordinate system analysis
- [ ] Mapping confidence metrics
- [ ] Graph storage of mappings

### Phase 3: Autonomous Exploration

- [ ] Multi-agent orchestration
- [ ] Progress tracking and resumption
- [ ] Automatic file queuing
- [ ] Discovery prioritization
- [ ] Cross-facility patterns

### Phase 4: Advanced Features

- [ ] RAG over code examples (LlamaIndex VectorStoreIndex)
- [ ] Agent memory (conversation persistence)
- [ ] Tool result caching
- [ ] Parallel tool execution
- [ ] Custom prompt templates per facility

## Cost Estimation

Using `google/gemini-3-flash-preview` via OpenRouter:

| Task | Tokens (est) | Cost (est) |
|------|--------------|------------|
| Single enrichment | ~5K | ~$0.001 |
| Batch 100 paths | ~500K | ~$0.10 |
| Full tree enrichment (~2000 nodes) | ~10M | ~$2.00 |
| IMAS mapping discovery | ~50K/mapping | ~$0.01 |

## Integration Points

### With Existing Systems

1. **Graph Database**: Agents read/write via `GraphClient`
2. **Code Ingestion**: Agents queue files via `ingest_nodes`
3. **MCP Server**: Agents can be invoked from MCP tools
4. **CLI**: Full CLI integration for user access

### Data Flow

```
User Query → Agent → Tools → {Graph, SSH, Search} → Response
                         ↓
                    Graph Update (optional)
```

## Testing

```bash
# Run agent module tests
uv run pytest tests/agents/ -v

# Test specific agent
uv run python -c "
from imas_codex.agents import quick_agent_task
import asyncio
result = asyncio.run(quick_agent_task('Test query'))
print(result)
"
```

## References

- [LlamaIndex ReActAgent Docs](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/)
- [OpenRouter API](https://openrouter.ai/docs)
- [agents/README.md](../imas_codex/agents/README.md) - Module documentation
