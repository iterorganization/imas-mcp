# Agents Module

The `imas_codex.agentic` module provides two complementary agent systems:

1. **MCP Server** (`AgentsServer`) - REPL-first exploration via Model Context Protocol
2. **LlamaIndex ReActAgents** - Autonomous agents with tool-use capabilities

## Quick Start

### MCP Server (Primary Interface)

The MCP server provides **4 core tools** with a Python REPL as the primary interface:

```bash
# Start the agents server
imas-codex serve agents
```

**Core Tools:**

| Tool | Purpose |
|------|---------|
| `python` | Persistent REPL with pre-loaded utilities (primary interface) |
| `get_graph_schema` | Schema introspection for Cypher query generation |
| `ingest_nodes` | Schema-validated node creation with privacy filtering |
| `private` | Read/update sensitive infrastructure files |

The `python()` REPL includes rich pre-loaded utilities:

**Graph:**
- `query(cypher, **params)` - Execute Cypher, return list of dicts
- `semantic_search(text, index, k)` - Vector similarity search
- `embed(text)` - Get 384-dim embedding vector

**Remote:**
- `ssh(cmd, facility, timeout)` - Run SSH command on remote facility
- `check_tools(facility)` - Verify required tools are available

**Facility:**
- `get_facility(facility)` - Comprehensive facility info + graph state
- `get_exploration_targets(facility, limit)` - Prioritized work items
- `get_tree_structure(tree, prefix, limit)` - TreeNode hierarchy

**Code Search:**
- `search_code(query, top_k, facility, min_score)` - Semantic code search

**IMAS Data Dictionary:**
- `search_imas(query, ids_filter, max_results)` - Semantic DD search
- `fetch_imas(paths)` - Full documentation for paths
- `list_imas(paths, leaf_only, max_paths)` - List IDS structure
- `check_imas(paths)` - Validate path existence
- `get_imas_overview(query)` - High-level DD summary

### CLI Agent Usage

```bash
# Run a quick agent task
imas-codex agent run "Describe what \RESULTS::ASTRA is used for"

# Use a specific agent type
imas-codex agent run "Find IMAS paths for electron temperature" --type mapping

# Batch enrich TreeNode paths
imas-codex agent enrich "\RESULTS::IBS" "\RESULTS::ASTRA"
```

### Python API

```python
from imas_codex.agentic import (
    get_enrichment_agent,
    get_mapping_agent,
    get_exploration_agent,
    run_agent,
    quick_agent_task,
)

# Quick one-liner
result = await quick_agent_task(
    "Describe what \\RESULTS::ASTRA is used for",
    agent_type="enrichment"
)

# Full control
agent = get_enrichment_agent(verbose=True)
result = await run_agent(agent, "Analyze this path...")
```

## MCP Server REPL Examples

The `python()` tool is the primary interface. Examples:

```python
# Graph query
python("paths = query('MATCH (t:TreeNode) RETURN t.path LIMIT 5')")

# Semantic search on code
python("hits = semantic_search('plasma current', 'code_chunk_embedding', 3)")

# SSH command
python("print(ssh('ls /home/codes | head -5'))")

# IMAS search
python("print(search_imas('electron temperature profile'))")

# Facility info
python("info = get_facility('epfl'); print(info['graph_summary'])")

# Variables persist between calls
python("x = 42")
python("print(x * 2)")  # prints 84

# Complex multi-step workflow
python("""
paths = query('MATCH (t:TreeNode {tree_name: \"results\"}) RETURN t.path LIMIT 10')
for p in paths:
    print(p['t.path'])
""")
```

## Vector Indexes

Available for `semantic_search()`:

| Index | Content | Size |
|-------|---------|------|
| `imas_path_embedding` | IMAS Data Dictionary paths | 61k |
| `code_chunk_embedding` | Code examples | 8.5k |
| `wiki_chunk_embedding` | Wiki documentation | 25k |
| `cluster_centroid` | Semantic clusters | ~500 |

## Agent Types

| Agent | Use Case | System Prompt Focus |
|-------|----------|---------------------|
| **Enrichment** | TreeNode metadata generation | Physics accuracy, TCV-specific knowledge |
| **Mapping** | IMAS ↔ MDSplus discovery | Semantic equivalence, unit compatibility |
| **Exploration** | Facility exploration | Systematic discovery, graph persistence |

## LlamaIndex Tools

For ReActAgents (not MCP server), use `tools.py`:

```python
from imas_codex.agentic.tools import (
    get_exploration_tools,  # Fast startup
    get_imas_tools,         # IMAS DD tools (~30s startup)
    get_wiki_tools,         # Wiki search
    get_enrichment_tools,   # Optimized for enrichment
)

# Fast agents (default)
tools = get_exploration_tools()

# Add IMAS tools if needed (slow startup)
tools = get_exploration_tools() + get_imas_tools()
```

## Configuration

### LLM Setup

Agents use OpenRouter for LLM access:

```bash
# Required: OpenRouter API key (in .env)
OPENAI_API_KEY=sk-or-v1-...
```

### Model Selection

Default: `google/gemini-3-flash-preview` (1M context, cost-effective)

```python
from imas_codex.agentic import get_llm, MODELS

# Use preset
llm = get_llm(model=MODELS["gemini-pro"])

# Or specific model ID
llm = get_llm(model="anthropic/claude-sonnet-4")
```

## Architecture

```
imas_codex/agentic/
├── __init__.py      # Public API exports
├── llm.py           # OpenRouter LLM configuration
├── tools.py         # LlamaIndex FunctionTools
├── react.py         # ReActAgent configurations
├── server.py        # MCP server with 4 core tools
├── prompt_loader.py # Prompt template loading
└── prompts/         # System prompts (markdown)
```

### MCP Server vs ReActAgents

| Feature | MCP Server | ReActAgents |
|---------|------------|-------------|
| Interface | `python()` REPL | Multi-step reasoning |
| Orchestration | External (Claude, Copilot) | Self (LlamaIndex) |
| Tool calls | Code generation | Function calls |
| State | Persistent REPL | Conversation context |
| Use case | Interactive exploration | Batch processing |

## Related Documentation

- [AGENTS.md](../../../AGENTS.md) - Project-wide agent guidelines
- [plans/LLAMAINDEX_AGENTS.md](../../../plans/LLAMAINDEX_AGENTS.md) - Architecture and roadmap
- [config/README.md](../config/README.md) - Facility exploration guide
- [plans/MDSPLUS_INGESTION.md](../../../plans/MDSPLUS_INGESTION.md) - Tree ingestion workflow
